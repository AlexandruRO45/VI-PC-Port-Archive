"""Workgroup shared-memory cap (--max-shared-memory) for --mesh-compute mesh shaders.

Under --mesh-compute, a mesh shader emits GLCompute (not MeshEXT), so Vulkan enforces maxComputeSharedMemorySize
(49,152 B on NVIDIA Pascal) on its total Workgroup memory: %lds + %x_buf (--wave64-emulate) + 4-byte %ms_counts.
When a rounded-up power-of-two LDS array pushes that sum over MAX_SHARED_BYTES while the shader's declared/offset
need still fits, fit_shared_memory() shrinks %lds to the need and bounds-checks accesses (lds_access); when even the
need plus %x_buf + %ms_counts exceeds MAX_SHARED_BYTES, assemble() refuses the shader with Unsupported. Native
MeshEXT (without --mesh-compute) is unaffected.

All inputs are hand-written here; no game data.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

PROGRAM = [
    "s_mov_b64 exec, -1",
    "v_lshlrev_b32_e32 v1, 2, v0",
    "v_readfirstlane_b32 s16, v1",
    "ds_write_b32 v1, v0",
    "s_waitcnt lgkmcnt(0)",
    "ds_read_b32 v2, v1 offset:4",
    "s_waitcnt lgkmcnt(0)",
    "s_endpgm",
]
THREADS = 128
X_BUF = 4 * (2 * THREADS + 6 * (THREADS // 64) + 7)  # 2 waves of 64 -> 1100 bytes
MS_COUNTS = 4


def mesh_dxbc(group_shared, threads=THREADS):
    """DXBC with one PSV0 part: stage 13 (MS), 64/64 triangles, NumThreads, GroupSharedBytesUsed."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, 64, 64)
    r[24], r[27] = 13, 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    part = b"PSV0" + struct.pack("<I", 4 + len(r)) + struct.pack("<I", len(r)) + bytes(r)
    head = 32 + 4
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, head + len(part), 1, head) + part


def translate_ms(group_shared, *, mesh_compute=True, emulate=True, limit=None, program=PROGRAM, threads=THREADS):
    prev = rdna2spirv.MESH_COMPUTE, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.MESH_COMPUTE = mesh_compute
    rdna2spirv.WAVE64_EMULATE = emulate
    if limit is not None:
        rdna2spirv.MAX_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("ms_shared_limit", program, dxbc=mesh_dxbc(group_shared, threads=threads))
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.MESH_COMPUTE, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    return spv, meta


def validate_spv(tc, spv):
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        with tempfile.TemporaryDirectory() as tmp:
            asm = os.path.join(tmp, "ms.spvasm")
            bin_path = os.path.join(tmp, "ms.spv")
            with open(asm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", bin_path],
                        ["spirv-val", "--target-env", "vulkan1.2", bin_path]):
                r = subprocess.run(cmd, capture_output=True, text=True)
                tc.assertEqual(r.returncode, 0, r.stdout + r.stderr)


def workgroup_bytes(spv):
    """{variable: bytes} of every Workgroup-storage-class OpVariable, sized from its pointee type."""
    consts, types, ptrs, out = {}, {}, {}, {}
    for ln in spv.splitlines():
        t = ln.split()
        if len(t) < 3 or t[1] != "=":
            continue
        rid, op, args = t[0], t[2], t[3:]
        if op == "OpConstant":
            consts[rid] = int(args[1], 0)
        elif op in ("OpTypeInt", "OpTypeFloat"):
            types[rid] = int(args[0]) // 8
        elif op == "OpTypeBool":
            types[rid] = 4
        elif op == "OpTypeVector" and args[0] in types:
            types[rid] = types[args[0]] * int(args[1])
        elif op == "OpTypeArray" and args[0] in types and args[1] in consts:
            types[rid] = types[args[0]] * consts[args[1]]
        elif op == "OpTypeStruct" and all(a in types for a in args):
            types[rid] = sum(types[a] for a in args)
        elif op == "OpTypePointer" and args[0] == "Workgroup":
            ptrs[rid] = args[1]
        elif op == "OpVariable" and args[1:2] == ["Workgroup"]:
            out[rid] = types[ptrs[args[0]]]
    return out


def lds_index_ops(spv):
    return set(re.findall(r"= (OpBitwiseAnd|OpUMod) %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ",
                          spv))


def bounds_checks(spv):
    return len(re.findall(r"^\s*%\w+_lin = OpLabel$", spv, re.M))


class MeshComputeSharedMemoryLimitTests(unittest.TestCase):
    def test_pascal_48k_shrinks_rounded_up_lds_to_need(self):
        # 36 KiB declared rounds up to 64 KiB (over Pascal's 48 KiB limit); shrinking to 36 KiB fits with %x_buf + %ms_counts.
        spv, meta = translate_ms(36864, mesh_compute=True, emulate=True, limit=49152)
        wg = workgroup_bytes(spv)
        self.assertEqual(wg, {"%lds": 36864, "%x_buf": X_BUF, "%ms_counts": MS_COUNTS})
        self.assertLessEqual(sum(wg.values()), 49152)
        self.assertEqual(meta["lds_bytes"], sum(wg.values()))
        self.assertEqual(lds_index_ops(spv), set())
        self.assertEqual(bounds_checks(spv), 2)
        validate_spv(self, spv)

    def test_exact_limit_accepted_and_one_dword_less_refused(self):
        exact = 32768 + X_BUF + MS_COUNTS
        spv, meta = translate_ms(32768, mesh_compute=True, emulate=True, limit=exact)
        wg = workgroup_bytes(spv)
        self.assertEqual(wg, {"%lds": 32768, "%x_buf": X_BUF, "%ms_counts": MS_COUNTS})
        self.assertEqual(meta["lds_bytes"], exact)
        self.assertEqual(lds_index_ops(spv), {"OpBitwiseAnd"})
        self.assertEqual(bounds_checks(spv), 0)
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_ms(32768, mesh_compute=True, emulate=True, limit=exact - 4)
        msg = str(cm.exception)
        self.assertIn("MS (--mesh-compute) workgroup memory", msg)
        self.assertIn("LDS 32768 B", msg)
        self.assertIn(f"wave64 exchange {X_BUF} B", msg)
        self.assertIn("ms_counts 4 B", msg)

    def test_over_48k_pascal_limit_is_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            translate_ms(49152, mesh_compute=True, emulate=True, limit=49152)
        msg = str(cm.exception)
        self.assertIn("MS (--mesh-compute) workgroup memory", msg)
        self.assertIn("49152 B", msg)

        with self.assertRaises(rdna2spirv.Unsupported) as cm_no_emu:
            translate_ms(50000, mesh_compute=True, emulate=False, limit=49152)
        msg_no_emu = str(cm_no_emu.exception)
        self.assertIn("ms_counts 4 B", msg_no_emu)
        self.assertNotIn("wave64 exchange", msg_no_emu)

    def test_native_mesh_ext_is_unaffected_by_compute_shared_limit(self):
        spv, _ = translate_ms(36864, mesh_compute=False, emulate=False, limit=49152)
        self.assertEqual(workgroup_bytes(spv), {"%lds": 65536, "%ms_counts": MS_COUNTS})

    def test_shrunk_lds_addtid_and_float_atomics_preserve_addressing(self):
        # 36 KiB declared LDS (9216 dwords, non-power-of-two) shrunk under a 48 KiB cap.
        # Direct lds_ptr() bitmasking (idx & 0x23ff) would map dword 1027 (byte 4108) onto dword 3 (byte 12).
        f_1 = 0x3F800000     # 1.0f
        f_2 = 0x40000000     # 2.0f
        f_half = 0x3F000000  # 0.5f
        f_3_5 = 0x40600000   # 3.5f
        f_7 = 0x40E00000     # 7.0f
        f_9 = 0x41100000     # 9.0f

        def store_out(slot, vreg):
            return [f"v_mov_b32_e32 v30, {4 * slot}", f"buffer_store_dword v{vreg}, v30, s[4:7], 0 offen"]

        prog = [
            "s_mov_b32 exec_lo, -1",
            # Seed dword 0 (sentinel for OOB checks), dword 3 (alias target of 1027 & 0x23ff), and dword 1027 (byte 4108).
            "s_mov_b32 m0, 0",
            "v_mov_b32_e32 v1, 0x5e17",
            "ds_write_addtid_b32 v1",
            "s_mov_b32 m0, 12",
            "v_mov_b32_e32 v1, 0x3333",
            "ds_write_addtid_b32 v1",
            "s_mov_b32 m0, 4108",
            f"v_mov_b32_e32 v1, {hex(f_1)}",
            "ds_write_addtid_b32 v1",
            "s_waitcnt lgkmcnt(0)",
            # In-bounds ds_read_addtid_b32 at M0=4108 (lane 0 -> dword 1027, NOT dword 3).
            "v_mov_b32_e32 v10, 0xdead",
            "ds_read_addtid_b32 v10",
            "s_waitcnt lgkmcnt(0)",
            # In-bounds ds_add_rtn_f32 & ds_add_f32 at byte 4108 (dword 1027).
            "v_mov_b32_e32 v2, 4108",
            f"v_mov_b32_e32 v3, {hex(f_2)}",
            "v_mov_b32_e32 v11, 0xdead",
            "ds_add_rtn_f32 v11, v2, v3",
            f"v_mov_b32_e32 v3, {hex(f_half)}",
            "ds_add_f32 v2, v3",
            "s_waitcnt lgkmcnt(0)",
            # In-bounds ds_cmpst_rtn_f32 & ds_cmpst_f32 at byte 4108 (dword 1027).
            f"v_mov_b32_e32 v3, {hex(f_3_5)}",
            f"v_mov_b32_e32 v4, {hex(f_7)}",
            "v_mov_b32_e32 v12, 0xdead",
            "ds_cmpst_rtn_f32 v12, v2, v3, v4",
            f"v_mov_b32_e32 v3, {hex(f_7)}",
            f"v_mov_b32_e32 v4, {hex(f_9)}",
            "ds_cmpst_f32 v2, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            # Read back dword 1027 and dword 3 after in-bounds ops.
            "ds_read_addtid_b32 v13",
            "s_mov_b32 m0, 12",
            "ds_read_addtid_b32 v14",
            "s_waitcnt lgkmcnt(0)",
            # Out-of-bounds (byte 36864 == dword 9216 == lds_bound): writes/atomics dropped, reads/returns give 0.
            "s_mov_b32 m0, 36864",
            "v_mov_b32_e32 v1, 0xbad",
            "ds_write_addtid_b32 v1",
            "v_mov_b32_e32 v15, 0xdead",
            "ds_read_addtid_b32 v15",
            "v_mov_b32_e32 v2, 36864",
            f"v_mov_b32_e32 v3, {hex(f_2)}",
            "v_mov_b32_e32 v16, 0xdead",
            "ds_add_rtn_f32 v16, v2, v3",
            "ds_add_f32 v2, v3",
            "v_mov_b32_e32 v17, 0xdead",
            "v_mov_b32_e32 v3, 0x5e17",
            "v_mov_b32_e32 v4, 0xbad",
            "ds_cmpst_rtn_f32 v17, v2, v3, v4",
            "ds_cmpst_f32 v2, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            # Verify dword 0 sentinel is untouched by OOB writes/atomics.
            "s_mov_b32 m0, 0",
            "ds_read_addtid_b32 v18",
            "s_waitcnt lgkmcnt(0)",
        ]
        for slot, vreg in enumerate((10, 11, 12, 13, 14, 15, 16, 17, 18)):
            prog += store_out(slot, vreg)
        prog.append("s_endpgm")

        # Validate both wave32 (non-emulated) and wave64-emulated shrunk --mesh-compute SPIR-V.
        for emulate, threads, expected_wg in (
            (False, 32, {"%lds": 36864, "%ms_counts": MS_COUNTS}),
            (True, THREADS, {"%lds": 36864, "%x_buf": X_BUF, "%ms_counts": MS_COUNTS}),
        ):
            with self.subTest(emulate=emulate):
                p = (["s_mov_b64 exec, -1"] + prog[1:]) if emulate else prog
                spv, meta = translate_ms(36864, mesh_compute=True, emulate=emulate, limit=49152,
                                         program=p, threads=threads)
                self.assertEqual(workgroup_bytes(spv), expected_wg)
                self.assertEqual(meta["lds_bytes"], sum(expected_wg.values()))
                self.assertEqual(lds_index_ops(spv), set())
                self.assertEqual(bounds_checks(spv), 17)
                validate_spv(self, spv)
                if not emulate:
                    # Execute lane 0 through spirv_mini_interp (output V# in s[4:7], base 0x1000 away from moa=0).
                    out_base = 0x1000
                    vsharp = [out_base, 0, 0x1000, 0x31016FAC]
                    mem = si.run_workgroup(
                        spv, 1, 32,
                        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%num_wg": (1, 1, 1),
                                              "%local_id": (i, 0, 0), "%local_index": i, "%sg_lane": i},
                        push_constants=[0, 0, 0, 0] + vsharp + [0] * 24,
                        memory={},
                    )
                    got = tuple(mem.get(out_base + 4 * k) for k in range(9))
                    self.assertEqual(got, (f_1, f_1, f_3_5, f_9, 0x3333, 0, 0, 0, 0x5E17))


if __name__ == "__main__":
    unittest.main()
