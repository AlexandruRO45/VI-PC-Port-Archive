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


def translate_ms(group_shared, *, mesh_compute=True, emulate=True, limit=None):
    prev = rdna2spirv.MESH_COMPUTE, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.MESH_COMPUTE = mesh_compute
    rdna2spirv.WAVE64_EMULATE = emulate
    if limit is not None:
        rdna2spirv.MAX_SHARED_BYTES = limit
    try:
        spv = support.translate_lines("ms_shared_limit", PROGRAM, dxbc=mesh_dxbc(group_shared))
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.MESH_COMPUTE, rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    return spv, meta


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
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            with tempfile.TemporaryDirectory() as tmp:
                asm = os.path.join(tmp, "ms.spvasm")
                bin_path = os.path.join(tmp, "ms.spv")
                with open(asm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                for cmd in (["spirv-as", "--target-env", "vulkan1.2", asm, "-o", bin_path],
                            ["spirv-val", "--target-env", "vulkan1.2", bin_path]):
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

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


if __name__ == "__main__":
    unittest.main()
