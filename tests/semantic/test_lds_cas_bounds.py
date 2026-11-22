"""Regression tests for SHADER-309 bounds-checked LDS CAS atomics in Gen.lds_op().

Covers all 10 CAS-loop LDS atomics:
  ds_inc_u32 / ds_inc_rtn_u32, ds_dec_u32 / ds_dec_rtn_u32, ds_rsub_u32 / ds_rsub_rtn_u32,
  ds_min_f32 / ds_min_rtn_f32, ds_max_f32 / ds_max_rtn_f32.

When Gen.lds_bound is set (non-power-of-two RSRC2.LDS_SIZE after fit_shared_memory() shrink or
when a constant ds offset exceeds RSRC2.LDS_SIZE), these atomics must go through Gen.lds_access():
  - in-bounds dwords address %lds[i] directly without power-of-two bitmasking (e.g. for 384 dwords,
    lds_ptr's `i & 383` silently cleared bit 7 and redirected dword 128 to dword 0);
  - out-of-bounds dwords (i >= lds_bound, including lds_bound == 0) leave LDS untouched and return 0;
  - inactive lanes leave both LDS and vdst untouched.

Standard library only; no GPU, no LLVM, no game files. Validates SPIR-V with spirv-as/spirv-val
when available on PATH.
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
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

VSHARP = [0, 0, 0x1000, 0x31016FAC]
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

CAS_SPECS = [
    # (op_non_rtn, op_rtn, init_word, data_word, expected_after)
    ("ds_inc_u32",  "ds_inc_rtn_u32",  0x00000010, 0x00000100, 0x00000011),
    ("ds_dec_u32",  "ds_dec_rtn_u32",  0x00000010, 0x00000100, 0x0000000F),
    ("ds_rsub_u32", "ds_rsub_rtn_u32", 0x00000010, 0x00000100, 0x000000F0),
    ("ds_min_f32",  "ds_min_rtn_f32",  0x3F800000, 0x3F000000, 0x3F000000),  # min(1.0, 0.5) = 0.5
    ("ds_max_f32",  "ds_max_rtn_f32",  0x3F000000, 0x3F800000, 0x3F800000),  # max(0.5, 1.0) = 1.0
]


def fssm(lds_units, threads=32):
    return bytes(0x138) + struct.pack("<4I", 0x88 | (lds_units << 15), threads, 1, 1)


def translate(lines, lds_units, *, threads=32, emulate=False, max_shared=None):
    prev = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES
    rdna2spirv.WAVE64_EMULATE = emulate
    if max_shared is not None:
        rdna2spirv.MAX_SHARED_BYTES = max_shared
    try:
        spv = support.translate_lines("cs_lds_cas_bounds", lines + ["s_endpgm"],
                                      fssm=fssm(lds_units, threads=threads))
        gen = rdna2spirv.translate.last_gen
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_SHARED_BYTES = prev
    return spv, gen


def validate_spirv(spv):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm = os.path.join(tmp, "s.spvasm")
        bin_ = os.path.join(tmp, "s.spv")
        with open(asm, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.3", asm, "-o", bin_],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", bin_],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def run_cs(spv, threads=32, nlanes=1):
    return si.run_workgroup(
        spv, nlanes, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )


class LdsCasBoundsTests(unittest.TestCase):
    def test_non_pow2_in_bounds_and_oob_all_ten_ops(self):
        # 3 LDS units = 1,536 bytes = 384 dwords (non-power-of-two: 384 - 1 = 383 = 0x17F clears bit 7 = 128).
        # The offset:1536 access pushes need > lds_declared, activating SHADER-309 bounds checking (lds_bound = 384).
        # Dword 0 = 0xAAAA (sentinel), dword 128 (byte 512, in-bounds!) = init_word, dword 256 (byte 1024, where
        # 384 & 383 would wrap!) = 0x5555 (sentinel).
        for op_non_rtn, op_rtn, init_w, data_w, expected_w in CAS_SPECS:
            for op, rtn in ((op_rtn, True), (op_non_rtn, False)):
                with self.subTest(op=op):
                    lines = [
                        "v_mov_b32_e32 v1, 0",
                        "v_mov_b32_e32 v2, 0xaaaa",
                        "ds_write_b32 v1, v2",
                        "v_mov_b32_e32 v1, 512",
                        f"v_mov_b32_e32 v2, {init_w:#x}",
                        "ds_write_b32 v1, v2",
                        "v_mov_b32_e32 v1, 1024",
                        "v_mov_b32_e32 v2, 0x5555",
                        "ds_write_b32 v1, v2",
                        "s_waitcnt lgkmcnt(0)",
                        f"v_mov_b32_e32 v3, {data_w:#x}",
                        "v_mov_b32_e32 v4, 0xdead",
                        "v_mov_b32_e32 v5, 0xbeef",
                        # 1) Strictly in-bounds at byte 512 (dword 128 < 384)
                        "v_mov_b32_e32 v1, 512",
                        f"{op} v4, v1, v3" if rtn else f"{op} v1, v3",
                        # 2) Out-of-bounds at byte 1536 (dword 384 == lds_bound)
                        "v_mov_b32_e32 v1, 0",
                        f"{op} v5, v1, v3 offset:1536" if rtn else f"{op} v1, v3 offset:1536",
                        "s_waitcnt lgkmcnt(0)",
                        # Read back dwords 0, 128, 256
                        "v_mov_b32_e32 v1, 0",    "ds_read_b32 v6, v1",
                        "v_mov_b32_e32 v1, 512",  "ds_read_b32 v7, v1",
                        "v_mov_b32_e32 v1, 1024", "ds_read_b32 v8, v1",
                        "s_waitcnt lgkmcnt(0)",
                        "v_mov_b32_e32 v10, 0",  "buffer_store_dword v4, v10, s[0:3], 0 offen",
                        "v_mov_b32_e32 v10, 4",  "buffer_store_dword v5, v10, s[0:3], 0 offen",
                        "v_mov_b32_e32 v10, 8",  "buffer_store_dword v6, v10, s[0:3], 0 offen",
                        "v_mov_b32_e32 v10, 12", "buffer_store_dword v7, v10, s[0:3], 0 offen",
                        "v_mov_b32_e32 v10, 16", "buffer_store_dword v8, v10, s[0:3], 0 offen",
                    ]
                    spv, gen = translate(lines, lds_units=3)
                    self.assertEqual(gen.lds_bound, 384)
                    self.assertEqual(gen.lds_dwords, 384)
                    # No power-of-two OpBitwiseAnd mask before OpAccessChain %lds
                    self.assertEqual(
                        re.findall(r"= OpBitwiseAnd %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ", spv),
                        [],
                    )
                    validate_spirv(spv)
                    mem = run_cs(spv)
                    self.assertEqual(mem[0], init_w if rtn else 0xDEAD)   # in-bounds return
                    self.assertEqual(mem[4], 0 if rtn else 0xBEEF)        # OOB return is 0 (non-rtn keeps v5)
                    self.assertEqual(mem[8], 0xAAAA)                      # dword 0 untouched
                    self.assertEqual(mem[12], expected_w)                 # dword 128 updated
                    self.assertEqual(mem[16], 0x5555)                     # dword 256 untouched (no OOB wrap)

    def test_zero_declared_lds_drops_atomic_and_returns_zero(self):
        # RSRC2.LDS_SIZE == 0 with a constant offset -> lds_bound = 0, %lds_arr has length 1.
        # Previously lds_ptr masked with `0 - 1 = 0xFFFFFFFF`, indexing completely unmasked past %lds[0].
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v3, 10",
            "v_mov_b32_e32 v4, 0xdead",
            "ds_inc_rtn_u32 v4, v1, v3 offset:16",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, 0",
            "buffer_store_dword v4, v10, s[0:3], 0 offen",
        ]
        spv, gen = translate(lines, lds_units=0)
        self.assertEqual(gen.lds_bound, 0)
        validate_spirv(spv)
        mem = run_cs(spv)
        self.assertEqual(mem[0], 0)

    def test_wave64_emulate_shrunk_lds_fit_shared_memory(self):
        # 100 LDS units = 51,200 B = 12,800 dwords (non-power-of-two: 12799 = 0x31FF clears bits 9..11).
        # fit_shared_memory() shrinks %lds from 64 KiB to 51,200 B and sets lds_bound = 12800.
        # Byte 2048 = dword 512 (0x200) had bit 9 cleared to 0 by the old lds_ptr mask!
        # 1) Value-verified execution of the fit_shared_memory() shrink path (wave32, max_shared=51200).
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0x1111",
            "ds_write_b32 v1, v2",
            "v_mov_b32_e32 v1, 2048",
            "v_mov_b32_e32 v2, 0x20",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0x100",
            "ds_inc_rtn_u32 v4, v1, v3",
            "v_mov_b32_e32 v1, 51200",        # dword 12800 == lds_bound (OOB)
            "ds_inc_rtn_u32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v1, 0",    "ds_read_b32 v6, v1",
            "v_mov_b32_e32 v1, 2048", "ds_read_b32 v7, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, 0",  "buffer_store_dword v4, v10, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, 4",  "buffer_store_dword v5, v10, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, 8",  "buffer_store_dword v6, v10, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, 12", "buffer_store_dword v7, v10, s[0:3], 0 offen",
        ]
        spv, gen = translate(lines, lds_units=100, threads=32, emulate=False, max_shared=51200)
        self.assertEqual(gen.lds_bound, 12800)
        self.assertEqual(gen.lds_dwords, 12800)
        self.assertEqual(
            re.findall(r"= OpBitwiseAnd %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ", spv),
            [],
        )
        validate_spirv(spv)
        mem = run_cs(spv, threads=32, nlanes=1)
        self.assertEqual(mem[0], 0x20)     # in-bounds dword 512 old value
        self.assertEqual(mem[4], 0)        # OOB dword 12800 returns 0
        self.assertEqual(mem[8], 0x1111)   # dword 0 untouched
        self.assertEqual(mem[12], 0x21)    # dword 512 incremented

        # 2) Multi-segment wave64 under --wave64-emulate (default 64 KiB cap triggers fit_shared_memory() shrink
        # because 64 KiB + %x_buf > 64 KiB; s_barrier + s_and_saveexec_b64 split segments): ensures
        # current_label_placeholder() respects x_marks and spirv-val accepts the nested CAS OpPhi.
        multi_seg = [
            "s_mov_b64 exec, -1",
            "s_barrier",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_saveexec_b64 s[4:5], vcc",
            "v_mov_b32_e32 v1, 2048",
            "v_mov_b32_e32 v3, 0x100",
            "ds_inc_rtn_u32 v4, v1, v3",
            "ds_min_rtn_f32 v5, v1, v3",
            "ds_dec_u32 v1, v3",
            "s_mov_b64 exec, s[4:5]",
        ]
        spv2, gen2 = translate(multi_seg, lds_units=100, threads=64, emulate=True)
        self.assertTrue(gen2.wave64)
        self.assertEqual(gen2.lds_bound, 12800)
        self.assertEqual(gen2.lds_dwords, 12800)
        self.assertEqual(
            re.findall(r"= OpBitwiseAnd %uint %\w+ %\w+\n\s*%\w+ = OpAccessChain %_ptr_WG_uint\w* %lds ", spv2),
            [],
        )
        self.assertEqual(len(re.findall(r"^\s*%\w+_lin = OpLabel$", spv2, re.M)), 3)
        validate_spirv(spv2)

    def test_inactive_lane_under_lds_bound_keeps_vdst_and_lds(self):
        # Inactive lane executing an OOB ds_inc_rtn_u32 must keep its old vdst (not overwrite with 0).
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 7",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 100",
            "v_mov_b32_e32 v4, 0xdead",
            "s_mov_b32 s4, exec_lo",
            "s_mov_b32 exec_lo, 0",
            "ds_inc_rtn_u32 v4, v1, v3 offset:1536",
            "s_mov_b32 exec_lo, s4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v5, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v10, 0", "buffer_store_dword v4, v10, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, 4", "buffer_store_dword v5, v10, s[0:3], 0 offen",
        ]
        spv, gen = translate(lines, lds_units=3)
        self.assertEqual(gen.lds_bound, 384)
        validate_spirv(spv)
        mem = run_cs(spv)
        self.assertEqual(mem[0], 0xDEAD)
        self.assertEqual(mem[4], 7)

    def test_unbounded_lds_emits_no_bounds_check(self):
        # Power-of-two LDS staying within RSRC2 keeps the unwrapped fast path (0 _lin labels).
        lines = [
            "v_mov_b32_e32 v1, 16",
            "v_mov_b32_e32 v3, 10",
            "ds_inc_rtn_u32 v4, v1, v3",
        ]
        spv, gen = translate(lines, lds_units=1)
        self.assertIsNone(gen.lds_bound)
        self.assertEqual(re.findall(r"^\s*%\w+_lin = OpLabel$", spv, re.M), [])
        validate_spirv(spv)


if __name__ == "__main__":
    unittest.main()
