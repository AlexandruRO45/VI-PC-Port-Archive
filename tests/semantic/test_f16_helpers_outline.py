"""Semantic and size regression tests for FP16 constant folding and Pure OpFunction helper outlining.

Verifies that:
  1. Non-NaN literal/inline-constant FP16 operands fold at translation time with zero UnpackHalf2x16
     instructions and bit-exact IEEE-754 binary16 -> binary32 semantics across all 63,488 non-NaN halves
     (subnormals, +-0, normals, +-inf, abs/neg, op_sel lo/hi, VOP3P neg_lo/neg_hi).
  2. Dynamic FP16 conversion and round-to-odd FMA kernels (%f16_in_{lo,hi}_fn, %f16_bits_*_fn,
     %mem_f16_fn, %f16_pack2_rne_fn, %fma_f16_odd_fn, %fma_mix_odd_fn) are emitted once per module
     as Pure OpFunctions, preserve NoContraction on all float arithmetic, avoid PackHalf2x16 (SHADER-56),
     and execute bit-identically raw and under --opt.
  3. Active runs (Gen.check_lane_local) permit Pure lane-local FP16 helpers while preserving inactive lanes.
  4. Multi-callsite FP16 and v_fma_mixlo_f16 shaders stay compact and assemble cleanly for Vulkan 1.2
     (including under --pc-tail-ubo).
"""
from __future__ import annotations

import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_opt  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
_FLOAT_ARITH_RE = re.compile(r"^\s*(%\w+) = Op(?:FMul|FAdd|FSub|FDiv|FNegate|FRem|FMod) ", re.M)


def _translate_ps(name: str, lines: list[str], *, pc_tail_ubo: bool = False) -> str:
    prev = rdna2spirv.PC_TAIL_UBO
    rdna2spirv.PC_TAIL_UBO = pc_tail_ubo
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            path = os.path.join(tmp, "disasm", f"{name}.s")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
            with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(path)
    finally:
        rdna2spirv.PC_TAIL_UBO = prev


def _f16_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _assert_unique_fn_types_and_no_contraction(tc: unittest.TestCase, spv: str) -> None:
    fn_types = [ln.split(" = ", 1)[1].strip() for ln in spv.splitlines() if " = OpTypeFunction " in ln]
    tc.assertEqual(len(fn_types), len(set(fn_types)), f"duplicate OpTypeFunction: {fn_types}")
    decorated = set(re.findall(r"OpDecorate (%\w+) NoContraction\b", spv))
    for fid in _FLOAT_ARITH_RE.findall(spv):
        tc.assertIn(fid, decorated, f"float op {fid} missing NoContraction")


class F16HelperOutliningTests(unittest.TestCase):
    def test_f16_half_to_f32_bits_exhaustive_non_nan(self):
        """Every non-NaN IEEE-754 binary16 value (63,488 values) widens bit-identically to binary32."""
        for h in range(0x10000):
            if (h & 0x7FFF) > 0x7C00:
                continue
            got = rdna2spirv.Gen._f16_half_to_f32_bits(h)
            want = _f32_bits(struct.unpack("<e", struct.pack("<H", h))[0])
            self.assertEqual(got, want, f"h={h:#06x}: got {got:#010x}, want {want:#010x}")

    def test_constant_folding_exact_and_no_runtime_unpack(self):
        """Non-NaN literal/inline FP16 operands fold with 0 UnpackHalf2x16 and bit-exact results."""
        # Packed constants:
        #   0x3c00_0001 -> lo = 2^-24 (subnormal), hi = 1.0
        #   0x8001_8000 -> lo = -0.0, hi = -2^-24 (subnormal)
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            # v_add_f16_e64: subnormal 0x0001 + subnormal 0x0001 = 0x0002 (low half)
            "v_add_f16_e64 v0, 0x0001, 0x0001",
            # v_mul_f16_e64 with abs/neg on inline const: |-2.0| * -0.5 = -1.0 (0xbc00) into high half
            "v_mul_f16_e64 v0, |-2.0|, -0.5 op_sel:[0,0,1]",
            # v_pk_fma_f16 on pure constants:
            #   lo: (0.5 * 2.0) + (-1.0) = 0.0 -> with neg_lo:[0,0,1] on 1.0 => 1.0 - 1.0 = +0.0 (0x0000)
            #   hi: (-2.0 * 1.5) + 0.5 = -2.5 (0xc100) using hex literal 0x3e00_3800 (lo=0.5, hi=1.5)
            "v_pk_fma_f16 v1, 0xc0003800, 0x3e004000, 0x38003c00 neg_lo:[0,0,1]",
            # v_pk_add_f16 with subnormals: lo = 0x0001 + 0x0002 = 0x0003, hi = -2^-24 + 2^-24 = +0.0
            "v_pk_add_f16 v2, 0x80010001, 0x00010002",
            # v_fma_f16_e64 with op_sel reading hi half of 0x3c00_4000 (hi=1.0, lo=2.0): 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
            "v_fma_f16 v3, 0x3c004000, 0x3c004000, 0.5 op_sel:[1,0,0,0]",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = _translate_ps("ps_f16_const_fold", lines)
        self.assertEqual(spv.count("UnpackHalf2x16"), 0, "all non-NaN constant f16 ops should fold without UnpackHalf2x16")
        self.assertNotIn("PackHalf2x16", spv)
        _assert_unique_fn_types_and_no_contraction(self, spv)
        want = [
            (_f16_bits(-1.0) << 16) | 0x0002,
            (_f16_bits(-2.5) << 16) | _f16_bits(0.0),
            (0x0000 << 16) | 0x0003,
            _f16_bits(2.5),
        ]
        for label, mod in (("raw", spv), ("opt", spirv_opt.optimize(spv))):
            spirv_asm.assemble(mod, target_env="vulkan1.2")
            got = list(si.run_fragment(mod)["%out_mrt0"])
            self.assertEqual(got, want, f"{label}: got {[hex(x) for x in got]}, want {[hex(x) for x in want]}")

    def test_outlined_helpers_parity_raw_opt_and_mode_round(self):
        """Dynamic f16_in, f16_bits (rne/rtz/exact/mode/mode_exact), mem_f16, pack2, fma_f16_odd, fma_mix_odd."""
        # Part A: non-MODE shader exercising %f16_in_{lo,hi}_fn, %f16_bits_{rne,rtz,exact}_fn,
        # %f16_pack2_rne_fn, %fma_f16_odd_fn, %fma_mix_odd_fn, and %mem_f16_fn (tbuffer_store 16_FLOAT).
        lines_a = [
            "s_mov_b32 s4, 0x1000",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 0x100",
            "s_mov_b32 s7, 0x3100DFAC",              # RAW, format=13 (16_FLOAT)
            "v_mov_b32 v4, 0x3c000001",              # lo = 2^-24 (subnormal), hi = 1.0
            "v_mov_b32 v5, 0x40000002",              # lo = 2*2^-24, hi = 2.0
            "v_mov_b32 v6, 0x3f801000",              # f32 1.00048828125 (tie -> RNE 0x3c00, RTZ 0x3c00)
            "v_mov_b32 v7, 0x47800000",              # f32 65536.0 (RNE -> inf 0x7c00, RTZ -> 65504 0x7bff)
            "v_mov_b32 v8, 0x7fa00000",              # f32 sNaN -> mem_f16 preserves payload 0x7d00
            # f16_in_lo + f16_in_hi + f16_bits_exact (v_add_f16)
            "v_add_f16_e64 v0, v4, v5 op_sel:[0,1,0]",   # 2^-24 + 2.0 -> 2.0 (0x4000)
            # f16_bits_rne via v_cvt_f16_f32
            "v_cvt_f16_f32_e32 v1, v7",                  # 65536.0 -> inf (0x7c00)
            # f16_bits_rtz via v_cvt_pkrtz_f16_f32 (also folds second arg 0)
            "v_cvt_pkrtz_f16_f32 v2, v7, v6",            # lo=0x7bff, hi=0x3c00
            # fma_f16_odd + f16_pack2_rne_fn via v_pk_fma_f16:
            #   lo: (2^-24 * 2*2^-24) + 2^-24 = 2^-24 (0x0001)
            #   hi: (1.0 * 2.0) + 1.0 = 3.0 (0x4200)
            "v_pk_fma_f16 v3, v4, v5, v4",
            # fma_mix_odd via v_fma_mixlo_f16: (1.0 * 2.0) + 1.0 = 3.0 (0x4200) into v9
            "v_mov_b32 v9, 0",
            "v_fma_mixlo_f16 v9, v4, v5, v4 op_sel:[1,1,1] op_sel_hi:[1,1,1]",
            # mem_f16 via tbuffer_store_format_x format:[BUF_FMT_16_FLOAT]
            "tbuffer_store_format_x v8, off, s[4:7], 0 format:[BUF_FMT_16_FLOAT]",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32 v10, 0",
            "v_add_f16_e64 v0, v0, v9 op_sel:[0,0,1]",   # hi = 2.0 + 3.0 = 5.0 (0x4500), lo stays 0x4000
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv_a = _translate_ps("ps_f16_helpers_all", lines_a)
        for fn in (
            "%f16_in_lo_fn = OpFunction %float Pure %fn_f16_in",
            "%f16_in_hi_fn = OpFunction %float Pure %fn_f16_in",
            "%f16_bits_rne_fn = OpFunction %uint Pure %fn_f16_bits",
            "%f16_bits_rtz_fn = OpFunction %uint Pure %fn_f16_bits",
            "%f16_bits_exact_fn = OpFunction %uint Pure %fn_f16_bits",
            "%mem_f16_fn = OpFunction %uint Pure %fn_f16_bits",
            "%f16_pack2_rne_fn = OpFunction %uint Pure %fn_f16_pack2",
            "%fma_f16_odd_fn = OpFunction %float Pure %fn_f32_3",
            "%fma_mix_odd_fn = OpFunction %float Pure %fn_f32_3",
        ):
            self.assertEqual(spv_a.count(fn), 1, f"expected 1 declaration of {fn}")
        self.assertNotIn("PackHalf2x16", spv_a)
        _assert_unique_fn_types_and_no_contraction(self, spv_a)
        want_a = [
            (_f16_bits(5.0) << 16) | _f16_bits(2.0),
            0x7C00,
            (_f16_bits(1.0) << 16) | 0x7BFF,
            (_f16_bits(3.0) << 16) | 0x0001,
        ]
        for label, mod in (("raw", spv_a), ("opt", spirv_opt.optimize(spv_a))):
            spirv_asm.assemble(mod, target_env="vulkan1.2")
            mem = {}
            got = list(si.run_fragment(mod, memory=mem)["%out_mrt0"])
            self.assertEqual(got, want_a, f"{label}: got {[hex(x) for x in got]}, want {[hex(x) for x in want_a]}")
            self.assertEqual(mem.get(0x1000, 0) & 0xFFFF, 0x7D00, f"{label}: mem_f16 sNaN payload mismatch")

        # Part B: dynamic MODE.FP_ROUND (f16 toward zero vs RNE) exercising %f16_bits_mode_fn & %f16_bits_mode_exact_fn.
        lines_b = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "v_mov_b32 v4, 0x7bff",                  # 65504.0
            "v_mov_b32 v5, 0x4c00",                  # 16.0 -> sum = 65520 (RNE -> inf 0x7c00, RTZ -> 65504 0x7bff)
            "v_mov_b32 v6, 0x477ff000",              # f32 65520.0
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3",      # f16 RTZ (MODE.FP_ROUND[3:2] = 3)
            "v_add_f16_e32 v0, v4, v5",              # RTZ -> 0x7bff (exact_tiny + mode)
            "v_cvt_f16_f32_e32 v1, v6",              # RTZ -> 0x7bff (mode)
            "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0",      # f16 RNE (MODE.FP_ROUND[3:2] = 0)
            "v_add_f16_e32 v2, v4, v5",              # RNE -> 0x7c00
            "v_cvt_f16_f32_e32 v3, v6",              # RNE -> 0x7c00
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv_b = _translate_ps("ps_f16_helpers_mode", lines_b)
        self.assertEqual(spv_b.count("%f16_bits_mode_fn = OpFunction %uint Pure %fn_f16_mode"), 1)
        self.assertEqual(spv_b.count("%f16_bits_mode_exact_fn = OpFunction %uint Pure %fn_f16_mode"), 1)
        _assert_unique_fn_types_and_no_contraction(self, spv_b)
        for label, mod in (("raw", spv_b), ("opt", spirv_opt.optimize(spv_b))):
            spirv_asm.assemble(mod, target_env="vulkan1.2")
            got = list(si.run_fragment(mod)["%out_mrt0"])
            self.assertEqual(got, [0x7BFF, 0x7BFF, 0x7C00, 0x7C00], f"{label}: {got}")

    def test_multi_callsite_size_and_pc_tail_ubo(self):
        """Multi-callsite FP16 and v_fma_mixlo_f16 shaders stay compact and assemble under --pc-tail-ubo."""
        lines_f16 = ["s_mov_b64 exec, -1", "v_mov_b32 v2, s0", "v_mov_b32 v3, s1"]
        for i in range(50):
            v = 2 + (i % 8)
            lines_f16 += [
                f"v_pk_fma_f16 v{v}, v{v}, 0.5, 1.0",
                f"v_add_f16_e32 v{v + 1}, v{v}, 2.0",
                f"v_mul_f16_e32 v{v + 1}, v{v + 1}, v{v}",
                f"v_cvt_pkrtz_f16_f32 v{v}, v{v}, v{v + 1}",
            ]
        lines_f16 += ["exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
        spv_f16 = _translate_ps("ps_f16_compact_200", lines_f16, pc_tail_ubo=True)
        bin_f16 = spirv_asm.assemble(spv_f16, target_env="vulkan1.2")
        # Baseline on main without outlining/const-folding is ~343 KB raw (~17.6k lines).
        self.assertLess(len(bin_f16), 95_000, f"expected compact 200-op f16 binary, got {len(bin_f16)} B")
        self.assertEqual(spv_f16.count("%f16_bits_rne_fn = OpFunction "), 1)
        self.assertEqual(spv_f16.count("%fma_f16_odd_fn = OpFunction "), 1)
        self.assertLessEqual(spv_f16.count("UnpackHalf2x16"), 52, "constant 0.5/1.0/2.0 should not emit UnpackHalf2x16")

        lines_mix = ["s_mov_b64 exec, -1", "v_mov_b32 v2, s0", "v_mov_b32 v3, s1"]
        for i in range(50):
            v = 2 + (i % 8)
            lines_mix.append(f"v_fma_mixlo_f16 v{v}, v{v}, 0.5, v{v + 1} op_sel_hi:[1,1,0]")
        lines_mix += ["exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
        spv_mix = _translate_ps("ps_fma_mix_compact_50", lines_mix)
        bin_mix = spirv_asm.assemble(spv_mix, target_env="vulkan1.2")
        # Baseline on main without outlining is ~125 KB raw.
        self.assertLess(len(bin_mix), 35_000, f"expected compact 50-op fma_mix binary, got {len(bin_mix)} B")
        self.assertEqual(spv_mix.count("%fma_mix_odd_fn = OpFunction "), 1)

    def test_active_run_allows_pure_f16_helpers(self):
        """Active runs permit Pure lane-local FP16 helpers while preserving inactive lane VGPRs."""
        lines = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mov_b32 v0, 0x3c003c00",              # (1.0, 1.0)
            "v_mov_b32 v1, 0x40004000",              # (2.0, 2.0)
            "v_mov_b32 v2, 0x42004200",              # (3.0, 3.0)
            "v_mov_b32 v3, 0x44004400",              # (4.0, 4.0)
            "v_cmp_gt_u32_e32 vcc_lo, 8, v4",
            "s_and_saveexec_b32 s0, vcc_lo",
            # Run of 4 lane-local FP16 ops inside active run:
            "v_add_f16_e32 v0, v0, 1.0",             # lo: 2.0, hi stays 1.0 -> 0x3c004000
            "v_mul_f16_e32 v1, v1, 2.0",             # lo: 4.0, hi stays 2.0 -> 0x40004400
            "v_pk_fma_f16 v2, v2, 0.5, 1.0",         # lo: 3*0.5+1=2.5 (0x4100), hi: 3*0+0=0.0 (0x0000)
            "v_fma_mixlo_f16 v3, v0, 0.5, v1 op_sel_hi:[1,1,1]",  # lo: 2.0*0.5+4.0=5.0 (0x4500), hi stays 4.0
            "s_or_b32 exec_lo, exec_lo, s0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = _translate_ps("ps_f16_active_run", lines)
        run_bodies = re.findall(r"%\w+_run = OpLabel\n(.*?)\n\s*OpBranch ", spv, re.S)
        self.assertTrue(run_bodies, "expected active run")
        self.assertTrue(any("OpFunctionCall %float %fma_f16_odd_fn" in b for b in run_bodies),
                        "expected Pure FP16 helper calls inside active run")
        spirv_asm.assemble(spv, target_env="vulkan1.2")
        _, outs = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%sg_lane": i}, return_outputs=True)
        active_want = (0x3C004000, 0x40004400, 0x00004100, 0x44004500)
        inactive_want = (0x3C003C00, 0x40004000, 0x42004200, 0x44004400)
        for lane in range(32):
            self.assertEqual(tuple(outs[lane]["%out_mrt0"]), active_want if lane < 8 else inactive_want, f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
