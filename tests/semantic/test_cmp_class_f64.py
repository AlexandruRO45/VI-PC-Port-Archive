"""Semantic value tests for RDNA2 binary64 class compares: v_cmp_class_f64 and v_cmpx_class_f64.

Measured natively on RDNA2 (gfx1036, wave32 & wave64, flush & ieee modes via tools/hw_oracle/):
  - Same 10 IEEE-754 class bits in S1[9:0] as v_cmp_class_f32 (bits [31:10] ignored):
      bit 0 (0x001): sNaN         bit 5 (0x020): -0.0
      bit 1 (0x002): qNaN         bit 6 (0x040): +0.0
      bit 2 (0x004): -inf         bit 7 (0x080): +subnormal
      bit 3 (0x008): -normal      bit 8 (0x100): +normal
      bit 4 (0x010): -subnormal   bit 9 (0x200): +inf
  - VOP3 abs/neg operate bitwise on hi[31], preserving signalling NaNs and subnormals.
  - Inline integers -16..64 sign-extend to 64 bits (1 -> +subnormal, -1 -> qNaN); 32-bit literals occupy the upper
    32 bits (lo = 0, hi = lit: 0x7ff00000 -> +inf, 0x80000000 -> -0.0, 0x7ff00001 -> sNaN); inline floats expand to
    exact IEEE-754 binary64; null reads as (0, 0) (+0.0).
  - Translated with pure 32-bit uint ops (no OpCapability Float64 required).
"""
import os
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
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CLASS_SNAN = 1 << 0
CLASS_QNAN = 1 << 1
CLASS_NINF = 1 << 2
CLASS_NNORM = 1 << 3
CLASS_NSUBNORM = 1 << 4
CLASS_NZERO = 1 << 5
CLASS_PZERO = 1 << 6
CLASS_PSUBNORM = 1 << 7
CLASS_PNORM = 1 << 8
CLASS_PINF = 1 << 9


def ref_class_f64_bit(lo: int, hi: int) -> int:
    lo &= 0xFFFFFFFF
    hi &= 0xFFFFFFFF
    neg = bool(hi >> 31)
    exp_ = (hi >> 20) & 0x7FF
    man_hi = hi & 0xFFFFF
    man_zero = (man_hi == 0) and (lo == 0)
    if exp_ == 0x7FF:
        if man_zero:
            return CLASS_NINF if neg else CLASS_PINF
        return CLASS_QNAN if (man_hi & 0x80000) else CLASS_SNAN
    if exp_ == 0:
        if man_zero:
            return CLASS_NZERO if neg else CLASS_PZERO
        return CLASS_NSUBNORM if neg else CLASS_PSUBNORM
    return CLASS_NNORM if neg else CLASS_PNORM


_TMP = None
_SRC = None


def setUpModule():
    global _TMP, _SRC
    _TMP = tempfile.TemporaryDirectory()
    dd = os.path.join(_TMP.name, "disasm")
    os.mkdir(dd)
    _SRC = os.path.join(dd, "ps_cmp_class_f64.s")
    with open(os.path.join(_TMP.name, "ps_cmp_class_f64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        _TMP.cleanup()
        _TMP = None


def translate_ps(lines: list[str], out_regs=("v0", "v1", "v2", "v3")) -> str:
    if _TMP is None:
        setUpModule()
    asm = ("; test_cmp_class_f64\n"
           + "".join(f"\t{ln}\n" for ln in lines)
           + f"\texp mrt0 {out_regs[0]}, {out_regs[1]}, {out_regs[2]}, {out_regs[3]} done vm\n\ts_endpgm\n")
    with open(_SRC, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC)


def run_ps(lines: list[str], out_regs=("v0", "v1", "v2", "v3")) -> tuple[int, int, int, int]:
    spv = translate_ps(lines, out_regs=out_regs)
    return tuple(si.run_fragment(spv)["%out_mrt0"])


def classify_all_10(insn: str, setup: list[str] | None = None, vcc_reg: str = "vcc_lo") -> int:
    lines = list(setup or []) + ["v_mov_b32 v10, 0"]
    for bit in range(10):
        lines += [
            f"v_mov_b32 v6, {1 << bit:#x}",
            insn,
            f"v_cndmask_b32_e64 v11, 0, {1 << bit:#x}, {vcc_reg}",
            "v_or_b32 v10, v10, v11",
        ]
    return run_ps(lines, ("v10", "v10", "v10", "v10"))[0]


class CmpClassF64Tests(unittest.TestCase):
    def test_all_ten_classes_and_low_word_significand_bits(self):
        samples = [
            ("snan_lo1", 0x00000001, 0x7FF00000, CLASS_SNAN),
            ("snan_lo_msb", 0x80000000, 0x7FF00000, CLASS_SNAN),
            ("snan_hi1", 0x00000000, 0x7FF00001, CLASS_SNAN),
            ("neg_snan", 0x00000001, 0xFFF00000, CLASS_SNAN),
            ("qnan_pos", 0x00000000, 0x7FF80000, CLASS_QNAN),
            ("qnan_neg", 0xDEADBEEF, 0xFFF80001, CLASS_QNAN),
            ("neg_inf", 0x00000000, 0xFFF00000, CLASS_NINF),
            ("neg_norm", 0x00000000, 0xBFF00000, CLASS_NNORM),
            ("neg_subnorm_lo1", 0x00000001, 0x80000000, CLASS_NSUBNORM),
            ("neg_subnorm_max", 0xFFFFFFFF, 0x800FFFFF, CLASS_NSUBNORM),
            ("neg_zero", 0x00000000, 0x80000000, CLASS_NZERO),
            ("pos_zero", 0x00000000, 0x00000000, CLASS_PZERO),
            ("pos_subnorm_lo1", 0x00000001, 0x00000000, CLASS_PSUBNORM),
            ("pos_subnorm_max", 0xFFFFFFFF, 0x000FFFFF, CLASS_PSUBNORM),
            ("pos_norm_min", 0x00000000, 0x00100000, CLASS_PNORM),
            ("pos_norm_one", 0x00000000, 0x3FF00000, CLASS_PNORM),
            ("pos_inf", 0x00000000, 0x7FF00000, CLASS_PINF),
        ]
        for label, lo, hi, want in samples:
            with self.subTest(label=label, lo=hex(lo), hi=hex(hi)):
                self.assertEqual(ref_class_f64_bit(lo, hi), want)
                got = classify_all_10(
                    "v_cmp_class_f64_e64 vcc_lo, v[0:1], v6",
                    [f"v_mov_b32 v0, {lo:#x}", f"v_mov_b32 v1, {hi:#x}"],
                )
                self.assertEqual(got, want)

    def test_vop3_neg_abs_modifiers_preserve_snan_and_flip_signs(self):
        cases = [
            ("snan_neg", 0x00000001, 0x7FF00000, "-v[0:1]", CLASS_SNAN),
            ("snan_abs", 0x00000001, 0xFFF00000, "|v[0:1]|", CLASS_SNAN),
            ("snan_neg_abs", 0x00000001, 0x7FF00000, "-|v[0:1]|", CLASS_SNAN),
            ("qnan_neg", 0x00000000, 0x7FF80000, "-v[0:1]", CLASS_QNAN),
            ("neg_zero_abs", 0x00000000, 0x80000000, "|v[0:1]|", CLASS_PZERO),
            ("pos_zero_neg", 0x00000000, 0x00000000, "-v[0:1]", CLASS_NZERO),
            ("neg_subnorm_abs", 0x00000001, 0x80000000, "|v[0:1]|", CLASS_PSUBNORM),
            ("pos_subnorm_neg", 0x00000001, 0x00000000, "-v[0:1]", CLASS_NSUBNORM),
            ("pos_inf_neg_abs", 0x00000000, 0x7FF00000, "-|v[0:1]|", CLASS_NINF),
            ("neg_inf_neg", 0x00000000, 0xFFF00000, "-v[0:1]", CLASS_PINF),
        ]
        for label, lo, hi, src, want in cases:
            with self.subTest(label=label, src=src):
                got = classify_all_10(
                    f"v_cmp_class_f64_e64 vcc_lo, {src}, v6",
                    [f"v_mov_b32 v0, {lo:#x}", f"v_mov_b32 v1, {hi:#x}"],
                )
                self.assertEqual(got, want)

    def test_inline_constants_literals_sgprs_null_and_upper_mask_bits(self):
        # Verified natively on gfx1036 (wave32 & wave64):
        const_cases = [
            ("int_0", "v_cmp_class_f64_e32 vcc_lo, 0, v6", [], CLASS_PZERO),
            ("int_1", "v_cmp_class_f64_e32 vcc_lo, 1, v6", [], CLASS_PSUBNORM),
            ("int_64", "v_cmp_class_f64_e32 vcc_lo, 64, v6", [], CLASS_PSUBNORM),
            ("int_65_lit", "v_cmp_class_f64_e32 vcc_lo, 65, v6", [], CLASS_PSUBNORM),
            ("int_m1", "v_cmp_class_f64_e32 vcc_lo, -1, v6", [], CLASS_QNAN),
            ("int_m16", "v_cmp_class_f64_e32 vcc_lo, -16, v6", [], CLASS_QNAN),
            ("int_m17_lit", "v_cmp_class_f64_e32 vcc_lo, -17, v6", [], CLASS_QNAN),
            ("lit_pos_inf", "v_cmp_class_f64_e32 vcc_lo, 0x7ff00000, v6", [], CLASS_PINF),
            ("lit_neg_inf", "v_cmp_class_f64_e32 vcc_lo, 0xfff00000, v6", [], CLASS_NINF),
            ("lit_neg_zero", "v_cmp_class_f64_e32 vcc_lo, 0x80000000, v6", [], CLASS_NZERO),
            ("lit_snan", "v_cmp_class_f64_e32 vcc_lo, 0x7ff00001, v6", [], CLASS_SNAN),
            ("lit_pos_norm", "v_cmp_class_f64_e32 vcc_lo, 0x3ff00000, v6", [], CLASS_PNORM),
            ("lit_0xffffffff", "v_cmp_class_f64_e32 vcc_lo, 0xffffffff, v6", [], CLASS_QNAN),
            ("flt_0_5", "v_cmp_class_f64_e32 vcc_lo, 0.5, v6", [], CLASS_PNORM),
            ("flt_neg_0_5", "v_cmp_class_f64_e32 vcc_lo, -0.5, v6", [], CLASS_NNORM),
            ("flt_abs_neg1", "v_cmp_class_f64_e64 vcc_lo, |-1.0|, v6", [], CLASS_PNORM),
            ("int_neg_abs_1", "v_cmp_class_f64_e64 vcc_lo, -|1|, v6", [], CLASS_NSUBNORM),
            ("null_src", "v_cmp_class_f64_e32 vcc_lo, null, v6", [], CLASS_PZERO),
            ("sgpr_snan", "v_cmp_class_f64_e64 vcc_lo, s[10:11], v6",
             ["s_mov_b32 s10, 1", "s_mov_b32 s11, 0x7ff00000"], CLASS_SNAN),
        ]
        for label, insn, setup, want in const_cases:
            with self.subTest(label=label):
                self.assertEqual(classify_all_10(insn, setup), want)

        # Mask bits [31:10] are ignored; explicit SGPR destination and wave64 vcc/s[4:5] work.
        got = run_ps([
            "s_mov_b64 exec, -1",
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3ff00000",
            "v_mov_b32 v6, 0xfffffc00",
            "v_cmp_class_f64_e64 s[4:5], v[0:1], v6",
            "v_cndmask_b32_e64 v10, 0, 1, s[4:5]",
            "v_mov_b32 v6, 0xfffffd00",
            "v_cmp_class_f64_e64 s[4:5], v[0:1], v6",
            "v_cndmask_b32_e64 v11, 0, 1, s[4:5]",
            "v_cmp_class_f64_e32 vcc, 0x7ff00000, v6",
            "v_cndmask_b32_e32 v12, 0, 1, vcc",
            "v_mov_b32 v6, 0x200",
            "v_cmp_class_f64_e32 vcc, 0x7ff00000, v6",
            "v_cndmask_b32_e32 v13, 0, 1, vcc",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (0, 1, 0, 1))

    def test_cmpx_updates_exec_preserves_vcc_and_inactive_lane_writes_zero(self):
        # v_cmpx_class_f64 keeps EXEC on match and preserves VCC; kills EXEC on mismatch.
        got = run_ps([
            "v_mov_b32 v10, 0x11111111",
            "v_mov_b32 v11, 0x22222222",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
            "s_mov_b32 vcc_lo, -1",
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0x7ff00000",          # sNaN
            "v_cmpx_class_f64_e64 -v[0:1], 1",   # stays active (sNaN preserved under neg)
            "v_mov_b32 v10, 0xaaaa0001",
            "v_cndmask_b32_e32 v12, 0, 0x7777, vcc_lo",
            "v_cmpx_class_f64_e64 -v[0:1], 2",   # qNaN mask -> false, lane goes inactive
            "v_mov_b32 v11, 0xdeadbeef",         # skipped
            "v_cmp_class_f64_e64 s6, v[0:1], 0x3ff",  # inactive lane writes 0 into s6
            "s_mov_b32 exec_lo, 1",
            "v_cndmask_b32_e64 v13, 0, 1, s6",
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(got, (0xAAAA0001, 0x22222222, 0x7777, 0))

    def test_does_not_require_float64_capability(self):
        spv = translate_ps([
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0x7ff00000",
            "v_cmp_class_f64_e64 vcc_lo, -v[0:1], 1",
            "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
        ])
        self.assertNotIn("OpCapability Float64", spv)


if __name__ == "__main__":
    unittest.main()
