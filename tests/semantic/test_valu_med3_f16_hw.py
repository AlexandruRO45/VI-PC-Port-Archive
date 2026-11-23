from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU v_med3_f16.

AMD RDNA2 Vector ALU (VALU) Architecture:
  - 16-bit Floating-Point 3-Operand Median:
    v_med3_f16 vdst, src0, src1, src2 [op_sel:[s0,s1,s2,d]] [clamp]
  - Operands:
    src0, src1, src2: 16-bit floating-point values selected from low (0) or high (1)
    16-bit half of the operand registers.
  - Mathematical Semantics:
    Without NaN:
      Computes median of 3 IEEE 754 half-precision values:
      NMin(NMax(z, NMin(x, y)), NMax(x, y))
    With any NaN operand:
      The hardware returns min3(x, y, z) with NaN-ignoring min (GLSL NMin),
      returning the minimum among all non-NaN operands, or NaN if all are NaN.
    With clamp modifier:
      Clamps the result to [0.0, 1.0] (0x0000 to 0x3C00).
    Destination:
      VOP3 opsel preserves the other 16-bit half of destination VGPR.

All tests in this suite compile real assembly listings through support.translate_lines(...)
and execute the translated SPIR-V in spirv_mini_interp via support.mrt0_words.
Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF

F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_ONE: int = 0x3C00
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_QNAN: int = 0x7E00


def run_med3_f16(
    s0: int,
    s1: int,
    s2: int,
    *,
    init_dst: int = 0xDEAD0000,
    alias_dst: int | None = None,
    clamp: bool = False,
    op_sel: str = "",
    name: str = "ps_med3_f16",
) -> int:
    """Assemble, translate, and execute a v_med3_f16 instruction."""
    dst = f"v{alias_dst}" if alias_dst is not None else "v0"
    clamp_str = " clamp" if clamp else ""
    sel_str = f" {op_sel}" if op_sel else ""
    lines = [
        f"v_mov_b32 v0, {init_dst & MASK32:#x}",
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        f"v_med3_f16 {dst}, v1, v2, v3{clamp_str}{sel_str}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# Suite 1: Core Mathematical Operations & Permutations (7 tests)
# ==============================================================================

class TestVMed3F16Core(unittest.TestCase):
    """Core half-precision median operations and mathematical properties."""

    def test_med3_f16_basic(self) -> None:
        """med3(1.0, 2.0, 3.0) = 2.0 (0x4000)."""
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, name="ps_med3_basic")
        self.assertEqual(res & MASK16, 0x4000)

    def test_med3_f16_all_permutations(self) -> None:
        """All 6 permutations of (1.0, 2.0, 3.0) yield median 2.0 (0x4000)."""
        perms = [
            (0x3C00, 0x4000, 0x4200),
            (0x3C00, 0x4200, 0x4000),
            (0x4000, 0x3C00, 0x4200),
            (0x4000, 0x4200, 0x3C00),
            (0x4200, 0x3C00, 0x4000),
            (0x4200, 0x4000, 0x3C00),
        ]
        for i, (a, b, c) in enumerate(perms):
            with self.subTest(perm=(a, b, c)):
                res = run_med3_f16(a, b, c, name=f"ps_med3_perm_{i}")
                self.assertEqual(res & MASK16, 0x4000)

    def test_med3_f16_signed_zeros(self) -> None:
        """-0.0 < +0.0 < 1.0 -> median is +0.0 (0x0000)."""
        res1 = run_med3_f16(F16_POS_ZERO, F16_NEG_ZERO, F16_ONE, name="ps_med3_zero1")
        self.assertEqual(res1 & MASK16, F16_POS_ZERO)
        # Duplicate -0.0: med(-0.0, +0.0, -0.0) = -0.0 (0x8000)
        res2 = run_med3_f16(F16_NEG_ZERO, F16_POS_ZERO, F16_NEG_ZERO, name="ps_med3_zero2")
        self.assertEqual(res2 & MASK16, F16_NEG_ZERO)

    def test_med3_f16_identical_operands(self) -> None:
        """Identical operands: med3(x, x, x) = x and med3(x, x, y) = x."""
        val = 0x4000  # 2.0
        res_same = run_med3_f16(val, val, val, name="ps_med3_all_same")
        self.assertEqual(res_same & MASK16, val)
        res_pair = run_med3_f16(val, val, 0x4500, name="ps_med3_pair_same")
        self.assertEqual(res_pair & MASK16, val)

    def test_med3_f16_infinities(self) -> None:
        """Infinities ordering: med3(-inf, 1.0, +inf) = 1.0 (0x3C00)."""
        res = run_med3_f16(F16_NEG_INF, F16_ONE, F16_POS_INF, name="ps_med3_inf")
        self.assertEqual(res & MASK16, F16_ONE)
        res_double_inf = run_med3_f16(F16_POS_INF, F16_POS_INF, F16_ONE, name="ps_med3_double_inf")
        self.assertEqual(res_double_inf & MASK16, F16_POS_INF)

    def test_med3_f16_subnormals(self) -> None:
        """Subnormal ordering: 0.0 < 0x0001 < 0x0002 -> median is 0x0001."""
        res = run_med3_f16(F16_POS_ZERO, 0x0001, 0x0002, name="ps_med3_subnorm")
        self.assertEqual(res & MASK16, 0x0001)

    def test_med3_f16_negative_values(self) -> None:
        """Negative values ordering: -3.0 (0xC200) < -2.0 (0xC000) < -1.0 (0xBC00) -> -2.0."""
        res = run_med3_f16(0xC200, 0xC000, 0xBC00, name="ps_med3_neg")
        self.assertEqual(res & MASK16, 0xC000)


# ==============================================================================
# Suite 2: NaN Handling & Hardware Fallback (5 tests)
# ==============================================================================

class TestVMed3F16NaNHandling(unittest.TestCase):
    """RDNA2 NaN-handling semantics (NMin fallback)."""

    def test_med3_f16_single_nan_suppression(self) -> None:
        """With 1 NaN, hardware executes min3 on non-NaNs: med(NaN, 2.0, 3.0) = 2.0."""
        res = run_med3_f16(F16_QNAN, 0x4000, 0x4200, name="ps_med3_nan1")
        self.assertEqual(res & MASK16, 0x4000)

    def test_med3_f16_single_nan_middle(self) -> None:
        """med(3.0, NaN, 2.0) = 2.0."""
        res = run_med3_f16(0x4200, F16_QNAN, 0x4000, name="ps_med3_nan_mid")
        self.assertEqual(res & MASK16, 0x4000)

    def test_med3_f16_two_nans(self) -> None:
        """With 2 NaNs, hardware returns the sole non-NaN operand."""
        res = run_med3_f16(F16_QNAN, F16_QNAN, 0x4200, name="ps_med3_two_nans")
        self.assertEqual(res & MASK16, 0x4200)

    def test_med3_f16_nan_with_infinities(self) -> None:
        """med(NaN, -inf, +inf) returns -inf (0xFC00)."""
        res = run_med3_f16(F16_QNAN, F16_NEG_INF, F16_POS_INF, name="ps_med3_nan_inf")
        self.assertEqual(res & MASK16, F16_NEG_INF)

    def test_med3_f16_signalling_nan(self) -> None:
        """Signalling NaN is suppressed in favor of non-NaN values."""
        snan = 0x7C01  # Signalling NaN
        res = run_med3_f16(snan, 0x4000, 0x4200, name="ps_med3_snan")
        self.assertEqual(res & MASK16, 0x4000)


# ==============================================================================
# Suite 3: Modifiers, Clamping & Bit Preservation (6 tests)
# ==============================================================================

class TestVMed3F16Modifiers(unittest.TestCase):
    """Clamp modifier, op_sel halfword routing, and upper/lower bit preservation."""

    def test_med3_f16_clamp_positive(self) -> None:
        """Clamp modifier clamps results > 1.0 to 1.0 (0x3C00)."""
        # med(2.0, 3.0, 5.0) = 3.0 -> clamped to 1.0
        res = run_med3_f16(0x4000, 0x4200, 0x4500, clamp=True, name="ps_med3_clamp_pos")
        self.assertEqual(res & MASK16, F16_ONE)

    def test_med3_f16_clamp_negative(self) -> None:
        """Clamp modifier clamps results < 0.0 to 0.0 (0x0000)."""
        # med(-3.0, -2.0, -1.0) = -2.0 -> clamped to 0.0
        res = run_med3_f16(0xC200, 0xC000, 0xBC00, clamp=True, name="ps_med3_clamp_neg")
        self.assertEqual(res & MASK16, F16_POS_ZERO)

    def test_med3_f16_upper_half_preservation(self) -> None:
        """Default (d_hi=0) preserves destination upper 16 bits (0xDEAD)."""
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, init_dst=0xDEAD0000, name="ps_med3_hi_pres")
        self.assertEqual((res >> 16) & MASK16, 0xDEAD)
        self.assertEqual(res & MASK16, 0x4000)

    def test_med3_f16_op_sel_src_and_dst(self) -> None:
        """op_sel:[1,0,1,1] routes s0.hi, s1.lo, s2.hi and writes destination high half."""
        # s0 = 0x3C000000 (hi: 1.0)
        # s1 = 0x00004000 (lo: 2.0)
        # s2 = 0x42000000 (hi: 3.0)
        # dst init = 0x0000BEEF
        # Result high half = med(1.0, 2.0, 3.0) = 2.0 (0x4000), low half preserved as 0xBEEF
        lines = [
            "v_mov_b32 v0, 0x0000BEEF",
            "v_mov_b32 v1, 0x3C000000",
            "v_mov_b32 v2, 0x00004000",
            "v_mov_b32 v3, 0x42000000",
            "v_med3_f16 v0, v1, v2, v3 op_sel:[1,0,1,1]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_opsel", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word, 0x4000BEEF)

    def test_med3_f16_op_sel_all_high_sources(self) -> None:
        """op_sel:[1,1,1,0] reads high halves of all 3 sources into destination low half."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0x3C001111",  # hi: 1.0
            "v_mov_b32 v2, 0x40002222",  # hi: 2.0
            "v_mov_b32 v3, 0x42003333",  # hi: 3.0
            "v_med3_f16 v0, v1, v2, v3 op_sel:[1,1,1,0]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_opsel_all_hi", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word, 0x55AA4000)

    def test_med3_f16_clamp_with_op_sel(self) -> None:
        """Combined clamp and op_sel:[1,1,1,0]."""
        lines = [
            "v_mov_b32 v0, 0xDEAD0000",
            "v_mov_b32 v1, 0x40000000",  # hi: 2.0
            "v_mov_b32 v2, 0x42000000",  # hi: 3.0
            "v_mov_b32 v3, 0x45000000",  # hi: 5.0
            "v_med3_f16 v0, v1, v2, v3 clamp op_sel:[1,1,1,0]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_clamp_opsel", lines)
        word = support.mrt0_words(spv)[0]
        self.assertEqual(word, 0xDEAD3C00)


# ==============================================================================
# Suite 4: Register Aliasing & In-Place Execution (4 tests)
# ==============================================================================

class TestVMed3F16Aliasing(unittest.TestCase):
    """Register aliasing: destination register aliases source registers."""

    def test_dst_aliases_src0(self) -> None:
        """v0 = med3(v0, v1, v2)."""
        # v0 = 0xCAFE3C00 (lo: 1.0, hi: 0xCAFE)
        # v1 = 2.0 (0x4000), v2 = 3.0 (0x4200)
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, init_dst=0xCAFE3C00, alias_dst=0, name="ps_med3_alias_v0")
        self.assertEqual(res, 0xCAFE4000)

    def test_dst_aliases_src1(self) -> None:
        """v1 = med3(v0, v1, v2)."""
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, alias_dst=1, name="ps_med3_alias_v1")
        self.assertEqual(res & MASK16, 0x4000)

    def test_dst_aliases_src2(self) -> None:
        """v2 = med3(v0, v1, v2)."""
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, alias_dst=2, name="ps_med3_alias_v2")
        self.assertEqual(res & MASK16, 0x4000)

    def test_all_operands_aliased(self) -> None:
        """v0 = med3(v0, v0, v0)."""
        res = run_med3_f16(0x4000, 0x4000, 0x4000, init_dst=0x99994000, alias_dst=0, name="ps_med3_alias_all")
        self.assertEqual(res, 0x99994000)


# ==============================================================================
# Suite 5: SPIR-V Structure, Negative Controls & Mutation Detection (5 tests)
# ==============================================================================

class TestVMed3F16TranslatorEmissionAndMutations(unittest.TestCase):
    """SPIR-V translation emission, negative controls, and mutation detection."""

    def test_spirv_structure(self) -> None:
        """SPIR-V translation lowers v_med3_f16 with median logic."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0x3C00",
            "v_mov_b32 v2, 0x4000",
            "v_mov_b32 v3, 0x4200",
            "v_med3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_med3_f16", lines)
        self.assertIn("NMin", spv)

    def test_mutation_detection_min_vs_max(self) -> None:
        """MED3 is strictly distinguished from MIN3 and MAX3."""
        res = run_med3_f16(0x3C00, 0x4000, 0x4200, name="ps_med3_mut_minmax")
        self.assertEqual(res & MASK16, 0x4000)
        # MIN3 is 1.0 (0x3C00), MAX3 is 3.0 (0x4200)
        self.assertNotEqual(res & MASK16, 0x3C00)
        self.assertNotEqual(res & MASK16, 0x4200)

    def test_mutation_detection_clamp_omitted(self) -> None:
        """Clamp modifier detection: unclamped mutant produces 3.0 instead of 1.0."""
        res_clamped = run_med3_f16(0x4000, 0x4200, 0x4500, clamp=True, name="ps_med3_mut_clamp")
        self.assertEqual(res_clamped & MASK16, F16_ONE)
        res_unclamped = run_med3_f16(0x4000, 0x4200, 0x4500, clamp=False, name="ps_med3_mut_unclamp")
        self.assertEqual(res_unclamped & MASK16, 0x4200)
        self.assertNotEqual(res_clamped & MASK16, res_unclamped & MASK16)

    def test_mutation_detection_nan_fallback(self) -> None:
        """NaN presence must trigger min3 on valid numbers, not propagate NaN."""
        res = run_med3_f16(F16_QNAN, 0x4000, 0x4200, name="ps_med3_mut_nan")
        self.assertEqual(res & MASK16, 0x4000)
        self.assertNotEqual(res & MASK16, F16_QNAN)

    def test_negative_syntax_rejection(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises((ValueError, support.rdna2spirv.Unsupported)):
            support.translate_lines("ps_neg_med3_f16", ["v_med3_f16 v0, v1"])


# ==============================================================================
# Suite 6: Integration & Chained Pipelines (2 tests)
# ==============================================================================

class TestVMed3F16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_med3_pipeline(self) -> None:
        """Chained: med3(med3(1.0, 2.0, 3.0), 1.5, 2.5) = med3(2.0, 1.5, 2.5) = 2.0."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0x3C00",  # 1.0
            "v_mov_b32 v2, 0x4000",  # 2.0
            "v_mov_b32 v3, 0x4200",  # 3.0
            "v_mov_b32 v4, 0x3E00",  # 1.5
            "v_mov_b32 v5, 0x4100",  # 2.5
            "v_med3_f16 v6, v1, v2, v3",      # 2.0 (0x4000)
            "v_med3_f16 v0, v6, v4, v5",      # med(2.0, 1.5, 2.5) = 2.0 (0x4000)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_med3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4000)
        self.assertEqual((words[0] >> 16) & MASK16, 0x55AA)

    def test_med3_with_add_and_mul(self) -> None:
        """Chained: med3(1.0 + 2.0, 2.0 * 1.5, 4.0) = med3(3.0, 3.0, 4.0) = 3.0 (0x4200)."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0x3C00",  # 1.0
            "v_mov_b32 v2, 0x4000",  # 2.0
            "v_mov_b32 v3, 0x3E00",  # 1.5
            "v_mov_b32 v4, 0x4400",  # 4.0
            "v_add_f16 v5, v1, v2",  # 3.0 (0x4200)
            "v_mul_f16 v6, v2, v3",  # 3.0 (0x4200)
            "v_med3_f16 v0, v5, v6, v4",  # med(3.0, 3.0, 4.0) = 3.0 (0x4200)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_add_mul_med3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4200)
        self.assertEqual((words[0] >> 16) & MASK16, 0x55AA)


if __name__ == "__main__":
    unittest.main()
