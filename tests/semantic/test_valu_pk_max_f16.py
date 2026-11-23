from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU v_pk_max_f16.

AMD RDNA2 Vector ALU (VALU) Architecture:
  - 16-bit Packed Floating-Point 2-Operand Maximum:
    v_pk_max_f16 vdst, src0, src1 [op_sel:[s0,s1]] [op_sel_hi:[s0,s1]] [neg_lo:[s0,s1]] [neg_hi:[s0,s1]] [clamp]
  - Operands:
    src0, src1: 32-bit registers each containing two IEEE 754 half-precision (binary16) floats.
    op_sel selects low (0) or high (1) 16-bit halfword for low result.
    op_sel_hi selects low (0) or high (1) 16-bit halfword for high result.
    neg_lo / neg_hi selectively negate chosen halfwords before computation.
  - Mathematical Semantics:
    D.f16[0] = NMax(S0.f16[lo], S1.f16[lo])
    D.f16[1] = NMax(S0.f16[hi], S1.f16[hi])
    When one operand is NaN, NMax returns the other non-NaN operand (NaN suppression).
    If both operands are NaN, NaN is returned.
    Signed zeros: +0.0 > -0.0.
    Clamp modifier clamps both halfword results to [0.0, 1.0] (0x0000 to 0x3C00).

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

ONE: int = 0x3C00        # 1.0
TWO: int = 0x4000        # 2.0
THREE: int = 0x4200      # 3.0
HALF: int = 0x3800       # 0.5
QUARTER: int = 0x3400    # 0.25
MAXH: int = 0x7BFF       # 65504.0
INF: int = 0x7C00        # +Infinity
NINF: int = 0xFC00       # -Infinity
QNAN: int = 0x7E00       # Quiet NaN
MIN_NORMAL: int = 0x0400
MIN_SUB: int = 0x0001
MAX_SUB: int = 0x03FF
POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack two 16-bit halfwords into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def run_pk_max_f16(
    s0: int,
    s1: int,
    mods: str = "",
    clamp: bool = False,
    alias_dst: int | None = None,
    s0_is_sgpr: bool = False,
    s1_is_sgpr: bool = False,
    old_dst: int = 0xDEADBEEF,
    name: str = "ps_pk_max_f16",
) -> int:
    """Run v_pk_max_f16 on a single-lane pixel shader and return v0 mrt0 output."""
    clamp_str = " clamp" if clamp else ""
    mod_str = f" {mods}" if mods else ""
    full_mods = f"{mod_str}{clamp_str}"

    lines = [f"v_mov_b32 v0, {old_dst:#x}"]

    r0 = "s1" if s0_is_sgpr else "v1"
    lines.append(f"{'s' if s0_is_sgpr else 'v'}_mov_b32 {r0}, {s0:#x}")

    r1 = "s2" if s1_is_sgpr else "v2"
    lines.append(f"{'s' if s1_is_sgpr else 'v'}_mov_b32 {r1}, {s1:#x}")

    if alias_dst == 1:
        dst = "v1"
        lines.append(f"v_pk_max_f16 {dst}, {dst}, {r1}{full_mods}")
        lines.append(f"v_mov_b32 v0, {dst}")
    elif alias_dst == 2:
        dst = "v2"
        lines.append(f"v_pk_max_f16 {dst}, {r0}, {dst}{full_mods}")
        lines.append(f"v_mov_b32 v0, {dst}")
    else:
        lines.append(f"v_pk_max_f16 v0, {r0}, {r1}{full_mods}")

    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


# ==============================================================================
# Suite 1: Core Operands & Hand-Written Vectors (8 tests)
# ==============================================================================

class TestVPkMaxF16Core(unittest.TestCase):
    """Core 2-operand packed binary16 maximum evaluation."""

    def test_basic_positive_values(self) -> None:
        """max((1.0, 3.0), (2.0, 0.5)) = (2.0, 3.0)."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_pos")
        self.assertEqual(res, pack_f16x2(TWO, THREE))

    def test_positive_and_negative(self) -> None:
        """max((-1.0, +0.0), (1.0, -0.0)) = (1.0, +0.0)."""
        w0 = pack_f16x2(0xBC00, POS_ZERO)
        w1 = pack_f16x2(ONE, NEG_ZERO)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_pos_neg")
        self.assertEqual(res, pack_f16x2(ONE, POS_ZERO))

    def test_signed_zeros(self) -> None:
        """max((-0.0, -0.0), (+0.0, -0.0)) = (+0.0, -0.0)."""
        w0 = pack_f16x2(NEG_ZERO, NEG_ZERO)
        w1 = pack_f16x2(POS_ZERO, NEG_ZERO)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_zeros")
        self.assertEqual(res, pack_f16x2(POS_ZERO, NEG_ZERO))

    def test_identical_operands(self) -> None:
        """max((-0.0, +0.0), (-0.0, +0.0)) = (-0.0, +0.0)."""
        w0 = pack_f16x2(NEG_ZERO, POS_ZERO)
        w1 = pack_f16x2(NEG_ZERO, POS_ZERO)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_ident")
        self.assertEqual(res, pack_f16x2(NEG_ZERO, POS_ZERO))

    def test_both_zeros_positive_outcome(self) -> None:
        """max((+0.0, +0.0), (-0.0, -0.0)) = (+0.0, +0.0)."""
        w0 = pack_f16x2(POS_ZERO, POS_ZERO)
        w1 = pack_f16x2(NEG_ZERO, NEG_ZERO)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_both_zeros")
        self.assertEqual(res, pack_f16x2(POS_ZERO, POS_ZERO))

    def test_subnormals_and_normals(self) -> None:
        """max((-2^-24, 1 + 2^-10), (2^-24, 1.0)) = (2^-24, 1 + 2^-10)."""
        w0 = pack_f16x2(0x8001, 0x3C01)
        w1 = pack_f16x2(MIN_SUB, ONE)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_subnorm")
        self.assertEqual(res, pack_f16x2(MIN_SUB, 0x3C01))

    def test_max_finite_and_infinities(self) -> None:
        """max((65504, -inf), (+inf, 65504)) = (+inf, 65504)."""
        w0 = pack_f16x2(MAXH, NINF)
        w1 = pack_f16x2(INF, MAXH)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_inf")
        self.assertEqual(res, pack_f16x2(INF, MAXH))

    def test_negative_infinities_and_subnormals(self) -> None:
        """max((-inf, 2^-24), (-65504, 0)) = (-65504, 2^-24)."""
        w0 = pack_f16x2(NINF, MIN_SUB)
        w1 = pack_f16x2(0xFBFF, POS_ZERO)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_ninf_sub")
        self.assertEqual(res, pack_f16x2(0xFBFF, MIN_SUB))


# ==============================================================================
# Suite 2: Modifiers (op_sel, op_sel_hi, neg_lo, neg_hi, clamp) (9 tests)
# ==============================================================================

class TestVPkMaxF16Modifiers(unittest.TestCase):
    """OpSel, negation, and clamp modifier validation."""

    def test_clamp_modifier_basic(self) -> None:
        """Clamp modifier clamps negative results to 0.0 and >1.0 to 1.0."""
        # lo: max(1.5, 2.0) = 2.0 -> clamped to 1.0 (0x3C00)
        # hi: max(-0.5, -1.0) = -0.5 -> clamped to 0.0 (0x0000)
        w0 = pack_f16x2(0x3E00, 0xB800)
        w1 = pack_f16x2(TWO, 0xBC00)
        res = run_pk_max_f16(w0, w1, clamp=True, name="ps_pk_max_clamp_basic")
        self.assertEqual(res, pack_f16x2(ONE, POS_ZERO))

    def test_clamp_inside_range(self) -> None:
        """Values strictly inside [0.0, 1.0] remain unchanged under clamp."""
        # lo: max(0.5, 0.75) = 0.75 (0x3A00)
        # hi: max(0.25, 1.0) = 1.0 (0x3C00)
        w0 = pack_f16x2(HALF, QUARTER)
        w1 = pack_f16x2(0x3A00, ONE)
        res = run_pk_max_f16(w0, w1, clamp=True, name="ps_pk_max_clamp_in")
        self.assertEqual(res, pack_f16x2(0x3A00, ONE))

    def test_clamp_infinities(self) -> None:
        """Clamp maps -inf to 0.0 and +inf to 1.0."""
        w0 = pack_f16x2(NINF, INF)
        w1 = pack_f16x2(0xFBFF, MAXH)
        res = run_pk_max_f16(w0, w1, clamp=True, name="ps_pk_max_clamp_inf")
        self.assertEqual(res, pack_f16x2(POS_ZERO, ONE))

    def test_opsel_lo_selection(self) -> None:
        """op_sel:[1,0] selects s0.hi and s1.lo for low half."""
        # s0 = (1.0, 3.0), s1 = (0.5, -0.25)
        # lo: max(3.0, 0.5) = 3.0 (0x4200)
        # hi: max(3.0, -0.25) = 3.0 (0x4200)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="op_sel:[1,0]", name="ps_pk_max_opsel_lo")
        self.assertEqual(res, pack_f16x2(THREE, THREE))

    def test_opsel_cross_halves(self) -> None:
        """op_sel:[0,1] op_sel_hi:[0,0] cross-selects halfwords."""
        # lo: max(1.0, -0.25) = 1.0 (0x3C00)
        # hi: max(1.0, 0.5) = 1.0 (0x3C00)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="op_sel:[0,1] op_sel_hi:[0,0]", name="ps_pk_max_cross")
        self.assertEqual(res, pack_f16x2(ONE, ONE))

    def test_opsel_both_high_low(self) -> None:
        """op_sel:[1,1] op_sel_hi:[0,0] evaluates high on low half, low on high half."""
        # lo: max(3.0, -0.25) = 3.0 (0x4200)
        # hi: max(1.0, 0.5) = 1.0 (0x3C00)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="op_sel:[1,1] op_sel_hi:[0,0]", name="ps_pk_max_swap_hl")
        self.assertEqual(res, pack_f16x2(THREE, ONE))

    def test_neg_lo_modifier(self) -> None:
        """neg_lo:[1,0] negates s0 for low half."""
        # lo: max(-1.0, 0.5) = 0.5 (0x3800)
        # hi: max(3.0, -0.25) = 3.0 (0x4200)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="neg_lo:[1,0]", name="ps_pk_max_neg_lo")
        self.assertEqual(res, pack_f16x2(HALF, THREE))

    def test_neg_hi_modifier(self) -> None:
        """neg_hi:[0,1] negates s1 for high half."""
        # lo: max(1.0, 0.5) = 1.0 (0x3C00)
        # hi: max(3.0, -(-0.25) = 0.25) = 3.0 (0x4200)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="neg_hi:[0,1]", name="ps_pk_max_neg_hi")
        self.assertEqual(res, pack_f16x2(ONE, THREE))

    def test_neg_both_halves(self) -> None:
        """neg_lo:[1,1] neg_hi:[1,1] negates all source halves."""
        # lo: max(-1.0, -0.5) = -0.5 (0xB800)
        # hi: max(-3.0, 0.25) = 0.25 (0x3400)
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(HALF, 0xB400)
        res = run_pk_max_f16(w0, w1, mods="neg_lo:[1,1] neg_hi:[1,1]", name="ps_pk_max_neg_all")
        self.assertEqual(res, pack_f16x2(0xB800, 0x3400))


# ==============================================================================
# Suite 3: SGPR Source Registers (3 tests)
# ==============================================================================

class TestVPkMaxF16SgprSources(unittest.TestCase):
    """Source operand placement in scalar registers (SGPR)."""

    def test_sgpr_src0_vgpr_src1(self) -> None:
        """src0 in s1, src1 in v2."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, s0_is_sgpr=True, name="ps_pk_max_sgpr_s0")
        self.assertEqual(res, pack_f16x2(TWO, THREE))

    def test_vgpr_src0_sgpr_src1(self) -> None:
        """src0 in v1, src1 in s2."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, s1_is_sgpr=True, name="ps_pk_max_sgpr_s1")
        self.assertEqual(res, pack_f16x2(TWO, THREE))

    def test_sgpr_both_sources(self) -> None:
        """src0 in s1, src1 in s2."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, s0_is_sgpr=True, s1_is_sgpr=True, name="ps_pk_max_sgpr_both")
        self.assertEqual(res, pack_f16x2(TWO, THREE))


# ==============================================================================
# Suite 4: IEEE 754 Extremes & NaN Suppression (3 tests)
# ==============================================================================

class TestVPkMaxF16EdgeCases(unittest.TestCase):
    """NaN suppression and boundary extremes."""

    def test_nan_suppression_low_and_high(self) -> None:
        """NaN operands lose to valid numeric operands."""
        # lo: max(NaN, 1.0) = 1.0 (0x3C00)
        # hi: max(1.0, NaN) = 1.0 (0x3C00)
        w0 = pack_f16x2(QNAN, ONE)
        w1 = pack_f16x2(ONE, QNAN)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_nan_suppress")
        self.assertEqual(res, pack_f16x2(ONE, ONE))

    def test_both_nan_returns_nan(self) -> None:
        """When both operands are NaN, NaN is preserved."""
        # lo: max(NaN, NaN) = NaN (0x7E00)
        # hi: max(inf, 65504) = inf (0x7C00)
        w0 = pack_f16x2(QNAN, INF)
        w1 = pack_f16x2(QNAN, MAXH)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_both_nan")
        self.assertEqual(res, pack_f16x2(QNAN, INF))

    def test_subnormals_boundary(self) -> None:
        """Subnormal precision boundaries compared to smallest normal."""
        # lo: max(MIN_SUB, 2*MIN_SUB) = 2*MIN_SUB (0x0002)
        # hi: max(MAX_SUB, MIN_NORMAL) = MIN_NORMAL (0x0400)
        w0 = pack_f16x2(MIN_SUB, MAX_SUB)
        w1 = pack_f16x2(0x0002, MIN_NORMAL)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_subnorm_bounds")
        self.assertEqual(res, pack_f16x2(0x0002, MIN_NORMAL))


# ==============================================================================
# Suite 5: Register Aliasing & Destination Overwrite (3 tests)
# ==============================================================================

class TestVPkMaxF16RegisterAliasing(unittest.TestCase):
    """In-place destination register reuse and self-aliasing."""

    def test_aliasing_src0_with_dst(self) -> None:
        """v_pk_max_f16 v1, v1, v2 correctly computes when vdst == src0."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, alias_dst=1, name="ps_pk_max_alias_s0")
        self.assertEqual(res, pack_f16x2(TWO, THREE))

    def test_aliasing_src1_with_dst(self) -> None:
        """v_pk_max_f16 v2, v1, v2 correctly computes when vdst == src1."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, alias_dst=2, name="ps_pk_max_alias_s1")
        self.assertEqual(res, pack_f16x2(TWO, THREE))

    def test_same_register_both_sources(self) -> None:
        """v_pk_max_f16 v0, v1, v1: identical sources return the same value."""
        w0 = pack_f16x2(ONE, THREE)
        lines = [
            f"v_mov_b32 v1, {w0:#x}",
            "v_pk_max_f16 v0, v1, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_same_src", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, w0)


# ==============================================================================
# Suite 6: Structural Emission, Negative Controls & Mutations (4 tests)
# ==============================================================================

class TestVPkMaxF16EmissionAndMutations(unittest.TestCase):
    """SPIR-V structural inspection and mutant detection."""

    def test_spirv_structure_nmax(self) -> None:
        """SPIR-V translation emits GLSL.std.450 NMax."""
        lines = [
            f"v_mov_b32 v1, {pack_f16x2(ONE, THREE):#x}",
            f"v_mov_b32 v2, {pack_f16x2(TWO, HALF):#x}",
            "v_pk_max_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_struct", lines)
        self.assertIn("NMax", spv)

    def test_mutation_detection_min_vs_max(self) -> None:
        """A mutant that performs MIN instead of MAX must fail."""
        w0 = pack_f16x2(ONE, THREE)
        w1 = pack_f16x2(TWO, HALF)
        res = run_pk_max_f16(w0, w1, name="ps_pk_max_mut_min")
        self.assertEqual(res, pack_f16x2(TWO, THREE))
        # Mutant MIN produces (1.0, 0.5)
        self.assertNotEqual(res, pack_f16x2(ONE, HALF))

    def test_mutation_detection_clamp_omitted(self) -> None:
        """A mutant omitting clamp must produce >1.0 / <0.0 values."""
        w0 = pack_f16x2(0x3E00, 0xB800)  # 1.5, -0.5
        w1 = pack_f16x2(TWO, 0xBC00)     # 2.0, -1.0
        res_clamped = run_pk_max_f16(w0, w1, clamp=True, name="ps_pk_max_mut_clamp")
        res_unclamped = run_pk_max_f16(w0, w1, clamp=False, name="ps_pk_max_mut_unclamp")
        self.assertEqual(res_clamped, pack_f16x2(ONE, POS_ZERO))
        self.assertEqual(res_unclamped, pack_f16x2(TWO, 0xB800))
        self.assertNotEqual(res_clamped, res_unclamped)

    def test_negative_syntax_rejection(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises((IndexError, ValueError, support.rdna2spirv.Unsupported)):
            support.translate_lines("ps_neg_pk_max", ["v_pk_max_f16 v0"])


# ==============================================================================
# Suite 7: Chained Multi-Instruction Pipelines (2 tests)
# ==============================================================================

class TestVPkMaxF16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_pk_max_pipeline(self) -> None:
        """Chained: max(max((1.0, 3.0), (2.0, 0.5)), (1.5, 4.0)) = (2.0, 4.0)."""
        lines = [
            f"v_mov_b32 v1, {pack_f16x2(ONE, THREE):#x}",
            f"v_mov_b32 v2, {pack_f16x2(TWO, HALF):#x}",
            f"v_mov_b32 v3, {pack_f16x2(0x3E00, 0x4400):#x}",  # 1.5, 4.0
            "v_pk_max_f16 v0, v1, v2",                          # (2.0, 3.0)
            "v_pk_max_f16 v0, v0, v3",                          # (2.0, 4.0)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_chain", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, pack_f16x2(TWO, 0x4400))

    def test_mixed_arithmetic_pipeline(self) -> None:
        """Pipeline: v_pk_add_f16 followed by v_pk_max_f16."""
        # v1 = (1.0, 1.0), v2 = (0.5, 1.0) -> add gives (1.5, 2.0) = (0x3E00, 0x4000)
        # max with v3 = (2.0, 0.5) gives (2.0, 2.0) = (0x4000, 0x4000)
        lines = [
            f"v_mov_b32 v1, {pack_f16x2(ONE, ONE):#x}",
            f"v_mov_b32 v2, {pack_f16x2(HALF, ONE):#x}",
            f"v_mov_b32 v3, {pack_f16x2(TWO, HALF):#x}",
            "v_pk_add_f16 v0, v1, v2",
            "v_pk_max_f16 v0, v0, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_add_pipe", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, pack_f16x2(TWO, TWO))


if __name__ == "__main__":
    unittest.main()
