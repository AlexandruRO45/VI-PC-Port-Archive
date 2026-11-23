from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU v_max3_f16.

AMD RDNA2 Vector ALU (VALU) Architecture:
  - 16-bit Floating-Point 3-Operand Maximum:
    v_max3_f16 vdst, src0, src1, src2 [op_sel:[s0,s1,s2,d]] [clamp]
  - Operands:
    src0, src1, src2: 16-bit floating-point values selected from low (0) or high (1)
    16-bit half of the operand registers.
  - Mathematical Semantics:
    Computes IEEE 754 half-precision (binary16) maximum across three operands:
    NMax(NMax(x, y), z)
    With NaN operands:
      Follows IEEE 754-2008 maxNum / GLSL NMax rules:
      - Skips NaN operands: if 1 or 2 operands are NaN, returns the maximum of the non-NaN operands.
      - If all 3 operands are NaN, returns quiet NaN (0x7E00).
    With signed zeros:
      +0.0 (0x0000) beats -0.0 (0x8000).
    With clamp modifier:
      Clamps the result to [0.0, 1.0] (0x0000 to 0x3C00).
    Destination:
      VOP3 opsel preserves the other 16-bit half of destination VGPR (default preserves high half).

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

POS_ZERO: int = 0x0000       # +0.0
NEG_ZERO: int = 0x8000       # -0.0
ONE: int = 0x3C00            # 1.0
TWO: int = 0x4000            # 2.0
THREE: int = 0x4200          # 3.0
HALF: int = 0x3800           # 0.5
QUARTER: int = 0x3400        # 0.25
MAXH: int = 0x7BFF           # 65504.0 (max normal float16)
INF: int = 0x7C00            # +Infinity
NINF: int = 0xFC00           # -Infinity
QNAN: int = 0x7E00           # Quiet NaN
MIN_NORMAL: int = 0x0400     # Smallest positive normal (2^-14)
MIN_SUB: int = 0x0001        # Smallest positive subnormal (2^-24)
MAX_SUB: int = 0x03FF        # Largest positive subnormal


def run_max3_f16(
    s0: int,
    s1: int,
    s2: int,
    mods: str = "",
    clamp: bool = False,
    alias_dst: int | None = None,
    s0_is_sgpr: bool = False,
    s1_is_sgpr: bool = False,
    s2_is_sgpr: bool = False,
    old_dst: int = 0xABCD0000,
    name: str = "ps_max3_f16",
) -> int:
    """Run v_max3_f16 on a single-lane pixel shader and return v0 mrt0 output."""
    clamp_str = " clamp" if clamp else ""
    mod_str = f" {mods}" if mods else ""
    full_mods = f"{mod_str}{clamp_str}"

    lines = [f"v_mov_b32 v0, {old_dst:#x}"]

    r0 = "s1" if s0_is_sgpr else "v1"
    lines.append(f"{'s' if s0_is_sgpr else 'v'}_mov_b32 {r0}, {s0:#x}")

    r1 = "s2" if s1_is_sgpr else "v2"
    lines.append(f"{'s' if s1_is_sgpr else 'v'}_mov_b32 {r1}, {s1:#x}")

    r2 = "s3" if s2_is_sgpr else "v3"
    lines.append(f"{'s' if s2_is_sgpr else 'v'}_mov_b32 {r2}, {s2:#x}")

    if alias_dst == 1:
        lines.append(f"v_max3_f16 {r0}, {r0}, {r1}, {r2}{full_mods}")
        lines.append(f"v_mov_b32 v0, {r0}")
    elif alias_dst == 2:
        lines.append(f"v_max3_f16 {r1}, {r0}, {r1}, {r2}{full_mods}")
        lines.append(f"v_mov_b32 v0, {r1}")
    elif alias_dst == 3:
        lines.append(f"v_max3_f16 {r2}, {r0}, {r1}, {r2}{full_mods}")
        lines.append(f"v_mov_b32 v0, {r2}")
    else:
        lines.append(f"v_max3_f16 v0, {r0}, {r1}, {r2}{full_mods}")

    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


# ==============================================================================
# Suite 1: Core Operands & Numerical Vectors (7 tests)
# ==============================================================================

class TestVMax3F16Core(unittest.TestCase):
    """Core 3-operand floating-point maximum evaluations."""

    def test_max3_all_positive(self) -> None:
        """max(1.0, 2.0, 0.5) = 2.0 (0x4000)."""
        res = run_max3_f16(ONE, TWO, HALF, old_dst=0x12340000, name="ps_max3_pos")
        self.assertEqual(res, 0x12344000)

    def test_max3_all_negative(self) -> None:
        """max(-1.0, -2.0, -0.5) = -0.5 (0xB800)."""
        res = run_max3_f16(0xBC00, 0xC000, 0xB800, old_dst=0x12340000, name="ps_max3_neg")
        self.assertEqual(res, 0x1234B800)

    def test_max3_mixed_signs(self) -> None:
        """max(-2.0, 1.5, -0.5) = 1.5 (0x3E00)."""
        res = run_max3_f16(0xC000, 0x3E00, 0xB800, old_dst=0xCAFE0000, name="ps_max3_mixed")
        self.assertEqual(res, 0xCAFE3E00)

    def test_max3_signed_zeros(self) -> None:
        """max(-0.0, +0.0, -1.0) = +0.0 (0x0000 beats -0.0)."""
        res = run_max3_f16(NEG_ZERO, POS_ZERO, 0xBC00, old_dst=0x56780000, name="ps_max3_sz")
        self.assertEqual(res, 0x56780000)

    def test_max3_all_negative_zeros(self) -> None:
        """max(-0.0, -0.0, -0.0) = -0.0 (0x8000)."""
        res = run_max3_f16(NEG_ZERO, NEG_ZERO, NEG_ZERO, old_dst=0x56780000, name="ps_max3_all_neg_zeros")
        self.assertEqual(res, 0x56788000)

    def test_max3_identical_inputs(self) -> None:
        """max(2.5, 2.5, 2.5) = 2.5 (0x4100)."""
        res = run_max3_f16(0x4100, 0x4100, 0x4100, old_dst=0xDEAD0000, name="ps_max3_ident")
        self.assertEqual(res, 0xDEAD4100)

    def test_max3_subnormals(self) -> None:
        """max(MIN_SUB, MAX_SUB, -0.0) = MAX_SUB (0x03FF)."""
        res = run_max3_f16(MIN_SUB, MAX_SUB, NEG_ZERO, old_dst=0x12340000, name="ps_max3_subnorm")
        self.assertEqual(res, 0x123403FF)


# ==============================================================================
# Suite 2: IEEE 754 NaN Handling & Suppression (5 tests)
# ==============================================================================

class TestVMax3F16NaNHandling(unittest.TestCase):
    """NaN suppression and propagation semantics."""

    def test_nan_operand0_skipped(self) -> None:
        """max(NaN, 1.0, 0.5) = 1.0 (0x3C00)."""
        res = run_max3_f16(QNAN, ONE, HALF, old_dst=0x9ABC0000, name="ps_max3_nan0")
        self.assertEqual(res, 0x9ABC3C00)

    def test_nan_operand1_skipped(self) -> None:
        """max(1.0, NaN, 2.0) = 2.0 (0x4000)."""
        res = run_max3_f16(ONE, QNAN, TWO, old_dst=0x9ABC0000, name="ps_max3_nan1")
        self.assertEqual(res, 0x9ABC4000)

    def test_nan_operand2_skipped(self) -> None:
        """max(0.5, 1.5, NaN) = 1.5 (0x3E00)."""
        res = run_max3_f16(HALF, 0x3E00, QNAN, old_dst=0x9ABC0000, name="ps_max3_nan2")
        self.assertEqual(res, 0x9ABC3E00)

    def test_two_nan_operands_skipped(self) -> None:
        """max(NaN, NaN, 3.0) = 3.0 (0x4200)."""
        res = run_max3_f16(QNAN, QNAN, THREE, old_dst=0x9ABC0000, name="ps_max3_two_nan")
        self.assertEqual(res, 0x9ABC4200)

    def test_all_nan_operands_returns_nan(self) -> None:
        """max(NaN, NaN, NaN) = NaN (0x7E00)."""
        res = run_max3_f16(QNAN, QNAN, QNAN, old_dst=0x9ABC0000, name="ps_max3_all_nan")
        self.assertEqual(res & MASK16, QNAN)


# ==============================================================================
# Suite 3: Boundary Extremes & Infinities (4 tests)
# ==============================================================================

class TestVMax3F16Infinities(unittest.TestCase):
    """IEEE 754 infinities and boundary value validation."""

    def test_positive_infinity_beats_all(self) -> None:
        """max(+inf, 65504, 1.0) = +inf (0x7C00)."""
        res = run_max3_f16(INF, MAXH, ONE, old_dst=0x12340000, name="ps_max3_pos_inf")
        self.assertEqual(res, 0x12347C00)

    def test_negative_infinity_loses_to_all(self) -> None:
        """max(-inf, -65504, -2.0) = -2.0 (0xC000)."""
        res = run_max3_f16(NINF, 0xFBFF, 0xC000, old_dst=0x12340000, name="ps_max3_neg_inf")
        self.assertEqual(res, 0x1234C000)

    def test_positive_and_negative_infinities(self) -> None:
        """max(+inf, -inf, 0.0) = +inf (0x7C00)."""
        res = run_max3_f16(INF, NINF, POS_ZERO, old_dst=0x12340000, name="ps_max3_both_inf")
        self.assertEqual(res, 0x12347C00)

    def test_max_finite_float16(self) -> None:
        """max(65504, 65504, 65504) = 65504 (0x7BFF)."""
        res = run_max3_f16(MAXH, MAXH, MAXH, old_dst=0x12340000, name="ps_max3_maxh")
        self.assertEqual(res, 0x12347BFF)


# ==============================================================================
# Suite 4: Modifiers (clamp, op_sel, destination preservation) (6 tests)
# ==============================================================================

class TestVMax3F16Modifiers(unittest.TestCase):
    """VOP3 clamp modifier and op_sel halfword selection."""

    def test_clamp_modifier_above_one(self) -> None:
        """Clamp modifier clamps values > 1.0 to 1.0 (0x3C00)."""
        # max(2.0, 1.5, 0.5) = 2.0 -> clamped to 1.0 (0x3C00)
        res = run_max3_f16(TWO, 0x3E00, HALF, clamp=True, old_dst=0, name="ps_max3_clamp_hi")
        self.assertEqual(res, ONE)

    def test_clamp_modifier_below_zero(self) -> None:
        """Clamp modifier clamps values < 0.0 to 0.0 (0x0000)."""
        # max(-2.0, -1.5, -0.5) = -0.5 -> clamped to 0.0 (0x0000)
        res = run_max3_f16(0xC000, 0xBE00, 0xB800, clamp=True, old_dst=0, name="ps_max3_clamp_lo")
        self.assertEqual(res, POS_ZERO)

    def test_clamp_modifier_within_range(self) -> None:
        """Values inside [0.0, 1.0] remain unchanged under clamp."""
        # max(0.25, 0.5, 0.75) = 0.75 (0x3A00)
        res = run_max3_f16(QUARTER, HALF, 0x3A00, clamp=True, old_dst=0, name="ps_max3_clamp_in")
        self.assertEqual(res, 0x3A00)

    def test_opsel_source_selection(self) -> None:
        """op_sel:[1,0,0,0] reads high half of src0."""
        # src0 = 0x42000000 (3.0 in high half)
        # src1 = 0x3800 (0.5), src2 = 0x3400 (0.25)
        # max(3.0, 0.5, 0.25) = 3.0 (0x4200)
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x42000000",
            "v_mov_b32 v2, 0x3800",
            "v_mov_b32 v3, 0x3400",
            "v_max3_f16 v0, v1, v2, v3 op_sel:[1,0,0,0]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_opsel_src", lines)
        self.assertEqual(support.mrt0_words(spv)[0], THREE)

    def test_opsel_destination_high_half(self) -> None:
        """op_sel:[0,0,0,1] writes to high half and preserves low half."""
        lines = [
            "v_mov_b32 v0, 0x1234",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3800",
            "v_mov_b32 v3, 0x3400",
            "v_max3_f16 v0, v1, v2, v3 op_sel:[0,0,0,1]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_opsel_dst", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x40001234)

    def test_destination_preservation_default(self) -> None:
        """Default writes to low 16 bits and preserves high 16 bits."""
        for old_hi in (0xAAAA0000, 0x55550000, 0xFFFF0000, 0x00000000):
            with self.subTest(old_hi=hex(old_hi)):
                res = run_max3_f16(ONE, TWO, HALF, old_dst=old_hi, name=f"ps_pres_{old_hi:#x}")
                self.assertEqual(res & 0xFFFF0000, old_hi)
                self.assertEqual(res & MASK16, TWO)


# ==============================================================================
# Suite 5: Register Placement & SGPR Sources (4 tests)
# ==============================================================================

class TestVMax3F16RegisterPlacement(unittest.TestCase):
    """Source operand placement in scalar registers (SGPR) and constants."""

    def test_src0_in_sgpr(self) -> None:
        """src0 in s1, src1 and src2 in VGPRs."""
        res = run_max3_f16(ONE, TWO, HALF, s0_is_sgpr=True, old_dst=0x12340000, name="ps_max3_sgpr_s0")
        self.assertEqual(res, 0x12344000)

    def test_src1_in_sgpr(self) -> None:
        """src1 in s2, src0 and src2 in VGPRs."""
        res = run_max3_f16(ONE, TWO, HALF, s1_is_sgpr=True, old_dst=0x12340000, name="ps_max3_sgpr_s1")
        self.assertEqual(res, 0x12344000)

    def test_src2_in_sgpr(self) -> None:
        """src2 in s3, src0 and src1 in VGPRs."""
        res = run_max3_f16(ONE, TWO, HALF, s2_is_sgpr=True, old_dst=0x12340000, name="ps_max3_sgpr_s2")
        self.assertEqual(res, 0x12344000)

    def test_literal_constant_source(self) -> None:
        """src2 as constant 0."""
        lines = [
            "s_mov_b32 s5, 0x8000",  # -0.0
            "v_mov_b32 v4, 0xDEAD0000",
            "v_mov_b32 v1, 0xBC00",  # -1.0
            "v_max3_f16 v4, v1, s5, 0",  # 0 -> +0.0
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_sgpr_imm", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xDEAD0000)


# ==============================================================================
# Suite 6: Register Aliasing & Hazards (4 tests)
# ==============================================================================

class TestVMax3F16RegisterAliasing(unittest.TestCase):
    """In-place destination register reuse and operand aliasing."""

    def test_alias_dst_with_src0(self) -> None:
        """v_max3_f16 v1, v1, v2, v3 correctly overwrites src0."""
        res = run_max3_f16(0x12343C00, TWO, HALF, alias_dst=1, name="ps_max3_alias_s0")
        self.assertEqual(res, 0x12344000)

    def test_alias_dst_with_src1(self) -> None:
        """v_max3_f16 v2, v1, v2, v3 correctly overwrites src1."""
        res = run_max3_f16(ONE, 0x56784000, HALF, alias_dst=2, name="ps_max3_alias_s1")
        self.assertEqual(res, 0x56784000)

    def test_alias_dst_with_src2(self) -> None:
        """v_max3_f16 v3, v1, v2, v3 correctly overwrites src2."""
        res = run_max3_f16(ONE, TWO, 0xCAFE3800, alias_dst=3, name="ps_max3_alias_s2")
        self.assertEqual(res, 0xCAFE4000)

    def test_alias_dst_with_all_operands(self) -> None:
        """v_max3_f16 v1, v1, v1, v1 preserves high half and writes low."""
        lines = [
            "v_mov_b32 v1, 0x56783c00",
            "v_max3_f16 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_all", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x56783C00)


# ==============================================================================
# Suite 7: Emission, Negative Controls & Mutations (5 tests)
# ==============================================================================

class TestVMax3F16EmissionAndMutations(unittest.TestCase):
    """SPIR-V structural inspection, mutation pinning, and negative controls."""

    def test_spirv_structure_nmax(self) -> None:
        """Verify support.translate_lines emits GLSL NMax operations for v_max3_f16."""
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3c00",
            "v_mov_b32 v2, 0x4000",
            "v_mov_b32 v3, 0x3800",
            "v_max3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_emit_nmax", lines)
        self.assertIn("NMax", spv)
        self.assertNotIn("NMin", spv)

    def test_mutation_detection_min_vs_max(self) -> None:
        """A mutant that calculates MIN instead of MAX must fail."""
        res = run_max3_f16(ONE, TWO, HALF, old_dst=0, name="ps_mut_min_vs_max")
        self.assertEqual(res, TWO)
        # Mutant MIN produces HALF (0x3800)
        self.assertNotEqual(res, HALF)

    def test_mutation_detection_signed_zero_tie(self) -> None:
        """A mutant where -0.0 beats +0.0 must fail."""
        res = run_max3_f16(NEG_ZERO, POS_ZERO, 0xBC00, old_dst=0, name="ps_mut_sz_tie")
        self.assertEqual(res, POS_ZERO)
        self.assertNotEqual(res, NEG_ZERO)

    def test_mutation_detection_clamp_omitted(self) -> None:
        """A mutant omitting clamp produces unclamped > 1.0."""
        res_clamped = run_max3_f16(TWO, 0x3E00, HALF, clamp=True, old_dst=0, name="ps_mut_clamped")
        res_unclamped = run_max3_f16(TWO, 0x3E00, HALF, clamp=False, old_dst=0, name="ps_mut_unclamped")
        self.assertEqual(res_clamped, ONE)
        self.assertEqual(res_unclamped, TWO)
        self.assertNotEqual(res_clamped, res_unclamped)

    def test_negative_syntax_rejection(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises((IndexError, ValueError, support.rdna2spirv.Unsupported)):
            support.translate_lines("ps_neg_max3_f16", ["v_max3_f16 v0, v1, v2"])


# ==============================================================================
# Suite 8: Chained Multi-Instruction Pipelines (2 tests)
# ==============================================================================

class TestVMax3F16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_max3_pipeline(self) -> None:
        """Chained: max3(max3(1.0, 2.0, 0.5), 1.5, 3.0) = max3(2.0, 1.5, 3.0) = 3.0."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            f"v_mov_b32 v1, {ONE:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {HALF:#x}",
            f"v_mov_b32 v4, {0x3E00:#x}",   # 1.5
            f"v_mov_b32 v5, {THREE:#x}",    # 3.0
            "v_max3_f16 v0, v1, v2, v3",    # v0 = 0x55AA4000
            "v_max3_f16 v0, v0, v4, v5",    # v0 = 0x55AA4200
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_chain", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, 0x55AA4200)

    def test_mixed_arithmetic_pipeline(self) -> None:
        """Pipeline: v_add_f16 feeding into v_max3_f16."""
        # v1 = 1.0, v2 = 0.5 -> add gives 1.5 (0x3E00)
        # max3(1.5, 2.0, 0.25) = 2.0 (0x4000)
        lines = [
            "v_mov_b32 v0, 0xBEEF0000",
            f"v_mov_b32 v1, {ONE:#x}",
            f"v_mov_b32 v2, {HALF:#x}",
            f"v_mov_b32 v3, {TWO:#x}",
            f"v_mov_b32 v4, {QUARTER:#x}",
            "v_add_f16 v0, v1, v2",
            "v_max3_f16 v0, v0, v3, v4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_add_pipe", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, 0xBEEF4000)


if __name__ == "__main__":
    unittest.main()
