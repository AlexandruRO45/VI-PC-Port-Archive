from __future__ import annotations

"""Hardware-measured and translator-backed semantic tests for v_ceil_f32.

RDNA2 VALU instruction v_ceil_f32:
Computes the ceiling of a 32-bit floating-point input in the MODE inferred for the
game (IEEE 0, f32 denormals flushed):
    - Denormal operands are flushed to zero with preserving sign:
      ceil(+1e-45) = +0.0, ceil(-1e-45) = -0.0
    - NaNs and Infinities pass through unchanged (signalling NaNs preserved under IEEE 0)
    - Fractional values rounded upwards to smallest integer not less than input
    - Modifiers supported: VOP3 clamp ([0.0, 1.0]), input negation (-v), input absolute (|v|)

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

MASK32: int = 0xFFFFFFFF

POS_ZERO: int = 0x00000000       # +0.0
NEG_ZERO: int = 0x80000000       # -0.0
POS_ONE: int = 0x3F800000        # +1.0
NEG_ONE: int = 0xBF800000        # -1.0
POS_TWO: int = 0x40000000        # +2.0
NEG_TWO: int = 0xC0000000        # -2.0
POS_THREE: int = 0x40400000      # +3.0
POS_INF: int = 0x7F800000        # +Infinity
NEG_INF: int = 0xFF800000        # -Infinity
QNAN: int = 0x7FC00000           # Quiet NaN
SNAN: int = 0x7FA00000           # Signalling NaN
SMALLEST_POS_NORMAL: int = 0x00800000
SMALLEST_NEG_NORMAL: int = 0x80800000


def run_ceil_f32(
    src_val: int,
    mods: str = "",
    clamp: bool = False,
    alias_dst: bool = False,
    src_is_sgpr: bool = False,
    old_dst: int = 0xDEADBEEF,
    name: str = "ps_ceil_f32",
) -> int:
    """Run v_ceil_f32 on a single-lane pixel shader and return v0 mrt0 output."""
    clamp_str = " clamp" if clamp else ""
    mod_str = f" {mods}" if mods else ""
    full_mods = f"{mod_str}{clamp_str}"

    lines = [f"v_mov_b32 v0, {old_dst:#x}"]

    r_src = "s1" if src_is_sgpr else "v1"
    lines.append(f"{'s' if src_is_sgpr else 'v'}_mov_b32 {r_src}, {src_val:#x}")

    if alias_dst:
        lines.append(f"v_ceil_f32 v0, v0{full_mods}")
    else:
        lines.append(f"v_ceil_f32 v0, {r_src}{full_mods}")

    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


# ==============================================================================
# Suite 1: Core Operands & Numerical Vectors (6 tests)
# ==============================================================================

class TestVCeilF32Core(unittest.TestCase):
    """Core positive, negative, integral, and subnormal floating-point ceiling evaluations."""

    def test_positive_fractional_inputs(self) -> None:
        """Verify ceiling for positive fractional values."""
        cases = [
            (0x3FC00000, POS_TWO),       # +1.5 -> +2.0
            (0x3E800000, POS_ONE),       # +0.25 -> +1.0
            (0x40066666, POS_THREE),     # +2.1 -> +3.0
            (0x3F7FFFFF, POS_ONE),       # +0.99999994 -> +1.0
        ]
        for idx, (inp, expected) in enumerate(cases):
            with self.subTest(inp=hex(inp)):
                res = run_ceil_f32(inp, name=f"ps_ceil_pos_{idx}")
                self.assertEqual(res, expected)

    def test_negative_fractional_inputs(self) -> None:
        """Verify ceiling for negative fractional values (ceil rounds towards +inf)."""
        cases = [
            (0xBF000000, NEG_ZERO),      # -0.5 -> -0.0
            (0xBFC00000, NEG_ONE),       # -1.5 -> -1.0
            (0xBE800000, NEG_ZERO),      # -0.25 -> -0.0
            (0xC0300000, NEG_TWO),       # -2.75 -> -2.0
            (0xBF7FFFFF, NEG_ZERO),      # -0.99999994 -> -0.0
        ]
        for idx, (inp, expected) in enumerate(cases):
            with self.subTest(inp=hex(inp)):
                res = run_ceil_f32(inp, name=f"ps_ceil_neg_{idx}")
                self.assertEqual(res, expected)

    def test_integral_inputs_unchanged(self) -> None:
        """Exact integral values are preserved identically."""
        cases = [
            POS_ONE,                     # +1.0
            NEG_ONE,                     # -1.0
            0x40800000,                  # +4.0
            0xC0A00000,                  # -5.0
        ]
        for idx, val in enumerate(cases):
            with self.subTest(val=hex(val)):
                res = run_ceil_f32(val, name=f"ps_ceil_int_{idx}")
                self.assertEqual(res, val)

    def test_signed_zeros_preserved(self) -> None:
        """Signed zero identities (+0.0 and -0.0) are preserved."""
        for idx, val in enumerate([POS_ZERO, NEG_ZERO]):
            with self.subTest(val=hex(val)):
                res = run_ceil_f32(val, name=f"ps_ceil_zero_{idx}")
                self.assertEqual(res, val)

    def test_subnormals_flushed_to_signed_zero(self) -> None:
        """In game mode (IEEE 0), subnormals are flushed to zero preserving sign."""
        cases = [
            (0x00000001, POS_ZERO),      # smallest pos subnormal -> +0.0
            (0x80000001, NEG_ZERO),      # smallest neg subnormal -> -0.0
            (0x00400000, POS_ZERO),      # mid pos subnormal -> +0.0
            (0x80400000, NEG_ZERO),      # mid neg subnormal -> -0.0
            (SMALLEST_POS_NORMAL, POS_ONE),  # smallest normal float > 0 -> +1.0
            (SMALLEST_NEG_NORMAL, NEG_ZERO), # smallest negative normal float -> -0.0
        ]
        for idx, (inp, expected) in enumerate(cases):
            with self.subTest(inp=hex(inp)):
                res = run_ceil_f32(inp, name=f"ps_ceil_subnorm_{idx}")
                self.assertEqual(res, expected)

    def test_large_integers_preserved(self) -> None:
        """Large numbers with no fractional bits (>= 2^24) remain unchanged."""
        cases = [
            0x4B800001,                  # large positive
            0xCB000001,                  # large negative
        ]
        for idx, val in enumerate(cases):
            with self.subTest(val=hex(val)):
                res = run_ceil_f32(val, name=f"ps_ceil_large_{idx}")
                self.assertEqual(res, val)


# ==============================================================================
# Suite 2: Boundary Extremes, Infinities & NaNs (4 tests)
# ==============================================================================

class TestVCeilF32InfinitiesAndNaNs(unittest.TestCase):
    """IEEE 754 infinities and NaN pass-through behavior."""

    def test_positive_infinity_passthrough(self) -> None:
        """+inf passes through unchanged."""
        res = run_ceil_f32(POS_INF, name="ps_ceil_pos_inf")
        self.assertEqual(res, POS_INF)

    def test_negative_infinity_passthrough(self) -> None:
        """-inf passes through unchanged."""
        res = run_ceil_f32(NEG_INF, name="ps_ceil_neg_inf")
        self.assertEqual(res, NEG_INF)

    def test_qnan_passthrough(self) -> None:
        """Quiet NaN passes through unchanged."""
        res = run_ceil_f32(QNAN, name="ps_ceil_qnan")
        self.assertEqual(res, QNAN)

    def test_snan_passthrough(self) -> None:
        """Signalling NaN is preserved in IEEE 0 mode."""
        res = run_ceil_f32(SNAN, name="ps_ceil_snan")
        self.assertEqual(res, SNAN)


# ==============================================================================
# Suite 3: Modifiers (clamp, e64 neg, e64 abs) (4 tests)
# ==============================================================================

class TestVCeilF32Modifiers(unittest.TestCase):
    """VOP3 clamp modifier and e64 input negation/absolute modifiers."""

    def test_vop3_clamp_modifier_high(self) -> None:
        """Clamp modifier clamps results > 1.0 down to 1.0."""
        # 2.3 -> ceil = 3.0 -> clamped to 1.0 (0x3F800000)
        res = run_ceil_f32(0x40133333, clamp=True, name="ps_ceil_clamp_hi")
        self.assertEqual(res, POS_ONE)

    def test_vop3_clamp_modifier_low(self) -> None:
        """Clamp modifier clamps results < 0.0 up to 0.0."""
        # -1.5 -> ceil = -1.0 -> clamped to 0.0 (0x00000000)
        res = run_ceil_f32(0xBFC00000, clamp=True, name="ps_ceil_clamp_lo")
        self.assertEqual(res, POS_ZERO)

    def test_vop3_neg_modifier(self) -> None:
        """e64 neg modifier negates input before computing ceil."""
        # +1.5 -> negated to -1.5 -> ceil(-1.5) = -1.0 (0xBF800000)
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3FC00000",
            "v_ceil_f32_e64 v0, -v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_e64_neg", lines)
        self.assertEqual(support.mrt0_words(spv)[0], NEG_ONE)

    def test_vop3_abs_modifier(self) -> None:
        """e64 abs modifier computes absolute value before ceil."""
        # |-1.5| = 1.5 -> ceil(1.5) = 2.0 (0x40000000)
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0xBFC00000",
            "v_ceil_f32_e64 v0, |v1|",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_e64_abs", lines)
        self.assertEqual(support.mrt0_words(spv)[0], POS_TWO)


# ==============================================================================
# Suite 4: Register Aliasing & Source Placement (3 tests)
# ==============================================================================

class TestVCeilF32RegisterAliasing(unittest.TestCase):
    """In-place destination register reuse and SGPR source operands."""

    def test_in_place_destination_equals_source(self) -> None:
        """v_ceil_f32 v0, v0 in-place computation succeeds."""
        # 2.3 -> ceil = 3.0 (0x40400000)
        res = run_ceil_f32(0x40133333, alias_dst=True, old_dst=0x40133333, name="ps_ceil_alias")
        self.assertEqual(res, POS_THREE)

    def test_preservation_of_unrelated_registers(self) -> None:
        """Unrelated registers are untouched during instruction execution."""
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 0x40900000",  # 4.5 -> ceil = 5.0 (0x40A00000)
            "v_mov_b32 v2, 0x2222",
            "v_mov_b32 v3, 0x3333",
            "v_ceil_f32 v1, v1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_pres_regs", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x1111)
        self.assertEqual(words[1], 0x40A00000)
        self.assertEqual(words[2], 0x2222)
        self.assertEqual(words[3], 0x3333)

    def test_sgpr_source_operand(self) -> None:
        """Source operand read from SGPR register."""
        res = run_ceil_f32(0x3FC00000, src_is_sgpr=True, name="ps_ceil_sgpr")
        self.assertEqual(res, POS_TWO)


# ==============================================================================
# Suite 5: Condition Codes, Comparisons & Pipelines (3 tests)
# ==============================================================================

class TestVCeilF32ConditionCodesAndPipelines(unittest.TestCase):
    """Comparisons on ceil results and multi-instruction pipelines."""

    def test_comparison_and_cndmask_selection(self) -> None:
        """ceil(-0.5) is -0.0. Compare with 0.0 (in IEEE 754, -0.0 == +0.0 is true)."""
        lines = [
            "v_mov_b32 v1, 0xBF000000",
            "v_ceil_f32 v0, v1",
            "v_cmp_eq_f32 vcc, v0, 0.0",
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",
            "exp mrt0 v0, v1, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_cmp_pipe", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], NEG_ZERO)
        self.assertEqual(words[1], 0x2222)

    def test_sequential_chained_ceil(self) -> None:
        """Sequential chained ceil operations."""
        # 0.25 -> ceil = 1.0 (0x3F800000); ceil(1.0) = 1.0
        lines = [
            "v_mov_b32 v1, 0x3E800000",
            "v_ceil_f32 v0, v1",
            "v_ceil_f32 v2, v0",
            "exp mrt0 v0, v2, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_chain", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], POS_ONE)
        self.assertEqual(words[1], POS_ONE)

    def test_arithmetic_pipe_add_then_ceil(self) -> None:
        """Pipeline: v_add_f32 feeding into v_ceil_f32."""
        # 1.0 + 0.25 = 1.25 -> ceil(1.25) = 2.0 (0x40000000)
        lines = [
            f"v_mov_b32 v1, {POS_ONE:#x}",
            "v_mov_b32 v2, 0x3E800000",   # 0.25
            "v_add_f32 v0, v1, v2",
            "v_ceil_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_add_pipe", lines)
        self.assertEqual(support.mrt0_words(spv)[0], POS_TWO)


# ==============================================================================
# Suite 6: Structural Emission, Mutations & Negative Controls (4 tests)
# ==============================================================================

class TestVCeilF32EmissionAndMutations(unittest.TestCase):
    """SPIR-V structural inspection, mutation pinning, and syntax rejection."""

    def test_spirv_structure_ceil_not_floor(self) -> None:
        """Verify SPIR-V text contains OpExtInst Ceil and does not contain Floor."""
        lines = [
            "v_mov_b32 v1, 0x3FC00000",
            "v_ceil_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ceil_spv_struct", lines)
        self.assertIn("OpExtInst %float %glsl Ceil", spv)
        self.assertNotIn("Floor", spv)

    def test_mutation_detection_ceil_vs_floor(self) -> None:
        """A mutant emitting Floor instead of Ceil must be distinguished."""
        # +1.5: ceil = 2.0 (0x40000000) != floor = 1.0 (0x3F800000)
        res_pos = run_ceil_f32(0x3FC00000, name="ps_ceil_mut_pos")
        self.assertEqual(res_pos, POS_TWO)
        self.assertNotEqual(res_pos, POS_ONE)

        # -0.5: ceil = -0.0 (0x80000000) != floor = -1.0 (0xBF800000)
        res_neg = run_ceil_f32(0xBF000000, name="ps_ceil_mut_neg")
        self.assertEqual(res_neg, NEG_ZERO)
        self.assertNotEqual(res_neg, NEG_ONE)

    def test_negative_syntax_rejection_missing_operands(self) -> None:
        """Negative control: missing operand syntax is rejected."""
        with self.assertRaises((IndexError, ValueError, support.rdna2spirv.Unsupported)):
            support.translate_lines("ps_ceil_bad_nop", ["v_ceil_f32"])

    def test_negative_syntax_rejection_invalid_opcode(self) -> None:
        """Negative control: unrecognized opcode is rejected."""
        with self.assertRaises((IndexError, ValueError, support.rdna2spirv.Unsupported)):
            support.translate_lines("ps_ceil_bad_op", ["v_ceil_f32_nonexistent v0, v1"])


if __name__ == "__main__":
    unittest.main()
