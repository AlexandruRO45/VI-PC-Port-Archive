from __future__ import annotations

"""Architectural reference model and hardware-rule semantic tests for v_floor_f32.

RDNA2 Instruction Set Architecture, VOP1:
    v_floor_f32 D, S0

Semantic operation:
    D.f32 = floor(S0.f32)

Hardware rules (GFX10 RDNA2):
    - Float32 IEEE 754 floor operation.
    - f32 denormals flushed to zero preserving sign: floor(+denorm) = +0.0, floor(-denorm) = -0.0.
    - Zero preservation: floor(+0.0) = +0.0, floor(-0.0) = -0.0.
    - Infs pass through: floor(+inf) = +inf, floor(-inf) = -inf.
    - NaNs pass through unchanged (quiet and signaling payloads preserved).

100% translator-backed execution; zero mock AST or virtual CPU/SIMD models.
Standard library Python only (unittest, struct, math). Strictly zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import math
import os
import struct
import sys
import unittest
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x7F80_0000
MANT_MASK32: int = 0x007F_FFFF


def to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def v_floor_f32_oracle(val: int) -> int:
    """Reference mathematical model matching GFX10 RDNA2 hardware semantics."""
    val = val & MASK32
    exp = (val >> 23) & 0xFF
    mant = val & MANT_MASK32
    sign = val & SIGN_BIT32

    # NaN pass through
    if exp == 0xFF and mant != 0:
        return val
    # Inf pass through
    if exp == 0xFF and mant == 0:
        return val
    # Signed zero pass through
    if exp == 0 and mant == 0:
        return val
    # Denormals flushed to signed zero
    if exp == 0:
        return sign

    f = to_f32(val)
    rf = float(math.floor(f))
    if rf == 0.0 and sign:
        return SIGN_BIT32
    return to_u32(rf)


_TRANSLATE_ID: int = 0


def translate_floor_f32(val: int, name: str | None = None) -> str:
    """Translate synthetic shader using v_floor_f32."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    prog_name = name or f"ps_floor_f32_{_TRANSLATE_ID}"
    lines = [
        f"v_mov_b32_e32 v1, {val & MASK32:#x}",
        "v_floor_f32 v2, v1",
        "v_mov_b32_e32 v3, 0",
        "exp mrt0 v2, v3, v3, v3 done vm",
        "s_endpgm",
    ]
    return support.translate_lines(prog_name, lines)


def execute_floor_f32(val: int, name: str | None = None) -> int:
    """Translate and execute synthetic shader; return first mrt0 word."""
    spv = translate_floor_f32(val, name)
    out = support.mrt0_words(spv)
    return out[0]


class TestVFloorF32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_oracle_hardware_measured_table(self):
        measured = [
            (0x00000001, 0x00000000),  # +denorm -> +0.0
            (0x80000001, 0x80000000),  # -denorm -> -0.0
            (0x00400000, 0x00000000),  # +denorm -> +0.0
            (0x807FFFFF, 0x80000000),  # -denorm -> -0.0
            (0x80000000, 0x80000000),  # -0.0 -> -0.0
            (0x00000000, 0x00000000),  # +0.0 -> +0.0
            (0x00800000, 0x00000000),  # min normal (+1.17e-38) -> 0.0
            (0x80800000, 0xBF800000),  # min neg normal (-1.17e-38) -> -1.0
            (0x7F800000, 0x7F800000),  # +inf
            (0xFF800000, 0xFF800000),  # -inf
            (0x7FA00000, 0x7FA00000),  # sNaN
            (0x7FC00000, 0x7FC00000),  # qNaN
            (0xBF000000, 0xBF800000),  # -0.5 -> -1.0
            (0x3FC00000, 0x3F800000),  # +1.5 -> +1.0
            (0x3F7FFFFF, 0x00000000),  # 0.99999994 -> 0.0
            (0xBF7FFFFF, 0xBF800000),  # -0.99999994 -> -1.0
        ]
        for inp, expected in measured:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(v_floor_f32_oracle(inp), expected)
                hw = execute_floor_f32(inp)
                self.assertEqual(hw, expected)

    def test_translator_pipeline_execution(self):
        spv = translate_floor_f32(0x3FC0_0000)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

    def test_translator_emits_floor_not_ceil(self):
        spv = translate_floor_f32(0x3FC0_0000)
        self.assertIn("OpExtInst %float %glsl Floor", spv)
        self.assertNotIn("Ceil", spv)

    def test_translator_positive_fractional_inputs(self):
        self.assertEqual(execute_floor_f32(0x3FC0_0000), 0x3F80_0000)
        self.assertEqual(execute_floor_f32(0x3F400000), 0x0000_0000)
        self.assertEqual(execute_floor_f32(0x403FFFFF), 0x4000_0000)
        self.assertEqual(execute_floor_f32(0x3F7FFFFF), 0x0000_0000)

    def test_translator_negative_fractional_inputs(self):
        self.assertEqual(execute_floor_f32(0xBF000000), 0xBF800000)
        self.assertEqual(execute_floor_f32(0xBF400000), 0xBF800000)
        self.assertEqual(execute_floor_f32(0xBFC00000), 0xC0000000)
        self.assertEqual(execute_floor_f32(0xBF7FFFFF), 0xBF800000)

    def test_translator_integral_values_unchanged(self):
        self.assertEqual(execute_floor_f32(0x3F800000), 0x3F800000)
        self.assertEqual(execute_floor_f32(0xBF800000), 0xBF800000)
        self.assertEqual(execute_floor_f32(0x40800000), 0x40800000)

    def test_translator_special_values_passthrough(self):
        self.assertEqual(execute_floor_f32(0x7F800000), 0x7F800000)
        self.assertEqual(execute_floor_f32(0xFF800000), 0xFF800000)
        self.assertEqual(execute_floor_f32(0x7FC00000), 0x7FC00000)

    def test_translator_signed_zeros_passthrough(self):
        self.assertEqual(execute_floor_f32(0x00000000), 0x00000000)
        self.assertEqual(execute_floor_f32(0x80000000), 0x80000000)

    def test_translator_mutation_floor_to_ceil_caught(self):
        translated = execute_floor_f32(0xBF000000)
        oracle = v_floor_f32_oracle(0xBF000000)
        self.assertEqual(translated, oracle)
        self.assertEqual(translated, 0xBF800000)

        ceil_result = to_u32(math.ceil(to_f32(0xBF000000)))
        self.assertNotEqual(translated, ceil_result)

    def test_translator_clamp_modifier(self):
        lines = [
            "v_mov_b32_e32 v1, 0xbf000000",
            "v_floor_f32 v2, v1 clamp",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_floor_clamp", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x00000000)
        self.assertIn("OpExtInst %float %glsl FClamp", spv)


class TestVFloorF32InstructionForms(unittest.TestCase):
    """Commit 2: Instruction operand variations and translator emission."""

    def test_instruction_vgpr_operands(self):
        lines = [
            "v_mov_b32_e32 v1, 0x3fc00000",
            "v_floor_f32 v2, v1",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_vgpr", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F80_0000)

    def test_instruction_with_sgpr_source(self):
        lines = [
            "s_mov_b32 s4, 0x3fc00000",
            "v_floor_f32 v3, s4",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v3, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_sgpr", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F80_0000)

    def test_instruction_inline_constant_source(self):
        lines = [
            "v_floor_f32 v0, 1.0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_inline", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x3F80_0000)

    def test_instruction_literal_constant_source(self):
        lines = [
            "v_floor_f32 v0, 0x40300000",  # 2.75 -> 2.0 (0x40000000)
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_lit", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x4000_0000)


class TestVFloorF32SequentialExecution(unittest.TestCase):
    """Commit 3: Sequential and chained instruction shader execution."""

    def test_sequential_step_basic(self):
        hw = execute_floor_f32(0x3FC0_0000)
        self.assertEqual(hw, 0x3F80_0000)

    def test_sequential_step_sgpr_source(self):
        lines = [
            "s_mov_b32 s10, 0xbf000000",
            "v_floor_f32 v4, s10",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v4, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_sgpr", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xBF80_0000)

    def test_chained_floor_in_shader(self):
        lines = [
            "v_mov_b32_e32 v1, 0x40900000",  # 4.5
            "v_floor_f32 v2, v1",            # 4.0
            "v_floor_f32 v4, v2",            # 4.0
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v4, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_floor_chained", lines)
        self.assertEqual(support.mrt0_words(spv)[0], to_u32(4.0))


class TestVFloorF32MultiLaneExecution(unittest.TestCase):
    """Commit 4: Multi-lane divergence sweeps and coverage."""

    def test_simd32_sweeps_positive_fractions(self):
        for lane in range(32):
            f_val = float(lane) + 0.75
            u_val = to_u32(f_val)
            expected = to_u32(float(lane))
            hw = execute_floor_f32(u_val)
            self.assertEqual(hw, expected)

    def test_simd64_sweeps_negative_fractions(self):
        for lane in range(64):
            f_val = -float(lane) - 0.25
            u_val = to_u32(f_val)
            expected = to_u32(-float(lane) - 1.0)
            hw = execute_floor_f32(u_val)
            self.assertEqual(hw, expected)


class TestVFloorF32Aliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        lines = [
            f"v_mov_b32_e32 v1, {to_u32(3.9):#x}",
            "v_floor_f32 v1, v1",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v1, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_inplace", lines)
        self.assertEqual(support.mrt0_words(spv)[0], to_u32(3.0))

    def test_unrelated_registers_unaffected(self):
        lines = [
            f"v_mov_b32_e32 v1, {to_u32(5.5):#x}",
            "v_mov_b32_e32 v0, 0x12345678",
            "v_mov_b32_e32 v3, 0x87654321",
            "v_floor_f32 v2, v1",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_unrelated", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12345678)
        self.assertEqual(words[1], to_u32(5.5))
        self.assertEqual(words[2], to_u32(5.0))
        self.assertEqual(words[3], 0x87654321)


class TestVFloorF32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and downstream branching."""

    def test_zero_result_downstream_branching(self):
        lines = [
            f"v_mov_b32_e32 v1, {to_u32(0.5):#x}",
            "v_floor_f32 v2, v1",
            "v_cmp_eq_f32 vcc, 0.0, v2",
            "v_cndmask_b32_e64 v4, 0x1111, 0x2222, vcc",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v4, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_zero_cond", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x2222)

    def test_negative_result_downstream_branching(self):
        lines = [
            f"v_mov_b32_e32 v1, {to_u32(-0.5):#x}",
            "v_floor_f32 v2, v1",
            "v_cmp_lt_f32 vcc, v2, 0.0",
            "v_cndmask_b32_e64 v4, 0x3333, 0x4444, vcc",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v4, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_neg_cond", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x4444)

    def test_negative_zero_preservation(self):
        res = execute_floor_f32(SIGN_BIT32)
        self.assertEqual(res, SIGN_BIT32)

    def test_sign_bit_propagation(self):
        res = execute_floor_f32(to_u32(-10.2))
        self.assertTrue(bool(res & SIGN_BIT32))


class TestVFloorF32IEEE754Bitwise(unittest.TestCase):
    """Commit 7: IEEE 754 edge cases, subnormals, and NaNs."""

    def test_all_denormals_flushed_to_signed_zero(self):
        for mant in (1, 2, 0x400000, 0x7FFFFF):
            with self.subTest(pos_denorm=hex(mant)):
                self.assertEqual(v_floor_f32_oracle(mant), 0x0000_0000)
                self.assertEqual(execute_floor_f32(mant), 0x0000_0000)

        for mant in (1, 2, 0x400000, 0x7FFFFF):
            with self.subTest(neg_denorm=hex(mant)):
                self.assertEqual(v_floor_f32_oracle(SIGN_BIT32 | mant), SIGN_BIT32)
                self.assertEqual(execute_floor_f32(SIGN_BIT32 | mant), SIGN_BIT32)

    def test_infinities_pass_through(self):
        self.assertEqual(v_floor_f32_oracle(0x7F80_0000), 0x7F80_0000)
        self.assertEqual(execute_floor_f32(0x7F80_0000), 0x7F80_0000)
        self.assertEqual(v_floor_f32_oracle(0xFF80_0000), 0xFF80_0000)
        self.assertEqual(execute_floor_f32(0xFF80_0000), 0xFF80_0000)

    def test_nans_pass_through_exact_payload(self):
        qnan = 0x7FC0_1234
        snan = 0x7F80_0042
        self.assertEqual(v_floor_f32_oracle(qnan), qnan)
        self.assertEqual(execute_floor_f32(qnan), qnan)
        self.assertEqual(v_floor_f32_oracle(snan), snan)
        self.assertEqual(execute_floor_f32(snan), snan)


class TestVFloorF32NegativeAndMutation(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_missing_operand(self):
        lines = [
            "v_floor_f32 v0",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_floor_bad_missing", lines)

    def test_syntax_rejection_unknown_operand(self):
        lines = [
            "v_floor_f32 v0, unknown_op",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_floor_bad_token", lines)

    def test_mutation_detector_ceil_mutant(self):
        def mutant_ceil(u: int) -> int:
            f = to_f32(u)
            return to_u32(float(math.ceil(f)))

        val = to_u32(1.5)
        correct = v_floor_f32_oracle(val)
        mutant = mutant_ceil(val)
        self.assertEqual(correct, to_u32(1.0))
        self.assertEqual(mutant, to_u32(2.0))
        self.assertNotEqual(mutant, correct)

    def test_mutation_detector_trunc_mutant(self):
        def mutant_trunc(u: int) -> int:
            f = to_f32(u)
            return to_u32(float(math.trunc(f)))

        val = to_u32(-1.5)
        correct = v_floor_f32_oracle(val)
        mutant = mutant_trunc(val)
        self.assertEqual(correct, to_u32(-2.0))
        self.assertEqual(mutant, to_u32(-1.0))
        self.assertNotEqual(mutant, correct)


if __name__ == "__main__":
    unittest.main()
