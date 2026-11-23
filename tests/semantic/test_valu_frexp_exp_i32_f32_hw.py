from __future__ import annotations

"""Semantic hardware tests and reference mathematical oracle for RDNA2 v_frexp_exp_i32_f32.

Instruction definition:
    v_frexp_exp_i32_f32 vdst, vsrc0

Semantics:
    Extracts the binary exponent of a single-precision floating-point value S0,
    such that S0 = mantissa * 2^exponent with mantissa in [0.5, 1.0).
    The exponent is returned as a 32-bit signed two's complement integer in destination VGPR.
    For normal binary32 numbers (biased exponent e in 1..254):
        exp = e - 126
    For zero (0.0, -0.0), denormals, infinities (+inf, -inf), and NaNs:
        exp = 0

Zero game data, standard library Python only (python -I, struct, math, unittest).
All test cases compile and execute through support.translate_lines and support.mrt0_words.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import rdna2spirv

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x00FF
EXP_SHIFT32: int = 23
MANT_MASK32: int = 0x007F_FFFF
EXP_BIAS_FREXP: int = 126

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MIN_NORMAL: int = 0x0080_0000       # 2^-126
MAX_NORMAL: int = 0x7F7F_FFFF       # approx 3.4028235e+38


# ==============================================================================
# Reference Mathematical Oracle and Hardware Execution Helpers
# ==============================================================================

def to_i32(val: int) -> int:
    """Interpret 32-bit integer as two's complement signed integer."""
    val32 = val & MASK32
    if val32 & SIGN_BIT32:
        return val32 - 0x1_0000_0000
    return val32


def f32_to_bits(f: float) -> int:
    """Pack Python float to 32-bit IEEE 754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    """Unpack 32-bit integer bit pattern to Python float."""
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def ref_frexp_exp_i32_f32(src: int) -> int:
    """Architectural reference oracle for v_frexp_exp_i32_f32.
    Computes 32-bit signed exponent in [0.5, 1.0) representation."""
    u = src & MASK32
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    if 1 <= e <= 254:
        exp = e - EXP_BIAS_FREXP
        return exp & MASK32
    return 0


def run_v_frexp_exp_i32_f32(src_val: int, tag: str = "run") -> int:
    """Translate and run v_frexp_exp_i32_f32, returning destination word."""
    prog = [
        f"v_mov_b32 v1, {src_val & MASK32:#x}",
        "v_frexp_exp_i32_f32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_frexp_exp_{tag}", prog)
    return support.mrt0_words(spv)[0]


# ==============================================================================
# Commit 1: Reference Mathematical Oracle and Hardware Conformance
# ==============================================================================

class TestFrexpExpI32F32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        cases = [
            (1.0, 1),
            (0.5, 0),
            (2.0, 2),
            (4.0, 3),
            (0.25, 0xFFFF_FFFF),
            (0.125, 0xFFFF_FFFE),
        ]
        for idx, (fval, expected) in enumerate(cases):
            bits = f32_to_bits(fval)
            self.assertEqual(ref_frexp_exp_i32_f32(bits), expected)
            hw_exp = run_v_frexp_exp_i32_f32(bits, tag=f"p2_{idx}")
            self.assertEqual(hw_exp, expected)

    def test_sign_invariance(self):
        cases = [-1.0, -0.5, -2.0, -0.25]
        for idx, fval in enumerate(cases):
            pos_bits = f32_to_bits(abs(fval))
            neg_bits = f32_to_bits(fval)
            expected = ref_frexp_exp_i32_f32(pos_bits)
            self.assertEqual(ref_frexp_exp_i32_f32(neg_bits), expected)
            hw_exp = run_v_frexp_exp_i32_f32(neg_bits, tag=f"sign_{idx}")
            self.assertEqual(hw_exp, expected)

    def test_arbitrary_normals(self):
        cases = [
            (1.5, 1),
            (3.0, 2),
            (100.0, 7),
        ]
        for idx, (fval, expected) in enumerate(cases):
            bits = f32_to_bits(fval)
            self.assertEqual(ref_frexp_exp_i32_f32(bits), expected)
            hw_exp = run_v_frexp_exp_i32_f32(bits, tag=f"arb_{idx}")
            self.assertEqual(hw_exp, expected)

    def test_boundary_normals(self):
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL), 0xFFFF_FF83)
        self.assertEqual(to_i32(ref_frexp_exp_i32_f32(MIN_NORMAL)), -125)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL), 128)

        hw_min = run_v_frexp_exp_i32_f32(MIN_NORMAL, tag="bnd_min")
        self.assertEqual(hw_min, 0xFFFF_FF83)
        hw_max = run_v_frexp_exp_i32_f32(MAX_NORMAL, tag="bnd_max")
        self.assertEqual(hw_max, 128)

    def test_special_values_return_zero(self):
        specials = [
            POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, SNAN,
            0x0000_0001, 0x007F_FFFF, 0x8000_0001, 0x807F_FFFF,
        ]
        for idx, spec in enumerate(specials):
            self.assertEqual(ref_frexp_exp_i32_f32(spec), 0)
            hw_exp = run_v_frexp_exp_i32_f32(spec, tag=f"spec_{idx}")
            self.assertEqual(hw_exp, 0)


# ==============================================================================
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestFrexpExpI32F32AST(unittest.TestCase):
    def test_ast_register_operands(self):
        cases = [
            ["v_frexp_exp_i32_f32 v0, v1", "s_endpgm"],
            ["v_frexp_exp_i32_f32 v5, s10", "s_endpgm"],
            ["v_frexp_exp_i32_f32 v2, 0x3F800000", "s_endpgm"],
        ]
        for idx, lines in enumerate(cases):
            spv = support.translate_lines(f"ps_ast_frexp_{idx}", lines)
            self.assertIn("OpBitFieldUExtract", spv)
            self.assertIn("OpISub", spv)

    def test_ast_pipeline_execution(self):
        lines = [
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_pipeline_frexp", lines)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertIn("OpISub", spv)
        self.assertIn("OpReturn", spv)


# ==============================================================================
# Commit 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestFrexpExpI32F32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution_vgpr(self):
        hw_exp = run_v_frexp_exp_i32_f32(f32_to_bits(2.0), tag="scalar_vgpr")
        self.assertEqual(hw_exp, 2)

    def test_scalar_execution_sgpr(self):
        prog = [
            f"s_mov_b32 s4, {f32_to_bits(0.25):#x}",
            "v_frexp_exp_i32_f32 v5, s4",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scalar_sgpr", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF_FFFF)

    def test_scalar_execution_literal(self):
        prog = [
            f"v_frexp_exp_i32_f32 v2, {f32_to_bits(1.0):#x}",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scalar_literal", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 1)

    def test_scalar_zero_input(self):
        hw_exp = run_v_frexp_exp_i32_f32(NEG_ZERO, tag="scalar_zero")
        self.assertEqual(hw_exp, 0)


# ==============================================================================
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestFrexpExpI32F32SIMDDivergence(unittest.TestCase):
    def test_simd_active_lane_execution(self):
        # Even lanes active -> lane 0 computes exp of 2.0f = 2
        prog = [
            "s_mov_b32 exec_lo, 0x55555555",
            f"v_mov_b32 v1, {f32_to_bits(2.0):#x}",
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_active", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 2)

    def test_simd_inactive_lane_preservation(self):
        # Odd lanes active -> lane 0 inactive; v0 initialized to 0xDEADBEEF must be preserved
        prog = [
            "v_mov_b32 v0, 0xDEADBEEF",
            "s_mov_b32 exec_lo, 0xAAAAAAAA",
            f"v_mov_b32 v1, {f32_to_bits(2.0):#x}",
            "v_frexp_exp_i32_f32 v0, v1",
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_inactive", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xDEADBEEF)


# ==============================================================================
# Commit 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestFrexpExpI32F32RegisterAliasing(unittest.TestCase):
    def test_in_place_alias_vgpr(self):
        # v0 = v_frexp_exp_i32_f32(v0)
        prog = [
            f"v_mov_b32 v0, {f32_to_bits(1.0):#x}",
            "v_frexp_exp_i32_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v0", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 1)

    def test_in_place_alias_negative_exponent(self):
        # v3 = v_frexp_exp_i32_f32(v3) with 0.125f (exp -2)
        prog = [
            f"v_mov_b32 v3, {f32_to_bits(0.125):#x}",
            "v_frexp_exp_i32_f32 v3, v3",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v3", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF_FFFE)

    def test_unrelated_registers_preserved(self):
        prog = [
            "v_mov_b32 v0, 0x11111111",
            f"v_mov_b32 v1, {f32_to_bits(8.0):#x}",
            "v_mov_b32 v2, 0x22222222",
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v1, v2, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_unrelated", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 4)
        self.assertEqual(words[1], f32_to_bits(8.0))
        self.assertEqual(words[2], 0x22222222)


# ==============================================================================
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestFrexpExpI32F32StatusFlags(unittest.TestCase):
    def test_scc_invariance(self):
        # VALU instructions do not modify scalar condition code SCC
        for init_scc in (0, 1):
            with self.subTest(init_scc=init_scc):
                cmp_code = "s_cmp_eq_u32 s0, 1" if init_scc == 1 else "s_cmp_eq_u32 s0, 0"
                prog = [
                    "s_mov_b32 s0, 1",
                    cmp_code,
                    f"v_mov_b32 v1, {f32_to_bits(16.0):#x}",
                    "v_frexp_exp_i32_f32 v0, v1",
                    "s_cselect_b32 s1, 777, 999",
                    "v_mov_b32 v1, s1",
                    "exp mrt0 v0, v1, v0, v1 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_scc_inv_{init_scc}", prog)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], 5)
                expected_sel = 777 if init_scc == 1 else 999
                self.assertEqual(words[1], expected_sel)


# ==============================================================================
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestFrexpExpI32F32IEEE754Handling(unittest.TestCase):
    def test_subnormals_flushed_to_zero(self):
        subnormals = [
            0x0000_0001,
            0x0000_0002,
            0x0000_FFFF,
            0x0040_0000,
            0x007F_FFFF,
            0x8000_0001,
            0x8000_FFFF,
            0x807F_FFFF,
        ]
        for idx, sub in enumerate(subnormals):
            self.assertEqual(ref_frexp_exp_i32_f32(sub), 0)
            hw_exp = run_v_frexp_exp_i32_f32(sub, tag=f"sub_{idx}")
            self.assertEqual(hw_exp, 0)

    def test_zeros_yield_zero(self):
        self.assertEqual(run_v_frexp_exp_i32_f32(POS_ZERO, tag="zero_pos"), 0)
        self.assertEqual(run_v_frexp_exp_i32_f32(NEG_ZERO, tag="zero_neg"), 0)

    def test_infinities_yield_zero(self):
        self.assertEqual(run_v_frexp_exp_i32_f32(POS_INF, tag="inf_pos"), 0)
        self.assertEqual(run_v_frexp_exp_i32_f32(NEG_INF, tag="inf_neg"), 0)

    def test_all_nans_yield_zero(self):
        nans = [
            QNAN, SNAN, 0xFFC0_0000, 0xFF80_0001, 0x7F80_0002, 0x7FFF_FFFF, 0xFFFF_FFFF,
        ]
        for idx, nan in enumerate(nans):
            self.assertEqual(ref_frexp_exp_i32_f32(nan), 0)
            hw_exp = run_v_frexp_exp_i32_f32(nan, tag=f"nan_{idx}")
            self.assertEqual(hw_exp, 0)

    def test_exponent_boundary_continuity(self):
        boundaries = [
            (0x3F7F_FFFF, 0),  # Just below 1.0f
            (0x3F80_0000, 1),  # Exactly 1.0f
            (0x3F80_0001, 1),  # Just above 1.0f
            (0x3FFF_FFFF, 1),  # Just below 2.0f
            (0x4000_0000, 2),  # Exactly 2.0f
        ]
        for idx, (bits, expected) in enumerate(boundaries):
            self.assertEqual(ref_frexp_exp_i32_f32(bits), expected)
            hw_exp = run_v_frexp_exp_i32_f32(bits, tag=f"bnd_cont_{idx}")
            self.assertEqual(hw_exp, expected)


# ==============================================================================
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestFrexpExpI32F32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_missing", ["v_frexp_exp_i32_f32 v0", "s_endpgm"])
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_neg_bad_op", ["v_frexp_unknown_f32 v0, v1", "s_endpgm"])

    def test_mutation_detection_bias(self):
        # Mutating bias from 126 to 127 changes exp of 1.0f from 1 to 0
        hw_exp = run_v_frexp_exp_i32_f32(f32_to_bits(1.0), tag="mut_bias")
        mutant_exp = 0  # wrong bias
        self.assertNotEqual(mutant_exp, hw_exp)

    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(1.0), 1),
            (f32_to_bits(0.5), 0),
            (f32_to_bits(2.0), 2),
            (f32_to_bits(0.25), 0xFFFF_FFFF),
            (f32_to_bits(100.0), 7),
            (f32_to_bits(-1.0), 1),
            (f32_to_bits(-0.5), 0),
            (POS_ZERO, 0),
            (NEG_ZERO, 0),
            (POS_INF, 0),
            (NEG_INF, 0),
            (QNAN, 0),
            (0x0000_0001, 0),
            (MIN_NORMAL, 0xFFFF_FF83),
            (MAX_NORMAL, 128),
        ]
        for idx, (src_val, expected) in enumerate(cases):
            hw_exp = run_v_frexp_exp_i32_f32(src_val, tag=f"hw_exec_{idx}")
            self.assertEqual(hw_exp, expected, f"Failed on input {src_val:#x}: got {hw_exp:#x}, expected {expected:#x}")


if __name__ == '__main__':
    unittest.main()
