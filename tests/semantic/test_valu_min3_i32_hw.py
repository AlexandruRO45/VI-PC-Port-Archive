from __future__ import annotations

"""Architectural reference model and translator-backed hardware execution tests for v_min3_i32.

RDNA2 Instruction Set Architecture, VOP3:
    v_min3_i32 D, S0, S1, S2

Semantic operation:
    D.i32 = min(to_s32(S0), to_s32(S1), to_s32(S2))
    D.u32 = to_u32(D.i32)

Lowering in tools/rdna2spirv.py:
    Emits two chained GLSL.std.450 SMin operations on signed 32-bit integers:
        %m = OpExtInst %uint %glsl SMin %x %y
        %r = OpExtInst %uint %glsl SMin %m %z
    All tests execute through support.translate_lines(...) and support.mrt0_words / run_workgroup.

Standard library Python only (unittest, struct, math, dataclasses, enum, typing).
Strictly zero game data.
"""

import os
import struct
import sys
from typing import Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import support
    import spirv_mini_interp as si
except TypeError:
    support = None
    si = None

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
MIN_INT32: int = -0x8000_0000
MAX_INT32: int = 0x7FFF_FFFF
FSSM_64 = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


class _TranslatorTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        if support is None or si is None:
            self.skipTest("rdna2spirv requires Python 3.10+ (PEP 604 union types)")


def to_s32(val: int) -> int:
    """Convert 32-bit unsigned integer to signed two's-complement integer."""
    val = val & MASK32
    return val - 0x1_0000_0000 if (val & SIGN_BIT32) else val


def to_u32(val: int) -> int:
    """Convert signed integer to 32-bit unsigned integer representation."""
    return val & MASK32


def v_min3_i32_oracle(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical model for v_min3_i32."""
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    v2 = to_s32(s2)
    return to_u32(min(v0, v1, v2))


def run_translated_min3_i32(
    s0: int,
    s1: int,
    s2: int,
    *,
    s0_kind: str = "v",
    s1_kind: str = "v",
    s2_kind: str = "v",
    dst_reg: str = "v3",
    alias_dst: Optional[int] = None,
    op: str = "v_min3_i32",
) -> Tuple[int, str]:
    """Translates and executes v_min3_i32 through tools/rdna2spirv.py and spirv_mini_interp."""
    s0_u = s0 & MASK32
    s1_u = s1 & MASK32
    s2_u = s2 & MASK32
    lines = []
    src_regs = ["v0", "v1", "v2"]
    if alias_dst is not None:
        dst_reg = src_regs[alias_dst]

    # Load s0
    if s0_kind == "s":
        lines.append(f"s_mov_b32 s0, {s0_u:#x}")
        op0 = "s0"
    elif s0_kind == "lit":
        op0 = f"{s0_u:#x}"
    else:
        lines.append(f"v_mov_b32_e32 {src_regs[0]}, {s0_u:#x}")
        op0 = src_regs[0]

    # Load s1
    if s1_kind == "s":
        lines.append(f"s_mov_b32 s1, {s1_u:#x}")
        op1 = "s1"
    elif s1_kind == "lit":
        op1 = f"{s1_u:#x}"
    else:
        lines.append(f"v_mov_b32_e32 {src_regs[1]}, {s1_u:#x}")
        op1 = src_regs[1]

    # Load s2
    if s2_kind == "s":
        lines.append(f"s_mov_b32 s2, {s2_u:#x}")
        op2 = "s2"
    elif s2_kind == "lit":
        op2 = f"{s2_u:#x}"
    else:
        lines.append(f"v_mov_b32_e32 {src_regs[2]}, {s2_u:#x}")
        op2 = src_regs[2]

    lines.append(f"{op} {dst_reg}, {op0}, {op1}, {op2}")
    lines.append("v_mov_b32_e32 v4, 0")
    lines.append(f"exp mrt0 {dst_reg}, v4, v4, v4 done vm")
    lines.append("s_endpgm")

    name = f"ps_{op}_{s0_u:x}_{s1_u:x}_{s2_u:x}"
    spv = support.translate_lines(name, lines)
    out = support.mrt0_words(spv)
    return out[0], spv


# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

class TestVMin3I32Oracle(_TranslatorTestCase):
    """Unit tests for reference oracle backed by translator execution."""

    def test_oracle_signed_negatives_vs_positives(self):
        # In signed comparison, -100 is less than 50 and 100
        val, _ = run_translated_min3_i32(100, 50, to_u32(-100))
        self.assertEqual(to_s32(val), -100)
        self.assertEqual(val, v_min3_i32_oracle(100, 50, to_u32(-100)))

    def test_oracle_signed_negative_one_less_than_zero(self):
        # 0xFFFFFFFF (-1) vs 0 vs 1
        val, _ = run_translated_min3_i32(0, 0xFFFF_FFFF, 1)
        self.assertEqual(to_s32(val), -1)
        self.assertEqual(val, 0xFFFF_FFFF)

    def test_oracle_extreme_min_int32(self):
        # -2147483648 (0x80000000) is the absolute minimum signed 32-bit integer
        val, _ = run_translated_min3_i32(0x8000_0000, 0, 0x7FFF_FFFF)
        self.assertEqual(val, 0x8000_0000)
        self.assertEqual(to_s32(val), MIN_INT32)

    def test_oracle_identical_values(self):
        val_in = to_u32(-42)
        val, _ = run_translated_min3_i32(val_in, val_in, val_in)
        self.assertEqual(val, val_in)

    def test_translator_pipeline_execution_and_opcodes(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly with SMin."""
        val, spv = run_translated_min3_i32(10, to_u32(-20), 30)
        self.assertEqual(to_s32(val), -20)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpExtInst %uint %glsl SMin", spv)
        self.assertNotIn("UMin", spv)
        self.assertNotIn("SMax", spv)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class TestVMin3I32AST(_TranslatorTestCase):
    """Unit tests for AST opcode representations and syntax rejection via translator."""

    def test_ast_valid_vgpr_operands(self):
        lines = [
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_mov_b32_e32 v3, 30",
            "v_min3_i32 v4, v1, v2, v3",
            "v_mov_b32_e32 v5, 0",
            "exp mrt0 v4, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_ast_all_v", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 10)

    def test_ast_valid_sgpr_and_literal_operands(self):
        lines = [
            "s_mov_b32 s0, 15",
            "v_mov_b32_e32 v1, 25",
            "v_min3_i32 v2, s0, v1, 5",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_ast_s_v_lit", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 5)

    def test_ast_rejects_missing_operands(self):
        lines = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 2",
            "v_min3_i32 v0, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_min3_missing_ops", lines)


# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

class TestVMin3I32ScalarInterpreter(_TranslatorTestCase):
    """Unit tests verifying deterministic scalar execution via translator."""

    def test_interpreter_step_basic(self):
        val, _ = run_translated_min3_i32(to_u32(-500), to_u32(200), to_u32(-100))
        self.assertEqual(to_s32(val), -500)

    def test_interpreter_step_with_sgpr_and_literal(self):
        val, _ = run_translated_min3_i32(
            to_u32(-10), to_u32(0), to_u32(-100), s0_kind="s", s2_kind="lit"
        )
        self.assertEqual(to_s32(val), -100)

    def test_interpreter_all_permutations(self):
        # All 6 permutations of (10, -20, 30) must yield -20
        perms = [
            (10, to_u32(-20), 30),
            (10, 30, to_u32(-20)),
            (to_u32(-20), 10, 30),
            (to_u32(-20), 30, 10),
            (30, 10, to_u32(-20)),
            (30, to_u32(-20), 10),
        ]
        for a, b, c in perms:
            with self.subTest(a=a, b=b, c=c):
                val, _ = run_translated_min3_i32(a, b, c)
                self.assertEqual(to_s32(val), -20)


# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class TestVMin3I32SIMDDivergence(_TranslatorTestCase):
    """Unit tests for SIMD divergence under Wave64 EXEC masking via compute workgroup execution."""

    def test_divergent_lanes_preserve_inactive_state(self):
        # 64 lanes, only even lanes active (EXEC = 0x5555_5555_5555_5555)
        # Even lanes store min(-i, i, 0) = -i. Inactive lanes preserve 0xDEADBEEF sentinel.
        prog = [
            "s_mov_b64 exec, -1",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_mbcnt_hi_u32_b32 v0, -1, v0",
            # v1 = -lane (0 - lane)
            "v_sub_nc_u32_e32 v1, 0, v0",
            # v2 = lane
            "v_mov_b32_e32 v2, v0",
            # canary in v4
            "v_mov_b32_e32 v4, 0xDEADBEEF",
            # set EXEC to even lanes only
            "s_mov_b32 exec_lo, 0x55555555",
            "s_mov_b32 exec_hi, 0x55555555",
            "v_min3_i32 v4, v1, v2, 0",
            # restore full EXEC to write results to memory
            "s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v10, 2, v0",
            "buffer_store_dword v4, v10, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_min3_divergence", prog, fssm=FSSM_64)
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory={},
        )
        for lane in range(64):
            got = mem.get(4 * lane, 0)
            if lane % 2 == 0:
                expected = to_u32(-lane)
                self.assertEqual(got, expected, f"Lane {lane} failed: got {got:#x}, expected {expected:#x}")
            else:
                self.assertEqual(got, 0xDEADBEEF, f"Lane {lane} canary clobbered: {got:#x}")


# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestVMin3I32Aliasing(_TranslatorTestCase):
    """Unit tests for register aliasing and in-place operand preservation through translator."""

    def test_alias_dst_equals_src0(self):
        val, _ = run_translated_min3_i32(to_u32(50), to_u32(-20), to_u32(10), alias_dst=0)
        self.assertEqual(to_s32(val), -20)

    def test_alias_dst_equals_src1(self):
        val, _ = run_translated_min3_i32(to_u32(50), to_u32(-20), to_u32(10), alias_dst=1)
        self.assertEqual(to_s32(val), -20)

    def test_alias_dst_equals_src2(self):
        val, _ = run_translated_min3_i32(to_u32(50), to_u32(-20), to_u32(10), alias_dst=2)
        self.assertEqual(to_s32(val), -20)

    def test_alias_all_operands_same_register(self):
        lines = [
            "v_mov_b32_e32 v0, 0xFFFFFFFD", # -3
            "v_min3_i32 v0, v0, v0, v0",
            "v_mov_b32_e32 v1, 0",
            "exp mrt0 v0, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_alias_all", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(to_s32(out[0]), -3)


# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

class TestVMin3I32ConditionCodes(_TranslatorTestCase):
    """Unit tests verifying sign and zero detection on translated output words."""

    def test_zero_result_translation(self):
        val, _ = run_translated_min3_i32(0, 10, 20)
        self.assertEqual(val, 0)

    def test_negative_result_sign_bit(self):
        val, _ = run_translated_min3_i32(to_u32(-1), 10, 20)
        self.assertEqual(val, 0xFFFF_FFFF)
        self.assertTrue(bool(val & SIGN_BIT32))

    def test_positive_result_sign_bit(self):
        val, _ = run_translated_min3_i32(10, 20, 30)
        self.assertEqual(val, 10)
        self.assertFalse(bool(val & SIGN_BIT32))


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class TestVMin3I32IEEE754Bitwise(_TranslatorTestCase):
    """Unit tests verifying that IEEE 754 bit patterns are compared strictly as signed integers."""

    FLOAT32_PATTERNS = {
        "pos_zero": 0x0000_0000,
        "neg_zero": 0x8000_0000,
        "pos_inf": 0x7F80_0000,
        "neg_inf": 0xFF80_0000,
        "qnan": 0x7FC0_0000,
        "snan": 0x7F80_0001,
        "min_subnormal": 0x0000_0001,
        "max_subnormal": 0x007F_FFFF,
        "min_normal": 0x0080_0000,
        "max_normal": 0x7F7F_FFFF,
    }

    def test_neg_zero_less_than_pos_zero(self):
        """In signed 32-bit integer, 0x80000000 (-2147483648) is less than 0x00000000 (0)."""
        neg_zero = self.FLOAT32_PATTERNS["neg_zero"]
        pos_zero = self.FLOAT32_PATTERNS["pos_zero"]
        val, _ = run_translated_min3_i32(pos_zero, neg_zero, 1)
        self.assertEqual(val, neg_zero)

    def test_neg_zero_is_signed_min_int32(self):
        neg_zero = self.FLOAT32_PATTERNS["neg_zero"]
        for name, pattern in self.FLOAT32_PATTERNS.items():
            with self.subTest(pattern_name=name):
                val, _ = run_translated_min3_i32(pattern, neg_zero, 0)
                self.assertEqual(val, neg_zero)

    def test_neg_inf_signed_comparison(self):
        neg_inf = self.FLOAT32_PATTERNS["neg_inf"]
        val, _ = run_translated_min3_i32(0, neg_inf, 100)
        self.assertEqual(val, neg_inf)


# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class TestVMin3I32NegativeAndMutation(_TranslatorTestCase):
    """Unit tests for negative syntax controls, bounds checking, and mutation detection."""

    def test_mutation_detector_unsigned_vs_signed(self):
        """Verify that signed min (v_min3_i32) and unsigned min (v_min3_u32) diverge on -1 vs 0 vs 1."""
        s0 = 0xFFFF_FFFF  # signed -1, unsigned 4294967295
        s1 = 0
        s2 = 1

        res_signed, _ = run_translated_min3_i32(s0, s1, s2, op="v_min3_i32")
        res_unsigned, _ = run_translated_min3_i32(s0, s1, s2, op="v_min3_u32")

        # Signed min is -1 (0xFFFFFFFF)
        self.assertEqual(res_signed, 0xFFFF_FFFF)
        # Unsigned min is 0 (0x0)
        self.assertEqual(res_unsigned, 0)
        self.assertNotEqual(res_signed, res_unsigned)

    def test_mutation_detector_max3_mutant(self):
        """Verify that min3 and max3 diverge on -10 vs 0 vs 10."""
        s0 = to_u32(-10)
        s1 = 0
        s2 = 10

        res_min, _ = run_translated_min3_i32(s0, s1, s2, op="v_min3_i32")
        res_max, _ = run_translated_min3_i32(s0, s1, s2, op="v_max3_i32")

        self.assertEqual(to_s32(res_min), -10)
        self.assertEqual(to_s32(res_max), 10)
        self.assertNotEqual(res_min, res_max)

    def test_translator_emits_chained_smins(self):
        """Verify that translator emits exactly two chained SMin instructions."""
        _, spv = run_translated_min3_i32(1, 2, 3)
        count = spv.count("OpExtInst %uint %glsl SMin")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
