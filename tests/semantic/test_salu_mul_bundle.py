"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 SALU 32-bit integer multiplication instructions (Bundle 40):
  - s_mul_i32:    32-bit integer multiplication (low 32 bits, signed/unsigned identical)
  - s_mul_hi_i32: Signed 32-bit integer multiplication high-half (high 32 bits of 64-bit product)
  - s_mul_hi_u32: Unsigned 32-bit integer multiplication high-half (high 32 bits of 64-bit product)

100% translator-backed: every test executes real RDNA2 assembly through tools/rdna2spirv.py
via support.translate_lines(...) and verifies lowered SPIR-V opcodes and output words via
support.mrt0_words(...). Zero in-file mock models.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import random
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

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def _run_binary_op(
    op: str,
    s0_val: int,
    s1_val: int,
    s0_is_imm: bool = False,
    s1_is_imm: bool = False,
    tag: str = "",
) -> int:
    """Translates and runs a pixel shader executing `op s2, s0, s1`."""
    s0_val = s0_val & MASK32
    s1_val = s1_val & MASK32
    s0_expr = f"{s0_val:#x}" if s0_is_imm else "s0"
    s1_expr = f"{s1_val:#x}" if s1_is_imm else "s1"
    lines = []
    if not s0_is_imm:
        lines.append(f"s_mov_b32 s0, {s0_val:#x}")
    if not s1_is_imm:
        lines.append(f"s_mov_b32 s1, {s1_val:#x}")
    lines.extend([
        f"{op} s2, {s0_expr}, {s1_expr}",
        "v_mov_b32 v0, s2",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0, v0, v1, v2, v3 done",
        "s_endpgm",
    ])
    shader_name = f"ps_{op}_{tag}_{s0_val}_{s1_val}".replace("-", "n")
    spvasm = support.translate_lines(shader_name, lines)
    words = support.mrt0_words(spvasm)
    return words[0]


def _run_with_scc(op: str, s0_val: int, s1_val: int, initial_scc: int) -> tuple[int, int]:
    """Translates and runs a shader testing SCC flag preservation across `op`."""
    s0_val = s0_val & MASK32
    s1_val = s1_val & MASK32
    # s_cmp_eq_u32 5, 5 sets SCC=1; s_cmp_eq_u32 5, 6 sets SCC=0
    cmp_rhs = 5 if initial_scc else 6
    lines = [
        f"s_cmp_eq_u32 5, {cmp_rhs}",
        f"s_mov_b32 s0, {s0_val:#x}",
        f"s_mov_b32 s1, {s1_val:#x}",
        f"{op} s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v0, s2",
        "v_mov_b32 v1, s3",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0, v0, v1, v2, v3 done",
        "s_endpgm",
    ]
    shader_name = f"ps_{op}_scc_{initial_scc}_{s0_val}_{s1_val}"
    spvasm = support.translate_lines(shader_name, lines)
    words = support.mrt0_words(spvasm)
    return words[0], words[1]


def _run_aliased(op: str, s0_val: int, s1_val: int, mode: str) -> int:
    """Translates and runs in-place register aliasing for `op`."""
    s0_val = s0_val & MASK32
    s1_val = s1_val & MASK32
    lines = [
        f"s_mov_b32 s0, {s0_val:#x}",
        f"s_mov_b32 s1, {s1_val:#x}",
    ]
    if mode == "dst_src0":
        lines.append(f"{op} s0, s0, s1")
        dst_sgpr = "s0"
    elif mode == "dst_src1":
        lines.append(f"{op} s1, s0, s1")
        dst_sgpr = "s1"
    else:  # all_same
        lines.append(f"{op} s0, s0, s0")
        dst_sgpr = "s0"

    lines.extend([
        f"v_mov_b32 v0, {dst_sgpr}",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0, v0, v1, v2, v3 done",
        "s_endpgm",
    ])
    shader_name = f"ps_{op}_alias_{mode}_{s0_val}_{s1_val}"
    spvasm = support.translate_lines(shader_name, lines)
    words = support.mrt0_words(spvasm)
    return words[0]


class TestSMulI32Translator(unittest.TestCase):
    """100% translator-backed regression suite for s_mul_i32."""

    def test_basic_multiplication(self) -> None:
        self.assertEqual(_run_binary_op("s_mul_i32", 6, 7, tag="basic1"), 42)
        self.assertEqual(_run_binary_op("s_mul_i32", 100, 200, tag="basic2"), 20000)

    def test_negative_operands(self) -> None:
        # (-3) * 4 = -12 -> 0xFFFFFFF4
        self.assertEqual(_run_binary_op("s_mul_i32", -3, 4, tag="neg1"), (-12) & MASK32)
        # (-5) * (-6) = 30
        self.assertEqual(_run_binary_op("s_mul_i32", -5, -6, tag="neg2"), 30)

    def test_zero_and_identity(self) -> None:
        self.assertEqual(_run_binary_op("s_mul_i32", 0, 12345, tag="zero"), 0)
        self.assertEqual(_run_binary_op("s_mul_i32", 12345, 1, tag="ident"), 12345)

    def test_wrapping_modulo_2_32(self) -> None:
        # 0x10001 * 0x10001 = 0x200020001 & MASK32 = 0x00020001
        self.assertEqual(_run_binary_op("s_mul_i32", 0x10001, 0x10001, tag="wrap1"), 0x00020001)
        # 0xFFFFFFFF * 0xFFFFFFFF = (-1)*(-1) = 1
        self.assertEqual(_run_binary_op("s_mul_i32", 0xFFFFFFFF, 0xFFFFFFFF, tag="wrap2"), 1)

    def test_imm_src1_operand(self) -> None:
        res = _run_binary_op("s_mul_i32", 25, 4, s1_is_imm=True, tag="imm1")
        self.assertEqual(res, 100)

    def test_imm_src0_operand(self) -> None:
        res = _run_binary_op("s_mul_i32", 12, 12, s0_is_imm=True, tag="imm0")
        self.assertEqual(res, 144)

    def test_large_literal_operand(self) -> None:
        a = 0x1234
        b = 0x5678
        res = _run_binary_op("s_mul_i32", a, b, s1_is_imm=True, tag="lit")
        self.assertEqual(res, (a * b) & MASK32)

    def test_register_aliasing_dst_src0(self) -> None:
        # s0 = s0 * s1
        res = _run_aliased("s_mul_i32", 15, 3, "dst_src0")
        self.assertEqual(res, 45)

    def test_register_aliasing_dst_src1(self) -> None:
        # s1 = s0 * s1
        res = _run_aliased("s_mul_i32", 11, 8, "dst_src1")
        self.assertEqual(res, 88)

    def test_register_aliasing_all_same(self) -> None:
        # s0 = s0 * s0
        res = _run_aliased("s_mul_i32", 9, 9, "all_same")
        self.assertEqual(res, 81)

    def test_scc_preservation_scc1(self) -> None:
        # s_mul_i32 must not overwrite SCC when SCC=1
        res, scc = _run_with_scc("s_mul_i32", 10, 10, initial_scc=1)
        self.assertEqual(res, 100)
        self.assertEqual(scc, 1)

    def test_scc_preservation_scc0(self) -> None:
        # s_mul_i32 must not overwrite SCC when SCC=0
        res, scc = _run_with_scc("s_mul_i32", 10, 10, initial_scc=0)
        self.assertEqual(res, 100)
        self.assertEqual(scc, 0)

    def test_spirv_opimul_emission(self) -> None:
        lines = [
            "v_readfirstlane_b32 s0, v0",
            "s_mov_b32 s1, 4",
            "s_mul_i32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_s_mul_i32_opcode_check", lines)
        self.assertIn("OpIMul %uint", spvasm)

    def test_random_fuzz_batches(self) -> None:
        pairs = [
            (0xDEADBEEF, 0x00000007),
            (0x12345678, 0x000000FF),
            (0x80000000, 0x00000002),
            (0x7FFFFFFF, 0x00000003),
            (0xAAAAAAAA, 0x55555555),
        ]
        for idx, (a, b) in enumerate(pairs):
            with self.subTest(a=hex(a), b=hex(b)):
                res = _run_binary_op("s_mul_i32", a, b, tag=f"fuzz_{idx}")
                self.assertEqual(res, (a * b) & MASK32)


class TestSMulHiI32Translator(unittest.TestCase):
    """100% translator-backed regression suite for s_mul_hi_i32 (signed high product)."""

    def test_small_positive_zero_high(self) -> None:
        # 100 * 200 = 20000 -> high 32 bits = 0
        self.assertEqual(_run_binary_op("s_mul_hi_i32", 100, 200, tag="pos_zero"), 0)

    def test_small_negative_product(self) -> None:
        # -2 * 100 = -200 -> in 64-bit signed two's complement, high 32 bits = 0xFFFFFFFF
        self.assertEqual(_run_binary_op("s_mul_hi_i32", -2, 100, tag="neg_high"), 0xFFFFFFFF)

    def test_negative_times_negative(self) -> None:
        # (-1) * (-1) = 1 -> high 32 bits = 0
        self.assertEqual(_run_binary_op("s_mul_hi_i32", -1, -1, tag="neg_neg"), 0)

    def test_large_signed_product(self) -> None:
        # -2147483648 * -2147483648 = 2^62 = 0x40000000_00000000 -> high 32 bits = 0x40000000
        self.assertEqual(_run_binary_op("s_mul_hi_i32", 0x80000000, 0x80000000, tag="min_min"), 0x40000000)

    def test_int32_min_times_minus_one(self) -> None:
        # -2147483648 * -1 = +2147483648 (0x00000000_80000000) -> high 32 bits = 0
        self.assertEqual(_run_binary_op("s_mul_hi_i32", 0x80000000, -1, tag="min_mone"), 0)

    def test_int32_max_times_int32_max(self) -> None:
        # 0x7FFFFFFF * 0x7FFFFFFF = 0x3FFFFFFF_00000001 -> high 32 bits = 0x3FFFFFFF
        self.assertEqual(_run_binary_op("s_mul_hi_i32", 0x7FFFFFFF, 0x7FFFFFFF, tag="max_max"), 0x3FFFFFFF)

    def test_int32_min_times_int32_max(self) -> None:
        # (-2^31) * (2^31 - 1) = -2^62 + 2^31 = 0xC0000000_80000000 -> high 32 bits = 0xC0000000
        self.assertEqual(_run_binary_op("s_mul_hi_i32", 0x80000000, 0x7FFFFFFF, tag="min_max"), 0xC0000000)

    def test_imm_src1_operand(self) -> None:
        # 0x40000000 * 4 = 0x100000000 -> high 32 bits = 1
        res = _run_binary_op("s_mul_hi_i32", 0x40000000, 4, s1_is_imm=True, tag="imm1")
        self.assertEqual(res, 1)

    def test_imm_src0_operand(self) -> None:
        # -1 * 5 = -5 -> high 32 bits = 0xFFFFFFFF
        res = _run_binary_op("s_mul_hi_i32", -1, 5, s0_is_imm=True, tag="imm0")
        self.assertEqual(res, 0xFFFFFFFF)

    def test_register_aliasing_dst_src0(self) -> None:
        res = _run_aliased("s_mul_hi_i32", 0x80000000, 0x80000000, "dst_src0")
        self.assertEqual(res, 0x40000000)

    def test_register_aliasing_all_same(self) -> None:
        # (-1) * (-1) = 1 -> high 32 bits = 0
        res = _run_aliased("s_mul_hi_i32", -1, -1, "all_same")
        self.assertEqual(res, 0)

    def test_scc_preservation_scc1(self) -> None:
        res, scc = _run_with_scc("s_mul_hi_i32", 0x7FFFFFFF, 2, initial_scc=1)
        self.assertEqual(scc, 1)

    def test_scc_preservation_scc0(self) -> None:
        res, scc = _run_with_scc("s_mul_hi_i32", 0x7FFFFFFF, 2, initial_scc=0)
        self.assertEqual(scc, 0)

    def test_spirv_opsmulextended_emission(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 2",
            "s_mul_hi_i32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_s_mul_hi_i32_opcode_check", lines)
        self.assertIn("OpSMulExtended", spvasm)

    def test_random_signed_fuzz_batches(self) -> None:
        pairs = [
            (-1000, 2000),
            (0x3FFFFFFF, -2),
            (-0x40000000, 3),
            (-1234567, -7654321),
            (0x10000000, 0x20),
        ]
        for idx, (a, b) in enumerate(pairs):
            with self.subTest(a=a, b=b):
                res = _run_binary_op("s_mul_hi_i32", a, b, tag=f"fuzz_hi_i_{idx}")
                expected = ((a * b) >> 32) & MASK32
                self.assertEqual(res, expected)


class TestSMulHiU32Translator(unittest.TestCase):
    """100% translator-backed regression suite for s_mul_hi_u32 (unsigned high product)."""

    def test_small_unsigned_product_zero_high(self) -> None:
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 100, 200, tag="small"), 0)

    def test_exact_2_32_boundary(self) -> None:
        # 0x80000000 * 2 = 0x100000000 -> high 32 bits = 1
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0x80000000, 2, tag="boundary"), 1)

    def test_max_uint32_squared(self) -> None:
        # (2^32 - 1)^2 = 0xFFFFFFFE_00000001 -> high 32 bits = 0xFFFFFFFE
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0xFFFFFFFF, 0xFFFFFFFF, tag="max_sq"), 0xFFFFFFFE)

    def test_zero_operand(self) -> None:
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0xFFFFFFFF, 0, tag="zero"), 0)

    def test_powers_of_two(self) -> None:
        # (0x80000000) * 4 = 2^33 = 0x2_00000000 -> high 32 bits = 2
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0x80000000, 4, tag="pow2_a"), 2)
        # (0x40000000) * 8 = 2^33 = 0x2_00000000 -> high 32 bits = 2
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0x40000000, 8, tag="pow2_b"), 2)

    def test_asymmetric_high_product(self) -> None:
        # 0xFFFFFFFF * 0x80000000 = 0x7FFFFFFF_80000000 -> high 32 bits = 0x7FFFFFFF
        self.assertEqual(_run_binary_op("s_mul_hi_u32", 0xFFFFFFFF, 0x80000000, tag="asymm"), 0x7FFFFFFF)

    def test_imm_src1_operand(self) -> None:
        # 0xFFFFFFFF * 5 = 0x4_FFFFFFFB -> high 32 bits = 4
        res = _run_binary_op("s_mul_hi_u32", 0xFFFFFFFF, 5, s1_is_imm=True, tag="imm1")
        self.assertEqual(res, 4)

    def test_imm_src0_operand(self) -> None:
        res = _run_binary_op("s_mul_hi_u32", 5, 0xFFFFFFFF, s0_is_imm=True, tag="imm0")
        self.assertEqual(res, 4)

    def test_register_aliasing_dst_src0(self) -> None:
        res = _run_aliased("s_mul_hi_u32", 0xFFFFFFFF, 0xFFFFFFFF, "dst_src0")
        self.assertEqual(res, 0xFFFFFFFE)

    def test_register_aliasing_all_same(self) -> None:
        res = _run_aliased("s_mul_hi_u32", 0x80000000, 0x80000000, "all_same")
        # 0x80000000 * 0x80000000 = 0x40000000_00000000 -> high = 0x40000000
        self.assertEqual(res, 0x40000000)

    def test_scc_preservation_scc1(self) -> None:
        res, scc = _run_with_scc("s_mul_hi_u32", 0xFFFFFFFF, 0xFFFFFFFF, initial_scc=1)
        self.assertEqual(scc, 1)

    def test_scc_preservation_scc0(self) -> None:
        res, scc = _run_with_scc("s_mul_hi_u32", 0xFFFFFFFF, 0xFFFFFFFF, initial_scc=0)
        self.assertEqual(scc, 0)

    def test_spirv_opumulextended_emission(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x80000000",
            "s_mov_b32 s1, 2",
            "s_mul_hi_u32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_s_mul_hi_u32_opcode_check", lines)
        self.assertIn("OpUMulExtended", spvasm)

    def test_reconstruct_full_64bit_product(self) -> None:
        """Executes both s_mul_i32 and s_mul_hi_u32 and reconstructs the full 64-bit product."""
        pairs = [
            (0x12345678, 0x87654321),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x00000002),
            (0xDEADBEEF, 0xCAFEBABE),
            (0x00001000, 0x00001000),
        ]
        for idx, (a, b) in enumerate(pairs):
            lines = [
                f"s_mov_b32 s0, {a:#x}",
                f"s_mov_b32 s1, {b:#x}",
                "s_mul_i32 s2, s0, s1",
                "s_mul_hi_u32 s3, s0, s1",
                "v_mov_b32 v0, s2",
                "v_mov_b32 v1, s3",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
                "exp mrt0, v0, v1, v2, v3 done",
                "s_endpgm",
            ]
            spvasm = support.translate_lines(f"ps_reconstruct_{idx}", lines)
            words = support.mrt0_words(spvasm)
            lo, hi = words[0], words[1]
            reconstructed = (hi << 32) | lo
            self.assertEqual(reconstructed, a * b)


class TestSALUMulNegativeAndMutation(unittest.TestCase):
    """Negative controls and mutation detection guards for SALU multiply lowering."""

    def test_syntax_rejection_missing_operands(self) -> None:
        lines = [
            "s_mul_i32 s0, s1",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_s_mul_missing_op", lines)

    def test_mutation_detection_signed_vs_unsigned_high(self) -> None:
        # For operands 0xFFFFFFFF (-1 signed, 4294967295 unsigned) and 5:
        # s_mul_hi_i32: (-1)*5 = -5 -> high = 0xFFFFFFFF
        # s_mul_hi_u32: (4294967295)*5 = 21474836475 -> high = 4
        # Asserts that lowered signed and unsigned high instructions never conflate
        hi_signed = _run_binary_op("s_mul_hi_i32", 0xFFFFFFFF, 5, tag="mut_s")
        hi_unsigned = _run_binary_op("s_mul_hi_u32", 0xFFFFFFFF, 5, tag="mut_u")
        self.assertEqual(hi_signed, 0xFFFFFFFF)
        self.assertEqual(hi_unsigned, 4)
        self.assertNotEqual(hi_signed, hi_unsigned)

    def test_mutation_detection_low_vs_high(self) -> None:
        # For operands 0x10000 and 0x10000:
        # Low 32 bits = (0x10000 * 0x10000) & 0xFFFFFFFF = 0
        # High 32 bits = (0x10000 * 0x10000) >> 32 = 1
        low = _run_binary_op("s_mul_i32", 0x10000, 0x10000, tag="mut_lo")
        high = _run_binary_op("s_mul_hi_u32", 0x10000, 0x10000, tag="mut_hi")
        self.assertEqual(low, 0)
        self.assertEqual(high, 1)
        self.assertNotEqual(low, high)


if __name__ == "__main__":
    unittest.main()
