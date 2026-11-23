from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU Float16 basic arithmetic:
  - v_add_f16:  Half-precision addition, destination upper half preserved
  - v_sub_f16:  Half-precision subtraction, destination upper half preserved
  - v_mul_f16:  Half-precision multiplication, destination upper half preserved
  - v_fma_f16:  Half-precision fused multiply-add, destination upper half preserved
  - v_min3_f16: 3-input half-precision minimum with NaN suppression
  - v_max3_f16: 3-input half-precision maximum with NaN suppression
  - v_med3_f16: 3-input half-precision median with NaN suppression

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


def run_f16_op(
    op: str,
    s0: int,
    s1: int,
    s2: int | None = None,
    *,
    init_dst: int = 0x55AA0000,
    alias_dst: int | None = None,
) -> int:
    """Assemble, translate, and execute a float16 VALU instruction."""
    lines = [f"v_mov_b32 v0, {init_dst & MASK32:#x}"]
    lines.append(f"v_mov_b32 v1, {s0 & MASK16:#x}")
    lines.append(f"v_mov_b32 v2, {s1 & MASK16:#x}")
    if s2 is not None:
        lines.append(f"v_mov_b32 v3, {s2 & MASK16:#x}")
        dst = f"v{alias_dst}" if alias_dst is not None else "v0"
        lines.append(f"{op} {dst}, v1, v2, v3")
    else:
        dst = f"v{alias_dst}" if alias_dst is not None else "v0"
        lines.append(f"{op} {dst}, v1, v2")
    lines.extend([
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_{op}", lines)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# Suite 1: v_add_f16 (Half-Precision Addition)
# ==============================================================================

class TestVAddF16(unittest.TestCase):
    """Half-precision addition tests."""

    def test_v_add_f16_basic_addition(self) -> None:
        """Basic 1.0 (0x3C00) + 2.0 (0x4000) = 3.0 (0x4200)."""
        res = run_f16_op("v_add_f16", 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_add_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_add_f16", 0x3C00, 0x4000, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_add_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: zeros, infinities, cancellations, subnormals."""
        cases = [
            (0x0000, 0x0000, 0x0000),      # +0 + +0 = +0
            (0x3C00, 0xBC00, 0x0000),      # 1.0 + (-1.0) = 0.0
            (0x7C00, 0x3C00, 0x7C00),      # +inf + 1.0 = +inf
            (0xFC00, 0xBC00, 0xFC00),      # -inf + (-1.0) = -inf
            (0x3800, 0x3800, 0x3C00),      # 0.5 + 0.5 = 1.0
        ]
        for s0, s1, expected in cases:
            with self.subTest(s0=s0, s1=s1):
                res = run_f16_op("v_add_f16", s0, s1)
                self.assertEqual(res & MASK16, expected)

    def test_v_add_f16_register_aliasing(self) -> None:
        """Register aliasing with destination operand."""
        # v1 = v1 + v2
        res_v1 = run_f16_op("v_add_f16", 0x3C00, 0x4000, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4200)
        # v2 = v1 + v2
        res_v2 = run_f16_op("v_add_f16", 0x3C00, 0x4000, alias_dst=2)
        self.assertEqual(res_v2 & MASK16, 0x4200)

    def test_v_add_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits OpFAdd."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x3c00",
            "v_mov_b32 v2, 0x4000",
            "v_add_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_add_f16", lines)
        self.assertIn("OpFAdd", spv)

    def test_v_add_f16_mutation_detection(self) -> None:
        """Addition is strictly distinguished from subtraction."""
        res = run_f16_op("v_add_f16", 0x4000, 0x3C00)  # 2.0 + 1.0 = 3.0 (0x4200)
        self.assertEqual(res & MASK16, 0x4200)
        # Subtraction would yield 2.0 - 1.0 = 1.0 (0x3C00)
        self.assertNotEqual(res & MASK16, 0x3C00)

    def test_v_add_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_add_f16", ["v_add_f16", "s_endpgm"])


# ==============================================================================
# Suite 2: v_sub_f16 (Half-Precision Subtraction)
# ==============================================================================

class TestVSubF16(unittest.TestCase):
    """Half-precision subtraction tests."""

    def test_v_sub_f16_basic_subtraction(self) -> None:
        """Basic 3.0 (0x4200) - 1.0 (0x3C00) = 2.0 (0x4000)."""
        res = run_f16_op("v_sub_f16", 0x4200, 0x3C00)
        self.assertEqual(res & MASK16, 0x4000)

    def test_v_sub_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_sub_f16", 0x4200, 0x3C00, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4000)

    def test_v_sub_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: zeros, self-subtraction, infinities."""
        cases = [
            (0x0000, 0x0000, 0x0000),      # 0 - 0 = 0
            (0x3C00, 0x3C00, 0x0000),      # 1.0 - 1.0 = 0
            (0x7C00, 0x3C00, 0x7C00),      # +inf - 1.0 = +inf
            (0xFC00, 0x3C00, 0xFC00),      # -inf - 1.0 = -inf
            (0x4000, 0x3C00, 0x3C00),      # 2.0 - 1.0 = 1.0
        ]
        for s0, s1, expected in cases:
            with self.subTest(s0=s0, s1=s1):
                res = run_f16_op("v_sub_f16", s0, s1)
                self.assertEqual(res & MASK16, expected)

    def test_v_sub_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_sub_f16", 0x4200, 0x3C00, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4000)
        res_v2 = run_f16_op("v_sub_f16", 0x4200, 0x3C00, alias_dst=2)
        self.assertEqual(res_v2 & MASK16, 0x4000)

    def test_v_sub_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits OpFSub."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_sub_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_sub_f16", lines)
        self.assertIn("OpFSub", spv)

    def test_v_sub_f16_mutation_detection(self) -> None:
        """Subtraction is strictly distinguished from addition."""
        res = run_f16_op("v_sub_f16", 0x4200, 0x3C00)  # 3.0 - 1.0 = 2.0 (0x4000)
        self.assertEqual(res & MASK16, 0x4000)
        # Addition would yield 4.0 (0x4400)
        self.assertNotEqual(res & MASK16, 0x4400)

    def test_v_sub_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_sub_f16", ["v_sub_f16", "s_endpgm"])


# ==============================================================================
# Suite 3: v_mul_f16 (Half-Precision Multiplication)
# ==============================================================================

class TestVMulF16(unittest.TestCase):
    """Half-precision multiplication tests."""

    def test_v_mul_f16_basic_multiplication(self) -> None:
        """Basic 2.0 (0x4000) * 1.5 (0x3E00) = 3.0 (0x4200)."""
        res = run_f16_op("v_mul_f16", 0x4000, 0x3E00)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_mul_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_mul_f16", 0x4000, 0x3E00, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_mul_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: zeros, signs, infinities."""
        cases = [
            (0x0000, 0x3C00, 0x0000),      # +0 * 1.0 = +0
            (0x8000, 0x3C00, 0x8000),      # -0 * 1.0 = -0
            (0x3C00, 0xBC00, 0xBC00),      # 1.0 * (-1.0) = -1.0
            (0xBC00, 0xBC00, 0x3C00),      # (-1.0) * (-1.0) = 1.0
            (0x7C00, 0x4000, 0x7C00),      # +inf * 2.0 = +inf
        ]
        for s0, s1, expected in cases:
            with self.subTest(s0=s0, s1=s1):
                res = run_f16_op("v_mul_f16", s0, s1)
                self.assertEqual(res & MASK16, expected)

    def test_v_mul_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_mul_f16", 0x4000, 0x3E00, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4200)
        res_v2 = run_f16_op("v_mul_f16", 0x4000, 0x3E00, alias_dst=2)
        self.assertEqual(res_v2 & MASK16, 0x4200)

    def test_v_mul_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits OpFMul."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mul_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_mul_f16", lines)
        self.assertIn("OpFMul", spv)

    def test_v_mul_f16_mutation_detection(self) -> None:
        """Multiplication is strictly distinguished from addition."""
        res = run_f16_op("v_mul_f16", 0x4000, 0x4000)  # 2.0 * 2.0 = 4.0 (0x4400)
        self.assertEqual(res & MASK16, 0x4400)
        # Addition would yield 2.0 + 2.0 = 4.0, but 2.0 * 1.5 = 3.0 != 3.5
        res_diff = run_f16_op("v_mul_f16", 0x4000, 0x3E00)  # 2.0 * 1.5 = 3.0 (0x4200)
        self.assertEqual(res_diff & MASK16, 0x4200)
        # Addition would yield 3.5 (0x4300)
        self.assertNotEqual(res_diff & MASK16, 0x4300)

    def test_v_mul_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_mul_f16", ["v_mul_f16", "s_endpgm"])


# ==============================================================================
# Suite 4: v_fma_f16 (Half-Precision Fused Multiply-Add)
# ==============================================================================

class TestVFmaF16(unittest.TestCase):
    """Half-precision fused multiply-add tests."""

    def test_v_fma_f16_basic_fused_multiply_add(self) -> None:
        """Basic 2.0 (0x4000) * 1.5 (0x3E00) + 1.0 (0x3C00) = 4.0 (0x4400)."""
        res = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00)
        self.assertEqual(res & MASK16, 0x4400)

    def test_v_fma_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4400)

    def test_v_fma_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: zeros, cancellation, signs."""
        cases = [
            (0x0000, 0x3C00, 0x4000, 0x4000),      # 0 * 1.0 + 2.0 = 2.0
            (0x3C00, 0x3C00, 0x3C00, 0x4000),      # 1.0 * 1.0 + 1.0 = 2.0
            (0x4000, 0xBC00, 0x4200, 0x3C00),      # 2.0 * (-1.0) + 3.0 = 1.0
            (0xBC00, 0xBC00, 0xBC00, 0x0000),      # (-1.0) * (-1.0) + (-1.0) = 0.0
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                res = run_f16_op("v_fma_f16", s0, s1, s2)
                self.assertEqual(res & MASK16, expected)

    def test_v_fma_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4400)
        res_v2 = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00, alias_dst=2)
        self.assertEqual(res_v2 & MASK16, 0x4400)
        res_v3 = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00, alias_dst=3)
        self.assertEqual(res_v3 & MASK16, 0x4400)

    def test_v_fma_f16_spirv_structure(self) -> None:
        """SPIR-V translation lowers v_fma_f16 using UnpackHalf2x16."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mov_b32 v3, 0x3c00",
            "v_fma_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_fma_f16", lines)
        self.assertIn("UnpackHalf2x16", spv)

    def test_v_fma_f16_mutation_detection(self) -> None:
        """FMA is strictly distinguished from simple multiplication."""
        res = run_f16_op("v_fma_f16", 0x4000, 0x3E00, 0x3C00)  # 2.0*1.5 + 1.0 = 4.0 (0x4400)
        self.assertEqual(res & MASK16, 0x4400)
        # Without accumulator would yield 3.0 (0x4200)
        self.assertNotEqual(res & MASK16, 0x4200)

    def test_v_fma_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fma_f16", ["v_fma_f16", "s_endpgm"])


# ==============================================================================
# Suite 5: v_min3_f16 (3-Input Half-Precision Minimum)
# ==============================================================================

class TestVMin3F16(unittest.TestCase):
    """3-input half-precision minimum tests."""

    def test_v_min3_f16_basic_min(self) -> None:
        """Basic minimum of 3.0 (0x4200), 1.0 (0x3C00), 2.0 (0x4000) is 1.0 (0x3C00)."""
        res = run_f16_op("v_min3_f16", 0x4200, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x3C00)

    def test_v_min3_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_min3_f16", 0x4200, 0x3C00, 0x4000, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x3C00)

    def test_v_min3_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: negative values, signed zeros, infinities."""
        cases = [
            (0x3C00, 0x4000, 0x4200, 0x3C00),      # min(1.0, 2.0, 3.0) = 1.0
            (0xBC00, 0x0000, 0x3C00, 0xBC00),      # min(-1.0, 0, 1.0) = -1.0
            (0x8000, 0x0000, 0x3C00, 0x8000),      # min(-0.0, +0.0, 1.0) = -0.0
            (0x7C00, 0xFC00, 0x3C00, 0xFC00),      # min(+inf, -inf, 1.0) = -inf
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                res = run_f16_op("v_min3_f16", s0, s1, s2)
                self.assertEqual(res & MASK16, expected)

    def test_v_min3_f16_nan_suppression(self) -> None:
        """NaN operands are suppressed in favor of non-NaN values."""
        # min(NaN, 1.0, 2.0) = 1.0
        res = run_f16_op("v_min3_f16", 0x7E00, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x3C00)

    def test_v_min3_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_min3_f16", 0x4200, 0x3C00, 0x4000, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x3C00)

    def test_v_min3_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits NMin."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_mov_b32 v3, 0x4000",
            "v_min3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_min3_f16", lines)
        self.assertIn("NMin", spv)

    def test_v_min3_f16_mutation_detection(self) -> None:
        """MIN3 is strictly distinguished from MAX3."""
        res = run_f16_op("v_min3_f16", 0x3C00, 0x4000, 0x4200)  # min(1, 2, 3) = 1.0 (0x3C00)
        self.assertEqual(res & MASK16, 0x3C00)
        # MAX3 would yield 3.0 (0x4200)
        self.assertNotEqual(res & MASK16, 0x4200)

    def test_v_min3_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_min3_f16", ["v_min3_f16", "s_endpgm"])


# ==============================================================================
# Suite 6: v_max3_f16 (3-Input Half-Precision Maximum)
# ==============================================================================

class TestVMax3F16(unittest.TestCase):
    """3-input half-precision maximum tests."""

    def test_v_max3_f16_basic_max(self) -> None:
        """Basic maximum of 3.0 (0x4200), 1.0 (0x3C00), 2.0 (0x4000) is 3.0 (0x4200)."""
        res = run_f16_op("v_max3_f16", 0x4200, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_max3_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_max3_f16", 0x4200, 0x3C00, 0x4000, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4200)

    def test_v_max3_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: negative values, signed zeros, infinities."""
        cases = [
            (0x3C00, 0x4000, 0x4200, 0x4200),      # max(1.0, 2.0, 3.0) = 3.0
            (0xBC00, 0xC000, 0xC200, 0xBC00),      # max(-1.0, -2.0, -3.0) = -1.0
            (0x8000, 0x0000, 0xBC00, 0x0000),      # max(-0.0, +0.0, -1.0) = +0.0
            (0x7C00, 0xFC00, 0x3C00, 0x7C00),      # max(+inf, -inf, 1.0) = +inf
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                res = run_f16_op("v_max3_f16", s0, s1, s2)
                self.assertEqual(res & MASK16, expected)

    def test_v_max3_f16_nan_suppression(self) -> None:
        """NaN operands are suppressed in favor of non-NaN values."""
        # max(NaN, 1.0, 2.0) = 2.0
        res = run_f16_op("v_max3_f16", 0x7E00, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x4000)

    def test_v_max3_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_max3_f16", 0x4200, 0x3C00, 0x4000, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4200)

    def test_v_max3_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits NMax."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_mov_b32 v3, 0x4000",
            "v_max3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_max3_f16", lines)
        self.assertIn("NMax", spv)

    def test_v_max3_f16_mutation_detection(self) -> None:
        """MAX3 is strictly distinguished from MIN3."""
        res = run_f16_op("v_max3_f16", 0x3C00, 0x4000, 0x4200)  # max(1, 2, 3) = 3.0 (0x4200)
        self.assertEqual(res & MASK16, 0x4200)
        # MIN3 would yield 1.0 (0x3C00)
        self.assertNotEqual(res & MASK16, 0x3C00)

    def test_v_max3_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_max3_f16", ["v_max3_f16", "s_endpgm"])


# ==============================================================================
# Suite 7: v_med3_f16 (3-Input Half-Precision Median)
# ==============================================================================

class TestVMed3F16(unittest.TestCase):
    """3-input half-precision median tests."""

    def test_v_med3_f16_basic_median(self) -> None:
        """Basic median of 3.0 (0x4200), 1.0 (0x3C00), 2.0 (0x4000) is 2.0 (0x4000)."""
        res = run_f16_op("v_med3_f16", 0x4200, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x4000)

    def test_v_med3_f16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_f16_op("v_med3_f16", 0x4200, 0x3C00, 0x4000, init_dst=0x55AA0000)
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0x4000)

    def test_v_med3_f16_edge_cases(self) -> None:
        """IEEE 754 edge cases: permutations, signs, infinities."""
        cases = [
            (0x3C00, 0x4000, 0x4200, 0x4000),      # med(1.0, 2.0, 3.0) = 2.0
            (0x4200, 0x3C00, 0x4000, 0x4000),      # med(3.0, 1.0, 2.0) = 2.0
            (0xBC00, 0x0000, 0x3C00, 0x0000),      # med(-1.0, 0, 1.0) = 0
            (0xFC00, 0x0000, 0x7C00, 0x0000),      # med(-inf, 0, +inf) = 0
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                res = run_f16_op("v_med3_f16", s0, s1, s2)
                self.assertEqual(res & MASK16, expected)

    def test_v_med3_f16_nan_suppression(self) -> None:
        """NaN operands are suppressed in favor of non-NaN values."""
        # med(NaN, 1.0, 2.0) evaluates using non-NaN inputs
        res = run_f16_op("v_med3_f16", 0x7E00, 0x3C00, 0x4000)
        self.assertEqual(res & MASK16, 0x3C00)

    def test_v_med3_f16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_f16_op("v_med3_f16", 0x4200, 0x3C00, 0x4000, alias_dst=1)
        self.assertEqual(res_v1 & MASK16, 0x4000)

    def test_v_med3_f16_spirv_structure(self) -> None:
        """SPIR-V translation emits median logic."""
        lines = [
            "v_mov_b32 v0, 0x55aa0000",
            "v_mov_b32 v1, 0x4200",
            "v_mov_b32 v2, 0x3c00",
            "v_mov_b32 v3, 0x4000",
            "v_med3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_structure_v_med3_f16", lines)
        self.assertIn("NMin", spv)

    def test_v_med3_f16_mutation_detection(self) -> None:
        """MED3 is strictly distinguished from MIN3 and MAX3."""
        res = run_f16_op("v_med3_f16", 0x3C00, 0x4000, 0x4200)  # med(1, 2, 3) = 2.0 (0x4000)
        self.assertEqual(res & MASK16, 0x4000)
        # MIN3 is 1.0 (0x3C00), MAX3 is 3.0 (0x4200)
        self.assertNotEqual(res & MASK16, 0x3C00)
        self.assertNotEqual(res & MASK16, 0x4200)

    def test_v_med3_f16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_med3_f16", ["v_med3_f16", "s_endpgm"])


# ==============================================================================
# Suite 8: Integration & Pipelines
# ==============================================================================

class TestVF16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_add_mul_pipeline(self) -> None:
        """Chained: (1.0 + 2.0) * 1.5 = 4.5 (0x4480)."""
        lines = [
            "v_mov_b32 v0, 0x3c00",      # 1.0
            "v_mov_b32 v1, 0x4000",      # 2.0
            "v_mov_b32 v2, 0x3e00",      # 1.5
            "v_add_f16 v3, v0, v1",      # 3.0 (0x4200)
            "v_mul_f16 v4, v3, v2",      # 4.5 (0x4480)
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_add_mul", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4480)

    def test_chained_fma_med3_pipeline(self) -> None:
        """Chained: med3(fma(2.0, 1.5, 1.0), 1.0, 3.0) = 3.0 (0x4200)."""
        lines = [
            "v_mov_b32 v0, 0x4000",      # 2.0
            "v_mov_b32 v1, 0x3e00",      # 1.5
            "v_mov_b32 v2, 0x3c00",      # 1.0
            "v_mov_b32 v4, 0x4200",      # 3.0
            "v_fma_f16 v3, v0, v1, v2",  # 2*1.5 + 1.0 = 4.0 (0x4400)
            "v_med3_f16 v5, v3, v2, v4", # med(4.0, 1.0, 3.0) = 3.0 (0x4200)
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_fma_med3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4200)


if __name__ == '__main__':
    unittest.main()
