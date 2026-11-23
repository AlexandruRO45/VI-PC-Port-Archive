from __future__ import annotations

"""Hardware-referenced semantic regression tests for RDNA2 VALU 16-bit 3-Input Min/Max/Med selection:
  - v_min3_u16: Unsigned 16-bit integer minimum of 3 operands, destination upper half preserved
  - v_max3_u16: Unsigned 16-bit integer maximum of 3 operands, destination upper half preserved
  - v_med3_u16: Unsigned 16-bit integer median of 3 operands, destination upper half preserved
  - v_min3_i16: Signed 16-bit integer minimum of 3 operands, destination upper half preserved
  - v_max3_i16: Signed 16-bit integer maximum of 3 operands, destination upper half preserved
  - v_med3_i16: Signed 16-bit integer median of 3 operands, destination upper half preserved

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


def run_minmaxmed16_op(
    op: str,
    s0: int,
    s1: int,
    s2: int,
    *,
    init_dst: int = 0x55AA0000,
    alias_dst: int | None = None,
    mods: str = "",
    name: str = "ps_minmaxmed16",
) -> int:
    """Assemble, translate, and execute a 16-bit 3-operand min/max/med instruction."""
    dst = f"v{alias_dst}" if alias_dst is not None else "v0"
    mod_str = f" {mods}" if mods else ""
    lines = [
        f"v_mov_b32 v0, {init_dst & MASK32:#x}",
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        f"{op} {dst}, v1, v2, v3{mod_str}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# Suite 1: v_min3_u16 (Unsigned 16-bit Minimum)
# ==============================================================================

class TestVMin3U16(unittest.TestCase):
    """Unsigned 16-bit 3-input minimum tests."""

    def test_min3_u16_basic(self) -> None:
        """min3(10, 5, 20) = 5."""
        res = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, name="ps_min3_u16_basic")
        self.assertEqual(res & MASK16, 5)

    def test_min3_u16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, init_dst=0x55AA0000, name="ps_min3_u16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 5)

    def test_min3_u16_edge_cases(self) -> None:
        """Unsigned edge cases and boundaries."""
        cases = [
            (0, 1, 65535, 0),
            (32768, 32767, 65535, 32767),
            (5, 5, 5, 5),
            (65535, 65535, 0, 0),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_min3_u16", a, b, c, name=f"ps_min3_u16_{a}")
                self.assertEqual(res & MASK16, expected)

    def test_min3_u16_register_aliasing(self) -> None:
        """Register aliasing with destination operand."""
        res_v1 = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, alias_dst=1, name="ps_min3_u16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 5)
        res_v2 = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, alias_dst=2, name="ps_min3_u16_alias_v2")
        self.assertEqual(res_v2 & MASK16, 5)
        res_v3 = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, alias_dst=3, name="ps_min3_u16_alias_v3")
        self.assertEqual(res_v3 & MASK16, 5)

    def test_min3_u16_spirv_structure(self) -> None:
        """SPIR-V translation emits UMin."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_min3_u16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_min3_u16", lines)
        self.assertIn("UMin", spv)

    def test_min3_u16_mutation_detection(self) -> None:
        """UMin is strictly distinguished from UMax and UClamp."""
        res = run_minmaxmed16_op("v_min3_u16", 10, 5, 20, name="ps_min3_u16_mut")
        self.assertEqual(res & MASK16, 5)
        # UMax would yield 20, UClamp would yield 10
        self.assertNotEqual(res & MASK16, 20)
        self.assertNotEqual(res & MASK16, 10)

    def test_min3_u16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_min3_u16", ["v_min3_u16", "s_endpgm"])


# ==============================================================================
# Suite 2: v_max3_u16 (Unsigned 16-bit Maximum)
# ==============================================================================

class TestVMax3U16(unittest.TestCase):
    """Unsigned 16-bit 3-input maximum tests."""

    def test_max3_u16_basic(self) -> None:
        """max3(10, 5, 20) = 20."""
        res = run_minmaxmed16_op("v_max3_u16", 10, 5, 20, name="ps_max3_u16_basic")
        self.assertEqual(res & MASK16, 20)

    def test_max3_u16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_max3_u16", 10, 5, 20, init_dst=0x55AA0000, name="ps_max3_u16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 20)

    def test_max3_u16_edge_cases(self) -> None:
        """Unsigned edge cases and boundaries."""
        cases = [
            (0, 1, 65535, 65535),
            (32768, 32767, 65535, 65535),
            (5, 5, 5, 5),
            (0, 100, 200, 200),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_max3_u16", a, b, c, name=f"ps_max3_u16_{a}")
                self.assertEqual(res & MASK16, expected)

    def test_max3_u16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_minmaxmed16_op("v_max3_u16", 10, 5, 20, alias_dst=1, name="ps_max3_u16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 20)
        res_v3 = run_minmaxmed16_op("v_max3_u16", 10, 5, 20, alias_dst=3, name="ps_max3_u16_alias_v3")
        self.assertEqual(res_v3 & MASK16, 20)

    def test_max3_u16_spirv_structure(self) -> None:
        """SPIR-V translation emits UMax."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_max3_u16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_max3_u16", lines)
        self.assertIn("UMax", spv)

    def test_max3_u16_mutation_detection(self) -> None:
        """UMax is strictly distinguished from UMin."""
        res = run_minmaxmed16_op("v_max3_u16", 10, 5, 20, name="ps_max3_u16_mut")
        self.assertEqual(res & MASK16, 20)
        # UMin would yield 5
        self.assertNotEqual(res & MASK16, 5)

    def test_max3_u16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_max3_u16", ["v_max3_u16", "s_endpgm"])


# ==============================================================================
# Suite 3: v_med3_u16 (Unsigned 16-bit Median)
# ==============================================================================

class TestVMed3U16(unittest.TestCase):
    """Unsigned 16-bit 3-input median tests."""

    def test_med3_u16_basic(self) -> None:
        """med3(10, 5, 20) = 10."""
        res = run_minmaxmed16_op("v_med3_u16", 10, 5, 20, name="ps_med3_u16_basic")
        self.assertEqual(res & MASK16, 10)

    def test_med3_u16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_med3_u16", 10, 5, 20, init_dst=0x55AA0000, name="ps_med3_u16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 10)

    def test_med3_u16_edge_cases(self) -> None:
        """Unsigned median edge cases and permutations."""
        cases = [
            (0, 1, 65535, 1),
            (32768, 32767, 65535, 32768),
            (5, 5, 5, 5),
            (20, 5, 10, 10),
            (65535, 0, 100, 100),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_med3_u16", a, b, c, name=f"ps_med3_u16_{a}")
                self.assertEqual(res & MASK16, expected)

    def test_med3_u16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_minmaxmed16_op("v_med3_u16", 10, 5, 20, alias_dst=1, name="ps_med3_u16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 10)
        res_v2 = run_minmaxmed16_op("v_med3_u16", 10, 5, 20, alias_dst=2, name="ps_med3_u16_alias_v2")
        self.assertEqual(res_v2 & MASK16, 10)

    def test_med3_u16_spirv_structure(self) -> None:
        """SPIR-V translation emits UClamp."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_med3_u16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_med3_u16", lines)
        self.assertIn("UClamp", spv)

    def test_med3_u16_mutation_detection(self) -> None:
        """UMed is strictly distinguished from UMin and UMax."""
        res = run_minmaxmed16_op("v_med3_u16", 10, 5, 20, name="ps_med3_u16_mut")
        self.assertEqual(res & MASK16, 10)
        self.assertNotEqual(res & MASK16, 5)
        self.assertNotEqual(res & MASK16, 20)

    def test_med3_u16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_med3_u16", ["v_med3_u16", "s_endpgm"])


# ==============================================================================
# Suite 4: v_min3_i16 (Signed 16-bit Minimum)
# ==============================================================================

class TestVMin3I16(unittest.TestCase):
    """Signed 16-bit 3-input minimum tests."""

    def test_min3_i16_basic(self) -> None:
        """min3(-2 (0xFFFE), 5, 20) = -2 (0xFFFE)."""
        res = run_minmaxmed16_op("v_min3_i16", 0xFFFE, 5, 20, name="ps_min3_i16_basic")
        self.assertEqual(res & MASK16, 0xFFFE)

    def test_min3_i16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_min3_i16", 0xFFFE, 5, 20, init_dst=0x55AA0000, name="ps_min3_i16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 0xFFFE)

    def test_min3_i16_edge_cases(self) -> None:
        """Signed edge cases: -32768, -1, 32767."""
        cases = [
            (0x8000, 0, 0x7FFF, 0x8000),      # min(-32768, 0, 32767) = -32768
            (0xFFFF, 0xFFFE, 1, 0xFFFE),      # min(-1, -2, 1) = -2
            (0x8001, 0x7FFE, 0, 0x8001),      # min(-32767, 32766, 0) = -32767
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_min3_i16", a, b, c, name=f"ps_min3_i16_{a:#x}")
                self.assertEqual(res & MASK16, expected)

    def test_min3_i16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_minmaxmed16_op("v_min3_i16", 0xFFFE, 5, 20, alias_dst=1, name="ps_min3_i16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 0xFFFE)
        res_v2 = run_minmaxmed16_op("v_min3_i16", 0xFFFE, 5, 20, alias_dst=2, name="ps_min3_i16_alias_v2")
        self.assertEqual(res_v2 & MASK16, 0xFFFE)

    def test_min3_i16_spirv_structure(self) -> None:
        """SPIR-V translation emits SMin."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xfffe",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_min3_i16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_min3_i16", lines)
        self.assertIn("SMin", spv)

    def test_min3_i16_mutation_detection(self) -> None:
        """Signed min strictly distinguishes negative values from unsigned interpretations."""
        # 0xFFFE is -2 in signed, 65534 in unsigned
        res = run_minmaxmed16_op("v_min3_i16", 0xFFFE, 5, 20, name="ps_min3_i16_mut")
        self.assertEqual(res & MASK16, 0xFFFE)
        # Unsigned min would produce 5
        self.assertNotEqual(res & MASK16, 5)

    def test_min3_i16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_min3_i16", ["v_min3_i16", "s_endpgm"])


# ==============================================================================
# Suite 5: v_max3_i16 (Signed 16-bit Maximum)
# ==============================================================================

class TestVMax3I16(unittest.TestCase):
    """Signed 16-bit 3-input maximum tests."""

    def test_max3_i16_basic(self) -> None:
        """max3(-2 (0xFFFE), 5, 20) = 20."""
        res = run_minmaxmed16_op("v_max3_i16", 0xFFFE, 5, 20, name="ps_max3_i16_basic")
        self.assertEqual(res & MASK16, 20)

    def test_max3_i16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_max3_i16", 0xFFFE, 5, 20, init_dst=0x55AA0000, name="ps_max3_i16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 20)

    def test_max3_i16_edge_cases(self) -> None:
        """Signed maximum edge cases."""
        cases = [
            (0x8000, 0, 0x7FFF, 0x7FFF),      # max(-32768, 0, 32767) = 32767
            (0xFFFF, 0xFFFE, 1, 1),           # max(-1, -2, 1) = 1
            (0x8001, 0x7FFE, 0, 0x7FFE),      # max(-32767, 32766, 0) = 32766
            (0xFFFE, 0xFFFD, 0xFFFC, 0xFFFE), # max(-2, -3, -4) = -2 (0xFFFE)
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_max3_i16", a, b, c, name=f"ps_max3_i16_{a:#x}")
                self.assertEqual(res & MASK16, expected)

    def test_max3_i16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_minmaxmed16_op("v_max3_i16", 0xFFFE, 5, 20, alias_dst=1, name="ps_max3_i16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 20)
        res_v3 = run_minmaxmed16_op("v_max3_i16", 0xFFFE, 5, 20, alias_dst=3, name="ps_max3_i16_alias_v3")
        self.assertEqual(res_v3 & MASK16, 20)

    def test_max3_i16_spirv_structure(self) -> None:
        """SPIR-V translation emits SMax."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xfffe",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_max3_i16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_max3_i16", lines)
        self.assertIn("SMax", spv)

    def test_max3_i16_mutation_detection(self) -> None:
        """SMax is strictly distinguished from SMin."""
        res = run_minmaxmed16_op("v_max3_i16", 0xFFFE, 5, 20, name="ps_max3_i16_mut")
        self.assertEqual(res & MASK16, 20)
        # SMin would yield -2 (0xFFFE)
        self.assertNotEqual(res & MASK16, 0xFFFE)

    def test_max3_i16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_max3_i16", ["v_max3_i16", "s_endpgm"])


# ==============================================================================
# Suite 6: v_med3_i16 (Signed 16-bit Median)
# ==============================================================================

class TestVMed3I16(unittest.TestCase):
    """Signed 16-bit 3-input median tests."""

    def test_med3_i16_basic(self) -> None:
        """med3(-2 (0xFFFE), 5, 20) = 5."""
        res = run_minmaxmed16_op("v_med3_i16", 0xFFFE, 5, 20, name="ps_med3_i16_basic")
        self.assertEqual(res & MASK16, 5)

    def test_med3_i16_upper_half_preservation(self) -> None:
        """Destination upper 16 bits are preserved."""
        res = run_minmaxmed16_op("v_med3_i16", 0xFFFE, 5, 20, init_dst=0x55AA0000, name="ps_med3_i16_pres")
        self.assertEqual((res >> 16) & MASK16, 0x55AA)
        self.assertEqual(res & MASK16, 5)

    def test_med3_i16_edge_cases(self) -> None:
        """Signed median edge cases."""
        cases = [
            (0x8000, 0, 0x7FFF, 0),           # med(-32768, 0, 32767) = 0
            (0xFFFF, 0xFFFE, 1, 0xFFFF),      # med(-1, -2, 1) = -1 (0xFFFF)
            (0x8001, 0x7FFE, 0, 0),           # med(-32767, 32766, 0) = 0
            (0xFFFD, 0xFFFE, 0xFFFC, 0xFFFD), # med(-3, -2, -4) = -3 (0xFFFD)
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                res = run_minmaxmed16_op("v_med3_i16", a, b, c, name=f"ps_med3_i16_{a:#x}")
                self.assertEqual(res & MASK16, expected)

    def test_med3_i16_register_aliasing(self) -> None:
        """Register aliasing with destination."""
        res_v1 = run_minmaxmed16_op("v_med3_i16", 0xFFFE, 5, 20, alias_dst=1, name="ps_med3_i16_alias_v1")
        self.assertEqual(res_v1 & MASK16, 5)
        res_v2 = run_minmaxmed16_op("v_med3_i16", 0xFFFE, 5, 20, alias_dst=2, name="ps_med3_i16_alias_v2")
        self.assertEqual(res_v2 & MASK16, 5)

    def test_med3_i16_spirv_structure(self) -> None:
        """SPIR-V translation emits SClamp."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xfffe",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_med3_i16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_struct_med3_i16", lines)
        self.assertIn("SClamp", spv)

    def test_med3_i16_mutation_detection(self) -> None:
        """SMed is strictly distinguished from SMin and SMax."""
        res = run_minmaxmed16_op("v_med3_i16", 0xFFFE, 5, 20, name="ps_med3_i16_mut")
        self.assertEqual(res & MASK16, 5)
        self.assertNotEqual(res & MASK16, 0xFFFE)
        self.assertNotEqual(res & MASK16, 20)

    def test_med3_i16_negative_controls(self) -> None:
        """Negative control: malformed opcode syntax is rejected."""
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_med3_i16", ["v_med3_i16", "s_endpgm"])


# ==============================================================================
# Suite 7: Integration & Cross-Opcode Pipelines
# ==============================================================================

class TestVMinMaxMed16Pipelines(unittest.TestCase):
    """Multi-instruction pipeline integration tests."""

    def test_chained_min_max_pipeline(self) -> None:
        """Chained: max3(min3(10, 5, 20), 2, 3) = max3(5, 2, 3) = 5."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 20",
            "v_mov_b32 v4, 2",
            "v_mov_b32 v5, 3",
            "v_min3_u16 v6, v1, v2, v3",      # 5
            "v_max3_u16 v0, v6, v4, v5",      # max(5, 2, 3) = 5
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_min_max", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 5)
        self.assertEqual((words[0] >> 16) & MASK16, 0x55AA)

    def test_chained_signed_pipeline(self) -> None:
        """Chained: med3(min3(-10, -5, -20), 0, -12) = med3(-20, 0, -12) = -12 (0xFFF4)."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0xFFF6",  # -10
            "v_mov_b32 v2, 0xFFFB",  # -5
            "v_mov_b32 v3, 0xFFEC",  # -20
            "v_mov_b32 v4, 0",       # 0
            "v_mov_b32 v5, 0xFFF4",  # -12
            "v_min3_i16 v6, v1, v2, v3",      # -20 (0xFFEC)
            "v_med3_i16 v0, v6, v4, v5",      # med(-20, 0, -12) = -12 (0xFFF4)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_chain_signed", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0xFFF4)
        self.assertEqual((words[0] >> 16) & MASK16, 0x55AA)

    def test_op_sel_modifier(self) -> None:
        """op_sel selects high halfwords from source registers."""
        lines = [
            "v_mov_b32 v0, 0x55AA0000",
            "v_mov_b32 v1, 0x000A0000",  # hi: 10
            "v_mov_b32 v2, 0x00050000",  # hi: 5
            "v_mov_b32 v3, 0x00140000",  # hi: 20
            "v_min3_u16 v0, v1, v2, v3 op_sel:[1,1,1,0]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_op_sel_min3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 5)
        self.assertEqual((words[0] >> 16) & MASK16, 0x55AA)


if __name__ == '__main__':
    unittest.main()
