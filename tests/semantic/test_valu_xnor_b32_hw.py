"""Semantic translator tests for RDNA2 32-bit bitwise XNOR (v_xnor_b32).

Architecture:
  - RDNA2 Vector ALU (GFX10) bitwise equivalence operation:
      D.u32 = ~(S0.u32 ^ S1.u32) & 0xFFFFFFFF
  - Truth table:
      S0 bit | S1 bit | Result bit
      -------+--------+-----------
         0   |   0    |     1
         0   |   1    |     0
         1   |   0    |     0
         1   |   1    |     1
  - Operands:
      * S0: VGPR, SGPR, inline constant, or literal constant.
      * S1: VGPR.
      * D:  VGPR (32-bit).

Each test translates hand-written RDNA2 assembly via tools/rdna2spirv.py
using support.translate_lines() and runs the translated SPIR-V in
tests/semantic/spirv_mini_interp.py, checking outputs against reference values.

Standard library only: no GPU, no LLVM, no game data.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32: int = 0xFFFFFFFF


def ref_xnor32(s0: int, s1: int) -> int:
    """Mathematical reference oracle for 32-bit bitwise XNOR."""
    return (~(s0 ^ s1)) & MASK32


def run_xnor(s0: int, s1: int, form: str = "vgpr") -> int:
    """Translate and execute a fragment shader computing v_xnor_b32."""
    if form == "vgpr":
        setup = [
            f"v_mov_b32 v1, {s0 & MASK32:#x}",
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v0, v1, v2",
        ]
    elif form == "sgpr":
        setup = [
            f"s_mov_b32 s0, {s0 & MASK32:#x}",
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v0, s0, v2",
        ]
    elif form == "inline_zero":
        setup = [
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v0, 0, v2",
        ]
    elif form == "inline_neg_one":
        setup = [
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v0, -1, v2",
        ]
    elif form == "literal":
        setup = [
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            f"v_xnor_b32 v0, {s0 & MASK32:#x}, v2",
        ]
    else:
        raise ValueError(f"Unknown form: {form}")

    lines = setup + [
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_xnor_b32", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


def run_xnor_aliased(s0: int, s1: int, alias_mode: str) -> int:
    """Translate and execute v_xnor_b32 with register aliasing."""
    if alias_mode == "dst_is_src0":
        lines = [
            f"v_mov_b32 v1, {s0 & MASK32:#x}",
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v1, v1, v2",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
    elif alias_mode == "dst_is_src1":
        lines = [
            f"v_mov_b32 v1, {s0 & MASK32:#x}",
            f"v_mov_b32 v2, {s1 & MASK32:#x}",
            "v_xnor_b32 v2, v1, v2",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
    elif alias_mode == "dst_is_both":
        lines = [
            f"v_mov_b32 v1, {s0 & MASK32:#x}",
            "v_xnor_b32 v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
    else:
        raise ValueError(f"Unknown alias mode: {alias_mode}")

    spv = support.translate_lines(f"ps_xnor_alias_{alias_mode}", lines)
    return support.mrt0_words(spv)[0] & MASK32


def run_xnor_quad(cases: list[tuple[int, int]]) -> list[int]:
    """Execute four xnor operations routed to mrt0 [R, G, B, A] channels."""
    assert len(cases) == 4
    lines = []
    for i, (s0, s1) in enumerate(cases):
        lines.append(f"v_mov_b32 v{10 + i}, {s0 & MASK32:#x}")
        lines.append(f"v_mov_b32 v{20 + i}, {s1 & MASK32:#x}")
        lines.append(f"v_xnor_b32 v{i}, v{10 + i}, v{20 + i}")
    lines.extend([
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_xnor_quad", lines)
    return [w & MASK32 for w in support.mrt0_words(spv)]


class TestValuXnorB32(unittest.TestCase):
    def test_basic_xnor(self) -> None:
        # ~(0 ^ 0) = 0xFFFFFFFF
        self.assertEqual(run_xnor(0, 0), MASK32)
        # ~(0xFFFFFFFF ^ 0xFFFFFFFF) = 0xFFFFFFFF
        self.assertEqual(run_xnor(MASK32, MASK32), MASK32)
        # ~(0 ^ 0xFFFFFFFF) = 0
        self.assertEqual(run_xnor(0, MASK32), 0)
        self.assertEqual(run_xnor(MASK32, 0), 0)

    def test_identity_and_inversion(self) -> None:
        test_values = [0x00000001, 0x80000000, 0x12345678, 0xAAAAAAAA, 0x55555555, 0xDEADBEEF]
        for val in test_values:
            with self.subTest(val=hex(val)):
                # Self-equivalence: ~(x ^ x) == 0xFFFFFFFF
                self.assertEqual(run_xnor(val, val), MASK32)
                # Inversion with zero: ~(0 ^ x) == ~x
                self.assertEqual(run_xnor(0, val), (~val) & MASK32)
                # Identity with all-ones: ~(0xFFFFFFFF ^ x) == x
                self.assertEqual(run_xnor(MASK32, val), val)

    def test_alternating_bit_patterns(self) -> None:
        patterns = [
            (0xAAAAAAAA, 0x55555555, 0x00000000),
            (0xAAAAAAAA, 0xAAAAAAAA, MASK32),
            (0x55555555, 0x55555555, MASK32),
            (0x0F0F0F0F, 0xF0F0F0F0, 0x00000000),
            (0x0F0F0F0F, 0x0F0F0F0F, MASK32),
            (0x00FF00FF, 0xFF00FF00, 0x00000000),
            (0x12345678, 0x87654321, 0x6AAEEAA6),
            (0xFEDCBA98, 0x01234567, 0x00000000),
            (0xCAFEBABE, 0xDEADBEEF, 0xEBACFBAE),
        ]
        for s0, s1, expected in patterns:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_xnor(s0, s1), expected)
                self.assertEqual(run_xnor(s0, s1), ref_xnor32(s0, s1))

    def test_walking_single_bits(self) -> None:
        for bit in range(32):
            s0 = 1 << bit
            with self.subTest(bit=bit):
                # Matching single bit set:
                self.assertEqual(run_xnor(s0, s0), MASK32)
                # Single bit vs 0:
                expected_inv_bit = (~s0) & MASK32
                self.assertEqual(run_xnor(s0, 0), expected_inv_bit)
                # Adjacent bits:
                if bit < 31:
                    s1 = 1 << (bit + 1)
                    expected_adj = (~(s0 | s1)) & MASK32
                    self.assertEqual(run_xnor(s0, s1), expected_adj)

    def test_operand_forms(self) -> None:
        s0, s1 = 0x12345678, 0x87654321
        expected = ref_xnor32(s0, s1)
        # SGPR source
        self.assertEqual(run_xnor(s0, s1, form="sgpr"), expected)
        # Inline zero
        self.assertEqual(run_xnor(0, s1, form="inline_zero"), (~s1) & MASK32)
        # Inline -1 (0xFFFFFFFF)
        self.assertEqual(run_xnor(-1, s1, form="inline_neg_one"), s1)
        # Literal constant
        self.assertEqual(run_xnor(s0, s1, form="literal"), expected)

    def test_register_aliasing(self) -> None:
        s0, s1 = 0xA5A5A5A5, 0x5A5A5A5A
        expected = ref_xnor32(s0, s1)
        self.assertEqual(run_xnor_aliased(s0, s1, "dst_is_src0"), expected)
        self.assertEqual(run_xnor_aliased(s0, s1, "dst_is_src1"), expected)
        self.assertEqual(run_xnor_aliased(s0, s1, "dst_is_both"), MASK32)

    def test_quad_export_channels(self) -> None:
        cases = [
            (0, 0),
            (MASK32, MASK32),
            (0xAAAAAAAA, 0x55555555),
            (0x12345678, 0x87654321),
        ]
        expected = [ref_xnor32(s0, s1) for s0, s1 in cases]
        self.assertEqual(run_xnor_quad(cases), expected)

    def test_random_vectors(self) -> None:
        rng = random.Random(0x4E494441)
        for i in range(40):
            s0 = rng.getrandbits(32)
            s1 = rng.getrandbits(32)
            with self.subTest(iteration=i, s0=hex(s0), s1=hex(s1)):
                self.assertEqual(run_xnor(s0, s1), ref_xnor32(s0, s1))

    def test_negative_controls(self) -> None:
        s0, s1 = 0x12345678, 0x9ABCDEF0
        res = run_xnor(s0, s1)
        expected = ref_xnor32(s0, s1)
        self.assertEqual(res, expected)
        # Confirm result is distinct from XOR
        self.assertNotEqual(res, (s0 ^ s1) & MASK32)
        # Confirm result is distinct from OR
        self.assertNotEqual(res, (s0 | s1) & MASK32)
        # Confirm result is distinct from AND
        self.assertNotEqual(res, (s0 & s1) & MASK32)
        # Confirm non-trivial result differs from both inputs
        self.assertNotEqual(res, s0)
        self.assertNotEqual(res, s1)


if __name__ == "__main__":
    unittest.main()
