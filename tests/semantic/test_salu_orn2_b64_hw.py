"""
RDNA2 SALU instruction semantic tests: s_orn2_b64.

Exercises s_orn2_b64 through tools/rdna2spirv.py and spirv_mini_interp (support.translate_lines).
RDNA2 ISA, SOP2:
    s_orn2_b64 s[d:d+1], s[s0:s0+1], s[s1:s1+1]
Semantics:
    D = S0 | ~S1
    SCC = (D != 0)

Validates:
1. Mathematical correctness of OR-with-NOT logic across 64-bit boundaries
2. Full truth table and algebraic identity properties (A | ~A = ~0, 0 | ~ALL = 0)
3. Register aliasing (D == S0, D == S1, S0 == S1)
4. SCC generation across zero and non-zero results
5. End-to-end SPIR-V compilation and execution via support.translate_lines and mrt0_words
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def ref_orn2_b64(s0: int, s1: int) -> tuple[int, int]:
    """
    Reference mathematical model for s_orn2_b64.
    Returns (result_64, scc).
    """
    d = (s0 | (~s1 & MASK64)) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


def run_orn2_shader(s0: int, s1: int, alias: str = "none") -> tuple[int, int]:
    """
    Translate and execute a pixel shader testing s_orn2_b64.
    Returns (d_result_64, scc).
    """
    s0_lo = s0 & MASK32
    s0_hi = (s0 >> 32) & MASK32
    s1_lo = s1 & MASK32
    s1_hi = (s1 >> 32) & MASK32

    if alias == "none":
        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_orn2_b64 s[4:5], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
    elif alias == "dst_is_s0":
        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_orn2_b64 s[0:1], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s1",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
    elif alias == "dst_is_s1":
        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_orn2_b64 s[2:3], s[0:1], s[2:3]",
            "v_mov_b32_e32 v0, s2",
            "v_mov_b32_e32 v1, s3",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
    elif alias == "s0_is_s1":
        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            "s_orn2_b64 s[4:5], s[0:1], s[0:1]",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
    else:
        raise ValueError(f"Unknown alias: {alias}")

    spv = support.translate_lines(f"ps_salu_orn2_b64_{alias}", lines)
    out = support.mrt0_words(spv)
    d_got = (out[1] << 32) | out[0]
    scc_got = out[2]
    return d_got, scc_got


class TestSaluOrn2B64Hw(unittest.TestCase):
    """Semantic tests for s_orn2_b64 verified through tools/rdna2spirv.py."""

    def test_truth_table_properties(self) -> None:
        """Verify fundamental Boolean identities for OR-NOT."""
        # 0 | ~ALL64 = 0 (the only case where result is 0 and SCC is 0)
        d, scc = run_orn2_shader(0x0, MASK64)
        exp_d, exp_scc = ref_orn2_b64(0x0, MASK64)
        self.assertEqual((d, scc), (0x0, 0))
        self.assertEqual((d, scc), (exp_d, exp_scc))

        # A | ~A = ALL64 (tautology)
        for val in (0x123456789ABCDEF0, 0x5555555555555555, 0x0, MASK64):
            with self.subTest(val=hex(val)):
                d, scc = run_orn2_shader(val, val, alias="s0_is_s1")
                exp_d, exp_scc = ref_orn2_b64(val, val)
                self.assertEqual((d, scc), (MASK64, 1))
                self.assertEqual((d, scc), (exp_d, exp_scc))

        # A | ~0 = ALL64
        d, scc = run_orn2_shader(0x123456789ABCDEF0, 0x0)
        self.assertEqual(d, MASK64)
        self.assertEqual(scc, 1)

        # 0 | ~A = ~A
        d, scc = run_orn2_shader(0x0, 0x00000000FFFFFFFF)
        self.assertEqual(d, 0xFFFFFFFF00000000)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        """Verify that in-place updates do not cause read-after-write hazards."""
        s0 = 0xAAAAAAAAAAAAAAAA
        s1 = 0x5555555555555555
        exp_d, exp_scc = ref_orn2_b64(s0, s1)

        # D == S0
        d_s0, scc_s0 = run_orn2_shader(s0, s1, alias="dst_is_s0")
        self.assertEqual((d_s0, scc_s0), (exp_d, exp_scc))

        # D == S1
        d_s1, scc_s1 = run_orn2_shader(s0, s1, alias="dst_is_s1")
        self.assertEqual((d_s1, scc_s1), (exp_d, exp_scc))

    def test_various_bit_patterns(self) -> None:
        """Test representative bit patterns, bit positions, and edge cases."""
        patterns = [
            (0x00000000_00000000, 0x00000000_00000000),
            (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF),
            (0xFFFFFFFF_FFFFFFFF, 0x00000000_00000000),
            (0xFFFFFFFF_FFFFFFFF, 0xFFFFFFFF_FFFFFFFF),
            (0x12345678_9ABCDEF0, 0x0F0F0F0F_F0F0F0F0),
            (0x80000000_00000000, 0x00000000_00000001),
            (0x00000000_00000001, 0x80000000_00000000),
            (0x55555555_55555555, 0x33333333_33333333),
            (0x7FFFFFFF_FFFFFFFF, 0x80000000_00000000),
        ]

        for s0, s1 in patterns:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = run_orn2_shader(s0, s1)
                exp_d, exp_scc = ref_orn2_b64(s0, s1)
                self.assertEqual(d, exp_d, f"Mismatch for s0={hex(s0)}, s1={hex(s1)}")
                self.assertEqual(scc, exp_scc, f"SCC mismatch for s0={hex(s0)}, s1={hex(s1)}")


if __name__ == "__main__":
    unittest.main()
