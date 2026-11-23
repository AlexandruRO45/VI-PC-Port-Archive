from __future__ import annotations

"""s_and_b32 (RDNA2 SOP2) hardware-referenced semantic regression tests.

Covers:
- Scalar 32-bit bitwise AND: D = S0 & S1
- Scalar condition code (SCC): SCC is set to 1 if result != 0, 0 if result == 0
- Bitmask patterns, sign-extended negative immediates, and literals
- Register operand variations: SGPRs, inline constants, and register aliasing (in-place)
- Full translation to SPIR-V via tools/rdna2spirv.py and execution via spirv_mini_interp

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402


def run_and_b32(instruction: str, seeds: dict[str, int]) -> tuple[int, int]:
    """Execute shader exercising `s_and_b32`, returning (result, scc)."""
    prog = []
    for reg, val in sorted(seeds.items()):
        prog.append(f"s_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("s_cselect_b32 s2, 1, 0")
    prog.append("v_mov_b32 v0, s0")
    prog.append("v_mov_b32 v1, s2")
    prog.append("exp mrt0 v0, v1, v0, v1 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_and_b32", prog)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class SaluAndB32HardwareTests(unittest.TestCase):
    def test_basic_bitwise_and(self) -> None:
        """Test standard 32-bit bitwise AND operations."""
        cases = [
            (0x12345678, 0x0F0F0F0F, 0x02040608),
            (0xAAAAAAAA, 0x55555555, 0x00000000),
            (0xFFFFFFFF, 0xCAFEBABE, 0xCAFEBABE),
            (0x00000000, 0xDEADBEEF, 0x00000000),
            (0x80000000, 0x80000000, 0x80000000),
            (0x0000FFFF, 0xFFFF0000, 0x00000000),
            (0x13579BDF, 0x2468ACE0, 0x004088C0),
        ]
        for s0_val, s1_val, expected in cases:
            with self.subTest(s0=hex(s0_val), s1=hex(s1_val)):
                res, scc = run_and_b32(
                    "s_and_b32 s0, s0, s1",
                    {"s0": s0_val, "s1": s1_val},
                )
                self.assertEqual(res, expected)
                expected_scc = 1 if expected != 0 else 0
                self.assertEqual(scc, expected_scc)

    def test_scc_condition_code(self) -> None:
        """SCC flag is 1 for non-zero result, 0 for zero result."""
        # Non-zero result sets SCC = 1
        res, scc = run_and_b32(
            "s_and_b32 s0, s0, s1",
            {"s0": 1, "s1": 1},
        )
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

        # Zero result sets SCC = 0
        res, scc = run_and_b32(
            "s_and_b32 s0, s0, s1",
            {"s0": 0x10, "s1": 0x20},
        )
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_immediate_constants(self) -> None:
        """Bitwise AND with inline constants and 32-bit literals."""
        # Inline 0
        res, scc = run_and_b32("s_and_b32 s0, s0, 0", {"s0": 0xFFFFFFFF})
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        # Inline -1 (0xFFFFFFFF)
        res, scc = run_and_b32("s_and_b32 s0, s0, -1", {"s0": 0x12345678})
        self.assertEqual(res, 0x12345678)
        self.assertEqual(scc, 1)

        # Literal constant mask
        res, scc = run_and_b32("s_and_b32 s0, s0, 0x000000FF", {"s0": 0x12345678})
        self.assertEqual(res, 0x78)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        """Register aliasing: self-AND is idempotent."""
        val = 0x87654321
        res, scc = run_and_b32("s_and_b32 s0, s0, s0", {"s0": val})
        self.assertEqual(res, val)
        self.assertEqual(scc, 1)

    def test_spirv_structure(self) -> None:
        """Verify that translation properly folds or emits bitwise AND operations."""
        prog = [
            "s_mov_b32 s0, 0x0f0f0f0f",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_and_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_and_b32_struct", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x000F000F)


if __name__ == "__main__":
    unittest.main()
