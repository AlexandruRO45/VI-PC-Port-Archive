from __future__ import annotations

"""v_ashrrev_i64 (RDNA2 VOP3) hardware-referenced semantic regression tests.

Covers:
- Arithmetic right shift on 64-bit signed integers:
    D[63:0] = S1[63:0] >> (S0[5:0] & 63) with sign bit (bit 63) extension
- Crossing 32-bit dword boundary between v[d] (low) and v[d+1] (high)
- Sign propagation across negative operands (e.g. -1, INT64_MIN)
- Shift count masking with 63 (bits 5:0)
- Register operand variations: VGPRs, SGPRs, and in-place register aliasing
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


def to_u64(signed_val: int) -> int:
    """Convert Python integer to 64-bit unsigned representation."""
    return signed_val & 0xFFFFFFFFFFFFFFFF


def run_ashrrev_listing(instruction: str, seeds: dict[str, int]) -> int:
    """Translate and run a minimal pixel shader exercising `v_ashrrev_i64`, returning 64-bit uint."""
    prog = []
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append("exp mrt0 v0, v1, v0, v1 done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines("ps_ashrrev_i64", prog)
    words = support.mrt0_words(spv)
    return (words[1] << 32) | words[0]


class ValuAshrrevI64HardwareTests(unittest.TestCase):
    def test_positive_values_shift_across_dword_boundary(self) -> None:
        """Test logical/arithmetic right shift of positive 64-bit integers across dword boundaries."""
        cases = [
            # (val64, shift, expected64)
            (0x100000000, 1, 0x80000000),             # high bit 32 shifts to low bit 31
            (0x123456789ABCDEF0, 4, 0x0123456789ABCDEF),
            (0x123456789ABCDEF0, 32, 0x0000000012345678),  # full dword shift
            (0x123456789ABCDEF0, 0, 0x123456789ABCDEF0),   # shift by 0 unchanged
            (0x7FFFFFFFFFFFFFFF, 1, 0x3FFFFFFFFFFFFFFF),
            (0x7FFFFFFFFFFFFFFF, 62, 1),
            (0x7FFFFFFFFFFFFFFF, 63, 0),
        ]
        for val, sh, expected in cases:
            seeds = {
                "v0": 0,
                "v1": 0,
                "v2": sh,
                "v3": val & 0xFFFFFFFF,
                "v4": (val >> 32) & 0xFFFFFFFF,
            }
            with self.subTest(val=hex(val), sh=sh):
                got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
                self.assertEqual(got, expected)

    def test_sign_propagation_negative_values(self) -> None:
        """Arithmetic shift replicates sign bit (bit 63) into all high bits."""
        # -1 (0xFFFFFFFFFFFFFFFF) shifted right by any amount remains -1
        for sh in (0, 1, 16, 31, 32, 48, 63):
            seeds = {
                "v0": 0, "v1": 0, "v2": sh,
                "v3": 0xFFFFFFFF, "v4": 0xFFFFFFFF,
            }
            got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
            self.assertEqual(got, 0xFFFFFFFFFFFFFFFF, f"Failed at shift {sh}")

        # INT64_MIN (0x8000000000000000) sign extension
        int64_min = 0x8000000000000000
        cases_min = [
            (1, 0xC000000000000000),
            (2, 0xE000000000000000),
            (16, 0xFFFF800000000000),
            (32, 0xFFFFFFFF80000000),
            (63, 0xFFFFFFFFFFFFFFFF),  # fully saturated to -1
        ]
        for sh, expected in cases_min:
            seeds = {
                "v0": 0, "v1": 0, "v2": sh,
                "v3": int64_min & 0xFFFFFFFF,
                "v4": (int64_min >> 32) & 0xFFFFFFFF,
            }
            with self.subTest(sh=sh):
                got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
                self.assertEqual(got, expected)

        # Negative small integer: -16 (0xFFFFFFFFFFFFFFF0) >> 2 = -4 (0xFFFFFFFFFFFFFFFC)
        seeds = {
            "v0": 0, "v1": 0, "v2": 2,
            "v3": 0xFFFFFFF0, "v4": 0xFFFFFFFF,
        }
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
        self.assertEqual(got, 0xFFFFFFFFFFFFFFFC)

    def test_shift_count_masking(self) -> None:
        """Shift count is masked with 63 (bits 5:0)."""
        val = 0x123456789ABCDEF0
        # Shift by 64 == shift by 0 (64 & 63 == 0)
        seeds = {
            "v0": 0, "v1": 0, "v2": 64,
            "v3": val & 0xFFFFFFFF, "v4": (val >> 32) & 0xFFFFFFFF,
        }
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
        self.assertEqual(got, val)

        # Shift by 65 == shift by 1 (65 & 63 == 1)
        seeds["v2"] = 65
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
        self.assertEqual(got, val >> 1)

        # Shift by 127 == shift by 63
        seeds["v2"] = 127
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[3:4]", seeds)
        self.assertEqual(got, val >> 63)

    def test_sgpr_sources(self) -> None:
        """Operands supplied via scalar SGPRs."""
        val = 0x4000000000000000
        # Shift amount in s0, source in s[2:3]
        seeds = {
            "s0": 2,
            "s2": val & 0xFFFFFFFF,
            "s3": (val >> 32) & 0xFFFFFFFF,
            "v0": 0, "v1": 0,
        }
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], s0, s[2:3]", seeds)
        self.assertEqual(got, 0x1000000000000000)

        # Shift amount in s0, source in v[2:3]
        seeds = {
            "s0": 4,
            "v2": val & 0xFFFFFFFF,
            "v3": (val >> 32) & 0xFFFFFFFF,
            "v0": 0, "v1": 0,
        }
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], s0, v[2:3]", seeds)
        self.assertEqual(got, 0x0400000000000000)

    def test_register_aliasing(self) -> None:
        """Destination and source register pair aliasing (in-place shift)."""
        val = 0x2000000000000000
        seeds = {
            "v0": val & 0xFFFFFFFF,
            "v1": (val >> 32) & 0xFFFFFFFF,
            "v2": 1,
        }
        got = run_ashrrev_listing("v_ashrrev_i64 v[0:1], v2, v[0:1]", seeds)
        self.assertEqual(got, 0x1000000000000000)

    def test_spirv_structure(self) -> None:
        """Verify that translation produces proper OpShiftRightArithmetic %ulong."""
        prog = [
            "v_mov_b32 v0, 0x00000000",
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 4",
            "v_ashrrev_i64 v[0:1], v2, v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ashrrev_i64_struct", prog)
        self.assertIn("OpShiftRightArithmetic %ulong", spv)
        self.assertIn("OpUConvert %ulong", spv)
        words = support.mrt0_words(spv)
        res64 = (words[1] << 32) | words[0]
        self.assertEqual(res64, 0xF800000000000000)


if __name__ == "__main__":
    unittest.main()
