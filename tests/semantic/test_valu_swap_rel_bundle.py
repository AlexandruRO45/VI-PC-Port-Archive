"""Translator-backed semantic execution tests for AMD RDNA2 register swap and relative addressing instructions.

Subsystem Feature Bundle 56 covers 6 RDNA2 register swap and relative addressing opcodes:
  1. v_swap_b32    - Exchanges two 32-bit VGPRs across active lanes
  2. v_swaprel_b32 - Exchanges VGPR[D + M0[25:16]] and VGPR[S0 + M0[9:0]] under active lanes
  3. s_swappc_b64  - Driver subroutine linkage in VS; rejected in PS and CS
  4. s_movrels_b32 - Relative SGPR read through M0: D = SGPR[(S0 + M0) % 128]
  5. s_movrels_b64 - Relative 64-bit SGPR read through M0: D = SGPR_pair[(S0 + M0) % 128]
  6. v_movrels_b32 - Relative VGPR read through M0: D = VGPR[S0 + M0]

Every test executes via support.translate_lines(...) and verifies computed register values
via support.mrt0_words, preventing unverified mock regressions.

Standard library Python only (unittest, struct, math). Zero game data.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF


def run_ps(name: str, body: list[str], export_regs: tuple[str, ...] = ("v0", "v1", "v0", "v1")) -> list[int]:
    """Helper to translate lines into a pixel shader, run it, and return exported MRT0 words."""
    regs_str = ", ".join(export_regs)
    lines = list(body) + [f"exp mrt0 {regs_str} done vm", "s_endpgm"]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)


class TestVSwapB32(unittest.TestCase):
    """Execution verification for v_swap_b32."""

    def test_swap_basic_pair(self):
        words = run_ps(
            "ps_swap_basic",
            [
                "v_mov_b32 v0, 100",
                "v_mov_b32 v1, 200",
                "v_swap_b32 v0, v1",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words, [200, 100, 200, 100])

    def test_swap_boundary_constants(self):
        pairs = [
            (0x00000000, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
            (0x80000000, 0x00000001),
            (0xCAFEBABE, 0xDEADBEEF),
        ]
        for idx, (a, b) in enumerate(pairs):
            with self.subTest(a=hex(a), b=hex(b)):
                words = run_ps(
                    f"ps_swap_pair_{idx}",
                    [
                        f"v_mov_b32 v0, {a:#x}",
                        f"v_mov_b32 v1, {b:#x}",
                        "v_swap_b32 v0, v1",
                    ],
                    export_regs=("v0", "v1", "v0", "v1"),
                )
                self.assertEqual(words, [b, a, b, a])

    def test_swap_identity_self(self):
        words = run_ps(
            "ps_swap_self",
            [
                "v_mov_b32 v0, 0xCAFE",
                "v_swap_b32 v0, v0",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words, [0xCAFE, 0xCAFE, 0xCAFE, 0xCAFE])

    def test_swap_double_involution(self):
        words = run_ps(
            "ps_swap_double",
            [
                "v_mov_b32 v0, 0x11223344",
                "v_mov_b32 v1, 0x55667788",
                "v_swap_b32 v0, v1",
                "v_swap_b32 v0, v1",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words, [0x11223344, 0x55667788, 0x11223344, 0x55667788])

    def test_swap_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_swap_bad", ["v_swap_b32"])


class TestVSwapRelB32(unittest.TestCase):
    """Execution verification for v_swaprel_b32."""

    def test_swaprel_zero_m0(self):
        words = run_ps(
            "ps_swaprel_zero",
            [
                "s_mov_b32 m0, 0",
                "v_mov_b32 v0, 10",
                "v_mov_b32 v1, 20",
                "v_swaprel_b32 v0, v1",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words, [20, 10, 20, 10])

    def test_swaprel_dual_m0_offsets(self):
        # M0[25:16] = 4 (dst offset), M0[9:0] = 4 (src offset)
        # With instruction v_swaprel_b32 v0, v4:
        # actual dst = v0 + 4 = v4
        # actual src = v4 + 4 = v8
        words = run_ps(
            "ps_swaprel_dual",
            [
                "s_mov_b32 m0, 0x00040004",
                "v_mov_b32 v4, 100",
                "v_mov_b32 v8, 200",
                "v_swaprel_b32 v0, v4",
            ],
            export_regs=("v4", "v8", "v4", "v8"),
        )
        self.assertEqual(words, [200, 100, 200, 100])

    def test_swaprel_split_indices(self):
        # M0[25:16] = 1, M0[9:0] = 2
        # v_swaprel_b32 v0, v0 exchanges v1 and v2
        words = run_ps(
            "ps_swaprel_split",
            [
                "s_mov_b32 m0, 0x00010002",
                "v_mov_b32 v1, 777",
                "v_mov_b32 v2, 888",
                "v_swaprel_b32 v0, v0",
            ],
            export_regs=("v1", "v2", "v1", "v2"),
        )
        self.assertEqual(words, [888, 777, 888, 777])

    def test_swaprel_self_identity(self):
        words = run_ps(
            "ps_swaprel_self",
            [
                "s_mov_b32 m0, 0",
                "v_mov_b32 v2, 555",
                "v_swaprel_b32 v2, v2",
            ],
            export_regs=("v2", "v2", "v2", "v2"),
        )
        self.assertEqual(words, [555, 555, 555, 555])

    def test_swaprel_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_swaprel_bad", ["v_swaprel_b32 s0, v1"])


class TestSSwappcB64(unittest.TestCase):
    """Rejection and syntax verification for s_swappc_b64."""

    def test_swappc_rejected_in_ps(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_swappc", ["s_swappc_b64 s[0:1], s[2:3]", "s_endpgm"])

    def test_swappc_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_swappc_bad", ["s_swappc_b64"])


class TestSMovRelSB32(unittest.TestCase):
    """Execution verification for s_movrels_b32."""

    def test_movrels32_zero_offset(self):
        words = run_ps(
            "ps_movrels32_zero",
            [
                "s_mov_b32 m0, 0",
                "s_mov_b32 s4, 42",
                "s_movrels_b32 s0, s4",
                "v_mov_b32 v0, s0",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words[0], 42)

    def test_movrels32_positive_offset(self):
        words = run_ps(
            "ps_movrels32_pos",
            [
                "s_mov_b32 m0, 5",
                "s_mov_b32 s15, 165",
                "s_movrels_b32 s2, s10",
                "v_mov_b32 v0, s2",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words[0], 165)

    def test_movrels32_modulo_128_wrapping(self):
        # Index = (127 + 1) & 127 = 0
        words = run_ps(
            "ps_movrels32_wrap",
            [
                "s_mov_b32 m0, 1",
                "s_mov_b32 s0, 999",
                "s_movrels_b32 s2, s127",
                "v_mov_b32 v0, s2",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words[0], 999)

    def test_movrels32_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_movrels32_bad", ["s_movrels_b32 v0, s1"])


class TestSMovRelSB64(unittest.TestCase):
    """Execution verification for s_movrels_b64."""

    def test_movrels64_zero_offset(self):
        words = run_ps(
            "ps_movrels64_zero",
            [
                "s_mov_b32 m0, 0",
                "s_mov_b32 s4, 100",
                "s_mov_b32 s5, 200",
                "s_movrels_b64 s[0:1], s[4:5]",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words[:2], [100, 200])

    def test_movrels64_positive_offset(self):
        words = run_ps(
            "ps_movrels64_pos",
            [
                "s_mov_b32 m0, 2",
                "s_mov_b32 s20, 0xAAAA",
                "s_mov_b32 s21, 0xBBBB",
                "s_movrels_b64 s[2:3], s[18:19]",
                "v_mov_b32 v0, s2",
                "v_mov_b32 v1, s3",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words[:2], [0xAAAA, 0xBBBB])

    def test_movrels64_wrapping_pair(self):
        # Reads pair starting at s127: (127 + 0) = s127, (127 + 1) % 128 = s0
        words = run_ps(
            "ps_movrels64_wrap",
            [
                "s_mov_b32 m0, 0",
                "s_mov_b32 s127, 0x111",
                "s_mov_b32 s0, 0x222",
                "s_movrels_b64 s[6:7], s[127:128]",
                "v_mov_b32 v0, s6",
                "v_mov_b32 v1, s7",
            ],
            export_regs=("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words[:2], [0x111, 0x222])

    def test_movrels64_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_movrels64_bad", ["s_movrels_b64 s0, s[2:3]"])


class TestVMovRelSB32(unittest.TestCase):
    """Execution verification for v_movrels_b32."""

    def test_v_movrels_zero_offset(self):
        words = run_ps(
            "ps_v_movrels_zero",
            [
                "s_mov_b32 m0, 0",
                "v_mov_b32 v1, 777",
                "v_movrels_b32 v0, v1",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words[0], 777)

    def test_v_movrels_positive_offset(self):
        words = run_ps(
            "ps_v_movrels_pos",
            [
                "s_mov_b32 m0, 5",
                "v_mov_b32 v25, 999",
                "v_movrels_b32 v2, v20",
            ],
            export_regs=("v2", "v2", "v2", "v2"),
        )
        self.assertEqual(words[0], 999)

    def test_v_movrels_dynamic_indexing(self):
        words = run_ps(
            "ps_v_movrels_dyn",
            [
                "s_mov_b32 m0, 10",
                "v_mov_b32 v20, 1234",
                "v_movrels_b32 v0, v10",
            ],
            export_regs=("v0", "v0", "v0", "v0"),
        )
        self.assertEqual(words[0], 1234)

    def test_v_movrels_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_v_movrels_bad", ["v_movrels_b32 s0, v1"])


if __name__ == "__main__":
    unittest.main()
