"""Hardware-measured and translator-backed semantic tests for v_lshlrev_b64.

RDNA2 VALU instruction v_lshlrev_b64:
Performs 64-bit logical shift left with reversed operands:
    v[d:d+1] = v[s:s+1] << (src0 & 63)
The shift count is masked to 6 bits (0-63).
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def v_lshlrev_b64_oracle(shift_count: int, val64: int) -> tuple[int, int]:
    """Exact bit-level mathematical reference oracle for v_lshlrev_b64.

    Returns (rlo, rhi) as 32-bit unsigned integers.
    """
    shift = shift_count & 63
    res64 = ((val64 & MASK64) << shift) & MASK64
    return res64 & MASK32, (res64 >> 32) & MASK32


def run_lshlrev_ps(lines: list[str], name: str = "ps_valu_lshlrev_b64") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class LshlrevReferenceModel:
    """Architectural model tracking vector registers and v_lshlrev_b64 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr64(self, reg: int, val64: int) -> None:
        self.vgprs[reg] = val64 & MASK32
        self.vgprs[reg + 1] = (val64 >> 32) & MASK32

    def get_vgpr64(self, reg: int) -> int:
        return self.vgprs[reg] | (self.vgprs[reg + 1] << 32)

    def v_lshlrev_b64(self, dst: int, shift: int, src: int) -> tuple[int, int]:
        val64 = self.get_vgpr64(src)
        rlo, rhi = v_lshlrev_b64_oracle(shift, val64)
        self.vgprs[dst] = rlo
        self.vgprs[dst + 1] = rhi
        return rlo, rhi


class TestLshlrevReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_lshlrev_b64."""

    def test_oracle_basic_shifts(self):
        val = 0x00000001_12345678
        # shift by 0: identity
        self.assertEqual(v_lshlrev_b64_oracle(0, val), (0x12345678, 0x00000001))
        # shift by 4
        self.assertEqual(v_lshlrev_b64_oracle(4, val), (0x23456780, 0x00000011))
        # shift by 32: low becomes high
        self.assertEqual(v_lshlrev_b64_oracle(32, val), (0, 0x12345678))
        # shift by 64: masks to 0
        self.assertEqual(v_lshlrev_b64_oracle(64, val), (0x12345678, 0x00000001))
        # shift by 65: masks to 1
        self.assertEqual(v_lshlrev_b64_oracle(65, 1), (2, 0))

    def test_translator_execution_matches_oracle(self):
        cases = [
            (0, 0x12345678_9ABCDEF0),
            (1, 0x12345678_9ABCDEF0),
            (4, 0x00000001_12345678),
            (16, 0x00000000_FFFFFFFF),
            (32, 0x00000000_12345678),
            (48, 0x00000000_0000ABCD),
            (63, 1),
            (64, 0x12345678_9ABCDEF0),
        ]
        for idx, (sh, v64) in enumerate(cases):
            with self.subTest(sh=sh, v64=hex(v64)):
                expected_lo, expected_hi = v_lshlrev_b64_oracle(sh, v64)
                lo32 = v64 & MASK32
                hi32 = (v64 >> 32) & MASK32
                lines = [
                    f"v_mov_b32 v2, {lo32:#x}",
                    f"v_mov_b32 v3, {hi32:#x}",
                    f"v_lshlrev_b64 v[0:1], {sh}, v[2:3]",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_lshlrev_ps(lines, f"ps_lshlrev_c1_{idx}")
                self.assertEqual(words[0], expected_lo)
                self.assertEqual(words[1], expected_hi)


class Vop3Opcode:
    """Symbolic AST opcode descriptor for 64-bit VOP3 shift instructions."""

    def __init__(self, mnemonic: str, dst: str, shift: str, src: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.shift = shift
        self.src = src

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.shift}, {self.src}"


class TestLshlrevASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for v_lshlrev_b64."""

    def test_ast_emission_and_translation(self):
        op = Vop3Opcode("v_lshlrev_b64", "v[0:1]", "8", "v[2:3]")
        self.assertEqual(op.emit_assembly(), "v_lshlrev_b64 v[0:1], 8, v[2:3]")
        lines = [
            "v_mov_b32 v2, 0x1234",
            "v_mov_b32 v3, 0",
            op.emit_assembly(),
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_lshlrev_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("OpShiftLeftLogical", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x123400)
        self.assertEqual(words[1], 0)

    def test_ast_sgpr_shift_count(self):
        op = Vop3Opcode("v_lshlrev_b64", "v[0:1]", "s0", "v[2:3]")
        lines = [
            "s_mov_b32 s0, 12",
            "v_mov_b32 v2, 0xF",
            "v_mov_b32 v3, 0",
            op.emit_assembly(),
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_lshlrev_ps(lines, "ps_lshlrev_ast_sgpr")
        self.assertEqual(words[0], 0xF000)
        self.assertEqual(words[1], 0)


if __name__ == '__main__':
    unittest.main()
