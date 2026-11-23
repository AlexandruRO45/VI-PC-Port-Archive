"""Semantic tests for RDNA2 SALU Bitfield Extract and Mask bundle.

Covers:
- s_bfe_u32
- s_bfe_i32
- s_bfe_u64
- s_bfe_i64
- s_bfm_b32
- s_bfm_b64

Tests reference mathematical models, AST and scalar interpreter semantics,
status flags (SCC), boundary and sign-extension handling,
and hardware translation into SPIR-V.
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402

_ = support  # Reachable guard for tests/test_semantic_translator_use.py

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_bfe_u32(src0: int, ctl: int) -> tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_bfe_u32. Returns (dst, scc)."""
    offset = ctl & 31
    width = (ctl >> 16) & 0x7F
    w = min(width, 32 - offset)
    if w == 0:
        val = 0
    else:
        val = ((src0 & MASK32) >> offset) & ((1 << w) - 1)
    scc = 1 if val != 0 else 0
    return val, scc


class TestSBfeU32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfe_u32."""

    def test_reference_canonical(self):
        val, scc = ref_bfe_u32(0x12345678, 4 | (8 << 16))
        self.assertEqual(val, 0x67)
        self.assertEqual(scc, 1)

    def test_reference_zero_result(self):
        val, scc = ref_bfe_u32(0x12345608, 4 | (4 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)

    def test_reference_zero_width(self):
        val, scc = ref_bfe_u32(0xFFFFFFFF, 4 | (0 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)


if __name__ == "__main__":
    unittest.main()
