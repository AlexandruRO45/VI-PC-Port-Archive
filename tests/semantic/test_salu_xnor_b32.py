from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 s_xnor_b32.

AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP2 instruction:
  s_xnor_b32 sdst, ssrc0, ssrc1

Operation:
  sdst = ~(ssrc0 ^ ssrc1) & 0xFFFFFFFF
  scc  = 1 if sdst != 0 else 0

Architectural Characteristics:
  - Operates on 32-bit scalar general-purpose registers (SGPRs), inline constants,
    or literal constants.
  - Bitwise equivalence (NOT-XOR): Each bit in destination is set to 1 if the
    corresponding bits in ssrc0 and ssrc1 are equal, and 0 if they differ.
  - Scalar Condition Code (SCC): Reflects whether the resulting 32-bit dword is non-zero.
  - Scalar execution: Unlike vector instructions, SALU instructions execute uniformly
    once per wave across the scalar register file, unaffected by the vector EXEC mask.
  - Aliasing: Source and destination registers may alias arbitrarily (e.g. sdst == ssrc0).
  - Pure 32-bit bitwise operation: Preserves exact bit patterns of IEEE 754 representations
    without denormal flushing, quiet NaN canonicalization, or sign-zero collapsing.

Standard library Python only; zero third-party dependencies; zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32: int = 0xFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 0xFFFFFFFF]."""
    return val & MASK32


def ref_s_xnor_b32(src0: int, src1: int) -> Tuple[int, int]:
    """Pure mathematical and architectural reference oracle for RDNA2 s_xnor_b32.

    Returns:
        (result_u32, scc) where:
          result_u32 = ~(src0 ^ src1) & 0xFFFFFFFF
          scc = 1 if result_u32 != 0 else 0
    """
    s0 = u32(src0)
    s1 = u32(src1)
    res = u32(~(s0 ^ s1))
    scc = 1 if res != 0 else 0
    return res, scc


def run_translator_salu_xnor(
    src0: int,
    src1: int,
    name: str = "ps_salu_xnor",
) -> Tuple[int, int]:
    """Translate and run a minimal shader verifying s_xnor_b32 and SCC output.

    Uses s_cselect_b32 to inspect SCC flag:
      s_cselect_b32 s3, 1, 0
    Selects 1 if SCC != 0, else 0.
    Returns (result_dword, scc_code).
    """
    lines = [
        f"s_mov_b32 s0, {u32(src0):#x}",
        f"s_mov_b32 s1, {u32(src1):#x}",
        "s_xnor_b32 s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v0, s2",
        "v_mov_b32 v1, s3",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_truth_table_exhaustive(self) -> None:
        """Verify 1-bit XNOR logic: (0^0)->1, (0^1)->0, (1^0)->0, (1^1)->1."""
        self.assertEqual(ref_s_xnor_b32(0, 0), (0xFFFFFFFF, 1))
        self.assertEqual(ref_s_xnor_b32(0, 1), (0xFFFFFFFE, 1))
        self.assertEqual(ref_s_xnor_b32(1, 0), (0xFFFFFFFE, 1))
        self.assertEqual(ref_s_xnor_b32(1, 1), (0xFFFFFFFF, 1))

    def test_bitwise_identities(self) -> None:
        """Verify fundamental boolean algebra properties of XNOR."""
        vals = [0x00000000, 0xFFFFFFFF, 0x12345678, 0xAAAAAAAA, 0x55555555, 0x80000000]
        for v in vals:
            # Identity 1: X XNOR X == 0xFFFFFFFF (reflexive equivalence)
            res_id, scc_id = ref_s_xnor_b32(v, v)
            self.assertEqual(res_id, 0xFFFFFFFF)
            self.assertEqual(scc_id, 1)

            # Identity 2: X XNOR ~X == 0x00000000 (complementarity)
            res_comp, scc_comp = ref_s_xnor_b32(v, ~v)
            self.assertEqual(res_comp, 0x00000000)
            self.assertEqual(scc_comp, 0)

            # Identity 3: X XNOR 0 == ~X (bitwise NOT)
            res_not, _ = ref_s_xnor_b32(v, 0)
            self.assertEqual(res_not, u32(~v))

            # Identity 4: X XNOR 0xFFFFFFFF == X (bitwise identity)
            res_same, _ = ref_s_xnor_b32(v, 0xFFFFFFFF)
            self.assertEqual(res_same, u32(v))

    def test_commutativity(self) -> None:
        """Verify commutativity: ssrc0 XNOR ssrc1 == ssrc1 XNOR ssrc0."""
        a = 0xA5A55A5A
        b = 0x12345678
        self.assertEqual(ref_s_xnor_b32(a, b), ref_s_xnor_b32(b, a))

    def test_oracle_matches_translator(self) -> None:
        """Cross-check reference oracle with translator output on basic vectors."""
        cases = [
            (0x12345678, 0x12345678),
            (0x55555555, 0xAAAAAAAA),
            (0x00000000, 0xFFFFFFFF),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for idx, (s0, s1) in enumerate(cases):
            expected_res, expected_scc = ref_s_xnor_b32(s0, s1)
            trans_res, trans_scc = run_translator_salu_xnor(s0, s1, name=f"ps_oracle_xnor_{idx}")
            self.assertEqual(trans_res, expected_res)
            self.assertEqual(trans_scc, expected_scc)

if __name__ == "__main__":
    unittest.main()
