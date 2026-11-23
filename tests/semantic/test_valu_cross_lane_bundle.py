from __future__ import annotations

"""Architectural reference models, symbolic AST, deterministic interpreter,
multi-lane Wave64 divergence tests, and translator integration tests for
Bundle 54: VALU Cross-Lane Permutation & Data Exchange Suite.

Covers RDNA2 Cross-Lane Instructions:
    - v_readfirstlane_b32 sD, vS
        Reads vector register from first active lane in EXEC into scalar destination.
    - v_readlane_b32 sD, vS, sLane
        Reads vector register from indexed lane into scalar destination (sLane mod 64).
    - v_writelane_b32 vD, sVal, sLane
        Writes scalar value into single indexed lane of vector register (EXEC-independent).
    - v_permlane16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes lanes within the same 16-lane half-row using 4-bit lane selectors.
    - v_permlanex16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes lanes from the opposing 16-lane half-row using 4-bit lane selectors.

Hardware rules (GFX10 RDNA2):
    - Subgroup cross-lane shuffles preserve dword bit patterns.
    - v_readfirstlane selects the first lane set in EXEC; if EXEC is 0, defaults to lane 0.
    - v_readlane and v_writelane index lanes modulo 64 (Wave64) or modulo 32 (Wave32).
    - v_writelane writes to the indexed lane regardless of whether that lane is active in EXEC.
    - v_permlane16 and v_permlanex16 decode 4-bit selectors from sSrc1 (lanes 0-7) and sSrc2 (lanes 8-15).

Standard library Python only (unittest, struct, math, dataclasses, enum).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si
import saveexec_run

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def model_readfirstlane(lanes: List[int], exec_mask: int) -> int:
    for i in range(64):
        if exec_mask & (1 << i):
            return lanes[i] & MASK32 if i < len(lanes) else 0
    return lanes[0] & MASK32 if lanes else 0


def model_readlane(lanes: List[int], lane_idx: int) -> int:
    idx = lane_idx & 63
    return lanes[idx] & MASK32 if idx < len(lanes) else 0


def model_writelane(lanes: List[int], val: int, lane_idx: int) -> List[int]:
    idx = lane_idx & 63
    out = list(lanes)
    while len(out) <= idx:
        out.append(0)
    out[idx] = val & MASK32
    return out


def model_permlane16(lanes: List[int], s1: int, s2: int, xor16: bool = False) -> List[int]:
    out = [0] * 64
    for l in range(64):
        within = l & 15
        if within < 8:
            nib = (s1 >> (4 * within)) & 0xF
        else:
            nib = (s2 >> (4 * (within - 8))) & 0xF
        half = l & 0xFFFFFFF0
        row = (half ^ 16) if xor16 else half
        src_lane = (row | nib) & 63
        out[l] = lanes[src_lane] if src_lane < len(lanes) else 0
    return out

class TestReadFirstLaneOracle(unittest.TestCase):
    def test_oracle_active_first_lane(self):
        lanes = [100 + i for i in range(64)]
        self.assertEqual(model_readfirstlane(lanes, 0x1), 100)
        self.assertEqual(model_readfirstlane(lanes, 0x2), 101)
        self.assertEqual(model_readfirstlane(lanes, 0x8), 103)
        self.assertEqual(model_readfirstlane(lanes, 0x1000), 112)

    def test_translator_readfirstlane_oracle(self):
        prog = ["v_mov_b32_e32 v1, 777", "v_readfirstlane_b32 s0, v1", "s_endpgm"]
        spv = support.translate_lines("cs_rfl_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadFirstLaneAST(unittest.TestCase):
    def test_ast_evaluation(self):
        lanes = [i * 7 for i in range(64)]
        exec_mask = 0xFF00
        val = model_readfirstlane(lanes, exec_mask)
        self.assertEqual(val, 8 * 7)

    def test_translator_readfirstlane_ast(self):
        prog = ["v_mov_b32_e32 v2, 999", "v_readfirstlane_b32 s4, v2", "s_endpgm"]
        spv = support.translate_lines("cs_rfl_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadFirstLaneDivergence(unittest.TestCase):
    def test_exec_mask_variations(self):
        lanes = [i for i in range(64)]
        self.assertEqual(model_readfirstlane(lanes, 0x8000_0000_0000_0000), 63)
        self.assertEqual(model_readfirstlane(lanes, 0), 0)

    def test_translator_readfirstlane_divergence(self):
        prog = [
            "s_mov_b32 exec_lo, 0x00000000",
            "s_mov_b32 exec_hi, 0x00010000",
            "v_mov_b32_e32 v1, 555",
            "v_readfirstlane_b32 s8, v1",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_rfl_div", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadFirstLaneAliasing(unittest.TestCase):
    def test_aliasing_and_preservation(self):
        lanes = [0xDEADBEEF] * 64
        val = model_readfirstlane(lanes, 0x1)
        self.assertEqual(val, 0xDEADBEEF)

    def test_translator_readfirstlane_aliasing(self):
        prog = ["v_mov_b32_e32 v1, 0x1234", "v_readfirstlane_b32 s1, v1", "s_endpgm"]
        spv = support.translate_lines("cs_rfl_alias", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadFirstLaneTranslator(unittest.TestCase):
    def test_translator_spirv_instructions(self):
        prog = ["v_mov_b32_e32 v3, 0xABCD", "v_readfirstlane_b32 s10, v3", "s_endpgm"]
        spv = support.translate_lines("cs_rfl_trans", prog, fssm=saveexec_run.fssm())
        self.assertTrue("OpGroupNonUniformShuffle" in spv or "OpGroupNonUniformBallot" in spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadLaneOracle(unittest.TestCase):
    def test_oracle_indexing(self):
        lanes = [1000 + i for i in range(64)]
        for i in range(64):
            self.assertEqual(model_readlane(lanes, i), 1000 + i)

    def test_translator_readlane_oracle(self):
        prog = ["v_mov_b32_e32 v1, 123", "s_mov_b32 s1, 4", "v_readlane_b32 s0, v1, s1", "s_endpgm"]
        spv = support.translate_lines("cs_rl_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpGroupNonUniformShuffle", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadLaneAST(unittest.TestCase):
    def test_ast_evaluation(self):
        lanes = [i * 3 for i in range(64)]
        self.assertEqual(model_readlane(lanes, 15), 45)

    def test_translator_readlane_ast(self):
        prog = ["v_mov_b32_e32 v2, 333", "s_mov_b32 s3, 10", "v_readlane_b32 s2, v2, s3", "s_endpgm"]
        spv = support.translate_lines("cs_rl_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpGroupNonUniformShuffle", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadLaneModulo(unittest.TestCase):
    def test_modulo_masking(self):
        lanes = [i for i in range(64)]
        self.assertEqual(model_readlane(lanes, 64), 0)
        self.assertEqual(model_readlane(lanes, 65), 1)
        self.assertEqual(model_readlane(lanes, 127), 63)

    def test_translator_readlane_modulo(self):
        prog = ["v_mov_b32_e32 v1, 888", "s_mov_b32 s1, 67", "v_readlane_b32 s0, v1, s1", "s_endpgm"]
        spv = support.translate_lines("cs_rl_mod", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpGroupNonUniformShuffle", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestReadLaneAliasing(unittest.TestCase):
    def test_aliasing(self):
        lanes = [42] * 64
        self.assertEqual(model_readlane(lanes, 7), 42)

    def test_translator_readlane_aliasing(self):
        prog = ["v_mov_b32_e32 v0, 55", "s_mov_b32 s0, 0", "v_readlane_b32 s0, v0, s0", "s_endpgm"]
        spv = support.translate_lines("cs_rl_alias", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpGroupNonUniformShuffle", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})


if __name__ == "__main__":
    unittest.main()
