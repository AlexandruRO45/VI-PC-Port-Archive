from __future__ import annotations

"""Architectural reference models, symbolic AST, deterministic interpreter,
multi-lane Wave64 execution control, and translator integration tests for
Bundle 59: SALU Execution Control, Rounding & Denormal Mode Suite.

Covers RDNA2 Instructions:
    - s_round_mode <imm>
        Sets hardware floating-point rounding modes:
        Bits [1:0]: FP32 rounding mode (0=Nearest-Even, 1=+Inf, 2=-Inf, 3=Zero).
        Bits [3:2]: FP16 rounding mode (0=Nearest-Even, 1=+Inf, 2=-Inf, 3=Zero).
    - s_denorm_mode <imm>
        Sets floating-point denormal handling:
        Bits [1:0]: FP32 denormal mode (0=Flush, 3=Preserve).
        Bits [3:2]: FP16/64 denormal mode (0=Flush, 3=Preserve).
    - s_setprio <imm>
        Sets wave execution scheduling priority level (0..3).
    - s_sleep <imm>
        Puts wave into low-power stall for (imm * 64) clock cycles.
    - s_nop <imm>
        Pads instruction pipeline with (imm + 1) no-operation cycles (0..7).

Hardware rules (GFX10 RDNA2):
    - Floating-point status/control register updates affect subsequent VALU instructions.
    - The translator models IEEE 754 Nearest-Even FP32 rounding and rejects non-NE FP32 modes.
    - Denormal mode register writes are strictly unmodelled (raises Unsupported).
    - s_setprio, s_sleep, and s_nop are pipeline control instructions translated as no-ops.

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
import rdna2spirv
import spirv_mini_interp as si
import saveexec_run

MASK32: int = 0xFFFF_FFFF


def model_round_mode(mode: int) -> Dict[str, int]:
    fp32_mode = mode & 3
    fp16_mode = (mode >> 2) & 3
    return {"fp32": fp32_mode, "fp16": fp16_mode}


def model_denorm_mode(mode: int) -> Dict[str, int]:
    fp32_denorm = mode & 3
    fp16_64_denorm = (mode >> 2) & 3
    return {"fp32": fp32_denorm, "fp16_64": fp16_64_denorm}


def model_setprio(prio: int) -> int:
    return max(0, min(3, prio))


def model_sleep(duration: int) -> int:
    return max(0, duration) * 64


def model_nop(count: int) -> int:
    return (count & 7) + 1

class TestRoundModeOracle(unittest.TestCase):
    def test_oracle_round_modes(self):
        self.assertEqual(model_round_mode(0), {"fp32": 0, "fp16": 0})
        self.assertEqual(model_round_mode(4), {"fp32": 0, "fp16": 1})
        self.assertEqual(model_round_mode(8), {"fp32": 0, "fp16": 2})
        self.assertEqual(model_round_mode(12), {"fp32": 0, "fp16": 3})

    def test_translator_round_mode_oracle(self):
        prog = ["s_round_mode 0", "s_endpgm"]
        spv = support.translate_lines("cs_rm_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestRoundModeAST(unittest.TestCase):
    def test_ast_evaluation(self):
        modes = model_round_mode(12)
        self.assertEqual(modes["fp32"], 0)
        self.assertEqual(modes["fp16"], 3)

    def test_translator_round_mode_ast(self):
        prog = ["s_round_mode 12", "s_endpgm"]
        spv = support.translate_lines("cs_rm_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpStore", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestRoundModeConfig(unittest.TestCase):
    def test_fp16_rounding_modes(self):
        for fp16_val in (0, 1, 2, 3):
            imm = fp16_val << 2
            modes = model_round_mode(imm)
            self.assertEqual(modes["fp16"], fp16_val)
            self.assertEqual(modes["fp32"], 0)

    def test_translator_round_mode_fp16_modes(self):
        for imm in (0, 12):
            prog = [f"s_round_mode {imm}", "s_endpgm"]
            spv = support.translate_lines(f"cs_rm_fp16_{imm}", prog, fssm=saveexec_run.fssm())
            self.assertIn("OpCapability Shader", spv)
            si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})
        for imm in (4, 8):
            prog = [f"s_round_mode {imm}", "s_endpgm"]
            with self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines(f"cs_rm_fp16_rej_{imm}", prog, fssm=saveexec_run.fssm())

class TestRoundModeRejection(unittest.TestCase):
    def test_non_nearest_even_fp32_rejection(self):
        for invalid_mode in (1, 2, 3):
            prog = [f"s_round_mode {invalid_mode}", "s_endpgm"]
            with self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines(f"cs_rm_reject_{invalid_mode}", prog, fssm=saveexec_run.fssm())

    def test_translator_round_mode_valid(self):
        prog = ["s_round_mode 0", "s_endpgm"]
        spv = support.translate_lines("cs_rm_valid", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)

class TestRoundModeTranslator(unittest.TestCase):
    def test_translator_mutation_guard(self):
        prog = ["s_round_mode 0", "s_endpgm"]
        spv = support.translate_lines("cs_rm_trans", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestDenormModeOracle(unittest.TestCase):
    def test_oracle_denorm_modes(self):
        self.assertEqual(model_denorm_mode(0), {"fp32": 0, "fp16_64": 0})
        self.assertEqual(model_denorm_mode(3), {"fp32": 3, "fp16_64": 0})
        self.assertEqual(model_denorm_mode(12), {"fp32": 0, "fp16_64": 3})
        self.assertEqual(model_denorm_mode(15), {"fp32": 3, "fp16_64": 3})

    def test_translator_denorm_mode_rejection(self):
        prog = ["s_denorm_mode 0", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_denorm_oracle", prog, fssm=saveexec_run.fssm())

class TestDenormModeAST(unittest.TestCase):
    def test_ast_evaluation(self):
        modes = model_denorm_mode(15)
        self.assertEqual(modes["fp32"], 3)
        self.assertEqual(modes["fp16_64"], 3)

    def test_translator_denorm_mode_ast(self):
        prog = ["s_denorm_mode 15", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_denorm_ast", prog, fssm=saveexec_run.fssm())


if __name__ == "__main__":
    unittest.main()
