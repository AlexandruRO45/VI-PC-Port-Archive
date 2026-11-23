from __future__ import annotations

"""Architectural reference models, symbolic AST, interpreter, SIMD divergence,
and translator integration tests for SALU SaveExec 64-bit Mask Control Suite.

Covers RDNA2 SOP1 Instructions:
    - s_and_saveexec_b64   D, S0 : D = EXEC; EXEC = S0 & EXEC;   SCC = (EXEC != 0)
    - s_or_saveexec_b64    D, S0 : D = EXEC; EXEC = S0 | EXEC;   SCC = (EXEC != 0)
    - s_xor_saveexec_b64   D, S0 : D = EXEC; EXEC = S0 ^ EXEC;   SCC = (EXEC != 0)
    - s_andn2_saveexec_b64 D, S0 : D = EXEC; EXEC = S0 & ~EXEC;  SCC = (EXEC != 0)
    - s_orn2_saveexec_b64  D, S0 : D = EXEC; EXEC = S0 | ~EXEC;  SCC = (EXEC != 0)
    - s_nand_saveexec_b64  D, S0 : D = EXEC; EXEC = ~(S0 & EXEC); SCC = (EXEC != 0)
    - s_nor_saveexec_b64   D, S0 : D = EXEC; EXEC = ~(S0 | EXEC); SCC = (EXEC != 0)
    - s_xnor_saveexec_b64  D, S0 : D = EXEC; EXEC = ~(S0 ^ EXEC); SCC = (EXEC != 0)

Hardware rules (GFX10 RDNA2):
    - SOP1 encoding with 64-bit SGPR pair destination D and source S0.
    - Previous 64-bit EXEC register value saved into destination D.
    - Updated EXEC reflects bitwise operation between S0 and previous EXEC.
    - Scalar Condition Code (SCC) set to 1 if new EXEC != 0, else 0.
    - In-place operand preservation and register aliasing (D == S0) supported.
    - Multi-lane SIMD divergence on Wave64 execution masks.

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

MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK32: int = 0xFFFF_FFFF

ISA_OPS = {
    "and": lambda s, e: s & e,
    "or": lambda s, e: s | e,
    "xor": lambda s, e: s ^ e,
    "andn2": lambda s, e: s & ~e,
    "orn2": lambda s, e: s | ~e,
    "nand": lambda s, e: ~(s & e),
    "nor": lambda s, e: ~(s | e),
    "xnor": lambda s, e: ~(s ^ e),
}


def model_saveexec(op: str, s0: int, old_exec: int) -> Tuple[int, int, int]:
    new_exec = ISA_OPS[op](s0 & MASK64, old_exec & MASK64) & MASK64
    d_val = old_exec & MASK64
    scc = 1 if new_exec != 0 else 0
    return new_exec, d_val, scc


class SaveExecOp(Enum):
    AND = "and"
    OR = "or"
    XOR = "xor"
    ANDN2 = "andn2"
    ORN2 = "orn2"
    NAND = "nand"
    NOR = "nor"
    XNOR = "xnor"


@dataclass(frozen=True)
class Reg64:
    lo: int
    hi: int

    def __post_init__(self):
        if self.hi != self.lo + 1:
            raise ValueError(f"Invalid SGPR pair s[{self.lo}:{self.hi}]")


@dataclass
class SaveExecAST:
    op: SaveExecOp
    dest: Reg64
    src0: Union[Reg64, int, str]


@dataclass
class SimState:
    sgprs: Dict[int, int] = field(default_factory=dict)
    exec_mask: int = MASK64
    scc: int = 0

    def read64(self, reg: Reg64) -> int:
        lo = self.sgprs.get(reg.lo, 0) & MASK32
        hi = self.sgprs.get(reg.hi, 0) & MASK32
        return (hi << 32) | lo

    def write64(self, reg: Reg64, val: int) -> None:
        self.sgprs[reg.lo] = val & MASK32
        self.sgprs[reg.hi] = (val >> 32) & MASK32

    def step(self, ast: SaveExecAST) -> None:
        if isinstance(ast.src0, Reg64):
            s0_val = self.read64(ast.src0)
        elif isinstance(ast.src0, str) and ast.src0 == "exec":
            s0_val = self.exec_mask
        elif isinstance(ast.src0, int):
            s0_val = ast.src0 & MASK64
        else:
            raise TypeError(f"Unsupported operand {ast.src0}")

        old_exec = self.exec_mask
        new_exec, d_val, scc = model_saveexec(ast.op.value, s0_val, old_exec)
        self.write64(ast.dest, d_val)
        self.exec_mask = new_exec
        self.scc = scc

class TestSaveExecOracle_and(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("and", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["and"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("and", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_and(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.AND, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("and", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.AND, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("and", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_and(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("and", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("and", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_and(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_and_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_and_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "and")

class TestSaveExecOracle_or(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("or", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["or"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("or", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_or(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.OR, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("or", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.OR, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("or", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_or(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("or", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("or", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_or(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_or_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_or_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "or")

class TestSaveExecOracle_xor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("xor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["xor"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("xor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_xor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.XOR, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("xor", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.XOR, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("xor", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_xor(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("xor", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("xor", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_xor(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_xor_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_xor_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "xor")

class TestSaveExecOracle_andn2(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("andn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["andn2"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("andn2", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_andn2(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.ANDN2, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("andn2", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.ANDN2, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("andn2", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_andn2(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("andn2", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("andn2", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_andn2(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_andn2_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_andn2_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "andn2")

class TestSaveExecOracle_orn2(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("orn2", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["orn2"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("orn2", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_orn2(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.ORN2, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("orn2", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.ORN2, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("orn2", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_orn2(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("orn2", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("orn2", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_orn2(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_orn2_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_orn2_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "orn2")

class TestSaveExecOracle_nand(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("nand", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["nand"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("nand", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_nand(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.NAND, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("nand", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.NAND, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("nand", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_nand(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("nand", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("nand", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_nand(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_nand_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_nand_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "nand")

class TestSaveExecOracle_nor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("nor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["nor"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("nor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_nor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.NOR, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("nor", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.NOR, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("nor", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

class TestSaveExecDivergenceFlags_nor(unittest.TestCase):
    def test_simd_lane_divergence(self):
        # Wave64 low half (lanes 0..31) vs high half (lanes 32..63)
        low_lanes = 0x0000_0000_FFFF_FFFF
        high_lanes = 0xFFFF_FFFF_0000_0000
        new_exec, d_val, scc = model_saveexec("nor", low_lanes, high_lanes)
        self.assertEqual(d_val, high_lanes)
        self.assertEqual(scc, int(new_exec != 0))
        self.assertEqual(bin(d_val).count("1"), 32)

    def test_walking_bit_divergence(self):
        for lane in (0, 1, 31, 32, 62, 63):
            old_exec = 1 << lane
            s0 = MASK64
            new_exec, d_val, scc = model_saveexec("nor", s0, old_exec)
            self.assertEqual(d_val, old_exec)
            self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecTranslator_nor(unittest.TestCase):
    def test_translator_spirv_emission(self):
        prog = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 exec_lo, 0x12345678",
            "s_mov_b32 exec_hi, 0x9ABCDEF0",
            "s_mov_b32 s10, 0x55555555",
            "s_mov_b32 s11, 0xAAAAAAAA",
            "s_nor_saveexec_b64 s[6:7], s[10:11]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_saveexec_nor_bundle", prog, fssm=saveexec_run.fssm())
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)

    def test_translator_hardware_runner(self):
        saveexec_run.check(self, "nor")

class TestSaveExecOracle_xnor(unittest.TestCase):
    def test_oracle_mathematical_properties(self):
        old_exec = 0xF0F0_F0F0_0F0F_0F0F
        s0 = 0x5555_5555_AAAA_AAAA
        new_exec, d_val, scc = model_saveexec("xnor", s0, old_exec)
        self.assertEqual(d_val, old_exec)
        self.assertEqual(scc, 1 if new_exec != 0 else 0)
        self.assertEqual(new_exec, ISA_OPS["xnor"](s0, old_exec) & MASK64)

    def test_oracle_edge_patterns(self):
        for old_exec in (0, MASK64, 0x8000_0000_0000_0000, 1):
            for s0 in (0, MASK64, 0x8000_0000_0000_0000, 1):
                new_exec, d_val, scc = model_saveexec("xnor", s0, old_exec)
                self.assertEqual(d_val, old_exec)
                self.assertEqual(scc, int(new_exec != 0))

class TestSaveExecASTInterpreter_xnor(unittest.TestCase):
    def test_ast_construction_and_step(self):
        sim = SimState()
        sim.exec_mask = 0x1234_5678_9ABC_DEF0
        sim.write64(Reg64(10, 11), 0x0000_FFFF_FFFF_0000)
        ast = SaveExecAST(SaveExecOp.XNOR, Reg64(6, 7), Reg64(10, 11))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0x1234_5678_9ABC_DEF0)
        expected_exec, _, expected_scc = model_saveexec("xnor", 0x0000_FFFF_FFFF_0000, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)

    def test_ast_register_aliasing(self):
        sim = SimState()
        sim.exec_mask = 0xAAAA_5555_AAAA_5555
        sim.write64(Reg64(6, 7), 0x1111_2222_3333_4444)
        ast = SaveExecAST(SaveExecOp.XNOR, Reg64(6, 7), Reg64(6, 7))
        sim.step(ast)
        self.assertEqual(sim.read64(Reg64(6, 7)), 0xAAAA_5555_AAAA_5555)
        expected_exec, _, expected_scc = model_saveexec("xnor", 0x1111_2222_3333_4444, 0xAAAA_5555_AAAA_5555)
        self.assertEqual(sim.exec_mask, expected_exec)
        self.assertEqual(sim.scc, expected_scc)


if __name__ == "__main__":
    unittest.main()
