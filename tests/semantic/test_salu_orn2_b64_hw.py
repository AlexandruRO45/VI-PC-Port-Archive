from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_orn2_b64.

RDNA2 Instruction Set Architecture, SOP2:
    s_orn2_b64 D, S0, S1

Semantic operation:
    D.u64 = S0.u64 | (~S1.u64 & 0xFFFFFFFFFFFFFFFF)
    SCC = (D != 0)

Outputs:
    D: 64-bit destination SGPR pair (or EXEC/VCC) storing the bitwise OR of S0 and bitwise-NOT of S1.
    SCC: 1 if destination is non-zero, 0 if all zeros.

Standard library Python only (unittest, struct, math, dataclasses, enum, typing).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
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

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_orn2_b64_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_orn2_b64.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 | (~s1 & MASK64)) & MASK64
            scc: 1 if d != 0 else 0
    """
    s0_u64 = s0 & MASK64
    s1_u64 = s1 & MASK64
    d = (s0_u64 | (~s1_u64 & MASK64)) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


class SOrn2B64Oracle:
    """Architectural model and hardware-rule validator for s_orn2_b64."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_orn2_b64_oracle(s0, s1)

    @staticmethod
    def build_compute_shader(s0: int, s1: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_orn2_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_orn2_b64_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"s_mov_b32 s2, {s0 & MASK32:#x}",
            f"s_mov_b32 s3, {(s0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1 & MASK32:#x}",
            f"s_mov_b32 s5, {(s1 >> 32) & MASK32:#x}",
            "s_orn2_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SOrn2B64Oracle.build_compute_shader(s0, s1)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSOrn2B64Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        # Truth table: (0,0)->1, (0,1)->0, (1,0)->1, (1,1)->1
        cases = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 1),
            (1, 1, 1),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SOrn2B64Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)

    def test_oracle_identical_operands_produce_all_ones(self):
        # x | ~x = all ones
        for val in (0, ALL64, 0x1234_5678_9ABC_DEF0, 0x5555_5555_5555_5555):
            with self.subTest(val=hex(val)):
                d, scc = SOrn2B64Oracle.eval(val, val)
                self.assertEqual(d, ALL64)
                self.assertEqual(scc, 1)

    def test_oracle_zero_result(self):
        # S0 = 0 and S1 = ALL64 gives 0 | ~ALL64 = 0
        d, scc = SOrn2B64Oracle.eval(0, ALL64)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

    def test_oracle_s1_zero_produces_all_ones(self):
        # S0 | ~0 = S0 | ALL64 = ALL64
        d, scc = SOrn2B64Oracle.eval(0x1111_2222_3333_4444, 0)
        self.assertEqual(d, ALL64)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SOrn2B64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegisterKind(Enum):
    SGPR = auto()
    VGPR = auto()
    EXEC = auto()
    VCC = auto()
    SCC = auto()
    LITERAL64 = auto()


@dataclass(frozen=True)
class RegOperand:
    kind: RegisterKind
    index: int = 0
    size_bits: int = 64

    def format(self) -> str:
        if self.kind == RegisterKind.SGPR:
            if self.size_bits == 64:
                return f"s[{self.index}:{self.index + 1}]"
            return f"s{self.index}"
        elif self.kind == RegisterKind.VGPR:
            if self.size_bits == 64:
                return f"v[{self.index}:{self.index + 1}]"
            return f"v{self.index}"
        elif self.kind == RegisterKind.EXEC:
            return "exec" if self.size_bits == 64 else "exec_lo"
        elif self.kind == RegisterKind.VCC:
            return "vcc" if self.size_bits == 64 else "vcc_lo"
        elif self.kind == RegisterKind.SCC:
            return "scc"
        elif self.kind == RegisterKind.LITERAL64:
            return f"{self.index:#x}"
        return f"reg_{self.index}"


@dataclass
class SOrn2B64Instruction:
    dst: RegOperand
    src0: RegOperand
    src1: RegOperand
    opcode: str = "s_orn2_b64"

    def validate(self) -> None:
        if self.opcode != "s_orn2_b64":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind not in (RegisterKind.SGPR, RegisterKind.EXEC, RegisterKind.VCC):
            raise TypeError(f"Invalid destination register kind: {self.dst.kind}")
        if self.dst.kind == RegisterKind.SGPR:
            if self.dst.size_bits != 64 or (self.dst.index % 2 != 0):
                raise ValueError(f"Destination SGPR must be an even-aligned pair, got s[{self.dst.index}:{self.dst.index+1}]")
            if self.dst.index > 102:
                raise ValueError(f"Destination SGPR index out of range: s{self.dst.index}")
        for src, name in ((self.src0, "src0"), (self.src1, "src1")):
            if src.kind == RegisterKind.SGPR:
                if src.size_bits != 64 or (src.index % 2 != 0):
                    raise ValueError(f"{name} SGPR must be an even-aligned pair, got s[{src.index}:{src.index+1}]")
                if src.index > 104:
                    raise ValueError(f"{name} SGPR index out of range: s{src.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}, {self.src1.format()}"


class TestSOrn2B64AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_operands(self):
        instr = SOrn2B64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_orn2_b64 s[6:7], s[2:3], s[4:5]")

    def test_ast_valid_exec_destination(self):
        instr = SOrn2B64Instruction(
            dst=RegOperand(RegisterKind.EXEC, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64),
            src1=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_orn2_b64 exec, exec, vcc")

    def test_ast_misaligned_destination(self):
        instr = SOrn2B64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=5, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_vgpr_destination(self):
        instr = SOrn2B64Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(TypeError):
            instr.validate()

if __name__ == "__main__":
    unittest.main()
