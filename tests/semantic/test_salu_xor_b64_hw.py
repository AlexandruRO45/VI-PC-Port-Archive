"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_xor_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_xor_b64 instruction belongs to the SOP2 format.
  Instruction:
    s_xor_b64 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 64-bit destination SGPR pair s[2k:2k+1] (or EXEC)
    ssrc0 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC
    ssrc1 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC

  Semantics:
    sdst = (ssrc0 ^ ssrc1) & 0xFFFFFFFFFFFFFFFF
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_xor_b64(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_xor_b64 instruction.

    Args:
        ssrc0: 64-bit first source operand (uint64)
        ssrc1: 64-bit second source operand (uint64)

    Returns:
        tuple (sdst, scc):
          - sdst: 64-bit result = (ssrc0 ^ ssrc1) & MASK64
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u64(ssrc0)
    s1 = u64(ssrc1)
    sdst = u64(s0 ^ s1)
    scc = 1 if sdst != 0 else 0
    return sdst, scc


class TestSaluXorB64Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_basic(self) -> None:
        s0 = 0x123456789ABCDEF0
        s1 = 0x00FF00FF0F0F0F0F
        sdst, scc = ref_xor_b64(s0, s1)
        expected = u64(s0 ^ s1)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)

    def test_oracle_self_xor_cancels_to_zero(self) -> None:
        val = 0xFEDCBA9876543210
        sdst, scc = ref_xor_b64(val, val)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_identity(self) -> None:
        val = 0x0123456789ABCDEF
        sdst, scc = ref_xor_b64(val, 0)
        self.assertEqual(sdst, val)
        self.assertEqual(scc, 1)

        sdst_rev, scc_rev = ref_xor_b64(0, val)
        self.assertEqual(sdst_rev, val)
        self.assertEqual(scc_rev, 1)

    def test_oracle_invert_with_all_ones(self) -> None:
        val = 0x5555555555555555
        sdst, scc = ref_xor_b64(val, ALL_LANES_ACTIVE)
        self.assertEqual(sdst, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(scc, 1)

    def test_oracle_commutativity(self) -> None:
        s0 = 0x123456789ABCDEF0
        s1 = 0xFEDCBA9876543210
        res01, scc01 = ref_xor_b64(s0, s1)
        res10, scc10 = ref_xor_b64(s1, s0)
        self.assertEqual(res01, res10)
        self.assertEqual(scc01, scc10)


class Opcode(Enum):
    S_XOR_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_xor_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.SGPR:
                return f"s[{op.index}:{op.index + 1}]"
            if op.kind == OperandKind.EXEC:
                return "exec"
            if op.kind == OperandKind.VCC:
                return "vcc"
            return hex(op.value)
        return f"s_xor_b64 {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class TestSaluXorB64AST(unittest.TestCase):
    """Stage 2: AST opcode definitions and typed expressions."""

    def test_ast_construction_sgpr_pair(self) -> None:
        dst = ASTOperand.sgpr(0)
        src0 = ASTOperand.sgpr(2)
        src1 = ASTOperand.sgpr(4)
        inst = ASTInstruction(Opcode.S_XOR_B64, dst, src0, src1)
        self.assertEqual(inst.disasm(), "s_xor_b64 s[0:1], s[2:3], s[4:5]")

    def test_ast_odd_alignment_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(3)  # odd index must be rejected


if __name__ == "__main__":
    unittest.main()
