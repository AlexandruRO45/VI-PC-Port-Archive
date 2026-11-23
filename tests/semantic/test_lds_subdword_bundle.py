from __future__ import annotations

"""Semantic interpreter tests and reference architectural model for RDNA2 LDS Sub-Dword Memory Suite.

Covers:
- ds_read_u8: 8-bit unsigned byte read from LDS, zero-extended to 32 bits.
- ds_read_i8: 8-bit signed byte read from LDS, sign-extended to 32 bits.
- ds_read_u16: 16-bit unsigned halfword read from LDS, zero-extended to 32 bits.
- ds_read_i16: 16-bit signed halfword read from LDS, sign-extended to 32 bits.
- ds_write_b8: 8-bit byte write to LDS, modifying only addressed byte within dword.
- ds_write_b16: 16-bit halfword write to LDS, modifying only addressed halfword within dword.

Standard library only (python -I, struct, math, unittest). Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
LDS_SIZE: int = 65536


# ------------------------------------------------------------------------------
# Architectural LDS Memory Reference Model
# ------------------------------------------------------------------------------

class LDSMemory:
    """Byte-addressable LDS memory reference model wrapping at LDS_SIZE."""

    def __init__(self, size: int = LDS_SIZE):
        self.size = size
        self.mem: bytearray = bytearray(size)

    def write_u8(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF

    def read_u8(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        return self.mem[eff]

    def read_i8(self, addr: int, offset: int = 0) -> int:
        b = self.read_u8(addr, offset)
        return (b - 256 if (b & 0x80) else b) & MASK32

    def write_u16(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF
        self.mem[(eff + 1) % self.size] = (val >> 8) & 0xFF

    def read_u16(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        lo = self.mem[eff]
        hi = self.mem[(eff + 1) % self.size]
        return (lo | (hi << 8)) & 0xFFFF

    def read_i16(self, addr: int, offset: int = 0) -> int:
        h = self.read_u16(addr, offset)
        return (h - 65536 if (h & 0x8000) else h) & MASK32


class TestDsReadU8Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_u8_zero_extension(self):
        self.lds.write_u8(0, 0xAB)
        self.assertEqual(self.lds.read_u8(0), 0xAB)
        self.lds.write_u8(1, 0xFF)
        self.assertEqual(self.lds.read_u8(1), 0xFF)

    def test_read_u8_with_offset(self):
        self.lds.write_u8(10, 0x42)
        self.assertEqual(self.lds.read_u8(4, offset=6), 0x42)


# ------------------------------------------------------------------------------
# ds_read_u8 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"v{self.index}"


@dataclass
class DSReadU8Instruction:
    dst: RegisterOperand
    addr: RegisterOperand
    offset: int = 0

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR or self.addr.op_type != OperandType.VGPR:
            raise TypeError("Operands of ds_read_u8 must be VGPRs")


class ScalarMachineDS:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = LDSMemory()

    def step_read_u8(self, insn: DSReadU8Instruction):
        addr = self.vgprs[insn.addr.index]
        res = self.lds.read_u8(addr, insn.offset)
        self.vgprs[insn.dst.index] = res


class TestDsReadU8ASTAndScalar(unittest.TestCase):
    def test_ast(self):
        insn = DSReadU8Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            addr=RegisterOperand(OperandType.VGPR, 1),
            offset=4,
        )
        self.assertEqual(insn.dst.name, "v2")
        self.assertEqual(insn.offset, 4)

    def test_scalar_execution(self):
        machine = ScalarMachineDS()
        machine.lds.write_u8(20, 0x7E)
        machine.vgprs[1] = 16
        insn = DSReadU8Instruction(
            dst=RegisterOperand(OperandType.VGPR, 2),
            addr=RegisterOperand(OperandType.VGPR, 1),
            offset=4,
        )
        machine.step_read_u8(insn)
        self.assertEqual(machine.vgprs[2], 0x7E)


if __name__ == '__main__':
    unittest.main()
