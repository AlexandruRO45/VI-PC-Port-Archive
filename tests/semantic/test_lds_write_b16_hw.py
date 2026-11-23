from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_write_b16.

Instruction definition:
    ds_write_b16 vaddr, vdata [offset:offset16]

Semantics:
    Local Data Share (LDS) 16-bit halfword write.
    Writes the lower 16 bits of the VGPR `vdata` into LDS storage at the byte address:
        addr = (vaddr + offset16) & 0xFFFF
    Only the addressed 2 bytes are modified; neighboring bytes and remaining bits
    of existing dwords remain untouched.

Standard library only: struct, math, unittest. Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
LDS_SIZE_BYTES: int = 65536

# IEEE 754 Binary16 Bit Patterns
POS_ZERO: int = 0x0000
NEG_ZERO: int = 0x8000
ONE: int = 0x3C00
TWO: int = 0x4000
THREE: int = 0x4200
HALF: int = 0x3800
NEG_ONE: int = 0xBC00
POS_INF: int = 0x7C00
NEG_INF: int = 0xFC00
QNAN: int = 0x7E00
SNAN: int = 0x7C01
MIN_NORMAL: int = 0x0400
MAX_NORMAL: int = 0x7BFF
MIN_SUB: int = 0x0001
MAX_SUB: int = 0x03FF


# ==============================================================================
# Reference Mathematical Oracle and Architectural Storage
# ==============================================================================

@dataclass
class LDSMemory:
    """Byte-addressable LDS storage."""
    data: bytearray = field(default_factory=lambda: bytearray(LDS_SIZE_BYTES))

    def write_u16(self, addr: int, val16: int) -> None:
        addr &= 0xFFFF
        struct.pack_into("<H", self.data, addr, val16 & MASK16)

    def read_u16(self, addr: int) -> int:
        addr &= 0xFFFF
        return struct.unpack_from("<H", self.data, addr)[0]

    def write_u32(self, addr: int, val32: int) -> None:
        addr &= 0xFFFF
        struct.pack_into("<I", self.data, addr, val32 & MASK32)

    def read_u32(self, addr: int) -> int:
        addr &= 0xFFFF
        return struct.unpack_from("<I", self.data, addr)[0]


def ref_ds_write_b16(lds: LDSMemory, base_addr: int, data: int, offset: int = 0) -> None:
    """Architectural reference oracle for ds_write_b16."""
    eff_addr = (base_addr + offset) & 0xFFFF
    lds.write_u16(eff_addr, data & MASK16)


class TestDSWriteB16ReferenceOracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_aligned_low_half_write(self):
        # Initial dword at 0x40: 0x12345678
        self.lds.write_u32(0x40, 0x1234_5678)
        # Write 0xABCD to low half
        ref_ds_write_b16(self.lds, base_addr=0x40, data=0xABCD, offset=0)
        self.assertEqual(self.lds.read_u32(0x40), 0x1234_ABCD)

    def test_aligned_high_half_write_with_offset(self):
        self.lds.write_u32(0x40, 0x1234_5678)
        # Write 0xABCD to high half (offset: 2)
        ref_ds_write_b16(self.lds, base_addr=0x40, data=0xABCD, offset=2)
        self.assertEqual(self.lds.read_u32(0x40), 0xABCD_5678)

    def test_cross_dword_unaligned_write(self):
        # Dword at 0x40: 0x11111111, Dword at 0x44: 0x22222222
        self.lds.write_u32(0x40, 0x1111_1111)
        self.lds.write_u32(0x44, 0x2222_2222)
        # Write 0xBBAA at byte 0x43 (crosses 0x43 in dword 0 and 0x44 in dword 1)
        ref_ds_write_b16(self.lds, base_addr=0x43, data=0xBBAA, offset=0)
        self.assertEqual(self.lds.read_u32(0x40), 0xAA11_1111)
        self.assertEqual(self.lds.read_u32(0x44), 0x2222_22BB)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandType(Enum):
    VGPR = auto()


@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self):
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    @property
    def name(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class DSWriteB16Instruction:
    vaddr: VGPR
    vdata: VGPR
    offset: int = 0

    def __post_init__(self):
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"LDS offset out of range [0, 65535]: {self.offset}")


class TestDSWriteB16AST(unittest.TestCase):
    def test_valid_ast_construction(self):
        inst = DSWriteB16Instruction(vaddr=VGPR(1), vdata=VGPR(2), offset=4)
        self.assertEqual(inst.vaddr.name, "v1")
        self.assertEqual(inst.vdata.name, "v2")
        self.assertEqual(inst.offset, 4)

    def test_invalid_vgpr_index(self):
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_invalid_offset(self):
        with self.assertRaises(ValueError):
            DSWriteB16Instruction(vaddr=VGPR(0), vdata=VGPR(1), offset=-1)
        with self.assertRaises(ValueError):
            DSWriteB16Instruction(vaddr=VGPR(0), vdata=VGPR(1), offset=65536)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    def __init__(self, lds: Optional[LDSMemory] = None):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()


class ScalarDSInterpreter:
    def execute(self, state: ScalarThreadState, inst: DSWriteB16Instruction) -> None:
        vaddr = state.vgprs[inst.vaddr.index] & MASK32
        vdata = state.vgprs[inst.vdata.index] & MASK32
        ref_ds_write_b16(state.lds, vaddr, vdata, inst.offset)


class TestDSWriteB16ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.state = ScalarThreadState()
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_step(self):
        self.state.lds.write_u32(0x100, 0x0000_0000)
        self.state.vgprs[1] = 0x100
        self.state.vgprs[2] = 0xBEEF
        inst = DSWriteB16Instruction(vaddr=VGPR(1), vdata=VGPR(2), offset=0)
        self.interp.execute(self.state, inst)
        self.assertEqual(self.state.lds.read_u16(0x100), 0xBEEF)

    def test_scalar_execution_with_offset(self):
        self.state.lds.write_u32(0x100, 0x0000_0000)
        self.state.vgprs[1] = 0x100
        self.state.vgprs[2] = 0xCAFE
        inst = DSWriteB16Instruction(vaddr=VGPR(1), vdata=VGPR(2), offset=2)
        self.interp.execute(self.state, inst)
        self.assertEqual(self.state.lds.read_u16(0x102), 0xCAFE)
        self.assertEqual(self.state.lds.read_u32(0x100), 0xCAFE_0000)


if __name__ == '__main__':
    unittest.main()
