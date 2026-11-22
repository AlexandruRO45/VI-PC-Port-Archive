from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_or_rtn_b64.

Opcode:
  - ds_or_rtn_b64 vdst, vaddr, vdata [offset:N]
      Atomic 64-bit bitwise OR on LDS memory with 64-bit destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte & (lds_bytes - 1)) & ~7
        old_val = LDS_qword[idx]
        LDS_qword[idx] = old_val | vdata
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in 32-bit VGPR vaddr + optional unsigned immediate byte offset.
  - Alignment: 64-bit access is 8-byte aligned (low 3 bits of byte address masked out).
  - Return value: Prior 64-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR pair v[dst:dst+1].
  - Flags: LDS operations do not alter VCC or SCC condition flags.
  - Untyped bitvector: Operates on raw 64-bit qwords without IEEE 754 interpretation or NaN quietening.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB LDS capacity in bytes
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001
MIN_NORMAL_F64 = 0x0010000000000000
MIN_SUBNORMAL_F64 = 0x0000000000000001
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MAX_NORMAL_F64 = 0x7FEFFFFFFFFFFFFF
ALL_ONES_F64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000000000000000) else val


class LDSMemory:
    """Model of Local Data Share memory storage with 64-bit qword access."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_bytes}")
        self.size_bytes = size_bytes
        self.mem: bytearray = bytearray(size_bytes)

    def read_qword(self, byte_addr: int) -> int:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        return struct.unpack_from("<Q", self.mem, idx)[0]

    def write_qword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        struct.pack_into("<Q", self.mem, idx, u64(val))


def ref_ds_or_rtn_b64(
    mem: LDSMemory,
    vaddr: int,
    vdata: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_or_rtn_b64.

    Atomically computes:
      effective_byte = (vaddr + offset) mod 2^32
      old_val = LDS[effective_byte]
      LDS[effective_byte] = old_val | vdata
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    old_val = mem.read_qword(effective_byte)
    new_val = u64(old_val | u64(vdata))
    mem.write_qword(effective_byte, new_val)
    return old_val


class TestLdsOrRtnB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_bytes=4096)

    def test_basic_or_and_return(self) -> None:
        # Pre-seed memory at byte 16 with 0x00FF00FF00FF00FF
        self.mem.write_qword(16, 0x00FF00FF00FF00FF)
        # vaddr = 16, OR with 0xFF00FF00FF00FF00
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=16, vdata=0xFF00FF00FF00FF00, offset=0)
        # Old value returned
        self.assertEqual(ret, 0x00FF00FF00FF00FF)
        # Memory updated to 0xFFFFFFFFFFFFFFFF
        self.assertEqual(self.mem.read_qword(16), 0xFFFFFFFFFFFFFFFF)

    def test_or_with_zero_preserves_memory_and_returns_old(self) -> None:
        self.mem.write_qword(32, 0x123456789ABCDEF0)
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=32, vdata=0, offset=0)
        self.assertEqual(ret, 0x123456789ABCDEF0)
        self.assertEqual(self.mem.read_qword(32), 0x123456789ABCDEF0)

    def test_or_with_all_ones_sets_all_bits(self) -> None:
        self.mem.write_qword(40, 0)
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=40, vdata=ALL_ONES_F64, offset=0)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_qword(40), ALL_ONES_F64)

    def test_address_offset_and_8byte_alignment(self) -> None:
        # vaddr = 24, offset = 16 -> effective = 40 (aligned)
        self.mem.write_qword(40, 0x000000000000FFFF)
        # unaligned byte 43 with ~7 masking resolves to 40
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=27, vdata=0xFFFF000000000000, offset=16)
        self.assertEqual(ret, 0x000000000000FFFF)
        self.assertEqual(self.mem.read_qword(40), 0xFFFF00000000FFFF)



# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class DSOpcode(Enum):
    """Supported DS family opcodes."""
    DS_OR_RTN_B64 = "ds_or_rtn_b64"


class RegType(Enum):
    """Register file class."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegSingle:
    """Symbolic 32-bit register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}{self.index} "
                f"(expected 0 <= index < {DEFAULT_NUM_VGPRS})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class RegPair:
    """Symbolic 64-bit register pair operand representation v[N:N+1]."""
    base_index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.base_index < DEFAULT_NUM_VGPRS - 1):
            raise ValueError(
                f"Register pair base index out of bounds: {self.reg_type.value}[{self.base_index}:{self.base_index+1}] "
                f"(expected 0 <= index < {DEFAULT_NUM_VGPRS - 1})"
            )

    @property
    def lo(self) -> int:
        return self.base_index

    @property
    def hi(self) -> int:
        return self.base_index + 1

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.lo}:{self.hi}]"


@dataclass(frozen=True)
class DSOrRtnB64Instruction:
    """Symbolic representation of ds_or_rtn_b64 instruction."""
    dst: RegPair
    addr: RegSingle
    data: RegPair
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_OR_RTN_B64

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination operand must be a VGPR pair")
        if self.addr.reg_type != RegType.VGPR:
            raise TypeError("Address operand must be a VGPR")
        if self.data.reg_type != RegType.VGPR:
            raise TypeError("Data operand must be a VGPR pair")
        if not (0 <= self.offset <= 0xFFFF):
            raise ValueError(f"Offset out of 16-bit range: {self.offset} (expected 0 <= offset <= 65535)")

    def disasm(self) -> str:
        """Render disassembled assembly syntax."""
        parts = [f"{self.opcode.value} {self.dst.name}, {self.addr.name}, {self.data.name}"]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_OR_RTN_B64.value, "ds_or_rtn_b64")

    def test_register_operand_validation(self) -> None:
        v0 = RegSingle(index=0, reg_type=RegType.VGPR)
        self.assertEqual(v0.name, "v0")

        vpair = RegPair(base_index=0, reg_type=RegType.VGPR)
        self.assertEqual(vpair.name, "v[0:1]")
        self.assertEqual(vpair.lo, 0)
        self.assertEqual(vpair.hi, 1)

        with self.assertRaises(ValueError):
            RegSingle(index=-1)
        with self.assertRaises(ValueError):
            RegSingle(index=256)
        with self.assertRaises(ValueError):
            RegPair(base_index=255)

    def test_instruction_validation(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        instr = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=64)
        self.assertEqual(instr.offset, 64)
        self.assertFalse(instr.gds)

        with self.assertRaises(ValueError):
            DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=-1)
        with self.assertRaises(ValueError):
            DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=65536)

        s_single = RegSingle(0, reg_type=RegType.SGPR)
        s_pair = RegPair(0, reg_type=RegType.SGPR)
        with self.assertRaises(TypeError):
            DSOrRtnB64Instruction(dst=s_pair, addr=addr, data=data)
        with self.assertRaises(TypeError):
            DSOrRtnB64Instruction(dst=dst, addr=s_single, data=data)
        with self.assertRaises(TypeError):
            DSOrRtnB64Instruction(dst=dst, addr=addr, data=s_pair)

    def test_disasm_rendering(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        instr_basic = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data)
        self.assertEqual(instr_basic.disasm(), "ds_or_rtn_b64 v[0:1], v2, v[4:5]")

        instr_offset = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=32)
        self.assertEqual(instr_offset.disasm(), "ds_or_rtn_b64 v[0:1], v2, v[4:5] offset:32")

        instr_gds = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=64, gds=True)
        self.assertEqual(instr_gds.disasm(), "ds_or_rtn_b64 v[0:1], v2, v[4:5] offset:64 gds")

    def test_structural_equality_and_immutability(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        i1 = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=16)
        i2 = DSOrRtnB64Instruction(dst=dst, addr=addr, data=data, offset=16)
        self.assertEqual(i1, i2)
        self.assertEqual(hash(i1), hash(i2))

        with self.assertRaises(Exception):
            i1.offset = 32  # type: ignore


if __name__ == '__main__':
    unittest.main()
