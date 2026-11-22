"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_nor_rtn_b64.

Opcode:
  - ds_nor_rtn_b64 vdst, vaddr, vdata [offset:N]
      Atomic 64-bit bitwise NOR on LDS memory with 64-bit destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte & (lds_bytes - 1)) & ~7
        old_val = LDS_qword[idx]
        LDS_qword[idx] = ~(old_val | vdata) & 0xFFFFFFFFFFFFFFFF
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in 32-bit VGPR vaddr + optional unsigned immediate byte offset.
  - Alignment: 64-bit access is 8-byte aligned (low 3 bits of byte address masked out).
  - Return value: Prior 64-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR pair v[dst:dst+1].
  - Flags: LDS operations do not alter VCC or SCC condition flags.
  - Bitwise NOR: Operates on raw 64-bit qwords without IEEE 754 interpretation or NaN quietening.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB LDS capacity in bytes
DEFAULT_NUM_VGPRS: int = 256
NUM_LDS_BANKS: int = 32

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


def oracle_nor_b64(a: int, b: int) -> int:
    """Reference 64-bit bitwise NOR returning 64-bit unsigned bit pattern."""
    ua = a & MASK64
    ub = b & MASK64
    return (~(ua | ub)) & MASK64


def ref_ds_nor_rtn_b64(current_mem: int, operand: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nor_rtn_b64.

    Returns:
        (old_val, new_mem) as unsigned 64-bit integers.
    """
    old_val = u64(current_mem)
    new_mem = oracle_nor_b64(old_val, operand)
    return old_val, new_mem


class LDSMemory:
    """Model of Local Data Share memory storage with 64-bit qword access."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_bytes}")
        self.size_bytes = size_bytes
        self.mem: bytearray = bytearray(size_bytes)

    def get_qword_offset(self, byte_addr: int, offset: int = 0) -> int:
        """Calculate effective wrapped byte address, aligned to 8-byte qword boundary."""
        eff_addr = (byte_addr + (offset & OFFSET16_MASK)) & MASK32
        return (eff_addr & (self.size_bytes - 1)) & ~7

    def read_qword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.get_qword_offset(byte_addr, offset)
        return struct.unpack_from("<Q", self.mem, idx)[0]

    def write_qword(self, byte_addr: int, val: int, offset: int = 0) -> None:
        idx = self.get_qword_offset(byte_addr, offset)
        struct.pack_into("<Q", self.mem, idx, u64(val))

    def atomic_nor_b64(self, byte_addr: int, val: int, offset: int = 0) -> int:
        idx = self.get_qword_offset(byte_addr, offset)
        old_val = struct.unpack_from("<Q", self.mem, idx)[0]
        new_val = oracle_nor_b64(old_val, val)
        struct.pack_into("<Q", self.mem, idx, new_val)
        return new_val

    def atomic_nor_rtn_b64(self, byte_addr: int, val: int, offset: int = 0) -> Tuple[int, int]:
        idx = self.get_qword_offset(byte_addr, offset)
        old_val = struct.unpack_from("<Q", self.mem, idx)[0]
        new_val = oracle_nor_b64(old_val, val)
        struct.pack_into("<Q", self.mem, idx, new_val)
        return old_val, new_val


class TestLdsNorRtnB64Oracle(unittest.TestCase):
    """Unit tests for 64-bit mathematical NOR oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_oracle_nor_b64_truth_table(self) -> None:
        # 0 NOR 0 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(oracle_nor_b64(0, 0), MASK64)
        # MASK64 NOR 0 = 0
        self.assertEqual(oracle_nor_b64(MASK64, 0), 0)
        # 0 NOR MASK64 = 0
        self.assertEqual(oracle_nor_b64(0, MASK64), 0)
        # MASK64 NOR MASK64 = 0
        self.assertEqual(oracle_nor_b64(MASK64, MASK64), 0)

    def test_oracle_nor_b64_bit_patterns(self) -> None:
        # Alternating patterns
        self.assertEqual(oracle_nor_b64(0x5555555555555555, 0xAAAAAAAAAAAAAAAA), 0)
        self.assertEqual(oracle_nor_b64(0x5555555555555555, 0), 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(oracle_nor_b64(0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF), 0xF000F000F000F000)

    def test_ref_ds_nor_rtn_b64(self) -> None:
        old_val, new_val = ref_ds_nor_rtn_b64(0, 0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK64)

        old_val2, new_val2 = ref_ds_nor_rtn_b64(0x0123456789ABCDEF, 0xFEDCBA9876543210)
        expected = (~(0x0123456789ABCDEF | 0xFEDCBA9876543210)) & MASK64
        self.assertEqual(old_val2, 0x0123456789ABCDEF)
        self.assertEqual(new_val2, expected)

    def test_lds_qword_alignment_and_wrapping(self) -> None:
        # Byte addresses 0 to 7 all map to qword index 0
        for b in range(8):
            self.assertEqual(self.lds.get_qword_offset(b), 0)
        # Byte addresses 8 to 15 map to offset 8
        for b in range(8, 16):
            self.assertEqual(self.lds.get_qword_offset(b), 8)
        # Immediate offset addition
        self.assertEqual(self.lds.get_qword_offset(8, offset=8), 16)
        # Wrap around 64 KiB (65536)
        self.assertEqual(self.lds.get_qword_offset(65536), 0)
        self.assertEqual(self.lds.get_qword_offset(65544), 8)

    def test_lds_read_write_qword(self) -> None:
        self.lds.write_qword(16, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.lds.read_qword(16), 0xDEADBEEFCAFEBABE)

    def test_lds_atomic_nor_b64(self) -> None:
        self.lds.write_qword(32, 0x5555555555555555)
        old_val, new_val = self.lds.atomic_nor_rtn_b64(32, 0)
        self.assertEqual(old_val, 0x5555555555555555)
        self.assertEqual(new_val, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(self.lds.read_qword(32), 0xAAAAAAAAAAAAAAAA)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(str, enum.Enum):
    """Data Share (DS) 64-bit atomic opcodes for bitwise NOR."""
    DS_NOR_RTN_B64 = "ds_nor_rtn_b64"
    DS_NOR_B64 = "ds_nor_b64"

    @property
    def is_return(self) -> bool:
        return "rtn" in self.value


@dataclasses.dataclass(frozen=True)
class RegSingle:
    """Single 32-bit Vector General Purpose Register (v0 - v255)."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"RegSingle index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class RegPair:
    """Aligned pair of 32-bit registers forming a 64-bit register (v[n:n+1])."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 254):
            raise ValueError(f"RegPair base index out of range [0, 254]: {self.index}")

    def __str__(self) -> str:
        return f"v[{self.index}:{self.index + 1}]"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """16-bit unsigned byte offset immediate for DS instructions."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"OffsetImmediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}"


@dataclasses.dataclass
class DSNorRtnB64Instruction:
    """Symbolic AST node for ds_nor_rtn_b64 / ds_nor_b64 instruction."""
    opcode: DSOpcode
    vaddr: RegSingle
    vdata: RegPair
    vdst: Optional[RegPair] = None
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))
    gds: bool = False

    def __post_init__(self) -> None:
        if self.opcode.is_return and self.vdst is None:
            raise ValueError(f"Return opcode {self.opcode.value} requires vdst register pair")
        if not self.opcode.is_return and self.vdst is not None:
            raise ValueError(f"Non-return opcode {self.opcode.value} must not specify vdst register")

    def to_assembly(self) -> str:
        tokens = [self.opcode.value]
        if self.opcode.is_return:
            assert self.vdst is not None
            tokens.append(f"{self.vdst}, {self.vaddr}, {self.vdata}")
        else:
            tokens.append(f"{self.vaddr}, {self.vdata}")
        suffix_parts = []
        if self.offset.value != 0:
            suffix_parts.append(str(self.offset))
        if self.gds:
            suffix_parts.append("gds")
        if suffix_parts:
            return f"{tokens[0]} {tokens[1]} {' '.join(suffix_parts)}"
        return f"{tokens[0]} {tokens[1]}"

    @classmethod
    def parse(cls, asm: str) -> DSNorRtnB64Instruction:
        text = asm.strip()
        if not text:
            raise ValueError("Empty assembly string")
        parts = text.split(None, 1)
        op_str = parts[0]
        try:
            opcode = DSOpcode(op_str)
        except ValueError:
            raise ValueError(f"Unknown or unsupported opcode: {op_str}")

        if len(parts) < 2:
            raise ValueError("Missing instruction operands")

        rest = parts[1].strip()
        tokens = rest.split()
        reg_tokens: List[str] = []
        mod_tokens: List[str] = []
        for t in tokens:
            if t.startswith("offset:") or t == "gds":
                mod_tokens.append(t)
            else:
                reg_tokens.append(t)

        reg_str = " ".join(reg_tokens)
        reg_names = [r.strip() for r in reg_str.split(",") if r.strip()]

        def parse_single(name: str) -> RegSingle:
            if not name.startswith("v") or "[" in name:
                raise ValueError(f"Expected single VGPR, got {name}")
            return RegSingle(int(name[1:]))

        def parse_pair(name: str) -> RegPair:
            # Matches v[a:b] or va
            if name.startswith("v[") and name.endswith("]"):
                inner = name[2:-1]
                p = inner.split(":")
                return RegPair(int(p[0]))
            elif name.startswith("v"):
                return RegPair(int(name[1:]))
            raise ValueError(f"Expected VGPR pair, got {name}")

        offset_val = 0
        gds = False
        for mod in mod_tokens:
            if mod.startswith("offset:"):
                offset_val = int(mod.split(":", 1)[1])
            elif mod == "gds":
                gds = True
            else:
                raise ValueError(f"Unknown modifier: {mod}")

        offset = OffsetImmediate(offset_val)

        if opcode.is_return:
            if len(reg_names) != 3:
                raise ValueError(f"Expected 3 operands for {opcode.value}, got {len(reg_names)}")
            vdst = parse_pair(reg_names[0])
            vaddr = parse_single(reg_names[1])
            vdata = parse_pair(reg_names[2])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=vdst, offset=offset, gds=gds)
        else:
            if len(reg_names) != 2:
                raise ValueError(f"Expected 2 operands for {opcode.value}, got {len(reg_names)}")
            vaddr = parse_single(reg_names[0])
            vdata = parse_pair(reg_names[1])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, offset=offset, gds=gds)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for 64-bit AST opcode definitions, typed operands, and parsing."""

    def test_register_bounds(self) -> None:
        v0 = RegSingle(0)
        v255 = RegSingle(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        with self.assertRaises(ValueError):
            RegSingle(-1)
        with self.assertRaises(ValueError):
            RegSingle(256)

        vp0 = RegPair(0)
        vp254 = RegPair(254)
        self.assertEqual(str(vp0), "v[0:1]")
        self.assertEqual(str(vp254), "v[254:255]")
        with self.assertRaises(ValueError):
            RegPair(255)

    def test_instruction_validation(self) -> None:
        inst_rtn = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vaddr=RegSingle(1),
            vdata=RegPair(2),
            vdst=RegPair(0),
            offset=OffsetImmediate(8),
        )
        self.assertEqual(inst_rtn.to_assembly(), "ds_nor_rtn_b64 v[0:1], v1, v[2:3] offset:8")

        inst_nortn = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(1),
            vdata=RegPair(2),
        )
        self.assertEqual(inst_nortn.to_assembly(), "ds_nor_b64 v1, v[2:3]")

        with self.assertRaises(ValueError):
            DSNorRtnB64Instruction(
                opcode=DSOpcode.DS_NOR_RTN_B64,
                vaddr=RegSingle(1),
                vdata=RegPair(2),
                vdst=None,
            )

    def test_instruction_parsing_and_roundtrip(self) -> None:
        asm1 = "ds_nor_rtn_b64 v[0:1], v1, v[2:3] offset:16"
        inst1 = DSNorRtnB64Instruction.parse(asm1)
        self.assertEqual(inst1.opcode, DSOpcode.DS_NOR_RTN_B64)
        self.assertEqual(inst1.vdst, RegPair(0))
        self.assertEqual(inst1.vaddr, RegSingle(1))
        self.assertEqual(inst1.vdata, RegPair(2))
        self.assertEqual(inst1.offset.value, 16)
        self.assertEqual(inst1.to_assembly(), asm1)

        asm2 = "ds_nor_b64 v5, v[6:7]"
        inst2 = DSNorRtnB64Instruction.parse(asm2)
        self.assertEqual(inst2.opcode, DSOpcode.DS_NOR_B64)
        self.assertIsNone(inst2.vdst)
        self.assertEqual(inst2.to_assembly(), asm2)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Architectural register and local memory state for 64-bit scalar operations."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.lds: LDSMemory = lds if lds is not None else LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def get_vgpr(self, reg: RegSingle) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: RegSingle, val: int) -> None:
        self.vgprs[reg.index] = u32(val)

    def get_vgpr_pair(self, reg: RegPair) -> int:
        lo = self.vgprs[reg.index]
        hi = self.vgprs[reg.index + 1]
        return (hi << 32) | lo

    def set_vgpr_pair(self, reg: RegPair, val: int) -> None:
        self.vgprs[reg.index] = u32(val)
        self.vgprs[reg.index + 1] = u32(val >> 32)


class ScalarInterpreter:
    """Deterministic scalar interpreter for ds_nor_rtn_b64 and ds_nor_b64."""

    def execute(self, state: ScalarMachineState, inst: DSNorRtnB64Instruction) -> Optional[int]:
        vaddr_val = state.get_vgpr(inst.vaddr)
        vdata_val = state.get_vgpr_pair(inst.vdata)

        old_val, new_val = state.lds.atomic_nor_rtn_b64(
            byte_addr=vaddr_val,
            val=vdata_val,
            offset=inst.offset.value,
        )

        if inst.opcode.is_return:
            assert inst.vdst is not None
            state.set_vgpr_pair(inst.vdst, old_val)
            return old_val
        return None


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic 64-bit scalar instruction execution."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarMachineState(lds=self.lds)
        self.interp = ScalarInterpreter()

    def test_ds_nor_rtn_b64_execution(self) -> None:
        self.lds.write_qword(16, 0x0000000000000000)
        self.state.set_vgpr(RegSingle(2), 16)
        self.state.set_vgpr_pair(RegPair(4), 0x0000000000000000)
        self.state.set_vgpr_pair(RegPair(0), 0xDEADBEEFCAFEBABE)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, 0)
        self.assertEqual(self.state.get_vgpr_pair(RegPair(0)), 0)
        self.assertEqual(self.state.get_vgpr(RegSingle(0)), 0)
        self.assertEqual(self.state.get_vgpr(RegSingle(1)), 0)
        self.assertEqual(self.state.get_vgpr(RegSingle(2)), 16)
        self.assertEqual(self.lds.read_qword(16), MASK64)

    def test_ds_nor_b64_non_return(self) -> None:
        self.lds.write_qword(32, 0x00000000FFFFFFFF)
        self.state.set_vgpr(RegSingle(2), 32)
        self.state.set_vgpr_pair(RegPair(4), 0xFFFFFFFF00000000)
        self.state.set_vgpr_pair(RegPair(0), 0x1122334455667788)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertIsNone(ret)
        self.assertEqual(self.state.get_vgpr_pair(RegPair(0)), 0x1122334455667788)
        self.assertEqual(self.lds.read_qword(32), 0)

    def test_double_nor_identity_64bit(self) -> None:
        initial = 0x0123456789ABCDEF
        self.lds.write_qword(8, initial)
        self.state.set_vgpr(RegSingle(2), 8)
        self.state.set_vgpr_pair(RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )

        ret1 = self.interp.execute(self.state, inst)
        self.assertEqual(ret1, initial)
        inverted = (~initial) & MASK64
        self.assertEqual(self.lds.read_qword(8), inverted)

        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, inverted)
        self.assertEqual(self.lds.read_qword(8), initial)

    def test_offset_immediate_computation_64bit(self) -> None:
        self.lds.write_qword(24, 0x5555555555555555)
        self.state.set_vgpr(RegSingle(2), 16)
        self.state.set_vgpr_pair(RegPair(4), 0xAAAAAAAAAAAAAAAA)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
            offset=OffsetImmediate(8),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x5555555555555555)
        self.assertEqual(self.lds.read_qword(24), 0)


if __name__ == "__main__":
    unittest.main()
