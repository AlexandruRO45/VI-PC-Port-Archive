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


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDState:
    """Architectural state for a 64-bit SIMD wavefront (Wave32 or Wave64)."""

    def __init__(
        self,
        num_lanes: int = 64,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes: int = num_lanes
        default_mask = (1 << num_lanes) - 1
        self.exec_mask: int = default_mask if exec_mask is None else (exec_mask & default_mask)
        self.lds: LDSMemory = lds if lds is not None else LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def get_lane_vgpr(self, lane: int, reg: RegSingle) -> int:
        return self.vgprs[lane][reg.index]

    def set_lane_vgpr(self, lane: int, reg: RegSingle, val: int) -> None:
        self.vgprs[lane][reg.index] = u32(val)

    def get_lane_vgpr_pair(self, lane: int, reg: RegPair) -> int:
        lo = self.vgprs[lane][reg.index]
        hi = self.vgprs[lane][reg.index + 1]
        return (hi << 32) | lo

    def set_lane_vgpr_pair(self, lane: int, reg: RegPair, val: int) -> None:
        self.vgprs[lane][reg.index] = u32(val)
        self.vgprs[lane][reg.index + 1] = u32(val >> 32)


class SIMDInterpreter:
    """Deterministic SIMD interpreter simulating 64-bit wave execution."""

    def execute(self, state: SIMDState, inst: DSNorRtnB64Instruction) -> Dict[int, Optional[int]]:
        results: Dict[int, Optional[int]] = {}
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results[lane] = None
                continue

            vaddr_val = state.get_lane_vgpr(lane, inst.vaddr)
            vdata_val = state.get_lane_vgpr_pair(lane, inst.vdata)

            old_val, new_val = state.lds.atomic_nor_rtn_b64(
                byte_addr=vaddr_val,
                val=vdata_val,
                offset=inst.offset.value,
            )

            if inst.opcode.is_return:
                assert inst.vdst is not None
                state.set_lane_vgpr_pair(lane, inst.vdst, old_val)
                results[lane] = old_val
            else:
                results[lane] = None

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for 64-bit SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_all_lanes_active_independent_addresses(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDInterpreter()

        # Each lane has its own 8-byte aligned address: lane * 8
        for lane in range(32):
            addr = lane * 8
            lds.write_qword(addr, lane)
            state.set_lane_vgpr(lane, RegSingle(2), addr)
            state.set_lane_vgpr_pair(lane, RegPair(4), 0)
            state.set_lane_vgpr_pair(lane, RegPair(0), 0xCAFEBABECAFEBABE)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        results = interp.execute(state, inst)

        for lane in range(32):
            self.assertEqual(results[lane], lane)
            self.assertEqual(state.get_lane_vgpr_pair(lane, RegPair(0)), lane)
            expected = (~lane) & MASK64
            self.assertEqual(lds.read_qword(lane * 8), expected)

    def test_wave64_divergent_exec_mask(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        even_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=even_mask, lds=lds)
        interp = SIMDInterpreter()

        sentinel = 0xBAADF00DBAADF00D
        for lane in range(64):
            addr = lane * 8
            lds.write_qword(addr, 0)
            state.set_lane_vgpr(lane, RegSingle(2), addr)
            state.set_lane_vgpr_pair(lane, RegPair(4), 0)
            state.set_lane_vgpr_pair(lane, RegPair(0), sentinel)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        results = interp.execute(state, inst)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0)
                self.assertEqual(state.get_lane_vgpr_pair(lane, RegPair(0)), 0)
                self.assertEqual(lds.read_qword(lane * 8), MASK64)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr_pair(lane, RegPair(0)), sentinel)
                self.assertEqual(lds.read_qword(lane * 8), 0)

    def test_multi_lane_64bit_address_collision_serialization(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        state = SIMDState(num_lanes=32, exec_mask=0xF, lds=lds)  # Lanes 0..3 active
        interp = SIMDInterpreter()

        collision_addr = 64
        lds.write_qword(collision_addr, 0)

        inputs = [
            0x0000000000000000,
            0x00000000FFFFFFFF,
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
        ]
        for lane in range(4):
            state.set_lane_vgpr(lane, RegSingle(2), collision_addr)
            state.set_lane_vgpr_pair(lane, RegPair(4), inputs[lane])
            state.set_lane_vgpr_pair(lane, RegPair(0), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        results = interp.execute(state, inst)

        # Lane 0: reads 0, writes ~(0 | 0) = MASK64
        self.assertEqual(results[0], 0)
        self.assertEqual(state.get_lane_vgpr_pair(0, RegPair(0)), 0)

        # Lane 1: reads MASK64, writes ~(MASK64 | 0x00000000FFFFFFFF) = 0
        self.assertEqual(results[1], MASK64)
        self.assertEqual(state.get_lane_vgpr_pair(1, RegPair(0)), MASK64)

        # Lane 2: reads 0, writes ~(0 | 0x5555555555555555) = 0xAAAAAAAAAAAAAAAA
        self.assertEqual(results[2], 0)
        self.assertEqual(state.get_lane_vgpr_pair(2, RegPair(0)), 0)

        # Lane 3: reads 0xAAAAAAAAAAAAAAAA, writes ~(0xAAAAAAAAAAAAAAAA | 0xAAAAAAAAAAAAAAAA) = 0x5555555555555555
        self.assertEqual(results[3], 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(state.get_lane_vgpr_pair(3, RegPair(0)), 0xAAAAAAAAAAAAAAAA)

        # Final memory must be 0x5555555555555555
        self.assertEqual(lds.read_qword(collision_addr), 0x5555555555555555)

    def test_zero_exec_mask_64bit(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        state = SIMDState(num_lanes=32, exec_mask=0, lds=lds)
        interp = SIMDInterpreter()

        lds.write_qword(0, 0x1122334455667788)
        for lane in range(32):
            state.set_lane_vgpr_pair(lane, RegPair(0), 0x7777777777777777)
            state.set_lane_vgpr(lane, RegSingle(2), 0)
            state.set_lane_vgpr_pair(lane, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        results = interp.execute(state, inst)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_lane_vgpr_pair(lane, RegPair(0)), 0x7777777777777777)
        self.assertEqual(lds.read_qword(0), 0x1122334455667788)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for 64-bit register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_alias_vdst_and_vaddr(self) -> None:
        # ds_nor_rtn_b64 v[0:1], v0, v[2:3]: vdst pair starts at v0, vaddr is single v0
        state = ScalarMachineState(lds=self.lds)
        initial = 0x1111222233334444
        self.lds.write_qword(16, initial)
        state.set_vgpr(RegSingle(0), 16)  # v0 holds address
        state.set_vgpr_pair(RegPair(2), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(0),
            vdata=RegPair(2),
        )
        ret = self.scalar_interp.execute(state, inst)

        # Must return old memory value into v[0:1]
        self.assertEqual(ret, initial)
        self.assertEqual(state.get_vgpr_pair(RegPair(0)), initial)
        self.assertEqual(self.lds.read_qword(16), (~initial) & MASK64)

    def test_alias_vdst_and_vdata(self) -> None:
        # ds_nor_rtn_b64 v[0:1], v2, v[0:1]: vdst pair aliases vdata pair
        state = ScalarMachineState(lds=self.lds)
        initial = 0x5555555555555555
        self.lds.write_qword(24, initial)
        state.set_vgpr(RegSingle(2), 24)
        state.set_vgpr_pair(RegPair(0), 0xAAAAAAAAAAAAAAAA)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(0),
        )
        ret = self.scalar_interp.execute(state, inst)

        # Must return old memory value into v[0:1]
        self.assertEqual(ret, initial)
        self.assertEqual(state.get_vgpr_pair(RegPair(0)), initial)
        self.assertEqual(self.lds.read_qword(24), 0)

    def test_partial_overlap_aliasing(self) -> None:
        # vdst is v[1:2], vaddr is v0, vdata is v[2:3] (v2 overlaps between vdst and vdata)
        state = ScalarMachineState(lds=self.lds)
        initial = 0xAAAAAAAAAAAAAAAA
        self.lds.write_qword(32, initial)
        state.set_vgpr(RegSingle(0), 32)
        state.set_vgpr_pair(RegPair(2), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(1),
            vaddr=RegSingle(0),
            vdata=RegPair(2),
        )
        ret = self.scalar_interp.execute(state, inst)

        self.assertEqual(ret, initial)
        self.assertEqual(state.get_vgpr_pair(RegPair(1)), initial)
        self.assertEqual(self.lds.read_qword(32), 0x5555555555555555)

    def test_simd_wave32_aliasing(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        for lane in range(32):
            addr = lane * 8
            self.lds.write_qword(addr, 0)
            wave.set_lane_vgpr(lane, RegSingle(0), addr)
            wave.set_lane_vgpr_pair(lane, RegPair(2), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(0),
            vdata=RegPair(2),
        )
        results = self.simd_interp.execute(wave, inst)

        for lane in range(32):
            self.assertEqual(results[lane], 0)
            self.assertEqual(wave.get_lane_vgpr_pair(lane, RegPair(0)), 0)
            self.assertEqual(self.lds.read_qword(lane * 8), MASK64)

    def test_toggling_oscillation_in_place(self) -> None:
        state = ScalarMachineState(lds=self.lds)
        initial = 0x0123456789ABCDEF
        self.lds.write_qword(0, initial)
        state.set_vgpr(RegSingle(2), 0)
        state.set_vgpr_pair(RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )

        # Pass 1: initial -> ~initial
        ret1 = self.scalar_interp.execute(state, inst)
        self.assertEqual(ret1, initial)
        self.assertEqual(self.lds.read_qword(0), (~initial) & MASK64)

        # Pass 2: ~initial -> initial
        ret2 = self.scalar_interp.execute(state, inst)
        self.assertEqual(ret2, (~initial) & MASK64)
        self.assertEqual(self.lds.read_qword(0), initial)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass
class LDSStatusFlags64:
    """Architectural status flags, wrapping conditions, and bank metrics for 64-bit LDS operations."""
    carry_out: bool = False
    unaligned_address: bool = False
    oob_wrapped: bool = False
    bank_conflicts: int = 0
    address_collisions: int = 0


@dataclasses.dataclass(frozen=True)
class BitwiseConditionCodes64:
    """Architectural 64-bit bitwise condition codes."""
    zero: bool
    sign_bit: bool
    all_ones: bool
    updated: bool


def evaluate_status_flags_64(
    wave: SIMDState,
    inst: DSNorRtnB64Instruction,
) -> LDSStatusFlags64:
    """Evaluate architectural flags, address wrap conditions, and bank conflicts for 64-bit LDS ops."""
    flags = LDSStatusFlags64()
    active_lanes = [l for l in range(wave.num_lanes) if wave.is_lane_active(l)]

    bank_to_dwords: Dict[int, Set[int]] = {}
    qword_to_lanes: Dict[int, List[int]] = {}

    for lane in active_lanes:
        addr = wave.get_lane_vgpr(lane, inst.vaddr)
        off = inst.offset.value

        if (addr + off) > MASK32:
            flags.carry_out = True

        effective_byte = (addr + off) & MASK32
        if (effective_byte & 7) != 0:
            flags.unaligned_address = True

        if effective_byte >= wave.lds.size_bytes:
            flags.oob_wrapped = True

        qword_idx = wave.lds.get_qword_offset(addr, off)
        # 64-bit qword spans 2 consecutive dwords: dword0 and dword1
        dword0 = (qword_idx >> 2) & ((wave.lds.size_bytes // 4) - 1)
        dword1 = (dword0 + 1) & ((wave.lds.size_bytes // 4) - 1)
        bank0 = dword0 % NUM_LDS_BANKS
        bank1 = dword1 % NUM_LDS_BANKS

        bank_to_dwords.setdefault(bank0, set()).add(dword0)
        bank_to_dwords.setdefault(bank1, set()).add(dword1)
        qword_to_lanes.setdefault(qword_idx, []).append(lane)

    for bank, dword_set in bank_to_dwords.items():
        if len(dword_set) > 1:
            flags.bank_conflicts += (len(dword_set) - 1)

    for qword, lanes in qword_to_lanes.items():
        if len(lanes) > 1:
            flags.address_collisions += (len(lanes) - 1)

    return flags


def compute_bitwise_condition_codes_64(old_val: int, new_val: int) -> BitwiseConditionCodes64:
    """Compute condition codes from old memory content and update status."""
    norm_old = u64(old_val)
    return BitwiseConditionCodes64(
        zero=(norm_old == 0),
        sign_bit=bool(norm_old & 0x8000000000000000),
        all_ones=(norm_old == MASK64),
        updated=(u64(new_val) != norm_old),
    )


class TestConditionCodesAndFlags(unittest.TestCase):
    """Unit tests for 64-bit status flags, condition codes, and carry/borrow isolation."""

    def test_address_carry_out_generation_64bit(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0x1)
        wave.set_lane_vgpr(0, RegSingle(2), 0xFFFFFFFF)
        wave.set_lane_vgpr_pair(0, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
            offset=OffsetImmediate(16),
        )
        flags = evaluate_status_flags_64(wave, inst)
        self.assertTrue(flags.carry_out)

    def test_unaligned_byte_address_detection_64bit(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0x1)
        wave.set_lane_vgpr(0, RegSingle(2), 5)  # not 8-byte aligned
        wave.set_lane_vgpr_pair(0, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        flags = evaluate_status_flags_64(wave, inst)
        self.assertTrue(flags.unaligned_address)

    def test_oob_address_wrapping_detection_64bit(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0x1)
        wave.set_lane_vgpr(0, RegSingle(2), 70000)
        wave.set_lane_vgpr_pair(0, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        flags = evaluate_status_flags_64(wave, inst)
        self.assertTrue(flags.oob_wrapped)

    def test_bank_conflicts_detection_64bit(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0x7)
        # Bank conflicts when distinct dwords fall in the same bank
        # Bank 0 dwords: 0, 32, 64
        wave.set_lane_vgpr(0, RegSingle(2), 0 * 4)
        wave.set_lane_vgpr(1, RegSingle(2), 32 * 4)
        wave.set_lane_vgpr(2, RegSingle(2), 64 * 4)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        flags = evaluate_status_flags_64(wave, inst)
        self.assertGreater(flags.bank_conflicts, 0)

    def test_address_collisions_detection_64bit(self) -> None:
        wave = SIMDState(num_lanes=32, exec_mask=0xF)
        # All 4 lanes target same address 64
        for l in range(4):
            wave.set_lane_vgpr(l, RegSingle(2), 64)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        flags = evaluate_status_flags_64(wave, inst)
        self.assertEqual(flags.address_collisions, 3)

    def test_bitwise_condition_codes_64bit(self) -> None:
        cc_zero = compute_bitwise_condition_codes_64(0, MASK64)
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.sign_bit)
        self.assertFalse(cc_zero.all_ones)
        self.assertTrue(cc_zero.updated)

        cc_ones = compute_bitwise_condition_codes_64(MASK64, 0)
        self.assertFalse(cc_ones.zero)
        self.assertTrue(cc_ones.sign_bit)
        self.assertTrue(cc_ones.all_ones)
        self.assertTrue(cc_ones.updated)

        cc_sign = compute_bitwise_condition_codes_64(0x8000000000000000, 0x8000000000000000)
        self.assertFalse(cc_sign.zero)
        self.assertTrue(cc_sign.sign_bit)
        self.assertFalse(cc_sign.all_ones)
        self.assertFalse(cc_sign.updated)

    def test_carry_borrow_vcc_isolation_64bit(self) -> None:
        vcc_sentinel = 0xAAAAAAAAAAAAAAAA
        vcc_after = vcc_sentinel
        self.assertEqual(vcc_after, vcc_sentinel)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754BitPatternHandling(unittest.TestCase):
    """Exhaustive unit tests for 64-bit boundary extremes and IEEE 754 float64 bit-patterns."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarMachineState(lds=self.lds)
        self.interp = ScalarInterpreter()

    def test_signed_zero_float64_bit_patterns(self) -> None:
        # nor(+0.0, -0.0) = ~(0 | 0x8000000000000000) = 0x7FFFFFFFFFFFFFFF
        self.assertEqual(oracle_nor_b64(POS_ZERO_F64, NEG_ZERO_F64), 0x7FFFFFFFFFFFFFFF)
        self.assertEqual(oracle_nor_b64(NEG_ZERO_F64, POS_ZERO_F64), 0x7FFFFFFFFFFFFFFF)

        self.lds.write_qword(0, POS_ZERO_F64)
        self.state.set_vgpr(RegSingle(2), 0)
        self.state.set_vgpr_pair(RegPair(4), NEG_ZERO_F64)
        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, POS_ZERO_F64)
        self.assertEqual(self.lds.read_qword(0), 0x7FFFFFFFFFFFFFFF)

    def test_infinities_float64_bit_patterns(self) -> None:
        # nor(+inf, -inf) = ~(0x7FF0000000000000 | 0xFFF0000000000000) = ~0xFFF0000000000000 = 0x000FFFFFFFFFFFFF
        self.assertEqual(oracle_nor_b64(POS_INF_F64, NEG_INF_F64), MAX_SUBNORMAL_F64)

        self.lds.write_qword(8, POS_INF_F64)
        self.state.set_vgpr(RegSingle(2), 8)
        self.state.set_vgpr_pair(RegPair(4), NEG_INF_F64)
        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, POS_INF_F64)
        self.assertEqual(self.lds.read_qword(8), MAX_SUBNORMAL_F64)

    def test_nan_float64_bit_patterns_unaltered_and_no_traps(self) -> None:
        # qNaN NOR 0 = ~QNAN_F64
        self.assertEqual(oracle_nor_b64(QNAN_F64, 0), (~QNAN_F64) & MASK64)

        # sNaN NOR MASK64 = 0
        self.assertEqual(oracle_nor_b64(SNAN_F64, MASK64), 0)

    def test_subnormal_float64_bit_patterns(self) -> None:
        # MIN_SUBNORMAL_F64 (1) NOR 0 = ~1 = 0xFFFFFFFFFFFFFFFE
        self.assertEqual(oracle_nor_b64(MIN_SUBNORMAL_F64, 0), (~MIN_SUBNORMAL_F64) & MASK64)

        # MAX_SUBNORMAL_F64 NOR 0 = ~MAX_SUBNORMAL_F64 = NEG_INF_F64
        self.assertEqual(oracle_nor_b64(MAX_SUBNORMAL_F64, 0), NEG_INF_F64)

    def test_boundary_permutations_64bit(self) -> None:
        extremes = [
            0x0000000000000000,
            0x0000000000000001,
            0x8000000000000000,
            0x7FFFFFFFFFFFFFFF,
            MASK64,
        ]
        for a in extremes:
            for b in extremes:
                exp = (~(a | b)) & MASK64
                self.assertEqual(oracle_nor_b64(a, b), exp)

                self.lds.write_qword(0, a)
                self.state.set_vgpr(RegSingle(2), 0)
                self.state.set_vgpr_pair(RegPair(4), b)
                inst = DSNorRtnB64Instruction(
                    opcode=DSOpcode.DS_NOR_RTN_B64,
                    vdst=RegPair(0),
                    vaddr=RegSingle(2),
                    vdata=RegPair(4),
                )
                ret = self.interp.execute(self.state, inst)
                self.assertEqual(ret, a)
                self.assertEqual(self.lds.read_qword(0), exp)

    def test_single_bit_walks_64bit(self) -> None:
        for k in range(64):
            bit = 1 << k
            self.assertEqual(oracle_nor_b64(bit, 0), (~bit) & MASK64)
            self.assertEqual(oracle_nor_b64(bit, bit), (~bit) & MASK64)
            self.assertEqual(oracle_nor_b64(bit, (~bit) & MASK64), 0)

    def test_alternating_bit_patterns_64bit(self) -> None:
        pat_a = 0x5555555555555555
        pat_b = 0xAAAAAAAAAAAAAAAA
        self.assertEqual(oracle_nor_b64(pat_a, pat_b), 0)
        self.assertEqual(oracle_nor_b64(pat_a, pat_a), pat_b)
        self.assertEqual(oracle_nor_b64(pat_b, pat_b), pat_a)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection for ds_nor_rtn_b64."""

    def test_syntax_rejection_invalid_opcodes_and_syntax(self) -> None:
        invalid_assemblies = [
            "",
            "   ",
            "v_nor_b64 v[0:1], v[2:3]",
            "ds_nand_b64 v[0:1], v2, v[3:4]",
            "ds_nor_u64 v[0:1], v2, v[3:4]",
            "ds_nor_rtn_b64",
            "ds_nor_rtn_b64 v[0:1], v2",
            "ds_nor_b64 v2",
            "ds_nor_b64 v0, v[1:2], v[3:4]",
            "ds_nor_rtn_b64 v[0:1], v2, v[3:4], v[5:6]",
            "ds_nor_b64 s0, v[1:2]",
            "ds_nor_b64 r0, v[1:2]",
            "ds_nor_b64 v0, v[1:2] offset:70000",
            "ds_nor_b64 v0, v[1:2] offset:-1",
            "ds_nor_b64 v0, v[1:2] offset:invalid",
            "ds_nor_b64 v0, v[1:2] badmod",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, IndexError)):
                    DSNorRtnB64Instruction.parse(asm)

    def test_architectural_model_negative_controls(self) -> None:
        with self.assertRaises(ValueError):
            LDSMemory(0)
        with self.assertRaises(ValueError):
            LDSMemory(-64)
        with self.assertRaises(ValueError):
            LDSMemory(1000)

        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_mutation_detection_inverted_logic_or(self) -> None:
        def mutant_or(a: int, b: int) -> int:
            return (a | b) & MASK64

        self.assertNotEqual(mutant_or(0, 0), oracle_nor_b64(0, 0))
        self.assertNotEqual(
            mutant_or(0x5555555555555555, 0xAAAAAAAAAAAAAAAA),
            oracle_nor_b64(0x5555555555555555, 0xAAAAAAAAAAAAAAAA),
        )

    def test_mutation_detection_inverted_logic_nand(self) -> None:
        def mutant_nand(a: int, b: int) -> int:
            return (~(a & b)) & MASK64

        self.assertNotEqual(
            mutant_nand(0x00000000FFFFFFFF, 0xFFFFFFFF00000000),
            oracle_nor_b64(0x00000000FFFFFFFF, 0xFFFFFFFF00000000),
        )

    def test_mutation_detection_destructive_aliasing(self) -> None:
        class DestructiveMutantInterpreter(ScalarInterpreter):
            def execute(self, state: ScalarMachineState, inst: DSNorRtnB64Instruction) -> Optional[int]:
                if inst.opcode.is_return:
                    assert inst.vdst is not None
                    state.set_vgpr_pair(inst.vdst, 0xDEADBEEFCAFEBABE)
                return super().execute(state, inst)

        lds = LDSMemory(65536)
        lds.write_qword(16, 0)
        state = ScalarMachineState(lds=lds)
        state.set_vgpr(RegSingle(0), 16)  # v0 holds address
        state.set_vgpr_pair(RegPair(2), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(0),
            vdata=RegPair(2),
        )
        mutant = DestructiveMutantInterpreter()
        mutant.execute(state, inst)

        self.assertNotEqual(lds.read_qword(16), MASK64)

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        class MutantSIMDInterpreter(SIMDInterpreter):
            def execute(self, state: SIMDState, inst: DSNorRtnB64Instruction) -> Dict[int, Optional[int]]:
                saved_mask = state.exec_mask
                state.exec_mask = (1 << state.num_lanes) - 1
                try:
                    return super().execute(state, inst)
                finally:
                    state.exec_mask = saved_mask

        lds = LDSMemory(65536)
        lds.write_qword(16, 0)
        wave = SIMDState(num_lanes=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.set_lane_vgpr(l, RegSingle(2), l * 8)
            wave.set_lane_vgpr_pair(l, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )
        mutant = MutantSIMDInterpreter()
        mutant.execute(wave, inst)

        self.assertEqual(lds.read_qword(16), MASK64)

        lds.write_qword(16, 0)
        SIMDInterpreter().execute(wave, inst)
        self.assertEqual(lds.read_qword(16), 0)

    def test_mutation_detection_dropped_offset(self) -> None:
        class DroppedOffsetInterpreter(ScalarInterpreter):
            def execute(self, state: ScalarMachineState, inst: DSNorRtnB64Instruction) -> Optional[int]:
                no_off_inst = DSNorRtnB64Instruction(
                    opcode=inst.opcode,
                    vaddr=inst.vaddr,
                    vdata=inst.vdata,
                    vdst=inst.vdst,
                    offset=OffsetImmediate(0),
                )
                return super().execute(state, no_off_inst)

        lds = LDSMemory(65536)
        lds.write_qword(8, 0)
        lds.write_qword(16, 0)
        state = ScalarMachineState(lds=lds)
        state.set_vgpr(RegSingle(2), 8)
        state.set_vgpr_pair(RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_B64,
            vaddr=RegSingle(2),
            vdata=RegPair(4),
            offset=OffsetImmediate(8),  # Points to qword 16
        )
        mutant = DroppedOffsetInterpreter()
        mutant.execute(state, inst)

        self.assertEqual(lds.read_qword(8), MASK64)
        self.assertEqual(lds.read_qword(16), 0)

    def test_mutation_detection_non_atomic_collision(self) -> None:
        class NonAtomicSIMDInterpreter(SIMDInterpreter):
            def execute(self, state: SIMDState, inst: DSNorRtnB64Instruction) -> Dict[int, Optional[int]]:
                results: Dict[int, Optional[int]] = {}
                read_vals: Dict[int, int] = {}
                for lane in range(state.num_lanes):
                    if state.is_lane_active(lane):
                        vaddr_val = state.get_lane_vgpr(lane, inst.vaddr)
                        read_vals[lane] = state.lds.read_qword(vaddr_val, inst.offset.value)
                for lane in range(state.num_lanes):
                    if state.is_lane_active(lane):
                        if inst.opcode.is_return:
                            assert inst.vdst is not None
                            state.set_lane_vgpr_pair(lane, inst.vdst, read_vals[lane])
                            results[lane] = read_vals[lane]
                return results

        lds = LDSMemory(65536)
        lds.write_qword(0, 0)
        wave = SIMDState(num_lanes=32, exec_mask=0x3, lds=lds)
        wave.set_lane_vgpr(0, RegSingle(2), 0)
        wave.set_lane_vgpr_pair(0, RegPair(4), 0)
        wave.set_lane_vgpr(1, RegSingle(2), 0)
        wave.set_lane_vgpr_pair(1, RegPair(4), 0)

        inst = DSNorRtnB64Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B64,
            vdst=RegPair(0),
            vaddr=RegSingle(2),
            vdata=RegPair(4),
        )

        mutant = NonAtomicSIMDInterpreter()
        mutant_res = mutant.execute(wave, inst)
        self.assertEqual(mutant_res[1], 0)

        lds.write_qword(0, 0)
        true_res = SIMDInterpreter().execute(wave, inst)
        self.assertEqual(true_res[1], MASK64)


if __name__ == "__main__":
    unittest.main()
