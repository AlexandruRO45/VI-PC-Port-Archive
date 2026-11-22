from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_dec_rtn_u32.

Opcode:
  - ds_dec_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]
      Atomic 32-bit decrement on LDS memory with destination register return:
        effective_byte = (vaddr + offset16) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        if old_val == 0 or old_val > vdata0:
            new_val = vdata0
        else:
            new_val = (old_val - 1) & 0xFFFFFFFF
        LDS[idx] = new_val
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + unsigned 16-bit immediate byte offset (offset:offset16).
    Dword-aligned: hardware drops low 2 bits when indexing 32-bit dwords in LDS.
  - Modulo/wrapping arithmetic:
      If old_val == 0 or old_val > vdata0: wraps to vdata0.
      Else: decrements by 1 (old_val - 1).
  - Return value: Previous value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
from typing import Dict, List, Optional, Tuple, Union
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
LDS_DEFAULT_SIZE_BYTES: int = 65536   # 64 KiB default Workgroup LDS allocation
LDS_DEFAULT_SIZE_DWORDS: int = 16384  # 16,384 32-bit dwords
OFFSET16_MASK: int = 0xFFFF
DEFAULT_NUM_LANES: int = 64           # Wave64 default width
DEFAULT_NUM_VGPRS: int = 256          # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106          # Standard GFX10 scalar register file size

# IEEE 754 Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
ONE_F32: int = 0x3F800000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate byte offset.
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_DEFAULT_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def ref_ds_dec_u32(old_val: int, vdata0: int) -> int:
    """Pure reference mathematical formula for ds_dec_u32.

    If old_val == 0 or old_val > vdata0:
        new_val = vdata0
    else:
        new_val = (old_val - 1) & 0xFFFFFFFF
    """
    a = u32(old_val)
    b = u32(vdata0)
    if a == 0 or a > b:
        return b
    return u32(a - 1)


class LDSMemory:
    """Architectural model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = LDS_DEFAULT_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: List[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)

    def atomic_dec_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit decrement with return on LDS memory.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        old_val = self.read_dword(idx)
        new_val = ref_ds_dec_u32(old_val, vdata0)
        self.write_dword(idx, new_val)
        return old_val, new_val


def oracle_ds_dec_rtn_u32(
    lds_mem: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES,
) -> Tuple[int, int]:
    """Reference bytearray mathematical oracle for RDNA2 ds_dec_rtn_u32.

    Computes effective address, reads previous 32-bit dword, applies wrapping
    decrement logic, writes updated 32-bit dword back, and returns (old_val, new_val).
    """
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~3  # dword aligned (drop low 2 bits)
    old_val = struct.unpack_from("<I", lds_mem, aligned_addr)[0]
    new_val = ref_ds_dec_u32(old_val, vdata0)
    struct.pack_into("<I", lds_mem, aligned_addr, new_val)
    return old_val, new_val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Supported Data Share (DS) atomic opcodes."""
    DS_DEC_RTN_U32 = "ds_dec_rtn_u32"


class RegType(Enum):
    """Register file classes."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}{self.index} "
                f"(expected 0 <= index < {max_limit})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class DSDecRtnInstruction:
    """Symbolic AST node for ds_dec_rtn_u32 atomic instruction."""
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_DEC_RTN_U32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise TypeError("vdst operand must be a VGPR")
        if self.vaddr.reg_type != RegType.VGPR:
            raise TypeError("vaddr operand must be a VGPR")
        if self.vdata0.reg_type != RegType.VGPR:
            raise TypeError("vdata0 operand must be a VGPR")
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"offset out of 16-bit range [0, 65535]: {self.offset}")

    def disasm(self) -> str:
        """Format canonical assembly mnemonic and operand string."""
        text = f"{self.opcode.value} {self.vdst.name}, {self.vaddr.name}, {self.vdata0.name}"
        if self.offset != 0:
            text += f" offset:{self.offset}"
        if self.gds:
            text += " gds"
        return text


# ==============================================================================
# Deterministic Scalar Interpreter & Machine State
# ==============================================================================

class ScalarRegisterFile:
    """Models a 256-register 32-bit scalar VGPR file."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * DEFAULT_NUM_VGPRS

    def get_u32(self, reg_idx: int) -> int:
        return self.regs[reg_idx] & MASK32

    def set_u32(self, reg_idx: int, val: int) -> None:
        self.regs[reg_idx] = val & MASK32


class ScalarMachineState:
    """Architectural state for scalar single-lane emulation."""

    def __init__(self, lds_dwords: int = LDS_DEFAULT_SIZE_DWORDS) -> None:
        self.rf = ScalarRegisterFile()
        self.sgprs = [0] * DEFAULT_NUM_SGPRS
        self.lds = LDSMemory(lds_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.rf.get_u32(index)

    def set_vgpr(self, index: int, val: int) -> None:
        self.rf.set_u32(index, val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_dec_rtn_u32."""

    def step(self, instr: DSDecRtnInstruction, state: ScalarMachineState) -> Tuple[int, int]:
        """Execute a single ds_dec_rtn_u32 instruction deterministically on scalar state.

        Snapshot operands before updating destination to ensure deterministic register handling.
        """
        vaddr = state.get_vgpr(instr.vaddr.index)
        vdata0 = state.get_vgpr(instr.vdata0.index)
        old_val, new_val = state.lds.atomic_dec_rtn_u32(
            vaddr=vaddr,
            vdata0=vdata0,
            offset=instr.offset,
        )
        state.set_vgpr(instr.vdst.index, old_val)
        return old_val, new_val


# ==============================================================================
# SIMD Multi-Lane State and Divergence Interpreter
# ==============================================================================

class SIMDState:
    """SIMD vector machine state with multi-lane VGPR files and shared LDS."""

    def __init__(
        self,
        num_lanes: int = 32,
        exec_mask: int = MASK32,
        lds_dwords: int = LDS_DEFAULT_SIZE_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = (exec_mask & MASK32) if num_lanes == 32 else (exec_mask & MASK64)
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.lds = LDSMemory(lds_dwords)

    def is_lane_active(self, lane: int) -> bool:
        if 0 <= lane < self.num_lanes:
            return bool((self.exec_mask >> lane) & 1)
        return False

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = u32(val)


class SIMDInterpreter:
    """SIMD multi-lane execution interpreter handling divergence and EXEC masking."""

    def step(
        self,
        instr: DSDecRtnInstruction,
        state: SIMDState,
    ) -> List[Optional[Tuple[int, int]]]:
        """Execute ds_dec_rtn_u32 across all lanes in the wave.

        Lanes execute according to exec_mask. Inactive lanes do not touch LDS memory
        and their destination VGPRs are not modified.
        Contending active lanes targeting the same address are serialized deterministically
        in lane order (lane 0, lane 1, etc.).
        """
        results: List[Optional[Tuple[int, int]]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            vaddr = state.get_lane_vgpr(lane, instr.vaddr.index)
            vdata0 = state.get_lane_vgpr(lane, instr.vdata0.index)
            old_val, new_val = state.lds.atomic_dec_rtn_u32(
                vaddr=vaddr,
                vdata0=vdata0,
                offset=instr.offset,
            )
            state.set_lane_vgpr(lane, instr.vdst.index, old_val)
            results.append((old_val, new_val))
        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Evaluation
# ==============================================================================

@dataclass(frozen=True)
class DecRtnStatusFlags:
    """Condition codes and arithmetic status flags for ds_dec_rtn_u32."""
    zero: bool
    sign: bool
    wrapped: bool
    borrow: bool
    unaligned: bool
    address_wrapped: bool
    old_is_zero: bool


def evaluate_dec_rtn_status_flags(
    vaddr: int,
    offset: int,
    old_val: int,
    vdata0: int,
    new_val: int,
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES,
) -> DecRtnStatusFlags:
    """Evaluate condition codes, arithmetic flags, and carry/borrow generation.

    - zero: new memory value == 0
    - sign: bit 31 of new memory value is set
    - wrapped: operation wrapped to vdata0 (i.e. old_val == 0 or old_val > vdata0)
    - borrow: decrement generated a borrow in unsigned arithmetic (old_val == 0)
    - unaligned: effective address has non-zero low 2 bits
    - address_wrapped: effective byte address wrapped around LDS memory boundary
    - old_is_zero: old memory value prior to decrement was 0
    """
    raw_addr = vaddr + (offset & OFFSET16_MASK)
    address_wrapped = raw_addr >= lds_size_bytes
    eff_addr = raw_addr & (lds_size_bytes - 1)
    unaligned = (eff_addr & 3) != 0

    old_u = u32(old_val)
    bound_u = u32(vdata0)
    new_u = u32(new_val)

    zero = (new_u == 0)
    sign = bool(new_u & 0x80000000)
    wrapped = (old_u == 0 or old_u > bound_u)
    borrow = (old_u == 0)
    old_is_zero = (old_u == 0)

    return DecRtnStatusFlags(
        zero=zero,
        sign=sign,
        wrapped=wrapped,
        borrow=borrow,
        unaligned=unaligned,
        address_wrapped=address_wrapped,
        old_is_zero=old_is_zero,
    )


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)
        self.lds_bytes = bytearray(LDS_DEFAULT_SIZE_BYTES)

    def test_normal_decrement_within_bounds(self) -> None:
        # Initial 5, bound 10 -> decrements to 4, returns old 5
        self.mem.write_dword(10, 5)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=40, vdata0=10, offset=0)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_val, 4)
        self.assertEqual(self.mem.read_dword(10), 4)

        # Initial 1, bound 10 -> decrements to 0, returns old 1
        self.mem.write_dword(10, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=40, vdata0=10, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.mem.read_dword(10), 0)

    def test_wrapping_at_zero(self) -> None:
        # Initial 0, bound 10 -> wraps to 10, returns old 0
        self.mem.write_dword(20, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=80, vdata0=10, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(20), 10)

    def test_wrapping_when_strictly_greater_than_bound(self) -> None:
        # Initial 15, bound 10 -> strictly greater, wraps to 10, returns old 15
        self.mem.write_dword(30, 15)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=120, vdata0=10, offset=0)
        self.assertEqual(old_val, 15)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(30), 10)

        # Initial MASK32, bound 10 -> wraps to 10
        self.mem.write_dword(30, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=120, vdata0=10, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(30), 10)

    def test_decrement_when_equal_to_bound(self) -> None:
        # Initial 10, bound 10 -> not 0 and not > 10, so normal decrement to 9
        self.mem.write_dword(40, 10)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=160, vdata0=10, offset=0)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 9)
        self.assertEqual(self.mem.read_dword(40), 9)

    def test_boundary_bound_zero(self) -> None:
        # bound = 0:
        # if old == 0: wraps to 0
        self.mem.write_dword(5, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        # if old == 1: 1 > 0, wraps to 0
        self.mem.write_dword(5, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)

        # if old == MASK32: MASK32 > 0, wraps to 0
        self.mem.write_dword(5, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, 0)

    def test_boundary_bound_max_u32(self) -> None:
        # bound = MASK32:
        # if old == 0: wraps to MASK32
        self.mem.write_dword(6, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=24, vdata0=MASK32, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK32)

        # if old == MASK32: not > MASK32 and not 0, decrements to MASK32 - 1
        self.mem.write_dword(6, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=24, vdata0=MASK32, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, MASK32 - 1)

    def test_countdown_cycle_simulation(self) -> None:
        # Countdown loop from 3 down to 0 and wrapping back to 3 with vdata0 = 3
        bound = 3
        idx = 15
        self.mem.write_dword(idx, bound)

        expected_sequence = [
            (3, 2),  # step 1
            (2, 1),  # step 2
            (1, 0),  # step 3
            (0, 3),  # step 4: wraps from 0 back to bound
            (3, 2),  # step 5: cycles again
        ]
        for expected_old, expected_new in expected_sequence:
            old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=idx * 4, vdata0=bound)
            self.assertEqual(old_val, expected_old)
            self.assertEqual(new_val, expected_new)
            self.assertEqual(self.mem.read_dword(idx), expected_new)

    def test_address_offset_and_dword_alignment(self) -> None:
        # vaddr = 0x1000, offset = 16 -> eff = 0x1010 -> dword index = 0x404 & 1023 = 4
        self.mem.write_dword(4, 50)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=0x1000, vdata0=100, offset=16)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.mem.read_dword(4), 49)

        # Low 2 bits of address ignored (unaligned byte offset 7 -> 0x1007 -> aligned to 0x1004)
        self.mem.write_dword(1, 25)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=0x1000, vdata0=100, offset=7)
        # 0x1007 >> 2 = 0x401 -> dword index 1
        self.assertEqual(old_val, 25)
        self.assertEqual(new_val, 24)
        self.assertEqual(self.mem.read_dword(1), 24)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x200
        struct.pack_into("<I", self.lds_bytes, addr, 77)
        old_val, new_val = oracle_ds_dec_rtn_u32(self.lds_bytes, vaddr=addr, vdata0=50)
        # 77 > 50 -> wraps to 50
        self.assertEqual(old_val, 77)
        self.assertEqual(new_val, 50)
        self.assertEqual(struct.unpack_from("<I", self.lds_bytes, addr)[0], 50)


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_DEC_RTN_U32.value, "ds_dec_rtn_u32")

    def test_register_operands(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        v255 = RegisterOperand(255, RegType.VGPR)
        s0 = RegisterOperand(0, RegType.SGPR)
        s105 = RegisterOperand(105, RegType.SGPR)

        self.assertEqual(v0.name, "v0")
        self.assertEqual(v255.name, "v255")
        self.assertEqual(s0.name, "s0")
        self.assertEqual(s105.name, "s105")

    def test_ast_construction_and_disassembly(self) -> None:
        instr_basic = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )
        self.assertEqual(instr_basic.disasm(), "ds_dec_rtn_u32 v5, v1, v2")

        instr_offset = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=16,
        )
        self.assertEqual(instr_offset.disasm(), "ds_dec_rtn_u32 v5, v1, v2 offset:16")

        instr_gds = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=8,
            gds=True,
        )
        self.assertEqual(instr_gds.disasm(), "ds_dec_rtn_u32 v5, v1, v2 offset:8 gds")

    def test_ast_immutability_and_equality(self) -> None:
        instr1 = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(4),
            vdata0=RegisterOperand(5),
            offset=64,
        )
        instr2 = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(4),
            vdata0=RegisterOperand(5),
            offset=64,
        )
        self.assertEqual(instr1, instr2)
        self.assertEqual(hash(instr1), hash(instr2))

    def test_ast_validation_and_type_checks(self) -> None:
        # Register index out of bounds
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

        # RegType SGPR in vdst, vaddr, or vdata0
        with self.assertRaises(TypeError):
            DSDecRtnInstruction(
                vdst=RegisterOperand(0, RegType.SGPR),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
            )
        with self.assertRaises(TypeError):
            DSDecRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1, RegType.SGPR),
                vdata0=RegisterOperand(2),
            )
        with self.assertRaises(TypeError):
            DSDecRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2, RegType.SGPR),
            )

        # Offset out of range
        with self.assertRaises(ValueError):
            DSDecRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
                offset=-1,
            )
        with self.assertRaises(ValueError):
            DSDecRtnInstruction(
                vdst=RegisterOperand(0),
                vaddr=RegisterOperand(1),
                vdata0=RegisterOperand(2),
                offset=65536,
            )


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState(lds_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_execute_basic(self) -> None:
        # Pre-seed LDS dword 10 with 50
        self.state.lds.write_dword(10, 50)
        # v1 = byte address 40 (dword 10), v2 = wrap threshold 100, vdst = v3
        self.state.set_vgpr(1, 40)
        self.state.set_vgpr(2, 100)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.state.get_vgpr(3), 50)
        self.assertEqual(self.state.lds.read_dword(10), 49)

    def test_scalar_execute_wrap_on_zero(self) -> None:
        self.state.lds.write_dword(10, 0)
        self.state.set_vgpr(1, 40)
        self.state.set_vgpr(2, 100)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 100)
        self.assertEqual(self.state.get_vgpr(3), 0)
        self.assertEqual(self.state.lds.read_dword(10), 100)

    def test_scalar_determinism_across_identical_runs(self) -> None:
        state1 = ScalarMachineState(lds_dwords=1024)
        state2 = ScalarMachineState(lds_dwords=1024)
        interp1 = ScalarInterpreter()
        interp2 = ScalarInterpreter()

        for s in (state1, state2):
            s.lds.write_dword(5, 20)
            s.set_vgpr(1, 20)
            s.set_vgpr(2, 50)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(4),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        res1 = interp1.step(instr, state1)
        res2 = interp2.step(instr, state2)

        self.assertEqual(res1, res2)
        self.assertEqual(state1.rf.regs, state2.rf.regs)
        self.assertEqual(state1.lds.data, state2.lds.data)

    def test_scalar_loop_countdown(self) -> None:
        bound = 3
        self.state.lds.write_dword(8, bound)
        self.state.set_vgpr(1, 32)
        self.state.set_vgpr(2, bound)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        expected_steps = [(3, 2), (2, 1), (1, 0), (0, 3), (3, 2)]
        for exp_old, exp_new in expected_steps:
            old_val, new_val = self.interp.step(instr, self.state)
            self.assertEqual(old_val, exp_old)
            self.assertEqual(new_val, exp_new)
            self.assertEqual(self.state.get_vgpr(5), exp_old)
            self.assertEqual(self.state.lds.read_dword(8), exp_new)

    def test_scalar_untouched_state_preserved(self) -> None:
        # Pre-seed adjacent registers and memory locations
        self.state.set_vgpr(0, 0x11111111)
        self.state.set_vgpr(1, 40)
        self.state.set_vgpr(2, 50)
        self.state.set_vgpr(4, 0x44444444)
        self.state.lds.write_dword(9, 0xAAAAAAAA)
        self.state.lds.write_dword(10, 10)
        self.state.lds.write_dword(11, 0xBBBBBBBB)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        self.interp.step(instr, self.state)

        # Target register v3 modified
        self.assertEqual(self.state.get_vgpr(3), 10)
        # Adjacent registers preserved
        self.assertEqual(self.state.get_vgpr(0), 0x11111111)
        self.assertEqual(self.state.get_vgpr(1), 40)
        self.assertEqual(self.state.get_vgpr(2), 50)
        self.assertEqual(self.state.get_vgpr(4), 0x44444444)
        # Adjacent memory preserved
        self.assertEqual(self.state.lds.read_dword(9), 0xAAAAAAAA)
        self.assertEqual(self.state.lds.read_dword(10), 9)
        self.assertEqual(self.state.lds.read_dword(11), 0xBBBBBBBB)


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=MASK32, lds_dwords=1024)
        simd = SIMDInterpreter()

        # Each lane has distinct address and initial memory
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(lane, 10 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 100)
            state.set_lane_vgpr(lane, 3, 0)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], (10 + lane, 9 + lane))
            self.assertEqual(state.get_lane_vgpr(lane, 3), 10 + lane)
            self.assertEqual(state.lds.read_dword(lane), 9 + lane)

    def test_simd_wave64_full_exec_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=MASK64, lds_dwords=1024)
        simd = SIMDInterpreter()

        for lane in range(64):
            addr = lane * 4
            state.lds.write_dword(lane, 100 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 200)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(4),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertEqual(results[lane], (100 + lane, 99 + lane))
            self.assertEqual(state.get_lane_vgpr(lane, 4), 100 + lane)
            self.assertEqual(state.lds.read_dword(lane), 99 + lane)

    def test_simd_wave32_divergent_checkerboard_mask(self) -> None:
        # Even lanes active, odd lanes inactive
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask, lds_dwords=1024)
        simd = SIMDInterpreter()
        sentinel = 0xDEADBEEF

        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(lane, 20 + lane)
            state.set_lane_vgpr(lane, 1, addr)
            state.set_lane_vgpr(lane, 2, 50)
            state.set_lane_vgpr(lane, 5, sentinel)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], (20 + lane, 19 + lane))
                self.assertEqual(state.get_lane_vgpr(lane, 5), 20 + lane)
                self.assertEqual(state.lds.read_dword(lane), 19 + lane)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 5), sentinel)
                self.assertEqual(state.lds.read_dword(lane), 20 + lane)

    def test_simd_all_lanes_inactive(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0, lds_dwords=1024)
        simd = SIMDInterpreter()

        for lane in range(32):
            state.lds.write_dword(lane, 42)
            state.set_lane_vgpr(lane, 1, lane * 4)
            state.set_lane_vgpr(lane, 2, 50)
            state.set_lane_vgpr(lane, 3, 0x12345678)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_lane_vgpr(lane, 3), 0x12345678)
            self.assertEqual(state.lds.read_dword(lane), 42)

    def test_simd_single_active_lane(self) -> None:
        target_lane = 7
        state = SIMDState(num_lanes=32, exec_mask=(1 << target_lane), lds_dwords=1024)
        simd = SIMDInterpreter()

        for lane in range(32):
            state.lds.write_dword(lane, 10)
            state.set_lane_vgpr(lane, 1, lane * 4)
            state.set_lane_vgpr(lane, 2, 50)
            state.set_lane_vgpr(lane, 6, 0)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(6),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        for lane in range(32):
            if lane == target_lane:
                self.assertEqual(results[lane], (10, 9))
                self.assertEqual(state.get_lane_vgpr(lane, 6), 10)
                self.assertEqual(state.lds.read_dword(lane), 9)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_lane_vgpr(lane, 6), 0)
                self.assertEqual(state.lds.read_dword(lane), 10)

    def test_simd_multi_lane_contention_same_address(self) -> None:
        # 4 active lanes contending on the same LDS address (0x80 -> dword 32)
        state = SIMDState(num_lanes=32, exec_mask=0b1111, lds_dwords=1024)
        simd = SIMDInterpreter()

        target_dword = 32
        target_addr = target_dword * 4
        bound = 3
        # Initial memory is 3
        state.lds.write_dword(target_dword, bound)

        for lane in range(4):
            state.set_lane_vgpr(lane, 1, target_addr)
            state.set_lane_vgpr(lane, 2, bound)
            state.set_lane_vgpr(lane, 3, 0)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        results = simd.step(instr, state)
        # Lane 0: 3 -> 2 (returns 3)
        self.assertEqual(results[0], (3, 2))
        self.assertEqual(state.get_lane_vgpr(0, 3), 3)

        # Lane 1: 2 -> 1 (returns 2)
        self.assertEqual(results[1], (2, 1))
        self.assertEqual(state.get_lane_vgpr(1, 3), 2)

        # Lane 2: 1 -> 0 (returns 1)
        self.assertEqual(results[2], (1, 0))
        self.assertEqual(state.get_lane_vgpr(2, 3), 1)

        # Lane 3: 0 -> wraps to bound 3 (returns 0)
        self.assertEqual(results[3], (0, 3))
        self.assertEqual(state.get_lane_vgpr(3, 3), 0)

        # Final memory in LDS is 3
        self.assertEqual(state.lds.read_dword(target_dword), 3)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_vdst_with_vaddr(self) -> None:
        # vdst = v1, vaddr = v1, vdata0 = v2
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=1024)

        target_dword = 10
        target_byte = target_dword * 4  # 40
        state.lds.write_dword(target_dword, 75)

        state.set_vgpr(1, target_byte)  # vaddr
        state.set_vgpr(2, 100)          # vdata0

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(1),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = interp.step(instr, state)
        self.assertEqual(old_val, 75)
        self.assertEqual(new_val, 74)
        # v1 overwritten with old_val 75
        self.assertEqual(state.get_vgpr(1), 75)
        # Memory at address 40 updated to 74
        self.assertEqual(state.lds.read_dword(target_dword), 74)
        # v2 preserved
        self.assertEqual(state.get_vgpr(2), 100)

    def test_alias_vdst_with_vdata0_preserves_bound_snapshot(self) -> None:
        # vdst = v2, vaddr = v1, vdata0 = v2
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=1024)

        target_dword = 15
        target_byte = target_dword * 4
        # Memory is 0 -> will wrap to vdata0
        state.lds.write_dword(target_dword, 0)

        state.set_vgpr(1, target_byte)
        orig_bound = 88
        state.set_vgpr(2, orig_bound)  # vdata0

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(2),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = interp.step(instr, state)
        # Old memory was 0, wraps to 88
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, orig_bound)
        # v2 overwritten with old_val (0)
        self.assertEqual(state.get_vgpr(2), 0)
        # Memory received orig_bound (88) despite v2 now being 0
        self.assertEqual(state.lds.read_dword(target_dword), orig_bound)
        # vaddr preserved
        self.assertEqual(state.get_vgpr(1), target_byte)

    def test_alias_vaddr_with_vdata0(self) -> None:
        # vdst = v3, vaddr = v1, vdata0 = v1
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=1024)

        # Let v1 = 64 (byte address 64 -> dword 16; and bound = 64)
        val_shared = 64
        dword_idx = val_shared // 4
        state.lds.write_dword(dword_idx, 10)
        state.set_vgpr(1, val_shared)
        state.set_vgpr(3, 0)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(1),
            offset=0,
        )

        old_val, new_val = interp.step(instr, state)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 9)
        self.assertEqual(state.get_vgpr(3), 10)
        self.assertEqual(state.lds.read_dword(dword_idx), 9)
        # v1 operand register preserved
        self.assertEqual(state.get_vgpr(1), val_shared)

    def test_triple_aliasing_all_operands_same_register(self) -> None:
        # vdst = vaddr = vdata0 = v1
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=1024)

        # Register holds 128 (byte address 128 -> dword 32; and bound = 128)
        initial_reg = 128
        dword_idx = initial_reg // 4
        # Memory holds 5 (5 <= 128, so decrements to 4)
        state.lds.write_dword(dword_idx, 5)
        state.set_vgpr(1, initial_reg)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(1),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(1),
            offset=0,
        )

        old_val, new_val = interp.step(instr, state)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_val, 4)
        # v1 overwritten with old_val (5)
        self.assertEqual(state.get_vgpr(1), 5)
        # Memory at dword 32 holds 4
        self.assertEqual(state.lds.read_dword(dword_idx), 4)

    def test_triple_aliasing_wrap_from_zero(self) -> None:
        # vdst = vaddr = vdata0 = v1 where memory holds 0
        interp = ScalarInterpreter()
        state = ScalarMachineState(lds_dwords=1024)

        initial_reg = 100
        dword_idx = initial_reg // 4
        state.lds.write_dword(dword_idx, 0)
        state.set_vgpr(1, initial_reg)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(1),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(1),
            offset=0,
        )

        old_val, new_val = interp.step(instr, state)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 100)
        # v1 gets old_val (0)
        self.assertEqual(state.get_vgpr(1), 0)
        # Memory gets bound (100)
        self.assertEqual(state.lds.read_dword(dword_idx), 100)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_zero_and_sign_flags(self) -> None:
        flags_zero = evaluate_dec_rtn_status_flags(0x100, 0, old_val=1, vdata0=10, new_val=0)
        self.assertTrue(flags_zero.zero)
        self.assertFalse(flags_zero.sign)

        flags_sign = evaluate_dec_rtn_status_flags(
            0x100, 0, old_val=0x80000001, vdata0=MASK32, new_val=0x80000000
        )
        self.assertFalse(flags_sign.zero)
        self.assertTrue(flags_sign.sign)

        flags_pos = evaluate_dec_rtn_status_flags(0x100, 0, old_val=10, vdata0=20, new_val=9)
        self.assertFalse(flags_pos.zero)
        self.assertFalse(flags_pos.sign)

    def test_wrapped_and_borrow_flags(self) -> None:
        # Wrap from 0 generates borrow
        flags_borrow = evaluate_dec_rtn_status_flags(
            0x100, 0, old_val=0, vdata0=10, new_val=10
        )
        self.assertTrue(flags_borrow.wrapped)
        self.assertTrue(flags_borrow.borrow)
        self.assertTrue(flags_borrow.old_is_zero)

        # Wrap from > bound does NOT generate borrow
        flags_wrap_gt = evaluate_dec_rtn_status_flags(
            0x100, 0, old_val=25, vdata0=10, new_val=10
        )
        self.assertTrue(flags_wrap_gt.wrapped)
        self.assertFalse(flags_wrap_gt.borrow)
        self.assertFalse(flags_wrap_gt.old_is_zero)

        # Normal decrement does not wrap or borrow
        flags_norm = evaluate_dec_rtn_status_flags(
            0x100, 0, old_val=5, vdata0=10, new_val=4
        )
        self.assertFalse(flags_norm.wrapped)
        self.assertFalse(flags_norm.borrow)
        self.assertFalse(flags_norm.old_is_zero)

    def test_unaligned_address_detection(self) -> None:
        aligned_flag = evaluate_dec_rtn_status_flags(
            vaddr=0x200, offset=0, old_val=5, vdata0=10, new_val=4
        )
        self.assertFalse(aligned_flag.unaligned)

        unaligned_flag1 = evaluate_dec_rtn_status_flags(
            vaddr=0x201, offset=0, old_val=5, vdata0=10, new_val=4
        )
        self.assertTrue(unaligned_flag1.unaligned)

        unaligned_flag2 = evaluate_dec_rtn_status_flags(
            vaddr=0x200, offset=2, old_val=5, vdata0=10, new_val=4
        )
        self.assertTrue(unaligned_flag2.unaligned)

        aligned_with_offset = evaluate_dec_rtn_status_flags(
            vaddr=0x200, offset=4, old_val=5, vdata0=10, new_val=4
        )
        self.assertFalse(aligned_with_offset.unaligned)

    def test_address_wraparound_detection(self) -> None:
        nowrap = evaluate_dec_rtn_status_flags(
            vaddr=0x100, offset=0, old_val=5, vdata0=10, new_val=4
        )
        self.assertFalse(nowrap.address_wrapped)

        # Raw address >= 64 KiB (65536)
        wrap_boundary = evaluate_dec_rtn_status_flags(
            vaddr=65532, offset=4, old_val=5, vdata0=10, new_val=4
        )
        self.assertTrue(wrap_boundary.address_wrapped)

        wrap_overflow = evaluate_dec_rtn_status_flags(
            vaddr=65530, offset=12, old_val=5, vdata0=10, new_val=4
        )
        self.assertTrue(wrap_overflow.address_wrapped)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.state = ScalarMachineState(lds_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_snan_preservation_without_quietening(self) -> None:
        # Signaling NaN: exponent=0xFF, fraction != 0, quiet bit 22 (0x00400000) is 0
        snan = 0x7F800002
        self.assertEqual(snan & 0x00400000, 0)

        dword_idx = 10
        self.state.lds.write_dword(dword_idx, snan)
        self.state.set_vgpr(1, dword_idx * 4)
        self.state.set_vgpr(2, MASK32)  # Bound is max u32, so normal decrement occurs

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        # Old val preserved with quiet bit clear
        self.assertEqual(old_val, snan)
        self.assertEqual(old_val & 0x00400000, 0)
        self.assertEqual(self.state.get_vgpr(3), snan)

        # New val is integer decremented: 0x7F800001 (still sNaN, quiet bit still clear)
        self.assertEqual(new_val, 0x7F800001)
        self.assertEqual(new_val & 0x00400000, 0)
        self.assertEqual(self.state.lds.read_dword(dword_idx), 0x7F800001)

    def test_qnan_payload_preservation(self) -> None:
        # Quiet NaN with specific payload: 0x7FC01234
        qnan = 0x7FC01234
        dword_idx = 12
        self.state.lds.write_dword(dword_idx, qnan)
        self.state.set_vgpr(1, dword_idx * 4)
        self.state.set_vgpr(2, MASK32)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
            offset=0,
        )

        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, qnan)
        self.assertEqual(new_val, 0x7FC01233)
        self.assertEqual(self.state.get_vgpr(3), qnan)
        self.assertEqual(self.state.lds.read_dword(dword_idx), 0x7FC01233)

    def test_signed_zeros_distinct_behavior(self) -> None:
        # +0.0 is 0x00000000 -> triggers wrap on zero
        dword_pos = 14
        self.state.lds.write_dword(dword_pos, POS_ZERO_F32)
        self.state.set_vgpr(1, dword_pos * 4)
        self.state.set_vgpr(2, 50)

        instr1 = DSDecRtnInstruction(
            vdst=RegisterOperand(3),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old1, new1 = self.interp.step(instr1, self.state)
        self.assertEqual(old1, POS_ZERO_F32)
        self.assertEqual(new1, 50)
        self.assertEqual(self.state.get_vgpr(3), POS_ZERO_F32)

        # -0.0 is 0x80000000 (2147483648) -> not equal to 0; strictly > 50 -> wraps to 50
        dword_neg = 15
        self.state.lds.write_dword(dword_neg, NEG_ZERO_F32)
        self.state.set_vgpr(1, dword_neg * 4)
        self.state.set_vgpr(2, 50)

        instr2 = DSDecRtnInstruction(
            vdst=RegisterOperand(4),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old2, new2 = self.interp.step(instr2, self.state)
        self.assertEqual(old2, NEG_ZERO_F32)
        self.assertEqual(new2, 50)
        self.assertEqual(self.state.get_vgpr(4), NEG_ZERO_F32)

    def test_subnormals_without_flush_to_zero(self) -> None:
        # Min subnormal 0x00000001
        dword_idx = 16
        self.state.lds.write_dword(dword_idx, MIN_SUBNORMAL_F32)
        self.state.set_vgpr(1, dword_idx * 4)
        self.state.set_vgpr(2, 100)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(5),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old_val, new_val = self.interp.step(instr, self.state)
        self.assertEqual(old_val, MIN_SUBNORMAL_F32)
        # Decrements by 1 to 0
        self.assertEqual(new_val, 0)
        self.assertEqual(self.state.get_vgpr(5), MIN_SUBNORMAL_F32)

        # Max subnormal 0x007FFFFF decrements to 0x007FFFFE with bound >= MAX_SUBNORMAL_F32
        self.state.lds.write_dword(dword_idx, MAX_SUBNORMAL_F32)
        self.state.set_vgpr(2, MASK32)
        old_max, new_max = self.interp.step(instr, self.state)
        self.assertEqual(old_max, MAX_SUBNORMAL_F32)
        self.assertEqual(new_max, MAX_SUBNORMAL_F32 - 1)

    def test_signed_infinities_preservation(self) -> None:
        dword_idx = 18
        self.state.lds.write_dword(dword_idx, POS_INF_F32)
        self.state.set_vgpr(1, dword_idx * 4)
        self.state.set_vgpr(2, MASK32)

        instr = DSDecRtnInstruction(
            vdst=RegisterOperand(6),
            vaddr=RegisterOperand(1),
            vdata0=RegisterOperand(2),
        )
        old_inf, new_inf = self.interp.step(instr, self.state)
        self.assertEqual(old_inf, POS_INF_F32)
        self.assertEqual(new_inf, POS_INF_F32 - 1)
        self.assertEqual(self.state.get_vgpr(6), POS_INF_F32)


if __name__ == "__main__":
    unittest.main()
