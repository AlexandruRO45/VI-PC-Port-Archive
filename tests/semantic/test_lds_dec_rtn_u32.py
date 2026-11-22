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


if __name__ == "__main__":
    unittest.main()
