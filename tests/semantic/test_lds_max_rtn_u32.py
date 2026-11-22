"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_rtn_u32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) atomic unsigned 32-bit maximum with return:
    ds_max_rtn_u32 vdst, vaddr, vdata [offset:offset16]

RDNA2 LDS / Unsigned 32-bit Integer ISA Semantics:
  1. Local Data Share Architecture:
     LDS is an on-chip, low-latency scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed, but memory is organized into 32-bit dwords across 32 banks.
     LDS accesses are dword-aligned in hardware: bits [1:0] of the effective byte address are dropped.
     Addresses wrap modulo the allocated power-of-two LDS capacity.
  2. Addressing and Effective Offset:
     effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
     dword_index = (effective_byte_address >> 2) & (capacity_dwords - 1)
  3. Atomic Unsigned Maximum & Return Value:
     old_val = LDS[dword_index]
     new_val = max(old_val, vdata)  # unsigned 32-bit integer comparison
     LDS[dword_index] = new_val
     vdst = old_val  # previous memory value returned to destination VGPR
  4. Unsigned Integer Ordering:
     Operands are treated as unsigned 32-bit integers in [0, 2^32 - 1].
     Strictly: 0 < 1 < 0x7FFFFFFF < 0x80000000 < 0xFFFFFFFF.

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

# RDNA2 LDS Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
MASK32: int = 0xFFFFFFFF            # 32-bit unsigned mask
MASK64: int = 0xFFFFFFFFFFFFFFFF    # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask
SIGN_BIT32: int = 0x80000000        # 32-bit sign bit / MSB


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS with 16-bit offset."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0-31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_max_u32(a: int, b: int) -> int:
    """Reference pure unsigned 32-bit maximum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua >= ub else ub


def oracle_ds_max_rtn_u32(
    current_mem: int,
    vdata: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_max_rtn_u32.

    Returns:
        (returned_old_val, new_mem_val, modified)
    """
    old_val = current_mem & MASK32
    val = vdata & MASK32
    new_val = oracle_max_u32(old_val, val)
    return old_val, new_val, (new_val != old_val)


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory: List[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = val & MASK32

    def atomic_max_rtn_u32(
        self,
        vaddr: int,
        vdata: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic unsigned maximum on LDS dword returning previous value.

        Returns:
            (old_val, new_val, modified)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val, modified = oracle_ds_max_rtn_u32(curr, vdata)
        self.memory[idx] = new_val
        return old_val, new_val, modified


# ---------------------------------------------------------------------------
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(str, enum.Enum):
    DS_MAX_RTN_U32 = "ds_max_rtn_u32"


class RegKind(str, enum.Enum):
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Typed AST representation of a GPU hardware register."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"{self.kind.value}{self.index}"


def VGPR(index: int) -> Register:
    return Register(RegKind.VGPR, index)


def SGPR(index: int) -> Register:
    return Register(RegKind.SGPR, index)


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Typed AST representation of a 32-bit immediate literal."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", self.val & MASK32)

    def __str__(self) -> str:
        return f"0x{self.val:X}" if self.val > 9 else str(self.val)


Operand = Union[Register, Immediate]


@dataclasses.dataclass
class DSMaxRtnU32Instruction:
    """AST node for ds_max_rtn_u32 instruction:

    ds_max_rtn_u32 vdst, vaddr, vdata [offset:offset16]
    """
    opcode: Opcode
    dst: Register
    addr: Register
    data: Operand
    offset: int = 0

    def __post_init__(self) -> None:
        if self.opcode != Opcode.DS_MAX_RTN_U32:
            raise ValueError(f"Unsupported opcode for ds_max_rtn_u32: {self.opcode}")
        if self.dst.kind != RegKind.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst}")
        if self.addr.kind != RegKind.VGPR:
            raise ValueError(f"Address must be a VGPR, got {self.addr}")
        if isinstance(self.data, Register) and self.data.kind != RegKind.VGPR:
            raise ValueError(f"Register data operand must be a VGPR, got {self.data}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset out of range [0, 65535]: {self.offset}")

    @classmethod
    def parse(cls, asm: str) -> DSMaxRtnU32Instruction:
        """Parse an assembly statement into a typed DSMaxRtnU32Instruction AST node."""
        tokens = asm.strip().replace(",", " ").split()
        if not tokens:
            raise ValueError("Empty assembly string")
        if tokens[0] != Opcode.DS_MAX_RTN_U32.value:
            raise ValueError(f"Expected opcode '{Opcode.DS_MAX_RTN_U32.value}', got '{tokens[0]}'")

        if len(tokens) < 4:
            raise ValueError(f"Too few operands for ds_max_rtn_u32: {asm}")

        def parse_reg(tok: str) -> Register:
            if tok.startswith("v"):
                return VGPR(int(tok[1:]))
            if tok.startswith("s"):
                return SGPR(int(tok[1:]))
            raise ValueError(f"Invalid register operand: {tok}")

        dst = parse_reg(tokens[1])
        addr = parse_reg(tokens[2])

        # Parse data operand: either VGPR or immediate
        data_tok = tokens[3]
        if data_tok.startswith("v") or data_tok.startswith("s"):
            data: Operand = parse_reg(data_tok)
        else:
            data = Immediate(int(data_tok, 0))

        offset = 0
        for tok in tokens[4:]:
            if tok.startswith("offset:"):
                val_str = tok[len("offset:"):]
                offset = int(val_str, 0)
            else:
                raise ValueError(f"Unknown modifier or trailing token: {tok}")

        return cls(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=dst,
            addr=addr,
            data=data,
            offset=offset,
        )

    def __str__(self) -> str:
        s = f"{self.opcode.value} {self.dst}, {self.addr}, {self.data}"
        if self.offset != 0:
            s += f" offset:{self.offset}"
        return s


# ---------------------------------------------------------------------------
# Phase 3: Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState:
    """Architectural register and memory state for a single scalar lane."""

    def __init__(self, lds: Optional[LDSModel] = None) -> None:
        self.lds: LDSModel = lds if lds is not None else LDSModel()
        self.vgprs: Dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: Dict[int, int] = {i: 0 for i in range(106)}

    def get_vgpr(self, index: int) -> int:
        return self.vgprs.get(index, 0) & MASK32

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic instruction interpreter for scalar ds_max_rtn_u32 execution."""

    def resolve_operand(self, op: Operand, state: ScalarMachineState) -> int:
        if isinstance(op, Register):
            return state.get_vgpr(op.index)
        elif isinstance(op, Immediate):
            return op.val
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSMaxRtnU32Instruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int, bool]:
        """Execute a single ds_max_rtn_u32 instruction in the scalar machine state.

        Returns:
            (old_mem_val, new_mem_val, modified)
        """
        # Hardware latches source operands before writeback
        vaddr = state.get_vgpr(instr.addr.index)
        vdata = self.resolve_operand(instr.data, state)

        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        idx = calc_lds_dword_index(eff_addr, state.lds.capacity_dwords)
        curr_mem = state.lds.memory[idx]

        old_val, new_mem, modified = oracle_ds_max_rtn_u32(curr_mem, vdata)
        state.lds.memory[idx] = new_mem
        state.set_vgpr(instr.dst.index, old_val)

        return old_val, new_mem, modified


# ---------------------------------------------------------------------------
# Phase 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ---------------------------------------------------------------------------

class SIMDState:
    """Architectural wavefront state supporting Wave32 and Wave64 execution."""

    def __init__(
        self,
        num_lanes: int = 32,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSModel] = None,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.lane_mask = (1 << num_lanes) - 1
        self.exec_mask = self.lane_mask if exec_mask is None else (exec_mask & self.lane_mask)
        self.lds: LDSModel = lds if lds is not None else LDSModel()
        self.vgprs: List[Dict[int, int]] = [{i: 0 for i in range(256)} for _ in range(num_lanes)]
        self.sgprs: Dict[int, int] = {i: 0 for i in range(106)}

    def is_lane_active(self, lane: int) -> bool:
        """Check if a lane is enabled by the active wavefront EXEC mask."""
        if not (0 <= lane < self.num_lanes):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, index: int) -> int:
        return self.vgprs[lane].get(index, 0) & MASK32

    def set_vgpr(self, lane: int, index: int, val: int) -> None:
        self.vgprs[lane][index] = val & MASK32


class SIMDInterpreter:
    """SIMD wavefront instruction interpreter with bank collision serialization."""

    def resolve_lane_operand(self, op: Operand, lane: int, state: SIMDState) -> int:
        if isinstance(op, Register):
            return state.get_vgpr(lane, op.index)
        elif isinstance(op, Immediate):
            return op.val
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSMaxRtnU32Instruction,
        state: SIMDState,
    ) -> List[Tuple[int, int, int]]:
        """Execute ds_max_rtn_u32 across all lanes under the wavefront EXEC mask.

        Returns:
            List of (lane_id, old_val, new_val) for all active lanes that executed.
        """
        results: List[Tuple[int, int, int]] = []

        # Active lanes serialize in ascending lane index order on LDS collisions
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue

            vaddr = state.get_vgpr(lane, instr.addr.index)
            vdata = self.resolve_lane_operand(instr.data, lane, state)

            eff_addr = calc_lds_effective_address(vaddr, instr.offset)
            idx = calc_lds_dword_index(eff_addr, state.lds.capacity_dwords)
            curr_mem = state.lds.memory[idx]

            old_val, new_mem, _ = oracle_ds_max_rtn_u32(curr_mem, vdata)
            state.lds.memory[idx] = new_mem
            state.set_vgpr(lane, instr.dst.index, old_val)

            results.append((lane, old_val, new_mem))

        return results


# ---------------------------------------------------------------------------
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StatusFlags:
    """Architectural status flags and condition codes for atomic max."""
    zero: bool
    sign: bool
    modified: bool
    borrow: bool
    bank_conflict: bool


def compute_status_flags(
    old_val: int,
    new_val: int,
    input_val: int,
    has_conflict: bool = False,
) -> StatusFlags:
    """Compute condition codes and status flags for ds_max_rtn_u32 operation.

    - Zero Flag (Z): True if resulting memory dword is 0.
    - Sign Flag (N): True if MSB (bit 31) of resulting memory dword is 1.
    - Modified Flag (M): True if input_val caused memory to update (input_val > old_val).
    - Borrow Flag (B): True if unsigned comparison old_val < input_val (borrow on subtraction).
    - Bank Conflict: True if 2 or more active wavefront lanes collide on the same LDS bank.
    """
    res = new_val & MASK32
    u_old = old_val & MASK32
    u_in = input_val & MASK32
    return StatusFlags(
        zero=(res == 0),
        sign=bool(res & SIGN_BIT32),
        modified=(res != u_old),
        borrow=(u_old < u_in),
        bank_conflict=has_conflict,
    )


def detect_simd_bank_conflicts(
    instr: DSMaxRtnU32Instruction,
    state: SIMDState,
) -> Set[int]:
    """Detect LDS bank conflicts across active lanes in a wavefront.

    Returns:
        Set of bank indices (0..31) that experience conflicting accesses.
    """
    accessed_banks: Dict[int, List[int]] = {}
    for lane in range(state.num_lanes):
        if not state.is_lane_active(lane):
            continue
        vaddr = state.get_vgpr(lane, instr.addr.index)
        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        bank = calc_lds_bank(eff_addr)
        accessed_banks.setdefault(bank, []).append(lane)

    conflicts: Set[int] = set()
    for bank, lanes in accessed_banks.items():
        if len(lanes) > 1:
            conflicts.add(bank)
    return conflicts


class TestLdsMaxRtnU32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_max_u32_standard(self) -> None:
        self.assertEqual(oracle_max_u32(10, 20), 20)
        self.assertEqual(oracle_max_u32(50, 30), 50)
        self.assertEqual(oracle_max_u32(100, 100), 100)

    def test_oracle_max_u32_unsigned_boundaries(self) -> None:
        # Unsigned 0 is less than 0xFFFFFFFF
        self.assertEqual(oracle_max_u32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(oracle_max_u32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        # 0x80000000 (signed -2147483648) > 0x7FFFFFFF (signed +2147483647) in unsigned comparison
        self.assertEqual(oracle_max_u32(0x80000000, 0x7FFFFFFF), 0x80000000)
        self.assertEqual(oracle_max_u32(0x7FFFFFFF, 0x80000000), 0x80000000)
        # Extremes
        self.assertEqual(oracle_max_u32(0, 0), 0)
        self.assertEqual(oracle_max_u32(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_oracle_ds_max_rtn_u32_return_and_modify(self) -> None:
        old, new, mod = oracle_ds_max_rtn_u32(100, 200)
        self.assertEqual(old, 100)
        self.assertEqual(new, 200)
        self.assertTrue(mod)

        old, new, mod = oracle_ds_max_rtn_u32(500, 300)
        self.assertEqual(old, 500)
        self.assertEqual(new, 500)
        self.assertFalse(mod)

        old, new, mod = oracle_ds_max_rtn_u32(500, 500)
        self.assertEqual(old, 500)
        self.assertEqual(new, 500)
        self.assertFalse(mod)

    def test_effective_address_calculation(self) -> None:
        # vaddr + offset
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        # 16-bit offset masking
        eff = calc_lds_effective_address(0x1000, 0x10040)
        self.assertEqual(eff, 0x1040)
        # 32-bit modulo wrapping
        eff = calc_lds_effective_address(0xFFFFFFFF, 0x0001)
        self.assertEqual(eff, 0x0000)

    def test_dword_alignment_and_bit_truncation(self) -> None:
        # LDS drops low 2 bits
        idx0 = calc_lds_dword_index(0x00)
        idx1 = calc_lds_dword_index(0x01)
        idx2 = calc_lds_dword_index(0x02)
        idx3 = calc_lds_dword_index(0x03)
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 0)
        self.assertEqual(idx3, 0)

        idx4 = calc_lds_dword_index(0x04)
        self.assertEqual(idx4, 1)

    def test_lds_capacity_wrapping(self) -> None:
        capacity = 1024  # 1024 dwords = 4096 bytes
        eff_addr = 4096
        self.assertEqual(calc_lds_dword_index(eff_addr, capacity), 0)
        eff_addr2 = 4096 + 12
        self.assertEqual(calc_lds_dword_index(eff_addr2, capacity), 3)

    def test_bank_indexing(self) -> None:
        self.assertEqual(calc_lds_bank(0x00), 0)
        self.assertEqual(calc_lds_bank(0x04), 1)
        self.assertEqual(calc_lds_bank(0x7C), 31)
        self.assertEqual(calc_lds_bank(0x80), 0)

    def test_lds_model_atomic_max_rtn_u32(self) -> None:
        lds = LDSModel(capacity_dwords=16384)
        addr = 0x200
        lds.write_dword(addr, 1000)
        self.assertEqual(lds.read_dword(addr), 1000)

        # Atomic max with smaller value -> old returned, memory unchanged
        old_val, new_val, mod = lds.atomic_max_rtn_u32(addr, 500)
        self.assertEqual(old_val, 1000)
        self.assertEqual(new_val, 1000)
        self.assertFalse(mod)
        self.assertEqual(lds.read_dword(addr), 1000)

        # Atomic max with larger value -> old returned, memory updated
        old_val2, new_val2, mod2 = lds.atomic_max_rtn_u32(addr, 2500)
        self.assertEqual(old_val2, 1000)
        self.assertEqual(new_val2, 2500)
        self.assertTrue(mod2)
        self.assertEqual(lds.read_dword(addr), 2500)

        # Atomic max with immediate offset
        old_val3, new_val3, mod3 = lds.atomic_max_rtn_u32(addr, 5000, offset=4)
        # offset=4 means addr 0x204 (dword index 0x81) which was initially 0
        self.assertEqual(old_val3, 0)
        self.assertEqual(new_val3, 5000)
        self.assertTrue(mod3)
        self.assertEqual(lds.read_dword(0x204), 5000)
        # original dword 0x200 remains 2500
        self.assertEqual(lds.read_dword(addr), 2500)


class TestLdsMaxRtnU32AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_attributes(self) -> None:
        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
            offset=16,
        )
        self.assertEqual(inst.opcode, Opcode.DS_MAX_RTN_U32)
        self.assertEqual(inst.dst, VGPR(0))
        self.assertEqual(inst.addr, VGPR(1))
        self.assertEqual(inst.data, VGPR(2))
        self.assertEqual(inst.offset, 16)
        self.assertEqual(str(inst), "ds_max_rtn_u32 v0, v1, v2 offset:16")

    def test_ast_parse_valid_registers(self) -> None:
        inst = DSMaxRtnU32Instruction.parse("ds_max_rtn_u32 v5, v12, v20")
        self.assertEqual(inst.opcode, Opcode.DS_MAX_RTN_U32)
        self.assertEqual(inst.dst, VGPR(5))
        self.assertEqual(inst.addr, VGPR(12))
        self.assertEqual(inst.data, VGPR(20))
        self.assertEqual(inst.offset, 0)
        self.assertEqual(str(inst), "ds_max_rtn_u32 v5, v12, v20")

    def test_ast_parse_with_offset(self) -> None:
        inst = DSMaxRtnU32Instruction.parse("ds_max_rtn_u32 v0, v1, v2 offset:64")
        self.assertEqual(inst.offset, 64)
        inst_hex = DSMaxRtnU32Instruction.parse("ds_max_rtn_u32 v10, v11, v12 offset:0x100")
        self.assertEqual(inst_hex.offset, 256)

    def test_ast_parse_with_immediate_data(self) -> None:
        inst = DSMaxRtnU32Instruction.parse("ds_max_rtn_u32 v0, v1, 42")
        self.assertEqual(inst.data, Immediate(42))
        inst2 = DSMaxRtnU32Instruction.parse("ds_max_rtn_u32 v0, v1, 0xFFFFFFFF")
        self.assertEqual(inst2.data, Immediate(0xFFFFFFFF))

    def test_ast_validation_errors(self) -> None:
        # Destination must be VGPR
        with self.assertRaises(ValueError):
            DSMaxRtnU32Instruction(
                opcode=Opcode.DS_MAX_RTN_U32,
                dst=SGPR(0),
                addr=VGPR(1),
                data=VGPR(2),
            )
        # Address must be VGPR
        with self.assertRaises(ValueError):
            DSMaxRtnU32Instruction(
                opcode=Opcode.DS_MAX_RTN_U32,
                dst=VGPR(0),
                addr=SGPR(1),
                data=VGPR(2),
            )
        # Register data operand cannot be SGPR
        with self.assertRaises(ValueError):
            DSMaxRtnU32Instruction(
                opcode=Opcode.DS_MAX_RTN_U32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=SGPR(2),
            )
        # Offset out of range
        with self.assertRaises(ValueError):
            DSMaxRtnU32Instruction(
                opcode=Opcode.DS_MAX_RTN_U32,
                dst=VGPR(0),
                addr=VGPR(1),
                data=VGPR(2),
                offset=70000,
            )


class TestLdsMaxRtnU32ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_basic_vgpr_operand(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x100  # dword index 0x40
        state.lds.write_dword(addr, 50)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 200)

        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 200)
        self.assertTrue(mod)
        # Check destination register received old value
        self.assertEqual(state.get_vgpr(0), 50)
        # Check memory updated to new max
        self.assertEqual(state.lds.read_dword(addr), 200)

    def test_scalar_interpreter_immediate_operand(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x240
        state.lds.write_dword(addr, 500)
        state.set_vgpr(5, addr)

        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(10),
            addr=VGPR(5),
            data=Immediate(300),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 500)
        self.assertEqual(new_val, 500)
        self.assertFalse(mod)
        self.assertEqual(state.get_vgpr(10), 500)
        self.assertEqual(state.lds.read_dword(addr), 500)

    def test_scalar_interpreter_sequential_updates(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x40
        state.lds.write_dword(addr, 100)
        state.set_vgpr(1, addr)

        # Step 1: max(100, 50) -> memory 100, dst receives 100
        inst1 = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), Immediate(50))
        old, new, mod = interp.step(inst1, state)
        self.assertEqual(old, 100)
        self.assertEqual(new, 100)
        self.assertFalse(mod)
        self.assertEqual(state.get_vgpr(0), 100)
        self.assertEqual(state.lds.read_dword(addr), 100)

        # Step 2: max(100, 300) -> memory 300, dst receives 100
        inst2 = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), Immediate(300))
        old, new, mod = interp.step(inst2, state)
        self.assertEqual(old, 100)
        self.assertEqual(new, 300)
        self.assertTrue(mod)
        self.assertEqual(state.get_vgpr(0), 100)
        self.assertEqual(state.lds.read_dword(addr), 300)

        # Step 3: max(300, 0xFFFFFFFF) -> memory 0xFFFFFFFF, dst receives 300
        inst3 = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), Immediate(0xFFFFFFFF))
        old, new, mod = interp.step(inst3, state)
        self.assertEqual(old, 300)
        self.assertEqual(new, 0xFFFFFFFF)
        self.assertTrue(mod)
        self.assertEqual(state.get_vgpr(0), 300)
        self.assertEqual(state.lds.read_dword(addr), 0xFFFFFFFF)

    def test_scalar_interpreter_with_offset(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        base_addr = 0x80
        offset = 0x14  # effective byte address 0x94 (dword index 0x25)
        state.lds.write_dword(0x94, 777)
        state.set_vgpr(1, base_addr)
        state.set_vgpr(2, 999)

        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(3),
            addr=VGPR(1),
            data=VGPR(2),
            offset=offset,
        )

        old, new, mod = interp.step(inst, state)
        self.assertEqual(old, 777)
        self.assertEqual(new, 999)
        self.assertTrue(mod)
        self.assertEqual(state.get_vgpr(3), 777)
        self.assertEqual(state.lds.read_dword(0x94), 999)
        # Base dword remains untouched
        self.assertEqual(state.lds.read_dword(base_addr), 0)


class TestLdsMaxRtnU32SIMD(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        interp = SIMDInterpreter()

        # Initialize LDS with initial values for 32 dwords
        for i in range(32):
            state.lds.write_dword(i * 4, 100 + i)
            state.set_vgpr(i, 1, i * 4)       # v1: addr
            state.set_vgpr(i, 2, 500 + i)     # v2: data

        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(0),
            addr=VGPR(1),
            data=VGPR(2),
        )

        results = interp.step(inst, state)
        self.assertEqual(len(results), 32)
        for i in range(32):
            # Destination register receives previous value 100 + i
            self.assertEqual(state.get_vgpr(i, 0), 100 + i)
            # Memory updated to max 500 + i
            self.assertEqual(state.lds.read_dword(i * 4), 500 + i)

    def test_wave32_divergence_exec_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        even_mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=even_mask)
        interp = SIMDInterpreter()

        sentinel_vgpr = 0xDEADBEEF
        for i in range(32):
            state.lds.write_dword(i * 4, 1000)
            state.set_vgpr(i, 0, sentinel_vgpr)  # v0: canary
            state.set_vgpr(i, 1, i * 4)          # v1: addr
            state.set_vgpr(i, 2, 2000)           # v2: data

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), VGPR(2))
        results = interp.step(inst, state)

        self.assertEqual(len(results), 16)
        for i in range(32):
            if i % 2 == 0:
                # Active lane: received old value 1000, memory updated to 2000
                self.assertEqual(state.get_vgpr(i, 0), 1000)
                self.assertEqual(state.lds.read_dword(i * 4), 2000)
            else:
                # Inactive lane: VGPR unmodified, memory unmodified
                self.assertEqual(state.get_vgpr(i, 0), sentinel_vgpr)
                self.assertEqual(state.lds.read_dword(i * 4), 1000)

    def test_wave64_full_and_partial_masking(self) -> None:
        # Test 64-lane wavefront
        mask64 = 0x00000000FFFFFFFF  # Lower 32 lanes active, upper 32 inactive
        state = SIMDState(num_lanes=64, exec_mask=mask64)
        interp = SIMDInterpreter()

        sentinel = 0xCAFEBABE
        for i in range(64):
            state.lds.write_dword(i * 4, 50)
            state.set_vgpr(i, 0, sentinel)
            state.set_vgpr(i, 1, i * 4)
            state.set_vgpr(i, 2, 150)

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), VGPR(2))
        results = interp.step(inst, state)

        self.assertEqual(len(results), 32)
        for i in range(64):
            if i < 32:
                self.assertEqual(state.get_vgpr(i, 0), 50)
                self.assertEqual(state.lds.read_dword(i * 4), 150)
            else:
                self.assertEqual(state.get_vgpr(i, 0), sentinel)
                self.assertEqual(state.lds.read_dword(i * 4), 50)

    def test_simd_fully_disabled_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0)
        interp = SIMDInterpreter()

        canary = 0x12345678
        state.lds.write_dword(0x10, canary)
        for i in range(32):
            state.set_vgpr(i, 0, 0x9999)
            state.set_vgpr(i, 1, 0x10)
            state.set_vgpr(i, 2, 0xFFFFFFFF)

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), VGPR(2))
        results = interp.step(inst, state)

        self.assertEqual(len(results), 0)
        self.assertEqual(state.lds.read_dword(0x10), canary)
        for i in range(32):
            self.assertEqual(state.get_vgpr(i, 0), 0x9999)

    def test_simd_bank_collision_serialization_same_address(self) -> None:
        # All 4 lanes target identical LDS dword address
        state = SIMDState(num_lanes=32, exec_mask=0xF)  # Lanes 0..3 active
        interp = SIMDInterpreter()

        target_addr = 0x40
        state.lds.write_dword(target_addr, 10)  # Initial memory = 10

        # Lane 0 proposes 20
        # Lane 1 proposes 15 (less than 20 -> memory remains 20)
        # Lane 2 proposes 50 (greater than 20 -> memory becomes 50)
        # Lane 3 proposes 40 (less than 50 -> memory remains 50)
        data_vals = [20, 15, 50, 40]
        for l in range(4):
            state.set_vgpr(l, 1, target_addr)
            state.set_vgpr(l, 2, data_vals[l])

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), VGPR(2))
        results = interp.step(inst, state)

        self.assertEqual(len(results), 4)
        # Lane 0 sees 10, updates memory to 20
        self.assertEqual(results[0], (0, 10, 20))
        self.assertEqual(state.get_vgpr(0, 0), 10)

        # Lane 1 sees 20, updates memory to max(20, 15)=20
        self.assertEqual(results[1], (1, 20, 20))
        self.assertEqual(state.get_vgpr(1, 0), 20)

        # Lane 2 sees 20, updates memory to max(20, 50)=50
        self.assertEqual(results[2], (2, 20, 50))
        self.assertEqual(state.get_vgpr(2, 0), 20)

        # Lane 3 sees 50, updates memory to max(50, 40)=50
        self.assertEqual(results[3], (3, 50, 50))
        self.assertEqual(state.get_vgpr(3, 0), 50)

        # Final memory must be 50
        self.assertEqual(state.lds.read_dword(target_addr), 50)


# ---------------------------------------------------------------------------
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestLdsMaxRtnU32Aliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and operand preservation."""

    def test_alias_dst_equals_addr(self) -> None:
        """vdst == vaddr: v0 is both destination and address register."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x200
        state.lds.write_dword(addr, 1234)
        state.set_vgpr(0, addr)  # v0 = addr
        state.set_vgpr(1, 5678)  # v1 = data

        # ds_max_rtn_u32 v0, v0, v1
        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(1),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 1234)
        self.assertEqual(new_val, 5678)
        self.assertTrue(mod)
        # v0 receives old value 1234
        self.assertEqual(state.get_vgpr(0), 1234)
        # Memory at 0x200 updated to 5678
        self.assertEqual(state.lds.read_dword(addr), 5678)

    def test_alias_dst_equals_data(self) -> None:
        """vdst == vdata: v1 is both destination and data register."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x300
        state.lds.write_dword(addr, 200)
        state.set_vgpr(0, addr)  # v0 = addr
        state.set_vgpr(1, 400)   # v1 = data

        # ds_max_rtn_u32 v1, v0, v1
        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(1),
            addr=VGPR(0),
            data=VGPR(1),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 200)
        self.assertEqual(new_val, 400)
        self.assertTrue(mod)
        # v1 receives old value 200
        self.assertEqual(state.get_vgpr(1), 200)
        # Memory updated to max(200, 400) = 400
        self.assertEqual(state.lds.read_dword(addr), 400)

    def test_alias_addr_equals_data(self) -> None:
        """vaddr == vdata: v0 is both address and data."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        # Let addr and data be 0x100 (256)
        state.lds.write_dword(0x100, 100)
        state.set_vgpr(0, 0x100)

        # ds_max_rtn_u32 v2, v0, v0
        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(2),
            addr=VGPR(0),
            data=VGPR(0),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_val, 256)
        self.assertTrue(mod)
        self.assertEqual(state.get_vgpr(2), 100)
        # v0 remains preserved as 0x100
        self.assertEqual(state.get_vgpr(0), 0x100)
        self.assertEqual(state.lds.read_dword(0x100), 256)

    def test_alias_tri_dst_addr_data(self) -> None:
        """vdst == vaddr == vdata: v0 is dst, addr, and data."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr_val = 0x80  # 128
        state.lds.write_dword(addr_val, 50)
        state.set_vgpr(0, addr_val)

        # ds_max_rtn_u32 v0, v0, v0
        inst = DSMaxRtnU32Instruction(
            opcode=Opcode.DS_MAX_RTN_U32,
            dst=VGPR(0),
            addr=VGPR(0),
            data=VGPR(0),
        )

        old_val, new_val, mod = interp.step(inst, state)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 128)
        self.assertTrue(mod)
        # v0 overwritten with old_val (50)
        self.assertEqual(state.get_vgpr(0), 50)
        # Memory at 128 updated to 128
        self.assertEqual(state.lds.read_dword(addr_val), 128)

    def test_simd_multi_lane_aliasing(self) -> None:
        """SIMD wave with mixed aliased operands across lanes."""
        state = SIMDState(num_lanes=32, exec_mask=0x7)  # lanes 0, 1, 2
        interp = SIMDInterpreter()

        # Lane 0: vdst == vaddr
        state.lds.write_dword(0x10, 10)
        state.set_vgpr(0, 0, 0x10)  # v0: addr
        state.set_vgpr(0, 1, 30)    # v1: data

        # Lane 1: vdst == vaddr
        state.lds.write_dword(0x20, 40)
        state.set_vgpr(1, 0, 0x20)  # v0: addr
        state.set_vgpr(1, 1, 25)    # v1: data (less than 40)

        # Lane 2: vdst == vaddr
        state.lds.write_dword(0x30, 100)
        state.set_vgpr(2, 0, 0x30)
        state.set_vgpr(2, 1, 200)

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(0), VGPR(1))
        results = interp.step(inst, state)

        self.assertEqual(len(results), 3)
        # Lane 0
        self.assertEqual(results[0], (0, 10, 30))
        self.assertEqual(state.get_vgpr(0, 0), 10)
        self.assertEqual(state.lds.read_dword(0x10), 30)

        # Lane 1
        self.assertEqual(results[1], (1, 40, 40))
        self.assertEqual(state.get_vgpr(1, 0), 40)
        self.assertEqual(state.lds.read_dword(0x20), 40)

        # Lane 2
        self.assertEqual(results[2], (2, 100, 200))
        self.assertEqual(state.get_vgpr(2, 0), 100)
        self.assertEqual(state.lds.read_dword(0x30), 200)


# ---------------------------------------------------------------------------
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Tests
# ---------------------------------------------------------------------------

class TestLdsMaxRtnU32StatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and borrow generation."""

    def test_status_flags_zero_result(self) -> None:
        # Both old and input are 0 -> result is 0
        flags = compute_status_flags(old_val=0, new_val=0, input_val=0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.sign)
        self.assertFalse(flags.modified)
        self.assertFalse(flags.borrow)

    def test_status_flags_nonzero_clears_zero_flag(self) -> None:
        flags = compute_status_flags(old_val=0, new_val=1, input_val=1)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.sign)
        self.assertTrue(flags.modified)
        self.assertTrue(flags.borrow)

    def test_status_flags_sign_flag_msb_set(self) -> None:
        # 0x80000000 has MSB set
        flags = compute_status_flags(old_val=0, new_val=0x80000000, input_val=0x80000000)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.sign)
        self.assertTrue(flags.modified)
        self.assertTrue(flags.borrow)

        # 0x7FFFFFFF has MSB clear
        flags2 = compute_status_flags(old_val=0, new_val=0x7FFFFFFF, input_val=0x7FFFFFFF)
        self.assertFalse(flags2.sign)

    def test_status_flags_borrow_and_modified(self) -> None:
        # Memory had 100, input is 50 -> input <= old_val
        # Result = 100, unmodified, no borrow
        flags_no_mod = compute_status_flags(old_val=100, new_val=100, input_val=50)
        self.assertFalse(flags_no_mod.modified)
        self.assertFalse(flags_no_mod.borrow)

        # Memory had 100, input is 200 -> input > old_val
        # Result = 200, modified, borrow generated
        flags_mod = compute_status_flags(old_val=100, new_val=200, input_val=200)
        self.assertTrue(flags_mod.modified)
        self.assertTrue(flags_mod.borrow)

    def test_simd_bank_conflict_detection(self) -> None:
        # LDS has 32 banks. Dwords 0 and 32 map to bank 0.
        state = SIMDState(num_lanes=32, exec_mask=0x3)  # Lanes 0 and 1 active
        # Lane 0 targets byte address 0x00 (bank 0)
        # Lane 1 targets byte address 0x80 (dword 32 -> bank 0)
        state.set_vgpr(0, 1, 0x00)
        state.set_vgpr(1, 1, 0x80)

        inst = DSMaxRtnU32Instruction(Opcode.DS_MAX_RTN_U32, VGPR(0), VGPR(1), Immediate(10))
        conflicts = detect_simd_bank_conflicts(inst, state)
        self.assertIn(0, conflicts)

        # Non-conflicting: Lane 0 targets bank 0 (0x00), Lane 1 targets bank 1 (0x04)
        state.set_vgpr(1, 1, 0x04)
        conflicts_none = detect_simd_bank_conflicts(inst, state)
        self.assertEqual(len(conflicts_none), 0)


if __name__ == "__main__":
    unittest.main()
