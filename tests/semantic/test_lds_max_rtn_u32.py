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
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 LDS Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
MASK32: int = 0xFFFFFFFF            # 32-bit unsigned mask
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
        vaddr = state.get_vgpr(instr.addr.index)
        vdata = self.resolve_operand(instr.data, state)

        eff_addr = calc_lds_effective_address(vaddr, instr.offset)
        idx = calc_lds_dword_index(eff_addr, state.lds.capacity_dwords)
        curr_mem = state.lds.memory[idx]

        old_val, new_mem, modified = oracle_ds_max_rtn_u32(curr_mem, vdata)
        state.lds.memory[idx] = new_mem
        state.set_vgpr(instr.dst.index, old_val)

        return old_val, new_mem, modified


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


if __name__ == "__main__":
    unittest.main()
