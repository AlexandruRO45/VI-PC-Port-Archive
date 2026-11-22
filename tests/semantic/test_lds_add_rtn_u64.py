from __future__ import annotations

"""Reference oracle and architectural model for ds_add_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Unsigned Addition with Return:
Performs a 64-bit atomic addition on LDS memory and returns the original (pre-operation) value.

Syntax:
    ds_add_rtn_u64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Operands:
    vdst: VGPR pair (v[d:d+1]) receiving the pre-operation 64-bit qword.
    vaddr: VGPR (v[a]) containing the byte address into LDS.
    vdata0: VGPR pair (v[s:s+1]) or 64-bit immediate to add.
    offset: 16-bit unsigned immediate byte offset.

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    dword_index_hi = (dword_index + 1) & (lds_dwords - 1)
    old_val_64 = (LDS[dword_index_hi] << 32) | LDS[dword_index]
    new_val_64 = (old_val_64 + vdata0) & 0xFFFFFFFFFFFFFFFF
    LDS[dword_index] = new_val_64 & 0xFFFFFFFF
    LDS[dword_index_hi] = (new_val_64 >> 32) & 0xFFFFFFFF
    vdst = old_val_64
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit qword into (low_u32, high_u32)."""
    val = u64(val)
    return val & MASK_32, (val >> 32) & MASK_32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (low_u32, high_u32) into a 64-bit qword."""
    return ((hi & MASK_32) << 32) | (lo & MASK_32)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_add_rtn_u64(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_add_rtn_u64.

    Returns:
        (old_val, new_val)
    """
    old_val = u64(current_mem)
    add_val = u64(data)
    new_val = u64(old_val + add_val)
    return old_val, new_val


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = u32(val)

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword (2 consecutive dwords) from LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        idx_hi = (idx_lo + 1) & (self.capacity_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword (2 consecutive dwords) to LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        idx_hi = (idx_lo + 1) & (self.capacity_dwords - 1)
        lo, hi = split_u64_to_u32s(val)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_add_u64(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic 64-bit unsigned addition with return on LDS.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_val = oracle_ds_add_rtn_u64(curr, data)
        self.write_qword(eff_addr, new_val)
        return old_val, new_val


class TestLdsAddRtnU64ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_add_identity_zero(self) -> None:
        """Adding 0 to 64-bit integer leaves value unchanged and returns original."""
        addr = 0x40
        init_val = 0x1234_5678_9ABC_DEF0
        self.lds.write_qword(addr, init_val)
        old_val, new_val = self.lds.atomic_add_u64(addr, 0)
        self.assertEqual(old_val, init_val)
        self.assertEqual(new_val, init_val)
        self.assertEqual(self.lds.read_qword(addr), init_val)

    def test_add_no_carry(self) -> None:
        """Direct 64-bit addition without carry across dword boundary."""
        addr = 0x80
        val1 = 0x1111_2222_1111_2222
        val2 = 0x2222_1111_2222_1111
        expected = val1 + val2
        self.lds.write_qword(addr, val1)
        old_val, new_val = self.lds.atomic_add_u64(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(new_val, expected)
        self.assertEqual(self.lds.read_qword(addr), expected)

    def test_add_carry_across_dword_boundary(self) -> None:
        """Carry propagates from low 32 bits into high 32 bits: 0x0_FFFF_FFFF + 1."""
        addr = 0x100
        val1 = 0x0000_0000_FFFF_FFFF
        add_val = 0x0000_0000_0000_0001
        expected = 0x0000_0001_0000_0000
        self.lds.write_qword(addr, val1)
        old_val, new_val = self.lds.atomic_add_u64(addr, add_val)
        self.assertEqual(old_val, val1)
        self.assertEqual(new_val, expected)
        self.assertEqual(self.lds.read_qword(addr), expected)

    def test_add_overflow_modulo_2_pow_64(self) -> None:
        """Unsigned 64-bit addition overflow wraps strictly modulo 2^64: MASK_64 + 1 == 0."""
        addr = 0x120
        self.lds.write_qword(addr, MASK_64)
        old_val, new_val = self.lds.atomic_add_u64(addr, 1)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.lds.read_qword(addr), 0)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x28
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1028)

        eff_masked = calc_lds_effective_address(base, 0x1_0008)
        self.assertEqual(eff_masked, 0x1008)

        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

    def test_dword_alignment_truncation(self) -> None:
        """Low 2 bits of effective address are ignored for dword indexing."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx2, 0x100)
        self.assertEqual(idx3, 0x100)

    def test_bank_distribution(self) -> None:
        """32 consecutive dwords map to 32 distinct LDS banks [0..31]."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))
        self.assertEqual(calc_lds_bank(32 * 4), 0)

    def test_lds_capacity_wrap(self) -> None:
        """LDS access wraps modulo LDS capacity (default 64 KiB = 16384 dwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + LDS_SIZE_BYTES
        idx_base = calc_lds_dword_index(addr_base)
        idx_wrapped = calc_lds_dword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    VGPR = auto()
    VGPR_PAIR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(f"VGPR index out of range [0, {DEFAULT_NUM_VGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class VGPRPair:
    base: int

    def __post_init__(self) -> None:
        if not (0 <= self.base < DEFAULT_NUM_VGPRS - 1):
            raise ValueError(
                f"VGPRPair base out of range [0, {DEFAULT_NUM_VGPRS - 2}]: {self.base}"
            )

    @property
    def lo(self) -> int:
        return self.base

    @property
    def hi(self) -> int:
        return self.base + 1

    def __str__(self) -> str:
        return f"v[{self.base}:{self.base + 1}]"


@dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_SGPRS):
            raise ValueError(f"SGPR index out of range [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class Immediate64:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u64(self.value))

    def __str__(self) -> str:
        return f"0x{self.value:X}"


@dataclass(frozen=True)
class DSAddRtnU64Instruction:
    dst: VGPRPair
    addr: VGPR
    data0: Union[VGPRPair, Immediate64]
    offset: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"LDS offset out of 16-bit range [0, 65535]: {self.offset}")
        if not isinstance(self.dst, VGPRPair):
            raise TypeError(f"dst must be a VGPRPair, got {type(self.dst)}")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"addr must be a VGPR, got {type(self.addr)}")
        if not isinstance(self.data0, (VGPRPair, Immediate64)):
            raise TypeError(f"data0 must be VGPRPair or Immediate64, got {type(self.data0)}")

    def disasm(self) -> str:
        tokens = ["ds_add_rtn_u64", str(self.dst), str(self.addr), str(self.data0)]
        extra = []
        if self.offset != 0:
            extra.append(f"offset:{self.offset}")
        if self.gds:
            extra.append("gds")
        core = f"{tokens[0]} {tokens[1]}, {tokens[2]}, {tokens[3]}"
        return f"{core} {' '.join(extra)}".rstrip()


class ASTExpr64:
    """Base class for symbolic typed expressions in RDNA2 64-bit arithmetic."""

    def simplify(self) -> ASTExpr64:
        raise NotImplementedError

    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol64(ASTExpr64):
    name: str

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in evaluation environment")
        return u64(env[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Constant64(ASTExpr64):
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u64(self.value))

    def simplify(self) -> ASTExpr64:
        return self

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.value

    def __str__(self) -> str:
        return f"0x{self.value:016X}"


@dataclass(frozen=True)
class Add64Expr(ASTExpr64):
    left: ASTExpr64
    right: ASTExpr64

    def simplify(self) -> ASTExpr64:
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        # Constant folding
        if isinstance(s_left, Constant64) and isinstance(s_right, Constant64):
            return Constant64(s_left.value + s_right.value)

        # Identity: X + 0 -> X
        if isinstance(s_right, Constant64) and s_right.value == 0:
            return s_left
        if isinstance(s_left, Constant64) and s_left.value == 0:
            return s_right

        return Add64Expr(s_left, s_right)

    def evaluate(self, env: Dict[str, int]) -> int:
        return u64(self.left.evaluate(env) + self.right.evaluate(env))

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_valid_creation_and_disassembly(self) -> None:
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0),
            addr=VGPR(4),
            data0=VGPRPair(2),
            offset=16,
        )
        self.assertEqual(instr.disasm(), "ds_add_rtn_u64 v[0:1], v4, v[2:3] offset:16")

        instr_imm = DSAddRtnU64Instruction(
            dst=VGPRPair(10),
            addr=VGPR(20),
            data0=Immediate64(0x1234_5678_9ABC_DEF0),
            offset=0,
            gds=True,
        )
        self.assertEqual(
            instr_imm.disasm(), "ds_add_rtn_u64 v[10:11], v20, 0x123456789ABCDEF0 gds"
        )

    def test_instruction_validation_vgpr_pair_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPRPair(255)
        with self.assertRaises(ValueError):
            VGPRPair(-1)

    def test_instruction_validation_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSAddRtnU64Instruction(
                dst=VGPRPair(0), addr=VGPR(1), data0=VGPRPair(2), offset=65536
            )
        with self.assertRaises(ValueError):
            DSAddRtnU64Instruction(
                dst=VGPRPair(0), addr=VGPR(1), data0=VGPRPair(2), offset=-1
            )

    def test_symbolic_constant_folding(self) -> None:
        expr = Add64Expr(
            Constant64(0x0000_0005_0000_0000), Constant64(0x0000_0002_0000_0000)
        )
        simplified = expr.simplify()
        self.assertIsInstance(simplified, Constant64)
        self.assertEqual(simplified.evaluate({}), 0x0000_0007_0000_0000)

    def test_symbolic_identity_zero(self) -> None:
        x = Symbol64("X")
        expr = Add64Expr(x, Constant64(0))
        simplified = expr.simplify()
        self.assertEqual(simplified, x)

    def test_symbolic_evaluation_env(self) -> None:
        x = Symbol64("A")
        y = Symbol64("B")
        expr = Add64Expr(x, y)
        env = {"A": 100, "B": 25}
        self.assertEqual(expr.evaluate(env), 125)


# ============================================================================
# Section 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

class ScalarMachineState:
    """Register and memory state for scalar single-thread RDNA2 execution."""

    def __init__(self, lds_capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSModel = LDSModel(capacity_dwords=lds_capacity_dwords)

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx]

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = u32(val)

    def get_vgpr_qword(self, base_idx: int) -> int:
        lo = self.get_vgpr(base_idx)
        hi = self.get_vgpr(base_idx + 1)
        return combine_u32s_to_u64(lo, hi)

    def set_vgpr_qword(self, base_idx: int, val: int) -> None:
        lo, hi = split_u64_to_u32s(val)
        self.set_vgpr(base_idx, lo)
        self.set_vgpr(base_idx + 1, hi)

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx]

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_add_rtn_u64."""

    def resolve_qword_operand(
        self,
        op: Union[VGPRPair, Immediate64],
        state: ScalarMachineState,
    ) -> int:
        if isinstance(op, VGPRPair):
            return state.get_vgpr_qword(op.base)
        elif isinstance(op, Immediate64):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSAddRtnU64Instruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int]:
        """Execute a single ds_add_rtn_u64 instruction deterministically.

        Latching behavior:
        Both vaddr and vdata0 operands are latched before memory is updated or
        the return value is written to vdst register pair.
        """
        vaddr = state.get_vgpr(instr.addr.index)
        data = self.resolve_qword_operand(instr.data0, state)

        # Atomic memory update
        old_val, new_val = state.lds.atomic_add_u64(vaddr, data, offset=instr.offset)

        # Writeback to destination 64-bit register pair
        state.set_vgpr_qword(instr.dst.base, old_val)

        return old_val, new_val

    def execute_sequence(
        self,
        instructions: List[DSAddRtnU64Instruction],
        state: ScalarMachineState,
    ) -> List[Tuple[int, int]]:
        """Execute an ordered sequence of instructions."""
        return [self.step(instr, state) for instr in instructions]


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_scalar_basic_execution(self) -> None:
        addr = 0x80
        init_mem = 0x1000_0000_2000_0000
        add_val = 0x2000_0000_3000_0000
        self.state.lds.write_qword(addr, init_mem)
        self.state.set_vgpr(4, addr)
        self.state.set_vgpr_qword(2, add_val)

        # ds_add_rtn_u64 v[0:1], v4, v[2:3]
        instr = DSAddRtnU64Instruction(dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2))
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(new_val, init_mem + add_val)
        self.assertEqual(self.state.get_vgpr_qword(0), init_mem)
        self.assertEqual(self.state.lds.read_qword(addr), init_mem + add_val)

    def test_scalar_with_immediate_operand(self) -> None:
        addr = 0x100
        init_mem = 0x0000_0002_0000_0000
        imm = 0x0000_0000_0000_0005
        self.state.lds.write_qword(addr, init_mem)
        self.state.set_vgpr(5, addr)

        # ds_add_rtn_u64 v[6:7], v5, imm
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(6), addr=VGPR(5), data0=Immediate64(imm)
        )
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(new_val, init_mem + imm)
        self.assertEqual(self.state.get_vgpr_qword(6), init_mem)
        self.assertEqual(self.state.lds.read_qword(addr), init_mem + imm)

    def test_scalar_offset_applied(self) -> None:
        base_addr = 0x200
        offset = 0x18
        eff_addr = base_addr + offset
        init_mem = 0x1111_1111_1111_1111
        self.state.lds.write_qword(eff_addr, init_mem)
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr_qword(2, 0x2222_2222_2222_2222)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(8), addr=VGPR(1), data0=VGPRPair(2), offset=offset
        )
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(new_val, init_mem + 0x2222_2222_2222_2222)
        self.assertEqual(self.state.get_vgpr_qword(8), init_mem)
        self.assertEqual(self.state.lds.read_qword(eff_addr), new_val)

    def test_scalar_deterministic_instruction_sequence(self) -> None:
        addr = 0x300
        self.state.lds.write_qword(addr, 100)
        self.state.set_vgpr(1, addr)

        seq = [
            DSAddRtnU64Instruction(
                dst=VGPRPair(10), addr=VGPR(1), data0=Immediate64(100)
            ),
            DSAddRtnU64Instruction(
                dst=VGPRPair(12), addr=VGPR(1), data0=Immediate64(200)
            ),
            DSAddRtnU64Instruction(
                dst=VGPRPair(14), addr=VGPR(1), data0=Immediate64(300)
            ),
        ]
        results = self.interp.execute_sequence(seq, self.state)

        self.assertEqual(results[0], (100, 200))
        self.assertEqual(results[1], (200, 400))
        self.assertEqual(results[2], (400, 700))

        self.assertEqual(self.state.get_vgpr_qword(10), 100)
        self.assertEqual(self.state.get_vgpr_qword(12), 200)
        self.assertEqual(self.state.get_vgpr_qword(14), 400)
        self.assertEqual(self.state.lds.read_qword(addr), 700)

    def test_scalar_matches_oracle_across_patterns(self) -> None:
        addr = 0x400
        patterns = [
            (0, 0),
            (10, 5),
            (0x0_FFFF_FFFF, 1),
            (MASK_64, 1),
            (0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321),
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000),
        ]
        for initial, add_v in patterns:
            self.state.lds.write_qword(addr, initial)
            self.state.set_vgpr(1, addr)
            self.state.set_vgpr_qword(2, add_v)
            instr = DSAddRtnU64Instruction(
                dst=VGPRPair(0), addr=VGPR(1), data0=VGPRPair(2)
            )
            old_interp, new_interp = self.interp.step(instr, self.state)
            old_oracle, new_oracle = oracle_ds_add_rtn_u64(initial, add_v)
            self.assertEqual(old_interp, old_oracle)
            self.assertEqual(new_interp, new_oracle)




if __name__ == "__main__":
    unittest.main()
