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


# ============================================================================
# Section 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

class SIMDState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64) with shared LDS."""

    def __init__(
        self,
        num_lanes: int = 32,
        exec_mask: int = 0xFFFF_FFFF,
        lds_capacity_dwords: int = LDS_SIZE_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes: int = num_lanes
        self.max_exec_mask: int = (1 << num_lanes) - 1
        self.exec_mask: int = exec_mask & self.max_exec_mask
        self.vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.lds: LDSModel = LDSModel(capacity_dwords=lds_capacity_dwords)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of range: {lane}")
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of range: {lane}")
        if not (0 <= reg_idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {reg_idx}")
        return self.vgprs[lane][reg_idx]

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        if not (0 <= lane < self.num_lanes):
            raise IndexError(f"Lane index out of range: {lane}")
        if not (0 <= reg_idx < DEFAULT_NUM_VGPRS):
            raise IndexError(f"VGPR index out of range: {reg_idx}")
        self.vgprs[lane][reg_idx] = u32(val)

    def get_vgpr_qword(self, lane: int, base_idx: int) -> int:
        lo = self.get_vgpr(lane, base_idx)
        hi = self.get_vgpr(lane, base_idx + 1)
        return combine_u32s_to_u64(lo, hi)

    def set_vgpr_qword(self, lane: int, base_idx: int, val: int) -> None:
        lo, hi = split_u64_to_u32s(val)
        self.set_vgpr(lane, base_idx, lo)
        self.set_vgpr(lane, base_idx + 1, hi)


class SIMDInterpreter:
    """Wavefront SIMD instruction interpreter with divergence, masking, and collision handling."""

    def resolve_lane_qword_operand(
        self,
        op: Union[VGPRPair, Immediate64],
        lane: int,
        state: SIMDState,
    ) -> int:
        if isinstance(op, VGPRPair):
            return state.get_vgpr_qword(lane, op.base)
        elif isinstance(op, Immediate64):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSAddRtnU64Instruction,
        state: SIMDState,
    ) -> List[Optional[Tuple[int, int]]]:
        """Execute a ds_add_rtn_u64 across all lanes in the wavefront.

        Returns a list of length num_lanes where:
        - Inactive lanes return None and modify no registers or LDS memory.
        - Active lanes return (old_val, new_val) and update dst VGPR pair.
        - Multiple active lanes accessing the same LDS qword serialize deterministically
          in ascending lane order.
        """
        active_lane_data: List[Tuple[int, int, int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                vaddr = state.get_vgpr(lane, instr.addr.index)
                data = self.resolve_lane_qword_operand(instr.data0, lane, state)
                eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                active_lane_data.append((lane, eff_addr, data))

        results: List[Optional[Tuple[int, int]]] = [None] * state.num_lanes

        for lane, eff_addr, data in active_lane_data:
            curr_mem = state.lds.read_qword(eff_addr)
            old_val, new_val = oracle_ds_add_rtn_u64(curr_mem, data)
            state.lds.write_qword(eff_addr, new_val)
            results[lane] = (old_val, new_val)

        for lane in range(state.num_lanes):
            res = results[lane]
            if res is not None:
                old_val, _ = res
                state.set_vgpr_qword(lane, instr.dst.base, old_val)

        return results


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD divergence, 32/64-bit EXEC masking, and intra-wave collision."""

    def test_wave32_all_lanes_active_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(32):
            addr = lane * 8
            state.lds.write_qword(addr, 0x1000_0000_0000 + lane)
            state.set_vgpr(lane, 4, addr)
            state.set_vgpr_qword(lane, 2, 0x100)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        for lane in range(32):
            res = results[lane]
            self.assertIsNotNone(res)
            old_val, new_val = res  # type: ignore
            expected_old = 0x1000_0000_0000 + lane
            expected_new = expected_old + 0x100
            self.assertEqual(old_val, expected_old)
            self.assertEqual(new_val, expected_new)
            self.assertEqual(state.get_vgpr_qword(lane, 0), expected_old)
            self.assertEqual(state.lds.read_qword(lane * 8), expected_new)

    def test_wave32_exec_mask_partial_inactivity(self) -> None:
        exec_mask = 0x0000_FFFF
        state = SIMDState(num_lanes=32, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        sentinel_reg = 0xDEAD_BEEF_CAFE_BABE
        for lane in range(32):
            addr = 0x200 + lane * 8
            orig_mem = 0x5000_0000_0000 + lane
            state.lds.write_qword(addr, orig_mem)
            state.set_vgpr_qword(lane, 0, sentinel_reg)
            state.set_vgpr(lane, 4, addr)
            state.set_vgpr_qword(lane, 2, 0x10)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        for lane in range(32):
            addr = 0x200 + lane * 8
            orig_mem = 0x5000_0000_0000 + lane
            if lane < 16:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.get_vgpr_qword(lane, 0), orig_mem)
                self.assertEqual(state.lds.read_qword(addr), orig_mem + 0x10)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr_qword(lane, 0), sentinel_reg)
                self.assertEqual(state.lds.read_qword(addr), orig_mem)

    def test_wave32_exec_mask_zero_all_inactive(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        addr = 0x400
        init_val = 0x9999_8888_7777_6666
        state.lds.write_qword(addr, init_val)
        for lane in range(32):
            state.set_vgpr_qword(lane, 0, 0x1111_1111_1111_1111)
            state.set_vgpr(lane, 4, addr)
            state.set_vgpr_qword(lane, 2, 1)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        self.assertTrue(all(r is None for r in results))
        self.assertEqual(state.lds.read_qword(addr), init_val)
        for lane in range(32):
            self.assertEqual(state.get_vgpr_qword(lane, 0), 0x1111_1111_1111_1111)

    def test_wave64_full_and_partial_masks(self) -> None:
        exec_mask = 0x5555_5555_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x600 + lane * 8
            state.lds.write_qword(addr, 1000)
            state.set_vgpr(lane, 4, addr)
            state.set_vgpr_qword(lane, 2, 10)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        for lane in range(64):
            addr = 0x600 + lane * 8
            if (lane % 2) == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.lds.read_qword(addr), 1010)
                self.assertEqual(state.get_vgpr_qword(lane, 0), 1000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_qword(addr), 1000)

    def test_intra_wave_collision_same_address(self) -> None:
        """Multiple active lanes target the identical LDS qword; atomic serialization occurs."""
        state = SIMDState(num_lanes=32, exec_mask=0x0F)
        interp = SIMDInterpreter()

        target_addr = 0x800
        state.lds.write_qword(target_addr, 1000)

        adds = [10, 20, 30, 40]
        for lane in range(4):
            state.set_vgpr(lane, 4, target_addr)
            state.set_vgpr_qword(lane, 2, adds[lane])

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        # Serialization order:
        # Lane 0: sees 1000, writes 1010
        # Lane 1: sees 1010, writes 1030
        # Lane 2: sees 1030, writes 1060
        # Lane 3: sees 1060, writes 1100
        self.assertEqual(results[0], (1000, 1010))
        self.assertEqual(results[1], (1010, 1030))
        self.assertEqual(results[2], (1030, 1060))
        self.assertEqual(results[3], (1060, 1100))

        self.assertEqual(state.get_vgpr_qword(0, 0), 1000)
        self.assertEqual(state.get_vgpr_qword(1, 0), 1010)
        self.assertEqual(state.get_vgpr_qword(2, 0), 1030)
        self.assertEqual(state.get_vgpr_qword(3, 0), 1060)
        self.assertEqual(state.lds.read_qword(target_addr), 1100)

    def test_intra_wave_collision_with_inactive_lanes(self) -> None:
        exec_mask = (1 << 1) | (1 << 4)
        state = SIMDState(num_lanes=32, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        target_addr = 0x900
        state.lds.write_qword(target_addr, 500)

        for lane in range(32):
            state.set_vgpr(lane, 4, target_addr)
            state.set_vgpr_qword(lane, 2, 0)
        state.set_vgpr_qword(1, 2, 50)
        state.set_vgpr_qword(4, 2, 75)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        interp.step(instr, state)

        self.assertEqual(state.get_vgpr_qword(1, 0), 500)
        self.assertEqual(state.get_vgpr_qword(4, 0), 550)
        self.assertEqual(state.lds.read_qword(target_addr), 625)


# ============================================================================
# Section 5: Register Aliasing and In-Place Operand Preservation
# ============================================================================

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for operand aliasing: vdst == vaddr, vdst == vdata, partial pair overlap."""

    def test_dst_pair_overlaps_addr_scalar(self) -> None:
        """vdst pair overlaps with vaddr (e.g. vaddr is v0, vdst is v[0:1])."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x300
        initial_mem = 0x1111_2222_3333_4444
        add_val = 0x0000_0000_0000_1000

        state.lds.write_qword(addr, initial_mem)
        state.set_vgpr(0, addr)       # v0 = address (will be overwritten by low dword of return)
        state.set_vgpr(1, 0xAAAA)     # v1 = dummy (will be overwritten by high dword of return)
        state.set_vgpr_qword(2, add_val)  # v[2:3] = add_val

        # ds_add_rtn_u64 v[0:1], v0, v[2:3]
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(0), data0=VGPRPair(2)
        )
        old_val, new_val = interp.step(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_val, initial_mem + add_val)
        self.assertEqual(state.get_vgpr_qword(0), initial_mem)
        self.assertEqual(state.lds.read_qword(addr), initial_mem + add_val)
        self.assertEqual(state.get_vgpr_qword(2), add_val)

    def test_dst_pair_overlaps_data_pair_exact_scalar(self) -> None:
        """vdst == vdata0: v[0:1] is both data to add and receives returned old value."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x400
        initial_mem = 0x1000_0000_0000_0000
        add_val = 0x2000_0000_0000_0000

        state.lds.write_qword(addr, initial_mem)
        state.set_vgpr(4, addr)
        state.set_vgpr_qword(0, add_val)  # v[0:1] = add_val

        # ds_add_rtn_u64 v[0:1], v4, v[0:1]
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(0)
        )
        old_val, new_val = interp.step(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_val, initial_mem + add_val)
        self.assertEqual(state.get_vgpr_qword(0), initial_mem)
        self.assertEqual(state.get_vgpr(4), addr)
        self.assertEqual(state.lds.read_qword(addr), initial_mem + add_val)

    def test_dst_pair_partially_overlaps_data_pair_scalar(self) -> None:
        """Partial overlap: vdst is v[2:3] and vdata0 is v[3:4]."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x500
        initial_mem = 0x1111_2222_3333_0000
        v3_val = 0x1234_5678
        v4_val = 0x0000_0001
        data_val = combine_u32s_to_u64(v3_val, v4_val)

        state.lds.write_qword(addr, initial_mem)
        state.set_vgpr(1, addr)
        state.set_vgpr(2, 0)
        state.set_vgpr(3, v3_val)
        state.set_vgpr(4, v4_val)

        # ds_add_rtn_u64 v[2:3], v1, v[3:4]
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(2), addr=VGPR(1), data0=VGPRPair(3)
        )
        old_val, new_val = interp.step(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_val, initial_mem + data_val)
        self.assertEqual(state.get_vgpr_qword(2), initial_mem)
        self.assertEqual(state.lds.read_qword(addr), initial_mem + data_val)
        self.assertEqual(state.get_vgpr(4), v4_val)

    def test_addr_inside_data_pair_scalar(self) -> None:
        """vaddr is v2 and data is v[2:3]. dst is v[0:1]."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        addr = 0x240
        initial_mem = 0x8888_8888_8888_8888
        state.lds.write_qword(addr, initial_mem)
        state.set_vgpr(2, addr)  # v2 = addr (also lo dword of data)
        state.set_vgpr(3, 0)     # v3 = 0 (hi dword of data)
        data_val = addr

        # ds_add_rtn_u64 v[0:1], v2, v[2:3]
        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(2), data0=VGPRPair(2)
        )
        old_val, new_val = interp.step(instr, state)

        self.assertEqual(old_val, initial_mem)
        self.assertEqual(new_val, initial_mem + data_val)
        self.assertEqual(state.get_vgpr_qword(0), initial_mem)
        self.assertEqual(state.get_vgpr(2), addr)
        self.assertEqual(state.get_vgpr(3), 0)

    def test_simd_dst_overlaps_addr_across_lanes(self) -> None:
        """SIMD wavefront where each lane has vdst pair v[0:1] overlapping with vaddr v0."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        for lane in range(32):
            addr = 0x600 + lane * 8
            state.lds.write_qword(addr, 0x1000 + lane)
            state.set_vgpr(lane, 0, addr)       # v0 = addr (also dst lo)
            state.set_vgpr(lane, 1, 0)          # v1 = dst hi
            state.set_vgpr_qword(lane, 2, 10)   # v[2:3] = data

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(0), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        for lane in range(32):
            addr = 0x600 + lane * 8
            expected_old = 0x1000 + lane
            expected_new = expected_old + 10
            self.assertEqual(results[lane], (expected_old, expected_new))
            self.assertEqual(state.get_vgpr_qword(lane, 0), expected_old)
            self.assertEqual(state.lds.read_qword(addr), expected_new)

    def test_inplace_operand_preservation_when_not_dst(self) -> None:
        """All source registers not aliased with vdst remain completely unchanged."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_vgpr(10, 0x400)
        state.set_vgpr_qword(12, 0x1111_2222_3333_4444)
        state.set_vgpr_qword(0, 0)
        state.lds.write_qword(0x400, 0x5555_6666_7777_8888)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(10), data0=VGPRPair(12)
        )
        interp.step(instr, state)

        self.assertEqual(state.get_vgpr(10), 0x400)
        self.assertEqual(state.get_vgpr_qword(12), 0x1111_2222_3333_4444)
        self.assertEqual(state.get_vgpr_qword(0), 0x5555_6666_7777_8888)


# ============================================================================
# Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ============================================================================

@dataclass(frozen=True)
class Add64ConditionFlags:
    """Status flags and condition codes derived from 64-bit unsigned addition."""
    z: bool            # Zero flag: result == 0
    n: bool            # Sign flag: bit 63 set
    carry_out: bool    # 64-bit carry out: True if (a + b) > MASK_64
    carry_dword: bool  # 32-bit dword boundary carry: True if low 32-bit sum > MASK_32


def compute_add64_condition_flags(a_u64: int, b_u64: int) -> Add64ConditionFlags:
    """Compute condition flags for unsigned 64-bit addition (a + b)."""
    a = u64(a_u64)
    b = u64(b_u64)
    res = u64(a + b)

    z = (res == 0)
    n = bool(res & 0x8000_0000_0000_0000)
    carry_out = (a + b) > MASK_64
    carry_dword = ((a & MASK_32) + (b & MASK_32)) > MASK_32

    return Add64ConditionFlags(
        z=z,
        n=n,
        carry_out=carry_out,
        carry_dword=carry_dword,
    )


class TestConditionCodesStatusFlagsAndCarryBorrow(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow relations."""

    def test_ds_instruction_hardware_flag_isolation(self) -> None:
        """DS instructions do not modify hardware scalar condition code (SCC)."""
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_sgpr(0, 1)  # SCC set
        state.lds.write_qword(0x10, 100)
        state.set_vgpr(4, 0x10)
        state.set_vgpr_qword(2, 25)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        interp.step(instr, state)

        # SCC is preserved unchanged
        self.assertEqual(state.get_sgpr(0), 1)

    def test_zero_flag_on_zero_sum(self) -> None:
        """Sum is 0 when 0 + 0 or MASK_64 + 1 wraps to 0."""
        flags_zero = compute_add64_condition_flags(0, 0)
        self.assertTrue(flags_zero.z)
        self.assertFalse(flags_zero.n)
        self.assertFalse(flags_zero.carry_out)

        flags_wrap = compute_add64_condition_flags(MASK_64, 1)
        self.assertTrue(flags_wrap.z)
        self.assertFalse(flags_wrap.n)
        self.assertTrue(flags_wrap.carry_out)

    def test_sign_flag_transitions(self) -> None:
        """Bit 63 set in result sets N flag."""
        flags = compute_add64_condition_flags(0x4000_0000_0000_0000, 0x4000_0000_0000_0000)
        self.assertTrue(flags.n)
        self.assertFalse(flags.z)
        self.assertFalse(flags.carry_out)

    def test_carry_out_detection(self) -> None:
        """Carry out occurs if and only if (A + B) exceeds 2^64 - 1."""
        flags_carry = compute_add64_condition_flags(MASK_64, 5)
        self.assertTrue(flags_carry.carry_out)

        flags_nocarry = compute_add64_condition_flags(100, 200)
        self.assertFalse(flags_nocarry.carry_out)

    def test_dword_boundary_carry(self) -> None:
        """Carry occurs across 32-bit dword boundary when low dword overflows."""
        a = 0x0000_0000_FFFF_FFFF
        b = 0x0000_0000_0000_0001
        flags = compute_add64_condition_flags(a, b)
        self.assertTrue(flags.carry_dword)
        self.assertFalse(flags.carry_out)
        self.assertFalse(flags.z)


# ============================================================================
# Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ============================================================================

def f64_to_u64(val: float) -> int:
    """Pack Python float to IEEE 754 double-precision 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(val: int) -> float:
    """Unpack IEEE 754 double-precision 64-bit unsigned integer to Python float."""
    return struct.unpack("<d", struct.pack("<Q", u64(val)))[0]


# IEEE 754 Double-Precision Architectural Bit Patterns
DOUBLE_POS_ZERO: int = 0x0000_0000_0000_0000
DOUBLE_NEG_ZERO: int = 0x8000_0000_0000_0000
DOUBLE_POS_ONE: int = 0x3FF0_0000_0000_0000
DOUBLE_POS_INF: int = 0x7FF0_0000_0000_0000
DOUBLE_NEG_INF: int = 0xFFF0_0000_0000_0000
DOUBLE_QNAN_DEFAULT: int = 0x7FF8_0000_0000_0000
DOUBLE_SNAN_EXAMPLE: int = 0x7FF0_0000_0000_0001
DOUBLE_MIN_SUBNORMAL: int = 0x0000_0000_0000_0001
DOUBLE_MAX_SUBNORMAL: int = 0x000F_FFFF_FFFF_FFFF
DOUBLE_MIN_NORMAL: int = 0x0010_0000_0000_0000
DOUBLE_MAX_NORMAL: int = 0x7FEF_FFFF_FFFF_FFFF


class TestIEEE754BoundaryExtremesSubnormalsAndNaNs(unittest.TestCase):
    """Unit tests for IEEE 754 double-precision bit patterns manipulated via ds_add_rtn_u64."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_double_subnormal_preservation_without_ftz(self) -> None:
        """Subnormal double bit patterns must never be flushed to zero."""
        addr = 0x100
        subnormals = [
            DOUBLE_MIN_SUBNORMAL,
            DOUBLE_MAX_SUBNORMAL,
            0x0000_0000_1234_5678,
        ]
        for sub in subnormals:
            self.state.lds.write_qword(addr, sub)
            self.state.set_vgpr(4, addr)
            self.state.set_vgpr_qword(2, 0)  # Add 0 (identity read)

            instr = DSAddRtnU64Instruction(
                dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
            )
            old_val, new_val = self.interp.step(instr, self.state)

            self.assertEqual(old_val, sub)
            self.assertEqual(new_val, sub)
            self.assertNotEqual(new_val, 0, "Subnormal was incorrectly flushed to zero")

    def test_double_snan_preserved_without_quietening(self) -> None:
        """Signaling NaNs (sNaN) must NOT be quietened (quiet bit 51 must NOT be set)."""
        addr = 0x110
        snan = DOUBLE_SNAN_EXAMPLE
        self.state.lds.write_qword(addr, snan)
        self.state.set_vgpr(4, addr)
        self.state.set_vgpr_qword(2, 0)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, snan)
        self.assertEqual(new_val, snan)
        # Check quiet bit (bit 51) is still 0
        self.assertEqual(new_val & (1 << 51), 0, "Double sNaN was quietened")

    def test_double_nan_payload_preservation(self) -> None:
        """Arbitrary NaN payload bits are preserved bit-for-bit."""
        addr = 0x120
        nan_with_payload = 0x7FF8_0000_CAFE_BABE
        self.state.lds.write_qword(addr, nan_with_payload)
        self.state.set_vgpr(4, addr)
        self.state.set_vgpr_qword(2, 0)

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, nan_with_payload)
        self.assertEqual(new_val, nan_with_payload)

    def test_double_roundtrip_struct_packing(self) -> None:
        """Exact IEEE 754 round-trip packing and unpacking."""
        test_floats = [0.0, -0.0, 1.0, -1.0, 3.141592653589793, 1e100, -1e-100]
        for f in test_floats:
            u = f64_to_u64(f)
            f_back = u64_to_f64(u)
            if math.isnan(f):
                self.assertTrue(math.isnan(f_back))
            else:
                self.assertEqual(f, f_back)

    def test_simd_wavefront_mixed_ieee754_double_lanes(self) -> None:
        """SIMD wavefront where different lanes hold distinct IEEE 754 64-bit patterns."""
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        special_values = [
            DOUBLE_POS_ZERO,
            DOUBLE_NEG_ZERO,
            DOUBLE_POS_ONE,
            DOUBLE_POS_INF,
            DOUBLE_NEG_INF,
            DOUBLE_QNAN_DEFAULT,
            DOUBLE_SNAN_EXAMPLE,
            DOUBLE_MIN_SUBNORMAL,
            DOUBLE_MAX_SUBNORMAL,
            DOUBLE_MIN_NORMAL,
            DOUBLE_MAX_NORMAL,
        ]

        for lane in range(32):
            val = special_values[lane % len(special_values)]
            addr = 0x300 + lane * 8
            state.lds.write_qword(addr, val)
            state.set_vgpr(lane, 4, addr)
            state.set_vgpr_qword(lane, 2, 0)  # Identity addition

        instr = DSAddRtnU64Instruction(
            dst=VGPRPair(0), addr=VGPR(4), data0=VGPRPair(2)
        )
        results = interp.step(instr, state)

        for lane in range(32):
            orig_val = special_values[lane % len(special_values)]
            res = results[lane]
            self.assertIsNotNone(res)
            self.assertEqual(res, (orig_val, orig_val))
            self.assertEqual(state.get_vgpr_qword(lane, 0), orig_val)
            self.assertEqual(state.lds.read_qword(0x300 + lane * 8), orig_val)








if __name__ == "__main__":
    unittest.main()
