from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_add_rtn_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Add with Return:
Performs an atomic single-precision IEEE 754 floating-point addition on 32-bit LDS memory
and returns the original (pre-operation) value.

Syntax:
    ds_add_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val_u32 = LDS[dword_index]
    LDS[dword_index] = f32_to_u32(u32_to_f32(old_val_u32) + u32_to_f32(vdata0))
    vdst = old_val_u32
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
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Constants
QNAN_DEFAULT: int = 0x7FC0_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def u32_to_f32(val: int) -> float:
    """Unpack 32-bit unsigned dword to Python float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def ieee754_add_f32(a_u32: int, b_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision addition matching RDNA2 hardware semantics."""
    a_u32 = u32(a_u32)
    b_u32 = u32(b_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & 0x7F_FFFF
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & 0x7F_FFFF

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces a canonical quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) + (-Inf) or (-Inf) + (+Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign != b_sign):
        return QNAN_DEFAULT

    # If either is Inf, result is that Inf
    if a_is_inf:
        return a_u32
    if b_is_inf:
        return b_u32

    # Float addition
    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    # Handle IEEE -0.0 + -0.0 = -0.0
    if a_u32 == 0x8000_0000 and b_u32 == 0x8000_0000:
        return 0x8000_0000

    try:
        res_f = a_f + b_f
    except OverflowError:
        return 0x7F80_0000 if (a_f > 0 or b_f > 0) else 0xFF80_0000

    return f32_to_u32(res_f)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_add_rtn_f32(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_add_rtn_f32.

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    data_masked = u32(data)
    new_val = ieee754_add_f32(old_val, data_masked)
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

    def atomic_add_f32(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic float addition with return on LDS dword.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val = oracle_ds_add_rtn_f32(curr, data)
        self.memory[idx] = new_val
        return old_val, new_val


class TestLdsAddRtnF32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_float_add_identity_zero(self) -> None:
        """Adding +0.0 to a float preserves its exact bit representation."""
        addr = 0x40
        val = f32_to_u32(42.5)
        self.lds.write_dword(addr, val)
        old_val, new_val = self.lds.atomic_add_f32(addr, f32_to_u32(0.0))
        self.assertEqual(old_val, val)
        self.assertEqual(new_val, val)
        self.assertEqual(self.lds.read_dword(addr), val)

    def test_float_add_basic_positive(self) -> None:
        """1.5 + 2.5 == 4.0."""
        addr = 0x80
        val1 = f32_to_u32(1.5)
        val2 = f32_to_u32(2.5)
        self.lds.write_dword(addr, val1)
        old_val, new_val = self.lds.atomic_add_f32(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(u32_to_f32(new_val), 4.0)
        self.assertEqual(self.lds.read_dword(addr), f32_to_u32(4.0))

    def test_float_add_positive_and_negative(self) -> None:
        """10.0 + (-3.5) == 6.5."""
        addr = 0x90
        val1 = f32_to_u32(10.0)
        val2 = f32_to_u32(-3.5)
        self.lds.write_dword(addr, val1)
        old_val, new_val = self.lds.atomic_add_f32(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(u32_to_f32(new_val), 6.5)

    def test_float_add_signed_zeros(self) -> None:
        """+0.0 + -0.0 == +0.0, and -0.0 + -0.0 == -0.0."""
        addr = 0xA0
        # +0.0 + -0.0 = +0.0
        self.lds.write_dword(addr, 0x0000_0000)
        old1, new1 = self.lds.atomic_add_f32(addr, 0x8000_0000)
        self.assertEqual(old1, 0x0000_0000)
        self.assertEqual(new1, 0x0000_0000)

        # -0.0 + -0.0 = -0.0
        self.lds.write_dword(addr, 0x8000_0000)
        old2, new2 = self.lds.atomic_add_f32(addr, 0x8000_0000)
        self.assertEqual(old2, 0x8000_0000)
        self.assertEqual(new2, 0x8000_0000)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x24
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1024)

        # 16-bit offset masking
        eff_masked = calc_lds_effective_address(base, 0x1_0004)
        self.assertEqual(eff_masked, 0x1004)

        # 32-bit address wraparound
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
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_SGPRS):
            raise ValueError(f"SGPR index out of range [0, {DEFAULT_NUM_SGPRS - 1}]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclass(frozen=True)
class Immediate:
    value: int  # Stored as unsigned 32-bit dword (can represent float bits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", u32(self.value))

    def __str__(self) -> str:
        return f"0x{self.value:X}"


@dataclass(frozen=True)
class DSAddRtnF32Instruction:
    dst: VGPR
    addr: VGPR
    data0: Union[VGPR, Immediate]
    offset: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.offset <= OFFSET16_MASK):
            raise ValueError(f"LDS offset out of 16-bit range [0, 65535]: {self.offset}")
        if not isinstance(self.dst, VGPR):
            raise TypeError(f"dst must be a VGPR, got {type(self.dst)}")
        if not isinstance(self.addr, VGPR):
            raise TypeError(f"addr must be a VGPR, got {type(self.addr)}")
        if not isinstance(self.data0, (VGPR, Immediate)):
            raise TypeError(f"data0 must be VGPR or Immediate, got {type(self.data0)}")

    def disasm(self) -> str:
        tokens = ["ds_add_rtn_f32", str(self.dst), str(self.addr), str(self.data0)]
        extra = []
        if self.offset != 0:
            extra.append(f"offset:{self.offset}")
        if self.gds:
            extra.append("gds")
        core = f"{tokens[0]} {tokens[1]}, {tokens[2]}, {tokens[3]}"
        return f"{core} {' '.join(extra)}".rstrip()


class ASTExpr:
    """Base class for symbolic typed expressions in RDNA2 floating-point algebra."""

    def simplify(self) -> ASTExpr:
        raise NotImplementedError

    def evaluate(self, env: Dict[str, float]) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class Symbol(ASTExpr):
    name: str

    def simplify(self) -> ASTExpr:
        return self

    def evaluate(self, env: Dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Symbol '{self.name}' not found in evaluation environment")
        return float(env[self.name])

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class FloatConstant(ASTExpr):
    value: float

    def simplify(self) -> ASTExpr:
        return self

    def evaluate(self, env: Dict[str, float]) -> float:
        return self.value

    def __str__(self) -> str:
        return f"{self.value}"


@dataclass(frozen=True)
class FloatAddExpr(ASTExpr):
    left: ASTExpr
    right: ASTExpr

    def simplify(self) -> ASTExpr:
        s_left = self.left.simplify()
        s_right = self.right.simplify()

        # Constant folding
        if isinstance(s_left, FloatConstant) and isinstance(s_right, FloatConstant):
            u_left = f32_to_u32(s_left.value)
            u_right = f32_to_u32(s_right.value)
            sum_u = ieee754_add_f32(u_left, u_right)
            return FloatConstant(u32_to_f32(sum_u))

        # Identity addition: X + 0.0 -> X (when exactly +0.0)
        if isinstance(s_right, FloatConstant) and s_right.value == 0.0 and f32_to_u32(s_right.value) == 0:
            return s_left
        if isinstance(s_left, FloatConstant) and s_left.value == 0.0 and f32_to_u32(s_left.value) == 0:
            return s_right

        return FloatAddExpr(s_left, s_right)

    def evaluate(self, env: Dict[str, float]) -> float:
        u_left = f32_to_u32(self.left.evaluate(env))
        u_right = f32_to_u32(self.right.evaluate(env))
        res_u = ieee754_add_f32(u_left, u_right)
        return u32_to_f32(res_u)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_valid_creation_and_disassembly(self) -> None:
        instr = DSAddRtnF32Instruction(
            dst=VGPR(0),
            addr=VGPR(1),
            data0=VGPR(2),
            offset=16,
        )
        self.assertEqual(instr.disasm(), "ds_add_rtn_f32 v0, v1, v2 offset:16")

        instr_imm = DSAddRtnF32Instruction(
            dst=VGPR(10),
            addr=VGPR(20),
            data0=Immediate(f32_to_u32(1.5)),
            offset=0,
            gds=True,
        )
        self.assertTrue(instr_imm.disasm().startswith("ds_add_rtn_f32 v10, v20, 0x"))
        self.assertTrue(instr_imm.disasm().endswith("gds"))

    def test_instruction_validation_vgpr_bounds(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)

    def test_instruction_validation_offset_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=65536)
        with self.assertRaises(ValueError):
            DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=-1)

    def test_symbolic_constant_folding(self) -> None:
        expr = FloatAddExpr(FloatConstant(1.25), FloatConstant(2.5))
        simplified = expr.simplify()
        self.assertIsInstance(simplified, FloatConstant)
        self.assertEqual(simplified.evaluate({}), 3.75)

    def test_symbolic_identity_zero(self) -> None:
        x = Symbol("X")
        expr = FloatAddExpr(x, FloatConstant(0.0))
        simplified = expr.simplify()
        self.assertEqual(simplified, x)

    def test_symbolic_evaluation_env(self) -> None:
        x = Symbol("A")
        y = Symbol("B")
        expr = FloatAddExpr(x, y)
        env = {"A": 5.5, "B": 4.5}
        self.assertEqual(expr.evaluate(env), 10.0)


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

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx]

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < DEFAULT_NUM_SGPRS):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for ds_add_rtn_f32."""

    def resolve_operand(
        self,
        op: Union[VGPR, Immediate],
        state: ScalarMachineState,
    ) -> int:
        if isinstance(op, VGPR):
            return state.get_vgpr(op.index)
        elif isinstance(op, Immediate):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSAddRtnF32Instruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int]:
        """Execute a single ds_add_rtn_f32 instruction deterministically.

        Latching behavior:
        Both vaddr and vdata0 operands are latched before memory is updated or
        the return value is written to vdst.
        """
        vaddr = state.get_vgpr(instr.addr.index)
        data = self.resolve_operand(instr.data0, state)

        # Atomic memory update
        old_val, new_val = state.lds.atomic_add_f32(vaddr, data, offset=instr.offset)

        # Writeback to destination register
        state.set_vgpr(instr.dst.index, old_val)

        return old_val, new_val

    def execute_sequence(
        self,
        instructions: List[DSAddRtnF32Instruction],
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
        init_mem = f32_to_u32(10.5)
        add_val = f32_to_u32(2.25)
        self.state.lds.write_dword(addr, init_mem)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, add_val)

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(u32_to_f32(new_val), 12.75)
        self.assertEqual(self.state.get_vgpr(0), init_mem)
        self.assertEqual(self.state.lds.read_dword(addr), f32_to_u32(12.75))

    def test_scalar_with_immediate_operand(self) -> None:
        addr = 0x100
        init_mem = f32_to_u32(-5.0)
        imm = f32_to_u32(8.0)
        self.state.lds.write_dword(addr, init_mem)
        self.state.set_vgpr(5, addr)

        instr = DSAddRtnF32Instruction(dst=VGPR(6), addr=VGPR(5), data0=Immediate(imm))
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, init_mem)
        self.assertEqual(u32_to_f32(new_val), 3.0)
        self.assertEqual(self.state.get_vgpr(6), init_mem)
        self.assertEqual(self.state.lds.read_dword(addr), f32_to_u32(3.0))

    def test_scalar_offset_applied(self) -> None:
        base_addr = 0x200
        offset = 0x10
        eff_addr = base_addr + offset
        self.state.lds.write_dword(eff_addr, f32_to_u32(100.0))
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, f32_to_u32(25.0))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2), offset=offset)
        old_val, new_val = self.interp.step(instr, self.state)

        self.assertEqual(old_val, f32_to_u32(100.0))
        self.assertEqual(u32_to_f32(new_val), 125.0)
        self.assertEqual(self.state.get_vgpr(0), f32_to_u32(100.0))
        self.assertEqual(self.state.lds.read_dword(eff_addr), f32_to_u32(125.0))

    def test_scalar_deterministic_instruction_sequence(self) -> None:
        addr = 0x300
        self.state.lds.write_dword(addr, f32_to_u32(0.0))
        self.state.set_vgpr(1, addr)

        seq = [
            DSAddRtnF32Instruction(dst=VGPR(10), addr=VGPR(1), data0=Immediate(f32_to_u32(1.0))),
            DSAddRtnF32Instruction(dst=VGPR(11), addr=VGPR(1), data0=Immediate(f32_to_u32(2.0))),
            DSAddRtnF32Instruction(dst=VGPR(12), addr=VGPR(1), data0=Immediate(f32_to_u32(4.0))),
            DSAddRtnF32Instruction(dst=VGPR(13), addr=VGPR(1), data0=Immediate(f32_to_u32(8.0))),
        ]
        results = self.interp.execute_sequence(seq, self.state)

        self.assertEqual(results[0], (f32_to_u32(0.0), f32_to_u32(1.0)))
        self.assertEqual(results[1], (f32_to_u32(1.0), f32_to_u32(3.0)))
        self.assertEqual(results[2], (f32_to_u32(3.0), f32_to_u32(7.0)))
        self.assertEqual(results[3], (f32_to_u32(7.0), f32_to_u32(15.0)))

        self.assertEqual(self.state.get_vgpr(10), f32_to_u32(0.0))
        self.assertEqual(self.state.get_vgpr(11), f32_to_u32(1.0))
        self.assertEqual(self.state.get_vgpr(12), f32_to_u32(3.0))
        self.assertEqual(self.state.get_vgpr(13), f32_to_u32(7.0))
        self.assertEqual(self.state.lds.read_dword(addr), f32_to_u32(15.0))

    def test_scalar_matches_oracle_across_floats(self) -> None:
        addr = 0x400
        patterns = [
            (f32_to_u32(0.0), f32_to_u32(0.0)),
            (f32_to_u32(1.0), f32_to_u32(-1.0)),
            (f32_to_u32(3.14159), f32_to_u32(2.71828)),
            (f32_to_u32(1e6), f32_to_u32(1e-3)),
            (f32_to_u32(-100.5), f32_to_u32(50.25)),
        ]
        for initial, operand in patterns:
            self.state.lds.write_dword(addr, initial)
            self.state.set_vgpr(1, addr)
            self.state.set_vgpr(2, operand)
            instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
            old_interp, new_interp = self.interp.step(instr, self.state)
            old_oracle, new_oracle = oracle_ds_add_rtn_f32(initial, operand)
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


class SIMDInterpreter:
    """Wavefront SIMD instruction interpreter with divergence, masking, and collision handling."""

    def resolve_lane_operand(
        self,
        op: Union[VGPR, Immediate],
        lane: int,
        state: SIMDState,
    ) -> int:
        if isinstance(op, VGPR):
            return state.get_vgpr(lane, op.index)
        elif isinstance(op, Immediate):
            return op.value
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def step(
        self,
        instr: DSAddRtnF32Instruction,
        state: SIMDState,
    ) -> List[Optional[Tuple[int, int]]]:
        """Execute a ds_add_rtn_f32 across all lanes in the wavefront.

        Returns a list of length num_lanes where:
        - Inactive lanes return None and modify no registers or LDS memory.
        - Active lanes return (old_val, new_val) and update dst VGPR.
        - Multiple active lanes accessing the same LDS dword serialize deterministically
          in ascending lane order.
        """
        # Step 1: Latch inputs for all active lanes
        active_lane_data: List[Tuple[int, int, int, int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                vaddr = state.get_vgpr(lane, instr.addr.index)
                data = self.resolve_lane_operand(instr.data0, lane, state)
                eff_addr = calc_lds_effective_address(vaddr, instr.offset)
                dword_idx = calc_lds_dword_index(eff_addr, state.lds.capacity_dwords)
                active_lane_data.append((lane, eff_addr, dword_idx, data))

        results: List[Optional[Tuple[int, int]]] = [None] * state.num_lanes

        # Step 2: Execute atomic operations in ascending lane order (hardware serialization)
        for lane, eff_addr, dword_idx, data in active_lane_data:
            curr_mem = state.lds.memory[dword_idx]
            old_val, new_val = oracle_ds_add_rtn_f32(curr_mem, data)
            state.lds.memory[dword_idx] = new_val
            results[lane] = (old_val, new_val)

        # Step 3: Writeback destination registers for active lanes
        for lane in range(state.num_lanes):
            res = results[lane]
            if res is not None:
                old_val, _ = res
                state.set_vgpr(lane, instr.dst.index, old_val)

        return results


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD divergence, 32/64-bit EXEC masking, and intra-wave collision."""

    def test_wave32_all_lanes_active_independent_addresses(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDInterpreter()

        # Each lane accesses its own unique dword address
        for lane in range(32):
            addr = lane * 4
            state.lds.write_dword(addr, f32_to_u32(100.0 + lane))
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, f32_to_u32(0.5))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(32):
            res = results[lane]
            self.assertIsNotNone(res)
            old_val, new_val = res  # type: ignore
            expected_old = f32_to_u32(100.0 + lane)
            expected_new = f32_to_u32(100.5 + lane)
            self.assertEqual(old_val, expected_old)
            self.assertEqual(new_val, expected_new)
            self.assertEqual(state.get_vgpr(lane, 0), expected_old)
            self.assertEqual(state.lds.read_dword(lane * 4), expected_new)

    def test_wave32_exec_mask_partial_inactivity(self) -> None:
        # Lanes 0..15 active, lanes 16..31 inactive
        exec_mask = 0x0000_FFFF
        state = SIMDState(num_lanes=32, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        sentinel_reg = 0xDEAD_BEEF
        for lane in range(32):
            addr = 0x100 + lane * 4
            state.lds.write_dword(addr, f32_to_u32(10.0 + lane))
            state.set_vgpr(lane, 0, sentinel_reg)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, f32_to_u32(5.0))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(32):
            addr = 0x100 + lane * 4
            orig_mem = f32_to_u32(10.0 + lane)
            if lane < 16:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), orig_mem)
                self.assertEqual(state.lds.read_dword(addr), f32_to_u32(15.0 + lane))
            else:
                self.assertIsNone(results[lane])
                # Destination VGPR unmodified
                self.assertEqual(state.get_vgpr(lane, 0), sentinel_reg)
                # LDS memory untouched
                self.assertEqual(state.lds.read_dword(addr), orig_mem)

    def test_wave32_exec_mask_zero_all_inactive(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        addr = 0x200
        init_val = f32_to_u32(99.0)
        state.lds.write_dword(addr, init_val)
        for lane in range(32):
            state.set_vgpr(lane, 0, 0x1111)
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, f32_to_u32(1.0))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        results = interp.step(instr, state)

        self.assertTrue(all(r is None for r in results))
        self.assertEqual(state.lds.read_dword(addr), init_val)
        for lane in range(32):
            self.assertEqual(state.get_vgpr(lane, 0), 0x1111)

    def test_wave64_full_and_partial_masks(self) -> None:
        # Alternating mask over 64 lanes
        exec_mask = 0x5555_5555_5555_5555
        state = SIMDState(num_lanes=64, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        for lane in range(64):
            addr = 0x400 + lane * 4
            state.lds.write_dword(addr, f32_to_u32(1.0))
            state.set_vgpr(lane, 1, addr)
            state.set_vgpr(lane, 2, f32_to_u32(2.0))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(64):
            addr = 0x400 + lane * 4
            if (lane % 2) == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.lds.read_dword(addr), f32_to_u32(3.0))
                self.assertEqual(state.get_vgpr(lane, 0), f32_to_u32(1.0))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lds.read_dword(addr), f32_to_u32(1.0))

    def test_intra_wave_collision_same_address(self) -> None:
        """Multiple active lanes target the identical LDS address; atomic serialization occurs."""
        state = SIMDState(num_lanes=32, exec_mask=0x0F)  # Lanes 0, 1, 2, 3 active
        interp = SIMDInterpreter()

        target_addr = 0x800
        state.lds.write_dword(target_addr, f32_to_u32(0.0))

        add_values = [1.0, 2.0, 4.0, 8.0]
        for lane in range(4):
            state.set_vgpr(lane, 1, target_addr)
            state.set_vgpr(lane, 2, f32_to_u32(add_values[lane]))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        results = interp.step(instr, state)

        # Expected serialization order:
        # Lane 0: sees 0.0, writes 1.0
        # Lane 1: sees 1.0, writes 3.0
        # Lane 2: sees 3.0, writes 7.0
        # Lane 3: sees 7.0, writes 15.0
        self.assertEqual(results[0], (f32_to_u32(0.0), f32_to_u32(1.0)))
        self.assertEqual(results[1], (f32_to_u32(1.0), f32_to_u32(3.0)))
        self.assertEqual(results[2], (f32_to_u32(3.0), f32_to_u32(7.0)))
        self.assertEqual(results[3], (f32_to_u32(7.0), f32_to_u32(15.0)))

        self.assertEqual(state.get_vgpr(0, 0), f32_to_u32(0.0))
        self.assertEqual(state.get_vgpr(1, 0), f32_to_u32(1.0))
        self.assertEqual(state.get_vgpr(2, 0), f32_to_u32(3.0))
        self.assertEqual(state.get_vgpr(3, 0), f32_to_u32(7.0))
        self.assertEqual(state.lds.read_dword(target_addr), f32_to_u32(15.0))

    def test_intra_wave_collision_with_inactive_lanes(self) -> None:
        """Lanes targeting the same address with intermediate inactive lanes."""
        # Active: Lane 1 (add 2.5), Lane 4 (add 7.5)
        exec_mask = (1 << 1) | (1 << 4)
        state = SIMDState(num_lanes=32, exec_mask=exec_mask)
        interp = SIMDInterpreter()

        target_addr = 0x900
        state.lds.write_dword(target_addr, f32_to_u32(10.0))

        for lane in range(32):
            state.set_vgpr(lane, 1, target_addr)
            state.set_vgpr(lane, 2, f32_to_u32(0.0))
        state.set_vgpr(1, 2, f32_to_u32(2.5))
        state.set_vgpr(4, 2, f32_to_u32(7.5))

        instr = DSAddRtnF32Instruction(dst=VGPR(0), addr=VGPR(1), data0=VGPR(2))
        interp.step(instr, state)

        self.assertEqual(state.get_vgpr(1, 0), f32_to_u32(10.0))
        self.assertEqual(state.get_vgpr(4, 0), f32_to_u32(12.5))
        self.assertEqual(state.lds.read_dword(target_addr), f32_to_u32(20.0))





if __name__ == "__main__":
    unittest.main()
