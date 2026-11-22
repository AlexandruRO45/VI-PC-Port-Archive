from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_u32_u16.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the vector integer unpacking and format conversion instruction:
    v_cvt_pk_u32_u16 dst, src

RDNA2 VALU Instruction Specification:
    Instruction: v_cvt_pk_u32_u16 D.u64, S0.u32 (or D.2u32, S0.u32, S1.u32)
    Operands:
        - D:   64-bit VGPR pair destination v[d:d+1] (two consecutive 32-bit VGPRs)
        - S0:  32-bit source (VGPR, SGPR, or literal/inline constant) containing packed 16-bit integers
        - S1:  Optional second 32-bit source for dual-source VOP2/VOP3 encodings

Mathematical Operation:
    Unpacks two 16-bit unsigned integers into two 32-bit unsigned integers with zero-extension:
        low_u16  = S0[15:0]
        high_u16 = S0[31:16] (or S1[15:0] in dual-source format)

        D0 (bits 31:0,  v[d])   = uint32(low_u16)  = low_u16
        D1 (bits 63:32, v[d+1]) = uint32(high_u16) = high_u16

    Combined 64-bit Destination:
        D = (uint32(high_u16) << 32) | uint32(low_u16)

Key Architectural Properties:
    1. Zero-Extension:
       Each 16-bit field is zero-extended to 32 bits. The upper 16 bits of each
       destination dword (D0[31:16] and D1[31:16]) are strictly zeroed.
    2. Multi-Register Write:
       Overwrites two consecutive 32-bit vector registers v[d] and v[d+1].
    3. Inverse / Complementary to v_cvt_pk_u16_u32:
       v_cvt_pk_u16_u32 packs two 32-bit integers into one 32-bit register (with saturation).
       v_cvt_pk_u32_u16 unpacks two 16-bit integers into two 32-bit registers (zero-extended).
    4. Exact Roundtrip Property:
       For any (u16_0, u16_1), v_cvt_pk_u32_u16(v_cvt_pk_u16_u32(u16_0, u16_1)) preserves
       both integers exactly.
    5. No Status Flag / VCC Side Effects:
       Unpacking does not alter condition codes, carry/borrow flags, or VCC.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import abc
from dataclasses import dataclass
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

U16_MIN: int = 0x0000
U16_MAX: int = 0xFFFF
U32_MIN: int = 0x0000_0000
U32_MAX: int = 0xFFFF_FFFF


# ==============================================================================
# Reference Mathematical Oracles and Bit Operations
# ==============================================================================

def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def unpack_u16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit unsigned integers (lo, hi)."""
    dw = dword & MASK32
    return dw & MASK16, (dw >> 16) & MASK16


def ref_cvt_pk_u32_u16(src: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_u32_u16.
    
    Unpacks low and high 16-bit halves of src into two zero-extended 32-bit unsigned integers.
    Returns (d0, d1) where d0 is the low dword and d1 is the high dword.
    """
    lo_u16, hi_u16 = unpack_u16(src)
    d0 = lo_u16  # zero-extended to 32 bits
    d1 = hi_u16  # zero-extended to 32 bits
    return d0, d1


def ref_cvt_pk_u32_u16_64(src: int) -> int:
    """Reference oracle returning the combined 64-bit integer (d1 << 32) | d0."""
    d0, d1 = ref_cvt_pk_u32_u16(src)
    return ((d1 & MASK32) << 32) | (d0 & MASK32)


def ref_cvt_pk_u32_u16_dual(
    src0: int,
    src1: int,
    op_sel: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """Dual-source reference oracle with halfword selection.
    
    op_sel[0] selects low (0) or high (1) 16 bits of src0 for d0.
    op_sel[1] selects low (0) or high (1) 16 bits of src1 for d1.
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    d0 = (s0 >> 16) & MASK16 if op_sel[0] else s0 & MASK16
    d1 = (s1 >> 16) & MASK16 if op_sel[1] else s1 & MASK16
    return d0, d1


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(enum.Enum):
    """Architecture register files in RDNA2 VALU."""
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclass(frozen=True)
class Register:
    """Typed architectural register, supporting scalar (count=1) or multi-word (count=2)."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Negative register index: {self.index}")
        if self.count < 1:
            raise ValueError(f"Register count must be >= 1, got {self.count}")
        max_idx = 256 if self.kind == RegKind.VGPR else 106
        if self.index + self.count > max_idx:
            raise ValueError(f"Register range [{self.index}:{self.index + self.count}] exceeds limit {max_idx}")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        if self.count == 1:
            return f"{prefix}{self.index}"
        return f"{prefix}[{self.index}:{self.index + self.count - 1}]"


@dataclass(frozen=True)
class Immediate:
    """32-bit unsigned immediate / literal constant."""
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & MASK32)

    def __str__(self) -> str:
        return f"0x{self.value:08X}"


Operand = Union[Register, Immediate]


class Opcode(enum.Enum):
    """Instruction opcodes."""
    V_CVT_PK_U32_U16 = "v_cvt_pk_u32_u16"
    V_MOV_B32 = "v_mov_b32"


@dataclass(frozen=True)
class Instruction:
    """Assembly instruction model."""
    opcode: Opcode
    dst: Register
    src0: Operand
    src1: Optional[Operand] = None
    op_sel: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst}")
        if self.opcode == Opcode.V_CVT_PK_U32_U16 and self.dst.count != 2:
            raise ValueError(f"v_cvt_pk_u32_u16 requires 64-bit (2-register) destination, got count {self.dst.count}")

    def __str__(self) -> str:
        if self.src1 is not None:
            return f"{self.opcode.value} {self.dst}, {self.src0}, {self.src1}"
        return f"{self.opcode.value} {self.dst}, {self.src0}"


class ASTNode(abc.ABC):
    """Abstract base class for symbolic AST expression trees."""

    @abc.abstractmethod
    def eval(self, env: Dict[Register, int]) -> Union[int, Tuple[int, int]]:
        """Evaluate the expression tree against register environment."""
        raise NotImplementedError


@dataclass(frozen=True)
class LiteralNode(ASTNode):
    """Constant literal AST node."""
    val: int

    def eval(self, env: Dict[Register, int]) -> int:
        return self.val & MASK32

    def __str__(self) -> str:
        return f"0x{self.val:08X}"


@dataclass(frozen=True)
class RegReadNode(ASTNode):
    """Register read AST node."""
    reg: Register

    def eval(self, env: Dict[Register, int]) -> int:
        return env.get(self.reg, 0) & MASK32

    def __str__(self) -> str:
        return str(self.reg)


@dataclass(frozen=True)
class ZeroExtendU16Node(ASTNode):
    """Symbolic zero-extension node: extracts 16-bit halfword and zero-extends to uint32."""
    expr: ASTNode
    high_half: bool = False

    def eval(self, env: Dict[Register, int]) -> int:
        val = self.expr.eval(env)
        if isinstance(val, tuple):
            val = val[0]
        val = val & MASK32
        return (val >> 16) & MASK16 if self.high_half else val & MASK16

    def __str__(self) -> str:
        half_str = "hi" if self.high_half else "lo"
        return f"zext_u16_{half_str}({self.expr})"


@dataclass(frozen=True)
class CvtPkU32U16Node(ASTNode):
    """High-level typed AST node for v_cvt_pk_u32_u16 unpacking operation."""
    src0: ASTNode
    src1: Optional[ASTNode] = None
    op_sel: Tuple[int, int] = (0, 0)

    def to_lowered_ast(self) -> Tuple[ASTNode, ASTNode]:
        """Decompose into primitive zero-extension nodes for (D0, D1)."""
        if self.src1 is None:
            d0_node = ZeroExtendU16Node(self.src0, high_half=False)
            d1_node = ZeroExtendU16Node(self.src0, high_half=True)
        else:
            d0_node = ZeroExtendU16Node(self.src0, high_half=bool(self.op_sel[0]))
            d1_node = ZeroExtendU16Node(self.src1, high_half=bool(self.op_sel[1]))
        return d0_node, d1_node

    def eval(self, env: Dict[Register, int]) -> Tuple[int, int]:
        s0 = self.src0.eval(env)
        if isinstance(s0, tuple):
            s0 = s0[0]
        if self.src1 is None:
            return ref_cvt_pk_u32_u16(s0)
        s1 = self.src1.eval(env)
        if isinstance(s1, tuple):
            s1 = s1[0]
        return ref_cvt_pk_u32_u16_dual(s0, s1, self.op_sel)

    def __str__(self) -> str:
        if self.src1 is not None:
            return f"v_cvt_pk_u32_u16({self.src0}, {self.src1})"
        return f"v_cvt_pk_u32_u16({self.src0})"


def simplify_ast(node: ASTNode) -> ASTNode:
    """Constant fold AST nodes where inputs are literals."""
    if isinstance(node, LiteralNode):
        return node
    if isinstance(node, RegReadNode):
        return node
    if isinstance(node, ZeroExtendU16Node):
        s_child = simplify_ast(node.expr)
        if isinstance(s_child, LiteralNode):
            res = (s_child.val >> 16) & MASK16 if node.high_half else s_child.val & MASK16
            return LiteralNode(res)
        return ZeroExtendU16Node(s_child, high_half=node.high_half)
    if isinstance(node, CvtPkU32U16Node):
        s0 = simplify_ast(node.src0)
        s1 = simplify_ast(node.src1) if node.src1 is not None else None
        return CvtPkU32U16Node(s0, s1, node.op_sel)
    return node


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Architectural register state for a single thread/scalar execution."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0

    def read_reg(self, reg: Register) -> int:
        """Read 32-bit dword or 64-bit qword register value."""
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                return self.vgprs[reg.index] & MASK32
            elif reg.count == 2:
                lo = self.vgprs[reg.index] & MASK32
                hi = self.vgprs[reg.index + 1] & MASK32
                return (hi << 32) | lo
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                return self.sgprs[reg.index] & MASK32
            elif reg.count == 2:
                lo = self.sgprs[reg.index] & MASK32
                hi = self.sgprs[reg.index + 1] & MASK32
                return (hi << 32) | lo
        raise TypeError(f"Unknown register kind: {reg.kind}")

    def write_reg(self, reg: Register, val: int) -> None:
        """Write 32-bit dword or 64-bit qword register value."""
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                self.vgprs[reg.index] = val & MASK32
            elif reg.count == 2:
                self.vgprs[reg.index] = val & MASK32
                self.vgprs[reg.index + 1] = (val >> 32) & MASK32
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                self.sgprs[reg.index] = val & MASK32
            elif reg.count == 2:
                self.sgprs[reg.index] = val & MASK32
                self.sgprs[reg.index + 1] = (val >> 32) & MASK32
        else:
            raise TypeError(f"Unknown register kind: {reg.kind}")

    def read_operand(self, op: Operand) -> int:
        """Resolve Register or Immediate operand to integer value."""
        if isinstance(op, Register):
            return self.read_reg(op)
        if isinstance(op, Immediate):
            return op.value & MASK32
        raise TypeError(f"Invalid operand type: {type(op)}")


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU operations."""

    def step(self, inst: Instruction, state: ScalarMachineState) -> None:
        """Execute a single instruction deterministically on the scalar machine state."""
        if inst.opcode == Opcode.V_CVT_PK_U32_U16:
            s0 = state.read_operand(inst.src0)
            if inst.src1 is None:
                d0, d1 = ref_cvt_pk_u32_u16(s0)
            else:
                s1 = state.read_operand(inst.src1)
                d0, d1 = ref_cvt_pk_u32_u16_dual(s0, s1, inst.op_sel)
            # Write d0 to v[d] and d1 to v[d+1]
            state.write_reg(Register(inst.dst.kind, inst.dst.index, count=1), d0)
            state.write_reg(Register(inst.dst.kind, inst.dst.index + 1, count=1), d1)
        elif inst.opcode == Opcode.V_MOV_B32:
            s0 = state.read_operand(inst.src0)
            state.write_reg(inst.dst, s0)
        else:
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")

    def run(self, program: List[Instruction], state: ScalarMachineState) -> None:
        """Execute a sequence of instructions sequentially and deterministically."""
        for inst in program:
            self.step(inst, state)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(self, wave_size: int = 32, exec_mask: Optional[int] = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wavefront size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.mask_limit: int = MASK32 if wave_size == 32 else MASK64
        self.exec_mask: int = self.mask_limit if exec_mask is None else (exec_mask & self.mask_limit)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0

    def read_lane_reg(self, lane: int, reg: Register) -> int:
        """Read register value for a specific lane."""
        if not 0 <= lane < self.wave_size:
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                return self.vgprs[lane][reg.index] & MASK32
            elif reg.count == 2:
                lo = self.vgprs[lane][reg.index] & MASK32
                hi = self.vgprs[lane][reg.index + 1] & MASK32
                return (hi << 32) | lo
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                return self.sgprs[reg.index] & MASK32
            elif reg.count == 2:
                lo = self.sgprs[reg.index] & MASK32
                hi = self.sgprs[reg.index + 1] & MASK32
                return (hi << 32) | lo
        raise TypeError(f"Unknown register kind: {reg.kind}")

    def write_lane_reg(self, lane: int, reg: Register, val: int) -> None:
        """Write register value for a specific lane."""
        if not 0 <= lane < self.wave_size:
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                self.vgprs[lane][reg.index] = val & MASK32
            elif reg.count == 2:
                self.vgprs[lane][reg.index] = val & MASK32
                self.vgprs[lane][reg.index + 1] = (val >> 32) & MASK32
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR range [{reg.index}:{reg.index + reg.count}] out of bounds")
            if reg.count == 1:
                self.sgprs[reg.index] = val & MASK32
            elif reg.count == 2:
                self.sgprs[reg.index] = val & MASK32
                self.sgprs[reg.index + 1] = (val >> 32) & MASK32
        else:
            raise TypeError(f"Unknown register kind: {reg.kind}")

    def read_lane_operand(self, lane: int, op: Operand) -> int:
        """Resolve operand for a specific lane."""
        if isinstance(op, Register):
            return self.read_lane_reg(lane, op)
        if isinstance(op, Immediate):
            return op.value & MASK32
        raise TypeError(f"Invalid operand type: {type(op)}")


class SIMDWavefrontInterpreter:
    """SIMD wavefront execution engine modeling multi-lane divergence and EXEC masking."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wavefront size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size

    def step(self, inst: Instruction, state: SIMDState) -> None:
        """Execute a single instruction across active lanes in wavefront."""
        for lane in range(self.wave_size):
            lane_active = bool((state.exec_mask >> lane) & 1)
            if not lane_active:
                # Inactive lane: destination VGPR pair must NOT be modified
                continue

            if inst.opcode == Opcode.V_CVT_PK_U32_U16:
                s0 = state.read_lane_operand(lane, inst.src0)
                if inst.src1 is None:
                    d0, d1 = ref_cvt_pk_u32_u16(s0)
                else:
                    s1 = state.read_lane_operand(lane, inst.src1)
                    d0, d1 = ref_cvt_pk_u32_u16_dual(s0, s1, inst.op_sel)
                state.write_lane_reg(lane, Register(inst.dst.kind, inst.dst.index, count=1), d0)
                state.write_lane_reg(lane, Register(inst.dst.kind, inst.dst.index + 1, count=1), d1)
            elif inst.opcode == Opcode.V_MOV_B32:
                s0 = state.read_lane_operand(lane, inst.src0)
                state.write_lane_reg(lane, inst.dst, s0)
            else:
                raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkU32U16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_u32_u16(0), (0, 0))
        self.assertEqual(ref_cvt_pk_u32_u16_64(0), 0)

    def test_basic_values_unpacking(self) -> None:
        src = 0x56781234
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x00001234)
        self.assertEqual(d1, 0x00005678)
        self.assertEqual(ref_cvt_pk_u32_u16_64(src), 0x00005678_00001234)

    def test_max_u16_boundary(self) -> None:
        src = 0xFFFF_FFFF
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_FFFF)
        self.assertEqual(d1, 0x0000_FFFF)
        self.assertEqual(ref_cvt_pk_u32_u16_64(src), 0x0000_FFFF_0000_FFFF)

    def test_isolated_low_half(self) -> None:
        src = 0x0000_ABCD
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_ABCD)
        self.assertEqual(d1, 0x0000_0000)

    def test_isolated_high_half(self) -> None:
        src = 0xABCD_0000
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_0000)
        self.assertEqual(d1, 0x0000_ABCD)

    def test_upper_16_bits_strictly_zeroed(self) -> None:
        test_values = [
            0x0000_0000,
            0x0000_FFFF,
            0xFFFF_0000,
            0xFFFF_FFFF,
            0x1234_5678,
            0x8000_8000,
            0x7FFF_7FFF,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                d0, d1 = ref_cvt_pk_u32_u16(val)
                self.assertEqual(d0 & ~MASK16, 0)
                self.assertEqual(d1 & ~MASK16, 0)

    def test_dual_source_oracle(self) -> None:
        s0 = 0xAAAA_1111
        s1 = 0xBBBB_2222
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (0, 0)), (0x1111, 0x2222))
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (1, 0)), (0xAAAA, 0x2222))
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (0, 1)), (0x1111, 0xBBBB))
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (1, 1)), (0xAAAA, 0xBBBB))

    def test_powers_of_two_bit_positions(self) -> None:
        for bit in range(32):
            val = 1 << bit
            d0, d1 = ref_cvt_pk_u32_u16(val)
            if bit < 16:
                self.assertEqual(d0, 1 << bit)
                self.assertEqual(d1, 0)
            else:
                self.assertEqual(d0, 0)
                self.assertEqual(d1, 1 << (bit - 16))


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestValuCvtPkU32U16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_creation_and_string_representation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst.opcode, Opcode.V_CVT_PK_U32_U16)
        self.assertEqual(str(inst), "v_cvt_pk_u32_u16 v[0:1], v2")

    def test_instruction_dual_source(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Register(RegKind.VGPR, 6),
            src1=Register(RegKind.SGPR, 8),
        )
        self.assertEqual(str(inst), "v_cvt_pk_u32_u16 v[4:5], v6, s8")

    def test_ast_evaluation_matches_oracle(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 1): 0x1234_5678,
        }
        node = CvtPkU32U16Node(src0=RegReadNode(Register(RegKind.VGPR, 1)))
        eval_result = node.eval(env)
        self.assertEqual(eval_result, (0x5678, 0x1234))
        self.assertEqual(eval_result, ref_cvt_pk_u32_u16(0x1234_5678))

    def test_lowered_ast_matches_high_level(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 0): 0xABCD_EF01,
        }
        high_level = CvtPkU32U16Node(src0=RegReadNode(Register(RegKind.VGPR, 0)))
        d0_node, d1_node = high_level.to_lowered_ast()
        d0_val = d0_node.eval(env)
        d1_val = d1_node.eval(env)
        self.assertEqual((d0_val, d1_val), high_level.eval(env))
        self.assertEqual((d0_val, d1_val), (0xEF01, 0xABCD))

    def test_ast_simplification_constant_folding(self) -> None:
        node_lo = ZeroExtendU16Node(LiteralNode(0x5555_AAAA), high_half=False)
        node_hi = ZeroExtendU16Node(LiteralNode(0x5555_AAAA), high_half=True)
        s_lo = simplify_ast(node_lo)
        s_hi = simplify_ast(node_hi)
        self.assertIsInstance(s_lo, LiteralNode)
        self.assertIsInstance(s_hi, LiteralNode)
        self.assertEqual(s_lo.eval({}), 0xAAAA)
        self.assertEqual(s_hi.eval({}), 0x5555)


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestValuCvtPkU32U16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_step_single_source_vgpr(self) -> None:
        self.state.write_reg(Register(RegKind.VGPR, 2), 0x0042_0010)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x00000010)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x00000042)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0, count=2)), 0x00000042_00000010)

    def test_step_dual_source_with_sgpr(self) -> None:
        self.state.write_reg(Register(RegKind.VGPR, 5), 0x1111)
        self.state.write_reg(Register(RegKind.SGPR, 2), 0x2222)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 6, count=2),
            src0=Register(RegKind.VGPR, 5),
            src1=Register(RegKind.SGPR, 2),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 6)), 0x1111)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 7)), 0x2222)

    def test_step_with_immediate_constant(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 10, count=2),
            src0=Immediate(0x9876_5432),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 10)), 0x00005432)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 11)), 0x00009876)

    def test_sequential_program_determinism(self) -> None:
        program = [
            Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 4), Immediate(0xBEEF_CAFE)),
            Instruction(
                Opcode.V_CVT_PK_U32_U16,
                Register(RegKind.VGPR, 0, count=2),
                Register(RegKind.VGPR, 4),
            ),
        ]
        self.interp.run(program, self.state)
        res_v0 = self.state.read_reg(Register(RegKind.VGPR, 0))
        res_v1 = self.state.read_reg(Register(RegKind.VGPR, 1))
        self.assertEqual(res_v0, 0x0000CAFE)
        self.assertEqual(res_v1, 0x0000BEEF)

        # Fresh state reproduction
        fresh_state = ScalarMachineState()
        self.interp.run(program, fresh_state)
        self.assertEqual(fresh_state.read_reg(Register(RegKind.VGPR, 0)), res_v0)
        self.assertEqual(fresh_state.read_reg(Register(RegKind.VGPR, 1)), res_v1)


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestValuCvtPkU32U16SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDWavefrontInterpreter(wave_size=32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )

        for lane in range(32):
            packed = ((lane * 100) << 16) | (lane * 10)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), packed)

        interp.step(inst, state)

        for lane in range(32):
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 0)), lane * 10)
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 1)), lane * 100)

    def test_wave32_divergence_canary_preservation(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x5555_5555)
        interp = SIMDWavefrontInterpreter(wave_size=32)
        canary0 = 0xDEAD_BEEF
        canary1 = 0xCAFE_BABE

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), canary0)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), canary1)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), 0x0020_0010)

        interp.step(inst, state)

        for lane in range(32):
            v0 = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            v1 = state.read_lane_reg(lane, Register(RegKind.VGPR, 1))
            if lane % 2 == 0:
                self.assertEqual(v0, 0x0010)
                self.assertEqual(v1, 0x0020)
            else:
                self.assertEqual(v0, canary0)
                self.assertEqual(v1, canary1)

    def test_wave32_all_inactive_lanes(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x0000_0000)
        interp = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xBAAD_F00D

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Immediate(0xFFFF_FFFF),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 4), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 5), canary)

        interp.step(inst, state)

        for lane in range(32):
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 4)), canary)
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 5)), canary)

    def test_wave64_full_and_divergent_execution(self) -> None:
        state = SIMDState(wave_size=64, exec_mask=0xFFFF_FFFF_0000_0000)
        interp = SIMDWavefrontInterpreter(wave_size=64)
        canary = 0x1122_3344

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )

        for lane in range(64):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), 0x7777_3333)

        interp.step(inst, state)

        for lane in range(64):
            v0 = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            v1 = state.read_lane_reg(lane, Register(RegKind.VGPR, 1))
            if lane < 32:
                self.assertEqual(v0, canary)
                self.assertEqual(v1, canary)
            else:
                self.assertEqual(v0, 0x3333)
                self.assertEqual(v1, 0x7777)


# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuCvtPkU32U16Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_lo_aliases_src(self) -> None:
        # v[0:1] = v_cvt_pk_u32_u16 v0
        # v0 holds source packed value, which must be fully read before v0 is overwritten
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x5678_1234)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0xDEAD_BEEF)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 0),
        )
        self.interp.step(inst, self.state)
        # Low half 0x1234 written to v0; High half 0x5678 written to v1
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x0000_1234)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x0000_5678)

    def test_dst_hi_aliases_src(self) -> None:
        # v[0:1] = v_cvt_pk_u32_u16 v1
        # v1 holds source packed value, and is also dst[1]
        self.state.write_reg(Register(RegKind.VGPR, 0), 0xDEAD_BEEF)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x9ABC_DEF0)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 1),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x0000_DEF0)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x0000_9ABC)

    def test_dual_source_aliasing(self) -> None:
        # v[0:1] = v_cvt_pk_u32_u16 v0, v1 (both sources alias the destination registers)
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x0000_1111)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x0000_2222)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x0000_1111)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x0000_2222)

    def test_dual_source_reversed_aliasing(self) -> None:
        # v[0:1] = v_cvt_pk_u32_u16 v1, v0 (swapped operands)
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x0000_1111)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x0000_2222)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 0),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x0000_2222)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x0000_1111)

    def test_simd_inplace_aliasing_divergent(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x0000_FFFF)  # low 16 lanes active
        interp = SIMDWavefrontInterpreter(wave_size=32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 0),
        )

        for lane in range(32):
            packed = ((lane + 0x100) << 16) | (lane + 1)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), packed)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), 0xCAFE_BABE)

        interp.step(inst, state)

        for lane in range(32):
            v0 = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            v1 = state.read_lane_reg(lane, Register(RegKind.VGPR, 1))
            if lane < 16:
                self.assertEqual(v0, lane + 1)
                self.assertEqual(v1, lane + 0x100)
            else:
                expected_unmodified_v0 = ((lane + 0x100) << 16) | (lane + 1)
                self.assertEqual(v0, expected_unmodified_v0)
                self.assertEqual(v1, 0xCAFE_BABE)



# ==============================================================================
# Unit Tests: Condition Codes and Status Flags
# ==============================================================================

class TestValuCvtPkU32U16ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vcc_canary_preservation(self) -> None:
        canaries = [
            0x0000_0000_0000_0000,
            0xFFFF_FFFF_FFFF_FFFF,
            0xAAAA_AAAA_5555_5555,
            0x1234_5678_9ABC_DEF0,
            0x8000_0000_0000_0001,
        ]
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )
        self.state.write_reg(Register(RegKind.VGPR, 2), 0xFFFF_FFFF)

        for canary in canaries:
            with self.subTest(canary=hex(canary)):
                self.state.vcc = canary
                self.interp.step(inst, self.state)
                self.assertEqual(self.state.vcc, canary, "VCC must remain untouched by v_cvt_pk_u32_u16")

    def test_simd_vcc_and_exec_preservation_wave32(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x5555_5555)
        interp = SIMDWavefrontInterpreter(wave_size=32)
        vcc_canary = 0x1234_5678_ABCD_EF01
        exec_canary = 0x5555_5555
        state.vcc = vcc_canary

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Immediate(0x8000_0001),
        )
        interp.step(inst, state)

        self.assertEqual(state.vcc, vcc_canary)
        self.assertEqual(state.exec_mask, exec_canary)

    def test_simd_vcc_and_exec_preservation_wave64(self) -> None:
        state = SIMDState(wave_size=64, exec_mask=0xF0F0_F0F0_0F0F_0F0F)
        interp = SIMDWavefrontInterpreter(wave_size=64)
        vcc_canary = 0xFEED_FACE_CAFE_BEEF
        exec_canary = 0xF0F0_F0F0_0F0F_0F0F
        state.vcc = vcc_canary

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 2, count=2),
            src0=Immediate(0xFFFF_FFFF),
        )
        interp.step(inst, state)

        self.assertEqual(state.vcc, vcc_canary)
        self.assertEqual(state.exec_mask, exec_canary)

    def test_no_carry_borrow_generation(self) -> None:
        self.state.vcc = 0
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Immediate(0xFFFF_FFFF),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.vcc, 0)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0xFFFF)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0xFFFF)

# ==============================================================================
# Unit Tests: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestValuCvtPkU32U16IEEE754EdgeCases(unittest.TestCase):
    """Unit tests for packed IEEE 754 float16 bit patterns and integer boundaries."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_ieee754_half_precision_constants(self) -> None:
        patterns = [
            (0x8000_0000, 0x0000, 0x8000, "+0.0 (lo), -0.0 (hi)"),
            (0xBC00_3C00, 0x3C00, 0xBC00, "+1.0 (lo), -1.0 (hi)"),
            (0xFC00_7C00, 0x7C00, 0xFC00, "+Inf (lo), -Inf (hi)"),
            (0x7C01_7E00, 0x7E00, 0x7C01, "qNaN (lo), sNaN (hi)"),
            (0x03FF_0001, 0x0001, 0x03FF, "min subnormal (lo), max subnormal (hi)"),
        ]

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U32_U16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )

        for packed, exp_d0, exp_d1, desc in patterns:
            with self.subTest(desc=desc):
                self.state.write_reg(Register(RegKind.VGPR, 2), packed)
                self.interp.step(inst, self.state)
                self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), exp_d0)
                self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), exp_d1)

    def test_nan_payload_preservation_without_trap(self) -> None:
        payload_lo = 0x7C01
        payload_hi = 0x7EFF
        packed = (payload_hi << 16) | payload_lo

        d0, d1 = ref_cvt_pk_u32_u16(packed)
        self.assertEqual(d0, payload_lo)
        self.assertEqual(d1, payload_hi)

    def test_integer_boundary_extremes(self) -> None:
        cases = [
            (0x7FFF_8000, 0x8000, 0x7FFF),
            (0x8000_7FFF, 0x7FFF, 0x8000),
            (0xFFFF_0000, 0x0000, 0xFFFF),
            (0x0000_FFFF, 0xFFFF, 0x0000),
            (0xFFFE_0001, 0x0001, 0xFFFE),
        ]
        for src, exp0, exp1 in cases:
            with self.subTest(src=hex(src)):
                d0, d1 = ref_cvt_pk_u32_u16(src)
                self.assertEqual(d0, exp0)
                self.assertEqual(d1, exp1)

# ==============================================================================
# Unit Tests: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestValuCvtPkU32U16NegativeRejection(unittest.TestCase):
    """Unit tests for syntax validation and negative error rejection."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_sgpr_destination_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Instruction(
                opcode=Opcode.V_CVT_PK_U32_U16,
                dst=Register(RegKind.SGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2),
            )

    def test_destination_count_must_be_two(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_PK_U32_U16,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2),
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_PK_U32_U16,
                dst=Register(RegKind.VGPR, 0, count=3),
                src0=Register(RegKind.VGPR, 2),
            )

    def test_out_of_bounds_register_indices(self) -> None:
        with self.assertRaises((ValueError, IndexError)):
            Register(RegKind.VGPR, 255, count=2)

    def test_invalid_wavefront_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=48)

    def test_invalid_lane_index(self) -> None:
        simd = SIMDState(wave_size=32)
        with self.assertRaises(IndexError):
            simd.read_lane_reg(-1, Register(RegKind.VGPR, 0))
        with self.assertRaises(IndexError):
            simd.read_lane_reg(32, Register(RegKind.VGPR, 0))


class TestValuCvtPkU32U16MutationDetection(unittest.TestCase):
    """Unit tests detecting potential implementation mutations and regressions."""

    def test_mutant_sign_extension_detection(self) -> None:
        src = 0x8000_8000
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_8000, "Must be zero-extended, not sign-extended")
        self.assertEqual(d1, 0x0000_8000, "Must be zero-extended, not sign-extended")
        self.assertNotEqual(d0, 0xFFFF_8000, "Sign-extension mutant must be rejected")
        self.assertNotEqual(d1, 0xFFFF_8000, "Sign-extension mutant must be rejected")

    def test_mutant_swapped_halves_detection(self) -> None:
        src = 0x1111_2222
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual((d0, d1), (0x2222, 0x1111))
        self.assertNotEqual((d0, d1), (0x1111, 0x2222), "Swapped dwords mutant must be rejected")

    def test_mutant_shift_amount_detection(self) -> None:
        src = 0x0001_0000
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d1, 1)
        self.assertNotEqual(d1, 2, "Shift-15 mutant must be rejected")
        self.assertNotEqual(d1, 0, "Shift-17 mutant must be rejected")

    def test_mutant_mask_width_detection(self) -> None:
        src = 0x8000_8000
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x8000)
        self.assertNotEqual(d0, 0x0000, "15-bit mask mutant must be rejected")

if __name__ == "__main__":
    unittest.main()
