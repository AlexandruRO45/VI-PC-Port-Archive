from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_u16_u32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the vector integer conversion and packing instruction:
    v_cvt_pk_u16_u32 dst, src0, src1

RDNA2 VALU Instruction Specification:
    Instruction: v_cvt_pk_u16_u32 D.u, S0.u, S1.u
    Operands:
        - D:   32-bit VGPR destination
        - S0:  32-bit unsigned integer source operand (VGPR, SGPR, or literal/inline constant)
        - S1:  32-bit unsigned integer source operand (VGPR in VOP2; VGPR, SGPR, or constant in VOP3)

Mathematical Operation:
    Converts two 32-bit unsigned integers (S0 and S1) into two 16-bit unsigned integers
    with unsigned saturation (clamping to [0, 65535]), then packs them into a single 32-bit VGPR:
        D[15:0]  = min(S0.u, 0xFFFF)
        D[31:16] = min(S1.u, 0xFFFF)

    Packed result:
        D = (min(S1.u, 0xFFFF) << 16) | min(S0.u, 0xFFFF)

Key Architectural Properties:
    1. Unsigned Saturation:
       Any 32-bit integer value strictly greater than 65535 (0xFFFF) saturates to 0xFFFF.
       Values within the representable unsigned 16-bit range [0, 65535] are preserved exactly.
    2. Zero Extension / Truncation Immunity:
       Unlike raw bitfield masking or truncation (which wraps modulo 65536), v_cvt_pk_u16_u32
       strictly clamps out-of-range values to UINT16_MAX.
    3. Half-word Packing:
       S0 maps strictly to the lower 16 bits (bits 15:0) of the destination.
       S1 maps strictly to the upper 16 bits (bits 31:16) of the destination.
    4. Full 32-bit Register Overwrite:
       Unlike sub-dword / D16 instructions that preserve existing halves of a VGPR,
       v_cvt_pk_u16_u32 unconditionally overwrites all 32 bits of the destination VGPR.
    5. No Status Flag / VCC Side Effects:
       Unsigned saturation does not alter condition codes or raise arithmetic exceptions.
       VCC and EXEC masks are preserved.

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
SIGN_BIT32: int = 0x8000_0000

U16_MIN: int = 0x0000
U16_MAX: int = 0xFFFF
U32_MIN: int = 0x0000_0000
U32_MAX: int = 0xFFFF_FFFF


# ==============================================================================
# Bit Manipulation and Reference Mathematical Oracles
# ==============================================================================

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u16_sat(val: int) -> int:
    """Unsigned 32-bit to unsigned 16-bit saturating conversion.
    
    Values in [0, 65535] are returned unchanged.
    Values greater than 65535 saturate to 65535 (0xFFFF).
    """
    v = val & MASK32
    return U16_MAX if v > U16_MAX else v


def pack_u16(lo: int, hi: int) -> int:
    """Pack two 16-bit unsigned integers into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_u16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit unsigned integers (lo, hi)."""
    dw = dword & MASK32
    return dw & MASK16, (dw >> 16) & MASK16


def ref_cvt_pk_u16_u32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_u16_u32.
    
    Converts src0 and src1 from unsigned 32-bit to saturated unsigned 16-bit,
    packing src0 into bits [15:0] and src1 into bits [31:16].
    """
    lo_u16 = u16_sat(src0)
    hi_u16 = u16_sat(src1)
    return pack_u16(lo_u16, hi_u16)


def ref_cvt_pk_u16_u32_decomposed(src0: int, src1: int) -> Tuple[int, int]:
    """Decomposed architectural reference returning (lo_u16, hi_u16)."""
    return u16_sat(src0), u16_sat(src1)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(enum.Enum):
    """Architecture register files in RDNA2 VALU."""
    VGPR = "vgpr"
    SGPR = "sgpr"


@dataclass(frozen=True)
class Register:
    """Typed architectural register."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Negative register index: {self.index}")
        if self.kind == RegKind.VGPR and self.index >= 256:
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.kind == RegKind.SGPR and self.index >= 106:
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


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
    V_CVT_PK_U16_U32 = "v_cvt_pk_u16_u32"
    V_MOV_B32 = "v_mov_b32"


@dataclass(frozen=True)
class Instruction:
    """Assembly instruction model."""
    opcode: Opcode
    dst: Register
    src0: Operand
    src1: Optional[Operand] = None

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise TypeError(f"VALU instruction destination must be a VGPR, got {self.dst}")
        if self.opcode == Opcode.V_CVT_PK_U16_U32 and self.src1 is None:
            raise ValueError("v_cvt_pk_u16_u32 requires two source operands")

    def __str__(self) -> str:
        if self.src1 is not None:
            return f"{self.opcode.value} {self.dst}, {self.src0}, {self.src1}"
        return f"{self.opcode.value} {self.dst}, {self.src0}"


class ASTNode(abc.ABC):
    """Abstract base class for symbolic AST expression trees."""

    @abc.abstractmethod
    def eval(self, env: Dict[Register, int]) -> int:
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
class SatU16Node(ASTNode):
    """Symbolic unsigned 16-bit saturation node: min(expr, 0xFFFF)."""
    expr: ASTNode

    def eval(self, env: Dict[Register, int]) -> int:
        return u16_sat(self.expr.eval(env))

    def __str__(self) -> str:
        return f"sat_u16({self.expr})"


@dataclass(frozen=True)
class PackU16Node(ASTNode):
    """Symbolic 16-bit packing node: (hi << 16) | lo."""
    lo: ASTNode
    hi: ASTNode

    def eval(self, env: Dict[Register, int]) -> int:
        return pack_u16(self.lo.eval(env), self.hi.eval(env))

    def __str__(self) -> str:
        return f"pack_u16({self.lo}, {self.hi})"


@dataclass(frozen=True)
class CvtPkU16U32Node(ASTNode):
    """High-level typed AST node for v_cvt_pk_u16_u32."""
    src0: ASTNode
    src1: ASTNode

    def to_lowered_ast(self) -> ASTNode:
        """Decompose into primitive saturation and packing AST nodes."""
        return PackU16Node(lo=SatU16Node(self.src0), hi=SatU16Node(self.src1))

    def eval(self, env: Dict[Register, int]) -> int:
        return ref_cvt_pk_u16_u32(self.src0.eval(env), self.src1.eval(env))

    def __str__(self) -> str:
        return f"v_cvt_pk_u16_u32({self.src0}, {self.src1})"


def simplify_ast(node: ASTNode) -> ASTNode:
    """Constant fold AST nodes where all inputs are literals."""
    if isinstance(node, LiteralNode):
        return node
    if isinstance(node, RegReadNode):
        return node
    if isinstance(node, SatU16Node):
        simplified_child = simplify_ast(node.expr)
        if isinstance(simplified_child, LiteralNode):
            return LiteralNode(u16_sat(simplified_child.val))
        return SatU16Node(simplified_child)
    if isinstance(node, PackU16Node):
        s_lo = simplify_ast(node.lo)
        s_hi = simplify_ast(node.hi)
        if isinstance(s_lo, LiteralNode) and isinstance(s_hi, LiteralNode):
            return LiteralNode(pack_u16(s_lo.val, s_hi.val))
        return PackU16Node(s_lo, s_hi)
    if isinstance(node, CvtPkU16U32Node):
        s0 = simplify_ast(node.src0)
        s1 = simplify_ast(node.src1)
        if isinstance(s0, LiteralNode) and isinstance(s1, LiteralNode):
            return LiteralNode(ref_cvt_pk_u16_u32(s0.val, s1.val))
        return CvtPkU16U32Node(s0, s1)
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
        """Read 32-bit register value."""
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index < 256:
                raise IndexError(f"VGPR index {reg.index} out of bounds")
            return self.vgprs[reg.index] & MASK32
        if reg.kind == RegKind.SGPR:
            if not 0 <= reg.index < 106:
                raise IndexError(f"SGPR index {reg.index} out of bounds")
            return self.sgprs[reg.index] & MASK32
        raise TypeError(f"Unknown register kind: {reg.kind}")

    def write_reg(self, reg: Register, val: int) -> None:
        """Write 32-bit register value."""
        val32 = val & MASK32
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index < 256:
                raise IndexError(f"VGPR index {reg.index} out of bounds")
            self.vgprs[reg.index] = val32
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index < 106:
                raise IndexError(f"SGPR index {reg.index} out of bounds")
            self.sgprs[reg.index] = val32
        else:
            raise TypeError(f"Unknown register kind: {reg.kind}")

    def read_operand(self, op: Operand) -> int:
        """Resolve Register or Immediate operand to a 32-bit unsigned integer."""
        if isinstance(op, Register):
            return self.read_reg(op)
        if isinstance(op, Immediate):
            return op.value & MASK32
        raise TypeError(f"Invalid operand type: {type(op)}")


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU operations."""

    def step(self, inst: Instruction, state: ScalarMachineState) -> None:
        """Execute a single instruction deterministically on the scalar machine state."""
        if inst.opcode == Opcode.V_CVT_PK_U16_U32:
            if inst.src1 is None:
                raise ValueError("v_cvt_pk_u16_u32 requires src1")
            s0 = state.read_operand(inst.src0)
            s1 = state.read_operand(inst.src1)
            result = ref_cvt_pk_u16_u32(s0, s1)
            state.write_reg(inst.dst, result)
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
        """Read 32-bit register for a specific SIMD lane."""
        if not 0 <= lane < self.wave_size:
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index < 256:
                raise IndexError(f"VGPR {reg.index} out of bounds")
            return self.vgprs[lane][reg.index] & MASK32
        if reg.kind == RegKind.SGPR:
            if not 0 <= reg.index < 106:
                raise IndexError(f"SGPR {reg.index} out of bounds")
            return self.sgprs[reg.index] & MASK32
        raise TypeError(f"Unknown register kind: {reg.kind}")

    def write_lane_reg(self, lane: int, reg: Register, val: int) -> None:
        """Write 32-bit register for a specific SIMD lane."""
        if not 0 <= lane < self.wave_size:
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        val32 = val & MASK32
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index < 256:
                raise IndexError(f"VGPR {reg.index} out of bounds")
            self.vgprs[lane][reg.index] = val32
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index < 106:
                raise IndexError(f"SGPR {reg.index} out of bounds")
            self.sgprs[reg.index] = val32
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
        """Execute a single instruction across all active lanes of the wavefront."""
        for lane in range(self.wave_size):
            # Check if this lane is active under the current EXEC mask
            lane_active = bool((state.exec_mask >> lane) & 1)
            if not lane_active:
                # Inactive lane: destination VGPR must NOT be modified
                continue

            if inst.opcode == Opcode.V_CVT_PK_U16_U32:
                if inst.src1 is None:
                    raise ValueError("v_cvt_pk_u16_u32 requires src1")
                s0 = state.read_lane_operand(lane, inst.src0)
                s1 = state.read_lane_operand(lane, inst.src1)
                res = ref_cvt_pk_u16_u32(s0, s1)
                state.write_lane_reg(lane, inst.dst, res)
            elif inst.opcode == Opcode.V_MOV_B32:
                s0 = state.read_lane_operand(lane, inst.src0)
                state.write_lane_reg(lane, inst.dst, s0)
            else:
                raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkU16U32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_u16_u32(0, 0), 0)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(0, 0), (0, 0))

    def test_in_range_values_exact_preservation(self) -> None:
        s0 = 0x1234
        s1 = 0x5678
        expected = 0x56781234
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0x1234, 0x5678))

    def test_boundary_max_u16(self) -> None:
        s0 = 0xFFFF
        s1 = 0xFFFF
        expected = 0xFFFF_FFFF
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0xFFFF, 0xFFFF))

    def test_saturation_at_u16_max_plus_one(self) -> None:
        # 65536 (0x10000) must saturate to 0xFFFF, not truncate to 0x0000
        s0 = 0x10000
        s1 = 0x10001
        expected = 0xFFFF_FFFF
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0xFFFF, 0xFFFF))

    def test_extreme_u32_saturation(self) -> None:
        extremes = [
            0x0001_0000,
            0x0002_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_0000,
            0xFFFF_FFFF,
        ]
        for val in extremes:
            with self.subTest(val=hex(val)):
                self.assertEqual(u16_sat(val), 0xFFFF)
                self.assertEqual(ref_cvt_pk_u16_u32(val, 0), 0x0000_FFFF)
                self.assertEqual(ref_cvt_pk_u16_u32(0, val), 0xFFFF_0000)
                self.assertEqual(ref_cvt_pk_u16_u32(val, val), 0xFFFF_FFFF)

    def test_asymmetric_operands(self) -> None:
        # src0 in-range, src1 saturated
        self.assertEqual(ref_cvt_pk_u16_u32(42, 100_000), 0xFFFF_002A)
        # src0 saturated, src1 in-range
        self.assertEqual(ref_cvt_pk_u16_u32(100_000, 42), 0x002A_FFFF)

    def test_roundtrip_unpack_integrity(self) -> None:
        test_pairs = [
            (0, 0),
            (1, 2),
            (65535, 65535),
            (65536, 10),
            (10, 65536),
            (0xDEADBEEF, 0xCAFEBABE),
            (0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for s0, s1 in test_pairs:
            packed = ref_cvt_pk_u16_u32(s0, s1)
            lo, hi = unpack_u16(packed)
            expected_lo, expected_hi = ref_cvt_pk_u16_u32_decomposed(s0, s1)
            self.assertEqual(lo, expected_lo)
            self.assertEqual(hi, expected_hi)

    def test_powers_of_two_saturation_boundary(self) -> None:
        for bit in range(32):
            val = 1 << bit
            sat = u16_sat(val)
            if bit < 16:
                self.assertEqual(sat, val)
            else:
                self.assertEqual(sat, 0xFFFF)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestValuCvtPkU16U32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_creation_and_string_representation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.SGPR, 2),
        )
        self.assertEqual(inst.opcode, Opcode.V_CVT_PK_U16_U32)
        self.assertEqual(str(inst), "v_cvt_pk_u16_u32 v0, v1, s2")

    def test_instruction_with_immediate_operand(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 5),
            src0=Immediate(0x1234),
            src1=Immediate(0x5678),
        )
        self.assertEqual(str(inst), "v_cvt_pk_u16_u32 v5, 0x00001234, 0x00005678")

    def test_ast_evaluation_matches_oracle(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 1): 0x0000_1000,
            Register(RegKind.VGPR, 2): 0x0002_0000,
        }
        node = CvtPkU16U32Node(
            src0=RegReadNode(Register(RegKind.VGPR, 1)),
            src1=RegReadNode(Register(RegKind.VGPR, 2)),
        )
        eval_result = node.eval(env)
        oracle_result = ref_cvt_pk_u16_u32(0x0000_1000, 0x0002_0000)
        self.assertEqual(eval_result, oracle_result)
        self.assertEqual(eval_result, 0xFFFF_1000)

    def test_ast_lowered_matches_high_level(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 0): 0x0000_0005,
            Register(RegKind.SGPR, 0): 0x0000_000A,
        }
        high_level = CvtPkU16U32Node(
            src0=RegReadNode(Register(RegKind.VGPR, 0)),
            src1=RegReadNode(Register(RegKind.SGPR, 0)),
        )
        lowered = high_level.to_lowered_ast()
        self.assertEqual(high_level.eval(env), lowered.eval(env))
        self.assertEqual(lowered.eval(env), (0x000A << 16) | 0x0005)

    def test_ast_simplification_constant_folding(self) -> None:
        unfolded = CvtPkU16U32Node(
            src0=LiteralNode(0x1234),
            src1=LiteralNode(0x80000),
        )
        simplified = simplify_ast(unfolded)
        self.assertIsInstance(simplified, LiteralNode)
        self.assertEqual(simplified.eval({}), 0xFFFF_1234)


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestValuCvtPkU16U32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_step_vgpr_operands(self) -> None:
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x0000_00FF)
        self.state.write_reg(Register(RegKind.VGPR, 2), 0x0000_00EE)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x00EE_00FF)

    def test_step_sgpr_and_immediate_operands(self) -> None:
        self.state.write_reg(Register(RegKind.SGPR, 5), 0x0005_0000)  # > 65535, saturates to 0xFFFF

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 3),
            src0=Register(RegKind.SGPR, 5),
            src1=Immediate(0x1111),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 3)), 0x1111_FFFF)

    def test_sequential_program_determinism(self) -> None:
        program = [
            Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 1), Immediate(0x1234)),
            Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 2), Immediate(0x70000)),
            Instruction(
                Opcode.V_CVT_PK_U16_U32,
                Register(RegKind.VGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
            ),
        ]
        self.interp.run(program, self.state)
        res1 = self.state.read_reg(Register(RegKind.VGPR, 0))
        self.assertEqual(res1, 0xFFFF_1234)

        # Re-run from fresh state; must produce identical result
        state2 = ScalarMachineState()
        self.interp.run(program, state2)
        self.assertEqual(state2.read_reg(Register(RegKind.VGPR, 0)), res1)

    def test_register_bounds_error(self) -> None:
        with self.assertRaises(IndexError):
            self.state.vgprs[256]
        with self.assertRaises(IndexError):
            self.state.sgprs[106]


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestValuCvtPkU16U32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDWavefrontInterpreter(wave_size=32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), lane * 1000)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), lane * 3000)

        interp.step(inst, state)

        for lane in range(32):
            expected = ref_cvt_pk_u16_u32(lane * 1000, lane * 3000)
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 0)), expected)

    def test_wave32_even_odd_divergence(self) -> None:
        # Only even lanes active (0x55555555)
        state = SIMDState(wave_size=32, exec_mask=0x5555_5555)
        interp = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xCAFE_BABE

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), 0x100)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), 0x200)

        interp.step(inst, state)

        expected_active = ref_cvt_pk_u16_u32(0x100, 0x200)
        for lane in range(32):
            res = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            if lane % 2 == 0:
                self.assertEqual(res, expected_active)
            else:
                self.assertEqual(res, canary)

    def test_wave32_all_lanes_inactive(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x0000_0000)
        interp = SIMDWavefrontInterpreter(wave_size=32)
        canary = 0xDEAD_BEEF

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), 0x5)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), 0x9)

        interp.step(inst, state)

        for lane in range(32):
            self.assertEqual(state.read_lane_reg(lane, Register(RegKind.VGPR, 0)), canary)

    def test_wave64_full_and_divergent_masking(self) -> None:
        # Wave64 with upper half active (lanes 32..63)
        state = SIMDState(wave_size=64, exec_mask=0xFFFF_FFFF_0000_0000)
        interp = SIMDWavefrontInterpreter(wave_size=64)
        canary = 0x1234_5678

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )

        for lane in range(64):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), canary)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), lane * 2000)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 2), 0x80000)  # saturates

        interp.step(inst, state)

        for lane in range(64):
            val = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            if lane < 32:
                self.assertEqual(val, canary)
            else:
                expected = ref_cvt_pk_u16_u32(lane * 2000, 0x80000)
                self.assertEqual(val, expected)


# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuCvtPkU16U32Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_aliases_src0(self) -> None:
        # v0 = v_cvt_pk_u16_u32 v0, v1
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x1234)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x5678)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        self.interp.step(inst, self.state)
        # S0 should have been read before V0 was overwritten
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x56781234)
        # V1 must remain untouched
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x5678)

    def test_dst_aliases_src1(self) -> None:
        # v0 = v_cvt_pk_u16_u32 v1, v0
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x5678)
        self.state.write_reg(Register(RegKind.VGPR, 1), 0x1234)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 0),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x56781234)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0x1234)

    def test_dst_aliases_both_src0_and_src1(self) -> None:
        # v0 = v_cvt_pk_u16_u32 v0, v0
        self.state.write_reg(Register(RegKind.VGPR, 0), 0xABCD)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0xABCDABCD)

    def test_dst_aliases_both_with_saturation(self) -> None:
        # v0 = v_cvt_pk_u16_u32 v0, v0 with v0 > 65535
        self.state.write_reg(Register(RegKind.VGPR, 0), 0x70000)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0xFFFFFFFF)

    def test_sequential_inplace_chain(self) -> None:
        # v0 = v_cvt_pk_u16_u32 v0, v1 (in-place update of v0)
        # v1 = v_cvt_pk_u16_u32 v0, v1 (in-place update of v1 using updated v0)
        self.state.write_reg(Register(RegKind.VGPR, 0), 10)
        self.state.write_reg(Register(RegKind.VGPR, 1), 20)

        inst1 = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        inst2 = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 1),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        self.interp.step(inst1, self.state)
        v0_after_1 = (20 << 16) | 10  # 0x0014000A = 1310730 > 65535, so in step 2 it saturates to 0xFFFF!
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), v0_after_1)

        self.interp.step(inst2, self.state)
        # S0 is v0_after_1 (1310730 > 65535 -> 0xFFFF)
        # S1 is v1 (20)
        expected_v1 = (20 << 16) | 0xFFFF
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), expected_v1)

    def test_simd_inplace_aliasing_divergent(self) -> None:
        state = SIMDState(wave_size=32, exec_mask=0x0000_FFFF)  # low 16 lanes active
        interp = SIMDWavefrontInterpreter(wave_size=32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_U32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )

        for lane in range(32):
            state.write_lane_reg(lane, Register(RegKind.VGPR, 0), lane + 1)
            state.write_lane_reg(lane, Register(RegKind.VGPR, 1), 100)

        interp.step(inst, state)

        for lane in range(32):
            val = state.read_lane_reg(lane, Register(RegKind.VGPR, 0))
            if lane < 16:
                expected = (100 << 16) | (lane + 1)
                self.assertEqual(val, expected)
            else:
                self.assertEqual(val, lane + 1)


if __name__ == "__main__":
    unittest.main()
