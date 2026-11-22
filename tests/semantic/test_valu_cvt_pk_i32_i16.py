from __future__ import annotations

import enum
import math
import struct
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
RDNA2 VALU instruction semantic tests: v_cvt_pk_i32_i16.

Converts packed two 16-bit signed integers into two 32-bit signed integers.
Single-source format:
    v[dst:dst+1] = v_cvt_pk_i32_i16 src0
    dst[0] = sign_extend(src0[15:0])
    dst[1] = sign_extend(src0[31:16])

Dual-source format (VOP3 with op_sel):
    v[dst:dst+1] = v_cvt_pk_i32_i16 src0, src1, op_sel=[sel0, sel1]
    dst[0] = sign_extend(sel0 ? src0[31:16] : src0[15:0])
    dst[1] = sign_extend(sel1 ? src1[31:16] : src1[15:0])

Destination is always a pair of consecutive 32-bit VGPRs: v[d:d+1].
No condition codes or carry/borrow flags are set.
"""

MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF
SIGN16 = 0x8000
SIGN32 = 0x8000_0000


def sign_extend_16_to_32(val: int) -> int:
    """Sign-extend a 16-bit signed integer into a 32-bit unsigned representation."""
    val = val & MASK16
    if val & SIGN16:
        return (val | 0xFFFF_0000) & MASK32
    return val


def to_signed_32(val: int) -> int:
    """Convert unsigned 32-bit representation to Python signed int."""
    val = val & MASK32
    if val & SIGN32:
        return val - 0x1_0000_0000
    return val


def to_signed_16(val: int) -> int:
    """Convert unsigned 16-bit representation to Python signed int."""
    val = val & MASK16
    if val & SIGN16:
        return val - 0x1_0000
    return val


def ref_cvt_pk_i32_i16(src: int) -> Tuple[int, int]:
    """
    Reference mathematical oracle for single-source v_cvt_pk_i32_i16.
    Extracts two 16-bit signed integers from a 32-bit register and sign-extends them to 32-bit words.
    Returns (d0, d1) as unsigned 32-bit integers.
    """
    src32 = src & MASK32
    lo16 = src32 & MASK16
    hi16 = (src32 >> 16) & MASK16
    d0 = sign_extend_16_to_32(lo16)
    d1 = sign_extend_16_to_32(hi16)
    return d0, d1


def ref_cvt_pk_i32_i16_64(src: int) -> int:
    """Return combined 64-bit pair (d1 << 32) | d0 for single-source v_cvt_pk_i32_i16."""
    d0, d1 = ref_cvt_pk_i32_i16(src)
    return (d1 << 32) | d0


def ref_cvt_pk_i32_i16_dual(src0: int, src1: int, op_sel: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """
    Reference mathematical oracle for dual-source v_cvt_pk_i32_i16.
    op_sel[0]: select high half of src0 if 1, else low half
    op_sel[1]: select high half of src1 if 1, else low half
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32

    half0 = ((s0 >> 16) & MASK16) if (op_sel[0] & 1) else (s0 & MASK16)
    half1 = ((s1 >> 16) & MASK16) if (op_sel[1] & 1) else (s1 & MASK16)

    d0 = sign_extend_16_to_32(half0)
    d1 = sign_extend_16_to_32(half1)
    return d0, d1


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkI32I16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_i32_i16(0), (0, 0))
        self.assertEqual(ref_cvt_pk_i32_i16_64(0), 0)

    def test_positive_signed_unpacking(self) -> None:
        src = 0x0042_0010
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0x0000_0010)
        self.assertEqual(d1, 0x0000_0042)
        self.assertEqual(to_signed_32(d0), 16)
        self.assertEqual(to_signed_32(d1), 66)
        self.assertEqual(ref_cvt_pk_i32_i16_64(src), 0x0000_0042_0000_0010)

    def test_negative_signed_unpacking(self) -> None:
        src = 0xFFFF_FFFF  # Both halves are -1 (0xFFFF)
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0xFFFF_FFFF)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(to_signed_32(d0), -1)
        self.assertEqual(to_signed_32(d1), -1)
        self.assertEqual(ref_cvt_pk_i32_i16_64(src), 0xFFFF_FFFF_FFFF_FFFF)

    def test_mixed_signs(self) -> None:
        src = 0xFFFF_0001  # lo is +1, hi is -1
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0x0000_0001)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(to_signed_32(d0), 1)
        self.assertEqual(to_signed_32(d1), -1)

    def test_int16_min_max_extremes(self) -> None:
        # lo is -32768 (0x8000), hi is +32767 (0x7FFF)
        src = 0x7FFF_8000
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0xFFFF_8000)
        self.assertEqual(d1, 0x0000_7FFF)
        self.assertEqual(to_signed_32(d0), -32768)
        self.assertEqual(to_signed_32(d1), 32767)

    def test_dual_source_oracle(self) -> None:
        s0 = 0x8001_0002  # hi is -32767, lo is +2
        s1 = 0x0004_FFFE  # hi is +4, lo is -2
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (0, 0)), (2, 0xFFFF_FFFE))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (1, 0)), (0xFFFF_8001, 0xFFFF_FFFE))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (0, 1)), (2, 4))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (1, 1)), (0xFFFF_8001, 4))

    def test_powers_of_two_and_sign_propagation(self) -> None:
        for bit in range(16):
            val_lo = 1 << bit
            d0, d1 = ref_cvt_pk_i32_i16(val_lo)
            self.assertEqual(d1, 0)
            if bit == 15:
                self.assertEqual(d0, 0xFFFF_8000)
            else:
                self.assertEqual(d0, 1 << bit)

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(enum.Enum):
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Register count must be >= 1, got {self.count}")
        max_idx = 256 if self.kind == RegKind.VGPR else 106
        if not (0 <= self.index and self.index + self.count <= max_idx):
            raise ValueError(f"Register range [{self.index}:{self.index + self.count}] exceeds limit {max_idx}")

    def __str__(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"


@dataclass(frozen=True)
class Immediate:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & MASK32)

    def __str__(self) -> str:
        return f"0x{self.value:08X}"


Operand = Union[Register, Immediate]


class Opcode(enum.Enum):
    V_CVT_PK_I32_I16 = "v_cvt_pk_i32_i16"
    V_MOV_B32 = "v_mov_b32"


@dataclass
class Instruction:
    opcode: Opcode
    dst: Register
    src0: Operand
    src1: Optional[Operand] = None
    op_sel: Tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.opcode == Opcode.V_CVT_PK_I32_I16:
            if self.dst.kind != RegKind.VGPR:
                raise TypeError(f"Destination must be VGPR, got {self.dst.kind}")
            if self.dst.count != 2:
                raise ValueError(f"Destination register count for {self.opcode.value} must be 2, got {self.dst.count}")

    def __str__(self) -> str:
        if self.src1 is None:
            return f"{self.opcode.value} {self.dst}, {self.src0}"
        if self.op_sel != (0, 0):
            return f"{self.opcode.value} {self.dst}, {self.src0}, {self.src1} op_sel:[{self.op_sel[0]},{self.op_sel[1]}]"
        return f"{self.opcode.value} {self.dst}, {self.src0}, {self.src1}"


class ASTNode:
    def eval(self, env: Dict[Register, int]) -> int:
        raise NotImplementedError


@dataclass
class LiteralNode(ASTNode):
    val: int

    def eval(self, env: Dict[Register, int]) -> int:
        return self.val & MASK32


@dataclass
class RegReadNode(ASTNode):
    reg: Register

    def eval(self, env: Dict[Register, int]) -> int:
        return env.get(self.reg, 0) & MASK32


@dataclass
class SignExtendI16Node(ASTNode):
    operand: ASTNode
    high_half: bool = False

    def eval(self, env: Dict[Register, int]) -> int:
        raw = self.operand.eval(env)
        half = ((raw >> 16) & MASK16) if self.high_half else (raw & MASK16)
        return sign_extend_16_to_32(half)


@dataclass
class CvtPkI32I16Node(ASTNode):
    src0: ASTNode
    src1: Optional[ASTNode] = None
    op_sel: Tuple[int, int] = (0, 0)

    def eval(self, env: Dict[Register, int]) -> Tuple[int, int]:
        s0 = self.src0.eval(env)
        if self.src1 is None:
            return ref_cvt_pk_i32_i16(s0)
        s1 = self.src1.eval(env)
        return ref_cvt_pk_i32_i16_dual(s0, s1, self.op_sel)

    def to_lowered_ast(self) -> Tuple[ASTNode, ASTNode]:
        if self.src1 is None:
            d0_node = SignExtendI16Node(self.src0, high_half=False)
            d1_node = SignExtendI16Node(self.src0, high_half=True)
            return d0_node, d1_node
        d0_node = SignExtendI16Node(self.src0, high_half=bool(self.op_sel[0]))
        d1_node = SignExtendI16Node(self.src1, high_half=bool(self.op_sel[1]))
        return d0_node, d1_node


def simplify_ast(node: ASTNode) -> ASTNode:
    if isinstance(node, SignExtendI16Node):
        sub = simplify_ast(node.operand)
        if isinstance(sub, LiteralNode):
            half = ((sub.val >> 16) & MASK16) if node.high_half else (sub.val & MASK16)
            return LiteralNode(sign_extend_16_to_32(half))
        return SignExtendI16Node(sub, node.high_half)
    return node


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class TestValuCvtPkI32I16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_creation_and_string_representation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I32_I16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst.opcode, Opcode.V_CVT_PK_I32_I16)
        self.assertEqual(str(inst), "v_cvt_pk_i32_i16 v[0:1], v2")

    def test_instruction_dual_source(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I32_I16,
            dst=Register(RegKind.VGPR, 4, count=2),
            src0=Register(RegKind.VGPR, 6),
            src1=Register(RegKind.SGPR, 8),
            op_sel=(1, 0),
        )
        self.assertEqual(str(inst), "v_cvt_pk_i32_i16 v[4:5], v6, s8 op_sel:[1,0]")

    def test_ast_evaluation_matches_oracle(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 1): 0xFFFF_0005,
        }
        node = CvtPkI32I16Node(src0=RegReadNode(Register(RegKind.VGPR, 1)))
        eval_result = node.eval(env)
        self.assertEqual(eval_result, (5, 0xFFFF_FFFF))
        self.assertEqual(eval_result, ref_cvt_pk_i32_i16(0xFFFF_0005))

    def test_lowered_ast_matches_high_level(self) -> None:
        env: Dict[Register, int] = {
            Register(RegKind.VGPR, 0): 0x8000_7FFF,
        }
        high_level = CvtPkI32I16Node(src0=RegReadNode(Register(RegKind.VGPR, 0)))
        d0_node, d1_node = high_level.to_lowered_ast()
        d0_val = d0_node.eval(env)
        d1_val = d1_node.eval(env)
        self.assertEqual((d0_val, d1_val), high_level.eval(env))
        self.assertEqual((d0_val, d1_val), (0x0000_7FFF, 0xFFFF_8000))

    def test_ast_simplification_constant_folding(self) -> None:
        node_lo = SignExtendI16Node(LiteralNode(0xFFFF_0042), high_half=False)
        node_hi = SignExtendI16Node(LiteralNode(0xFFFF_0042), high_half=True)
        s_lo = simplify_ast(node_lo)
        s_hi = simplify_ast(node_hi)
        self.assertIsInstance(s_lo, LiteralNode)
        self.assertIsInstance(s_hi, LiteralNode)
        self.assertEqual(s_lo.eval({}), 0x0000_0042)
        self.assertEqual(s_hi.eval({}), 0xFFFF_FFFF)

# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Architectural register state for a single execution lane."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0
        self.scc: int = 0

    def read_reg(self, reg: Register) -> int:
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR index out of bounds: {reg}")
            if reg.count == 1:
                return self.vgprs[reg.index] & MASK32
            elif reg.count == 2:
                lo = self.vgprs[reg.index] & MASK32
                hi = self.vgprs[reg.index + 1] & MASK32
                return (hi << 32) | lo
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR index out of bounds: {reg}")
            if reg.count == 1:
                return self.sgprs[reg.index] & MASK32
            elif reg.count == 2:
                lo = self.sgprs[reg.index] & MASK32
                hi = self.sgprs[reg.index + 1] & MASK32
                return (hi << 32) | lo
        raise TypeError(f"Unknown register kind: {reg.kind}")

    def write_reg(self, reg: Register, val: int) -> None:
        if reg.kind == RegKind.VGPR:
            if not 0 <= reg.index + reg.count - 1 < 256:
                raise IndexError(f"VGPR index out of bounds: {reg}")
            if reg.count == 1:
                self.vgprs[reg.index] = val & MASK32
            elif reg.count == 2:
                self.vgprs[reg.index] = val & MASK32
                self.vgprs[reg.index + 1] = (val >> 32) & MASK32
        elif reg.kind == RegKind.SGPR:
            if not 0 <= reg.index + reg.count - 1 < 106:
                raise IndexError(f"SGPR index out of bounds: {reg}")
            if reg.count == 1:
                self.sgprs[reg.index] = val & MASK32
            elif reg.count == 2:
                self.sgprs[reg.index] = val & MASK32
                self.sgprs[reg.index + 1] = (val >> 32) & MASK32
        else:
            raise TypeError(f"Unknown register kind: {reg.kind}")

    def read_operand(self, op: Operand) -> int:
        if isinstance(op, Register):
            return self.read_reg(op)
        if isinstance(op, Immediate):
            return op.value & MASK32
        raise TypeError(f"Invalid operand type: {type(op)}")


class ScalarVALUInterpreter:
    """Deterministic single-lane scalar VALU instruction interpreter."""

    def step(self, inst: Instruction, state: ScalarMachineState) -> None:
        if inst.opcode == Opcode.V_CVT_PK_I32_I16:
            s0 = state.read_operand(inst.src0)
            if inst.src1 is None:
                d0, d1 = ref_cvt_pk_i32_i16(s0)
            else:
                s1 = state.read_operand(inst.src1)
                d0, d1 = ref_cvt_pk_i32_i16_dual(s0, s1, inst.op_sel)
            state.write_reg(Register(inst.dst.kind, inst.dst.index, count=1), d0)
            state.write_reg(Register(inst.dst.kind, inst.dst.index + 1, count=1), d1)
        elif inst.opcode == Opcode.V_MOV_B32:
            s0 = state.read_operand(inst.src0)
            state.write_reg(inst.dst, s0)
        else:
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")

    def run(self, program: List[Instruction], state: ScalarMachineState) -> None:
        for inst in program:
            self.step(inst, state)


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestValuCvtPkI32I16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarVALUInterpreter()
        self.state = ScalarMachineState()

    def test_step_single_source_vgpr(self) -> None:
        self.state.write_reg(Register(RegKind.VGPR, 2), 0xFFFF_0010)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I32_I16,
            dst=Register(RegKind.VGPR, 0, count=2),
            src0=Register(RegKind.VGPR, 2),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 0)), 0x0000_0010)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 1)), 0xFFFF_FFFF)

    def test_step_dual_source_with_sgpr(self) -> None:
        self.state.write_reg(Register(RegKind.VGPR, 5), 0x7FFF)
        self.state.write_reg(Register(RegKind.SGPR, 2), 0x8000)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I32_I16,
            dst=Register(RegKind.VGPR, 6, count=2),
            src0=Register(RegKind.VGPR, 5),
            src1=Register(RegKind.SGPR, 2),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 6)), 0x0000_7FFF)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 7)), 0xFFFF_8000)

    def test_step_with_immediate_constant(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I32_I16,
            dst=Register(RegKind.VGPR, 10, count=2),
            src0=Immediate(0xFFFE_0002),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 10)), 0x0000_0002)
        self.assertEqual(self.state.read_reg(Register(RegKind.VGPR, 11)), 0xFFFF_FFFE)

    def test_sequential_program_determinism(self) -> None:
        program = [
            Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 4), Immediate(0x8001_7FFF)),
            Instruction(
                Opcode.V_CVT_PK_I32_I16,
                Register(RegKind.VGPR, 0, count=2),
                Register(RegKind.VGPR, 4),
            ),
        ]
        self.interp.run(program, self.state)
        res_v0 = self.state.read_reg(Register(RegKind.VGPR, 0))
        res_v1 = self.state.read_reg(Register(RegKind.VGPR, 1))
        self.assertEqual(res_v0, 0x0000_7FFF)
        self.assertEqual(res_v1, 0xFFFF_8001)

        fresh_state = ScalarMachineState()
        self.interp.run(program, fresh_state)
        self.assertEqual(fresh_state.read_reg(Register(RegKind.VGPR, 0)), res_v0)
        self.assertEqual(fresh_state.read_reg(Register(RegKind.VGPR, 1)), res_v1)

if __name__ == "__main__":
    unittest.main()
