from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_i16_f32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the packed float-to-integer conversion instruction:
  v_cvt_pk_i16_f32 dst, src0, src1

RDNA2 VALU Instruction Specification:
  v_cvt_pk_i16_f32 D.u32, S0.f32, S1.f32
  Operands:
    - D:  32-bit VGPR destination (holding two packed signed 16-bit integers)
    - S0: 32-bit float source (VGPR/SGPR/literal/inline constant), providing low 16 bits
    - S1: 32-bit float source (VGPR/SGPR/literal/inline constant), providing high 16 bits

  Mathematical Operation:
    D[15:0]  = f32_to_i16(S0)
    D[31:16] = f32_to_i16(S1)

    where f32_to_i16(f) truncates single-precision IEEE 754 float toward zero (RTZ)
    and saturates to the signed 16-bit integer range [-32768, 32767]:
      - If f is NaN: returns 0 (0x0000) per DirectX / AMD RDNA2 specification.
      - If f >= 32767.0f (or +infinity): saturates to 32767 (0x7FFF).
      - If f <= -32768.0f (or -infinity): saturates to -32768 (0x8000).
      - Otherwise: returns int(f) & 0xFFFF (truncation toward zero).

    Packed Destination Dword:
      D = (f32_to_i16(S0) & 0xFFFF) | ((f32_to_i16(S1) & 0xFFFF) << 16)

  Key Architectural Properties:
    1. Independent saturation: S0 and S1 are converted and saturated independently.
    2. Truncation toward zero (RTZ): Fractional values round toward 0 (+1.9 -> 1, -1.9 -> -1).
    3. Low/High Packing: S0 occupies the lower 16-bit halfword D[15:0], while S1 occupies
       the upper 16-bit halfword D[31:16].
    4. Destination is always a VGPR: The full 32-bit destination register is overwritten.
    5. Special values: NaN produces 0; ±0.0 produces 0; subnormals truncate to 0.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import math
import struct
from typing import Union
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000
SIGN_BIT16: int = 0x8000

INT16_MIN: int = -32768
INT16_MAX: int = 32767

INT16_MIN_U16: int = 0x8000
INT16_MAX_U16: int = 0x7FFF

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000
NEG_ONE_F32: int = 0xBF800000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as a signed two's-complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & SIGN_BIT16) else val


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_inf_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision infinity."""
    bits &= MASK32
    return (bits & 0x7FFFFFFF) == 0x7F800000


def f32_to_i16(bits: int, *, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Convert float32 bit pattern to saturated signed 16-bit integer (unsigned 16-bit word)."""
    b = bits & MASK32
    if abs_mod:
        b &= ~SIGN_BIT32
    if neg_mod:
        b ^= SIGN_BIT32

    # NaN returns 0 per AMD RDNA2 / DirectX specification
    if is_nan_f32(b):
        return 0

    val = bits_to_f32(b)

    # Positive saturation / +infinity
    if math.isinf(val) and val > 0:
        return INT16_MAX_U16
    if math.isinf(val) and val < 0:
        return INT16_MIN_U16

    if val >= 32767.0:
        return INT16_MAX_U16
    if val <= -32768.0:
        return INT16_MIN_U16

    # Truncation toward zero
    truncated = int(val)
    return truncated & MASK16


def pack_i16_to_u32(lo_i16: int, hi_i16: int) -> int:
    """Pack two 16-bit integers into a 32-bit unsigned dword (lo in [15:0], hi in [31:16])."""
    return ((hi_i16 & MASK16) << 16) | (lo_i16 & MASK16)


def unpack_i16_from_u32(val32: int) -> tuple[int, int]:
    """Unpack a 32-bit unsigned dword into low and high 16-bit words (lo, hi)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def ref_cvt_pk_i16_f32(
    src0: int,
    src1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_i16_f32.

    Converts two 32-bit single-precision floats into saturated 16-bit signed integers and
    packs them into a single 32-bit dword:
      D[15:0]  = f32_to_i16(S0)
      D[31:16] = f32_to_i16(S1)
    """
    lo = f32_to_i16(src0, abs_mod=abs0, neg_mod=neg0)
    hi = f32_to_i16(src1, abs_mod=abs1, neg_mod=neg1)
    return pack_i16_to_u32(lo, hi)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """RDNA architectural register."""
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Register index must be non-negative: {self.index}")
        if self.kind == RegKind.VGPR and self.index >= 256:
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.kind == RegKind.SGPR and self.index >= 106:
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    @property
    def name(self) -> str:
        return f"{self.kind.value}{self.index}"

    def to_assembly(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class Immediate:
    """32-bit immediate or inline constant."""
    value: int

    def to_assembly(self) -> str:
        val = s32(self.value)
        if -16 <= val <= 64:
            return str(val)
        return f"{u32(self.value):#x}"


Operand = Union[Register, Immediate]


class Opcode(enum.Enum):
    """Opcode definitions for VALU and SALU instructions."""
    V_CVT_PK_I16_F32 = "v_cvt_pk_i16_f32"
    V_MOV_B32 = "v_mov_b32"
    S_MOV_B32 = "s_mov_b32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """RDNA2 instruction representation."""
    opcode: Opcode
    dst: Register
    src0: Operand
    src1: Operand | None = None
    modifiers: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.dst, Register):
            raise TypeError(f"Destination must be a Register, got {type(self.dst).__name__}")
        if self.opcode == Opcode.V_CVT_PK_I16_F32:
            if self.dst.kind != RegKind.VGPR:
                raise TypeError(f"Destination for {self.opcode.value} must be a VGPR, got {self.dst.kind.name}")
            if self.src1 is None:
                raise ValueError(f"{self.opcode.value} requires two source operands")

    def to_assembly(self) -> str:
        parts = [self.opcode.value, self.dst.to_assembly()]
        parts.append(self.src0.to_assembly())
        if self.src1 is not None:
            parts.append(self.src1.to_assembly())
        mods = " ".join(self.modifiers)
        line = f"{parts[0]} {parts[1]}, {', '.join(parts[2:])}"
        if mods:
            line = f"{line} {mods}"
        return line


@dataclasses.dataclass
class Program:
    """Sequence of instructions forming a shader basic block."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def emit(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class ASTNode:
    """Base class for all symbolic AST nodes."""
    pass


class ASTExpr(ASTNode):
    """Base class for typed symbolic expressions."""
    bitwidth: int = 32


@dataclasses.dataclass(frozen=True)
class VarExpr(ASTExpr):
    """Symbolic variable operand (e.g. register contents)."""
    name: str
    bitwidth: int = 32


@dataclasses.dataclass(frozen=True)
class ConstExpr(ASTExpr):
    """Symbolic constant expression."""
    value: int
    bitwidth: int = 32


@dataclasses.dataclass(frozen=True)
class F32ToI16Expr(ASTExpr):
    """Symbolic float-to-signed-16-bit integer conversion operator."""
    child: ASTExpr
    bitwidth: int = 16


@dataclasses.dataclass(frozen=True)
class PackI16Expr(ASTExpr):
    """Symbolic packing of two 16-bit expressions into a 32-bit dword."""
    lo: ASTExpr
    hi: ASTExpr
    bitwidth: int = 32


def eval_ast(node: ASTExpr, env: dict[str, int]) -> int:
    """Evaluate symbolic AST expression deterministically in an integer environment."""
    if isinstance(node, VarExpr):
        if node.name not in env:
            raise KeyError(f"Variable {node.name} not found in environment")
        return u32(env[node.name])
    if isinstance(node, ConstExpr):
        return u32(node.value)
    if isinstance(node, F32ToI16Expr):
        val = eval_ast(node.child, env)
        return f32_to_i16(val)
    if isinstance(node, PackI16Expr):
        lo = eval_ast(node.lo, env)
        hi = eval_ast(node.hi, env)
        return pack_i16_to_u32(lo, hi)
    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


def instruction_to_ast(inst: Instruction) -> PackI16Expr:
    """Translate a v_cvt_pk_i16_f32 instruction into its canonical typed symbolic AST."""
    if inst.opcode != Opcode.V_CVT_PK_I16_F32:
        raise ValueError(f"Cannot translate non-pack opcode to PackI16Expr: {inst.opcode}")
    assert inst.src1 is not None

    def operand_to_ast(op: Operand) -> ASTExpr:
        if isinstance(op, Register):
            return VarExpr(name=op.name, bitwidth=32)
        elif isinstance(op, Immediate):
            return ConstExpr(value=op.value, bitwidth=32)
        raise TypeError(f"Unsupported operand: {type(op).__name__}")

    s0_expr = operand_to_ast(inst.src0)
    s1_expr = operand_to_ast(inst.src1)
    return PackI16Expr(lo=F32ToI16Expr(s0_expr), hi=F32ToI16Expr(s1_expr))


class TestValuCvtPkI16F32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_in_range_values_truncation(self) -> None:
        """Verify integer conversions and truncation toward zero."""
        test_cases = [
            (f32_to_bits(0.0), f32_to_bits(0.0), 0x00000000),
            (f32_to_bits(1.0), f32_to_bits(0.0), 0x00000001),
            (f32_to_bits(0.0), f32_to_bits(1.0), 0x00010000),
            (f32_to_bits(10.0), f32_to_bits(-10.0), 0xFFF6000A),
            (f32_to_bits(1.9), f32_to_bits(-1.9), 0xFFFF0001),   # RTZ: 1.9 -> 1, -1.9 -> -1
            (f32_to_bits(0.999), f32_to_bits(-0.999), 0x00000000), # RTZ to 0
            (f32_to_bits(32766.0), f32_to_bits(-32767.0), 0x80017FFE),
            (f32_to_bits(32767.0), f32_to_bits(-32768.0), 0x80007FFF),
        ]
        for s0, s1, expected in test_cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                res = ref_cvt_pk_i16_f32(s0, s1)
                self.assertEqual(res, expected)

    def test_positive_overflow_saturation(self) -> None:
        """Values >= 32767.0f and +infinity saturate to 32767 (0x7FFF)."""
        overflow_values = [
            f32_to_bits(32767.0),
            f32_to_bits(32768.0),
            f32_to_bits(65536.0),
            f32_to_bits(1e6),
            POS_INF_F32,
        ]
        for val in overflow_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_pk_i16_f32(val, 0), 0x00007FFF)
                self.assertEqual(ref_cvt_pk_i16_f32(0, val), 0x7FFF0000)
                self.assertEqual(ref_cvt_pk_i16_f32(val, val), 0x7FFF7FFF)

    def test_negative_underflow_saturation(self) -> None:
        """Values <= -32768.0f and -infinity saturate to -32768 (0x8000)."""
        underflow_values = [
            f32_to_bits(-32768.0),
            f32_to_bits(-32769.0),
            f32_to_bits(-65536.0),
            f32_to_bits(-1e6),
            NEG_INF_F32,
        ]
        for val in underflow_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_pk_i16_f32(val, 0), 0x00008000)
                self.assertEqual(ref_cvt_pk_i16_f32(0, val), 0x80000000)
                self.assertEqual(ref_cvt_pk_i16_f32(val, val), 0x80008000)

    def test_nan_returns_zero(self) -> None:
        """NaN values produce 0 (0x0000)."""
        for nan_val in [QNAN_F32, SNAN_F32, 0x7FC01234, 0xFFC00000, 0xFF800001]:
            with self.subTest(nan=hex(nan_val)):
                self.assertEqual(f32_to_i16(nan_val), 0)
                self.assertEqual(ref_cvt_pk_i16_f32(nan_val, nan_val), 0x00000000)

    def test_input_modifiers(self) -> None:
        """abs and neg modifiers apply prior to conversion."""
        # abs(-10.0f) -> 10 -> 0x000A
        neg_ten = f32_to_bits(-10.0)
        self.assertEqual(ref_cvt_pk_i16_f32(neg_ten, 0, abs0=True), 0x0000000A)

        # neg(10.0f) -> -10 -> 0xFFF6
        pos_ten = f32_to_bits(10.0)
        self.assertEqual(ref_cvt_pk_i16_f32(0, pos_ten, neg1=True), 0xFFF60000)


class TestValuCvtPkI16F32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_assembly_formatting(self) -> None:
        inst1 = Instruction(
            opcode=Opcode.V_CVT_PK_I16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst1.to_assembly(), "v_cvt_pk_i16_f32 v0, v1, v2")

        inst2 = Instruction(
            opcode=Opcode.V_CVT_PK_I16_F32,
            dst=Register(RegKind.VGPR, 5),
            src0=Register(RegKind.SGPR, 2),
            src1=Immediate(f32_to_bits(1.0)),
        )
        self.assertEqual(inst2.to_assembly(), "v_cvt_pk_i16_f32 v5, s2, 0x3f800000")

    def test_program_emission(self) -> None:
        prog = Program()
        prog.emit(Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 0), Immediate(POS_ONE_F32)))
        prog.emit(Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 1), Immediate(NEG_ONE_F32)))
        prog.emit(Instruction(
            Opcode.V_CVT_PK_I16_F32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.SGPR, 0),
        ))
        expected = (
            "s_mov_b32 s0, 0x3f800000\n"
            "v_mov_b32 v1, 0xbf800000\n"
            "v_cvt_pk_i16_f32 v0, v1, s0"
        )
        self.assertEqual(prog.to_assembly(), expected)

    def test_symbolic_ast_evaluation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        ast_tree = instruction_to_ast(inst)
        self.assertIsInstance(ast_tree, PackI16Expr)
        self.assertIsInstance(ast_tree.lo, F32ToI16Expr)
        self.assertIsInstance(ast_tree.hi, F32ToI16Expr)

        # Evaluate with test environment
        env = {"v1": f32_to_bits(100.5), "v2": f32_to_bits(-50.5)}
        res = eval_ast(ast_tree, env)
        expected = ref_cvt_pk_i16_f32(f32_to_bits(100.5), f32_to_bits(-50.5))
        self.assertEqual(res, expected)
        self.assertEqual(res, ((s32(-50) & 0xFFFF) << 16) | (100 & 0xFFFF))

    def test_ast_immutability(self) -> None:
        reg = Register(RegKind.VGPR, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            reg.index = 1  # type: ignore[misc]

        imm = Immediate(42)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            imm.value = 43  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
