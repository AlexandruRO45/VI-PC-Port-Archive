from __future__ import annotations

"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_div_fixup_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, 16-bit half-precision floating-point operations
  operate on 16-bit values (IEEE 754 binary16): vd from source operands src0, src1, and src2.

  Instruction:
    v_div_fixup_f16 vd, src0, src1, src2 [clamp] [omod]

  Operands:
    src0 = provisional quotient from previous reciprocal/FMAS step
    src1 = denominator
    src2 = numerator

  IEEE-754 binary16 encoding:
    - 1 sign bit (bit 15)
    - 5 exponent bits with bias 15 (bits 14:10)
    - 10 mantissa / fraction bits (bits 9:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

from enum import Enum, auto
import math
import struct
import unittest

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE-754 binary16 bit constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_INDEFINITE_QNAN = 0xFE00  # Default indefinite NaN returned by AMD GPU for 0/0 and inf/inf
F16_QUIET_BIT = 0x0200

F16_EXP_MASK = 0x7C00
F16_MANT_MASK = 0x03FF
F16_SIGN_MASK = 0x8000

F16_ONE = 0x3C00            # 1.0
F16_NEG_ONE = 0xBC00        # -1.0
F16_TWO = 0x4000            # 2.0
F16_NEG_TWO = 0xC000        # -2.0
F16_HALF = 0x3800           # 0.5
F16_FOUR = 0x4400           # 4.0
F16_QUARTER = 0x3400        # 0.25
F16_MIN_NORMAL = 0x0400     # 2^-14
F16_MAX_NORMAL = 0x7BFF     # (2 - 2^-10) * 2^15 = 65504.0
F16_MIN_SUBNORMAL = 0x0001  # 2^-24
F16_MAX_SUBNORMAL = 0x03FF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 2^16 - 1]."""
    return val & MASK16


def f16_to_u16(val: float) -> int:
    """Pack an IEEE-754 half precision float into an unsigned 16-bit integer."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_NEG_INF if val < 0 else F16_POS_INF


def u16_to_f16(bits: int) -> float:
    """Unpack an unsigned 16-bit integer into an IEEE-754 half precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_f16_nan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) != 0)


def is_f16_snan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a signaling NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) == 0)


def is_f16_qnan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a quiet NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) != 0)


def is_f16_zero(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +0.0 or -0.0."""
    return (bits & (F16_EXP_MASK | F16_MANT_MASK)) == 0


def is_f16_inf(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +inf or -inf."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) == 0)


def is_f16_subnormal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a subnormal/denormal float."""
    return ((bits & F16_EXP_MASK) == 0) and ((bits & F16_MANT_MASK) != 0)


def is_f16_normal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a normalized finite float."""
    exp = (bits >> 10) & 0x1F
    return 0 < exp < 0x1F


def f16_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 15) & 1


def f16_unbiased_exponent(bits: int) -> int:
    """Extract true mathematical binary exponent for finite non-zero value."""
    val = abs(u16_to_f16(bits))
    if val == 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    _, e = math.frexp(val)
    return e - 1


def quiet_f16(bits: int) -> int:
    """Force an IEEE-754 half NaN to quiet by setting the quiet bit."""
    return (bits & MASK16) | F16_QUIET_BIT


def apply_src_mods_f16(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg)."""
    res = bits & MASK16
    if abs_val:
        res &= ~F16_SIGN_MASK
    if neg:
        res ^= F16_SIGN_MASK
    return res


def apply_clamp_f16(bits: int) -> int:
    """Apply VOP3 clamp modifier to half precision float."""
    if is_f16_nan(bits) or bits == F16_NEG_ZERO:
        return F16_POS_ZERO
    val = u16_to_f16(bits)
    if val < 0.0:
        return F16_POS_ZERO
    elif val > 1.0:
        return F16_ONE
    return bits


def apply_omod_f16(bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0 or is_f16_nan(bits) or is_f16_inf(bits) or is_f16_zero(bits):
        return bits
    val = u16_to_f16(bits)
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")
    return f16_to_u16(val)


def ref_div_fixup_f16(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_div_fixup_f16 instruction."""
    s0 = apply_src_mods_f16(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f16(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f16(s2_bits, neg=src2_neg, abs_val=src2_abs)

    sign1 = f16_sign(s1)
    sign2 = f16_sign(s2)
    sign_out = sign1 ^ sign2

    # Decision tree according to AMD RDNA2 ISA specification
    if is_f16_nan(s2):
        res = quiet_f16(s2)
    elif is_f16_nan(s1):
        res = quiet_f16(s1)
    elif is_f16_zero(s1) and is_f16_zero(s2):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = F16_INDEFINITE_QNAN
    elif is_f16_inf(s1) and is_f16_inf(s2):
        # inf / inf -> Indefinite QNaN
        res = F16_INDEFINITE_QNAN
    elif is_f16_zero(s1) or is_f16_inf(s2):
        # x / 0 or inf / y -> signed infinity
        res = F16_NEG_INF if sign_out else F16_POS_INF
    elif is_f16_inf(s1) or is_f16_zero(s2):
        # x / inf or 0 / y -> signed zero
        res = F16_NEG_ZERO if sign_out else F16_POS_ZERO
    elif is_f16_nan(s0):
        # Intermediate quotient overflowed to NaN -> signed infinity
        res = F16_NEG_INF if sign_out else F16_POS_INF
    elif (f16_unbiased_exponent(s2) - f16_unbiased_exponent(s1)) < -25:
        # Exponent underflow threshold (-25 for binary16 subnormal limit)
        res = F16_NEG_ZERO if sign_out else F16_POS_ZERO
    else:
        # Standard case: sign_out applied to absolute magnitude of s0
        res = (s0 & ~F16_SIGN_MASK) | (F16_SIGN_MASK if sign_out else 0)

    # Output modifiers
    if omod != 0:
        res = apply_omod_f16(res, omod)

    if clamp:
        res = apply_clamp_f16(res)

    return res & MASK16


def ref_div_fixup_f16_float(s0: float, s1: float, s2: float, **kwargs) -> float:
    """Float wrapper for ref_div_fixup_f16."""
    b0 = f16_to_u16(s0)
    b1 = f16_to_u16(s1)
    b2 = f16_to_u16(s2)
    res_bits = ref_div_fixup_f16(b0, b1, b2, **kwargs)
    return u16_to_f16(res_bits)


class HalfRegister:
    """Architectural model of a 16-bit register value."""

    def __init__(self, val_u16: int = 0) -> None:
        self._val = val_u16 & MASK16

    @property
    def as_u16(self) -> int:
        return self._val

    @as_u16.setter
    def as_u16(self, val: int) -> None:
        self._val = val & MASK16

    @property
    def as_f16(self) -> float:
        return u16_to_f16(self._val)

    @as_f16.setter
    def as_f16(self, val: float) -> None:
        self._val = f16_to_u16(val)

    @classmethod
    def from_u16(cls, val: int) -> HalfRegister:
        return cls(val)

    @classmethod
    def from_f16(cls, val: float) -> HalfRegister:
        return cls(f16_to_u16(val))

    def __repr__(self) -> str:
        return f"HalfRegister({self._val:#06x}, f16={self.as_f16})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HalfRegister):
            return NotImplemented
        return self._val == other._val


class TestDivFixupF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_normal_quotient_fixup(self):
        # 6.0 / 2.0 = 3.0, s0 is estimated 3.0
        s0 = f16_to_u16(3.0)
        s1 = f16_to_u16(2.0)
        s2 = f16_to_u16(6.0)
        res = ref_div_fixup_f16(s0, s1, s2)
        self.assertEqual(res, f16_to_u16(3.0))

        # Sign correction
        s2_neg = f16_to_u16(-6.0)
        res_neg = ref_div_fixup_f16(s0, s1, s2_neg)
        self.assertEqual(res_neg, f16_to_u16(-3.0))

    def test_division_by_zero(self):
        s0 = f16_to_u16(1.0)
        res_pz = ref_div_fixup_f16(s0, F16_POS_ZERO, f16_to_u16(1.0))
        self.assertEqual(res_pz, F16_POS_INF)

        res_nz = ref_div_fixup_f16(s0, F16_NEG_ZERO, f16_to_u16(1.0))
        self.assertEqual(res_nz, F16_NEG_INF)

    def test_zero_numerator(self):
        s0 = f16_to_u16(0.0)
        res = ref_div_fixup_f16(s0, f16_to_u16(4.0), F16_POS_ZERO)
        self.assertEqual(res, F16_POS_ZERO)

        res_nz = ref_div_fixup_f16(s0, f16_to_u16(4.0), F16_NEG_ZERO)
        self.assertEqual(res_nz, F16_NEG_ZERO)

    def test_zero_divided_by_zero(self):
        res = ref_div_fixup_f16(F16_POS_ZERO, F16_POS_ZERO, F16_POS_ZERO)
        self.assertEqual(res, F16_INDEFINITE_QNAN)

    def test_infinity_divided_by_infinity(self):
        res = ref_div_fixup_f16(F16_POS_INF, F16_POS_INF, F16_POS_INF)
        self.assertEqual(res, F16_INDEFINITE_QNAN)

    def test_finite_divided_by_infinity(self):
        res = ref_div_fixup_f16(f16_to_u16(0.0), F16_POS_INF, f16_to_u16(3.0))
        self.assertEqual(res, F16_POS_ZERO)

    def test_infinity_divided_by_finite(self):
        res = ref_div_fixup_f16(f16_to_u16(1.0), f16_to_u16(3.0), F16_POS_INF)
        self.assertEqual(res, F16_POS_INF)

    def test_nan_numerator_and_denominator_priority(self):
        nan2 = 0x7E20
        nan1 = 0x7E10
        res = ref_div_fixup_f16(F16_ONE, nan1, nan2)
        self.assertEqual(res, quiet_f16(nan2))

        res1 = ref_div_fixup_f16(F16_ONE, nan1, F16_ONE)
        self.assertEqual(res1, quiet_f16(nan1))

    def test_s0_nan_overflow(self):
        res = ref_div_fixup_f16(F16_CANONICAL_QNAN, F16_ONE, F16_ONE)
        self.assertEqual(res, F16_POS_INF)

        res_neg = ref_div_fixup_f16(F16_CANONICAL_QNAN, F16_ONE, F16_NEG_ONE)
        self.assertEqual(res_neg, F16_NEG_INF)

    def test_exponent_underflow(self):
        # Exponent diff < -25 produces signed zero
        s2 = f16_to_u16(2.0 ** -14)
        s1 = f16_to_u16(2.0 ** 14)
        res = ref_div_fixup_f16(F16_ONE, s1, s2)
        self.assertEqual(res, F16_POS_ZERO)

    def test_source_and_output_modifiers(self):
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(1.0)
        s2 = f16_to_u16(2.0)

        # src2 neg
        res_neg2 = ref_div_fixup_f16(s0, s1, s2, src2_neg=True)
        self.assertEqual(res_neg2, f16_to_u16(-2.0))

        # omod: mul:2
        res_omod = ref_div_fixup_f16(s0, s1, s2, omod=1)
        self.assertEqual(res_omod, f16_to_u16(4.0))

        # clamp
        res_clamp = ref_div_fixup_f16(f16_to_u16(5.0), s1, s2, clamp=True)
        self.assertEqual(res_clamp, F16_ONE)

    def test_half_register_model(self):
        reg = HalfRegister.from_f16(2.5)
        self.assertEqual(reg.as_f16, 2.5)
        self.assertEqual(reg.as_u16, f16_to_u16(2.5))

        reg.as_u16 = F16_ONE
        self.assertEqual(reg.as_f16, 1.0)


class Opcode(Enum):
    V_DIV_FIXUP_F16 = "v_div_fixup_f16"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT16 = "f16"
    UINT16 = "u16"


class OMod(Enum):
    NONE = 0
    MUL2 = 1
    MUL4 = 2
    DIV2 = 3

    def to_asm(self) -> str:
        if self == OMod.MUL2:
            return "mul:2"
        elif self == OMod.MUL4:
            return "mul:4"
        elif self == OMod.DIV2:
            return "div:2"
        return ""


class ASTOperand:
    """Symbolic AST operand representing a 16-bit register, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT16,
    ) -> None:
        self.kind = kind
        self.index = index
        self.value = value
        self.neg = neg
        self.abs_ = abs_
        self.type_kind = type_kind

    @classmethod
    def vgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR:
            body = f"v{self.index}"
        elif self.kind == OperandKind.SGPR:
            body = f"s{self.index}"
        elif self.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            body = str(self.value)
        else:
            raise ValueError(f"Unknown operand kind: {self.kind}")

        if self.abs_:
            body = f"|{body}|"
        if self.neg:
            body = f"-{body}"
        return body

    def __repr__(self) -> str:
        return f"ASTOperand({self.disasm()}, type={self.type_kind.value})"


class ASTInstruction:
    """Symbolic AST instruction representation for v_div_fixup_f16."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        src0: ASTOperand,
        src1: ASTOperand,
        src2: ASTOperand,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def validate(self) -> None:
        if self.opcode != Opcode.V_DIV_FIXUP_F16:
            raise ValueError(f"Invalid opcode for v_div_fixup_f16 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.dst.type_kind != TypeKind.FLOAT16:
            raise TypeError(f"Destination type must be FLOAT16, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT16:
                raise TypeError(f"Source operand {i} type must be FLOAT16, got {src.type_kind}")
            if src.kind == OperandKind.VGPR and not (0 <= src.index <= 255):
                raise ValueError(f"Source {i} VGPR index out of range: {src.index}")
            if src.kind == OperandKind.SGPR and not (0 <= src.index <= 105):
                raise ValueError(f"Source {i} SGPR index out of range: {src.index}")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.src0.disasm()},",
            f"{self.src1.disasm()},",
            self.src2.disasm(),
        ]
        if self.omod != OMod.NONE:
            parts.append(self.omod.to_asm())
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"ASTInstruction({self.disasm()})"


class TypedExpr:
    """Base class for typed symbolic expressions."""

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Named variable in symbolic expression tree (e.g. representing a register value)."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in evaluation environment")
        return env[self.name]

    def __repr__(self) -> str:
        return f"Var({self.name})"


class ConstantExpr(TypedExpr):
    """Constant literal in symbolic expression tree."""

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier applied to a typed expression ('neg' or 'abs')."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT16) -> None:
        super().__init__(type_kind)
        if op not in ("neg", "abs"):
            raise ValueError(f"Unknown unary operator: {op}")
        self.op = op
        self.expr = expr

    def evaluate(self, env: dict[str, float]) -> float:
        val = self.expr.evaluate(env)
        if self.op == "abs":
            return math.fabs(val)
        elif self.op == "neg":
            return -val
        return val

    def simplify(self) -> TypedExpr:
        sub = self.expr.simplify()
        if isinstance(sub, ConstantExpr):
            if self.op == "abs":
                return ConstantExpr(math.fabs(sub.value), self.type_kind)
            elif self.op == "neg":
                return ConstantExpr(-sub.value, self.type_kind)
        return UnaryOpExpr(self.op, sub, self.type_kind)

    def __repr__(self) -> str:
        return f"{self.op}({self.expr})"


class DivFixupF16Expr(TypedExpr):
    """Symbolic DivFixupF16 ternary expression: div_fixup(src0, src1, src2) with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT16,
    ) -> None:
        super().__init__(type_kind)
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def evaluate(self, env: dict[str, float]) -> float:
        v0 = self.src0.evaluate(env)
        v1 = self.src1.evaluate(env)
        v2 = self.src2.evaluate(env)
        return ref_div_fixup_f16_float(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res = ref_div_fixup_f16_float(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res, self.type_kind)
        return DivFixupF16Expr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"DivFixupF16({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


class TestDivFixupF16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_fixup_f16 v0, v1, v2, v3")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1, neg=True, abs_=True),
            src1=ASTOperand.sgpr(2),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_fixup_f16 v0, -|v1|, s2, 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind (SGPR)
        bad_dst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        expr = DivFixupF16Expr(
            VariableExpr("q"),
            VariableExpr("denom"),
            VariableExpr("num"),
        )
        env = {"q": 2.0, "denom": 3.0, "num": 6.0}
        self.assertEqual(expr.evaluate(env), 2.0)

    def test_typed_expression_constant_folding(self):
        c_expr = DivFixupF16Expr(
            ConstantExpr(2.0),
            ConstantExpr(3.0),
            ConstantExpr(6.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 2.0)

        # Unary folding
        u_expr = UnaryOpExpr("neg", ConstantExpr(1.5))
        u_folded = u_expr.simplify()
        self.assertIsInstance(u_folded, ConstantExpr)
        self.assertEqual(u_folded.value, -1.5)


class ScalarRegisterFile:
    """Register file for a single scalar lane with 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_vgpr_u16(self, idx: int) -> int:
        return self.vgprs[idx] & MASK16

    def write_vgpr_u16(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK16

    def read_vgpr_f16(self, idx: int) -> float:
        return u16_to_f16(self.read_vgpr_u16(idx))

    def write_vgpr_f16(self, idx: int, val: float) -> None:
        self.write_vgpr_u16(idx, f16_to_u16(val))

    def read_sgpr_u16(self, idx: int) -> int:
        return self.sgprs[idx] & MASK16

    def write_sgpr_u16(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK16

    def read_sgpr_f16(self, idx: int) -> float:
        return u16_to_f16(self.read_sgpr_u16(idx))

    def write_sgpr_f16(self, idx: int, val: float) -> None:
        self.write_sgpr_u16(idx, f16_to_u16(val))


class DivFixupF16ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_div_fixup_f16."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_u16(self, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR:
            val_u16 = self.reg_file.read_vgpr_u16(src.index)
        elif src.kind == OperandKind.SGPR:
            val_u16 = self.reg_file.read_sgpr_u16(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u16 = f16_to_u16(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_f16(val_u16, neg=src.neg, abs_val=src.abs_)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        s0 = self.resolve_source_u16(inst.src0)
        s1 = self.resolve_source_u16(inst.src1)
        s2 = self.resolve_source_u16(inst.src2)

        res = ref_div_fixup_f16(s0, s1, s2, clamp=inst.clamp, omod=inst.omod.value)
        self.reg_file.write_vgpr_u16(inst.dst.index, res)


class TestDivFixupF16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = DivFixupF16ScalarInterpreter(self.rf)

    def test_scalar_interpreter_vgprs(self):
        # 8.0 / 2.0 = 4.0
        self.rf.write_vgpr_f16(1, 4.0)   # s0 (estimated quotient)
        self.rf.write_vgpr_f16(2, 2.0)   # s1 (denom)
        self.rf.write_vgpr_f16(3, 8.0)   # s2 (num)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_f16(0), 4.0)

    def test_scalar_interpreter_sgpr_and_constants(self):
        self.rf.write_sgpr_f16(0, 3.0)
        self.rf.write_vgpr_f16(1, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.inline_const(6.0),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_f16(0), 3.0)

    def test_scalar_interpreter_modifiers(self):
        self.rf.write_vgpr_f16(1, 1.5)
        self.rf.write_vgpr_f16(2, 2.0)
        self.rf.write_vgpr_f16(3, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3, neg=True),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_f16(0), -1.5)

    def test_scalar_interpreter_deterministic_repeatability(self):
        self.rf.write_vgpr_f16(1, 1.25)
        self.rf.write_vgpr_f16(2, 2.5)
        self.rf.write_vgpr_f16(3, -3.125)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(inst)
        res1 = self.rf.read_vgpr_u16(0)

        self.interp.execute(inst)
        res2 = self.rf.read_vgpr_u16(0)
        self.assertEqual(res1, res2)


class SIMDWaveState:
    """Architectural model of a SIMD wavefront (Wave32 or Wave64) with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 106

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    @exec_mask.setter
    def exec_mask(self, val: int) -> None:
        mask_limit = (1 << self.wave_size) - 1
        self._exec_mask = val & mask_limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    @exec_lo.setter
    def exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    @exec_hi.setter
    def exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def set_lane_vgpr_u16(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK16

    def get_lane_vgpr_u16(self, lane: int, idx: int) -> int:
        return self.vgprs[lane][idx] & MASK16

    def set_lane_vgpr_f16(self, lane: int, idx: int, val: float) -> None:
        self.set_lane_vgpr_u16(lane, idx, f16_to_u16(val))

    def get_lane_vgpr_f16(self, lane: int, idx: int) -> float:
        return u16_to_f16(self.get_lane_vgpr_u16(lane, idx))

    def set_sgpr_u16(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK16

    def get_sgpr_u16(self, idx: int) -> int:
        return self.sgprs[idx] & MASK16

    def set_sgpr_f16(self, idx: int, val: float) -> None:
        self.set_sgpr_u16(idx, f16_to_u16(val))

    def get_sgpr_f16(self, idx: int) -> float:
        return u16_to_f16(self.get_sgpr_u16(idx))

    def _resolve_operand_u16(self, src: ASTOperand, lane: int) -> int:
        if src.kind == OperandKind.VGPR:
            val_u16 = self.get_lane_vgpr_u16(lane, src.index)
        elif src.kind == OperandKind.SGPR:
            val_u16 = self.get_sgpr_u16(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u16 = f16_to_u16(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_f16(val_u16, neg=src.neg, abs_val=src.abs_)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                s0 = self._resolve_operand_u16(inst.src0, lane)
                s1 = self._resolve_operand_u16(inst.src1, lane)
                s2 = self._resolve_operand_u16(inst.src2, lane)
                res = ref_div_fixup_f16(s0, s1, s2, clamp=inst.clamp, omod=inst.omod.value)
                self.set_lane_vgpr_u16(lane, inst.dst.index, res)


class TestDivFixupF16SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD = 0xBEEF

    def test_wave32_full_exec(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.set_lane_vgpr_f16(lane, 1, float(lane))       # s0 (approx)
            wave.set_lane_vgpr_f16(lane, 2, 2.0)               # s1 (denom)
            wave.set_lane_vgpr_f16(lane, 3, float(lane) * 2.0) # s2 (num)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute(inst)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), float(lane))

    def test_wave64_full_exec(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0xFFFFFFFF
        for lane in range(64):
            wave.set_lane_vgpr_f16(lane, 1, float(lane) * 0.5)
            wave.set_lane_vgpr_f16(lane, 2, 4.0)
            wave.set_lane_vgpr_f16(lane, 3, float(lane) * 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute(inst)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), float(lane) * 0.5)

    def test_wave64_divergence_exec_masking(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0x55555555
        wave.exec_hi = 0xAAAAAAAA

        for lane in range(64):
            wave.set_lane_vgpr_u16(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_f16(lane, 1, 3.0)
            wave.set_lane_vgpr_f16(lane, 2, 2.0)
            wave.set_lane_vgpr_f16(lane, 3, 6.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute(inst)

        for lane in range(64):
            is_active = bool((wave.exec_mask >> lane) & 1)
            if is_active:
                self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), 3.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_u16(lane, 0), self.CANARY_OLD)

    def test_wave64_halves_masking(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        for lane in range(64):
            wave.set_lane_vgpr_u16(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_f16(lane, 1, 1.0)
            wave.set_lane_vgpr_f16(lane, 2, 1.0)
            wave.set_lane_vgpr_f16(lane, 3, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute(inst)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), 1.0)
        for lane in range(32, 64):
            self.assertEqual(wave.get_lane_vgpr_u16(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 1 << 17  # only lane 17

        for lane in range(64):
            wave.set_lane_vgpr_u16(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_f16(lane, 1, 7.5)
            wave.set_lane_vgpr_f16(lane, 2, 2.0)
            wave.set_lane_vgpr_f16(lane, 3, 15.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute(inst)

        for lane in range(64):
            if lane == 17:
                self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), 7.5)
            else:
                self.assertEqual(wave.get_lane_vgpr_u16(lane, 0), self.CANARY_OLD)

    def test_zero_exec_preserves_all_lanes(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 0

        for lane in range(64):
            wave.set_lane_vgpr_u16(lane, 0, self.CANARY_OLD)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(1.0),
            src1=ASTOperand.inline_const(1.0),
            src2=ASTOperand.inline_const(1.0),
        )
        wave.execute(inst)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr_u16(lane, 0), self.CANARY_OLD)


class TestDivFixupF16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_equals_src0_inplace_fixup(self):
        # In-place quotient update: v0 = div_fixup(v0, v1, v2)
        rf = ScalarRegisterFile()
        interp = DivFixupF16ScalarInterpreter(rf)

        rf.write_vgpr_f16(0, 3.0)   # initial estimate in v0
        rf.write_vgpr_f16(1, 2.0)   # denom
        rf.write_vgpr_f16(2, 6.0)   # num

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_f16(0), 3.0)

    def test_dst_equals_src1_inplace(self):
        rf = ScalarRegisterFile()
        interp = DivFixupF16ScalarInterpreter(rf)

        rf.write_vgpr_f16(0, 2.0)   # denom in v0
        rf.write_vgpr_f16(1, 4.0)   # estimate in v1
        rf.write_vgpr_f16(2, 8.0)   # num in v2

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(2),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_f16(0), 4.0)

    def test_dst_equals_src2_inplace(self):
        rf = ScalarRegisterFile()
        interp = DivFixupF16ScalarInterpreter(rf)

        rf.write_vgpr_f16(0, 10.0)  # num in v0
        rf.write_vgpr_f16(1, 5.0)   # estimate in v1
        rf.write_vgpr_f16(2, 2.0)   # denom in v2

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(0),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_f16(0), 5.0)

    def test_self_aliasing_dst_equals_all_sources(self):
        rf = ScalarRegisterFile()
        interp = DivFixupF16ScalarInterpreter(rf)

        rf.write_vgpr_f16(0, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_f16(0), 1.0)

    def test_simd_multi_lane_aliased_fixup(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFF00FF00
        wave.exec_hi = 0x00FF00FF

        for lane in range(64):
            wave.set_lane_vgpr_f16(lane, 0, float(lane) + 1.0)  # estimate in v0
            wave.set_lane_vgpr_f16(lane, 1, 2.0)                # denom in v1
            wave.set_lane_vgpr_f16(lane, 2, (float(lane) + 1.0) * 2.0) # num in v2

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        wave.execute(inst)

        for lane in range(64):
            expected = float(lane) + 1.0
            self.assertEqual(wave.get_lane_vgpr_f16(lane, 0), expected)


class StatusFlags:
    """IEEE 754 exception flags and architectural condition codes for an operation."""

    def __init__(
        self,
        *,
        invalid: bool = False,
        div_by_zero: bool = False,
        overflow: bool = False,
        underflow: bool = False,
        inexact: bool = False,
        is_negative: bool = False,
        is_zero: bool = False,
        is_inf: bool = False,
        is_nan: bool = False,
        is_subnormal: bool = False,
    ) -> None:
        self.invalid = invalid
        self.div_by_zero = div_by_zero
        self.overflow = overflow
        self.underflow = underflow
        self.inexact = inexact
        self.is_negative = is_negative
        self.is_zero = is_zero
        self.is_inf = is_inf
        self.is_nan = is_nan
        self.is_subnormal = is_subnormal

    def __repr__(self) -> str:
        flags = [k for k, v in self.__dict__.items() if v]
        return f"StatusFlags({', '.join(flags)})"


def compute_div_fixup_status_flags(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[int, StatusFlags]:
    """Compute v_div_fixup_f16 result along with IEEE 754 status flags and condition codes."""
    s0 = apply_src_mods_f16(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f16(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f16(s2_bits, neg=src2_neg, abs_val=src2_abs)

    invalid = False
    div_by_zero = False
    overflow = False
    underflow = False
    inexact = False

    # Invalid operations:
    if is_f16_snan(s0) or is_f16_snan(s1) or is_f16_snan(s2):
        invalid = True
    elif is_f16_zero(s1) and is_f16_zero(s2):
        invalid = True
    elif is_f16_inf(s1) and is_f16_inf(s2):
        invalid = True

    # Division by zero: finite nonzero numerator / zero denominator
    if is_f16_zero(s1) and not is_f16_zero(s2) and not is_f16_nan(s2) and not is_f16_inf(s2):
        div_by_zero = True

    # Underflow check
    if not is_f16_nan(s1) and not is_f16_nan(s2) and not is_f16_zero(s1) and not is_f16_zero(s2):
        if (f16_unbiased_exponent(s2) - f16_unbiased_exponent(s1)) < -25:
            underflow = True

    # Compute result
    res_bits = ref_div_fixup_f16(
        s0_bits, s1_bits, s2_bits,
        src0_neg=src0_neg, src0_abs=src0_abs,
        src1_neg=src1_neg, src1_abs=src1_abs,
        src2_neg=src2_neg, src2_abs=src2_abs,
        clamp=clamp, omod=omod,
    )

    if is_f16_nan(s0) and not is_f16_nan(s1) and not is_f16_nan(s2):
        overflow = True

    if is_f16_subnormal(res_bits):
        underflow = True

    is_neg = (res_bits & F16_SIGN_MASK) != 0
    is_nan = is_f16_nan(res_bits)
    is_inf = is_f16_inf(res_bits)
    is_zero = is_f16_zero(res_bits)
    is_subnormal = is_f16_subnormal(res_bits)

    flags = StatusFlags(
        invalid=invalid,
        div_by_zero=div_by_zero,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        is_negative=is_neg,
        is_zero=is_zero,
        is_inf=is_inf,
        is_nan=is_nan,
        is_subnormal=is_subnormal,
    )
    return res_bits, flags


def evaluate_vcc_predicate(wave: SIMDWaveState, reg_idx: int, predicate: str) -> int:
    """Generate 64-bit VCC mask evaluating condition on reg_idx across all active lanes."""
    vcc = 0
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            val = wave.get_lane_vgpr_f16(lane, reg_idx)
            cond = False
            if predicate == "eq_zero":
                cond = (val == 0.0)
            elif predicate == "ge_zero":
                cond = not math.isnan(val) and val >= 0.0
            elif predicate == "lt_zero":
                cond = not math.isnan(val) and val < 0.0
            elif predicate == "is_nan":
                cond = math.isnan(val)
            elif predicate == "is_finite":
                cond = math.isfinite(val)
            else:
                raise ValueError(f"Unknown predicate: {predicate}")

            if cond:
                vcc |= (1 << lane)
    return vcc


class TestDivFixupF16StatusFlagsAndConditionCodes(unittest.TestCase):
    """Test suite for IEEE 754 status flags, condition codes, and VCC generation."""

    def test_invalid_operation_flags(self):
        res1, flags1 = compute_div_fixup_status_flags(F16_POS_ZERO, F16_POS_ZERO, F16_POS_ZERO)
        self.assertTrue(flags1.invalid)
        self.assertTrue(flags1.is_nan)
        self.assertEqual(res1, F16_INDEFINITE_QNAN)

        res2, flags2 = compute_div_fixup_status_flags(F16_POS_INF, F16_POS_INF, F16_POS_INF)
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.is_nan)
        self.assertEqual(res2, F16_INDEFINITE_QNAN)

    def test_div_by_zero_flag(self):
        res, flags = compute_div_fixup_status_flags(f16_to_u16(1.0), F16_POS_ZERO, f16_to_u16(4.0))
        self.assertTrue(flags.div_by_zero)
        self.assertTrue(flags.is_inf)
        self.assertFalse(flags.is_negative)
        self.assertEqual(res, F16_POS_INF)

    def test_underflow_and_overflow_flags(self):
        # Exponent underflow
        s2 = f16_to_u16(2.0 ** -14)
        s1 = f16_to_u16(2.0 ** 14)
        res_u, flags_u = compute_div_fixup_status_flags(F16_ONE, s1, s2)
        self.assertTrue(flags_u.underflow)
        self.assertTrue(flags_u.is_zero)

        # S0 NaN overflow
        res_o, flags_o = compute_div_fixup_status_flags(F16_CANONICAL_QNAN, F16_ONE, F16_ONE)
        self.assertTrue(flags_o.overflow)
        self.assertTrue(flags_o.is_inf)

    def test_condition_codes_signed_zeros(self):
        res, flags = compute_div_fixup_status_flags(F16_POS_ZERO, F16_ONE, F16_NEG_ZERO)
        self.assertTrue(flags.is_zero)
        self.assertTrue(flags.is_negative)

    def test_vcc_predicate_mask_generation(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        for lane in range(64):
            val = 2.0 if (lane % 2 == 0) else -2.0
            wave.set_lane_vgpr_f16(lane, 0, val)

        vcc_ge = evaluate_vcc_predicate(wave, 0, "ge_zero")
        self.assertEqual(vcc_ge, 0x55555555)

        vcc_lt = evaluate_vcc_predicate(wave, 0, "lt_zero")
        self.assertEqual(vcc_lt, 0xAAAAAAAA)


if __name__ == "__main__":
    unittest.main()
