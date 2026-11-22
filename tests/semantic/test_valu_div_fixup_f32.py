from __future__ import annotations

"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_div_fixup_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector general-purpose registers (VGPRs) are 32 bits wide.
  Operations on 32-bit single-precision floating-point numbers (binary32, IEEE 754)
  operate on individual 32-bit registers: vd from source operands src0, src1, and src2
  (VGPRs, SGPRs, inline constants, or literal constants).

  Instruction:
    v_div_fixup_f32 vd, src0, src1, src2 [clamp] [omod]

  Operands:
    src0 = provisional quotient from previous reciprocal/FMAS step
    src1 = denominator
    src2 = numerator

  IEEE-754 binary32 encoding:
    - 1 sign bit (bit 31)
    - 8 exponent bits with bias 127 (bits 30:23)
    - 23 mantissa / fraction bits (bits 22:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

from enum import Enum, auto
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF

# IEEE-754 binary32 bit constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_CANONICAL_QNAN = 0x7FC00000
F32_INDEFINITE_QNAN = 0xFFC00000  # Default indefinite NaN returned by AMD GPU for 0/0 and inf/inf
F32_QUIET_BIT = 0x00400000

F32_EXP_MASK = 0x7F800000
F32_MANT_MASK = 0x007FFFFF
F32_SIGN_MASK = 0x80000000

F32_ONE = 0x3F800000            # 1.0
F32_NEG_ONE = 0xBF800000        # -1.0
F32_TWO = 0x40000000            # 2.0
F32_NEG_TWO = 0xC0000000        # -2.0
F32_HALF = 0x3F000000           # 0.5
F32_FOUR = 0x40800000           # 4.0
F32_QUARTER = 0x3E800000        # 0.25
F32_MIN_NORMAL = 0x00800000     # 2^-126
F32_MAX_NORMAL = 0x7F7FFFFF     # (2 - 2^-23) * 2^127
F32_MIN_SUBNORMAL = 0x00000001  # 2^-149
F32_MAX_SUBNORMAL = 0x007FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def f32_to_u32(val: float) -> int:
    """Pack an IEEE-754 single precision float into an unsigned 32-bit integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return F32_NEG_INF if val < 0 else F32_POS_INF


def u32_to_f32(bits: int) -> float:
    """Unpack an unsigned 32-bit integer into an IEEE-754 single precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan(bits: int) -> bool:
    """Return True if the 32-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F32_EXP_MASK) == F32_EXP_MASK) and ((bits & F32_MANT_MASK) != 0)


def is_f32_snan(bits: int) -> bool:
    """Return True if the 32-bit pattern represents a signaling NaN."""
    return is_f32_nan(bits) and ((bits & F32_QUIET_BIT) == 0)


def is_f32_qnan(bits: int) -> bool:
    """Return True if the 32-bit pattern represents a quiet NaN."""
    return is_f32_nan(bits) and ((bits & F32_QUIET_BIT) != 0)


def is_f32_zero(bits: int) -> bool:
    """Return True if the 32-bit pattern represents +0.0 or -0.0."""
    return (bits & (F32_EXP_MASK | F32_MANT_MASK)) == 0


def is_f32_inf(bits: int) -> bool:
    """Return True if the 32-bit pattern represents +inf or -inf."""
    return ((bits & F32_EXP_MASK) == F32_EXP_MASK) and ((bits & F32_MANT_MASK) == 0)


def is_f32_subnormal(bits: int) -> bool:
    """Return True if the 32-bit pattern represents a subnormal/denormal float."""
    return ((bits & F32_EXP_MASK) == 0) and ((bits & F32_MANT_MASK) != 0)


def is_f32_normal(bits: int) -> bool:
    """Return True if the 32-bit pattern represents a normalized finite float."""
    exp = (bits >> 23) & 0xFF
    return 0 < exp < 0xFF


def f32_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 31) & 1


def f32_unbiased_exponent(bits: int) -> int:
    """Extract true mathematical binary exponent for finite non-zero value."""
    val = abs(u32_to_f32(bits))
    if val == 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    _, e = math.frexp(val)
    return e - 1


def quiet_f32(bits: int) -> int:
    """Force an IEEE-754 single NaN to quiet by setting the quiet bit."""
    return (bits & MASK32) | F32_QUIET_BIT


def apply_src_mods_f32(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK32
    if abs_val:
        res &= ~F32_SIGN_MASK
    if neg:
        res ^= F32_SIGN_MASK
    return res


def apply_clamp_f32(bits: int) -> int:
    """Apply VOP3 clamp modifier to single precision float.

    Saturates result to [+0.0, 1.0].
    NaN is converted to +0.0.
    -0.0 is clamped to +0.0.
    Values < 0.0 are clamped to +0.0.
    Values > 1.0 are clamped to 1.0.
    """
    if is_f32_nan(bits) or bits == F32_NEG_ZERO:
        return F32_POS_ZERO
    val = u32_to_f32(bits)
    if val < 0.0:
        return F32_POS_ZERO
    elif val > 1.0:
        return F32_ONE
    return bits


def apply_omod_f32(bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0 or is_f32_nan(bits) or is_f32_inf(bits) or is_f32_zero(bits):
        return bits
    val = u32_to_f32(bits)
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")
    return f32_to_u32(val)


def ref_div_fixup_f32(
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
    """Reference mathematical oracle for AMD RDNA2 v_div_fixup_f32 instruction.

    Args:
        s0_bits: Provisional quotient (32-bit uint)
        s1_bits: Denominator (32-bit uint)
        s2_bits: Numerator (32-bit uint)
        src0_neg / src0_abs: Modifiers for src0
        src1_neg / src1_abs: Modifiers for src1
        src2_neg / src2_abs: Modifiers for src2
        clamp: If True, saturates to [0.0, 1.0] (NaN -> +0.0)
        omod: Output modifier (0: 1.0, 1: 2.0, 2: 4.0, 3: 0.5)

    Returns:
        Fixed-up quotient as 32-bit uint.
    """
    s0 = apply_src_mods_f32(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f32(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f32(s2_bits, neg=src2_neg, abs_val=src2_abs)

    sign1 = f32_sign(s1)
    sign2 = f32_sign(s2)
    sign_out = sign1 ^ sign2

    # Decision tree according to AMD RDNA2 ISA specification
    if is_f32_nan(s2):
        res = quiet_f32(s2)
    elif is_f32_nan(s1):
        res = quiet_f32(s1)
    elif is_f32_zero(s1) and is_f32_zero(s2):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = F32_INDEFINITE_QNAN
    elif is_f32_inf(s1) and is_f32_inf(s2):
        # inf / inf -> Indefinite QNaN
        res = F32_INDEFINITE_QNAN
    elif is_f32_zero(s1) or is_f32_inf(s2):
        # x / 0 or inf / y -> signed infinity
        res = F32_NEG_INF if sign_out else F32_POS_INF
    elif is_f32_inf(s1) or is_f32_zero(s2):
        # x / inf or 0 / y -> signed zero
        res = F32_NEG_ZERO if sign_out else F32_POS_ZERO
    elif is_f32_nan(s0):
        # Intermediate quotient overflowed to NaN -> signed infinity
        res = F32_NEG_INF if sign_out else F32_POS_INF
    elif (f32_unbiased_exponent(s2) - f32_unbiased_exponent(s1)) < -150:
        # Exponent underflow threshold (-150 for binary32 subnormal limit)
        res = F32_NEG_ZERO if sign_out else F32_POS_ZERO
    else:
        # Standard case: sign_out applied to absolute magnitude of s0
        res = (s0 & ~F32_SIGN_MASK) | (F32_SIGN_MASK if sign_out else 0)

    # Output modifiers
    if omod != 0:
        res = apply_omod_f32(res, omod)

    if clamp:
        res = apply_clamp_f32(res)

    return res & MASK32


def ref_div_fixup_f32_float(s0: float, s1: float, s2: float, **kwargs) -> float:
    """Float wrapper for ref_div_fixup_f32."""
    b0 = f32_to_u32(s0)
    b1 = f32_to_u32(s1)
    b2 = f32_to_u32(s2)
    res_bits = ref_div_fixup_f32(b0, b1, b2, **kwargs)
    return u32_to_f32(res_bits)


class SingleRegister:
    """Architectural model of a 32-bit VGPR or SGPR register."""

    def __init__(self, val_u32: int = 0) -> None:
        self._val = val_u32 & MASK32

    @property
    def as_u32(self) -> int:
        return self._val

    @as_u32.setter
    def as_u32(self, val: int) -> None:
        self._val = val & MASK32

    @property
    def as_f32(self) -> float:
        return u32_to_f32(self._val)

    @as_f32.setter
    def as_f32(self, val: float) -> None:
        self._val = f32_to_u32(val)

    @classmethod
    def from_u32(cls, val: int) -> SingleRegister:
        return cls(val)

    @classmethod
    def from_f32(cls, val: float) -> SingleRegister:
        return cls(f32_to_u32(val))

    def __repr__(self) -> str:
        return f"SingleRegister({self._val:#010x}, f32={self.as_f32})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleRegister):
            return NotImplemented
        return self._val == other._val


class TestDivFixupF32MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_normal_quotient_fixup(self):
        # 10.0 / 2.0 = 5.0, s0 is estimated 5.0
        s0 = f32_to_u32(5.0)
        s1 = f32_to_u32(2.0)
        s2 = f32_to_u32(10.0)
        res = ref_div_fixup_f32(s0, s1, s2)
        self.assertEqual(res, f32_to_u32(5.0))

        # Sign correction: numerator negative, denominator positive -> result negative
        s2_neg = f32_to_u32(-10.0)
        res_neg = ref_div_fixup_f32(s0, s1, s2_neg)
        self.assertEqual(res_neg, f32_to_u32(-5.0))

    def test_division_by_zero(self):
        # 1.0 / +0.0 -> +inf
        s0 = f32_to_u32(1.0)
        s1 = F32_POS_ZERO
        s2 = f32_to_u32(1.0)
        res = ref_div_fixup_f32(s0, s1, s2)
        self.assertEqual(res, F32_POS_INF)

        # 1.0 / -0.0 -> -inf
        res_nz = ref_div_fixup_f32(s0, F32_NEG_ZERO, s2)
        self.assertEqual(res_nz, F32_NEG_INF)

    def test_zero_numerator(self):
        # +0.0 / 5.0 -> +0.0
        s0 = f32_to_u32(0.0)
        s1 = f32_to_u32(5.0)
        s2 = F32_POS_ZERO
        res = ref_div_fixup_f32(s0, s1, s2)
        self.assertEqual(res, F32_POS_ZERO)

        # -0.0 / 5.0 -> -0.0
        res_nz = ref_div_fixup_f32(s0, s1, F32_NEG_ZERO)
        self.assertEqual(res_nz, F32_NEG_ZERO)

    def test_zero_divided_by_zero(self):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = ref_div_fixup_f32(F32_POS_ZERO, F32_POS_ZERO, F32_POS_ZERO)
        self.assertEqual(res, F32_INDEFINITE_QNAN)

    def test_infinity_divided_by_infinity(self):
        # inf / inf -> Indefinite QNaN
        res = ref_div_fixup_f32(F32_POS_INF, F32_POS_INF, F32_POS_INF)
        self.assertEqual(res, F32_INDEFINITE_QNAN)

    def test_finite_divided_by_infinity(self):
        # 5.0 / inf -> +0.0
        s0 = f32_to_u32(0.0)
        s1 = F32_POS_INF
        s2 = f32_to_u32(5.0)
        res = ref_div_fixup_f32(s0, s1, s2)
        self.assertEqual(res, F32_POS_ZERO)

    def test_infinity_divided_by_finite(self):
        # inf / 5.0 -> +inf
        s0 = f32_to_u32(1.0)
        s1 = f32_to_u32(5.0)
        s2 = F32_POS_INF
        res = ref_div_fixup_f32(s0, s1, s2)
        self.assertEqual(res, F32_POS_INF)

    def test_nan_numerator_and_denominator_priority(self):
        # S2 NaN takes precedence
        nan2 = 0x7F800042  # SNaN
        nan1 = 0x7F800011
        res = ref_div_fixup_f32(F32_ONE, nan1, nan2)
        self.assertEqual(res, quiet_f32(nan2))

        # S1 NaN if S2 is not NaN
        res1 = ref_div_fixup_f32(F32_ONE, nan1, F32_ONE)
        self.assertEqual(res1, quiet_f32(nan1))

    def test_s0_nan_overflow(self):
        # S0 NaN produces signed infinity
        res = ref_div_fixup_f32(F32_CANONICAL_QNAN, F32_ONE, F32_ONE)
        self.assertEqual(res, F32_POS_INF)

        res_neg = ref_div_fixup_f32(F32_CANONICAL_QNAN, F32_ONE, F32_NEG_ONE)
        self.assertEqual(res_neg, F32_NEG_INF)

    def test_exponent_underflow(self):
        # Exponent diff < -150 produces signed zero
        # s2 = 2^-130, s1 = 2^25 -> diff = -155 < -150
        s2 = f32_to_u32(2.0 ** -120)
        s1 = f32_to_u32(2.0 ** 35)
        res = ref_div_fixup_f32(F32_ONE, s1, s2)
        self.assertEqual(res, F32_POS_ZERO)

    def test_source_and_output_modifiers(self):
        s0 = f32_to_u32(2.0)
        s1 = f32_to_u32(1.0)
        s2 = f32_to_u32(2.0)

        # src0 neg
        res_neg0 = ref_div_fixup_f32(s0, s1, s2, src0_neg=True)
        # Note: sign_out is from s1 and s2, magnitude from s0, so sign_out determines sign!
        self.assertEqual(res_neg0, f32_to_u32(2.0))

        # src2 neg
        res_neg2 = ref_div_fixup_f32(s0, s1, s2, src2_neg=True)
        self.assertEqual(res_neg2, f32_to_u32(-2.0))

        # omod: mul:2
        res_omod = ref_div_fixup_f32(s0, s1, s2, omod=1)
        self.assertEqual(res_omod, f32_to_u32(4.0))

        # clamp
        res_clamp = ref_div_fixup_f32(f32_to_u32(5.0), s1, s2, clamp=True)
        self.assertEqual(res_clamp, F32_ONE)

    def test_single_register_model(self):
        reg = SingleRegister.from_f32(3.5)
        self.assertEqual(reg.as_f32, 3.5)
        self.assertEqual(reg.as_u32, f32_to_u32(3.5))

        reg.as_u32 = F32_ONE
        self.assertEqual(reg.as_f32, 1.0)


class Opcode(Enum):
    V_DIV_FIXUP_F32 = "v_div_fixup_f32"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT32 = "f32"
    UINT32 = "u32"


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
    """Symbolic AST operand representing a 32-bit register, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT32,
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
    """Symbolic AST instruction representation for v_div_fixup_f32."""

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
        if self.opcode != Opcode.V_DIV_FIXUP_F32:
            raise ValueError(f"Invalid opcode for v_div_fixup_f32 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a VGPR")
        if self.dst.type_kind != TypeKind.FLOAT32:
            raise TypeError(f"Destination type must be FLOAT32, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT32:
                raise TypeError(f"Source operand {i} type must be FLOAT32, got {src.type_kind}")
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

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Named variable in symbolic expression tree (e.g. representing a register value)."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
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

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier applied to a typed expression ('neg' or 'abs')."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
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


class DivFixupF32Expr(TypedExpr):
    """Symbolic DivFixupF32 ternary expression: div_fixup(src0, src1, src2) with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT32,
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
        return ref_div_fixup_f32_float(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res = ref_div_fixup_f32_float(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res, self.type_kind)
        return DivFixupF32Expr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"DivFixupF32({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


class TestDivFixupF32SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_fixup_f32 v0, v1, v2, v3")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1, neg=True, abs_=True),
            src1=ASTOperand.sgpr(2),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_fixup_f32 v0, -|v1|, s2, 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind (SGPR)
        bad_dst = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F32,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_DIV_FIXUP_F32,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        expr = DivFixupF32Expr(
            VariableExpr("q"),
            VariableExpr("denom"),
            VariableExpr("num"),
        )
        env = {"q": 4.0, "denom": 2.0, "num": 8.0}
        self.assertEqual(expr.evaluate(env), 4.0)

    def test_typed_expression_constant_folding(self):
        c_expr = DivFixupF32Expr(
            ConstantExpr(4.0),
            ConstantExpr(2.0),
            ConstantExpr(8.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 4.0)

        # Unary folding
        u_expr = UnaryOpExpr("neg", ConstantExpr(3.25))
        u_folded = u_expr.simplify()
        self.assertIsInstance(u_folded, ConstantExpr)
        self.assertEqual(u_folded.value, -3.25)


if __name__ == "__main__":
    unittest.main()
