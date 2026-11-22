from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale_f64.

Tests the double-precision floating-point division pre-scaling instruction:
    v_div_scale_f64 D, sdst, S0, S1, S2

Hardware Specification (RDNA2 / GFX10 / GFX1030 VOP3b dual-output):
    D    = 64-bit vector destination register pair v[d:d+1] receiving scaled S0
    sdst = scalar destination (VCC or SGPR pair/single) receiving scale condition bit mask
    S0   = operand to scale (64-bit float, typically S1 or S2)
    S1   = denominator (64-bit float)
    S2   = numerator (64-bit float)

Operation:
    Given numerator S2 and denominator S1, v_div_scale_f64 determines whether intermediate
    Newton-Raphson division approximation will underflow or overflow, and conditionally scales
    S0 by 2^128 or 2^-128 while recording whether scaling took place in sdst (VCC).
    
    Based on measured RDNA2 hardware semantics (docs/design.md):
    With e1, e2 the exponent fields of S1 (denom) and S2 (num), the first matching rule wins:
      1. e2 - e1 >= 768:
         sdst = 1, D = S0 * 2^128 if S0 == S1 else S0
      2. e1 == 0 (zero or denormal denominator):
         sdst = 0, D = S0 * 2^128
      3. e1 >= 2045 and e1 - e2 >= 768:
         sdst = 1, D = S0 * 2^-128 if S0 == S1 else S0
      4. e1 >= 2045:
         sdst = 0, D = S0 * 2^-128
      5. e1 - e2 >= 768:
         sdst = 1, D = S0 if S0 == S1 else S0 * 2^128
      6. e2 <= 53:
         sdst = 0, D = S0 * 2^128
      7. otherwise:
         sdst = 0, D = S0

    Zero behavior:
      A zero S1 or S2 makes D = 0xFFF8000000000000 (NaN), while sdst retains its table bit.
    
    Equality:
      S0 == S1 compares 64-bit values after neg modifier is applied.
    
    Modifiers:
      - neg(-src): inverts sign bit of S0, S1, S2.
      - clamp: maps NaNs, zeros, and negative values to +0.0; values >= 1.0 to 1.0.
      - omod: accepted by assembler and ignored by hardware.

Architectural model:
    64-bit floating point registers mapped onto aligned VGPR/SGPR pairs:
    [lo: 32-bit dword, hi: 32-bit dword].
"""

import math
import struct
import unittest
from enum import Enum, auto

# IEEE 754 double precision bitwise constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_TWO = 0x4000000000000000
F64_HALF = 0x3FE0000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_SUB_MIN = 0x0000000000000001
F64_SUB_MAX = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_dwords(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def dwords_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact source modifiers (|src|, -src) to a 64-bit float bit pattern."""
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO
    if neg:
        bits ^= F64_NEG_ZERO
    return bits


def apply_clamp_f64(val_u64: int) -> int:
    """Apply RDNA2 clamp modifier to double precision result."""
    x = u64_to_f64(val_u64)
    if x > 0.0:
        return f64_to_u64(min(x, 1.0))
    return 0


def apply_omod_f64(val_u64: int, omod: int) -> int:
    """Apply RDNA2 omod modifier (measured: ignored on hardware for v_div_scale_f64)."""
    return val_u64


def ex_f64(b: int) -> int:
    """Extract 11-bit IEEE 754 exponent field from 64-bit float pattern."""
    return (b >> 52) & 0x7FF


def scale_f64_bits(b: int, n: int) -> int:
    """Scale 64-bit float bit pattern by 2^n per RDNA2 hardware semantics."""
    if ex_f64(b) == 0x7FF or not (b & ~(1 << 63)):
        return b  # NaN (payload preserved), inf, zero
    if n < 0 and ex_f64(b) <= 116:
        return (b & (1 << 63)) | 0x7FF0000000000000  # measured: inf with sign of S0
    x = u64_to_f64(b)
    try:
        res = math.ldexp(x, n)
        return f64_to_u64(res)
    except OverflowError:
        return (b & (1 << 63)) | 0x7FF0000000000000


def ref_div_scale_f64_u64(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[int, int]:
    """Bit-exact reference mathematical oracle for v_div_scale_f64 using 64-bit integer words.

    Returns (d_u64, sdst_bit).
    """
    s0 = (u0 & MASK64) ^ (F64_NEG_ZERO if neg0 else 0)
    s1 = (u1 & MASK64) ^ (F64_NEG_ZERO if neg1 else 0)
    s2 = (u2 & MASK64) ^ (F64_NEG_ZERO if neg2 else 0)

    e1, e2 = ex_f64(s1), ex_f64(s2)
    vcc, d = 0, s0

    if e2 - e1 >= 768:
        vcc, d = 1, scale_f64_bits(s0, 128) if s0 == s1 else s0
    elif e1 == 0:
        d = scale_f64_bits(s0, 128)
    elif e1 >= 2045 and e1 - e2 >= 768:
        vcc, d = 1, scale_f64_bits(s0, -128) if s0 == s1 else s0
    elif e1 >= 2045:
        d = scale_f64_bits(s0, -128)
    elif e1 - e2 >= 768:
        vcc, d = 1, s0 if s0 == s1 else scale_f64_bits(s0, 128)
    elif e2 <= 53:
        d = scale_f64_bits(s0, 128)

    # A zero S1 or S2 makes D = 0xFFF8000000000000 (NaN), while sdst retains its table bit
    if not (s1 & ~(1 << 63)) or not (s2 & ~(1 << 63)):
        d = 0xFFF8000000000000

    if clamp:
        d = apply_clamp_f64(d)

    return d, vcc


def ref_div_scale_f64(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[float, int]:
    """Float wrapper for ref_div_scale_f64_u64. Returns (d_float, sdst_bit)."""
    d_u64, vcc = ref_div_scale_f64_u64(
        f64_to_u64(s0),
        f64_to_u64(s1),
        f64_to_u64(s2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        clamp=clamp,
        omod=omod,
    )
    return u64_to_f64(d_u64), vcc


class RegisterPair:
    """Architectural model of an aligned 64-bit register pair (e.g. v[0:1] or s[0:1])."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = u64_to_dwords(val)
        return cls(lo, hi)

    @classmethod
    def from_f64(cls, val: float) -> RegisterPair:
        return cls.from_u64(f64_to_u64(val))

    @property
    def as_u64(self) -> int:
        return dwords_to_u64(self.lo, self.hi)

    @property
    def as_f64(self) -> float:
        return u64_to_f64(self.as_u64)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = u64_to_dwords(val)

    def set_f64(self, val: float) -> None:
        self.set_u64(f64_to_u64(val))

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, f64={self.as_f64})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class TestDivScaleF64MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_normal_no_scale(self):
        # 1.0 / 2.0 with s0 = 2.0: exponents 1023 and 1024, no scaling needed
        d, vcc = ref_div_scale_f64(2.0, 2.0, 1.0)
        self.assertEqual(vcc, 0)
        self.assertEqual(d, 2.0)

    def test_rdna2_measured_cases(self):
        # Hardware measured 13 cases from gfx1036 (docs/design.md)
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        tiny = 0x0DF0000000000003

        cases = [
            (one5, one5, big, False),
            (big, one5, big, False),
            (tiny, one5, tiny, False),
            (one5, one5, tiny, False),
            (0x7FE0000000000000, 0x7FE0000000000000, 0x3FF0000000000000, False),
            (0x7F80000000000009, 0x7FE0000000000000, 0x7F80000000000009, False),
            (7, 7, 0x2580000000000000, False),
            (0x0140000000000001, 0x0010000000000000, 0x0140000000000001, False),
            (0, 0, 0x3FF0000000000000, False),
            (0x7FF0000000000001, 0x3FF0000000000000, 0x7FF0000000000001, False),
            (one5 | (1 << 63), one5, big, False),
            (0, 0, 0x3FF0000000000000, True),
            (0x3FD0000000000000, 0x3FD0000000000000, 0x3FE0000000000000, False),
        ]
        want = [
            0x00000005, 0x47F00000, 1,
            0x00000000, 0x71F00000, 1,
            0x00000003, 0x15F00000, 1,
            0x00000005, 0x3FF00000, 1,
            0x00000000, 0x77E00000, 1,
            0x00000009, 0x77800000, 0,
            0x00000000, 0x04FC0000, 0,
            0x00000001, 0x09400000, 0,
            0x00000000, 0xFFF80000, 1,
            0x00000001, 0x7FF00000, 1,
            0x00000005, 0xBFF00000, 1,
            0x00000000, 0x00000000, 1,
            0x00000000, 0x3FD00000, 0,
        ]

        got = []
        for s0, s1, s2, clamp in cases:
            d_u, vcc = ref_div_scale_f64_u64(s0, s1, s2, clamp=clamp)
            got.extend([d_u & MASK32, (d_u >> 32) & MASK32, vcc])

        self.assertEqual(got, want)

    def test_source_modifiers(self):
        # neg modifier flips sign bit
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # When both s0 and s1 have neg applied, s0 == s1, so both scale and have inverted sign
        d_pos, vcc_p = ref_div_scale_f64_u64(one5, one5, big)
        d_neg, vcc_n = ref_div_scale_f64_u64(one5, one5, big, neg0=True, neg1=True)
        self.assertEqual(vcc_p, 1)
        self.assertEqual(vcc_n, 1)
        self.assertEqual(d_pos ^ (1 << 63), d_neg)

        # Equality S0 == S1 is checked after neg
        # When only s0 has neg, s0 != s1, so case 1 (e2 - e1 >= 768) does not scale s0
        d_noteq, vcc_noteq = ref_div_scale_f64_u64(one5, one5, big, neg0=True)
        self.assertEqual(vcc_noteq, 1)
        self.assertEqual(d_noteq, one5 ^ (1 << 63))  # unscaled s0 with negative sign

    def test_output_modifiers_clamp(self):
        # clamp saturates to [0.0, 1.0], NaNs and negatives map to +0.0
        d_pos, _ = ref_div_scale_f64_u64(F64_TWO, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_pos, F64_ONE)

        d_neg, _ = ref_div_scale_f64_u64(F64_NEG_ONE, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_neg, F64_POS_ZERO)

        # Zero denominator produces NaN which clamps to +0.0
        # For s1 = +0.0, s2 = 1.0, e2 - e1 >= 768 is met, so vcc is 1
        d_zero_denom, vcc_z = ref_div_scale_f64_u64(F64_ONE, F64_POS_ZERO, F64_ONE, clamp=True)
        self.assertEqual(d_zero_denom, F64_POS_ZERO)
        self.assertEqual(vcc_z, 1)

    def test_register_pair_model(self):
        rp = RegisterPair.from_f64(1.5)
        self.assertEqual(rp.as_f64, 1.5)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF80000)

        rp.set_u64(F64_NEG_ONE)
        self.assertEqual(rp.as_f64, -1.0)
        self.assertEqual(rp.hi, 0xBFF00000)


class Opcode(Enum):
    V_DIV_SCALE_F64 = "v_div_scale_f64"


class OperandKind(Enum):
    VGPR_PAIR = auto()
    SGPR_PAIR = auto()
    SGPR_SINGLE = auto()
    VCC = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT64 = "f64"
    UINT64 = "u64"
    UINT32 = "u32"
    BOOL = "bool"


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
    """Symbolic AST operand representing a register pair, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT64,
    ) -> None:
        self.kind = kind
        self.index = index
        self.value = value
        self.neg = neg
        self.abs_ = abs_
        self.type_kind = type_kind

    @classmethod
    def vgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR_PAIR, index=index, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT64)

    @classmethod
    def sgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR_PAIR, index=index, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT64)

    @classmethod
    def sdst_sgpr(cls, index: int, *, is_wave32: bool = False) -> ASTOperand:
        kind = OperandKind.SGPR_SINGLE if is_wave32 else OperandKind.SGPR_PAIR
        type_k = TypeKind.UINT32 if is_wave32 else TypeKind.UINT64
        return cls(kind, index=index, type_kind=type_k)

    @classmethod
    def vcc(cls, *, is_wave32: bool = False) -> ASTOperand:
        return cls(OperandKind.VCC, index=106, type_kind=TypeKind.UINT32 if is_wave32 else TypeKind.UINT64)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT64)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT64)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            body = f"v[{self.index}:{self.index + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            body = f"s[{self.index}:{self.index + 1}]"
        elif self.kind == OperandKind.SGPR_SINGLE:
            body = f"s{self.index}"
        elif self.kind == OperandKind.VCC:
            body = "vcc"
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
    """Symbolic AST instruction representation for v_div_scale_f64."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        sdst: ASTOperand,
        src0: ASTOperand,
        src1: ASTOperand,
        src2: ASTOperand,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.sdst = sdst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def validate(self) -> None:
        if self.opcode != Opcode.V_DIV_SCALE_F64:
            raise ValueError(f"Invalid opcode for v_div_scale_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise TypeError("Destination must be a VGPR pair")
        if self.dst.type_kind != TypeKind.FLOAT64:
            raise TypeError(f"Destination type must be FLOAT64, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        if self.sdst.kind not in (OperandKind.VCC, OperandKind.SGPR_PAIR, OperandKind.SGPR_SINGLE):
            raise TypeError(f"sdst must be VCC or SGPR, got {self.sdst.kind}")
        if self.sdst.neg or self.sdst.abs_:
            raise ValueError("sdst operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT64:
                raise TypeError(f"Source operand {i} type must be FLOAT64, got {src.type_kind}")
            if src.kind == OperandKind.VGPR_PAIR and not (0 <= src.index <= 254):
                raise ValueError(f"Source {i} VGPR pair index out of range: {src.index}")
            if src.kind == OperandKind.SGPR_PAIR and not (0 <= src.index <= 104):
                raise ValueError(f"Source {i} SGPR pair index out of range: {src.index}")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.sdst.disasm()},",
            f"{self.src0.disasm()},",
            f"{self.src1.disasm()},",
            self.src2.disasm(),
        ]
        mods = []
        if self.omod != OMod.NONE:
            mods.append(self.omod.to_asm())
        if self.clamp:
            mods.append("clamp")
        if mods:
            parts.extend(mods)
        return " ".join(parts)


class TypedExpr:
    """Base class for symbolic typed AST expressions."""

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float | tuple[float, int]:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Symbolic variable reference."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable '{self.name}' not bound in environment")
        return env[self.name]

    def __repr__(self) -> str:
        return f"Var({self.name})"


class ConstantExpr(TypedExpr):
    """Literal constant value."""

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier expression: neg or abs."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.op = op
        self.expr = expr

    def evaluate(self, env: dict[str, float]) -> float:
        val = self.expr.evaluate(env)
        if isinstance(val, tuple):
            raise TypeError("Cannot apply unary op to tuple")
        if self.op == "abs":
            return math.fabs(val)
        elif self.op == "neg":
            return -val
        else:
            raise ValueError(f"Unknown unary op: {self.op}")

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


class DivScaleExpr(TypedExpr):
    """Symbolic DivScale ternary typed expression: v_div_scale_f64 with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT64,
    ) -> None:
        super().__init__(type_kind)
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        v0 = self.src0.evaluate(env)
        v1 = self.src1.evaluate(env)
        v2 = self.src2.evaluate(env)
        if isinstance(v0, tuple) or isinstance(v1, tuple) or isinstance(v2, tuple):
            raise TypeError("Subexpressions must evaluate to float")
        return ref_div_scale_f64(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res_val, _ = ref_div_scale_f64(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res_val, self.type_kind)
        return DivScaleExpr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"DivScale({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


def parse_operand_asm(op_str: str, *, is_sdst: bool = False) -> ASTOperand:
    """Parse a single assembly operand token into an ASTOperand."""
    op_str = op_str.strip()
    neg = False
    abs_ = False

    if not is_sdst:
        if op_str.startswith("-"):
            neg = True
            op_str = op_str[1:].strip()
        if op_str.startswith("|") and op_str.endswith("|"):
            abs_ = True
            op_str = op_str[1:-1].strip()

    if is_sdst:
        if op_str.lower() == "vcc":
            return ASTOperand.vcc()
        if op_str.startswith("s[") and op_str.endswith("]"):
            nums = op_str[2:-1].split(":")
            if len(nums) != 2 or int(nums[1]) != int(nums[0]) + 1:
                raise ValueError(f"Invalid SGPR pair syntax: '{op_str}'")
            return ASTOperand.sdst_sgpr(int(nums[0]), is_wave32=False)
        if op_str.startswith("s") and op_str[1:].isdigit():
            return ASTOperand.sdst_sgpr(int(op_str[1:]), is_wave32=True)
        raise ValueError(f"Invalid sdst operand: '{op_str}'")

    if op_str.startswith("v[") and op_str.endswith("]"):
        nums = op_str[2:-1].split(":")
        if len(nums) != 2 or int(nums[1]) != int(nums[0]) + 1:
            raise ValueError(f"Non-consecutive or invalid VGPR pair: '{op_str}'")
        idx = int(nums[0])
        if not (0 <= idx <= 254):
            raise ValueError(f"VGPR index out of range: {idx}")
        return ASTOperand.vgpr(idx, neg=neg, abs_=abs_)

    if op_str.startswith("s[") and op_str.endswith("]"):
        nums = op_str[2:-1].split(":")
        if len(nums) != 2 or int(nums[1]) != int(nums[0]) + 1:
            raise ValueError(f"Non-consecutive or invalid SGPR pair: '{op_str}'")
        idx = int(nums[0])
        if not (0 <= idx <= 104):
            raise ValueError(f"SGPR index out of range: {idx}")
        return ASTOperand.sgpr(idx, neg=neg, abs_=abs_)

    try:
        val = float(op_str)
        return ASTOperand.inline_const(val, neg=neg, abs_=abs_)
    except ValueError:
        pass

    raise ValueError(f"Unable to parse operand: '{op_str}'")


def parse_div_scale_asm(asm_line: str) -> ASTInstruction:
    """Parse an assembly line string into an ASTInstruction."""
    line = asm_line.strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    op_name = tokens[0]
    if op_name != Opcode.V_DIV_SCALE_F64.value:
        raise ValueError(f"Invalid opcode for div_scale: '{op_name}'")

    if len(tokens) < 2:
        raise ValueError("Missing operands in instruction")

    rest = tokens[1].strip()
    raw_parts = [p.strip() for p in rest.split(",")]
    if len(raw_parts) != 5:
        raise ValueError(f"Expected 5 comma-separated operands (dst, sdst, s0, s1, s2), got {len(raw_parts)}")

    dst_str = raw_parts[0]
    sdst_str = raw_parts[1]
    src0_str = raw_parts[2]
    src1_str = raw_parts[3]

    src2_and_mods = raw_parts[4].split()
    if not src2_and_mods:
        raise ValueError("Missing src2 operand")
    src2_str = src2_and_mods[0]
    mod_tokens = src2_and_mods[1:]

    clamp = False
    omod = OMod.NONE
    for mod in mod_tokens:
        mod_l = mod.lower()
        if mod_l == "clamp":
            clamp = True
        elif mod_l == "mul:2":
            omod = OMod.MUL2
        elif mod_l == "mul:4":
            omod = OMod.MUL4
        elif mod_l == "div:2":
            omod = OMod.DIV2
        else:
            raise ValueError(f"Unknown modifier: '{mod}'")

    dst = parse_operand_asm(dst_str)
    sdst = parse_operand_asm(sdst_str, is_sdst=True)
    src0 = parse_operand_asm(src0_str)
    src1 = parse_operand_asm(src1_str)
    src2 = parse_operand_asm(src2_str)

    inst = ASTInstruction(
        opcode=Opcode.V_DIV_SCALE_F64,
        dst=dst,
        sdst=sdst,
        src0=src0,
        src1=src1,
        src2=src2,
        clamp=clamp,
        omod=omod,
    )
    inst.validate()
    return inst


class TestDivScaleF64SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_scale_f64 v[0:1], vcc, v[2:3], v[4:5], v[6:7]")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.sdst_sgpr(4),
            src0=ASTOperand.vgpr(2, neg=True),
            src1=ASTOperand.sgpr(6),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_scale_f64 v[0:1], s[4:5], -v[2:3], s[6:7], 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind (SGPR cannot be vector destination)
        bad_dst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.sgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0, neg=True),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(256),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        expr = DivScaleExpr(
            VariableExpr("s0"),
            VariableExpr("s1"),
            VariableExpr("s2"),
        )
        env = {"s0": 2.0, "s1": 2.0, "s2": 1.0}
        d_val, vcc_val = expr.evaluate(env)
        self.assertEqual(d_val, 2.0)
        self.assertEqual(vcc_val, 0)

    def test_typed_expression_constant_folding(self):
        c_expr = DivScaleExpr(
            ConstantExpr(2.0),
            ConstantExpr(2.0),
            ConstantExpr(1.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 2.0)

    def test_asm_parsing_roundtrip(self):
        asm = "v_div_scale_f64 v[0:1], vcc, -v[2:3], s[4:5], 1.0 clamp"
        inst = parse_div_scale_asm(asm)
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.sdst.kind, OperandKind.VCC)
        self.assertTrue(inst.src0.neg)
        self.assertEqual(inst.src1.index, 4)
        self.assertEqual(inst.src2.value, 1.0)
        self.assertTrue(inst.clamp)


class ScalarRegisterFile:
    """Register file for a single scalar lane with 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 108  # 106 general SGPRs + VCC_LO (106) and VCC_HI (107)

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_vgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.vgprs[idx], self.vgprs[idx + 1])

    def write_vgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.vgprs[idx] = lo
        self.vgprs[idx + 1] = hi

    def read_vgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_vgpr_pair_u64(idx))

    def write_vgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_vgpr_pair_u64(idx, f64_to_u64(val))

    def read_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def write_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def read_sgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_sgpr_pair_u64(idx))

    def write_sgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_sgpr_pair_u64(idx, f64_to_u64(val))

    @property
    def vcc_lo(self) -> int:
        return self.sgprs[106]

    @vcc_lo.setter
    def vcc_lo(self, val: int) -> None:
        self.sgprs[106] = val & MASK32

    @property
    def vcc_hi(self) -> int:
        return self.sgprs[107]

    @vcc_hi.setter
    def vcc_hi(self, val: int) -> None:
        self.sgprs[107] = val & MASK32

    @property
    def vcc(self) -> int:
        return dwords_to_u64(self.vcc_lo, self.vcc_hi)

    @vcc.setter
    def vcc(self, val: int) -> None:
        self.vcc_lo, self.vcc_hi = u64_to_dwords(val)


class DivScaleF64ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_div_scale_f64."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_u64(self, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR_PAIR:
            val_u64 = self.reg_file.read_vgpr_pair_u64(src.index)
        elif src.kind == OperandKind.SGPR_PAIR:
            val_u64 = self.reg_file.read_sgpr_pair_u64(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u64 = f64_to_u64(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u64(val_u64, neg=src.neg, abs_=src.abs_)

    def step(self, inst: ASTInstruction) -> tuple[int, int]:
        inst.validate()
        u0 = self.resolve_source_u64(inst.src0)
        u1 = self.resolve_source_u64(inst.src1)
        u2 = self.resolve_source_u64(inst.src2)

        d_u64, vcc_bit = ref_div_scale_f64_u64(
            u0,
            u1,
            u2,
            clamp=inst.clamp,
            omod=inst.omod.value,
        )

        self.reg_file.write_vgpr_pair_u64(inst.dst.index, d_u64)

        if inst.sdst.kind == OperandKind.VCC:
            self.reg_file.vcc = vcc_bit
        elif inst.sdst.kind == OperandKind.SGPR_PAIR:
            self.reg_file.write_sgpr_pair_u64(inst.sdst.index, vcc_bit)
        elif inst.sdst.kind == OperandKind.SGPR_SINGLE:
            self.reg_file.write_sgpr(inst.sdst.index, vcc_bit)

        return d_u64, vcc_bit


class TestDivScaleF64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_interpreter_basic_scaling(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # Set v[2:3] = one5, v[4:5] = big
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, big)

        # v_div_scale_f64 v[0:1], vcc, v[2:3], v[2:3], v[4:5]
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(rf.vcc, 1)
        self.assertEqual(rf.read_vgpr_pair_u64(0), d_res)

        # Expected word components from case 1
        self.assertEqual(rf.read_vgpr(0), 0x00000005)
        self.assertEqual(rf.read_vgpr(1), 0x47F00000)

    def test_interpreter_source_modifiers_neg(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, big)

        # neg on src0 only makes s0 != s1, unscaled result
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2, neg=True),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(d_res, one5 ^ (1 << 63))

    def test_interpreter_sdst_sgpr(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, big)

        # Write scale bit to SGPR pair s[10:11]
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.sdst_sgpr(10),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )
        interp.step(inst)
        self.assertEqual(rf.read_sgpr_pair_u64(10), 1)

    def test_interpreter_clamp_modifier(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        # Set v[2:3] = 2.0, v[4:5] = 1.0, v[6:7] = 1.0
        rf.write_vgpr_pair_u64(2, F64_TWO)
        rf.write_vgpr_pair_u64(4, F64_ONE)
        rf.write_vgpr_pair_u64(6, F64_ONE)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
            clamp=True,
        )
        d_res, _ = interp.step(inst)
        self.assertEqual(d_res, F64_ONE)
        self.assertEqual(rf.read_vgpr_pair_u64(0), F64_ONE)


class SIMDWaveState:
    """Architectural model of a SIMD wavefront (Wave32 or Wave64) with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 108

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

    def set_lane_vgpr_pair_u64(self, lane: int, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.vgprs[lane][idx] = lo
        self.vgprs[lane][idx + 1] = hi

    def get_lane_vgpr_pair_u64(self, lane: int, idx: int) -> int:
        return dwords_to_u64(self.vgprs[lane][idx], self.vgprs[lane][idx + 1])

    def set_lane_vgpr_pair_f64(self, lane: int, idx: int, val: float) -> None:
        self.set_lane_vgpr_pair_u64(lane, idx, f64_to_u64(val))

    def get_lane_vgpr_pair_f64(self, lane: int, idx: int) -> float:
        return u64_to_f64(self.get_lane_vgpr_pair_u64(lane, idx))

    def set_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def get_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def set_sgpr_pair_f64(self, idx: int, val: float) -> None:
        self.set_sgpr_pair_u64(idx, f64_to_u64(val))

    def get_sgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.get_sgpr_pair_u64(idx))

    @property
    def vcc(self) -> int:
        if self.wave_size == 32:
            return self.sgprs[106]
        return dwords_to_u64(self.sgprs[106], self.sgprs[107])

    @vcc.setter
    def vcc(self, val: int) -> None:
        if self.wave_size == 32:
            self.sgprs[106] = val & MASK32
        else:
            self.sgprs[106], self.sgprs[107] = u64_to_dwords(val)


class SIMDInterpreter:
    """Multi-lane SIMD wavefront interpreter with EXEC masking."""

    def __init__(self, wave: SIMDWaveState) -> None:
        self.wave = wave

    def resolve_lane_source_u64(self, lane: int, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR_PAIR:
            val_u64 = self.wave.get_lane_vgpr_pair_u64(lane, src.index)
        elif src.kind == OperandKind.SGPR_PAIR:
            val_u64 = self.wave.get_sgpr_pair_u64(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u64 = f64_to_u64(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u64(val_u64, neg=src.neg, abs_=src.abs_)

    def step(self, inst: ASTInstruction) -> int:
        inst.validate()
        sdst_mask = 0

        for lane in range(self.wave.wave_size):
            is_active = bool((self.wave.exec_mask >> lane) & 1)
            if not is_active:
                # Inactive lanes: destination is not modified, and sdst bit is 0
                continue

            u0 = self.resolve_lane_source_u64(lane, inst.src0)
            u1 = self.resolve_lane_source_u64(lane, inst.src1)
            u2 = self.resolve_lane_source_u64(lane, inst.src2)

            d_u64, vcc_bit = ref_div_scale_f64_u64(
                u0,
                u1,
                u2,
                clamp=inst.clamp,
                omod=inst.omod.value,
            )

            self.wave.set_lane_vgpr_pair_u64(lane, inst.dst.index, d_u64)
            if vcc_bit:
                sdst_mask |= (1 << lane)

        # Write sdst mask
        if inst.sdst.kind == OperandKind.VCC:
            self.wave.vcc = sdst_mask
        elif inst.sdst.kind == OperandKind.SGPR_PAIR:
            self.wave.set_sgpr_pair_u64(inst.sdst.index, sdst_mask)
        elif inst.sdst.kind == OperandKind.SGPR_SINGLE:
            self.wave.sgprs[inst.sdst.index] = sdst_mask & MASK32

        return sdst_mask


class TestDivScaleF64SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_all_active(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        for l in range(32):
            wave.set_lane_vgpr_pair_u64(l, 2, one5)
            wave.set_lane_vgpr_pair_u64(l, 4, big)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(is_wave32=True),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        vcc_mask = interp.step(inst)
        self.assertEqual(vcc_mask, MASK32)
        self.assertEqual(wave.vcc, MASK32)

        for l in range(32):
            self.assertEqual(wave.vgprs[l][0], 0x00000005)
            self.assertEqual(wave.vgprs[l][1], 0x47F00000)

    def test_simd_wave64_divergent_exec_mask(self):
        wave = SIMDWaveState(wave_size=64)
        interp = SIMDInterpreter(wave)

        # Alternating mask: 0x55555555_55555555 (even lanes active, odd lanes inactive)
        alt_mask = 0x5555555555555555
        wave.exec_mask = alt_mask

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        sentinel = 0xDEADBEEFCAFEBABE

        for l in range(64):
            wave.set_lane_vgpr_pair_u64(l, 0, sentinel)
            wave.set_lane_vgpr_pair_u64(l, 2, one5)
            wave.set_lane_vgpr_pair_u64(l, 4, big)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        vcc_mask = interp.step(inst)
        # sdst receives scale bit only for active lanes
        self.assertEqual(vcc_mask, alt_mask)
        self.assertEqual(wave.vcc, alt_mask)

        for l in range(64):
            if l % 2 == 0:
                # Active lane: updated
                self.assertEqual(wave.vgprs[l][0], 0x00000005)
                self.assertEqual(wave.vgprs[l][1], 0x47F00000)
            else:
                # Inactive lane: strictly preserved sentinel
                self.assertEqual(wave.get_lane_vgpr_pair_u64(l, 0), sentinel)

    def test_simd_divergent_operands_across_lanes(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        normal = 0x3FF0000000000000

        for l in range(32):
            if l < 16:
                # First 16 lanes: e2 - e1 >= 768 -> scale up and vcc = 1
                wave.set_lane_vgpr_pair_u64(l, 2, one5)
                wave.set_lane_vgpr_pair_u64(l, 4, big)
            else:
                # Last 16 lanes: normal 1.0 / 1.0 -> no scale, vcc = 0
                wave.set_lane_vgpr_pair_u64(l, 2, normal)
                wave.set_lane_vgpr_pair_u64(l, 4, normal)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(is_wave32=True),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        vcc_mask = interp.step(inst)
        self.assertEqual(vcc_mask, 0x0000FFFF)
        self.assertEqual(wave.vcc, 0x0000FFFF)

        for l in range(16):
            self.assertEqual(wave.vgprs[l][0], 0x00000005)
            self.assertEqual(wave.vgprs[l][1], 0x47F00000)
        for l in range(16, 32):
            self.assertEqual(wave.get_lane_vgpr_pair_u64(l, 0), normal)

    def test_exec_lo_hi_properties(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0x12345678
        wave.exec_hi = 0x9ABCDEF0
        self.assertEqual(wave.exec_mask, 0x9ABCDEF012345678)
        self.assertEqual(wave.exec_lo, 0x12345678)
        self.assertEqual(wave.exec_hi, 0x9ABCDEF0)


class TestDivScaleF64RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_alias_src0_in_place(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # v[0:1] holds s0 (one5), v[2:3] holds s1 (one5), v[4:5] holds s2 (big)
        rf.write_vgpr_pair_u64(0, one5)
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, big)

        # In-place scaling of v[0:1]
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(rf.read_vgpr_pair_u64(0), d_res)
        self.assertEqual(rf.read_vgpr(0), 0x00000005)
        self.assertEqual(rf.read_vgpr(1), 0x47F00000)

        # src1 and src2 must be strictly preserved
        self.assertEqual(rf.read_vgpr_pair_u64(2), one5)
        self.assertEqual(rf.read_vgpr_pair_u64(4), big)

    def test_dst_alias_src1_denominator(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        rf.write_vgpr_pair_u64(0, one5)  # dst and src1
        rf.write_vgpr_pair_u64(2, one5)  # src0
        rf.write_vgpr_pair_u64(4, big)   # src2

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(4),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(rf.read_vgpr_pair_u64(0), d_res)
        self.assertEqual(rf.read_vgpr_pair_u64(2), one5)
        self.assertEqual(rf.read_vgpr_pair_u64(4), big)

    def test_dst_alias_src2_numerator(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        rf.write_vgpr_pair_u64(0, big)   # dst and src2
        rf.write_vgpr_pair_u64(2, one5)  # src0
        rf.write_vgpr_pair_u64(4, one5)  # src1

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(0),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(rf.read_vgpr_pair_u64(0), d_res)
        self.assertEqual(rf.read_vgpr_pair_u64(2), one5)
        self.assertEqual(rf.read_vgpr_pair_u64(4), one5)

    def test_all_operands_fully_aliased(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        normal = 0x3FF0000000000000  # 1.0
        rf.write_vgpr_pair_u64(0, normal)

        # v_div_scale_f64 v[0:1], vcc, v[0:1], v[0:1], v[0:1]
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 0)
        self.assertEqual(d_res, normal)
        self.assertEqual(rf.read_vgpr_pair_u64(0), normal)

    def test_unrelated_registers_strictly_preserved(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        sentinel = 0xDEADBEEF
        for i in range(16):
            rf.write_vgpr(i, sentinel)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, one5)
        rf.write_vgpr_pair_u64(6, big)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        interp.step(inst)

        # v[0:1] updated
        self.assertEqual(rf.read_vgpr(0), 0x00000005)
        self.assertEqual(rf.read_vgpr(1), 0x47F00000)

        # v[8..15] must still be sentinel
        for i in range(8, 16):
            self.assertEqual(rf.read_vgpr(i), sentinel)


class StatusFlags:
    """Architectural condition codes and status flags for double-precision float operations."""

    def __init__(
        self,
        *,
        is_zero: bool = False,
        is_negative: bool = False,
        is_nan: bool = False,
        is_inf: bool = False,
        is_subnormal: bool = False,
        is_scaled: bool = False,
        vcc_bit: int = 0,
    ) -> None:
        self.is_zero = is_zero
        self.is_negative = is_negative
        self.is_nan = is_nan
        self.is_inf = is_inf
        self.is_subnormal = is_subnormal
        self.is_scaled = is_scaled
        self.vcc_bit = vcc_bit

    @classmethod
    def from_f64_bits(cls, bits: int, vcc_bit: int = 0) -> StatusFlags:
        sign = bool(bits & (1 << 63))
        exp = (bits >> 52) & 0x7FF
        mant = bits & F64_SUB_MAX

        is_zero = False
        is_negative = sign
        is_nan = False
        is_inf = False
        is_subnormal = False

        if exp == 0x7FF:
            if mant == 0:
                is_inf = True
            else:
                is_nan = True
        elif exp == 0:
            if mant == 0:
                is_zero = True
            else:
                is_subnormal = True

        return cls(
            is_zero=is_zero,
            is_negative=is_negative,
            is_nan=is_nan,
            is_inf=is_inf,
            is_subnormal=is_subnormal,
            is_scaled=bool(vcc_bit),
            vcc_bit=vcc_bit,
        )


class TestDivScaleF64StatusFlagsAndConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow (VCC/SCC) isolation."""

    def test_status_flags_classification(self):
        # Zero
        flags_zero = StatusFlags.from_f64_bits(F64_POS_ZERO)
        self.assertTrue(flags_zero.is_zero)
        self.assertFalse(flags_zero.is_negative)
        self.assertFalse(flags_zero.is_nan)

        # Negative Zero
        flags_neg_zero = StatusFlags.from_f64_bits(F64_NEG_ZERO)
        self.assertTrue(flags_neg_zero.is_zero)
        self.assertTrue(flags_neg_zero.is_negative)

        # Infinity
        flags_inf = StatusFlags.from_f64_bits(F64_POS_INF)
        self.assertTrue(flags_inf.is_inf)
        self.assertFalse(flags_inf.is_nan)

        # NaN
        flags_nan = StatusFlags.from_f64_bits(F64_CANONICAL_QNAN)
        self.assertTrue(flags_nan.is_nan)
        self.assertFalse(flags_nan.is_inf)

        # Subnormal
        flags_sub = StatusFlags.from_f64_bits(F64_SUB_MIN)
        self.assertTrue(flags_sub.is_subnormal)
        self.assertFalse(flags_sub.is_zero)

    def test_vcc_condition_flag_generation(self):
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # Case 1: e2 - e1 >= 768 and s0 == s1 -> vcc = 1
        d1, vcc1 = ref_div_scale_f64_u64(one5, one5, big)
        flags1 = StatusFlags.from_f64_bits(d1, vcc1)
        self.assertEqual(vcc1, 1)
        self.assertTrue(flags1.is_scaled)

        # Case 6: normal division 1.0 / 1.0 -> vcc = 0
        d0, vcc0 = ref_div_scale_f64_u64(F64_ONE, F64_ONE, F64_ONE)
        flags0 = StatusFlags.from_f64_bits(d0, vcc0)
        self.assertEqual(vcc0, 0)
        self.assertFalse(flags0.is_scaled)

    def test_scc_isolation_and_preservation(self):
        # VALU instructions do not modify SCC (Scalar Condition Code)
        rf = ScalarRegisterFile()
        interp = DivScaleF64ScalarInterpreter(rf)

        # Arbitrary simulated SCC stored in SGPR 105
        scc_sentinel = 1
        rf.write_sgpr(105, scc_sentinel)

        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        rf.write_vgpr_pair_u64(2, one5)
        rf.write_vgpr_pair_u64(4, big)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        interp.step(inst)

        # VCC is modified, but SCC (SGPR 105) is preserved
        self.assertEqual(rf.vcc, 1)
        self.assertEqual(rf.read_sgpr(105), scc_sentinel)


class TestDivScaleF64IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, infinities, and NaNs."""

    def test_subnormal_scale_up(self):
        # Denormal in S1: e1 == 0 -> scales S0 by 2^128
        sub_min = F64_SUB_MIN
        normal_s0 = F64_ONE

        d_res, vcc_res = ref_div_scale_f64_u64(normal_s0, sub_min, normal_s0)
        # e2 (1023) - e1 (0) = 1023 >= 768, so rule 1 matches:
        # vcc = 1, D = S0 * 2^128 if S0 == S1 else S0
        # Since S0 != S1 (F64_ONE != F64_SUB_MIN), D = S0
        self.assertEqual(vcc_res, 1)
        self.assertEqual(d_res, normal_s0)

        # When S0 == S1 (both subnormals):
        d_sub, vcc_sub = ref_div_scale_f64_u64(sub_min, sub_min, normal_s0)
        self.assertEqual(vcc_sub, 1)
        # sub_min * 2^128
        self.assertEqual(d_sub, scale_f64_bits(sub_min, 128))

    def test_threshold_boundaries_exact(self):
        # Rule 1 exact threshold: e2 - e1 >= 768
        # Let e1 = 1000, then e2 = 1768 (diff = 768 -> matches), e2 = 1767 (diff = 767 -> does not match rule 1)
        s1 = 1000 << 52
        s2_match = 1768 << 52
        s2_nomatch = 1767 << 52

        d_m, vcc_m = ref_div_scale_f64_u64(s1, s1, s2_match)
        self.assertEqual(vcc_m, 1)
        self.assertEqual(d_m, scale_f64_bits(s1, 128))

        d_nm, vcc_nm = ref_div_scale_f64_u64(s1, s1, s2_nomatch)
        # diff 767 does not match rule 1, does not match e1==0, does not match e1>=2045, etc.
        self.assertEqual(vcc_nm, 0)
        self.assertEqual(d_nm, s1)

        # Rule 6 threshold: e2 <= 53 (with e1 - e2 < 768 to avoid rule 5)
        s1_mid = 500 << 52
        s2_53 = 53 << 52
        s2_54 = 54 << 52
        d_53, vcc_53 = ref_div_scale_f64_u64(s1_mid, s1_mid, s2_53)
        self.assertEqual(vcc_53, 0)
        self.assertEqual(d_53, scale_f64_bits(s1_mid, 128))

        d_54, vcc_54 = ref_div_scale_f64_u64(s1_mid, s1_mid, s2_54)
        self.assertEqual(vcc_54, 0)
        self.assertEqual(d_54, s1_mid)

    def test_zero_denominator_or_numerator_produces_nan(self):
        # Measured RDNA2 behavior: A zero S1 or S2 makes D = 0xFFF8000000000000 (NaN)
        # +0.0 denominator
        d_z1, _ = ref_div_scale_f64_u64(F64_ONE, F64_POS_ZERO, F64_ONE)
        self.assertEqual(d_z1, 0xFFF8000000000000)

        # -0.0 denominator
        d_zn1, _ = ref_div_scale_f64_u64(F64_ONE, F64_NEG_ZERO, F64_ONE)
        self.assertEqual(d_zn1, 0xFFF8000000000000)

        # +0.0 numerator
        d_z2, _ = ref_div_scale_f64_u64(F64_ONE, F64_ONE, F64_POS_ZERO)
        self.assertEqual(d_z2, 0xFFF8000000000000)

        # -0.0 numerator
        d_zn2, _ = ref_div_scale_f64_u64(F64_ONE, F64_ONE, F64_NEG_ZERO)
        self.assertEqual(d_zn2, 0xFFF8000000000000)

    def test_nan_and_infinity_preservation(self):
        # Scaling keeps NaN (payload preserved) and infinities
        qnan_payload = F64_CANONICAL_QNAN | 0x12345
        d_qnan, vcc_q = ref_div_scale_f64_u64(qnan_payload, F64_ONE, 50 << 52)
        self.assertEqual(d_qnan, qnan_payload)

        # Infinity
        d_inf, _ = ref_div_scale_f64_u64(F64_POS_INF, F64_ONE, 50 << 52)
        self.assertEqual(d_inf, F64_POS_INF)

        d_ninf, _ = ref_div_scale_f64_u64(F64_NEG_INF, F64_ONE, 50 << 52)
        self.assertEqual(d_ninf, F64_NEG_INF)

    def test_scale_down_extreme_to_infinity(self):
        # Measured: 2^-128 scaling with ex(b) <= 116 gives infinity of S0's sign
        small_val = 110 << 52
        d_inf_pos = scale_f64_bits(small_val, -128)
        self.assertEqual(d_inf_pos, F64_POS_INF)

        small_neg = (110 << 52) | (1 << 63)
        d_inf_neg = scale_f64_bits(small_neg, -128)
        self.assertEqual(d_inf_neg, F64_NEG_INF)

    def test_clamp_boundary_extremes(self):
        # Negative value -> +0.0
        d_neg, _ = ref_div_scale_f64_u64(F64_NEG_ONE, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_neg, F64_POS_ZERO)

        # Greater than 1.0 -> 1.0
        d_gt, _ = ref_div_scale_f64_u64(F64_TWO, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_gt, F64_ONE)

        # NaN -> +0.0
        d_nan, _ = ref_div_scale_f64_u64(F64_CANONICAL_QNAN, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_nan, F64_POS_ZERO)


def mutant_omit_scale(u0: int, u1: int, u2: int, **kwargs) -> tuple[int, int]:
    """Mutant 1: Omits scaling logic and returns unscaled s0."""
    _, vcc = ref_div_scale_f64_u64(u0, u1, u2, **kwargs)
    return u0, vcc


def mutant_invert_scale_direction(u0: int, u1: int, u2: int, **kwargs) -> tuple[int, int]:
    """Mutant 2: Scales down (2^-128) when it should scale up (2^128)."""
    d, vcc = ref_div_scale_f64_u64(u0, u1, u2, **kwargs)
    if vcc == 1 and ex_f64(u2) - ex_f64(u1) >= 768 and u0 == u1:
        return scale_f64_bits(u0, -128), 1
    return d, vcc


def mutant_omit_vcc_flag(u0: int, u1: int, u2: int, **kwargs) -> tuple[int, int]:
    """Mutant 3: Computes scaled result but fails to set VCC flag."""
    d, _ = ref_div_scale_f64_u64(u0, u1, u2, **kwargs)
    return d, 0


def mutant_zero_not_nan(u0: int, u1: int, u2: int, **kwargs) -> tuple[int, int]:
    """Mutant 4: Zero S1 or S2 returns S0 instead of 0xFFF8000000000000."""
    if not (u1 & ~(1 << 63)) or not (u2 & ~(1 << 63)):
        return u0, 1
    return ref_div_scale_f64_u64(u0, u1, u2, **kwargs)


def mutant_clamp_nan_to_nan(u0: int, u1: int, u2: int, *, clamp: bool = False, **kwargs) -> tuple[int, int]:
    """Mutant 5: Clamps numbers but fails to map NaN to +0.0."""
    res_u64, vcc = ref_div_scale_f64_u64(u0, u1, u2, clamp=False, **kwargs)
    if clamp:
        x = u64_to_f64(res_u64)
        if math.isnan(x):
            return res_u64, vcc
        elif x < 0.0:
            return F64_POS_ZERO, vcc
        elif x > 1.0:
            return F64_ONE, vcc
    return res_u64, vcc


class TestDivScaleF64NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, and mutation detection."""

    def test_syntax_rejection_bad_opcodes(self):
        for bad_op in ("v_div_scale_f32", "v_add_f64", "s_div_scale_f64", "foo_bar"):
            with self.assertRaises(ValueError):
                parse_div_scale_asm(f"{bad_op} v[0:1], vcc, v[2:3], v[2:3], v[4:5]")

    def test_syntax_rejection_operand_count(self):
        # Too few operands
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 v[0:1], vcc, v[2:3], v[4:5]")
        # Too many operands
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 v[0:1], vcc, v[2:3], v[4:5], v[6:7], v[8:9]")

    def test_syntax_rejection_destination_modifiers(self):
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 -v[0:1], vcc, v[2:3], v[4:5], v[6:7]")
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 |v[0:1]|, vcc, v[2:3], v[4:5], v[6:7]")

    def test_syntax_rejection_destination_sgpr(self):
        with self.assertRaises(TypeError):
            parse_div_scale_asm("v_div_scale_f64 s[0:1], vcc, v[2:3], v[4:5], v[6:7]")

    def test_syntax_rejection_non_consecutive_register_pair(self):
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 v[0:2], vcc, v[2:3], v[4:5], v[6:7]")
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 v[0:1], vcc, v[2:4], v[4:5], v[6:7]")

    def test_syntax_rejection_register_out_of_bounds(self):
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f64 v[256:257], vcc, v[2:3], v[4:5], v[6:7]")

    def test_mutation_detection_all_mutants_caught(self):
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # Mutant 1: omit scale
        good_res, good_vcc = ref_div_scale_f64_u64(one5, one5, big)
        bad_res1, bad_vcc1 = mutant_omit_scale(one5, one5, big)
        self.assertNotEqual(good_res, bad_res1)
        self.assertEqual(bad_res1, one5)

        # Mutant 2: invert scale direction
        bad_res2, _ = mutant_invert_scale_direction(one5, one5, big)
        self.assertNotEqual(good_res, bad_res2)

        # Mutant 3: omit VCC flag
        _, bad_vcc3 = mutant_omit_vcc_flag(one5, one5, big)
        self.assertEqual(good_vcc, 1)
        self.assertEqual(bad_vcc3, 0)
        self.assertNotEqual(good_vcc, bad_vcc3)

        # Mutant 4: zero denominator does not return 0xFFF8000000000000
        good_z, _ = ref_div_scale_f64_u64(one5, F64_POS_ZERO, one5)
        bad_z, _ = mutant_zero_not_nan(one5, F64_POS_ZERO, one5)
        self.assertEqual(good_z, 0xFFF8000000000000)
        self.assertNotEqual(good_z, bad_z)

        # Mutant 5: clamp NaN to NaN
        good_clamp, _ = ref_div_scale_f64_u64(F64_CANONICAL_QNAN, F64_ONE, F64_ONE, clamp=True)
        bad_clamp, _ = mutant_clamp_nan_to_nan(F64_CANONICAL_QNAN, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(good_clamp, F64_POS_ZERO)
        self.assertNotEqual(good_clamp, bad_clamp)

    def test_mutation_detection_simd_exec_mask_leak(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0  # All lanes inactive
        sentinel = 0xCAFEBABE12345678
        for l in range(32):
            wave.set_lane_vgpr_pair_u64(l, 0, sentinel)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F64,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(is_wave32=True),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )

        interp = SIMDInterpreter(wave)
        interp.step(inst)

        # Inactive lanes must be strictly preserved
        for l in range(32):
            self.assertEqual(wave.get_lane_vgpr_pair_u64(l, 0), sentinel)


if __name__ == "__main__":
    unittest.main()
