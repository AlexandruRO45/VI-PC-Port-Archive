from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale_f32.

Tests the single-precision floating-point division pre-scaling instruction:
    v_div_scale_f32 D, sdst, S0, S1, S2

Hardware Specification (RDNA2 / GFX10 / GFX1030 VOP3b dual-output):
    D    = 32-bit vector destination register v[d] receiving scaled S0
    sdst = scalar destination (VCC or SGPR) receiving scale condition bit mask
    S0   = operand to scale (32-bit float, typically denominator S1 or numerator S2)
    S1   = denominator (32-bit float)
    S2   = numerator (32-bit float)

Operation:
    v_div_scale_f32 examines the denominator S1 and numerator S2 to check whether
    intermediate Newton-Raphson division approximation will overflow or underflow.
    If scaling is necessary:
      - If 0.0 < |S1| <= 2^-88: D = S0 * 2^64, sdst = 1
      - Else if |S1| >= 2^62 and finite: D = S0 * 2^-64, sdst = 1
      - Else if 0.0 < |S2| <= 2^-88: D = S0 * 2^64, sdst = 1
      - Else if |S2| >= 2^62 and finite: D = S0 * 2^-64, sdst = 1
      - Else: D = S0, sdst = 0

Special Cases:
    - If S0 is NaN: D = quiet(S0), sdst = 0.
    - If S1 or S2 is NaN: D = S0, sdst = 0.
    - Zero/Infinity: if not matching threshold conditions, unscaled, sdst = 0.

Modifiers:
    - Input modifiers: abs (|src|: clear sign bit), neg (-src: invert sign bit).
    - Output modifiers:
        - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0.
        - omod: mul:2, mul:4, div:2.

Architectural Model:
    32-bit registers (VGPR / SGPR) with IEEE 754 binary32 representation.
"""

import math
import struct
import unittest
from enum import Enum, auto

# IEEE 754 single precision constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_CANONICAL_QNAN = 0x7FC00000
F32_SNAN = 0x7F800001
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF
F32_MIN_NORMAL = 0x00800000
F32_MAX_NORMAL = 0x7F7FFFFF

MASK32 = 0xFFFFFFFF

# Scaling constants for f32
SCALE64_POS = 2.0 ** 64
SCALE64_NEG = 2.0 ** -64

THRESH_F32_SMALL = 2.0 ** -88
THRESH_F32_LARGE = 2.0 ** 62


def f32_to_u32(val: float) -> int:
    """Pack Python float into unsigned 32-bit integer (IEEE 754 binary32)."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack unsigned 32-bit integer into Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def quiet_f32(bits: int) -> int:
    return (bits & MASK32) | 0x00400000


def apply_src_mods_u32(val_u32: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 32-bit float bit pattern."""
    bits = val_u32 & MASK32
    if abs_:
        bits &= ~F32_NEG_ZERO
    if neg:
        bits ^= F32_NEG_ZERO
    return bits


def apply_clamp_f32(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float32, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_f32(val: float, omod: int) -> float:
    """Apply RDNA2 omod modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")


def ref_div_scale_f32(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f32.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u32(f32_to_u32(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u32(f32_to_u32(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u32(f32_to_u32(s2), neg=neg2, abs_=abs2)

    if is_f32_nan(u0):
        res = u32_to_f32(quiet_f32(u0))
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    if is_f32_nan(u1) or is_f32_nan(u2):
        res = u32_to_f32(u0)
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    v0 = u32_to_f32(u0)
    v1 = u32_to_f32(u1)
    v2 = u32_to_f32(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    # Denominator check has priority
    if 0.0 < abs_y <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_y >= THRESH_F32_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE64_NEG
    elif 0.0 < abs_x <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_x >= THRESH_F32_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE64_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_f32(res, omod)
    if clamp:
        res = apply_clamp_f32(res)

    return res, vcc


def ref_div_scale_f32_u32(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[int, int]:
    """Bit-exact reference mathematical oracle for v_div_scale_f32 using unsigned 32-bit words."""
    res_f, vcc = ref_div_scale_f32(
        u32_to_f32(u0),
        u32_to_f32(u1),
        u32_to_f32(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f32_to_u32(res_f), vcc


class ScalarRegister:
    """Architectural model of a 32-bit vector or scalar register."""

    def __init__(self, raw: int = 0) -> None:
        self.raw = raw & MASK32

    @classmethod
    def from_u32(cls, val: int) -> ScalarRegister:
        return cls(val)

    @classmethod
    def from_f32(cls, val: float) -> ScalarRegister:
        return cls(f32_to_u32(val))

    @property
    def as_u32(self) -> int:
        return self.raw

    @property
    def as_f32(self) -> float:
        return u32_to_f32(self.raw)

    def set_u32(self, val: int) -> None:
        self.raw = val & MASK32

    def set_f32(self, val: float) -> None:
        self.raw = f32_to_u32(val)

    def __repr__(self) -> str:
        return f"ScalarRegister(raw={self.raw:#010x}, f32={self.as_f32})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarRegister):
            return NotImplemented
        return self.raw == other.raw


class TestDivScaleF32MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_normal_no_scale(self):
        # 10.0 / 2.0 with s0 = 2.0 (denominator) -> no scale required
        res, vcc = ref_div_scale_f32(2.0, 2.0, 10.0)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        # 10.0 / 2.0 with s0 = 10.0 (numerator) -> no scale required
        res_num, vcc_num = ref_div_scale_f32(10.0, 2.0, 10.0)
        self.assertEqual(res_num, 10.0)
        self.assertEqual(vcc_num, 0)

    def test_scale_up_small_denominator(self):
        # Small denominator (e.g. 2^-100): scale up by 2^64
        small_denom = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE64_POS)

    def test_scale_down_large_denominator(self):
        # Large denominator (e.g. 2^70): scale down by 2^-64
        large_denom = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE64_NEG)

    def test_scale_up_small_numerator(self):
        # Small numerator (e.g. 2^-100): scale up by 2^64
        small_num = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_num, 1.0, small_num)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_num * SCALE64_POS)

    def test_scale_down_large_numerator(self):
        # Large numerator (e.g. 2^70): scale down by 2^-64
        large_num = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_num, 1.0, large_num)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_num * SCALE64_NEG)

    def test_source_modifiers_neg_abs(self):
        # abs clears sign bit
        res_abs, _ = ref_div_scale_f32(-2.5, 2.0, 1.0, abs0=True)
        self.assertEqual(res_abs, 2.5)

        # neg inverts sign bit
        res_neg, _ = ref_div_scale_f32(2.5, 2.0, 1.0, neg0=True)
        self.assertEqual(res_neg, -2.5)

        # combined neg and abs: -|src|
        res_neg_abs, _ = ref_div_scale_f32(2.5, 2.0, 1.0, neg0=True, abs0=True)
        self.assertEqual(res_neg_abs, -2.5)

    def test_output_modifiers_clamp_omod(self):
        # clamp saturates to [0.0, 1.0]
        res_clamp_hi, _ = ref_div_scale_f32(2.5, 2.0, 1.0, clamp=True)
        self.assertEqual(res_clamp_hi, 1.0)

        res_clamp_lo, _ = ref_div_scale_f32(-0.5, 2.0, 1.0, clamp=True)
        self.assertEqual(res_clamp_lo, 0.0)

        # omod: mul:2, mul:4, div:2
        res_mul2, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=1)
        self.assertEqual(res_mul2, 3.0)

        res_mul4, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=2)
        self.assertEqual(res_mul4, 6.0)

        res_div2, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=3)
        self.assertEqual(res_div2, 0.75)

    def test_architectural_register_model(self):
        reg = ScalarRegister.from_f32(1.5)
        self.assertEqual(reg.as_f32, 1.5)
        self.assertEqual(reg.as_u32, 0x3FC00000)

        reg.set_f32(-1.0)
        self.assertEqual(reg.as_f32, -1.0)
        self.assertEqual(reg.as_u32, F32_NEG_ONE)


class Opcode(Enum):
    V_DIV_SCALE_F32 = "v_div_scale_f32"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    VCC = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT32 = "f32"
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
        return cls(OperandKind.VGPR, index=index, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT32)

    @classmethod
    def sgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR, index=index, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT32)

    @classmethod
    def sdst_sgpr(cls, index: int) -> ASTOperand:
        return cls(OperandKind.SGPR, index=index, type_kind=TypeKind.UINT32)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(OperandKind.VCC, index=106, type_kind=TypeKind.UINT32)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT32)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_, type_kind=TypeKind.FLOAT32)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR:
            body = f"v{self.index}"
        elif self.kind == OperandKind.SGPR:
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
    """Symbolic AST instruction representation for v_div_scale_f32."""

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
        if self.opcode != Opcode.V_DIV_SCALE_F32:
            raise ValueError(f"Invalid opcode for v_div_scale_f32 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError("Destination must be a single VGPR")
        if self.dst.type_kind != TypeKind.FLOAT32:
            raise TypeError(f"Destination type must be FLOAT32, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 255):
            raise ValueError(f"Destination VGPR index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        if self.sdst.kind not in (OperandKind.VCC, OperandKind.SGPR):
            raise TypeError(f"sdst must be VCC or SGPR, got {self.sdst.kind}")
        if self.sdst.neg or self.sdst.abs_:
            raise ValueError("sdst operand cannot have neg/abs modifiers")

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

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float | tuple[float, int]:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Symbolic variable reference."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
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

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier expression: neg or abs."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT32) -> None:
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
    """Symbolic DivScale ternary typed expression: v_div_scale_f32 with modifiers."""

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

    def evaluate(self, env: dict[str, float]) -> tuple[float, int]:
        v0 = self.src0.evaluate(env)
        v1 = self.src1.evaluate(env)
        v2 = self.src2.evaluate(env)
        if isinstance(v0, tuple) or isinstance(v1, tuple) or isinstance(v2, tuple):
            raise TypeError("Subexpressions must evaluate to float")
        return ref_div_scale_f32(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res_val, _ = ref_div_scale_f32(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res_val, self.type_kind)
        return DivScaleExpr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"DivScale({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


def parse_operand_asm(op_str: str, *, is_sdst: bool = False) -> ASTOperand:
    """Parse a single assembly operand token into an ASTOperand for f32."""
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
        if op_str.startswith("s") and op_str[1:].isdigit():
            idx = int(op_str[1:])
            if not (0 <= idx <= 105):
                raise ValueError(f"SGPR index out of range: {idx}")
            return ASTOperand.sdst_sgpr(idx)
        raise ValueError(f"Invalid sdst operand: '{op_str}'")

    if op_str.startswith("v") and op_str[1:].isdigit():
        idx = int(op_str[1:])
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of range: {idx}")
        return ASTOperand.vgpr(idx, neg=neg, abs_=abs_)

    if op_str.startswith("s") and op_str[1:].isdigit():
        idx = int(op_str[1:])
        if not (0 <= idx <= 105):
            raise ValueError(f"SGPR index out of range: {idx}")
        return ASTOperand.sgpr(idx, neg=neg, abs_=abs_)

    try:
        val = float(op_str)
        return ASTOperand.inline_const(val, neg=neg, abs_=abs_)
    except ValueError:
        pass

    raise ValueError(f"Unable to parse operand: '{op_str}'")


def parse_div_scale_asm(asm_line: str) -> ASTInstruction:
    """Parse an assembly line string into an ASTInstruction for v_div_scale_f32."""
    line = asm_line.strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    op_name = tokens[0]
    if op_name != Opcode.V_DIV_SCALE_F32.value:
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
        opcode=Opcode.V_DIV_SCALE_F32,
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


class TestDivScaleF32SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_scale_f32 v0, vcc, v1, v1, v2")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.sdst_sgpr(4),
            src0=ASTOperand.vgpr(1, neg=True, abs_=True),
            src1=ASTOperand.sgpr(2),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_div_scale_f32 v0, s4, -|v1|, s2, 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind
        bad_dst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.sgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0, neg=True),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(256),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        expr = DivScaleExpr(
            VariableExpr("s0"),
            VariableExpr("s1"),
            VariableExpr("s2"),
        )
        env = {"s0": 2.0, "s1": 2.0, "s2": 10.0}
        d_val, vcc_val = expr.evaluate(env)
        self.assertEqual(d_val, 2.0)
        self.assertEqual(vcc_val, 0)

    def test_typed_expression_constant_folding(self):
        c_expr = DivScaleExpr(
            ConstantExpr(2.0),
            ConstantExpr(2.0),
            ConstantExpr(10.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 2.0)

    def test_asm_parsing_roundtrip(self):
        asm = "v_div_scale_f32 v0, vcc, -|v1|, s2, 1.0 clamp"
        inst = parse_div_scale_asm(asm)
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.sdst.kind, OperandKind.VCC)
        self.assertTrue(inst.src0.neg)
        self.assertTrue(inst.src0.abs_)
        self.assertEqual(inst.src1.index, 2)
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

    def read_vgpr_f32(self, idx: int) -> float:
        return u32_to_f32(self.vgprs[idx])

    def write_vgpr_f32(self, idx: int, val: float) -> None:
        self.vgprs[idx] = f32_to_u32(val)

    def read_sgpr_f32(self, idx: int) -> float:
        return u32_to_f32(self.sgprs[idx])

    def write_sgpr_f32(self, idx: int, val: float) -> None:
        self.sgprs[idx] = f32_to_u32(val)

    @property
    def vcc(self) -> int:
        return self.sgprs[106]

    @vcc.setter
    def vcc(self, val: int) -> None:
        self.sgprs[106] = val & MASK32


class DivScaleF32ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_div_scale_f32."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_u32(self, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR:
            val_u32 = self.reg_file.read_vgpr(src.index)
        elif src.kind == OperandKind.SGPR:
            val_u32 = self.reg_file.read_sgpr(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u32 = f32_to_u32(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u32(val_u32, neg=src.neg, abs_=src.abs_)

    def step(self, inst: ASTInstruction) -> tuple[int, int]:
        inst.validate()
        u0 = self.resolve_source_u32(inst.src0)
        u1 = self.resolve_source_u32(inst.src1)
        u2 = self.resolve_source_u32(inst.src2)

        d_u32, vcc_bit = ref_div_scale_f32_u32(
            u0,
            u1,
            u2,
            clamp=inst.clamp,
            omod=inst.omod.value,
        )

        self.reg_file.write_vgpr(inst.dst.index, d_u32)

        if inst.sdst.kind == OperandKind.VCC:
            self.reg_file.vcc = vcc_bit
        elif inst.sdst.kind == OperandKind.SGPR:
            self.reg_file.write_sgpr(inst.sdst.index, vcc_bit)

        return d_u32, vcc_bit


class TestDivScaleF32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_interpreter_normal_execution(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        # v1 = 2.0, v2 = 10.0
        rf.write_vgpr_f32(1, 2.0)
        rf.write_vgpr_f32(2, 10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 0)
        self.assertEqual(rf.vcc, 0)
        self.assertEqual(rf.read_vgpr_f32(0), 2.0)

    def test_interpreter_scale_up_small_denom(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        small = 2.0 ** -100
        rf.write_vgpr_f32(1, small)
        rf.write_vgpr_f32(2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertEqual(rf.vcc, 1)
        self.assertAlmostEqual(rf.read_vgpr_f32(0), small * SCALE64_POS)

    def test_interpreter_source_modifiers(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        rf.write_vgpr_f32(1, -2.5)
        rf.write_vgpr_f32(2, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.sdst_sgpr(10),
            src0=ASTOperand.vgpr(1, abs_=True),
            src1=ASTOperand.vgpr(1, abs_=True),
            src2=ASTOperand.vgpr(2),
        )

        d_res, vcc_res = interp.step(inst)
        self.assertEqual(rf.read_vgpr_f32(0), 2.5)
        self.assertEqual(rf.read_sgpr(10), 0)

    def test_interpreter_clamp_and_omod(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        rf.write_vgpr_f32(1, 0.4)
        rf.write_vgpr_f32(2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
            omod=OMod.MUL2,
            clamp=True,
        )

        interp.step(inst)
        # 0.4 * 2.0 = 0.8 <= 1.0
        self.assertAlmostEqual(rf.read_vgpr_f32(0), 0.8)


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

    def set_lane_vgpr_u32(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK32

    def get_lane_vgpr_u32(self, lane: int, idx: int) -> int:
        return self.vgprs[lane][idx]

    def set_lane_vgpr_f32(self, lane: int, idx: int, val: float) -> None:
        self.vgprs[lane][idx] = f32_to_u32(val)

    def get_lane_vgpr_f32(self, lane: int, idx: int) -> float:
        return u32_to_f32(self.vgprs[lane][idx])

    @property
    def vcc(self) -> int:
        if self.wave_size == 32:
            return self.sgprs[106]
        return self.sgprs[106] | (self.sgprs[107] << 32)

    @vcc.setter
    def vcc(self, val: int) -> None:
        if self.wave_size == 32:
            self.sgprs[106] = val & MASK32
        else:
            self.sgprs[106] = val & MASK32
            self.sgprs[107] = (val >> 32) & MASK32


class SIMDInterpreter:
    """Multi-lane SIMD wavefront interpreter with EXEC masking."""

    def __init__(self, wave: SIMDWaveState) -> None:
        self.wave = wave

    def resolve_lane_source_u32(self, lane: int, src: ASTOperand) -> int:
        if src.kind == OperandKind.VGPR:
            val_u32 = self.wave.get_lane_vgpr_u32(lane, src.index)
        elif src.kind == OperandKind.SGPR:
            val_u32 = self.wave.sgprs[src.index]
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val_u32 = f32_to_u32(float(src.value)) if src.value is not None else 0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        return apply_src_mods_u32(val_u32, neg=src.neg, abs_=src.abs_)

    def step(self, inst: ASTInstruction) -> int:
        inst.validate()
        sdst_mask = 0

        for lane in range(self.wave.wave_size):
            is_active = bool((self.wave.exec_mask >> lane) & 1)
            if not is_active:
                # Inactive lanes: destination is not modified, and sdst bit is 0
                continue

            u0 = self.resolve_lane_source_u32(lane, inst.src0)
            u1 = self.resolve_lane_source_u32(lane, inst.src1)
            u2 = self.resolve_lane_source_u32(lane, inst.src2)

            d_u32, vcc_bit = ref_div_scale_f32_u32(
                u0,
                u1,
                u2,
                clamp=inst.clamp,
                omod=inst.omod.value,
            )

            self.wave.set_lane_vgpr_u32(lane, inst.dst.index, d_u32)
            if vcc_bit:
                sdst_mask |= (1 << lane)

        if inst.sdst.kind == OperandKind.VCC:
            self.wave.vcc = sdst_mask
        elif inst.sdst.kind == OperandKind.SGPR:
            self.wave.sgprs[inst.sdst.index] = sdst_mask & MASK32

        return sdst_mask


class TestDivScaleF32SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_all_active(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)

        small = 2.0 ** -100
        for l in range(32):
            wave.set_lane_vgpr_f32(l, 1, small)
            wave.set_lane_vgpr_f32(l, 2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        vcc_mask = interp.step(inst)
        self.assertEqual(vcc_mask, MASK32)
        self.assertEqual(wave.vcc, MASK32)

        for l in range(32):
            self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), small * SCALE64_POS)

    def test_simd_wave64_divergent_exec_mask(self):
        wave = SIMDWaveState(wave_size=64)
        interp = SIMDInterpreter(wave)

        alt_mask = 0x5555555555555555
        wave.exec_mask = alt_mask

        small = 2.0 ** -100
        sentinel = 0xDEADBEEF

        for l in range(64):
            wave.set_lane_vgpr_u32(l, 0, sentinel)
            wave.set_lane_vgpr_f32(l, 1, small)
            wave.set_lane_vgpr_f32(l, 2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        vcc_mask = interp.step(inst)
        self.assertEqual(vcc_mask, alt_mask)
        self.assertEqual(wave.vcc, alt_mask)

        for l in range(64):
            if l % 2 == 0:
                self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), small * SCALE64_POS)
            else:
                self.assertEqual(wave.get_lane_vgpr_u32(l, 0), sentinel)

    def test_simd_divergent_operands_across_lanes(self):
        wave = SIMDWaveState(wave_size=32)
        interp = SIMDInterpreter(wave)

        small = 2.0 ** -100
        normal = 2.0

        for l in range(32):
            if l < 16:
                # small denominator -> scale up, vcc = 1
                wave.set_lane_vgpr_f32(l, 1, small)
                wave.set_lane_vgpr_f32(l, 2, 1.0)
            else:
                # normal denominator -> unscaled, vcc = 0
                wave.set_lane_vgpr_f32(l, 1, normal)
                wave.set_lane_vgpr_f32(l, 2, 10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        vcc_mask = interp.step(inst)
        self.assertEqual(vcc_mask, 0x0000FFFF)
        self.assertEqual(wave.vcc, 0x0000FFFF)

        for l in range(16):
            self.assertAlmostEqual(wave.get_lane_vgpr_f32(l, 0), small * SCALE64_POS)
        for l in range(16, 32):
            self.assertEqual(wave.get_lane_vgpr_f32(l, 0), normal)

    def test_exec_lo_hi_properties(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0x12345678
        wave.exec_hi = 0x9ABCDEF0
        self.assertEqual(wave.exec_mask, 0x9ABCDEF012345678)
        self.assertEqual(wave.exec_lo, 0x12345678)
        self.assertEqual(wave.exec_hi, 0x9ABCDEF0)


class TestDivScaleF32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_alias_src0_in_place(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        small = 2.0 ** -100
        # v0 = small (s0), v1 = small (s1), v2 = 1.0 (s2)
        rf.write_vgpr_f32(0, small)
        rf.write_vgpr_f32(1, small)
        rf.write_vgpr_f32(2, 1.0)

        # In-place scaling of v0
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertAlmostEqual(rf.read_vgpr_f32(0), small * SCALE64_POS)

        # src1 and src2 must remain strictly preserved
        self.assertEqual(rf.read_vgpr_f32(1), small)
        self.assertEqual(rf.read_vgpr_f32(2), 1.0)

    def test_dst_alias_src1_denominator(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        small = 2.0 ** -100
        rf.write_vgpr_f32(0, small)  # dst and src1
        rf.write_vgpr_f32(1, small)  # src0
        rf.write_vgpr_f32(2, 1.0)    # src2

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(2),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertAlmostEqual(rf.read_vgpr_f32(0), small * SCALE64_POS)
        self.assertEqual(rf.read_vgpr_f32(1), small)
        self.assertEqual(rf.read_vgpr_f32(2), 1.0)

    def test_dst_alias_src2_numerator(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        small = 2.0 ** -100
        rf.write_vgpr_f32(0, 1.0)    # dst and src2
        rf.write_vgpr_f32(1, small)  # src0
        rf.write_vgpr_f32(2, small)  # src1

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(0),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 1)
        self.assertAlmostEqual(rf.read_vgpr_f32(0), small * SCALE64_POS)
        self.assertEqual(rf.read_vgpr_f32(1), small)
        self.assertEqual(rf.read_vgpr_f32(2), small)

    def test_all_operands_fully_aliased(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        normal = 2.0
        rf.write_vgpr_f32(0, normal)

        # v_div_scale_f32 v0, vcc, v0, v0, v0
        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        d_res, vcc_res = interp.step(inst)
        self.assertEqual(vcc_res, 0)
        self.assertEqual(rf.read_vgpr_f32(0), normal)

    def test_unrelated_registers_strictly_preserved(self):
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        sentinel = 0xDEADBEEF
        for i in range(16):
            rf.write_vgpr(i, sentinel)

        rf.write_vgpr_f32(1, 2.0)
        rf.write_vgpr_f32(2, 2.0)
        rf.write_vgpr_f32(3, 10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        interp.step(inst)

        self.assertEqual(rf.read_vgpr_f32(0), 2.0)
        for i in range(4, 16):
            self.assertEqual(rf.read_vgpr(i), sentinel)


class StatusFlags:
    """Architectural condition codes and status flags for single-precision float operations."""

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
    def from_f32_bits(cls, bits: int, vcc_bit: int = 0) -> StatusFlags:
        sign = bool(bits & F32_NEG_ZERO)
        exp = (bits >> 23) & 0xFF
        mant = bits & F32_SUB_MAX

        is_zero = False
        is_negative = sign
        is_nan = False
        is_inf = False
        is_subnormal = False

        if exp == 0xFF:
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


class TestDivScaleF32StatusFlagsAndConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow (VCC/SCC) isolation."""

    def test_status_flags_classification(self):
        # Zero
        flags_zero = StatusFlags.from_f32_bits(F32_POS_ZERO)
        self.assertTrue(flags_zero.is_zero)
        self.assertFalse(flags_zero.is_negative)
        self.assertFalse(flags_zero.is_nan)

        # Negative Zero
        flags_neg_zero = StatusFlags.from_f32_bits(F32_NEG_ZERO)
        self.assertTrue(flags_neg_zero.is_zero)
        self.assertTrue(flags_neg_zero.is_negative)

        # Infinity
        flags_inf = StatusFlags.from_f32_bits(F32_POS_INF)
        self.assertTrue(flags_inf.is_inf)
        self.assertFalse(flags_inf.is_nan)

        # NaN
        flags_nan = StatusFlags.from_f32_bits(F32_CANONICAL_QNAN)
        self.assertTrue(flags_nan.is_nan)
        self.assertFalse(flags_nan.is_inf)

        # Subnormal
        flags_sub = StatusFlags.from_f32_bits(F32_SUB_MIN)
        self.assertTrue(flags_sub.is_subnormal)
        self.assertFalse(flags_sub.is_zero)

    def test_vcc_condition_flag_generation(self):
        small = 2.0 ** -100

        # Scaled case -> vcc = 1
        d_scaled, vcc_s = ref_div_scale_f32(small, small, 1.0)
        flags_s = StatusFlags.from_f32_bits(f32_to_u32(d_scaled), vcc_s)
        self.assertEqual(vcc_s, 1)
        self.assertTrue(flags_s.is_scaled)

        # Unscaled case -> vcc = 0
        d_unscaled, vcc_u = ref_div_scale_f32(2.0, 2.0, 10.0)
        flags_u = StatusFlags.from_f32_bits(f32_to_u32(d_unscaled), vcc_u)
        self.assertEqual(vcc_u, 0)
        self.assertFalse(flags_u.is_scaled)

    def test_scc_isolation_and_preservation(self):
        # VALU instructions do not modify SCC (Scalar Condition Code)
        rf = ScalarRegisterFile()
        interp = DivScaleF32ScalarInterpreter(rf)

        # Arbitrary simulated SCC stored in SGPR 105
        scc_sentinel = 1
        rf.write_sgpr(105, scc_sentinel)

        small = 2.0 ** -100
        rf.write_vgpr_f32(1, small)
        rf.write_vgpr_f32(2, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        interp.step(inst)

        # VCC is modified, but SCC (SGPR 105) is preserved
        self.assertEqual(rf.vcc, 1)
        self.assertEqual(rf.read_sgpr(105), scc_sentinel)


class TestDivScaleF32IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, infinities, and NaNs."""

    def test_subnormal_scale_up(self):
        # Subnormal float in denominator -> scale up by 2^64
        sub_min = u32_to_f32(F32_SUB_MIN)
        res_u32, vcc = ref_div_scale_f32_u32(F32_SUB_MIN, F32_SUB_MIN, F32_ONE)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(u32_to_f32(res_u32), sub_min * SCALE64_POS)

    def test_threshold_boundaries_exact(self):
        # Exactly at THRESH_F32_SMALL -> scales up
        res_at, vcc_at = ref_div_scale_f32(THRESH_F32_SMALL, THRESH_F32_SMALL, 1.0)
        self.assertEqual(vcc_at, 1)
        self.assertAlmostEqual(res_at, THRESH_F32_SMALL * SCALE64_POS)

        # Above THRESH_F32_SMALL -> does not scale
        above_small = THRESH_F32_SMALL * 2.0
        res_above, vcc_above = ref_div_scale_f32(above_small, above_small, 1.0)
        self.assertEqual(vcc_above, 0)
        self.assertEqual(res_above, above_small)

        # Exactly at THRESH_F32_LARGE -> scales down
        res_large, vcc_large = ref_div_scale_f32(THRESH_F32_LARGE, THRESH_F32_LARGE, 1.0)
        self.assertEqual(vcc_large, 1)
        self.assertAlmostEqual(res_large, THRESH_F32_LARGE * SCALE64_NEG)

        # Below THRESH_F32_LARGE -> does not scale
        below_large = THRESH_F32_LARGE * 0.5
        res_below, vcc_below = ref_div_scale_f32(below_large, below_large, 1.0)
        self.assertEqual(vcc_below, 0)
        self.assertEqual(res_below, below_large)

    def test_signed_zeros(self):
        # +0.0 and -0.0
        res_p, vcc_p = ref_div_scale_f32_u32(F32_POS_ZERO, F32_POS_ZERO, F32_ONE)
        self.assertEqual(res_p, F32_POS_ZERO)
        self.assertEqual(vcc_p, 0)

        res_n, vcc_n = ref_div_scale_f32_u32(F32_NEG_ZERO, F32_NEG_ZERO, F32_ONE)
        self.assertEqual(res_n, F32_NEG_ZERO)
        self.assertEqual(vcc_n, 0)

    def test_signed_infinities(self):
        # Infinities
        res_pinf, vcc_p = ref_div_scale_f32_u32(F32_POS_INF, F32_POS_INF, F32_ONE)
        self.assertEqual(res_pinf, F32_POS_INF)
        self.assertEqual(vcc_p, 0)

        res_ninf, vcc_n = ref_div_scale_f32_u32(F32_NEG_INF, F32_NEG_INF, F32_ONE)
        self.assertEqual(res_ninf, F32_NEG_INF)
        self.assertEqual(vcc_n, 0)

    def test_nan_quietening_and_payload_preservation(self):
        # QNaN payload preserved
        qnan_payload = F32_CANONICAL_QNAN | 0x1234
        res_qnan, vcc_q = ref_div_scale_f32_u32(qnan_payload, F32_ONE, F32_TWO)
        self.assertEqual(res_qnan, qnan_payload)
        self.assertEqual(vcc_q, 0)

        # SNaN quieted (bit 22 set)
        res_snan, vcc_s = ref_div_scale_f32_u32(F32_SNAN, F32_ONE, F32_TWO)
        self.assertTrue(is_f32_qnan(res_snan))
        self.assertFalse(is_f32_snan(res_snan))
        self.assertEqual(vcc_s, 0)

    def test_clamp_boundary_extremes(self):
        # Negative -> +0.0
        res_neg, _ = ref_div_scale_f32_u32(F32_NEG_ONE, F32_ONE, F32_ONE, clamp=True)
        self.assertEqual(res_neg, F32_POS_ZERO)

        # Greater than 1.0 -> 1.0
        res_gt, _ = ref_div_scale_f32_u32(F32_TWO, F32_ONE, F32_ONE, clamp=True)
        self.assertEqual(res_gt, F32_ONE)

        # NaN -> +0.0
        res_nan, _ = ref_div_scale_f32_u32(F32_CANONICAL_QNAN, F32_ONE, F32_ONE, clamp=True)
        self.assertEqual(res_nan, F32_POS_ZERO)


def mutant_omit_scale(s0: float, s1: float, s2: float, **kwargs) -> tuple[float, int]:
    """Mutant 1: Omits scaling logic entirely."""
    _, vcc = ref_div_scale_f32(s0, s1, s2, **kwargs)
    return s0, vcc


def mutant_invert_scale_direction(s0: float, s1: float, s2: float, **kwargs) -> tuple[float, int]:
    """Mutant 2: Uses scale down (2^-64) when it should scale up (2^64)."""
    res, vcc = ref_div_scale_f32(s0, s1, s2, **kwargs)
    if vcc == 1 and abs(s1) <= THRESH_F32_SMALL:
        return s0 * SCALE64_NEG, 1
    return res, vcc


def mutant_omit_vcc_flag(s0: float, s1: float, s2: float, **kwargs) -> tuple[float, int]:
    """Mutant 3: Computes scaled result but fails to set VCC flag."""
    res, _ = ref_div_scale_f32(s0, s1, s2, **kwargs)
    return res, 0


def mutant_clamp_nan_to_nan(u0: int, u1: int, u2: int, *, clamp: bool = False, **kwargs) -> tuple[int, int]:
    """Mutant 4: Clamps numbers but fails to map NaN to +0.0."""
    res_u32, vcc = ref_div_scale_f32_u32(u0, u1, u2, clamp=False, **kwargs)
    if clamp and not is_f32_nan(res_u32):
        val = u32_to_f32(res_u32)
        if val < 0.0:
            return F32_POS_ZERO, vcc
        elif val > 1.0:
            return F32_ONE, vcc
    return res_u32, vcc


def mutant_snan_not_quieted(u0: int, u1: int, u2: int, **kwargs) -> tuple[int, int]:
    """Mutant 5: Returns SNaN directly without quieting."""
    if is_f32_snan(u0):
        return u0, 0
    return ref_div_scale_f32_u32(u0, u1, u2, **kwargs)


class TestDivScaleF32NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, and mutation detection."""

    def test_syntax_rejection_bad_opcodes(self):
        for bad_op in ("v_div_scale_f16", "v_add_f32", "s_div_scale_f32", "foo_bar"):
            with self.assertRaises(ValueError):
                parse_div_scale_asm(f"{bad_op} v0, vcc, v1, v1, v2")

    def test_syntax_rejection_operand_count(self):
        # Too few operands
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f32 v0, vcc, v1, v1")
        # Too many operands
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f32 v0, vcc, v1, v1, v2, v3")

    def test_syntax_rejection_destination_modifiers(self):
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f32 -v0, vcc, v1, v1, v2")
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f32 |v0|, vcc, v1, v1, v2")

    def test_syntax_rejection_destination_sgpr(self):
        with self.assertRaises(TypeError):
            parse_div_scale_asm("v_div_scale_f32 s0, vcc, v1, v1, v2")

    def test_syntax_rejection_register_out_of_bounds(self):
        with self.assertRaises(ValueError):
            parse_div_scale_asm("v_div_scale_f32 v256, vcc, v1, v1, v2")

    def test_mutation_detection_all_mutants_caught(self):
        small = 2.0 ** -100

        # Mutant 1: omit scale
        good_res, good_vcc = ref_div_scale_f32(small, small, 1.0)
        bad_res1, bad_vcc1 = mutant_omit_scale(small, small, 1.0)
        self.assertNotEqual(good_res, bad_res1)
        self.assertEqual(bad_res1, small)

        # Mutant 2: invert scale direction
        bad_res2, _ = mutant_invert_scale_direction(small, small, 1.0)
        self.assertNotEqual(good_res, bad_res2)

        # Mutant 3: omit VCC flag
        _, bad_vcc3 = mutant_omit_vcc_flag(small, small, 1.0)
        self.assertEqual(good_vcc, 1)
        self.assertEqual(bad_vcc3, 0)
        self.assertNotEqual(good_vcc, bad_vcc3)

        # Mutant 4: clamp NaN to NaN
        good_clamp, _ = ref_div_scale_f32_u32(F32_CANONICAL_QNAN, F32_ONE, F32_TWO, clamp=True)
        bad_clamp, _ = mutant_clamp_nan_to_nan(F32_CANONICAL_QNAN, F32_ONE, F32_TWO, clamp=True)
        self.assertEqual(good_clamp, F32_POS_ZERO)
        self.assertNotEqual(good_clamp, bad_clamp)

        # Mutant 5: SNaN not quieted
        good_snan, _ = ref_div_scale_f32_u32(F32_SNAN, F32_ONE, F32_TWO)
        bad_snan, _ = mutant_snan_not_quieted(F32_SNAN, F32_ONE, F32_TWO)
        self.assertTrue(is_f32_qnan(good_snan))
        self.assertTrue(is_f32_snan(bad_snan))
        self.assertNotEqual(good_snan, bad_snan)

    def test_mutation_detection_simd_exec_mask_leak(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0  # All lanes inactive
        sentinel = 0xCAFEBABE
        for l in range(32):
            wave.set_lane_vgpr_u32(l, 0, sentinel)

        inst = ASTInstruction(
            opcode=Opcode.V_DIV_SCALE_F32,
            dst=ASTOperand.vgpr(0),
            sdst=ASTOperand.vcc(),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.vgpr(2),
        )

        interp = SIMDInterpreter(wave)
        interp.step(inst)

        # Inactive lanes must be strictly preserved
        for l in range(32):
            self.assertEqual(wave.get_lane_vgpr_u32(l, 0), sentinel)


if __name__ == "__main__":
    unittest.main()
