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


if __name__ == "__main__":
    unittest.main()
