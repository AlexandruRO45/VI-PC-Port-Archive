from __future__ import annotations

"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_div_fixup_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector general-purpose registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point numbers (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operands
  v[s0:s0+1], v[s1:s1+1], and v[s2:s2+1] (or scalar SGPR pairs s[k:k+1], literal constants).

  Instruction:
    v_div_fixup_f64 v[d:d+1], src0, src1, src2 [clamp] [omod]

  Operands:
    src0 = provisional quotient from previous reciprocal/FMAS step
    src1 = denominator
    src2 = numerator

  IEEE-754 binary64 encoding:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

import abc
from dataclasses import dataclass
import enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_INDEFINITE_QNAN = 0xFFF8000000000000  # Default indefinite NaN returned by AMD GPU for 0/0 and inf/inf
F64_QUIET_BIT = 0x0008000000000000

F64_EXP_MASK = 0x7FF0000000000000
F64_MANT_MASK = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK = 0x8000000000000000

F64_ONE = 0x3FF0000000000000            # 1.0
F64_NEG_ONE = 0xBFF0000000000000        # -1.0
F64_TWO = 0x4000000000000000            # 2.0
F64_NEG_TWO = 0xC000000000000000        # -2.0
F64_HALF = 0x3FE0000000000000           # 0.5
F64_FOUR = 0x4010000000000000           # 4.0
F64_QUARTER = 0x3FD0000000000000        # 0.25
F64_MIN_NORMAL = 0x0010000000000000     # 2^-1022
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF     # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL = 0x0000000000000001    # 2^-1074
F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_NEG_INF if val < 0 else F64_POS_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_f64_nan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) != 0)


def is_f64_snan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a signaling NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) == 0)


def is_f64_qnan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a quiet NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) != 0)


def is_f64_zero(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +0.0 or -0.0."""
    return (bits & (F64_EXP_MASK | F64_MANT_MASK)) == 0


def is_f64_inf(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +inf or -inf."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) == 0)


def is_f64_subnormal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a subnormal/denormal float."""
    return ((bits & F64_EXP_MASK) == 0) and ((bits & F64_MANT_MASK) != 0)


def is_f64_normal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a normalized finite float."""
    exp = (bits >> 52) & 0x7FF
    return 0 < exp < 0x7FF


def f64_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 63) & 1


def f64_unbiased_exponent(bits: int) -> int:
    """Extract true mathematical binary exponent for finite non-zero value."""
    val = abs(u64_to_f64(bits))
    if val == 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    _, e = math.frexp(val)
    return e - 1


def quiet_f64(bits: int) -> int:
    """Force an IEEE-754 double NaN to quiet by setting the quiet bit."""
    return (bits & MASK64) | F64_QUIET_BIT


def apply_src_mods_f64(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK64
    if abs_val:
        res &= ~F64_SIGN_MASK
    if neg:
        res ^= F64_SIGN_MASK
    return res


def apply_clamp_f64(bits: int) -> int:
    """Apply VOP3 clamp modifier to double precision float.

    Saturates result to [+0.0, 1.0].
    NaN is converted to +0.0.
    -0.0 is clamped to +0.0.
    Values < 0.0 are clamped to +0.0.
    Values > 1.0 are clamped to 1.0.
    """
    if is_f64_nan(bits) or bits == F64_NEG_ZERO:
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_POS_ZERO
    elif val > 1.0:
        return F64_ONE
    return bits


def ref_div_fixup_f64(
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
    omod: float = 1.0,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_div_fixup_f64 instruction.

    Args:
        s0_bits: Provisional quotient (64-bit uint)
        s1_bits: Denominator (64-bit uint)
        s2_bits: Numerator (64-bit uint)
        src0_neg / src0_abs: Modifiers for src0
        src1_neg / src1_abs: Modifiers for src1
        src2_neg / src2_abs: Modifiers for src2
        clamp: If True, saturates to [0.0, 1.0] (NaN -> +0.0)
        omod: Output modifier (1.0, 2.0, 4.0, 0.5)

    Returns:
        Fixed-up quotient as 64-bit uint.
    """
    s0 = apply_src_mods_f64(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f64(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f64(s2_bits, neg=src2_neg, abs_val=src2_abs)

    sign1 = f64_sign(s1)
    sign2 = f64_sign(s2)
    sign_out = sign1 ^ sign2

    # Decision tree according to AMD RDNA2 ISA specification
    if is_f64_nan(s2):
        res = quiet_f64(s2)
    elif is_f64_nan(s1):
        res = quiet_f64(s1)
    elif is_f64_zero(s1) and is_f64_zero(s2):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = F64_INDEFINITE_QNAN
    elif is_f64_inf(s1) and is_f64_inf(s2):
        # inf / inf -> Indefinite QNaN
        res = F64_INDEFINITE_QNAN
    elif is_f64_zero(s1) or is_f64_inf(s2):
        # x / 0 or inf / y -> signed infinity
        res = F64_NEG_INF if sign_out else F64_POS_INF
    elif is_f64_inf(s1) or is_f64_zero(s2):
        # x / inf or 0 / y -> signed zero
        res = F64_NEG_ZERO if sign_out else F64_POS_ZERO
    elif is_f64_nan(s0):
        # Intermediate quotient overflowed to NaN -> signed infinity
        res = F64_NEG_INF if sign_out else F64_POS_INF
    elif (f64_unbiased_exponent(s2) - f64_unbiased_exponent(s1)) < -1075:
        # Exponent underflow threshold (-1075 for binary64 subnormal limit)
        res = F64_NEG_ZERO if sign_out else F64_POS_ZERO
    else:
        # Standard case: sign_out applied to absolute magnitude of s0
        res = (s0 & ~F64_SIGN_MASK) | (F64_SIGN_MASK if sign_out else 0)

    # Output modifiers
    if omod != 1.0 and not is_f64_nan(res) and not is_f64_inf(res) and not is_f64_zero(res):
        fval = u64_to_f64(res) * omod
        res = f64_to_u64(fval)

    if clamp:
        res = apply_clamp_f64(res)

    return res & MASK64


def ref_div_fixup_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    s1_lo: int,
    s1_hi: int,
    s2_lo: int,
    s2_hi: int,
    **kwargs,
) -> tuple[int, int]:
    """Dword-level wrapper for ref_div_fixup_f64 operating on 32-bit register halves."""
    s0 = join_dwords(s0_lo, s0_hi)
    s1 = join_dwords(s1_lo, s1_hi)
    s2 = join_dwords(s2_lo, s2_hi)
    res = ref_div_fixup_f64(s0, s1, s2, **kwargs)
    return split_dwords(res)


class VALURegisterFile:
    """Architectural register model for RDNA2 VALU VGPR file.

    Models 32-bit physical VGPRs with 64-bit aligned pair access.
    """

    def __init__(self, num_vgprs: int = 256) -> None:
        if not (1 <= num_vgprs <= 256):
            raise ValueError(f"num_vgprs must be in range [1, 256], got {num_vgprs}")
        self.num_vgprs = num_vgprs
        self.vgprs = [0] * num_vgprs

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = val & MASK32

    def get_vgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo = self.get_vgpr(idx)
        hi = self.get_vgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_vgpr(idx, lo)
        self.set_vgpr(idx + 1, hi)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(enum.Enum):
    VGPR_PAIR = enum.auto()
    SGPR_PAIR = enum.auto()
    LITERAL_64 = enum.auto()
    INLINE_IMM = enum.auto()


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    imm_val: int = 0
    neg: bool = False
    abs_val: bool = False

    def format_asm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.LITERAL_64:
            s = f"0x{self.imm_val:016x}"
        elif self.kind == OperandKind.INLINE_IMM:
            s = f"{self.imm_val}"
        else:
            s = "unknown"
        if self.abs_val:
            s = f"|{s}|"
        if self.neg:
            s = f"-{s}"
        return s


@dataclass
class ASTInstruction:
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand
    clamp: bool = False
    omod: float = 1.0

    def validate(self) -> None:
        if self.opcode != "v_div_fixup_f64":
            raise ValueError(f"Invalid opcode for v_div_fixup_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.kind}")
        if not (0 <= self.dst.reg_idx <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.reg_idx}")
        if self.dst.neg or self.dst.abs_val:
            raise ValueError("Destination operand cannot have neg or abs modifiers")

        for idx, (src_name, src_op) in enumerate([("src0", self.src0), ("src1", self.src1), ("src2", self.src2)]):
            if src_op.kind == OperandKind.VGPR_PAIR:
                if not (0 <= src_op.reg_idx <= 254):
                    raise ValueError(f"{src_name} VGPR pair index out of range: {src_op.reg_idx}")
            elif src_op.kind == OperandKind.SGPR_PAIR:
                if not (0 <= src_op.reg_idx <= 104):
                    raise ValueError(f"{src_name} SGPR pair index out of range: {src_op.reg_idx}")
            elif src_op.kind in (OperandKind.LITERAL_64, OperandKind.INLINE_IMM):
                pass
            else:
                raise ValueError(f"Unsupported operand kind for {src_name}: {src_op.kind}")

        if self.omod not in (1.0, 2.0, 4.0, 0.5):
            raise ValueError(f"Invalid omod value: {self.omod}")

    def format_asm(self) -> str:
        tokens = [
            f"{self.opcode} {self.dst.format_asm()}, {self.src0.format_asm()}, "
            f"{self.src1.format_asm()}, {self.src2.format_asm()}"
        ]
        if self.clamp:
            tokens.append("clamp")
        if self.omod == 2.0:
            tokens.append("mul:2")
        elif self.omod == 4.0:
            tokens.append("mul:4")
        elif self.omod == 0.5:
            tokens.append("div:2")
        return " ".join(tokens)


class F64Expression(abc.ABC):
    """Abstract base class for typed 64-bit floating-point expressions."""

    @abc.abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        pass


@dataclass
class F64Constant(F64Expression):
    bits: int

    def evaluate(self, env: dict[str, int]) -> int:
        return self.bits & MASK64


@dataclass
class F64Variable(F64Expression):
    name: str

    def evaluate(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in environment")
        return env[self.name] & MASK64


@dataclass
class F64UnaryOp(F64Expression):
    op: str  # "abs", "neg", "neg_abs"
    operand: F64Expression

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        if self.op == "abs":
            return apply_src_mods_f64(val, abs_val=True)
        elif self.op == "neg":
            return apply_src_mods_f64(val, neg=True)
        elif self.op == "neg_abs":
            return apply_src_mods_f64(val, neg=True, abs_val=True)
        raise ValueError(f"Unknown unary operator: {self.op}")


@dataclass
class F64DivFixup(F64Expression):
    s0: F64Expression
    s1: F64Expression
    s2: F64Expression
    clamp: bool = False
    omod: float = 1.0

    def evaluate(self, env: dict[str, int]) -> int:
        v0 = self.s0.evaluate(env)
        v1 = self.s1.evaluate(env)
        v2 = self.s2.evaluate(env)
        return ref_div_fixup_f64(v0, v1, v2, clamp=self.clamp, omod=self.omod)


def instruction_to_expression(inst: ASTInstruction) -> F64DivFixup:
    """Translate an ASTInstruction into a typed symbolic expression tree."""
    inst.validate()

    def build_operand_expr(op: ASTOperand, prefix: str) -> F64Expression:
        if op.kind in (OperandKind.VGPR_PAIR, OperandKind.SGPR_PAIR):
            expr: F64Expression = F64Variable(f"{prefix}_{op.reg_idx}")
        elif op.kind in (OperandKind.LITERAL_64, OperandKind.INLINE_IMM):
            expr = F64Constant(op.imm_val)
        else:
            raise ValueError(f"Unsupported operand kind: {op.kind}")

        if op.abs_val and op.neg:
            expr = F64UnaryOp("neg_abs", expr)
        elif op.abs_val:
            expr = F64UnaryOp("abs", expr)
        elif op.neg:
            expr = F64UnaryOp("neg", expr)
        return expr

    s0_expr = build_operand_expr(inst.src0, "src0")
    s1_expr = build_operand_expr(inst.src1, "src1")
    s2_expr = build_operand_expr(inst.src2, "src2")
    return F64DivFixup(s0=s0_expr, s1=s1_expr, s2=s2_expr, clamp=inst.clamp, omod=inst.omod)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

def parse_asm_operand(tok: str) -> ASTOperand:
    tok = tok.strip()
    neg = False
    abs_val = False
    if tok.startswith("-"):
        neg = True
        tok = tok[1:].strip()
    if tok.startswith("|") and tok.endswith("|"):
        abs_val = True
        tok = tok[1:-1].strip()

    if tok.startswith("v[") and tok.endswith("]"):
        inner = tok[2:-1]
        parts = inner.split(":")
        reg_idx = int(parts[0])
        return ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=reg_idx, neg=neg, abs_val=abs_val)
    elif tok.startswith("s[") and tok.endswith("]"):
        inner = tok[2:-1]
        parts = inner.split(":")
        reg_idx = int(parts[0])
        return ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=reg_idx, neg=neg, abs_val=abs_val)
    elif tok.startswith("0x") or tok.startswith("0X"):
        imm = int(tok, 16)
        return ASTOperand(kind=OperandKind.LITERAL_64, imm_val=imm, neg=neg, abs_val=abs_val)
    else:
        try:
            if "." in tok or "e" in tok.lower():
                imm = f64_to_u64(float(tok))
                return ASTOperand(kind=OperandKind.LITERAL_64, imm_val=imm, neg=neg, abs_val=abs_val)
            imm = int(tok)
            return ASTOperand(kind=OperandKind.INLINE_IMM, imm_val=imm, neg=neg, abs_val=abs_val)
        except ValueError:
            raise ValueError(f"Unrecognized assembly operand token: {tok}")


def parse_asm_line(line: str) -> ASTInstruction:
    clean = line.split(";")[0].split("//")[0].strip()
    tokens = [t for t in clean.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("Empty assembly line")
    opcode = tokens[0]
    if opcode != "v_div_fixup_f64":
        raise ValueError(f"Expected opcode v_div_fixup_f64, got {opcode}")

    clamp = False
    omod = 1.0
    raw_ops = []
    for tok in tokens[1:]:
        if tok == "clamp":
            clamp = True
        elif tok == "mul:2":
            omod = 2.0
        elif tok == "mul:4":
            omod = 4.0
        elif tok == "div:2":
            omod = 0.5
        else:
            raw_ops.append(tok)

    if len(raw_ops) != 4:
        raise ValueError(f"Expected 4 operands for v_div_fixup_f64, got {len(raw_ops)}: {raw_ops}")

    dst = parse_asm_operand(raw_ops[0])
    src0 = parse_asm_operand(raw_ops[1])
    src1 = parse_asm_operand(raw_ops[2])
    src2 = parse_asm_operand(raw_ops[3])

    inst = ASTInstruction(
        opcode=opcode,
        dst=dst,
        src0=src0,
        src1=src1,
        src2=src2,
        clamp=clamp,
        omod=omod,
    )
    inst.validate()
    return inst


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU v_div_fixup_f64."""

    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgpr = VALURegisterFile(num_vgprs=num_vgprs)
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.vcc = 0
        self.scc = 0

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32

    def get_sgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo = self.get_sgpr(idx)
        hi = self.get_sgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_sgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_sgpr(idx, lo)
        self.set_sgpr(idx + 1, hi)

    def resolve_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.vgpr.get_vgpr_pair(op.reg_idx)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = self.get_sgpr_pair(op.reg_idx)
        elif op.kind in (OperandKind.LITERAL_64, OperandKind.INLINE_IMM):
            val = op.imm_val & MASK64
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

        return apply_src_mods_f64(val, neg=op.neg, abs_val=op.abs_val)

    def execute_ast(self, inst: ASTInstruction) -> int:
        inst.validate()
        s0 = self.resolve_operand(inst.src0)
        s1 = self.resolve_operand(inst.src1)
        s2 = self.resolve_operand(inst.src2)

        res = ref_div_fixup_f64(s0, s1, s2, clamp=inst.clamp, omod=inst.omod)
        self.vgpr.set_vgpr_pair(inst.dst.reg_idx, res)
        return res

    def execute_asm(self, line: str) -> int:
        inst = parse_asm_line(line)
        return self.execute_ast(inst)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

def ref_simd_div_fixup_f64(
    src0_lanes: list[int],
    src1_lanes: list[int],
    src2_lanes: list[int],
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
    *,
    clamp: bool = False,
    omod: float = 1.0,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
) -> list[int]:
    """Reference SIMD execution oracle across wavefront lanes."""
    if wave_size not in (32, 64):
        raise ValueError(f"Unsupported wave size: {wave_size}")
    if len(src0_lanes) != wave_size or len(src1_lanes) != wave_size or len(src2_lanes) != wave_size:
        raise ValueError(f"Operand length must match wave_size ({wave_size})")

    active_mask = (1 << wave_size) - 1 if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    fallback = [0] * wave_size if prev_dst is None else prev_dst
    if len(fallback) != wave_size:
        raise ValueError(f"prev_dst length must match wave_size ({wave_size})")

    output = list(fallback)
    for lane in range(wave_size):
        if (active_mask >> lane) & 1:
            output[lane] = ref_div_fixup_f64(
                src0_lanes[lane],
                src1_lanes[lane],
                src2_lanes[lane],
                clamp=clamp,
                omod=omod,
                src0_neg=src0_neg,
                src0_abs=src0_abs,
                src1_neg=src1_neg,
                src1_abs=src1_abs,
                src2_neg=src2_neg,
                src2_abs=src2_abs,
            )
    return output


class SIMDWavefront:
    """SIMD wavefront simulator modeling multi-lane execution and EXEC masking."""

    def __init__(self, wave_size: int = 64, num_vgprs: int = 256) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.lanes = [VALURegisterFile(num_vgprs=num_vgprs) for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1
        self.vcc = 0
        self.sgprs = [0] * 106

    def store_vgpr_pair_wave(self, reg_idx: int, values: list[int]) -> None:
        if len(values) != self.wave_size:
            raise ValueError(f"Value count ({len(values)}) does not match wave_size ({self.wave_size})")
        for lane in range(self.wave_size):
            self.lanes[lane].set_vgpr_pair(reg_idx, values[lane])

    def load_vgpr_pair_wave(self, reg_idx: int) -> list[int]:
        return [self.lanes[lane].get_vgpr_pair(reg_idx) for lane in range(self.wave_size)]

    def step_v_div_fixup_f64(
        self,
        dst_reg: int,
        src0_reg: int,
        src1_reg: int,
        src2_reg: int,
        *,
        clamp: bool = False,
        omod: float = 1.0,
        src0_neg: bool = False,
        src0_abs: bool = False,
        src1_neg: bool = False,
        src1_abs: bool = False,
        src2_neg: bool = False,
        src2_abs: bool = False,
    ) -> list[int]:
        active_mask = self.exec_mask & ((1 << self.wave_size) - 1)
        for lane in range(self.wave_size):
            if (active_mask >> lane) & 1:
                s0 = self.lanes[lane].get_vgpr_pair(src0_reg)
                s1 = self.lanes[lane].get_vgpr_pair(src1_reg)
                s2 = self.lanes[lane].get_vgpr_pair(src2_reg)
                res = ref_div_fixup_f64(
                    s0,
                    s1,
                    s2,
                    clamp=clamp,
                    omod=omod,
                    src0_neg=src0_neg,
                    src0_abs=src0_abs,
                    src1_neg=src1_neg,
                    src1_abs=src1_abs,
                    src2_neg=src2_neg,
                    src2_abs=src2_abs,
                )
                self.lanes[lane].set_vgpr_pair(dst_reg, res)
        return self.load_vgpr_pair_wave(dst_reg)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        s0_vals = [f64_to_u64(float(i + 1) * 2.5) for i in range(32)]
        s1_vals = [f64_to_u64(2.0)] * 32
        s2_vals = [f64_to_u64(float(i + 1) * 5.0) for i in range(32)]

        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.store_vgpr_pair_wave(6, s2_vals)

        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)
        for i in range(32):
            self.assertEqual(u64_to_f64(res[i]), float(i + 1) * 2.5)

    def test_wave64_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        s0_vals = [f64_to_u64(float(i + 1)) for i in range(64)]
        s1_vals = [f64_to_u64(3.0)] * 64
        s2_vals = [f64_to_u64(float(i + 1) * 3.0) for i in range(64)]

        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.store_vgpr_pair_wave(6, s2_vals)

        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)
        for i in range(64):
            self.assertEqual(u64_to_f64(res[i]), float(i + 1))

    def test_wave32_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)

        s0_vals = [f64_to_u64(10.0)] * 32
        s1_vals = [f64_to_u64(2.0)] * 32
        s2_vals = [f64_to_u64(20.0)] * 32

        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.store_vgpr_pair_wave(6, s2_vals)

        # Alternating active lanes: 0b0101...
        sim.exec_mask = 0x55555555
        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(u64_to_f64(res[lane]), 10.0)
            else:
                self.assertEqual(res[lane], sentinel_pair)

    def test_wave64_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 64)

        s0_vals = [f64_to_u64(4.0)] * 64
        s1_vals = [f64_to_u64(-2.0)] * 64
        s2_vals = [f64_to_u64(8.0)] * 64

        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.store_vgpr_pair_wave(6, s2_vals)

        # Lower 32 lanes active, upper 32 inactive
        sim.exec_mask = 0x00000000FFFFFFFF
        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(u64_to_f64(res[lane]), -4.0)
            else:
                self.assertEqual(res[lane], sentinel_pair)

    def test_wave_mixed_special_cases_divergence(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)

        s0_vals = [f64_to_u64(1.0)] * 32
        s1_vals = [f64_to_u64(1.0)] * 32
        s2_vals = [f64_to_u64(1.0)] * 32

        # Lane 0: 0/0 -> Indefinite NaN
        s1_vals[0] = F64_POS_ZERO
        s2_vals[0] = F64_POS_ZERO

        # Lane 1: x / +0.0 -> +Inf
        s1_vals[1] = F64_POS_ZERO
        s2_vals[1] = F64_TWO

        # Lane 2: -x / +0.0 -> -Inf
        s1_vals[2] = F64_POS_ZERO
        s2_vals[2] = F64_NEG_TWO

        # Lane 3: x / Inf -> +0.0
        s1_vals[3] = F64_POS_INF
        s2_vals[3] = F64_TWO

        # Lane 4: 0 / x -> +0.0
        s1_vals[4] = F64_FOUR
        s2_vals[4] = F64_POS_ZERO

        # Lane 5: NaN numerator -> Quiet NaN
        s1_vals[5] = F64_FOUR
        s2_vals[5] = 0x7FF0000000000001

        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.store_vgpr_pair_wave(6, s2_vals)

        # Lanes 0..5 active, lanes 6..31 inactive
        sim.exec_mask = 0x0000003F
        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)

        self.assertEqual(res[0], F64_INDEFINITE_QNAN)
        self.assertEqual(res[1], F64_POS_INF)
        self.assertEqual(res[2], F64_NEG_INF)
        self.assertEqual(res[3], F64_POS_ZERO)
        self.assertEqual(res[4], F64_POS_ZERO)
        self.assertEqual(res[5], 0x7FF0000000000001 | F64_QUIET_BIT)

        # Inactive lanes preserved
        for lane in range(6, 32):
            self.assertEqual(res[lane], sentinel_pair)

    def test_exec_mask_zero(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)
        sim.exec_mask = 0

        res = sim.step_v_div_fixup_f64(0, 2, 4, 6)
        for lane in range(32):
            self.assertEqual(res[lane], sentinel_pair)


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_execute_vgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(5.0))   # s0 (quotient)
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(2.0))   # s1 (denom)
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(10.0))  # s2 (numer)

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4)
        src2 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=6)
        inst = ASTInstruction(opcode="v_div_fixup_f64", dst=dst, src0=src0, src1=src1, src2=src2)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 5.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_sgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(4.0))   # s0
        interp.set_sgpr_pair(4, f64_to_u64(-3.0))       # s1
        interp.set_sgpr_pair(6, f64_to_u64(12.0))       # s2

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=4)
        src2 = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=6)
        inst = ASTInstruction(opcode="v_div_fixup_f64", dst=dst, src0=src0, src1=src1, src2=src2)

        res = interp.execute_ast(inst)
        # 12 / -3 = -4.0
        self.assertEqual(u64_to_f64(res), -4.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_literal_source(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(2.5))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(2.0))

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4)
        src2 = ASTOperand(kind=OperandKind.LITERAL_64, imm_val=f64_to_u64(5.0))
        inst = ASTInstruction(opcode="v_div_fixup_f64", dst=dst, src0=src0, src1=src1, src2=src2)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 2.5)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_text_asm(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(7.0))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(-2.0))
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(-14.0))

        # Both negative (- / - -> +)
        res = interp.execute_asm("v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7]")
        self.assertEqual(u64_to_f64(res), 7.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_with_clamp_and_omod(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(0.25))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(4.0))
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(1.0))

        # omod mul:2 -> 0.25 * 2 = 0.5
        res = interp.execute_asm("v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] mul:2 clamp")
        self.assertEqual(u64_to_f64(res), 0.5)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4)
        src2 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=6)

        inst = ASTInstruction(opcode="v_div_fixup_f64", dst=dst, src0=src0, src1=src1, src2=src2)
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7]")

        # With modifiers and clamp
        src1_mod = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=8, neg=True, abs_val=True)
        inst_clamp = ASTInstruction(
            opcode="v_div_fixup_f64",
            dst=dst,
            src0=src0,
            src1=src1_mod,
            src2=src2,
            clamp=True,
            omod=2.0,
        )
        inst_clamp.validate()
        self.assertEqual(inst_clamp.format_asm(), "v_div_fixup_f64 v[0:1], v[2:3], -|s[8:9]|, v[6:7] clamp mul:2")

    def test_typed_expression_evaluation(self) -> None:
        c0 = F64Constant(f64_to_u64(5.0))
        c1 = F64Constant(f64_to_u64(2.0))
        c2 = F64Constant(f64_to_u64(10.0))

        expr = F64DivFixup(s0=c0, s1=c1, s2=c2)
        res = expr.evaluate({})
        self.assertEqual(u64_to_f64(res), 5.0)

        # Unary modifier on s1: negate denominator -> negative quotient
        neg_c1 = F64UnaryOp("neg", c1)
        expr_neg = F64DivFixup(s0=c0, s1=neg_c1, s2=c2)
        res_neg = expr_neg.evaluate({})
        self.assertEqual(u64_to_f64(res_neg), -5.0)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4, abs_val=True)
        src2 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=6, neg=True)

        inst = ASTInstruction(opcode="v_div_fixup_f64", dst=dst, src0=src0, src1=src1, src2=src2)
        expr = instruction_to_expression(inst)

        env = {
            "src0_2": f64_to_u64(3.0),
            "src1_4": f64_to_u64(-4.0),
            "src2_6": f64_to_u64(12.0),
        }
        res = expr.evaluate(env)
        # s1 = |-4.0| = 4.0; s2 = -12.0; s0 = 3.0 -> -12 / 4 = -3.0
        self.assertEqual(u64_to_f64(res), -3.0)

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        bad_op = ASTInstruction(
            opcode="v_div_scale_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
            src2=ASTOperand(OperandKind.VGPR_PAIR, 6),
        )
        with self.assertRaises(ValueError):
            bad_op.validate()

        # Destination cannot be SGPR
        bad_dst = ASTInstruction(
            opcode="v_div_fixup_f64",
            dst=ASTOperand(OperandKind.SGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
            src2=ASTOperand(OperandKind.VGPR_PAIR, 6),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination out of bounds
        oob_dst = ASTInstruction(
            opcode="v_div_fixup_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 256),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
            src2=ASTOperand(OperandKind.VGPR_PAIR, 6),
        )
        with self.assertRaises(ValueError):
            oob_dst.validate()

        # Invalid omod
        bad_omod = ASTInstruction(
            opcode="v_div_fixup_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
            src2=ASTOperand(OperandKind.VGPR_PAIR, 6),
            omod=3.0,
        )
        with self.assertRaises(ValueError):
            bad_omod.validate()


class TestVDivFixupF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_quotient_fixup(self) -> None:
        # Normal quotient: 10.0 / 2.0 = 5.0
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(10.0)
        s0 = f64_to_u64(5.0)

        # Both positive (+ / + -> +)
        r = ref_div_fixup_f64(s0, s1, s2)
        self.assertEqual(u64_to_f64(r), 5.0)

        # Numerator negative (- / + -> -)
        r = ref_div_fixup_f64(s0, s1, f64_to_u64(-10.0))
        self.assertEqual(u64_to_f64(r), -5.0)

        # Denominator negative (+ / - -> -)
        r = ref_div_fixup_f64(s0, f64_to_u64(-2.0), s2)
        self.assertEqual(u64_to_f64(r), -5.0)

        # Both negative (- / - -> +)
        r = ref_div_fixup_f64(s0, f64_to_u64(-2.0), f64_to_u64(-10.0))
        self.assertEqual(u64_to_f64(r), 5.0)

    def test_zero_divided_by_zero(self) -> None:
        # 0.0 / 0.0 -> Indefinite QNaN (0xFFF8_0000_0000_0000)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_POS_ZERO)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

    def test_inf_divided_by_inf(self) -> None:
        # Inf / Inf -> Indefinite QNaN (0xFFF8_0000_0000_0000)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_POS_INF)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

        r = ref_div_fixup_f64(F64_ONE, F64_NEG_INF, F64_POS_INF)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

    def test_division_by_zero(self) -> None:
        # x / +0.0 -> +Inf (for x > 0)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_ONE)
        self.assertEqual(r, F64_POS_INF)

        # -x / +0.0 -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_NEG_ONE)
        self.assertEqual(r, F64_NEG_INF)

        # x / -0.0 -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_ONE)
        self.assertEqual(r, F64_NEG_INF)

        # -x / -0.0 -> +Inf
        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_NEG_ONE)
        self.assertEqual(r, F64_POS_INF)

    def test_numerator_infinity(self) -> None:
        # +Inf / x -> +Inf
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_INF)
        self.assertEqual(r, F64_POS_INF)

        # -Inf / x -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_NEG_INF)
        self.assertEqual(r, F64_NEG_INF)

    def test_denominator_infinity(self) -> None:
        # x / +Inf -> +0.0
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_TWO)
        self.assertEqual(r, F64_POS_ZERO)

        # -x / +Inf -> -0.0
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_NEG_TWO)
        self.assertEqual(r, F64_NEG_ZERO)

    def test_zero_numerator(self) -> None:
        # +0.0 / x -> +0.0
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_ZERO)
        self.assertEqual(r, F64_POS_ZERO)

        # -0.0 / x -> -0.0
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_NEG_ZERO)
        self.assertEqual(r, F64_NEG_ZERO)

    def test_nan_propagation(self) -> None:
        # S2 (numerator) is NaN -> quieted S2 returned
        snan = 0x7FF0000000000001
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, snan)
        self.assertTrue(is_f64_qnan(r))
        self.assertEqual(r, snan | F64_QUIET_BIT)

        # S1 (denominator) is NaN -> quieted S1 returned
        r2 = ref_div_fixup_f64(F64_ONE, snan, F64_TWO)
        self.assertTrue(is_f64_qnan(r2))
        self.assertEqual(r2, snan | F64_QUIET_BIT)

    def test_s0_nan_recovery(self) -> None:
        # Intermediate S0 is NaN while S1 and S2 are finite non-zero -> recovers to signed Inf
        r = ref_div_fixup_f64(F64_CANONICAL_QNAN, F64_TWO, F64_FOUR)
        self.assertEqual(r, F64_POS_INF)

        r_neg = ref_div_fixup_f64(F64_CANONICAL_QNAN, F64_TWO, F64_NEG_ONE)
        self.assertEqual(r_neg, F64_NEG_INF)

    def test_exponent_underflow(self) -> None:
        # Exponent diff < -1075 -> underflow to signed zero
        # Numerator = 2^-1074 (min subnormal), Denominator = 2^10 (exp diff = -1084 < -1075)
        s2 = F64_MIN_SUBNORMAL
        s1 = f64_to_u64(1024.0)
        r = ref_div_fixup_f64(F64_ONE, s1, s2)
        self.assertEqual(r, F64_POS_ZERO)

        # Negative sign preserved
        r_neg = ref_div_fixup_f64(F64_ONE, s1, s2, src2_neg=True)
        self.assertEqual(r_neg, F64_NEG_ZERO)

    def test_input_modifiers(self) -> None:
        s1 = f64_to_u64(-2.0)
        s2 = f64_to_u64(-10.0)
        s0 = f64_to_u64(5.0)

        # abs on s1: -10 / 2 = -5.0
        r = ref_div_fixup_f64(s0, s1, s2, src1_abs=True)
        self.assertEqual(u64_to_f64(r), -5.0)

        # neg on s0: s0 sign is ignored in magnitude, sign_out dictates final sign
        r2 = ref_div_fixup_f64(s0, s1, s2, src0_neg=True)
        self.assertEqual(u64_to_f64(r2), 5.0)

    def test_clamp_modifier(self) -> None:
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(1.0)
        s0 = f64_to_u64(0.5)

        # 0.5 in [0.0, 1.0] -> 0.5
        r = ref_div_fixup_f64(s0, s1, s2, clamp=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # Negative clamped to +0.0
        r_neg = ref_div_fixup_f64(s0, s1, f64_to_u64(-1.0), clamp=True)
        self.assertEqual(r_neg, F64_POS_ZERO)

        # > 1.0 clamped to 1.0
        r_large = ref_div_fixup_f64(f64_to_u64(5.0), s1, f64_to_u64(10.0), clamp=True)
        self.assertEqual(r_large, F64_ONE)

        # NaN clamped to +0.0
        r_nan = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO, clamp=True)
        self.assertEqual(r_nan, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(5.0)
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(10.0)

        s0_lo, s0_hi = split_dwords(s0)
        s1_lo, s1_hi = split_dwords(s1)
        s2_lo, s2_hi = split_dwords(s2)

        r_lo, r_hi = ref_div_fixup_f64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, s2_lo, s2_hi)
        res = join_dwords(r_lo, r_hi)
        self.assertEqual(u64_to_f64(res), 5.0)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


if __name__ == "__main__":
    unittest.main()
