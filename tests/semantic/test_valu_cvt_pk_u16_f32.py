from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_u16_f32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the packed float-to-unsigned-integer conversion instruction:
  v_cvt_pk_u16_f32 dst, src0, src1

RDNA2 VALU Instruction Specification:
  v_cvt_pk_u16_f32 D.u32, S0.f32, S1.f32
  Operands:
    - D:  32-bit VGPR destination (holding two packed unsigned 16-bit integers)
    - S0: 32-bit float source (VGPR/SGPR/literal/inline constant), providing low 16 bits
    - S1: 32-bit float source (VGPR/SGPR/literal/inline constant), providing high 16 bits

  Mathematical Operation:
    D[15:0]  = f32_to_u16(S0)
    D[31:16] = f32_to_u16(S1)

    where f32_to_u16(f) truncates single-precision IEEE 754 float toward zero (RTZ)
    and saturates to the unsigned 16-bit integer range [0, 65535] ([0, 0xFFFF]):
      - If f is NaN: returns 0 (0x0000) per DirectX / AMD RDNA2 specification.
      - If f <= 0.0f (or -infinity, negative numbers, ±0.0): returns 0 (0x0000).
      - If f >= 65535.0f (or +infinity): saturates to 65535 (0xFFFF).
      - Otherwise: returns int(f) & 0xFFFF (truncation toward zero).

    Packed Destination Dword:
      D = (f32_to_u16(S0) & 0xFFFF) | ((f32_to_u16(S1) & 0xFFFF) << 16)

  Key Architectural Properties:
    1. Independent saturation: S0 and S1 are converted and saturated independently.
    2. Truncation toward zero (RTZ): Fractional values round toward 0 (+1.9 -> 1, +0.9 -> 0).
    3. Unsigned saturation: Negative values clamp to 0; positive overflow clamps to 0xFFFF.
    4. Low/High Packing: S0 occupies D[15:0], S1 occupies D[31:16].
    5. Destination is always a VGPR: The full 32-bit destination register is overwritten.

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

UINT16_MAX: int = 65535
UINT16_MIN: int = 0

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


def f32_to_u16(bits: int, *, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Convert float32 bit pattern to saturated unsigned 16-bit integer [0, 65535]."""
    b = bits & MASK32
    if abs_mod:
        b &= ~SIGN_BIT32
    if neg_mod:
        b ^= SIGN_BIT32

    # NaN returns 0 per AMD RDNA2 / DirectX specification
    if is_nan_f32(b):
        return 0

    val = bits_to_f32(b)

    if math.isinf(val):
        return UINT16_MAX if val > 0 else 0

    # Negative numbers and negative zero clamp to 0
    if val <= 0.0:
        return 0

    # Positive saturation
    if val >= 65535.0:
        return UINT16_MAX

    # Truncation toward zero
    return int(val) & MASK16


def pack_u16_to_u32(lo_u16: int, hi_u16: int) -> int:
    """Pack two 16-bit integers into a 32-bit unsigned dword (lo in [15:0], hi in [31:16])."""
    return ((hi_u16 & MASK16) << 16) | (lo_u16 & MASK16)


def unpack_u16_from_u32(val32: int) -> tuple[int, int]:
    """Unpack a 32-bit unsigned dword into low and high 16-bit words (lo, hi)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def ref_cvt_pk_u16_f32(
    src0: int,
    src1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_u16_f32.

    Converts two 32-bit single-precision floats into saturated 16-bit unsigned integers and
    packs them into a single 32-bit dword:
      D[15:0]  = f32_to_u16(S0)
      D[31:16] = f32_to_u16(S1)
    """
    lo = f32_to_u16(src0, abs_mod=abs0, neg_mod=neg0)
    hi = f32_to_u16(src1, abs_mod=abs1, neg_mod=neg1)
    return pack_u16_to_u32(lo, hi)


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
    V_CVT_PK_U16_F32 = "v_cvt_pk_u16_f32"
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
        if self.opcode == Opcode.V_CVT_PK_U16_F32:
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
class F32ToU16Expr(ASTExpr):
    """Symbolic float-to-unsigned-16-bit integer conversion operator."""
    child: ASTExpr
    bitwidth: int = 16


@dataclasses.dataclass(frozen=True)
class PackU16Expr(ASTExpr):
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
    if isinstance(node, F32ToU16Expr):
        val = eval_ast(node.child, env)
        return f32_to_u16(val)
    if isinstance(node, PackU16Expr):
        lo = eval_ast(node.lo, env)
        hi = eval_ast(node.hi, env)
        return pack_u16_to_u32(lo, hi)
    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


def instruction_to_ast(inst: Instruction) -> PackU16Expr:
    """Translate a v_cvt_pk_u16_f32 instruction into its canonical typed symbolic AST."""
    if inst.opcode != Opcode.V_CVT_PK_U16_F32:
        raise ValueError(f"Cannot translate non-pack opcode to PackU16Expr: {inst.opcode}")
    assert inst.src1 is not None

    def operand_to_ast(op: Operand) -> ASTExpr:
        if isinstance(op, Register):
            return VarExpr(name=op.name, bitwidth=32)
        elif isinstance(op, Immediate):
            return ConstExpr(value=op.value, bitwidth=32)
        raise TypeError(f"Unsupported operand: {type(op).__name__}")

    s0_expr = operand_to_ast(inst.src0)
    s1_expr = operand_to_ast(inst.src1)
    return PackU16Expr(lo=F32ToU16Expr(s0_expr), hi=F32ToU16Expr(s1_expr))


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCodes:
    """Architectural condition codes for 32-bit packed VALU operations."""
    zero: bool
    sign: bool
    lo_zero: bool
    hi_zero: bool
    lo_saturated: bool
    hi_saturated: bool
    parity: bool


@dataclasses.dataclass(frozen=True)
class VALUStatusFlags:
    """Status flags and carry/borrow tracking for VALU instructions."""
    vcc_modified: bool = False
    overflow: bool = False
    carry: bool = False


def compute_cvt_pk_condition_codes(result: int, src0: int, src1: int) -> ConditionCodes:
    """Computes condition codes evaluating attributes of the 32-bit packed result."""
    res = result & MASK32
    lo = res & 0xFFFF
    hi = (res >> 16) & 0xFFFF
    b0 = res & 0xFF
    parity = (bin(b0).count("1") % 2) == 0

    f0 = bits_to_f32(src0) if not is_nan_f32(src0) else 0.0
    f1 = bits_to_f32(src1) if not is_nan_f32(src1) else 0.0
    lo_sat = (f0 >= 65535.0) or (f0 <= 0.0)
    hi_sat = (f1 >= 65535.0) or (f1 <= 0.0)

    return ConditionCodes(
        zero=(res == 0),
        sign=bool(res & SIGN_BIT32),
        lo_zero=(lo == 0),
        hi_zero=(hi == 0),
        lo_saturated=lo_sat,
        hi_saturated=hi_sat,
        parity=parity,
    )


def compute_cvt_pk_status_flags() -> VALUStatusFlags:
    """In RDNA2 ISA, VALU v_cvt_pk_u16_f32 never modifies VCC or sets carry/borrow."""
    return VALUStatusFlags(vcc_modified=False, overflow=False, carry=False)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar architectural interpreter for RDNA2 VALU operations."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.last_condition_codes: ConditionCodes | None = None
        self.last_status_flags: VALUStatusFlags | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds [0, 255]: {idx}")
            return self.vgprs[idx]
        elif kind == RegKind.SGPR:
            if not (0 <= idx < 106):
                raise IndexError(f"SGPR index out of bounds [0, 105]: {idx}")
            return self.sgprs[idx]
        raise TypeError(f"Unknown register kind: {kind}")

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        val_norm = u32(val)
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds [0, 255]: {idx}")
            self.vgprs[idx] = val_norm
        elif kind == RegKind.SGPR:
            if not (0 <= idx < 106):
                raise IndexError(f"SGPR index out of bounds [0, 105]: {idx}")
            self.sgprs[idx] = val_norm
        else:
            raise TypeError(f"Unknown register kind: {kind}")

    def read_operand(self, op: Operand) -> int:
        if isinstance(op, Register):
            return self.read_reg_u32(op.kind, op.index)
        elif isinstance(op, Immediate):
            return u32(op.value)
        raise TypeError(f"Unsupported operand type: {type(op).__name__}")

    def step(self, inst: Instruction) -> int:
        if inst.opcode == Opcode.S_MOV_B32:
            val = self.read_operand(inst.src0)
            self.write_reg_u32(inst.dst.kind, inst.dst.index, val)
            self.trace.append(f"{inst.to_assembly()} => {val:#010x}")
            return val
        elif inst.opcode == Opcode.V_MOV_B32:
            val = self.read_operand(inst.src0)
            self.write_reg_u32(inst.dst.kind, inst.dst.index, val)
            self.trace.append(f"{inst.to_assembly()} => {val:#010x}")
            return val
        elif inst.opcode == Opcode.V_CVT_PK_U16_F32:
            assert inst.src1 is not None
            s0 = self.read_operand(inst.src0)
            s1 = self.read_operand(inst.src1)
            res = ref_cvt_pk_u16_f32(s0, s1)
            self.write_reg_u32(inst.dst.kind, inst.dst.index, res)
            self.last_condition_codes = compute_cvt_pk_condition_codes(res, s0, s1)
            self.last_status_flags = compute_cvt_pk_status_flags()
            self.trace.append(f"{inst.to_assembly()} => {res:#010x}")
            return res
        else:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {inst.opcode}")

    def run(self, prog: Program) -> None:
        for inst in prog.instructions:
            self.step(inst)


# ---------------------------------------------------------------------------
# SIMD Multi-Lane Divergence and Wavefront Interpreter
# ---------------------------------------------------------------------------

class SIMDWavefrontInterpreter:
    """Architectural SIMD wavefront interpreter supporting Wave32 and Wave64 execution modes."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wavefront size: {wave_size}. Must be 32 or 64.")
        self.wave_size: int = wave_size
        self.lanes_vgpr: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0
        self.exec_mask: int = (1 << wave_size) - 1

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self.exec_mask = mask & limit

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index out of bounds [0, {self.wave_size - 1}]: {lane}")
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index out of bounds [0, {self.wave_size - 1}]: {lane}")
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds [0, 255]: {idx}")
            return self.lanes_vgpr[lane][idx]
        elif kind == RegKind.SGPR:
            if not (0 <= idx < 106):
                raise IndexError(f"SGPR index out of bounds [0, 105]: {idx}")
            return self.sgprs[idx]
        raise TypeError(f"Unknown register kind: {kind}")

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index out of bounds [0, {self.wave_size - 1}]: {lane}")
        val_norm = u32(val)
        if kind == RegKind.VGPR:
            if not (0 <= idx < 256):
                raise IndexError(f"VGPR index out of bounds [0, 255]: {idx}")
            self.lanes_vgpr[lane][idx] = val_norm
        elif kind == RegKind.SGPR:
            if not (0 <= idx < 106):
                raise IndexError(f"SGPR index out of bounds [0, 105]: {idx}")
            self.sgprs[idx] = val_norm
        else:
            raise TypeError(f"Unknown register kind: {kind}")

    def read_lane_operand(self, lane: int, op: Operand) -> int:
        if isinstance(op, Register):
            return self.read_lane_reg_u32(lane, op.kind, op.index)
        elif isinstance(op, Immediate):
            return u32(op.value)
        raise TypeError(f"Unsupported operand type: {type(op).__name__}")

    def step(self, inst: Instruction) -> None:
        if inst.opcode == Opcode.S_MOV_B32:
            val = self.read_lane_operand(0, inst.src0)
            if inst.dst.kind != RegKind.SGPR:
                raise TypeError("S_MOV_B32 destination must be SGPR")
            self.sgprs[inst.dst.index] = val
            return

        if inst.opcode in (Opcode.V_MOV_B32, Opcode.V_CVT_PK_U16_F32):
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    if inst.opcode == Opcode.V_MOV_B32:
                        res = self.read_lane_operand(lane, inst.src0)
                    else:
                        assert inst.src1 is not None
                        s0 = self.read_lane_operand(lane, inst.src0)
                        s1 = self.read_lane_operand(lane, inst.src1)
                        res = ref_cvt_pk_u16_f32(s0, s1)
                    self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, res)
            return

        raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")


class TestValuCvtPkU16F32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_in_range_values_truncation(self) -> None:
        """Verify integer conversions and truncation toward zero."""
        test_cases = [
            (f32_to_bits(0.0), f32_to_bits(0.0), 0x00000000),
            (f32_to_bits(1.0), f32_to_bits(0.0), 0x00000001),
            (f32_to_bits(0.0), f32_to_bits(1.0), 0x00010000),
            (f32_to_bits(10.0), f32_to_bits(20.0), 0x0014000A),
            (f32_to_bits(1.9), f32_to_bits(2.9), 0x00020001),   # RTZ: 1.9 -> 1, 2.9 -> 2
            (f32_to_bits(0.999), f32_to_bits(0.5), 0x00000000), # RTZ to 0
            (f32_to_bits(65534.0), f32_to_bits(65534.9), 0xFFFEFFFE),
            (f32_to_bits(65535.0), f32_to_bits(65535.0), 0xFFFFFFFF),
        ]
        for s0, s1, expected in test_cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                res = ref_cvt_pk_u16_f32(s0, s1)
                self.assertEqual(res, expected)

    def test_negative_values_saturate_to_zero(self) -> None:
        """Negative numbers and negative zero saturate strictly to 0."""
        negative_cases = [
            f32_to_bits(-0.0),
            f32_to_bits(-1.0),
            f32_to_bits(-10.0),
            f32_to_bits(-1000.0),
            f32_to_bits(-1e6),
            NEG_INF_F32,
        ]
        for val in negative_cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(f32_to_u16(val), 0)
                self.assertEqual(ref_cvt_pk_u16_f32(val, val), 0x00000000)

    def test_positive_overflow_saturation(self) -> None:
        """Values >= 65535.0f and +infinity saturate to 65535 (0xFFFF)."""
        overflow_values = [
            f32_to_bits(65535.0),
            f32_to_bits(65536.0),
            f32_to_bits(100000.0),
            f32_to_bits(1e8),
            POS_INF_F32,
        ]
        for val in overflow_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_pk_u16_f32(val, 0), 0x0000FFFF)
                self.assertEqual(ref_cvt_pk_u16_f32(0, val), 0xFFFF0000)
                self.assertEqual(ref_cvt_pk_u16_f32(val, val), 0xFFFFFFFF)

    def test_nan_returns_zero(self) -> None:
        """NaN values produce 0 (0x0000)."""
        for nan_val in [QNAN_F32, SNAN_F32, 0x7FC01234, 0xFFC00000, 0xFF800001]:
            with self.subTest(nan=hex(nan_val)):
                self.assertEqual(f32_to_u16(nan_val), 0)
                self.assertEqual(ref_cvt_pk_u16_f32(nan_val, nan_val), 0x00000000)

    def test_input_modifiers(self) -> None:
        """abs clears sign bit and neg inverts sign bit."""
        # abs(-10.0f) -> 10 -> 0x000A
        neg_ten = f32_to_bits(-10.0)
        self.assertEqual(ref_cvt_pk_u16_f32(neg_ten, 0, abs0=True), 0x0000000A)

        # neg(10.0f) -> -10 -> clamps to 0
        pos_ten = f32_to_bits(10.0)
        self.assertEqual(ref_cvt_pk_u16_f32(0, pos_ten, neg1=True), 0x00000000)


class TestValuCvtPkU16F32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_assembly_formatting(self) -> None:
        inst1 = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst1.to_assembly(), "v_cvt_pk_u16_f32 v0, v1, v2")

        inst2 = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 5),
            src0=Register(RegKind.SGPR, 2),
            src1=Immediate(f32_to_bits(1.0)),
        )
        self.assertEqual(inst2.to_assembly(), "v_cvt_pk_u16_f32 v5, s2, 0x3f800000")

    def test_program_emission(self) -> None:
        prog = Program()
        prog.emit(Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 0), Immediate(POS_ONE_F32)))
        prog.emit(Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 1), Immediate(f32_to_bits(256.0))))
        prog.emit(Instruction(
            Opcode.V_CVT_PK_U16_F32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.SGPR, 0),
        ))
        expected = (
            "s_mov_b32 s0, 0x3f800000\n"
            "v_mov_b32 v1, 0x43800000\n"
            "v_cvt_pk_u16_f32 v0, v1, s0"
        )
        self.assertEqual(prog.to_assembly(), expected)

    def test_symbolic_ast_evaluation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        ast_tree = instruction_to_ast(inst)
        self.assertIsInstance(ast_tree, PackU16Expr)
        self.assertIsInstance(ast_tree.lo, F32ToU16Expr)
        self.assertIsInstance(ast_tree.hi, F32ToU16Expr)

        # Evaluate with test environment
        env = {"v1": f32_to_bits(100.5), "v2": f32_to_bits(200.5)}
        res = eval_ast(ast_tree, env)
        expected = ref_cvt_pk_u16_f32(f32_to_bits(100.5), f32_to_bits(200.5))
        self.assertEqual(res, expected)
        self.assertEqual(res, (200 << 16) | 100)

    def test_ast_immutability(self) -> None:
        reg = Register(RegKind.VGPR, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            reg.index = 1  # type: ignore[misc]

        imm = Immediate(42)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            imm.value = 43  # type: ignore[misc]


class TestValuCvtPkU16F32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def test_register_bounds_and_masking(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.VGPR, 256)
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.SGPR, 106)

        interp.write_reg_u32(RegKind.VGPR, 0, 0x1_FFFF_FFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0xFFFFFFFF)

    def test_step_cvt_pk_u16_f32_vgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, f32_to_bits(1000.0))
        interp.write_reg_u32(RegKind.VGPR, 2, f32_to_bits(2000.0))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        expected = (2000 << 16) | 1000
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)

    def test_step_cvt_pk_u16_f32_sgpr_and_inline_operands(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.SGPR, 2, f32_to_bits(70000.0))  # > 65535 -> 0xFFFF

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 2),
            src1=Immediate(f32_to_bits(-100.0)),  # negative -> 0x0000
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x0000FFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x0000FFFF)

    def test_run_complete_program(self) -> None:
        interp = ScalarVALUInterpreter()
        prog = Program()
        prog.emit(Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 0), Immediate(POS_ONE_F32)))
        prog.emit(Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 1), Immediate(f32_to_bits(500.0))))
        prog.emit(Instruction(
            Opcode.V_CVT_PK_U16_F32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 1),
            Register(RegKind.SGPR, 0),
        ))
        interp.run(prog)
        # S0 = 500.0f -> 500, S1 = 1.0f -> 1
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 2), (1 << 16) | 500)
        self.assertEqual(len(interp.trace), 3)

    def test_scalar_interpreter_determinism(self) -> None:
        runs = []
        for _ in range(5):
            interp = ScalarVALUInterpreter()
            interp.write_reg_u32(RegKind.VGPR, 1, f32_to_bits(123.45))
            interp.write_reg_u32(RegKind.VGPR, 2, f32_to_bits(678.9))
            inst = Instruction(
                opcode=Opcode.V_CVT_PK_U16_F32,
                dst=Register(RegKind.VGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
            )
            interp.step(inst)
            runs.append((interp.vgprs[:], interp.sgprs[:]))
        for r in runs[1:]:
            self.assertEqual(r, runs[0])


class TestValuCvtPkU16F32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active_execution(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0xFFFFFFFF)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, f32_to_bits(float(lane * 100)))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, f32_to_bits(float(lane * 200)))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        for lane in range(32):
            s0 = f32_to_bits(float(lane * 100))
            s1 = f32_to_bits(float(lane * 200))
            expected = ref_cvt_pk_u16_f32(s0, s1)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)

    def test_wave32_all_inactive_preserves_canary(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x00000000)
        canary = 0xCAFEBABE

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, POS_ONE_F32)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, POS_ONE_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        for lane in range(32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_wave32_checkerboard_mask_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x55555555)  # Even lanes active
        canary = 0xDEADBEEF

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, f32_to_bits(80000.0))  # > 65535 -> 0xFFFF
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, f32_to_bits(-10.0))    # negative -> 0x0000

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        expected_active = 0x0000FFFF
        for lane in range(32):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane % 2 == 0:
                self.assertEqual(actual, expected_active)
            else:
                self.assertEqual(actual, canary)

    def test_wave64_high_half_execution(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.set_exec_mask(0xFFFFFFFF00000000)  # Lanes 32..63 active
        canary = 0xBAADF00D

        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, f32_to_bits(float(lane)))
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, f32_to_bits(float(lane * 2)))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        for lane in range(64):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane < 32:
                self.assertEqual(actual, canary)
            else:
                expected = ref_cvt_pk_u16_f32(f32_to_bits(float(lane)), f32_to_bits(float(lane * 2)))
                self.assertEqual(actual, expected)

    def test_wavefront_scalar_broadcast_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x0000000F)  # First 4 lanes active

        simd.write_lane_reg_u32(0, RegKind.SGPR, 5, f32_to_bits(1.0))

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x11111111)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, f32_to_bits(float(lane)))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.SGPR, 5),
        )
        simd.step(inst)

        for lane in range(32):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane < 4:
                expected = ref_cvt_pk_u16_f32(f32_to_bits(float(lane)), f32_to_bits(1.0))
                self.assertEqual(actual, expected)
            else:
                self.assertEqual(actual, 0x11111111)


class TestValuCvtPkU16F32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self) -> None:
        """v_cvt_pk_u16_f32 v0, v0, v1: D aliases with S0."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, f32_to_bits(1234.0))
        interp.write_reg_u32(RegKind.VGPR, 1, f32_to_bits(5678.0))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        res = interp.step(inst)
        expected = (5678 << 16) | 1234
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), f32_to_bits(5678.0))

    def test_alias_dst_equals_src1(self) -> None:
        """v_cvt_pk_u16_f32 v0, v1, v0: D aliases with S1."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, f32_to_bits(50.0))
        interp.write_reg_u32(RegKind.VGPR, 0, f32_to_bits(100.0))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        expected = (100 << 16) | 50
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)

    def test_alias_dst_equals_src0_and_src1(self) -> None:
        """v_cvt_pk_u16_f32 v0, v0, v0: Full self-aliasing."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, f32_to_bits(70000.0))  # > 65535 -> sat to 0xFFFF

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0xFFFFFFFF)

    def test_duplicate_sources_distinct_destination(self) -> None:
        """v_cvt_pk_u16_f32 v2, v0, v0: Duplicate sources."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, f32_to_bits(250.0))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        expected = (250 << 16) | 250
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 2), expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), f32_to_bits(250.0))

    def test_adjacent_register_preservation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary_prev = 0xAAAA5555
        canary_next = 0x5555AAAA
        interp.write_reg_u32(RegKind.VGPR, 4, canary_prev)
        interp.write_reg_u32(RegKind.VGPR, 6, canary_next)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 5),
            src0=Immediate(f32_to_bits(1.0)),
            src1=Immediate(f32_to_bits(2.0)),
        )
        interp.step(inst)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 4), canary_prev)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 6), canary_next)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 5), 0x00020001)

    def test_simd_multi_lane_self_aliasing(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x0000FFFF)  # Low 16 lanes active

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, f32_to_bits(float(lane * 10)))

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        simd.step(inst)

        for lane in range(32):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane < 16:
                expected = ref_cvt_pk_u16_f32(f32_to_bits(float(lane * 10)), f32_to_bits(float(lane * 10)))
                self.assertEqual(actual, expected)
            else:
                self.assertEqual(actual, f32_to_bits(float(lane * 10)))


class TestValuCvtPkU16F32ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation isolation."""

    def test_condition_codes_zero_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(POS_ZERO_F32),
            src1=Immediate(NEG_ZERO_F32),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.sign)
        self.assertTrue(cc.lo_zero)
        self.assertTrue(cc.hi_zero)
        self.assertTrue(cc.lo_saturated)  # 0.0 clamps/saturates to min
        self.assertTrue(cc.hi_saturated)
        self.assertTrue(cc.parity)

    def test_condition_codes_sign_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        # S0 = 1.0f -> 0x0001, S1 = 40000.0f -> 0x9C40 (bit 31 is 1 since 0x9C400001 has bit 31 set)
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(POS_ONE_F32),
            src1=Immediate(f32_to_bits(40000.0)),
        )
        res = interp.step(inst)
        self.assertEqual(res, (40000 << 16) | 1)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertTrue(cc.sign)
        self.assertFalse(cc.lo_zero)
        self.assertFalse(cc.hi_zero)

    def test_condition_codes_saturation_detection(self) -> None:
        interp = ScalarVALUInterpreter()
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(f32_to_bits(70000.0)),
            src1=Immediate(f32_to_bits(-10.0)),
        )
        interp.step(inst)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.lo_saturated)
        self.assertTrue(cc.hi_saturated)

    def test_condition_codes_parity(self) -> None:
        cc_odd = compute_cvt_pk_condition_codes(0x0001, POS_ONE_F32, POS_ZERO_F32)
        self.assertFalse(cc_odd.parity)

        cc_even = compute_cvt_pk_condition_codes(0x0003, POS_ONE_F32, POS_ZERO_F32)
        self.assertTrue(cc_even.parity)

    def test_vcc_and_carry_borrow_isolation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xFEEDFACECAFEBEEF
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_U16_F32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(POS_INF_F32),
            src1=Immediate(NEG_INF_F32),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary_vcc)

        flags = interp.last_status_flags
        self.assertIsNotNone(flags)
        assert flags is not None
        self.assertFalse(flags.vcc_modified)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.carry)

        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary_vcc
        simd.step(inst)
        self.assertEqual(simd.vcc, canary_vcc)


class TestValuCvtPkU16F32IEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros_distinction(self) -> None:
        """Both +0.0 and -0.0 saturate to integer 0."""
        self.assertEqual(f32_to_u16(POS_ZERO_F32), 0)
        self.assertEqual(f32_to_u16(NEG_ZERO_F32), 0)
        self.assertEqual(ref_cvt_pk_u16_f32(POS_ZERO_F32, NEG_ZERO_F32), 0x00000000)

    def test_subnormals_truncate_to_zero(self) -> None:
        """Subnormal floats have magnitude < 1.0, so they strictly truncate toward zero."""
        subnormals = [
            0x00000001,  # min positive subnormal
            0x007FFFFF,  # max positive subnormal
            0x80000001,  # min negative subnormal
            0x807FFFFF,  # max negative subnormal
            0x00000042,
            0x00012345,
        ]
        for sub in subnormals:
            with self.subTest(sub=hex(sub)):
                self.assertEqual(f32_to_u16(sub), 0)
                self.assertEqual(ref_cvt_pk_u16_f32(sub, sub), 0x00000000)

    def test_nan_variants_produce_zero(self) -> None:
        """Quiet and signaling NaNs of all signs and payloads produce zero."""
        nans = [
            QNAN_F32,
            SNAN_F32,
            0x7FC00001,
            0x7F800002,
            0x7FC01234,
            0x7F804321,
            0xFFC00000,
            0xFF800001,
            0xFFC01234,
            0xFF804321,
        ]
        for nan_bits in nans:
            with self.subTest(nan=hex(nan_bits)):
                self.assertTrue(is_nan_f32(nan_bits))
                self.assertEqual(f32_to_u16(nan_bits), 0)
                self.assertEqual(ref_cvt_pk_u16_f32(nan_bits, nan_bits), 0x00000000)

    def test_infinities_and_large_floats(self) -> None:
        """Infinities and large floats saturate to unsigned 16-bit boundaries."""
        self.assertEqual(f32_to_u16(POS_INF_F32), 0xFFFF)
        self.assertEqual(f32_to_u16(NEG_INF_F32), 0x0000)
        self.assertEqual(ref_cvt_pk_u16_f32(POS_INF_F32, NEG_INF_F32), 0x0000FFFF)

        # 0x7F7FFFFF (FLT_MAX)
        self.assertEqual(f32_to_u16(0x7F7FFFFF), 0xFFFF)
        # 0xFF7FFFFF (-FLT_MAX)
        self.assertEqual(f32_to_u16(0xFF7FFFFF), 0x0000)

    def test_fractional_truncation_around_zero(self) -> None:
        """Verify strict round-toward-zero across fractional values."""
        cases = [
            (0.9999999, 0),
            (-0.9999999, 0),
            (1.0, 1),
            (1.99999, 1),
            (2.5, 2),
            (65534.99, 65534),
            (65535.99, 65535),
        ]
        for val, expected in cases:
            bits = f32_to_bits(val)
            self.assertEqual(f32_to_u16(bits), expected)


if __name__ == "__main__":
    unittest.main()
