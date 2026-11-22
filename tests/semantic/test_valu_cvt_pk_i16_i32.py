from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_i16_i32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the packed integer format conversion instruction:
  v_cvt_pk_i16_i32 dst, src0, src1

RDNA2 VALU Instruction Specification:
  v_cvt_pk_i16_i32 D.u32, S0.i32, S1.i32
  Operands:
    - D:  32-bit VGPR destination (holding two packed signed 16-bit integers)
    - S0: 32-bit source (VGPR/SGPR/literal/inline constant), providing low 16 bits
    - S1: 32-bit source (VGPR/SGPR/literal/inline constant), providing high 16 bits

  Mathematical Operation:
    D[15:0]  = sat16(S0)
    D[31:16] = sat16(S1)

    where sat16(x) clamps a signed 32-bit integer x into the signed 16-bit range:
      [-32768, 32767] ([-0x8000, 0x7FFF]):
        if x > 32767:
            sat16(x) = 32767 (0x7FFF)
        elif x < -32768:
            sat16(x) = -32768 (0x8000)
        else:
            sat16(x) = x & 0xFFFF

    Packed Destination Dword:
      D = (sat16(S0) & 0xFFFF) | ((sat16(S1) & 0xFFFF) << 16)

  Key Architectural Properties:
    1. Independent saturation: S0 and S1 are converted and saturated independently.
    2. Signed saturation: Saturation treats inputs as signed 32-bit two's-complement
       values. Positive overflow saturates to 0x7FFF (+32767), while negative underflow
       saturates to 0x8000 (-32768).
    3. Low/High Packing: S0 occupies the lower 16-bit halfword D[15:0], while S1 occupies
       the upper 16-bit halfword D[31:16].
    4. Destination is always a VGPR: The full 32-bit destination register is overwritten.
    5. Modifiers: Integer clamp is not supported / rejected as per RDNA2 ISA contract.

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

INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF
INT16_MIN: int = -32768
INT16_MAX: int = 32767

INT16_MIN_U16: int = 0x8000
INT16_MAX_U16: int = 0x7FFF


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


def sat_i16(val: int) -> int:
    """Saturate a 32-bit integer to signed 16-bit range [-32768, 32767], returning 16-bit unsigned bits."""
    s = s32(val)
    if s > INT16_MAX:
        return INT16_MAX_U16
    if s < INT16_MIN:
        return INT16_MIN_U16
    return s & MASK16


def pack_i16_to_u32(lo_i16: int, hi_i16: int) -> int:
    """Pack two 16-bit integers into a 32-bit unsigned dword (lo in [15:0], hi in [31:16])."""
    return ((hi_i16 & MASK16) << 16) | (lo_i16 & MASK16)


def unpack_i16_from_u32(val32: int) -> tuple[int, int]:
    """Unpack a 32-bit unsigned dword into low and high 16-bit words (lo, hi)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def ref_cvt_pk_i16_i32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_i16_i32.

    Converts two 32-bit signed integers into saturated 16-bit signed integers and
    packs them into a single 32-bit dword:
      D[15:0]  = sat16(S0)
      D[31:16] = sat16(S1)
    """
    lo = sat_i16(src0)
    hi = sat_i16(src1)
    return pack_i16_to_u32(lo, hi)


def ref_cvt_pk_i16_i32_explicit(src0: int, src1: int) -> int:
    """Alternative architectural verification oracle with explicit boundary branching."""
    s0 = s32(src0)
    s1 = s32(src1)

    if s0 > 32767:
        lo = 32767
    elif s0 < -32768:
        lo = -32768
    else:
        lo = s0

    if s1 > 32767:
        hi = 32767
    elif s1 < -32768:
        hi = -32768
    else:
        hi = s1

    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


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
    V_CVT_PK_I16_I32 = "v_cvt_pk_i16_i32"
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
        if self.opcode == Opcode.V_CVT_PK_I16_I32:
            if self.dst.kind != RegKind.VGPR:
                raise TypeError(f"Destination for {self.opcode.value} must be a VGPR, got {self.dst.kind.name}")
            if self.src1 is None:
                raise ValueError(f"{self.opcode.value} requires two source operands")
            if "clamp" in self.modifiers:
                raise ValueError(f"{self.opcode.value} does not support clamp modifier")

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
class SatI16Expr(ASTExpr):
    """Symbolic signed 16-bit saturation operator."""
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
    if isinstance(node, SatI16Expr):
        val = eval_ast(node.child, env)
        return sat_i16(val)
    if isinstance(node, PackI16Expr):
        lo = eval_ast(node.lo, env)
        hi = eval_ast(node.hi, env)
        return pack_i16_to_u32(lo, hi)
    raise TypeError(f"Unknown AST node type: {type(node).__name__}")


def instruction_to_ast(inst: Instruction) -> PackI16Expr:
    """Translate a v_cvt_pk_i16_i32 instruction into its canonical typed symbolic AST."""
    if inst.opcode != Opcode.V_CVT_PK_I16_I32:
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
    return PackI16Expr(lo=SatI16Expr(s0_expr), hi=SatI16Expr(s1_expr))


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

    s0 = s32(src0)
    s1 = s32(src1)
    lo_sat = (s0 > INT16_MAX) or (s0 < INT16_MIN)
    hi_sat = (s1 > INT16_MAX) or (s1 < INT16_MIN)

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
    """In RDNA2 ISA, VALU v_cvt_pk_i16_i32 never modifies VCC or sets carry/borrow."""
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
        elif inst.opcode == Opcode.V_CVT_PK_I16_I32:
            assert inst.src1 is not None
            s0 = self.read_operand(inst.src0)
            s1 = self.read_operand(inst.src1)
            res = ref_cvt_pk_i16_i32(s0, s1)
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
            # Scalar move runs globally across SGPR file, independent of EXEC
            val = self.read_lane_operand(0, inst.src0)
            if inst.dst.kind != RegKind.SGPR:
                raise TypeError("S_MOV_B32 destination must be SGPR")
            self.sgprs[inst.dst.index] = val
            return

        if inst.opcode in (Opcode.V_MOV_B32, Opcode.V_CVT_PK_I16_I32):
            for lane in range(self.wave_size):
                if self.is_lane_active(lane):
                    if inst.opcode == Opcode.V_MOV_B32:
                        res = self.read_lane_operand(lane, inst.src0)
                    else:
                        assert inst.src1 is not None
                        s0 = self.read_lane_operand(lane, inst.src0)
                        s1 = self.read_lane_operand(lane, inst.src1)
                        res = ref_cvt_pk_i16_i32(s0, s1)
                    self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, res)
            return

        raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")


# ---------------------------------------------------------------------------
# Test Suites: Commit 1 (Oracle), Commit 2 (AST), Commit 3 (Interpreter)
# ---------------------------------------------------------------------------

class TestValuCvtPkI16I32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_in_range_values_pass_through(self) -> None:
        """Values strictly within [-32768, 32767] pass through without clamping."""
        test_cases = [
            (0, 0, 0x00000000),
            (1, 0, 0x00000001),
            (0, 1, 0x00010000),
            (42, 100, (100 << 16) | 42),
            (0x7FFF, 0x7FFF, 0x7FFF7FFF),
            (-1, -1, 0xFFFFFFFF),
            (-32768, -32768, 0x80008000),
            (-32768, 32767, 0x7FFF8000),
            (32767, -32768, 0x80007FFF),
            (-1, 0, 0x0000FFFF),
            (0, -1, 0xFFFF0000),
            (-2, -3, 0xFFFDFFFE),
        ]
        for s0, s1, expected in test_cases:
            with self.subTest(s0=s0, s1=s1):
                res = ref_cvt_pk_i16_i32(s0, s1)
                self.assertEqual(res, expected)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(s0, s1), expected)

    def test_upper_saturation_boundary(self) -> None:
        """Values exceeding 32767 saturate strictly to 32767 (0x7FFF)."""
        overflow_values = [
            32768,
            32769,
            40000,
            65535,
            65536,
            0x00010000,
            0x12345678,
            0x7FFFFFFF,
            INT32_MAX,
        ]
        for val in overflow_values:
            with self.subTest(val=val):
                self.assertEqual(ref_cvt_pk_i16_i32(val, 0), 0x00007FFF)
                self.assertEqual(ref_cvt_pk_i16_i32(0, val), 0x7FFF0000)
                self.assertEqual(ref_cvt_pk_i16_i32(val, val), 0x7FFF7FFF)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(val, val), 0x7FFF7FFF)

    def test_lower_saturation_boundary(self) -> None:
        """Values less than -32768 saturate strictly to -32768 (0x8000)."""
        underflow_values = [
            -32769,
            -32770,
            -40000,
            -65536,
            -100000,
            -2147483648,
            INT32_MIN,
            0x80000000,
        ]
        for val in underflow_values:
            with self.subTest(val=val):
                self.assertEqual(ref_cvt_pk_i16_i32(val, 0), 0x00008000)
                self.assertEqual(ref_cvt_pk_i16_i32(0, val), 0x80000000)
                self.assertEqual(ref_cvt_pk_i16_i32(val, val), 0x80008000)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(val, val), 0x80008000)

    def test_asymmetric_mixed_boundaries(self) -> None:
        """Verify cross-combination of upper saturation, lower saturation, and in-range."""
        self.assertEqual(ref_cvt_pk_i16_i32(0x7FFFFFFF, 0x80000000), 0x80007FFF)
        self.assertEqual(ref_cvt_pk_i16_i32(0x80000000, 0x7FFFFFFF), 0x7FFF8000)
        self.assertEqual(ref_cvt_pk_i16_i32(0x1234, 0x7FFFFFFF), 0x7FFF1234)
        self.assertEqual(ref_cvt_pk_i16_i32(-5, -50000), 0x8000FFFB)

    def test_channel_independence(self) -> None:
        """Mutations to S0 never affect high half D[31:16], and mutations to S1 never affect low half D[15:0]."""
        fixed_hi = 0x1234
        for s0_cand in [-100000, -32768, -1, 0, 1, 32767, 100000]:
            packed = ref_cvt_pk_i16_i32(s0_cand, fixed_hi)
            lo, hi = unpack_i16_from_u32(packed)
            self.assertEqual(hi, fixed_hi)
            self.assertEqual(lo, sat_i16(s0_cand))

        fixed_lo = 0x5678
        for s1_cand in [-100000, -32768, -1, 0, 1, 32767, 100000]:
            packed = ref_cvt_pk_i16_i32(fixed_lo, s1_cand)
            lo, hi = unpack_i16_from_u32(packed)
            self.assertEqual(lo, fixed_lo)
            self.assertEqual(hi, sat_i16(s1_cand))

    def test_oracle_equivalence_across_sweeps(self) -> None:
        """Exhaustively verify equivalence between ref_cvt_pk_i16_i32 and explicit branch model."""
        test_points = [
            -0x80000000, -0x7FFFFFFF, -65537, -65536, -32769, -32768, -32767,
            -1, 0, 1, 32766, 32767, 32768, 65535, 65536, 0x7FFFFFFE, 0x7FFFFFFF
        ]
        for a in test_points:
            for b in test_points:
                self.assertEqual(ref_cvt_pk_i16_i32(a, b), ref_cvt_pk_i16_i32_explicit(a, b))


class TestValuCvtPkI16I32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_assembly_formatting(self) -> None:
        inst1 = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst1.to_assembly(), "v_cvt_pk_i16_i32 v0, v1, v2")

        inst2 = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 5),
            src0=Register(RegKind.SGPR, 2),
            src1=Immediate(-1),
        )
        self.assertEqual(inst2.to_assembly(), "v_cvt_pk_i16_i32 v5, s2, -1")

        inst3 = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 10),
            src0=Immediate(0x12345),
            src1=Immediate(64),
        )
        self.assertEqual(inst3.to_assembly(), "v_cvt_pk_i16_i32 v10, 0x12345, 64")

    def test_program_emission(self) -> None:
        prog = Program()
        prog.emit(Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 0), Immediate(0x10000)))
        prog.emit(Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 1), Immediate(-50000)))
        prog.emit(Instruction(
            Opcode.V_CVT_PK_I16_I32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.SGPR, 0),
        ))
        expected = (
            "s_mov_b32 s0, 0x10000\n"
            "v_mov_b32 v1, 0xffff3cb0\n"
            "v_cvt_pk_i16_i32 v0, v1, s0"
        )
        self.assertEqual(prog.to_assembly(), expected)

    def test_symbolic_ast_evaluation(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        ast_tree = instruction_to_ast(inst)
        self.assertIsInstance(ast_tree, PackI16Expr)
        self.assertIsInstance(ast_tree.lo, SatI16Expr)
        self.assertIsInstance(ast_tree.hi, SatI16Expr)

        # Evaluate with test environment
        env = {"v1": 40000, "v2": -50000}
        res = eval_ast(ast_tree, env)
        expected = ref_cvt_pk_i16_i32(40000, -50000)
        self.assertEqual(res, expected)
        self.assertEqual(res, 0x80007FFF)

    def test_ast_immutability(self) -> None:
        reg = Register(RegKind.VGPR, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            reg.index = 1  # type: ignore[misc]

        imm = Immediate(42)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            imm.value = 43  # type: ignore[misc]


class TestValuCvtPkI16I32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def test_register_bounds_and_masking(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.VGPR, 256)
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.SGPR, 106)

        # Write masking to 32 bits
        interp.write_reg_u32(RegKind.VGPR, 0, 0x1_FFFF_FFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0xFFFFFFFF)

    def test_step_cvt_pk_i16_i32_vgpr_sources(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, 0x10000)   # 65536 -> sat to 0x7FFF
        interp.write_reg_u32(RegKind.VGPR, 2, -40000)    # -> sat to 0x8000
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x80007FFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x80007FFF)

    def test_step_cvt_pk_i16_i32_sgpr_and_inline_operands(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.SGPR, 2, 0x12345)  # > 32767 -> sat to 0x7FFF
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.SGPR, 2),
            src1=Immediate(-1),  # -1 in-range -> 0xFFFF
        )
        res = interp.step(inst)
        self.assertEqual(res, 0xFFFF7FFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0xFFFF7FFF)

    def test_run_complete_program(self) -> None:
        interp = ScalarVALUInterpreter()
        prog = Program()
        prog.emit(Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 10), Immediate(500)))
        prog.emit(Instruction(Opcode.V_MOV_B32, Register(RegKind.VGPR, 2), Immediate(-300)))
        prog.emit(Instruction(
            Opcode.V_CVT_PK_I16_I32,
            Register(RegKind.VGPR, 5),
            Register(RegKind.VGPR, 2),
            Register(RegKind.SGPR, 10),
        ))
        interp.run(prog)
        # S0 = -300 -> 0xFED4, S1 = 500 -> 0x01F4
        expected = (500 << 16) | ((-300) & 0xFFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 5), expected)
        self.assertEqual(len(interp.trace), 3)

    def test_scalar_interpreter_determinism(self) -> None:
        # Running the exact same sequence 5 times gives bit-identical results
        runs = []
        for _ in range(5):
            interp = ScalarVALUInterpreter()
            interp.write_reg_u32(RegKind.VGPR, 1, 0x7FFFFFFF)
            interp.write_reg_u32(RegKind.VGPR, 2, 0x80000000)
            inst = Instruction(
                opcode=Opcode.V_CVT_PK_I16_I32,
                dst=Register(RegKind.VGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
            )
            interp.step(inst)
            runs.append((interp.vgprs[:], interp.sgprs[:]))
        for r in runs[1:]:
            self.assertEqual(r, runs[0])


class TestValuCvtPkI16I32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active_execution(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0xFFFFFFFF)

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, (lane + 1) * 2000)  # S0
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, -(lane + 1) * 2000) # S1

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        for lane in range(32):
            s0 = (lane + 1) * 2000
            s1 = -(lane + 1) * 2000
            expected = ref_cvt_pk_i16_i32(s0, s1)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), expected)

    def test_wave32_all_inactive_preserves_canary(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x00000000)  # All lanes disabled
        canary = 0xCAFEBABE

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 200)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        for lane in range(32):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_wave32_checkerboard_mask_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x55555555)  # Even lanes active, odd lanes inactive
        canary = 0xDEADBEEF

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, 50000)  # overflows to 0x7FFF
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, -60000) # underflows to 0x8000

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        expected_active = 0x80007FFF
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
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, -lane)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
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
                expected = ref_cvt_pk_i16_i32(lane, -lane)
                self.assertEqual(actual, expected)

    def test_wavefront_scalar_broadcast_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x0000000F)  # Only first 4 lanes active

        # Write SGPR scalar broadcast
        simd.write_lane_reg_u32(0, RegKind.SGPR, 5, 0x7FFFFFFF)  # -> sat to 0x7FFF

        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 0x11111111)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane - 2)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.SGPR, 5),
        )
        simd.step(inst)

        for lane in range(32):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane < 4:
                expected = ref_cvt_pk_i16_i32(lane - 2, 0x7FFFFFFF)
                self.assertEqual(actual, expected)
            else:
                self.assertEqual(actual, 0x11111111)


class TestValuCvtPkI16I32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self) -> None:
        """v_cvt_pk_i16_i32 v0, v0, v1: D aliases with S0."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, 40000)  # > 32767 -> sat to 0x7FFF
        interp.write_reg_u32(RegKind.VGPR, 1, -50000) # < -32768 -> sat to 0x8000

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 1),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x80007FFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x80007FFF)
        # S1 (v1) remains unmodified
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), u32(-50000))

    def test_alias_dst_equals_src1(self) -> None:
        """v_cvt_pk_i16_i32 v0, v1, v0: D aliases with S1."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, 100)    # S0 in-range -> 0x0064
        interp.write_reg_u32(RegKind.VGPR, 0, -200)   # S1 in-range -> 0xFF38

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        expected = ((s32(-200) & 0xFFFF) << 16) | (100 & 0xFFFF)
        self.assertEqual(res, expected)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), expected)
        # S0 (v1) remains unmodified
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 100)

    def test_alias_dst_equals_src0_and_src1(self) -> None:
        """v_cvt_pk_i16_i32 v0, v0, v0: Full self-aliasing."""
        interp = ScalarVALUInterpreter()
        # 0x00010000 = 65536 in signed 32-bit integer -> saturates to 0x7FFF
        interp.write_reg_u32(RegKind.VGPR, 0, 0x00010000)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        # Both halves must saturate independently to 0x7FFF
        self.assertEqual(res, 0x7FFF7FFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x7FFF7FFF)

    def test_duplicate_sources_distinct_destination(self) -> None:
        """v_cvt_pk_i16_i32 v2, v0, v0: Duplicate source operands produce twin halfwords."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, -32768)  # -> sat to 0x8000

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x80008000)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 2), 0x80008000)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), u32(-32768))

    def test_adjacent_register_preservation(self) -> None:
        """Writes to target VGPR must not leak into adjacent register dwords."""
        interp = ScalarVALUInterpreter()
        canary_prev = 0xAAAA5555
        canary_next = 0x5555AAAA
        interp.write_reg_u32(RegKind.VGPR, 4, canary_prev)
        interp.write_reg_u32(RegKind.VGPR, 6, canary_next)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 5),
            src0=Immediate(1),
            src1=Immediate(2),
        )
        interp.step(inst)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 4), canary_prev)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 6), canary_next)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 5), 0x00020001)

    def test_simd_multi_lane_self_aliasing(self) -> None:
        """In-place self-aliasing across divergent SIMD wavefront."""
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0x0000FFFF)  # Low 16 lanes active

        for lane in range(32):
            # Give each lane a value that exceeds 32767
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, 50000 + lane)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.VGPR, 0),
        )
        simd.step(inst)

        for lane in range(32):
            actual = simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)
            if lane < 16:
                self.assertEqual(actual, 0x7FFF7FFF)
            else:
                self.assertEqual(actual, 50000 + lane)


class TestValuCvtPkI16I32ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation isolation."""

    def test_condition_codes_zero_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0),
            src1=Immediate(0),
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
        self.assertFalse(cc.lo_saturated)
        self.assertFalse(cc.hi_saturated)
        self.assertTrue(cc.parity)  # 0 has 0 bits set -> even parity

    def test_condition_codes_sign_flag(self) -> None:
        interp = ScalarVALUInterpreter()
        # S0 = 100, S1 = -1 -> D = 0xFFFF0064 (bit 31 set)
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(100),
            src1=Immediate(-1),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0xFFFF0064)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertTrue(cc.sign)
        self.assertFalse(cc.lo_zero)
        self.assertFalse(cc.hi_zero)

    def test_condition_codes_saturation_detection(self) -> None:
        interp = ScalarVALUInterpreter()
        # S0 overflows (40000 > 32767), S1 underflows (-40000 < -32768)
        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(40000),
            src1=Immediate(-40000),
        )
        interp.step(inst)
        cc = interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.lo_saturated)
        self.assertTrue(cc.hi_saturated)

        # S0 in range, S1 in range
        inst_in_range = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1000),
            src1=Immediate(-1000),
        )
        interp.step(inst_in_range)
        cc2 = interp.last_condition_codes
        self.assertIsNotNone(cc2)
        assert cc2 is not None
        self.assertFalse(cc2.lo_saturated)
        self.assertFalse(cc2.hi_saturated)

    def test_condition_codes_parity(self) -> None:
        # 1 bit in low byte -> odd parity (parity=False)
        cc_odd = compute_cvt_pk_condition_codes(0x0001, 1, 0)
        self.assertFalse(cc_odd.parity)

        # 2 bits in low byte (0x03) -> even parity (parity=True)
        cc_even = compute_cvt_pk_condition_codes(0x0003, 3, 0)
        self.assertTrue(cc_even.parity)

    def test_vcc_and_carry_borrow_isolation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary_vcc = 0xFEEDFACECAFEBEEF
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(0x7FFFFFFF),
            src1=Immediate(0x80000000),
        )
        interp.step(inst)
        # VCC is never modified
        self.assertEqual(interp.vcc, canary_vcc)

        flags = interp.last_status_flags
        self.assertIsNotNone(flags)
        assert flags is not None
        self.assertFalse(flags.vcc_modified)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.carry)

        # In SIMD wavefront, VCC is also invariant
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary_vcc
        simd.step(inst)
        self.assertEqual(simd.vcc, canary_vcc)


# ---------------------------------------------------------------------------
# IEEE 754 Floating-Point Constants, Decoders, and Boundary Behavior
# ---------------------------------------------------------------------------

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000
NEG_ONE_F32: int = 0xBF800000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
QNAN_PAYLOAD_F32: int = 0x7FC01234
SNAN_PAYLOAD_F32: int = 0x7F804321
SUBNORM_MIN_F32: int = 0x00000001
SUBNORM_MAX_F32: int = 0x007FFFFF
NEG_SUBNORM_MIN_F32: int = 0x80000001
NEG_SUBNORM_MAX_F32: int = 0x807FFFFF
FLT_MIN_F32: int = 0x00800000
FLT_MAX_F32: int = 0x7F7FFFFF


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_subnormal_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision subnormal."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


class TestValuCvtPkI16I32IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_float_zeros_as_integers(self) -> None:
        """+0.0 has bit pattern 0x00000000 (int 0); -0.0 has bit pattern 0x80000000 (int -2147483648)."""
        # +0.0 -> 0x0000
        self.assertEqual(sat_i16(POS_ZERO_F32), 0x0000)
        # -0.0 -> int -2147483648 -> saturates to INT16_MIN (-32768 = 0x8000)
        self.assertEqual(sat_i16(NEG_ZERO_F32), 0x8000)

        # Packed +0.0 and -0.0
        self.assertEqual(ref_cvt_pk_i16_i32(POS_ZERO_F32, NEG_ZERO_F32), 0x80000000)
        self.assertEqual(ref_cvt_pk_i16_i32(NEG_ZERO_F32, POS_ZERO_F32), 0x00008000)

    def test_positive_floats_saturate_to_max_int16(self) -> None:
        """Positive float patterns exceeding 32767 saturate to 32767 (0x7FFF)."""
        pos_floats = [
            POS_ONE_F32,      # 1.0 (int 1065353216)
            POS_INF_F32,      # +inf (int 2139095040)
            QNAN_F32,         # qNaN (int 2143289344)
            SNAN_F32,         # sNaN (int 2139095041)
            QNAN_PAYLOAD_F32, # qNaN with payload
            SNAN_PAYLOAD_F32, # sNaN with payload
            FLT_MIN_F32,      # 1.175e-38 (int 8388608)
            FLT_MAX_F32,      # ~3.4e38 (int 2139095039)
            SUBNORM_MAX_F32,  # max subnormal (int 8388607)
        ]
        for flt_bits in pos_floats:
            with self.subTest(flt=f"{flt_bits:#010x}"):
                self.assertEqual(sat_i16(flt_bits), 0x7FFF)
                self.assertEqual(ref_cvt_pk_i16_i32(flt_bits, flt_bits), 0x7FFF7FFF)

    def test_negative_floats_saturate_to_min_int16(self) -> None:
        """Negative float patterns with bit 31 set represent negative integers < -32768."""
        neg_floats = [
            NEG_ONE_F32,         # -1.0 (int -1082130432)
            NEG_INF_F32,         # -inf (int -8388608)
            NEG_SUBNORM_MIN_F32, # -subnorm_min (int -2147483647)
            NEG_SUBNORM_MAX_F32, # -subnorm_max (int -2139095041)
            0xFFC00000,          # negative qNaN (int -4194304)
            0xFF800001,          # negative sNaN (int -8388607)
        ]
        for flt_bits in neg_floats:
            with self.subTest(flt=f"{flt_bits:#010x}"):
                self.assertEqual(sat_i16(flt_bits), 0x8000)
                self.assertEqual(ref_cvt_pk_i16_i32(flt_bits, flt_bits), 0x80008000)

    def test_subnormals_integer_interpretation_no_ftz(self) -> None:
        """Subnormal floats with integer values in [1, 32767] pass through without flush-to-zero."""
        # 0x00000001: float 1.4e-45 -> int 1 -> returns 0x0001
        self.assertEqual(sat_i16(SUBNORM_MIN_F32), 0x0001)

        # 0x00000042: float 9.25e-44 -> int 66 -> returns 0x0042
        self.assertEqual(sat_i16(0x00000042), 0x0042)

        # 0x00007FFF: float 4.5916e-41 -> int 32767 -> returns 0x7FFF
        self.assertEqual(sat_i16(0x00007FFF), 0x7FFF)

        # 0x00008000: float 4.5918e-41 -> int 32768 -> saturates to 0x7FFF
        self.assertEqual(sat_i16(0x00008000), 0x7FFF)

        # Packed combination of two in-range subnormals
        self.assertEqual(ref_cvt_pk_i16_i32(0x00000005, 0x0000000A), 0x000A0005)

    def test_float_bitcasts_through_interpreter(self) -> None:
        """End-to-end execution of float bit patterns through scalar interpreter."""
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 1, POS_INF_F32)
        interp.write_reg_u32(RegKind.VGPR, 2, NEG_INF_F32)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        res = interp.step(inst)
        # S0 = +inf -> 0x7FFF, S1 = -inf -> 0x8000 => 0x80007FFF
        self.assertEqual(res, 0x80007FFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x80007FFF)


class TestValuCvtPkI16I32NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_negative_control_sgpr_as_destination(self) -> None:
        """v_cvt_pk_i16_i32 cannot target an SGPR destination."""
        with self.assertRaises(TypeError):
            Instruction(
                opcode=Opcode.V_CVT_PK_I16_I32,
                dst=Register(RegKind.SGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
            )

    def test_negative_control_literal_as_destination(self) -> None:
        """Instruction destination cannot be an immediate/literal."""
        with self.assertRaises(TypeError):
            Instruction(
                opcode=Opcode.V_CVT_PK_I16_I32,
                dst=Immediate(0),  # type: ignore[arg-type]
                src0=Register(RegKind.VGPR, 1),
                src1=Register(RegKind.VGPR, 2),
            )

    def test_negative_control_missing_source_operand(self) -> None:
        """v_cvt_pk_i16_i32 requires exactly two source operands."""
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_PK_I16_I32,
                dst=Register(RegKind.VGPR, 0),
                src0=Register(RegKind.VGPR, 1),
                src1=None,
            )

    def test_negative_control_out_of_bounds_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.SGPR, 106)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)

    def test_negative_control_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=128)

    def test_negative_control_unsupported_operand_type(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(TypeError):
            interp.read_operand("v0")  # type: ignore[arg-type]

    def test_negative_control_unsupported_opcode_rejection(self) -> None:
        class FakeOpcode(enum.Enum):
            UNKNOWN = "v_unknown"

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
        )
        object.__setattr__(inst, "opcode", FakeOpcode.UNKNOWN)

        interp = ScalarVALUInterpreter()
        with self.assertRaises(NotImplementedError):
            interp.step(inst)

    def test_negative_control_clamp_modifier_rejection(self) -> None:
        """v_cvt_pk_i16_i32 strictly rejects the clamp modifier."""
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_CVT_PK_I16_I32,
                dst=Register(RegKind.VGPR, 0),
                src0=Immediate(1),
                src1=Immediate(2),
                modifiers=("clamp",),
            )

    def test_mutation_detection_unsigned_saturation(self) -> None:
        """Sabotage: using unsigned clamp min(u32, 0xFFFF) instead of signed clamp."""
        def mutant_sat_unsigned(val: int) -> int:
            return min(u32(val), 0xFFFF)

        def mutant_cvt_pk(s0: int, s1: int) -> int:
            return pack_i16_to_u32(mutant_sat_unsigned(s0), mutant_sat_unsigned(s1))

        # For s0 = -40000, correct signed saturation is 0x8000 (-32768)
        # Mutant treats -40000 as 0xFFFF63C0, min(..., 0xFFFF) gives 0xFFFF!
        expected = ref_cvt_pk_i16_i32(-40000, 0)
        mutant_val = mutant_cvt_pk(-40000, 0)
        self.assertEqual(expected, 0x00008000)
        self.assertNotEqual(mutant_val, expected)
        with self.assertRaises(AssertionError):
            self.assertEqual(mutant_val, expected)

    def test_mutation_detection_inverted_operand_order(self) -> None:
        """Sabotage: packing S1 into low half and S0 into high half."""
        def mutant_cvt_pk(s0: int, s1: int) -> int:
            return pack_i16_to_u32(sat_i16(s1), sat_i16(s0))

        s0 = 100
        s1 = 200
        expected = ref_cvt_pk_i16_i32(s0, s1)
        mutant_val = mutant_cvt_pk(s0, s1)
        self.assertEqual(expected, 0x00C80064)
        self.assertEqual(mutant_val, 0x006400C8)
        with self.assertRaises(AssertionError):
            self.assertEqual(mutant_val, expected)

    def test_mutation_detection_wrong_negative_boundary(self) -> None:
        """Sabotage: clamping to [-32767, 32767] instead of [-32768, 32767]."""
        def mutant_sat(val: int) -> int:
            s = s32(val)
            if s > 32767:
                return 32767
            if s < -32767:  # Bug: should be -32768
                return (-32767) & MASK16
            return s & MASK16

        # For s = -32768, correct is 0x8000 (-32768). Mutant returns 0x8001 (-32767).
        expected = sat_i16(-32768)
        mutant_val = mutant_sat(-32768)
        self.assertEqual(expected, 0x8000)
        self.assertEqual(mutant_val, 0x8001)
        with self.assertRaises(AssertionError):
            self.assertEqual(mutant_val, expected)

    def test_mutation_detection_wrap_without_saturation(self) -> None:
        """Sabotage: truncation/modulo 2^16 instead of saturation."""
        def mutant_cvt_pk(s0: int, s1: int) -> int:
            return pack_i16_to_u32(s0 & 0xFFFF, s1 & 0xFFFF)

        s0 = 40000  # in hex: 0x9C40 -> should saturate to 0x7FFF
        expected = ref_cvt_pk_i16_i32(s0, 0)
        mutant_val = mutant_cvt_pk(s0, 0)
        self.assertEqual(expected, 0x00007FFF)
        self.assertEqual(mutant_val, 0x00009C40)
        with self.assertRaises(AssertionError):
            self.assertEqual(mutant_val, expected)

    def test_mutation_detection_simd_exec_leak(self) -> None:
        """Sabotage: inactive SIMD lane modifies destination register."""
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0)  # All lanes inactive
        canary = 0xBAD0CAFE
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, canary)

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1),
            src1=Immediate(2),
        )
        simd.step(inst)
        self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)

        # Mutated behavior: unconditionally overwrite
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, 0x12345678)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)

    def test_mutation_detection_vcc_corruption(self) -> None:
        """Sabotage: v_cvt_pk_i16_i32 clobbering VCC."""
        interp = ScalarVALUInterpreter()
        canary_vcc = 0x1234567890ABCDEF
        interp.vcc = canary_vcc

        inst = Instruction(
            opcode=Opcode.V_CVT_PK_I16_I32,
            dst=Register(RegKind.VGPR, 0),
            src0=Immediate(1),
            src1=Immediate(2),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary_vcc)

        # Mutated step corrupts VCC
        interp.vcc = 0
        with self.assertRaises(AssertionError):
            self.assertEqual(interp.vcc, canary_vcc)


if __name__ == "__main__":
    unittest.main()
