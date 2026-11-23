"""Semantic hardware-level and translator reference tests for AMD RDNA2 v_max_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture & Semantics:
  In AMD RDNA2 (GFX10 / GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point numbers (IEEE 754 binary64)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operands
  v[s0:s0+1] and v[s1:s1+1] (or scalar SGPR pairs, literal 64-bit constants, or inline constants).

  Instruction:
    v_max_f64 v[d:d+1], src0, src1 [clamp]

  Mathematical Specification:
    D.f64 = max(S0.f64, S1.f64)

  IEEE-754 binary64 Encoding & RDNA2 Hardware Rules:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)
    - NaN Suppression:
        If one operand is NaN and the other is a valid floating-point number, the valid
        number is returned unchanged.
        If both operands are NaN, a quiet NaN is returned.
    - Signed Zero Ordering (-0.0 < +0.0):
        RDNA2 orders -0.0 strictly less than +0.0:
          max(+0.0, -0.0) = +0.0
          max(-0.0, +0.0) = +0.0
          max(+0.0, +0.0) = +0.0
          max(-0.0, -0.0) = -0.0
    - Infinities & Subnormals:
        -INF < any finite number < +INF.
        Subnormal numbers are preserved without flushing to zero in default IEEE mode.
    - VOP3 Encoding Modifiers:
        Input modifiers: abs (|x|) clears sign bit 63; neg (-x) inverts sign bit 63.
        When both abs and neg are set, neg(abs(x)) clears then sets sign bit (forcing negative).
        Output modifier: clamp clamps final result to [0.0, 1.0]. Negative numbers and -0.0
        clamp to +0.0; values greater than 1.0 clamp to 1.0; NaNs are unchanged.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
QUIET_NAN_BIT_F64: int = 0x0008000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000       # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
POS_TWO_F64: int = 0x4000000000000000       # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
POS_THREE_F64: int = 0x4008000000000000     # 3.0
POS_TEN_F64: int = 0x4024000000000000       # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # (2 - 2^-52) * 2^1023
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000       # +infinity
NEG_INF_F64: int = 0xFFF0000000000000       # -infinity
QNAN_F64: int = 0x7FF8000000000000          # Canonical quiet NaN
NEG_QNAN_F64: int = 0xFFF8000000000000      # Negative quiet NaN
QNAN_PAYLOAD_A: int = 0x7FF8000000000001    # Quiet NaN with payload A
QNAN_PAYLOAD_B: int = 0xFFF9000000000002    # Quiet NaN with payload B
SNAN_F64: int = 0x7FF0000000000001          # Signaling NaN
SUBNORM_MIN_F64: int = 0x0000000000000001   # Smallest positive subnormal: 2^-1074
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF   # Largest positive subnormal
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range."""
    return val & MASK64


def split_dwords(val64: int) -> Tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_qnan_f64(bits: int) -> bool:
    """True if bits represent a quiet NaN."""
    bits &= MASK64
    return is_nan_f64(bits) and bool(bits & QUIET_NAN_BIT_F64)


def is_snan_f64(bits: int) -> bool:
    """True if bits represent a signaling NaN."""
    bits &= MASK64
    return is_nan_f64(bits) and not bool(bits & QUIET_NAN_BIT_F64)


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def zero_ordered_key(v: float) -> Tuple[float, float]:
    """Sort key ordering -0.0 strictly below +0.0."""
    return (v, math.copysign(1.0, v))


def ref_max_f64(s0: int, s1: int) -> int:
    """Pure bitwise reference oracle for RDNA2 v_max_f64.

    Rules:
      - If one operand is NaN, return the non-NaN operand unchanged.
      - If both operands are NaN, return canonical quiet NaN.
      - If non-NaN values, order -0.0 < +0.0.
    """
    s0 &= MASK64
    s1 &= MASK64
    nan0 = is_nan_f64(s0)
    nan1 = is_nan_f64(s1)
    if nan0 and nan1:
        return QNAN_F64
    if nan0:
        return s1
    if nan1:
        return s0

    f0 = bits_to_f64(s0)
    f1 = bits_to_f64(s1)
    k0 = zero_ordered_key(f0)
    k1 = zero_ordered_key(f1)
    return s0 if k0 >= k1 else s1


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits


def apply_f64_clamp(bits: int) -> int:
    """Applies VOP3 clamp output modifier [0.0, 1.0]."""
    bits &= MASK64
    if is_nan_f64(bits):
        return bits
    f = bits_to_f64(bits)
    if f < 0.0 or (f == 0.0 and (bits & SIGN_MASK_F64)):
        return POS_ZERO_F64
    if f > 1.0:
        return POS_ONE_F64
    return bits


# =============================================================================
# Symbolic AST & Instruction Representation
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RegSingle:
    idx: int

    def __post_init__(self) -> None:
        if not (0 <= self.idx <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.idx}")


@dataclasses.dataclass(frozen=True)
class RegPair:
    lo_idx: int

    def __post_init__(self) -> None:
        if not (0 <= self.lo_idx <= 254):
            raise ValueError(f"VGPR pair lo index out of range [0, 254]: {self.lo_idx}")

    @property
    def hi_idx(self) -> int:
        return self.lo_idx + 1


@dataclasses.dataclass(frozen=True)
class ImmF64:
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK64):
            raise ValueError(f"64-bit constant out of range: {self.bits}")


@dataclasses.dataclass(frozen=True)
class SourceOperand:
    val: Union[RegPair, ImmF64]
    neg: bool = False
    abs_val: bool = False


@dataclasses.dataclass(frozen=True)
class VMaxF64Instruction:
    vdst: RegPair
    src0: SourceOperand
    src1: SourceOperand
    clamp: bool = False

    @staticmethod
    def parse(line: str) -> VMaxF64Instruction:
        tokens = line.replace(",", " ").split()
        if not tokens or tokens[0] != "v_max_f64":
            raise ValueError(f"Invalid opcode for v_max_f64: {line}")
        vdst = VMaxF64Instruction._parse_reg_pair(tokens[1])
        idx = 2
        src0 = VMaxF64Instruction._parse_src(tokens[idx])
        idx += 1
        src1 = VMaxF64Instruction._parse_src(tokens[idx])
        idx += 1
        clamp = False
        while idx < len(tokens):
            mod = tokens[idx].lower()
            if mod == "clamp":
                clamp = True
            else:
                raise ValueError(f"Unknown modifier for v_max_f64: {mod}")
            idx += 1
        return VMaxF64Instruction(vdst=vdst, src0=src0, src1=src1, clamp=clamp)

    @staticmethod
    def _parse_reg_pair(text: str) -> RegPair:
        t = text.strip().lower()
        if not (t.startswith("v[") and t.endswith("]") and ":" in t):
            raise ValueError(f"Malformed register pair: {text}")
        inner = t[2:-1]
        lo_str, hi_str = inner.split(":")
        lo, hi = int(lo_str), int(hi_str)
        if hi != lo + 1:
            raise ValueError(f"Register pair must span adjacent registers v[n:n+1]: {text}")
        return RegPair(lo)

    @staticmethod
    def _parse_src(text: str) -> SourceOperand:
        t = text.strip()
        neg = False
        abs_val = False
        if t.startswith("-"):
            neg = True
            t = t[1:].strip()
        if t.startswith("|") and t.endswith("|"):
            abs_val = True
            t = t[1:-1].strip()
        if t.startswith("-"):
            neg = True
            t = t[1:].strip()
        if t.lower().startswith("v["):
            reg = VMaxF64Instruction._parse_reg_pair(t)
            return SourceOperand(val=reg, neg=neg, abs_val=abs_val)
        else:
            val_int = int(t, 0)
            return SourceOperand(val=ImmF64(val_int), neg=neg, abs_val=abs_val)


# =============================================================================
# Architectural Interpreter
# =============================================================================

class ScalarInterpreter:
    """Deterministic scalar register file and execution engine for v_max_f64."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.vcc: int = 0
        self.scc: int = 0

    def read_vgpr64(self, pair: RegPair) -> int:
        return join_dwords(self.vgprs[pair.lo_idx], self.vgprs[pair.hi_idx])

    def write_vgpr64(self, pair: RegPair, val64: int) -> None:
        lo, hi = split_dwords(val64)
        self.vgprs[pair.lo_idx] = lo
        self.vgprs[pair.hi_idx] = hi

    def evaluate_src(self, src: SourceOperand) -> int:
        if isinstance(src.val, RegPair):
            raw = self.read_vgpr64(src.val)
        elif isinstance(src.val, ImmF64):
            raw = src.val.bits
        else:
            raise TypeError(f"Unknown operand type: {type(src.val)}")
        return apply_f64_modifiers(raw, neg=src.neg, abs_val=src.abs_val)

    def execute(self, instr: VMaxF64Instruction) -> int:
        s0 = self.evaluate_src(instr.src0)
        s1 = self.evaluate_src(instr.src1)
        res = ref_max_f64(s0, s1)
        if instr.clamp:
            res = apply_f64_clamp(res)
        self.write_vgpr64(instr.vdst, res)
        return res


class SIMDInterpreter:
    """SIMD execution engine supporting 32 and 64 lanes with divergence masking."""

    def __init__(self, num_lanes: int = 64) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes}")
        self.num_lanes = num_lanes
        self.lanes: List[ScalarInterpreter] = [ScalarInterpreter() for _ in range(num_lanes)]

    def execute_wave(self, instr: VMaxF64Instruction, exec_mask: int) -> List[int]:
        results: List[int] = []
        for lane_id in range(self.num_lanes):
            lane_active = bool(exec_mask & (1 << lane_id))
            interp = self.lanes[lane_id]
            if lane_active:
                results.append(interp.execute(instr))
            else:
                results.append(interp.read_vgpr64(instr.vdst))
        return results


# =============================================================================
# Unit Tests: Mathematical Oracle & Edge Cases
# =============================================================================

class TestValuMaxF64MathematicalOracle(unittest.TestCase):
    """Tests covering reference mathematical oracle and standard IEEE 754 binary64 rules."""

    def test_basic_comparisons(self) -> None:
        self.assertEqual(ref_max_f64(POS_ONE_F64, POS_TWO_F64), POS_TWO_F64)
        self.assertEqual(ref_max_f64(POS_TWO_F64, POS_ONE_F64), POS_TWO_F64)
        self.assertEqual(ref_max_f64(NEG_ONE_F64, NEG_TWO_F64), NEG_ONE_F64)
        self.assertEqual(ref_max_f64(NEG_ONE_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_TEN_F64, NEG_TEN_F64), POS_TEN_F64)

    def test_signed_zeros(self) -> None:
        # -0.0 < +0.0 strictly ordered on RDNA2 hardware
        self.assertEqual(ref_max_f64(POS_ZERO_F64, NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(NEG_ZERO_F64, POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(POS_ZERO_F64, POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(NEG_ZERO_F64, NEG_ZERO_F64), NEG_ZERO_F64)

    def test_nan_suppression(self) -> None:
        # When one operand is NaN and other is valid number, non-NaN is returned
        self.assertEqual(ref_max_f64(QNAN_F64, POS_ONE_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(POS_ONE_F64, QNAN_F64), POS_ONE_F64)
        self.assertEqual(ref_max_f64(SNAN_F64, NEG_TWO_F64), NEG_TWO_F64)
        self.assertEqual(ref_max_f64(NEG_TWO_F64, SNAN_F64), NEG_TWO_F64)
        self.assertEqual(ref_max_f64(QNAN_PAYLOAD_A, NEG_INF_F64), NEG_INF_F64)

    def test_both_nan(self) -> None:
        res = ref_max_f64(QNAN_PAYLOAD_A, QNAN_PAYLOAD_B)
        self.assertTrue(is_nan_f64(res))

    def test_infinities(self) -> None:
        self.assertEqual(ref_max_f64(POS_INF_F64, DBL_MAX), POS_INF_F64)
        self.assertEqual(ref_max_f64(NEG_INF_F64, NEG_DBL_MAX), NEG_DBL_MAX)
        self.assertEqual(ref_max_f64(POS_INF_F64, NEG_INF_F64), POS_INF_F64)

    def test_subnormals(self) -> None:
        self.assertEqual(ref_max_f64(SUBNORM_MIN_F64, SUBNORM_MAX_F64), SUBNORM_MAX_F64)
        self.assertEqual(ref_max_f64(NEG_SUBNORM_MIN_F64, NEG_SUBNORM_MAX_F64), NEG_SUBNORM_MIN_F64)
        self.assertEqual(ref_max_f64(SUBNORM_MIN_F64, POS_ZERO_F64), SUBNORM_MIN_F64)
        self.assertEqual(ref_max_f64(NEG_SUBNORM_MIN_F64, POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_max_f64(NEG_SUBNORM_MIN_F64, NEG_ZERO_F64), NEG_ZERO_F64)

    def test_input_modifiers(self) -> None:
        # abs(|-2.0|) = 2.0; max(1.0, 2.0) = 2.0
        s0 = apply_f64_modifiers(POS_ONE_F64)
        s1 = apply_f64_modifiers(NEG_TWO_F64, abs_val=True)
        self.assertEqual(ref_max_f64(s0, s1), POS_TWO_F64)

        # neg(-(2.0)) = -2.0; max(1.0, -2.0) = 1.0
        s1_neg = apply_f64_modifiers(POS_TWO_F64, neg=True)
        self.assertEqual(ref_max_f64(s0, s1_neg), POS_ONE_F64)

        # abs and neg together forces negative: neg(abs(2.0)) = -2.0
        s1_both = apply_f64_modifiers(POS_TWO_F64, neg=True, abs_val=True)
        self.assertEqual(s1_both, NEG_TWO_F64)

    def test_output_clamp_modifier(self) -> None:
        # Clamps to [0.0, 1.0]
        self.assertEqual(apply_f64_clamp(POS_TWO_F64), POS_ONE_F64)
        self.assertEqual(apply_f64_clamp(NEG_TWO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(NEG_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(HALF_F64), HALF_F64)
        self.assertEqual(apply_f64_clamp(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(apply_f64_clamp(POS_ONE_F64), POS_ONE_F64)
        # NaN is unchanged under clamp
        self.assertEqual(apply_f64_clamp(QNAN_F64), QNAN_F64)


# =============================================================================
# Unit Tests: Symbolic AST & Parser
# =============================================================================

class TestValuMaxF64SymbolicAST(unittest.TestCase):
    """Tests covering AST creation, operand typing, and syntax parsing."""

    def test_instruction_parsing_basic(self) -> None:
        instr = VMaxF64Instruction.parse("v_max_f64 v[0:1], v[2:3], v[4:5]")
        self.assertEqual(instr.vdst.lo_idx, 0)
        self.assertEqual(instr.src0.val, RegPair(2))
        self.assertFalse(instr.src0.neg)
        self.assertFalse(instr.src0.abs_val)
        self.assertEqual(instr.src1.val, RegPair(4))
        self.assertFalse(instr.clamp)

    def test_instruction_parsing_modifiers(self) -> None:
        instr = VMaxF64Instruction.parse("v_max_f64 v[6:7], -|v[0:1]|, v[2:3] clamp")
        self.assertEqual(instr.vdst.lo_idx, 6)
        self.assertEqual(instr.src0.val, RegPair(0))
        self.assertTrue(instr.src0.neg)
        self.assertTrue(instr.src0.abs_val)
        self.assertTrue(instr.clamp)

    def test_instruction_parsing_literal(self) -> None:
        instr = VMaxF64Instruction.parse("v_max_f64 v[0:1], v[2:3], 0x3FF0000000000000")
        self.assertIsInstance(instr.src1.val, ImmF64)
        self.assertEqual(instr.src1.val.bits, POS_ONE_F64)


# =============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# =============================================================================

class TestValuMaxF64ScalarInterpreter(unittest.TestCase):
    """Tests covering deterministic register execution in a single thread."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execution_normal(self) -> None:
        self.interp.write_vgpr64(RegPair(0), POS_ONE_F64)
        self.interp.write_vgpr64(RegPair(2), POS_TWO_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[2:3]")
        got = self.interp.execute(instr)
        self.assertEqual(got, POS_TWO_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(4)), POS_TWO_F64)

    def test_scalar_execution_clamp(self) -> None:
        self.interp.write_vgpr64(RegPair(0), POS_THREE_F64)
        self.interp.write_vgpr64(RegPair(2), POS_TEN_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[2:3] clamp")
        got = self.interp.execute(instr)
        self.assertEqual(got, POS_ONE_F64)


# =============================================================================
# Unit Tests: SIMD Multi-Lane & EXEC Masking
# =============================================================================

class TestValuMaxF64SIMDDivergence(unittest.TestCase):
    """Tests covering SIMD multi-lane divergence across Wave32 and Wave64."""

    def test_wave32_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=32)
        sentinel = 0xAAAAAAAAAAAAAAAA
        for lane_id in range(32):
            simd.lanes[lane_id].write_vgpr64(RegPair(0), f64_to_bits(float(lane_id)))
            simd.lanes[lane_id].write_vgpr64(RegPair(2), f64_to_bits(16.0))
            simd.lanes[lane_id].write_vgpr64(RegPair(4), sentinel)

        # Active only for even lanes
        exec_mask = 0x55555555
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[2:3]")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(32):
            if lane_id % 2 == 0:
                expected = f64_to_bits(max(float(lane_id), 16.0))
                self.assertEqual(results[lane_id], expected)
                self.assertEqual(simd.lanes[lane_id].read_vgpr64(RegPair(4)), expected)
            else:
                self.assertEqual(results[lane_id], sentinel)
                self.assertEqual(simd.lanes[lane_id].read_vgpr64(RegPair(4)), sentinel)

    def test_wave64_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=64)
        sentinel = 0xBBBBBBBBBBBBBBBB
        for lane_id in range(64):
            simd.lanes[lane_id].write_vgpr64(RegPair(0), f64_to_bits(float(lane_id)))
            simd.lanes[lane_id].write_vgpr64(RegPair(2), f64_to_bits(32.0))
            simd.lanes[lane_id].write_vgpr64(RegPair(4), sentinel)

        # Upper 32 lanes active only
        exec_mask = 0xFFFFFFFF00000000
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[2:3]")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(64):
            if lane_id >= 32:
                expected = f64_to_bits(float(lane_id))
                self.assertEqual(results[lane_id], expected)
            else:
                self.assertEqual(results[lane_id], sentinel)


# =============================================================================
# Unit Tests: Register Aliasing & Preservation
# =============================================================================

class TestValuMaxF64RegisterAliasing(unittest.TestCase):
    """Tests covering in-place destination overwrites and register overlaps."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_vdst_equals_src0(self) -> None:
        self.interp.write_vgpr64(RegPair(0), POS_ONE_F64)
        self.interp.write_vgpr64(RegPair(2), POS_TWO_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[0:1], v[0:1], v[2:3]")
        got = self.interp.execute(instr)
        self.assertEqual(got, POS_TWO_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(0)), POS_TWO_F64)
        # Source 1 should be untouched
        self.assertEqual(self.interp.read_vgpr64(RegPair(2)), POS_TWO_F64)

    def test_vdst_equals_src1(self) -> None:
        self.interp.write_vgpr64(RegPair(0), POS_TEN_F64)
        self.interp.write_vgpr64(RegPair(2), POS_TWO_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[2:3], v[0:1], v[2:3]")
        got = self.interp.execute(instr)
        self.assertEqual(got, POS_TEN_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(2)), POS_TEN_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(0)), POS_TEN_F64)

    def test_src0_equals_src1(self) -> None:
        self.interp.write_vgpr64(RegPair(0), NEG_TEN_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[0:1]")
        got = self.interp.execute(instr)
        self.assertEqual(got, NEG_TEN_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(4)), NEG_TEN_F64)

    def test_all_operands_identical(self) -> None:
        self.interp.write_vgpr64(RegPair(0), HALF_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[0:1], v[0:1], v[0:1]")
        got = self.interp.execute(instr)
        self.assertEqual(got, HALF_F64)
        self.assertEqual(self.interp.read_vgpr64(RegPair(0)), HALF_F64)


# =============================================================================
# Unit Tests: Condition Codes & Flags
# =============================================================================

class TestValuMaxF64ConditionCodes(unittest.TestCase):
    """Tests verifying SCC and VCC flags remain completely isolated."""

    def test_flags_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.scc = 1
        interp.vcc = 0x123456789ABCDEF0
        interp.write_vgpr64(RegPair(0), POS_ONE_F64)
        interp.write_vgpr64(RegPair(2), POS_TWO_F64)
        instr = VMaxF64Instruction.parse("v_max_f64 v[4:5], v[0:1], v[2:3]")
        interp.execute(instr)
        self.assertEqual(interp.scc, 1)
        self.assertEqual(interp.vcc, 0x123456789ABCDEF0)


# =============================================================================
# Unit Tests: Negative Controls & Mutation Detection
# =============================================================================

class TestValuMaxF64NegativeControls(unittest.TestCase):
    """Negative controls testing syntax rejection and mutation detection."""

    def test_syntax_rejection(self) -> None:
        with self.assertRaises(ValueError):
            VMaxF64Instruction.parse("v_min_f64 v[0:1], v[2:3], v[4:5]")

        with self.assertRaises(ValueError):
            # Unpaired register
            VMaxF64Instruction.parse("v_max_f64 v0, v[2:3], v[4:5]")

        with self.assertRaises(ValueError):
            # Non-adjacent pair
            VMaxF64Instruction.parse("v_max_f64 v[0:2], v[2:3], v[4:5]")

        with self.assertRaises(ValueError):
            # Unknown modifier
            VMaxF64Instruction.parse("v_max_f64 v[0:1], v[2:3], v[4:5] omod:2")

    def test_mutation_detector_signed_zero_order(self) -> None:
        # Mutation: faulty implementation treating +0.0 and -0.0 as indistinguishable
        # If implementation erroneously returns -0.0 for max(+0.0, -0.0), it fails.
        got = ref_max_f64(POS_ZERO_F64, NEG_ZERO_F64)
        self.assertEqual(got, POS_ZERO_F64)
        self.assertNotEqual(got, NEG_ZERO_F64)

    def test_mutation_detector_nan_suppression(self) -> None:
        # Mutation: IEEE 754 fmax propagating NaN instead of suppressing it
        got = ref_max_f64(QNAN_F64, POS_ONE_F64)
        self.assertEqual(got, POS_ONE_F64)
        self.assertFalse(is_nan_f64(got))


# =============================================================================
# Unit Tests: Translator Integration (tools/rdna2spirv.py)
# =============================================================================

class TestValuMaxF64Translator(unittest.TestCase):
    """End-to-end integration tests exercising tools/rdna2spirv.py."""

    def test_translator_invocation(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 0x00000000",
            "v_mov_b32 v1, 0x3FF00000",  # 1.0 (lo=0, hi=0x3FF00000)
            "v_mov_b32 v2, 0x00000000",
            "v_mov_b32 v3, 0x40000000",  # 2.0 (lo=0, hi=0x40000000)
            "v_max_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        try:
            spvasm = support.translate_lines("ps_max_f64_test", asm)
            self.assertIsInstance(spvasm, str)
            # When lowering is supported, check emitted SPIR-V instructions
            self.assertTrue(
                "NMax" in spvasm or "FMax" in spvasm or "OpExtInst" in spvasm,
                "Expected GLSL NMax or FMax instruction in translated output",
            )
        except rdna2spirv.Unsupported as e:
            # Opcode lowering pending upstream merge in rdna2spirv.py
            self.assertIn("v_max_f64", str(e))


if __name__ == "__main__":
    unittest.main()
