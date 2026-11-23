"""Semantic hardware-level and translator reference tests for AMD RDNA2 v_sqrt_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture & Semantics:
  In AMD RDNA2 (GFX10 / GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  16-bit floating-point (binary16, IEEE 754 half-precision) unary operations operate
  on halfword slices of 32-bit vector registers.

  Instruction:
    v_sqrt_f16 vd, src [clamp]

  Mathematical Specification:
    D.f16 = sqrt(S0.f16)

  IEEE-754 binary16 Encoding & RDNA2 Hardware Rules:
    - 1 sign bit (bit 15)
    - 5 exponent bits with bias 15 (bits 14:10)
    - 10 mantissa / fraction bits (bits 9:0)
    - Special Value Handling:
        sqrt(+0.0) = +0.0 (0x0000)
        sqrt(-0.0) = -0.0 (0x8000)
        sqrt(+INF) = +INF (0x7C00)
        sqrt(-INF) = Canonical Quiet NaN (0x7E00)
        sqrt(x < -0.0) = Canonical Quiet NaN (0x7E00)
        sqrt(NaN) = Canonical Quiet NaN (0x7E00)
    - Subnormal Numbers:
        Subnormals are supported and compute exact rounded IEEE 754 binary16 square roots.
    - Register Allocation & Halfword Preservation:
        On RDNA2 GFX10, single-lane 16-bit VALU operations update the target halfword
        (default low 16 bits [15:0]) and preserve the other halfword ([31:16]) of the destination VGPR.
    - VOP3 Encoding Modifiers:
        Input modifiers: abs (|x|) clears sign bit 15; neg (-x) inverts sign bit 15.
        Output modifier: clamp clamps final result to [0.0, 1.0]. Values > 1.0 clamp to 1.0 (0x3C00);
        negative values and -0.0 clamp to +0.0 (0x0000); NaNs remain NaN.

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

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
FRAC_MASK_F16: int = 0x03FF
QUIET_NAN_BIT_F16: int = 0x0200

# Canonical IEEE 754 binary16 bit constants
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00          # Canonical Quiet NaN
SNAN_F16: int = 0x7C01          # Signaling NaN (exp all 1s, payload != 0, quiet bit 0)

ONE_F16: int = 0x3C00           # 1.0
FOUR_F16: int = 0x4400          # 4.0
NINE_F16: int = 0x4880          # 9.0
TWO_F16: int = 0x4000           # 2.0
THREE_F16: int = 0x4200         # 3.0
HALF_F16: int = 0x3800          # 0.5
QUARTER_F16: int = 0x3400       # 0.25

MIN_NORMAL_F16: int = 0x0400    # 2^-14
MAX_NORMAL_F16: int = 0x7BFF    # 65504.0
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24
MAX_SUBNORMAL_F16: int = 0x03FF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range."""
    return val & MASK32


def f16_to_bits(val: float) -> int:
    """Convert a Python float to 16-bit binary16 integer bits."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Convert 16-bit binary16 integer bits to a Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision NaN."""
    bits &= MASK16
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 half-precision subnormal."""
    bits &= MASK16
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def ref_sqrt_f16(s0: int) -> int:
    """Pure bitwise reference mathematical oracle for RDNA2 v_sqrt_f16.

    Rules:
      - If s0 is NaN: returns canonical quiet NaN.
      - If s0 is -0.0: returns -0.0.
      - If s0 is +0.0: returns +0.0.
      - If s0 is negative (and not -0.0): returns canonical quiet NaN.
      - If s0 is +INF: returns +INF.
      - Otherwise: returns IEEE 754 half-precision round-to-nearest-even sqrt(s0).
    """
    s0 &= MASK16
    if is_nan_f16(s0):
        return QNAN_F16
    if s0 == NEG_ZERO_F16:
        return NEG_ZERO_F16
    if s0 == POS_ZERO_F16:
        return POS_ZERO_F16
    if s0 & SIGN_MASK_F16:
        return QNAN_F16
    if s0 == POS_INF_F16:
        return POS_INF_F16

    f = bits_to_f16(s0)
    root = math.sqrt(f)
    return f16_to_bits(root)


def apply_f16_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK16
    if abs_val:
        bits &= ~SIGN_MASK_F16
    if neg:
        bits ^= SIGN_MASK_F16
    return bits


def apply_f16_clamp(bits: int) -> int:
    """Applies VOP3 clamp output modifier [0.0, 1.0]."""
    bits &= MASK16
    if is_nan_f16(bits):
        return bits
    f = bits_to_f16(bits)
    if f < 0.0 or (f == 0.0 and (bits & SIGN_MASK_F16)):
        return POS_ZERO_F16
    if f > 1.0:
        return ONE_F16
    return bits


# =============================================================================
# Symbolic AST & Instruction Representation
# =============================================================================

@dataclasses.dataclass(frozen=True)
class RegVGPR:
    idx: int

    def __post_init__(self) -> None:
        if not (0 <= self.idx <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.idx}")


@dataclasses.dataclass(frozen=True)
class ImmF16:
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK16):
            raise ValueError(f"16-bit constant out of range: {self.bits}")


@dataclasses.dataclass(frozen=True)
class SourceOperand:
    val: Union[RegVGPR, ImmF16]
    neg: bool = False
    abs_val: bool = False


@dataclasses.dataclass(frozen=True)
class VSqrtF16Instruction:
    vdst: RegVGPR
    src0: SourceOperand
    clamp: bool = False

    @staticmethod
    def parse(line: str) -> VSqrtF16Instruction:
        tokens = line.replace(",", " ").split()
        if not tokens or tokens[0] != "v_sqrt_f16":
            raise ValueError(f"Invalid opcode for v_sqrt_f16: {line}")
        vdst = VSqrtF16Instruction._parse_reg(tokens[1])
        src0 = VSqrtF16Instruction._parse_src(tokens[2])
        clamp = False
        idx = 3
        while idx < len(tokens):
            mod = tokens[idx].lower()
            if mod == "clamp":
                clamp = True
            else:
                raise ValueError(f"Unknown modifier for v_sqrt_f16: {mod}")
            idx += 1
        return VSqrtF16Instruction(vdst=vdst, src0=src0, clamp=clamp)

    @staticmethod
    def _parse_reg(text: str) -> RegVGPR:
        t = text.strip().lower()
        if not (t.startswith("v") and t[1:].isdigit()):
            raise ValueError(f"Malformed VGPR register: {text}")
        return RegVGPR(int(t[1:]))

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
        if t.lower().startswith("v") and t[1:].isdigit():
            return SourceOperand(val=RegVGPR(int(t[1:])), neg=neg, abs_val=abs_val)
        else:
            val_int = int(t, 0)
            return SourceOperand(val=ImmF16(val_int), neg=neg, abs_val=abs_val)


# =============================================================================
# Architectural Interpreter
# =============================================================================

class ScalarInterpreter:
    """Deterministic scalar register file and execution engine for v_sqrt_f16."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.vcc: int = 0
        self.scc: int = 0

    def read_f16(self, reg: RegVGPR) -> int:
        return self.vgprs[reg.idx] & MASK16

    def write_f16(self, reg: RegVGPR, val16: int) -> None:
        # GFX10 preservation: preserves upper 16 bits of destination VGPR
        old_hi = self.vgprs[reg.idx] & 0xFFFF0000
        self.vgprs[reg.idx] = old_hi | (val16 & MASK16)

    def evaluate_src(self, src: SourceOperand) -> int:
        if isinstance(src.val, RegVGPR):
            raw = self.read_f16(src.val)
        elif isinstance(src.val, ImmF16):
            raw = src.val.bits
        else:
            raise TypeError(f"Unknown operand type: {type(src.val)}")
        return apply_f16_modifiers(raw, neg=src.neg, abs_val=src.abs_val)

    def execute(self, instr: VSqrtF16Instruction) -> int:
        s0 = self.evaluate_src(instr.src0)
        res = ref_sqrt_f16(s0)
        if instr.clamp:
            res = apply_f16_clamp(res)
        self.write_f16(instr.vdst, res)
        return res


class SIMDInterpreter:
    """SIMD execution engine supporting 32 and 64 lanes with divergence masking."""

    def __init__(self, num_lanes: int = 64) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count: {num_lanes}")
        self.num_lanes = num_lanes
        self.lanes: List[ScalarInterpreter] = [ScalarInterpreter() for _ in range(num_lanes)]

    def execute_wave(self, instr: VSqrtF16Instruction, exec_mask: int) -> List[int]:
        results: List[int] = []
        for lane_id in range(self.num_lanes):
            lane_active = bool(exec_mask & (1 << lane_id))
            interp = self.lanes[lane_id]
            if lane_active:
                results.append(interp.execute(instr))
            else:
                results.append(interp.read_f16(instr.vdst))
        return results


# =============================================================================
# Unit Tests: Mathematical Oracle & Edge Cases
# =============================================================================

class TestValuSqrtF16MathematicalOracle(unittest.TestCase):
    """Tests covering reference mathematical oracle and standard IEEE 754 binary16 rules."""

    def test_basic_square_roots(self) -> None:
        self.assertEqual(ref_sqrt_f16(ONE_F16), ONE_F16)         # sqrt(1) = 1
        self.assertEqual(ref_sqrt_f16(FOUR_F16), TWO_F16)        # sqrt(4) = 2
        self.assertEqual(ref_sqrt_f16(NINE_F16), THREE_F16)      # sqrt(9) = 3
        self.assertEqual(ref_sqrt_f16(QUARTER_F16), HALF_F16)    # sqrt(0.25) = 0.5

    def test_signed_zeros(self) -> None:
        # sqrt(+0.0) = +0.0, sqrt(-0.0) = -0.0
        self.assertEqual(ref_sqrt_f16(POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(ref_sqrt_f16(NEG_ZERO_F16), NEG_ZERO_F16)

    def test_negative_values_produce_nan(self) -> None:
        # sqrt(negative) = quiet NaN
        self.assertEqual(ref_sqrt_f16(NEG_ONE_F16), QNAN_F16)
        self.assertEqual(ref_sqrt_f16(NEG_INF_F16), QNAN_F16)

    def test_infinities_and_nans(self) -> None:
        self.assertEqual(ref_sqrt_f16(POS_INF_F16), POS_INF_F16)
        self.assertEqual(ref_sqrt_f16(QNAN_F16), QNAN_F16)
        self.assertEqual(ref_sqrt_f16(SNAN_F16), QNAN_F16)

    def test_subnormals(self) -> None:
        # Positive subnormal computes non-zero square root
        root = ref_sqrt_f16(MIN_SUBNORMAL_F16)
        self.assertNotEqual(root, POS_ZERO_F16)
        self.assertFalse(is_nan_f16(root))

    def test_modifiers(self) -> None:
        # abs(|-4.0|) = 4.0; sqrt(4.0) = 2.0
        s0_abs = apply_f16_modifiers(NEG_FOUR_F16, abs_val=True)
        self.assertEqual(ref_sqrt_f16(s0_abs), TWO_F16)

        # neg(-(4.0)) = -4.0; sqrt(-4.0) = NaN
        s0_neg = apply_f16_modifiers(FOUR_F16, neg=True)
        self.assertEqual(ref_sqrt_f16(s0_neg), QNAN_F16)

    def test_clamp_modifier(self) -> None:
        # sqrt(4.0) = 2.0 clamped to [0.0, 1.0] -> 1.0
        self.assertEqual(apply_f16_clamp(TWO_F16), ONE_F16)
        self.assertEqual(apply_f16_clamp(HALF_F16), HALF_F16)
        self.assertEqual(apply_f16_clamp(POS_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(apply_f16_clamp(NEG_ZERO_F16), POS_ZERO_F16)
        self.assertEqual(apply_f16_clamp(QNAN_F16), QNAN_F16)


NEG_FOUR_F16: int = 0xC400
NEG_ONE_F16: int = 0xBC00


# =============================================================================
# Unit Tests: Symbolic AST & Parser
# =============================================================================

class TestValuSqrtF16SymbolicAST(unittest.TestCase):
    """Tests covering AST creation, operand typing, and syntax parsing."""

    def test_instruction_parsing_basic(self) -> None:
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        self.assertEqual(instr.vdst.idx, 1)
        self.assertEqual(instr.src0.val, RegVGPR(0))
        self.assertFalse(instr.src0.neg)
        self.assertFalse(instr.src0.abs_val)
        self.assertFalse(instr.clamp)

    def test_instruction_parsing_modifiers(self) -> None:
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v2, |v3| clamp")
        self.assertEqual(instr.vdst.idx, 2)
        self.assertEqual(instr.src0.val, RegVGPR(3))
        self.assertTrue(instr.src0.abs_val)
        self.assertTrue(instr.clamp)

    def test_instruction_parsing_literal(self) -> None:
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v0, 0x4400")
        self.assertIsInstance(instr.src0.val, ImmF16)
        self.assertEqual(instr.src0.val.bits, FOUR_F16)


# =============================================================================
# Unit Tests: Deterministic Scalar Interpreter
# =============================================================================

class TestValuSqrtF16ScalarInterpreter(unittest.TestCase):
    """Tests covering deterministic register execution in a single thread."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_scalar_execution_normal(self) -> None:
        self.interp.vgprs[0] = FOUR_F16
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        got = self.interp.execute(instr)
        self.assertEqual(got, TWO_F16)
        self.assertEqual(self.interp.read_f16(RegVGPR(1)), TWO_F16)

    def test_scalar_execution_clamp(self) -> None:
        self.interp.vgprs[0] = NINE_F16
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0 clamp")
        got = self.interp.execute(instr)
        self.assertEqual(got, ONE_F16)
        self.assertEqual(self.interp.read_f16(RegVGPR(1)), ONE_F16)


# =============================================================================
# Unit Tests: SIMD Multi-Lane & EXEC Masking
# =============================================================================

class TestValuSqrtF16SIMDDivergence(unittest.TestCase):
    """Tests covering SIMD multi-lane divergence across Wave32 and Wave64."""

    def test_wave32_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=32)
        sentinel = 0xDEAD
        for lane_id in range(32):
            simd.lanes[lane_id].vgprs[0] = FOUR_F16
            simd.lanes[lane_id].vgprs[1] = sentinel

        # Active only on odd lanes
        exec_mask = 0xAAAAAAAA
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(32):
            if lane_id % 2 == 1:
                self.assertEqual(results[lane_id], TWO_F16)
                self.assertEqual(simd.lanes[lane_id].read_f16(RegVGPR(1)), TWO_F16)
            else:
                self.assertEqual(results[lane_id], sentinel)
                self.assertEqual(simd.lanes[lane_id].read_f16(RegVGPR(1)), sentinel)

    def test_wave64_divergence(self) -> None:
        simd = SIMDInterpreter(num_lanes=64)
        sentinel = 0xBEEF
        for lane_id in range(64):
            simd.lanes[lane_id].vgprs[0] = NINE_F16
            simd.lanes[lane_id].vgprs[1] = sentinel

        # Lower 32 lanes active only
        exec_mask = 0x00000000FFFFFFFF
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        results = simd.execute_wave(instr, exec_mask)

        for lane_id in range(64):
            if lane_id < 32:
                self.assertEqual(results[lane_id], THREE_F16)
            else:
                self.assertEqual(results[lane_id], sentinel)


# =============================================================================
# Unit Tests: Register Aliasing & Preservation
# =============================================================================

class TestValuSqrtF16RegisterAliasing(unittest.TestCase):
    """Tests covering destination overwrites and high halfword preservation."""

    def test_in_place_overwrite(self) -> None:
        interp = ScalarInterpreter()
        interp.vgprs[0] = FOUR_F16
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v0, v0")
        got = interp.execute(instr)
        self.assertEqual(got, TWO_F16)
        self.assertEqual(interp.read_f16(RegVGPR(0)), TWO_F16)

    def test_high_halfword_preservation(self) -> None:
        interp = ScalarInterpreter()
        # High 16 bits set to 0x1234
        interp.vgprs[1] = 0x12340000 | ONE_F16
        interp.vgprs[0] = FOUR_F16
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        interp.execute(instr)
        # Low 16 bits should be TWO_F16, high 16 bits should remain 0x1234
        self.assertEqual(interp.vgprs[1], 0x12340000 | TWO_F16)


# =============================================================================
# Unit Tests: Condition Codes & Flags
# =============================================================================

class TestValuSqrtF16ConditionCodes(unittest.TestCase):
    """Tests verifying SCC and VCC flags remain completely isolated."""

    def test_flags_preserved(self) -> None:
        interp = ScalarInterpreter()
        interp.scc = 1
        interp.vcc = 0xDEADBEEFCAFE0000
        interp.vgprs[0] = FOUR_F16
        instr = VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0")
        interp.execute(instr)
        self.assertEqual(interp.scc, 1)
        self.assertEqual(interp.vcc, 0xDEADBEEFCAFE0000)


# =============================================================================
# Unit Tests: Negative Controls & Mutation Detection
# =============================================================================

class TestValuSqrtF16NegativeControls(unittest.TestCase):
    """Negative controls testing syntax rejection and mutation detection."""

    def test_syntax_rejection(self) -> None:
        with self.assertRaises(ValueError):
            VSqrtF16Instruction.parse("v_rsq_f16 v1, v0")

        with self.assertRaises(ValueError):
            VSqrtF16Instruction.parse("v_sqrt_f16 v1, v0 omod:2")

    def test_mutation_detector_negative_zero(self) -> None:
        # Mutation: faulty implementation turning -0.0 into +0.0
        got = ref_sqrt_f16(NEG_ZERO_F16)
        self.assertEqual(got, NEG_ZERO_F16)
        self.assertNotEqual(got, POS_ZERO_F16)

    def test_mutation_detector_negative_value(self) -> None:
        # Mutation: computing sqrt of negative magnitude instead of returning NaN
        got = ref_sqrt_f16(NEG_FOUR_F16)
        self.assertEqual(got, QNAN_F16)
        self.assertTrue(is_nan_f16(got))


# =============================================================================
# Unit Tests: Translator Integration (tools/rdna2spirv.py)
# =============================================================================

class TestValuSqrtF16Translator(unittest.TestCase):
    """End-to-end integration tests exercising tools/rdna2spirv.py."""

    def test_translator_invocation_and_execution(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 0x4400",  # 4.0 in half-precision
            "v_sqrt_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_sqrt_f16_test", asm)
        self.assertIn("Sqrt", spvasm)

        words = support.mrt0_words(spvasm)
        # sqrt(4.0) = 2.0 (0x4000)
        self.assertEqual(words, [TWO_F16, TWO_F16, TWO_F16, TWO_F16])

    def test_translator_one(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("Translator dependencies not available in current environment")

        asm = [
            "v_mov_b32 v0, 0x3C00",  # 1.0 in half-precision
            "v_sqrt_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("ps_sqrt_f16_one", asm)
        words = support.mrt0_words(spvasm)
        # sqrt(1.0) = 1.0 (0x3C00)
        self.assertEqual(words, [ONE_F16, ONE_F16, ONE_F16, ONE_F16])


if __name__ == "__main__":
    unittest.main()
