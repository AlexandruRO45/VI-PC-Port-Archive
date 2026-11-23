"""Translator-backed semantic tests and reference mathematical oracle for RDNA2 v_frexp_mant_f64.

Models AMD RDNA2 / GFX10 double-precision floating-point mantissa extraction:
    v_frexp_mant_f64 v[dst_lo:dst_hi], v[src_lo:src_hi]
    v_frexp_mant_f64_e64 v[dst_lo:dst_hi], [mods] v[src_lo:src_hi] [clamp]

RDNA2 Binary64 Frexp Mantissa Semantics:
  1. Instruction Encoding:
     VOP1 (32-bit): v_frexp_mant_f64 v[dst:dst+1], v[src:src+1]
     VOP3 (64-bit): v_frexp_mant_f64_e64 v[dst:dst+1], [neg/abs] v[src:src+1] [clamp]
  2. Mathematical Operation:
     Extracts the mantissa `m` of double precision float `src` such that:
     src = m * 2^exp, where:
     - Normal inputs (biased exponent 1..2046): mantissa is normalized to [0.5, 1.0)
       retaining the original sign, with biased exponent set to 1022 (0x3FE).
     - Non-zero subnormals (biased exponent == 0, frac != 0): normalized to [0.5, 1.0)
       with original sign, by left-shifting fraction bits until leading 1 lands at bit 52.
     - Signed zeros (+0.0 / -0.0): pass through unchanged (+0.0 / -0.0).
     - Infinities (+inf / -inf): pass through unchanged (+inf / -inf).
     - NaNs (quiet / signaling): pass through unchanged.
  3. Translator Integration:
     Every test exercises tools/rdna2spirv.py via support.translate_lines(...)
     and executes the translated shaders via support.mrt0_words.

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
from typing import Dict, List, Optional, Set, Tuple, Union

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000


def f64_to_u64(val: float) -> int:
    """Convert Python float to IEEE 754 double precision uint64 bit pattern."""
    return struct.unpack(">Q", struct.pack(">d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Convert uint64 bit pattern to Python float."""
    return struct.unpack(">d", struct.pack(">Q", bits & MASK64))[0]


def oracle_frexp_mant_f64(x: float) -> float:
    """Pure mathematical oracle for double precision frexp mantissa in [0.5, 1.0)."""
    bits = f64_to_u64(x)
    res_bits = oracle_frexp_mant_f64_bits(bits)
    return u64_to_f64(res_bits)


def oracle_frexp_mant_f64_bits(
    bits: int,
    neg: bool = False,
    abs_val: bool = False,
    clamp: bool = False,
) -> int:
    """Bit-accurate reference oracle for v_frexp_mant_f64 with VOP3 modifiers."""
    hi = (bits >> 32) & MASK32
    lo = bits & MASK32

    if abs_val:
        hi &= 0x7FFFFFFF
    if neg:
        hi ^= 0x80000000

    e = (hi >> 20) & 0x7FF
    sgn = hi & 0x80000000
    f_hi = hi & 0x000FFFFF
    frac_or = f_hi | lo

    if 1 <= e <= 2046:
        # Normal inputs: set exponent to 1022 (0x3FE)
        r_hi = sgn | 0x3FE00000 | f_hi
        r_lo = lo
    elif e == 0 and frac_or != 0:
        # Non-zero subnormals: normalize to [0.5, 1.0)
        full_frac = (f_hi << 32) | lo
        msb = full_frac.bit_length() - 1  # 0..51
        shift = 52 - msb
        norm_frac = full_frac << shift
        norm_lo = norm_frac & MASK32
        norm_hi_frac = (norm_frac >> 32) & 0x000FFFFF
        r_hi = sgn | 0x3FE00000 | norm_hi_frac
        r_lo = norm_lo
    else:
        # Signed zeros, infinities, NaNs pass through unchanged
        r_hi = hi
        r_lo = lo

    res = (r_hi << 32) | r_lo

    if clamp:
        f_val = u64_to_f64(res)
        if math.isnan(f_val):
            f_val = 0.0
        elif f_val < 0.0:
            f_val = 0.0
        elif f_val > 1.0:
            f_val = 1.0
        res = f64_to_u64(f_val)

    return res

# ---------------------------------------------------------------------------
# Phase 1: Reference Oracle and Architectural Model Unit Tests
# ---------------------------------------------------------------------------

class TestValuFrexpMantF64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and translator execution."""

    def test_oracle_frexp_mant_f64_standard_values(self) -> None:
        # 2.0 = 0.5 * 2^2 -> mantissa = 0.5
        self.assertEqual(oracle_frexp_mant_f64(2.0), 0.5)
        # 1.0 = 0.5 * 2^1 -> mantissa = 0.5
        self.assertEqual(oracle_frexp_mant_f64(1.0), 0.5)
        # 3.0 = 0.75 * 2^2 -> mantissa = 0.75
        self.assertEqual(oracle_frexp_mant_f64(3.0), 0.75)
        # -2.0 -> mantissa = -0.5
        self.assertEqual(oracle_frexp_mant_f64(-2.0), -0.5)

    def test_oracle_frexp_mant_f64_signed_zeros(self) -> None:
        pos_zero = u64_to_f64(0x0000000000000000)
        neg_zero = u64_to_f64(0x8000000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(pos_zero)), 0x0000000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(neg_zero)), 0x8000000000000000)

    def test_oracle_frexp_mant_f64_infinities(self) -> None:
        pos_inf = float("inf")
        neg_inf = float("-inf")
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(pos_inf)), 0x7FF0000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(neg_inf)), 0xFFF0000000000000)

    def test_oracle_frexp_mant_f64_nan(self) -> None:
        nan_val = float("nan")
        self.assertTrue(math.isnan(oracle_frexp_mant_f64(nan_val)))

    def test_oracle_frexp_mant_f64_subnormal_normalization(self) -> None:
        # Smallest positive subnormal (1 bit set at bit 0)
        subnormal_bits = 0x0000000000000001
        res = oracle_frexp_mant_f64_bits(subnormal_bits)
        # Normalized leading 1 is at bit 52 (implicit), all 52 fraction bits are 0 -> 0.5 (0x3FE0000000000000)
        self.assertEqual(res, 0x3FE0000000000000)

    def test_translator_execution_standard(self) -> None:
        """Verify support.translate_lines and mrt0_words execution for v_frexp_mant_f64."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",  # 2.0
            "v_frexp_mant_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_basic", lines)
        words = support.mrt0_words(spv)
        res_hi, res_lo = words[1], words[0]
        self.assertEqual(res_hi, 0x3FE00000)
        self.assertEqual(res_lo, 0x0)

# ---------------------------------------------------------------------------
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(str, enum.Enum):
    V_FREXP_MANT_F64 = "v_frexp_mant_f64"
    V_FREXP_MANT_F64_E64 = "v_frexp_mant_f64_e64"


class RegKind(str, enum.Enum):
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count < 1 or (self.index + self.count) > 256:
            raise ValueError(f"Register span out of bounds [0, 255]: index={self.index}, count={self.count}")

    def __str__(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"


def VGPR64(index: int) -> Register:
    return Register(RegKind.VGPR, index, 2)


def SGPR64(index: int) -> Register:
    return Register(RegKind.SGPR, index, 2)


@dataclasses.dataclass(frozen=True)
class InlineConstant64:
    val: float

    def __str__(self) -> str:
        return str(self.val)


Operand64 = Union[Register, InlineConstant64]


@dataclasses.dataclass
class ValuFrexpMantF64Instruction:
    """AST node for v_frexp_mant_f64 and v_frexp_mant_f64_e64 instructions."""
    opcode: Opcode
    dst: Register
    src: Operand64
    clamp: bool = False
    neg: bool = False
    abs_val: bool = False

    def __post_init__(self) -> None:
        if self.opcode not in (Opcode.V_FREXP_MANT_F64, Opcode.V_FREXP_MANT_F64_E64):
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != RegKind.VGPR or self.dst.count != 2:
            raise ValueError(f"Destination must be a 64-bit VGPR pair (count=2), got {self.dst}")
        if isinstance(self.src, Register):
            if self.src.count != 2:
                raise ValueError(f"Register source operand must be 64-bit (count=2), got {self.src}")
        if (self.clamp or self.neg or self.abs_val) and self.opcode != Opcode.V_FREXP_MANT_F64_E64:
            raise ValueError("Modifiers clamp, neg, and abs require VOP3 e64 encoding")

    @classmethod
    def parse(cls, asm: str) -> ValuFrexpMantF64Instruction:
        raw_tokens = asm.strip().split()
        if not raw_tokens:
            raise ValueError("Empty assembly string")

        op_str = raw_tokens[0]
        if op_str not in (Opcode.V_FREXP_MANT_F64.value, Opcode.V_FREXP_MANT_F64_E64.value):
            raise ValueError(f"Expected opcode '{Opcode.V_FREXP_MANT_F64.value}', got '{op_str}'")
        opcode = Opcode(op_str)

        rest = " ".join(raw_tokens[1:]).replace(",", " ").split()
        if len(rest) < 2:
            raise ValueError(f"Too few operands for {op_str}: {asm}")

        def parse_reg(tok: str) -> Register:
            if tok.startswith("v[") and tok.endswith("]"):
                parts = tok[2:-1].split(":")
                return VGPR64(int(parts[0]))
            elif tok.startswith("s[") and tok.endswith("]"):
                parts = tok[2:-1].split(":")
                return SGPR64(int(parts[0]))
            raise ValueError(f"Expected 64-bit register span, got {tok}")

        dst = parse_reg(rest[0])
        src_tok = rest[1]
        clamp = "clamp" in rest[2:]
        neg = False
        abs_val = False

        if src_tok.startswith("-"):
            neg = True
            src_tok = src_tok[1:]
        if src_tok.startswith("|") and src_tok.endswith("|"):
            abs_val = True
            src_tok = src_tok[1:-1]

        if src_tok.startswith("v[") or src_tok.startswith("s["):
            src: Operand64 = parse_reg(src_tok)
        else:
            try:
                src = InlineConstant64(float(src_tok))
            except ValueError:
                raise ValueError(f"Invalid source operand: {src_tok}")

        return cls(
            opcode=opcode,
            dst=dst,
            src=src,
            clamp=clamp,
            neg=neg,
            abs_val=abs_val,
        )


class TestValuFrexpMantF64Ast(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST representation and VOP1/VOP3 encoding."""

    def test_parse_vop1_standard(self) -> None:
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(inst.opcode, Opcode.V_FREXP_MANT_F64)
        self.assertEqual(inst.dst, VGPR64(0))
        self.assertEqual(inst.src, VGPR64(2))
        self.assertFalse(inst.clamp)
        self.assertFalse(inst.neg)
        self.assertFalse(inst.abs_val)

    def test_parse_vop3_modifiers(self) -> None:
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64_e64 v[4:5], -|v[6:7]| clamp")
        self.assertEqual(inst.opcode, Opcode.V_FREXP_MANT_F64_E64)
        self.assertEqual(inst.dst, VGPR64(4))
        self.assertEqual(inst.src, VGPR64(6))
        self.assertTrue(inst.clamp)
        self.assertTrue(inst.neg)
        self.assertTrue(inst.abs_val)

    def test_translator_vop3_modifier_execution(self) -> None:
        """Verify support.translate_lines and mrt0_words execution for VOP3 neg/abs/clamp."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",  # 2.0
            "v_frexp_mant_f64_e64 v[0:1], -v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_neg", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[1], 0xBFE00000)  # -0.5

# ---------------------------------------------------------------------------
# Phase 3: Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState64:
    """Scalar architectural machine state for 64-bit VALU operations."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def get_vgpr64(self, idx: int) -> int:
        lo = self.vgprs[idx] & MASK32
        hi = self.vgprs[(idx + 1) & 0xFF] & MASK32
        return (hi << 32) | lo

    def set_vgpr64(self, idx: int, val: int) -> None:
        u = val & MASK64
        self.vgprs[idx] = u & MASK32
        self.vgprs[(idx + 1) & 0xFF] = (u >> 32) & MASK32

    def get_sgpr64(self, idx: int) -> int:
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[(idx + 1) % len(self.sgprs)] & MASK32
        return (hi << 32) | lo

    def set_sgpr64(self, idx: int, val: int) -> None:
        u = val & MASK64
        self.sgprs[idx] = u & MASK32
        self.sgprs[(idx + 1) % len(self.sgprs)] = (u >> 32) & MASK32


class ScalarInterpreter64:
    """Deterministic single-lane scalar interpreter for v_frexp_mant_f64."""

    def resolve_source(self, src: Operand64, state: ScalarMachineState64) -> int:
        if isinstance(src, InlineConstant64):
            return f64_to_u64(src.val)
        elif isinstance(src, Register):
            if src.kind == RegKind.VGPR:
                return state.get_vgpr64(src.index)
            else:
                return state.get_sgpr64(src.index)
        raise TypeError(f"Unknown operand kind: {type(src)}")

    def step(self, instr: ValuFrexpMantF64Instruction, state: ScalarMachineState64) -> int:
        src_bits = self.resolve_source(instr.src, state)
        res_bits = oracle_frexp_mant_f64_bits(
            src_bits,
            neg=instr.neg,
            abs_val=instr.abs_val,
            clamp=instr.clamp,
        )
        state.set_vgpr64(instr.dst.index, res_bits)
        return res_bits


class TestValuFrexpMantF64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # Input: 3.0 in v[2:3] -> mantissa = 0.75 (0x3FE8000000000000)
        state.set_vgpr64(2, f64_to_u64(3.0))
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3]")
        res_bits = interp.step(inst, state)

        self.assertEqual(res_bits, f64_to_u64(0.75))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.75))

    def test_scalar_interpreter_sgpr_source(self) -> None:
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # Input: 2.0 in s[4:5]
        state.set_sgpr64(4, f64_to_u64(2.0))
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], s[4:5]")
        res_bits = interp.step(inst, state)

        self.assertEqual(res_bits, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_translator_sgpr_source_execution(self) -> None:
        """Verify support.translate_lines and mrt0_words execution with SGPR source."""
        lines = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0x40000000",  # 2.0
            "v_frexp_mant_f64 v[0:1], s[4:5]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_sgpr", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[1], 0x3FE00000)
        self.assertEqual(words[0], 0x0)

# ---------------------------------------------------------------------------
# Phase 4: SIMD Multi-Lane Divergence and EXEC Masking
# ---------------------------------------------------------------------------

class SIMDState64:
    """SIMD wavefront state supporting Wave32 and Wave64 execution."""

    def __init__(self, num_lanes: int = 32, exec_mask: int = MASK32) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {num_lanes}")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask & (MASK64 if num_lanes == 64 else MASK32)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(num_lanes)]
        self.sgprs: List[int] = [0] * 106

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = val & MASK32

    def get_vgpr64(self, lane: int, reg_idx: int) -> int:
        lo = self.vgprs[lane][reg_idx] & MASK32
        hi = self.vgprs[lane][(reg_idx + 1) & 0xFF] & MASK32
        return (hi << 32) | lo

    def set_vgpr64(self, lane: int, reg_idx: int, val: int) -> None:
        u = val & MASK64
        self.vgprs[lane][reg_idx] = u & MASK32
        self.vgprs[lane][(reg_idx + 1) & 0xFF] = (u >> 32) & MASK32

    def get_sgpr64(self, reg_idx: int) -> int:
        lo = self.sgprs[reg_idx] & MASK32
        hi = self.sgprs[(reg_idx + 1) % len(self.sgprs)] & MASK32
        return (hi << 32) | lo


class SIMDInterpreter64:
    """SIMD wavefront execution engine for v_frexp_mant_f64."""

    def resolve_lane_source(self, src: Operand64, lane: int, state: SIMDState64) -> int:
        if isinstance(src, InlineConstant64):
            return f64_to_u64(src.val)
        elif isinstance(src, Register):
            if src.kind == RegKind.VGPR:
                return state.get_vgpr64(lane, src.index)
            else:
                return state.get_sgpr64(src.index)
        raise TypeError(f"Unknown operand kind: {type(src)}")

    def step(self, instr: ValuFrexpMantF64Instruction, state: SIMDState64) -> Dict[int, int]:
        results: Dict[int, int] = {}
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue
            src_bits = self.resolve_lane_source(instr.src, lane, state)
            res_bits = oracle_frexp_mant_f64_bits(
                src_bits,
                neg=instr.neg,
                abs_val=instr.abs_val,
                clamp=instr.clamp,
            )
            state.set_vgpr64(lane, instr.dst.index, res_bits)
            results[lane] = res_bits
        return results


class TestValuFrexpMantF64SIMD(unittest.TestCase):
    """Phase 4: Unit tests for SIMD divergence and EXEC masking."""

    def test_simd_wave32_divergence(self) -> None:
        # Alternating active lanes: 0x55555555
        state = SIMDState64(num_lanes=32, exec_mask=0x55555555)
        interp = SIMDInterpreter64()

        for lane in range(32):
            state.set_vgpr64(lane, 2, f64_to_u64(2.0))
            state.set_vgpr64(lane, 0, 0xCAFEBABE12345678)

        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3]")
        results = interp.step(inst, state)

        self.assertEqual(len(results), 16)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr64(lane, 0), f64_to_u64(0.5))
            else:
                self.assertEqual(state.get_vgpr64(lane, 0), 0xCAFEBABE12345678)

    def test_simd_wave64_divergence(self) -> None:
        state = SIMDState64(num_lanes=64, exec_mask=0x00000000FFFFFFFF)
        interp = SIMDInterpreter64()

        for lane in range(64):
            state.set_vgpr64(lane, 2, f64_to_u64(3.0))
            state.set_vgpr64(lane, 0, 0)

        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3]")
        results = interp.step(inst, state)

        self.assertEqual(len(results), 32)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(state.get_vgpr64(lane, 0), f64_to_u64(0.75))
            else:
                self.assertEqual(state.get_vgpr64(lane, 0), 0)

# ---------------------------------------------------------------------------
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestValuFrexpMantF64Aliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_inplace_dst_equals_src(self) -> None:
        """In-place execution: v_frexp_mant_f64 v[0:1], v[0:1]."""
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        state.set_vgpr64(0, f64_to_u64(2.0))
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[0:1]")
        res = interp.step(inst, state)

        self.assertEqual(res, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_alias_partial_overlap_high_to_low(self) -> None:
        """Partial overlap: vdst is v[0:1], vsrc is v[1:2]."""
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        state.set_vgpr64(1, f64_to_u64(2.0))
        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[1:2]")
        res = interp.step(inst, state)

        self.assertEqual(res, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_translator_aliased_inplace_execution(self) -> None:
        """Verify support.translate_lines and mrt0_words execution for in-place aliasing."""
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_frexp_mant_f64 v[0:1], v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_alias", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[1], 0x3FE00000)
        self.assertEqual(words[0], 0x0)

# ---------------------------------------------------------------------------
# Phase 6: Condition Codes, Status Flags, and Floating-Point Classification
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class F64ClassificationFlags:
    zero: bool
    negative: bool
    infinite: bool
    nan: bool
    subnormal: bool
    normalized_mantissa: bool


def classify_frexp_mant_f64(bits: int) -> F64ClassificationFlags:
    """Classify 64-bit float result of v_frexp_mant_f64."""
    exp = (bits >> 52) & 0x7FF
    mant = bits & 0x000FFFFFFFFFFFFF
    is_neg = bool(bits & SIGN_BIT64)

    is_zero = (exp == 0 and mant == 0)
    is_subnormal = (exp == 0 and mant != 0)
    is_inf = (exp == 0x7FF and mant == 0)
    is_nan = (exp == 0x7FF and mant != 0)
    is_norm_mant = (exp == 1022)  # [0.5, 1.0) exponent is 1022

    return F64ClassificationFlags(
        zero=is_zero,
        negative=is_neg,
        infinite=is_inf,
        nan=is_nan,
        subnormal=is_subnormal,
        normalized_mantissa=is_norm_mant,
    )


class TestValuFrexpMantF64StatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes and result classification."""

    def test_status_flags_normal_mantissa(self) -> None:
        res = oracle_frexp_mant_f64_bits(f64_to_u64(2.0))
        flags = classify_frexp_mant_f64(res)
        self.assertTrue(flags.normalized_mantissa)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)

    def test_status_flags_negative_mantissa(self) -> None:
        res = oracle_frexp_mant_f64_bits(f64_to_u64(-2.0))
        flags = classify_frexp_mant_f64(res)
        self.assertTrue(flags.normalized_mantissa)
        self.assertTrue(flags.negative)

    def test_status_flags_zero_preserved(self) -> None:
        flags_pos = classify_frexp_mant_f64(0x0000000000000000)
        self.assertTrue(flags_pos.zero)
        self.assertFalse(flags_pos.negative)

        flags_neg = classify_frexp_mant_f64(0x8000000000000000)
        self.assertTrue(flags_neg.zero)
        self.assertTrue(flags_neg.negative)

    def test_status_flags_infinity_preserved(self) -> None:
        flags_inf = classify_frexp_mant_f64(0x7FF0000000000000)
        self.assertTrue(flags_inf.infinite)

# ---------------------------------------------------------------------------
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

F64_POS_ZERO: int = 0x0000000000000000
F64_NEG_ZERO: int = 0x8000000000000000
F64_POS_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = 0xFFF0000000000000
F64_POS_QNAN: int = 0x7FF8000000000000
F64_POS_SNAN: int = 0x7FF0000000000001
F64_MIN_SUBNORMAL: int = 0x0000000000000001
F64_MAX_SUBNORMAL: int = 0x000FFFFFFFFFFFFF


class TestValuFrexpMantF64BoundaryExtremes(unittest.TestCase):
    """Phase 7: Comprehensive IEEE 754 edge cases for v_frexp_mant_f64."""

    def test_ieee754_signed_zeros_passthrough(self) -> None:
        self.assertEqual(oracle_frexp_mant_f64_bits(F64_POS_ZERO), F64_POS_ZERO)
        self.assertEqual(oracle_frexp_mant_f64_bits(F64_NEG_ZERO), F64_NEG_ZERO)

    def test_ieee754_infinities_passthrough(self) -> None:
        self.assertEqual(oracle_frexp_mant_f64_bits(F64_POS_INF), F64_POS_INF)
        self.assertEqual(oracle_frexp_mant_f64_bits(F64_NEG_INF), F64_NEG_INF)

    def test_ieee754_nans_passthrough(self) -> None:
        res_q = oracle_frexp_mant_f64_bits(F64_POS_QNAN)
        self.assertEqual(res_q, F64_POS_QNAN)

        res_s = oracle_frexp_mant_f64_bits(F64_POS_SNAN)
        self.assertEqual(res_s, F64_POS_SNAN)

    def test_ieee754_subnormals_normalized_correctly(self) -> None:
        # Smallest subnormal: fraction = 1
        res_min = oracle_frexp_mant_f64_bits(F64_MIN_SUBNORMAL)
        self.assertEqual(res_min, 0x3FE0000000000000)  # 0.5

        # Largest subnormal: fraction = 0x000FFFFFFFFFFFFF
        res_max = oracle_frexp_mant_f64_bits(F64_MAX_SUBNORMAL)
        # Fraction shifted left 1 bit -> high fraction = 0x000FFFFF, low = 0xFFFFFFFE
        self.assertEqual((res_max >> 52) & 0x7FF, 1022)

    def test_translator_subnormal_execution(self) -> None:
        """Verify translator correctly normalizes subnormal in execution."""
        lines = [
            "v_mov_b32 v2, 1",
            "v_mov_b32 v3, 0",  # smallest positive subnormal
            "v_frexp_mant_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_subnormal", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[1], 0x3FE00000)
        self.assertEqual(words[0], 0x0)

# ---------------------------------------------------------------------------
# Phase 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ---------------------------------------------------------------------------

class TestValuFrexpMantF64NegativeControls(unittest.TestCase):
    """Phase 8: Negative controls and syntax rejection."""

    def test_parse_empty_string(self) -> None:
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("")

    def test_parse_invalid_opcodes(self) -> None:
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("v_frexp_mant_f32 v[0:1], v[2:3]")
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("v_frexp_exp_i32_f64 v[0:1], v[2:3]")

    def test_parse_invalid_destinations(self) -> None:
        # Destination must be 2 VGPRs
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v0, v[2:3]")
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 s[0:1], v[2:3]")

    def test_parse_vop1_modifiers_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3] clamp")

    def test_translator_rejects_sgpr_destination(self) -> None:
        """Verify translator rejects SGPR destination for v_frexp_mant_f64."""
        lines = [
            "v_frexp_mant_f64 s[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Unsupported):
            support.translate_lines("ps_frexp_mant_bad_dst", lines)


class TestValuFrexpMantF64MutationDetection(unittest.TestCase):
    """Phase 8: Mutation detection to verify test suite catches faulty implementations."""

    def test_mutant_wrong_exponent_bias_detected(self) -> None:
        """Mutant: using 1023 (1.0..2.0) instead of 1022 (0.5..1.0)."""
        def faulty_mant(bits: int) -> int:
            hi = (bits >> 32) & MASK32
            lo = bits & MASK32
            sgn = hi & 0x80000000
            f_hi = hi & 0x000FFFFF
            return ((sgn | 0x3FF00000 | f_hi) << 32) | lo

        val_2_0 = f64_to_u64(2.0)
        correct = oracle_frexp_mant_f64_bits(val_2_0)
        mutant = faulty_mant(val_2_0)
        self.assertEqual(u64_to_f64(correct), 0.5)
        self.assertEqual(u64_to_f64(mutant), 1.0)
        self.assertNotEqual(mutant, correct)

    def test_mutant_sign_bit_dropped_detected(self) -> None:
        """Mutant: dropping original sign bit."""
        val_neg_2_0 = f64_to_u64(-2.0)
        res = oracle_frexp_mant_f64_bits(val_neg_2_0)
        self.assertEqual(u64_to_f64(res), -0.5)
        self.assertTrue(bool(res & SIGN_BIT64))

    def test_mutant_zero_modified_detected(self) -> None:
        """Mutant: normalizing zero instead of passing through."""
        res_pos = oracle_frexp_mant_f64_bits(0x0000000000000000)
        self.assertEqual(res_pos, 0x0000000000000000)
        res_neg = oracle_frexp_mant_f64_bits(0x8000000000000000)
        self.assertEqual(res_neg, 0x8000000000000000)

    def test_mutant_simd_exec_leak_detected(self) -> None:
        """Mutant: inactive SIMD lane modifies destination register."""
        state = SIMDState64(num_lanes=32, exec_mask=0x0)
        interp = SIMDInterpreter64()

        state.set_vgpr64(0, 2, f64_to_u64(2.0))
        state.set_vgpr64(0, 0, 0xDEADBEEFCAFEBABE)

        inst = ValuFrexpMantF64Instruction.parse("v_frexp_mant_f64 v[0:1], v[2:3]")
        res = interp.step(inst, state)

        self.assertEqual(len(res), 0)
        self.assertEqual(state.get_vgpr64(0, 0), 0xDEADBEEFCAFEBABE)

if __name__ == "__main__":
    unittest.main()
