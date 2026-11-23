"""Translator-backed semantic tests and reference mathematical oracle for RDNA2 v_rcp_f64.

Models AMD RDNA2 / GFX10 double-precision reciprocal:
    v_rcp_f64 v[dst_lo:dst_hi], v[src_lo:src_hi]
    v_rcp_f64_e64 v[dst_lo:dst_hi], [mods] v[src_lo:src_hi] [clamp]

RDNA2 Double-Precision Reciprocal Semantics:
  1. Instruction Encoding:
     VOP1 (32-bit): v_rcp_f64 v[dst:dst+1], v[src:src+1]
     VOP3 (64-bit): v_rcp_f64_e64 v[dst:dst+1], [neg/abs] v[src:src+1] [clamp]
  2. Mathematical Operation:
     dst = 1.0 / src (IEEE 754 binary64 double precision division).
  3. IEEE 754 Rules:
     1.0 / +0.0 = +infinity (0x7FF0000000000000)
     1.0 / -0.0 = -infinity (0xFFF0000000000000)
     1.0 / +infinity = +0.0 (0x0000000000000000)
     1.0 / -infinity = -0.0 (0x8000000000000000)
     1.0 / NaN = NaN
  4. Translator Integration:
     Every test exercises tools/rdna2spirv.py via support.translate_lines(...)
     asserting that translated SPIR-V shaders accurately emit OpCapability Float64
     and OpFDiv %double %constant_1_0 %x.

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


def oracle_rcp_f64(x: float) -> float:
    """Pure mathematical oracle for double precision reciprocal 1.0 / x."""
    if math.isnan(x):
        return float("nan")
    if x == 0.0:
        bits = f64_to_u64(x)
        return float("-inf") if (bits & SIGN_BIT64) else float("inf")
    if math.isinf(x):
        return -0.0 if x < 0.0 else 0.0
    return 1.0 / x


def oracle_rcp_f64_bits(bits: int, clamp: bool = False, neg: bool = False, abs_val: bool = False) -> int:
    """Bit-accurate reference oracle for v_rcp_f64 with VOP3 modifiers."""
    if neg:
        bits ^= SIGN_BIT64
    if abs_val:
        bits &= ~SIGN_BIT64

    f_in = u64_to_f64(bits)
    f_res = oracle_rcp_f64(f_in)

    if clamp:
        if math.isnan(f_res):
            f_res = 0.0
        elif f_res < 0.0:
            f_res = 0.0
        elif f_res > 1.0:
            f_res = 1.0

    return f64_to_u64(f_res)

# ---------------------------------------------------------------------------
# Phase 1: Reference Oracle and Architectural Model Unit Tests
# ---------------------------------------------------------------------------

class TestValuRcpF64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and translator interaction."""

    def test_oracle_rcp_f64_standard_values(self) -> None:
        self.assertEqual(oracle_rcp_f64(1.0), 1.0)
        self.assertEqual(oracle_rcp_f64(2.0), 0.5)
        self.assertEqual(oracle_rcp_f64(4.0), 0.25)
        self.assertEqual(oracle_rcp_f64(0.5), 2.0)
        self.assertEqual(oracle_rcp_f64(-2.0), -0.5)

    def test_oracle_rcp_f64_signed_zeros(self) -> None:
        pos_zero = u64_to_f64(0x0000000000000000)
        neg_zero = u64_to_f64(0x8000000000000000)
        self.assertEqual(oracle_rcp_f64(pos_zero), float("inf"))
        self.assertEqual(oracle_rcp_f64(neg_zero), float("-inf"))

    def test_oracle_rcp_f64_infinities(self) -> None:
        pos_inf = float("inf")
        neg_inf = float("-inf")
        res_pos = oracle_rcp_f64(pos_inf)
        res_neg = oracle_rcp_f64(neg_inf)
        self.assertEqual(f64_to_u64(res_pos), 0x0000000000000000)
        self.assertEqual(f64_to_u64(res_neg), 0x8000000000000000)

    def test_oracle_rcp_f64_nan(self) -> None:
        nan_val = float("nan")
        res = oracle_rcp_f64(nan_val)
        self.assertTrue(math.isnan(res))

    def test_oracle_rcp_f64_modifiers(self) -> None:
        val_2_0 = f64_to_u64(2.0)
        res_neg = oracle_rcp_f64_bits(val_2_0, neg=True)
        self.assertEqual(u64_to_f64(res_neg), -0.5)

        val_neg_2_0 = f64_to_u64(-2.0)
        res_abs = oracle_rcp_f64_bits(val_neg_2_0, abs_val=True)
        self.assertEqual(u64_to_f64(res_abs), 0.5)

        val_0_5 = f64_to_u64(0.5)
        res_clamp = oracle_rcp_f64_bits(val_0_5, clamp=True)
        self.assertEqual(u64_to_f64(res_clamp), 1.0)

    def test_translator_emits_float64_and_fdiv(self) -> None:
        """Verify that support.translate_lines emits OpCapability Float64 and OpFDiv."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",
            "v_rcp_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_basic", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)
        self.assertIn("OpCompositeConstruct %v2uint_d", spv)
        self.assertIn("OpCompositeExtract %uint", spv)

# ---------------------------------------------------------------------------
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class Opcode(str, enum.Enum):
    V_RCP_F64 = "v_rcp_f64"
    V_RCP_F64_E64 = "v_rcp_f64_e64"


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


def VGPR(index: int, count: int = 1) -> Register:
    return Register(RegKind.VGPR, index, count)


def VGPR64(index: int) -> Register:
    return Register(RegKind.VGPR, index, 2)


def SGPR(index: int, count: int = 1) -> Register:
    return Register(RegKind.SGPR, index, count)


def SGPR64(index: int) -> Register:
    return Register(RegKind.SGPR, index, 2)


@dataclasses.dataclass(frozen=True)
class InlineConstant64:
    val: float

    def __str__(self) -> str:
        return str(self.val)


Operand64 = Union[Register, InlineConstant64]


@dataclasses.dataclass
class ValuRcpF64Instruction:
    """AST node for v_rcp_f64 and v_rcp_f64_e64 instructions."""
    opcode: Opcode
    dst: Register
    src: Operand64
    clamp: bool = False
    neg: bool = False
    abs_val: bool = False

    def __post_init__(self) -> None:
        if self.opcode not in (Opcode.V_RCP_F64, Opcode.V_RCP_F64_E64):
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != RegKind.VGPR or self.dst.count != 2:
            raise ValueError(f"Destination must be a 64-bit VGPR pair (count=2), got {self.dst}")
        if isinstance(self.src, Register):
            if self.src.count != 2:
                raise ValueError(f"Register source operand must be 64-bit (count=2), got {self.src}")
        if (self.clamp or self.neg or self.abs_val) and self.opcode != Opcode.V_RCP_F64_E64:
            raise ValueError("Modifiers clamp, neg, and abs require VOP3 e64 encoding")

    @classmethod
    def parse(cls, asm: str) -> ValuRcpF64Instruction:
        raw_tokens = asm.strip().split()
        if not raw_tokens:
            raise ValueError("Empty assembly string")

        op_str = raw_tokens[0]
        if op_str not in (Opcode.V_RCP_F64.value, Opcode.V_RCP_F64_E64.value):
            raise ValueError(f"Expected opcode '{Opcode.V_RCP_F64.value}' or '{Opcode.V_RCP_F64_E64.value}', got '{op_str}'")
        opcode = Opcode(op_str)

        rest = " ".join(raw_tokens[1:]).replace(",", " ").split()
        if len(rest) < 2:
            raise ValueError(f"Too few operands for {op_str}: {asm}")

        def parse_reg(tok: str) -> Register:
            if tok.startswith("v[") and tok.endswith("]"):
                parts = tok[2:-1].split(":")
                return VGPR(int(parts[0]), count=int(parts[1]) - int(parts[0]) + 1)
            elif tok.startswith("s[") and tok.endswith("]"):
                parts = tok[2:-1].split(":")
                return SGPR(int(parts[0]), count=int(parts[1]) - int(parts[0]) + 1)
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


class TestValuRcpF64Ast(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST representation and VOP1/VOP3 encoding."""

    def test_parse_vop1_standard(self) -> None:
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[2:3]")
        self.assertEqual(inst.opcode, Opcode.V_RCP_F64)
        self.assertEqual(inst.dst, VGPR64(0))
        self.assertEqual(inst.src, VGPR64(2))
        self.assertFalse(inst.clamp)
        self.assertFalse(inst.neg)
        self.assertFalse(inst.abs_val)

    def test_parse_vop3_modifiers(self) -> None:
        inst = ValuRcpF64Instruction.parse("v_rcp_f64_e64 v[4:5], -|v[6:7]| clamp")
        self.assertEqual(inst.opcode, Opcode.V_RCP_F64_E64)
        self.assertEqual(inst.dst, VGPR64(4))
        self.assertEqual(inst.src, VGPR64(6))
        self.assertTrue(inst.clamp)
        self.assertTrue(inst.neg)
        self.assertTrue(inst.abs_val)

    def test_parse_inline_constant(self) -> None:
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], 2.0")
        self.assertEqual(inst.src, InlineConstant64(2.0))

    def test_parse_sgpr_source(self) -> None:
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], s[2:3]")
        self.assertEqual(inst.src, SGPR64(2))

    def test_translator_vop3_clamp_emission(self) -> None:
        """Verify translator emits FClamp for clamp modifier in VOP3."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",
            "v_rcp_f64_e64 v[0:1], v[2:3] clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_clamp", lines)
        self.assertIn("FClamp", spv)

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
    """Deterministic single-lane scalar interpreter for v_rcp_f64."""

    def resolve_source(self, src: Operand64, state: ScalarMachineState64) -> int:
        if isinstance(src, InlineConstant64):
            return f64_to_u64(src.val)
        elif isinstance(src, Register):
            if src.kind == RegKind.VGPR:
                return state.get_vgpr64(src.index)
            else:
                return state.get_sgpr64(src.index)
        raise TypeError(f"Unknown operand kind: {type(src)}")

    def step(self, instr: ValuRcpF64Instruction, state: ScalarMachineState64) -> int:
        # Latch source operand
        src_bits = self.resolve_source(instr.src, state)

        # Compute reciprocal with modifiers
        res_bits = oracle_rcp_f64_bits(
            src_bits,
            clamp=instr.clamp,
            neg=instr.neg,
            abs_val=instr.abs_val,
        )

        # Write result to destination VGPR pair
        state.set_vgpr64(instr.dst.index, res_bits)
        return res_bits


class TestValuRcpF64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # Input: 2.0 (0x4000000000000000) in v[2:3]
        state.set_vgpr64(2, f64_to_u64(2.0))
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[2:3]")
        res_bits = interp.step(inst, state)

        # Output in v[0:1] must be 0.5 (0x3FE0000000000000)
        self.assertEqual(res_bits, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_scalar_interpreter_sgpr_source(self) -> None:
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # Input: 4.0 in s[4:5]
        state.set_sgpr64(4, f64_to_u64(4.0))
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], s[4:5]")
        res_bits = interp.step(inst, state)

        # 1.0 / 4.0 = 0.25
        self.assertEqual(res_bits, f64_to_u64(0.25))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.25))

    def test_translator_sgpr_source_translation(self) -> None:
        """Verify support.translate_lines translates SGPR source operand correctly."""
        lines = [
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0x40000000",
            "v_rcp_f64 v[0:1], s[4:5]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_sgpr", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)

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
    """SIMD wavefront execution engine for v_rcp_f64."""

    def resolve_lane_source(self, src: Operand64, lane: int, state: SIMDState64) -> int:
        if isinstance(src, InlineConstant64):
            return f64_to_u64(src.val)
        elif isinstance(src, Register):
            if src.kind == RegKind.VGPR:
                return state.get_vgpr64(lane, src.index)
            else:
                return state.get_sgpr64(src.index)
        raise TypeError(f"Unknown operand kind: {type(src)}")

    def step(self, instr: ValuRcpF64Instruction, state: SIMDState64) -> Dict[int, int]:
        results: Dict[int, int] = {}
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                continue
            src_bits = self.resolve_lane_source(instr.src, lane, state)
            res_bits = oracle_rcp_f64_bits(
                src_bits,
                clamp=instr.clamp,
                neg=instr.neg,
                abs_val=instr.abs_val,
            )
            state.set_vgpr64(lane, instr.dst.index, res_bits)
            results[lane] = res_bits
        return results


class TestValuRcpF64SIMD(unittest.TestCase):
    """Phase 4: Unit tests for SIMD divergence and EXEC masking."""

    def test_simd_wave32_divergence(self) -> None:
        # Alternating active lanes: 0x55555555
        state = SIMDState64(num_lanes=32, exec_mask=0x55555555)
        interp = SIMDInterpreter64()

        # Initialize all lanes with 2.0
        for lane in range(32):
            state.set_vgpr64(lane, 2, f64_to_u64(2.0))
            state.set_vgpr64(lane, 0, 0xDEADBEEFCAFEBABE)

        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[2:3]")
        results = interp.step(inst, state)

        # 16 active lanes
        self.assertEqual(len(results), 16)
        for lane in range(32):
            if lane % 2 == 0:
                # Active: 1.0 / 2.0 = 0.5
                self.assertEqual(state.get_vgpr64(lane, 0), f64_to_u64(0.5))
            else:
                # Inactive: untouched
                self.assertEqual(state.get_vgpr64(lane, 0), 0xDEADBEEFCAFEBABE)

    def test_simd_wave64_divergence(self) -> None:
        # Low 32 lanes active: 0x00000000FFFFFFFF
        state = SIMDState64(num_lanes=64, exec_mask=0x00000000FFFFFFFF)
        interp = SIMDInterpreter64()

        for lane in range(64):
            state.set_vgpr64(lane, 2, f64_to_u64(4.0))
            state.set_vgpr64(lane, 0, 0)

        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[2:3]")
        results = interp.step(inst, state)

        self.assertEqual(len(results), 32)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(state.get_vgpr64(lane, 0), f64_to_u64(0.25))
            else:
                self.assertEqual(state.get_vgpr64(lane, 0), 0)

# ---------------------------------------------------------------------------
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestValuRcpF64Aliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_inplace_dst_equals_src(self) -> None:
        """In-place reciprocal: v_rcp_f64 v[0:1], v[0:1]."""
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # v[0:1] = 2.0
        state.set_vgpr64(0, f64_to_u64(2.0))
        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[0:1]")
        res = interp.step(inst, state)

        # v[0:1] becomes 0.5
        self.assertEqual(res, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_alias_partial_overlap_high_to_low(self) -> None:
        """Partial overlap: vdst is v[0:1], vsrc is v[1:2]."""
        state = ScalarMachineState64()
        interp = ScalarInterpreter64()

        # Setup v[1:2] with 2.0
        val_2_0 = f64_to_u64(2.0)
        state.set_vgpr64(1, val_2_0)

        inst = ValuRcpF64Instruction.parse("v_rcp_f64 v[0:1], v[1:2]")
        res = interp.step(inst, state)

        # Latching source before writeback ensures correct result
        self.assertEqual(res, f64_to_u64(0.5))
        self.assertEqual(state.get_vgpr64(0), f64_to_u64(0.5))

    def test_translator_aliased_inplace_translation(self) -> None:
        """Verify support.translate_lines cleanly translates in-place v_rcp_f64 v[0:1], v[0:1]."""
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x40000000",
            "v_rcp_f64 v[0:1], v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_alias", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)

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


def classify_f64(bits: int) -> F64ClassificationFlags:
    """Classify 64-bit IEEE 754 float result."""
    exp = (bits >> 52) & 0x7FF
    mant = bits & 0x000FFFFFFFFFFFFF
    is_neg = bool(bits & SIGN_BIT64)

    is_zero = (exp == 0 and mant == 0)
    is_subnormal = (exp == 0 and mant != 0)
    is_inf = (exp == 0x7FF and mant == 0)
    is_nan = (exp == 0x7FF and mant != 0)

    return F64ClassificationFlags(
        zero=is_zero,
        negative=is_neg,
        infinite=is_inf,
        nan=is_nan,
        subnormal=is_subnormal,
    )


class TestValuRcpF64StatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes and float classification."""

    def test_status_flags_zero_result(self) -> None:
        flags = classify_f64(f64_to_u64(0.0))
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.infinite)

    def test_status_flags_negative_zero(self) -> None:
        flags = classify_f64(f64_to_u64(-0.0))
        self.assertTrue(flags.zero)
        self.assertTrue(flags.negative)

    def test_status_flags_infinity(self) -> None:
        flags = classify_f64(f64_to_u64(float("inf")))
        self.assertTrue(flags.infinite)
        self.assertFalse(flags.negative)

        flags_neg = classify_f64(f64_to_u64(float("-inf")))
        self.assertTrue(flags_neg.infinite)
        self.assertTrue(flags_neg.negative)

    def test_status_flags_nan(self) -> None:
        flags = classify_f64(f64_to_u64(float("nan")))
        self.assertTrue(flags.nan)

    def test_status_flags_subnormal(self) -> None:
        flags = classify_f64(0x0000000000000001)
        self.assertTrue(flags.subnormal)

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
F64_ONE: int = 0x3FF0000000000000


class TestValuRcpF64BoundaryExtremes(unittest.TestCase):
    """Phase 7: Comprehensive IEEE 754 edge cases for double precision reciprocal."""

    def test_ieee754_signed_zero_division(self) -> None:
        # 1.0 / +0.0 = +infinity
        res_pos = oracle_rcp_f64_bits(F64_POS_ZERO)
        self.assertEqual(res_pos, F64_POS_INF)

        # 1.0 / -0.0 = -infinity
        res_neg = oracle_rcp_f64_bits(F64_NEG_ZERO)
        self.assertEqual(res_neg, F64_NEG_INF)

    def test_ieee754_infinity_division(self) -> None:
        # 1.0 / +inf = +0.0
        res_pos = oracle_rcp_f64_bits(F64_POS_INF)
        self.assertEqual(res_pos, F64_POS_ZERO)

        # 1.0 / -inf = -0.0
        res_neg = oracle_rcp_f64_bits(F64_NEG_INF)
        self.assertEqual(res_neg, F64_NEG_ZERO)

    def test_ieee754_nans(self) -> None:
        # 1.0 / qNaN = NaN
        res_q = oracle_rcp_f64_bits(F64_POS_QNAN)
        self.assertTrue(math.isnan(u64_to_f64(res_q)))

        # 1.0 / sNaN = NaN
        res_s = oracle_rcp_f64_bits(F64_POS_SNAN)
        self.assertTrue(math.isnan(u64_to_f64(res_s)))

    def test_ieee754_subnormals_overflow_to_infinity(self) -> None:
        # Smallest subnormal (approx 4.9e-324): 1.0 / min_subnormal overflows double range -> +inf
        res = oracle_rcp_f64_bits(F64_MIN_SUBNORMAL)
        self.assertEqual(res, F64_POS_INF)

    def test_ieee754_powers_of_two_exactness(self) -> None:
        # Exact reciprocal of powers of two
        for exp in range(-50, 50):
            val = math.pow(2.0, exp)
            expected = math.pow(2.0, -exp)
            res = oracle_rcp_f64(val)
            self.assertEqual(res, expected)

    def test_translator_vop3_modifiers_neg_abs(self) -> None:
        """Verify support.translate_lines emits OpFNegate and FAbs for VOP3 modifiers."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",
            "v_rcp_f64_e64 v[0:1], -|v[2:3]|",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_neg_abs", lines)
        self.assertIn("FAbs", spv)
        self.assertIn("OpFNegate", spv)

if __name__ == "__main__":
    unittest.main()
