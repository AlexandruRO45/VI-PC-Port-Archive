"""Semantic interpreter tests and reference hardware-matching oracle for RDNA2 s_orn2_b32.

Models AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP2 32-bit bitwise OR-NOT2 operation:
  s_orn2_b32 sdst, ssrc0, ssrc1

RDNA2 SALU Bitwise Architecture & Semantics:
  1. Instruction Operands:
     - sdst: Destination SGPR receiving the 32-bit bitwise result.
     - ssrc0: First 32-bit scalar source operand (SGPR, literal, or inline constant).
     - ssrc1: Second 32-bit scalar source operand (SGPR, literal, or inline constant).
  2. Bitwise Logic:
     sdst = (ssrc0 | (~ssrc1)) & 0xFFFFFFFF
     Note that ssrc1 is bitwise inverted (NOT) prior to the bitwise OR with ssrc0.
  3. Scalar Condition Code (SCC):
     SCC is updated based on the 32-bit result:
     SCC = 1 if (sdst != 0) else 0.
  4. Non-commutativity & Algebraic Properties:
     - Non-commutative: (A | ~B) != (B | ~A) in general.
     - Self-identity: (A | ~A) == 0xFFFFFFFF, with SCC = 1 for all A.
     - Zero-absorption: (0 | ~B) == ~B.
     - Maximum element: (A | ~0) == 0xFFFFFFFF; (A | ~0xFFFFFFFF) == A.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import spirv_mini_interp as si
    import support
    TRANSLATOR_ERROR = None
except Exception as _err:
    rdna2spirv = None  # type: ignore
    si = None  # type: ignore
    support = None  # type: ignore
    TRANSLATOR_ERROR = str(_err)

MASK32: int = 0xFFFFFFFF

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Pure bitwise reference oracle for s_orn2_b32: D = S0 | ~S1, SCC = (D != 0).

    Returns tuple (sdst, scc) normalized to 32-bit unsigned integers.
    """
    s0_u = u32(s0)
    s1_u = u32(s1)
    res = u32(s0_u | (~s1_u))
    scc = 1 if res != 0 else 0
    return res, scc


class TestSaluOrn2B32Oracle(unittest.TestCase):
    """Unit tests verifying the reference mathematical oracle against RDNA2 hardware semantics."""

    def test_oracle_truth_table_basic(self) -> None:
        # 0 | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0x00000000, 0x00000000), (0xFFFFFFFF, 1))
        # 0 | ~0xFFFFFFFF = 0x00000000, scc = 0 (only way to produce 0)
        self.assertEqual(ref_orn2_b32(0x00000000, 0xFFFFFFFF), (0x00000000, 0))
        # 0xFFFFFFFF | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0x00000000), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF | ~0xFFFFFFFF = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 1))

    def test_oracle_self_identity(self) -> None:
        # For any word X, X | ~X == 0xFFFFFFFF, SCC == 1
        test_vectors = [
            0x00000000, 0xFFFFFFFF, 0x12345678, 0xAAAAAAAA,
            0x55555555, 0x80000000, 0x7FFFFFFF, 0x00000001,
            0xCAFEBABE, 0xDEADBEEF,
        ]
        for val in test_vectors:
            res, scc = ref_orn2_b32(val, val)
            self.assertEqual(res, 0xFFFFFFFF, f"Self-identity failed for {val:#x}")
            self.assertEqual(scc, 1, f"SCC failed for {val:#x}")

    def test_oracle_non_commutativity(self) -> None:
        # A | ~B != B | ~A when A != B
        a = 0x0F0F0F0F
        b = 0x33333333
        res_ab, _ = ref_orn2_b32(a, b)  # 0x0F0F0F0F | 0xCCCCCCCC = 0xCFCFCFCF
        res_ba, _ = ref_orn2_b32(b, a)  # 0x33333333 | 0xF0F0F0F0 = 0xF3F3F3F3
        self.assertEqual(res_ab, 0xCFCFCFCF)
        self.assertEqual(res_ba, 0xF3F3F3F3)
        self.assertNotEqual(res_ab, res_ba)

    def test_oracle_bit_decomposition(self) -> None:
        # Check all 32 independent bit positions
        for bit in range(32):
            mask = 1 << bit
            # s0=0, s1=0: bit in ~s1 is 1 -> res has 1
            res, _ = ref_orn2_b32(0, 0)
            self.assertTrue(bool(res & mask))
            # s0=0, s1=mask: bit in ~s1 is 0 -> res has 0
            res, _ = ref_orn2_b32(0, mask)
            self.assertFalse(bool(res & mask))
            # s0=mask, s1=mask: bit in s0 is 1 -> res has 1
            res, _ = ref_orn2_b32(mask, mask)
            self.assertTrue(bool(res & mask))
            # s0=mask, s1=0: both are 1 -> res has 1
            res, _ = ref_orn2_b32(mask, 0)
            self.assertTrue(bool(res & mask))

    def test_oracle_exhaustive_nibbles(self) -> None:
        # Exhaustive verification across all 16x16 4-bit nibbles
        for n0 in range(16):
            for n1 in range(16):
                expected_nibble = (n0 | (~n1 & 0xF)) & 0xF
                expected_res = sum(expected_nibble << (shift * 4) for shift in range(8))
                res, _ = ref_orn2_b32(
                    sum(n0 << (shift * 4) for shift in range(8)),
                    sum(n1 << (shift * 4) for shift in range(8)),
                )
                self.assertEqual(res, expected_res)

class RegKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    LITERAL = "literal"
    INLINE_INT = "inline_int"


@dataclasses.dataclass(frozen=True)
class RegOperand:
    """Representation of an RDNA2 register or constant operand."""
    kind: RegKind
    index_or_val: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.SGPR:
            if not (0 <= self.index_or_val <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.index_or_val}")
        elif self.kind == RegKind.VGPR:
            if not (0 <= self.index_or_val <= 255):
                raise ValueError(f"VGPR index out of range [0, 255]: {self.index_or_val}")

    def format_asm(self) -> str:
        if self.kind == RegKind.SGPR:
            return f"s{self.index_or_val}"
        elif self.kind == RegKind.VGPR:
            return f"v{self.index_or_val}"
        elif self.kind == RegKind.LITERAL:
            return f"{u32(self.index_or_val):#x}"
        elif self.kind == RegKind.INLINE_INT:
            return str(self.index_or_val)
        return str(self.index_or_val)


class Opcode(enum.Enum):
    S_ORN2_B32 = "s_orn2_b32"
    S_MOV_B32 = "s_mov_b32"


@dataclasses.dataclass(frozen=True)
class SALUInstruction:
    """AST node representing a Scalar ALU instruction."""
    opcode: Opcode
    sdst: RegOperand
    ssrc0: RegOperand
    ssrc1: RegOperand | None = None

    def __post_init__(self) -> None:
        if self.sdst.kind != RegKind.SGPR:
            raise ValueError("Destination of SALU instruction must be an SGPR")
        if self.opcode == Opcode.S_ORN2_B32:
            if self.ssrc1 is None:
                raise ValueError("s_orn2_b32 requires two source operands")
            literals = sum(1 for op in (self.ssrc0, self.ssrc1) if op is not None and op.kind == RegKind.LITERAL)
            if literals > 1:
                raise ValueError("SOP2 encoding cannot have more than one literal pool")

    def format_asm(self) -> str:
        if self.opcode == Opcode.S_ORN2_B32:
            assert self.ssrc1 is not None
            return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}, {self.ssrc1.format_asm()}"
        elif self.opcode == Opcode.S_MOV_B32:
            return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}"
        return f"{self.opcode.value} {self.sdst.format_asm()}"


class TestSaluOrn2B32AST(unittest.TestCase):
    """Unit tests for AST definitions and instruction formatting."""

    def test_ast_reg_formatting(self) -> None:
        self.assertEqual(RegOperand(RegKind.SGPR, 5).format_asm(), "s5")
        self.assertEqual(RegOperand(RegKind.VGPR, 12).format_asm(), "v12")
        self.assertEqual(RegOperand(RegKind.LITERAL, 0x12345678).format_asm(), "0x12345678")
        self.assertEqual(RegOperand(RegKind.INLINE_INT, 0).format_asm(), "0")

    def test_ast_instruction_formatting(self) -> None:
        inst = SALUInstruction(
            opcode=Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.assertEqual(inst.format_asm(), "s_orn2_b32 s0, s1, s2")

    def test_ast_instruction_with_literal(self) -> None:
        inst = SALUInstruction(
            opcode=Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 4),
            ssrc0=RegOperand(RegKind.SGPR, 5),
            ssrc1=RegOperand(RegKind.LITERAL, 0xCAFEBABE),
        )
        self.assertEqual(inst.format_asm(), "s_orn2_b32 s4, s5, 0xcafebabe")

    def test_ast_validation_errors(self) -> None:
        # Invalid SGPR index
        with self.assertRaises(ValueError):
            RegOperand(RegKind.SGPR, 106)
        with self.assertRaises(ValueError):
            RegOperand(RegKind.SGPR, -1)
        # SALU destination cannot be VGPR
        with self.assertRaises(ValueError):
            SALUInstruction(
                opcode=Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.VGPR, 0),
                ssrc0=RegOperand(RegKind.SGPR, 1),
                ssrc1=RegOperand(RegKind.SGPR, 2),
            )
        # Two literals in one SOP2 instruction rejected
        with self.assertRaises(ValueError):
            SALUInstruction(
                opcode=Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.SGPR, 0),
                ssrc0=RegOperand(RegKind.LITERAL, 0x1234),
                ssrc1=RegOperand(RegKind.LITERAL, 0x5678),
            )

class ScalarSALUInterpreter:
    """Deterministic architectural scalar instruction interpreter for SALU state."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0

    def read_operand(self, op: RegOperand) -> int:
        if op.kind == RegKind.SGPR:
            return self.sgprs[op.index_or_val]
        elif op.kind in (RegKind.LITERAL, RegKind.INLINE_INT):
            return u32(op.index_or_val)
        raise ValueError(f"Cannot read operand kind {op.kind} as scalar")

    def write_sgpr(self, reg_idx: int, val: int) -> None:
        self.sgprs[reg_idx] = u32(val)

    def execute(self, inst: SALUInstruction) -> tuple[int, int]:
        """Execute a single SALU instruction, updating state and returning (result, scc)."""
        if inst.opcode == Opcode.S_MOV_B32:
            val = self.read_operand(inst.ssrc0)
            self.write_sgpr(inst.sdst.index_or_val, val)
            return val, self.scc
        elif inst.opcode == Opcode.S_ORN2_B32:
            assert inst.ssrc1 is not None
            s0 = self.read_operand(inst.ssrc0)
            s1 = self.read_operand(inst.ssrc1)
            res, scc = ref_orn2_b32(s0, s1)
            self.write_sgpr(inst.sdst.index_or_val, res)
            self.scc = scc
            return res, scc
        raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")


class TestSaluOrn2B32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_interp_single_execution(self) -> None:
        self.interp.write_sgpr(1, 0x11110000)
        self.interp.write_sgpr(2, 0x00FF00FF)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, scc = self.interp.execute(inst)
        expected_res = 0x11110000 | (~0x00FF00FF & MASK32)
        self.assertEqual(res, expected_res)
        self.assertEqual(self.interp.sgprs[0], expected_res)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.scc, 1)

    def test_interp_sequential_chain(self) -> None:
        self.interp.write_sgpr(1, 0xAAAAAAAA)
        self.interp.write_sgpr(2, 0x55555555)
        # s3 = s1 | ~s2
        inst1 = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 3),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.interp.execute(inst1)
        self.assertEqual(self.interp.sgprs[3], 0xAAAAAAAA)
        # s4 = s3 | ~s1 (0xAAAAAAAA | ~0xAAAAAAAA = 0xFFFFFFFF)
        inst2 = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 4),
            ssrc0=RegOperand(RegKind.SGPR, 3),
            ssrc1=RegOperand(RegKind.SGPR, 1),
        )
        self.interp.execute(inst2)
        self.assertEqual(self.interp.sgprs[4], 0xFFFFFFFF)
        self.assertEqual(self.interp.scc, 1)

    def test_interp_unmodified_registers(self) -> None:
        self.interp.write_sgpr(10, 0x12345678)
        self.interp.write_sgpr(11, 0x9ABCDEF0)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.interp.execute(inst)
        self.assertEqual(self.interp.sgprs[10], 0x12345678)
        self.assertEqual(self.interp.sgprs[11], 0x9ABCDEF0)

class SIMDWavefrontContext:
    """Context modeling SIMD wavefront interaction with scalar SALU lane masks."""

    def __init__(self, wave_size: int = 32) -> None:
        assert wave_size in (32, 64)
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.salu: ScalarSALUInterpreter = ScalarSALUInterpreter()
        self.lane_vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]

    def step_salu(self, inst: SALUInstruction) -> tuple[int, int]:
        return self.salu.execute(inst)

    def apply_lane_mask_to_exec(self, mask_reg_idx: int) -> None:
        mask = self.salu.sgprs[mask_reg_idx]
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def execute_vector_add_masked(self, vdst: int, vsrc0: int, vsrc1: int) -> None:
        """Simulates a vector instruction active only on lanes where exec_mask bit is 1."""
        for lane in range(self.wave_size):
            if bool(self.exec_mask & (1 << lane)):
                res = (self.lane_vgprs[lane][vsrc0] + self.lane_vgprs[lane][vsrc1]) & MASK32
                self.lane_vgprs[lane][vdst] = res


class TestSaluOrn2B32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD divergence, lane mask inversion, and wave execution masking."""

    def test_wave32_divergent_branch_mask_combination(self) -> None:
        wave = SIMDWavefrontContext(wave_size=32)
        active_mask = 0b11110000111100001111000011110000
        taken_mask =  0b00110011001100110011001100110011
        wave.salu.write_sgpr(1, active_mask)
        wave.salu.write_sgpr(2, taken_mask)

        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, scc = wave.step_salu(inst)
        expected = (active_mask | (~taken_mask & MASK32)) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(scc, 1)

        wave.apply_lane_mask_to_exec(0)
        self.assertEqual(wave.exec_mask, expected)

    def test_wave_vector_isolation_under_orn2_mask(self) -> None:
        wave = SIMDWavefrontContext(wave_size=32)
        for lane in range(32):
            wave.lane_vgprs[lane][1] = 10
            wave.lane_vgprs[lane][2] = 20
            wave.lane_vgprs[lane][0] = 0xCAFE

        s0 = 0xFFFFFFFE
        s1 = 0x00000001
        wave.salu.write_sgpr(1, s0)
        wave.salu.write_sgpr(2, s1)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, scc = wave.step_salu(inst)
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

        wave.apply_lane_mask_to_exec(0)
        wave.execute_vector_add_masked(vdst=0, vsrc0=1, vsrc1=2)

        self.assertEqual(wave.lane_vgprs[0][0], 0xCAFE)
        for lane in range(1, 32):
            self.assertEqual(wave.lane_vgprs[lane][0], 30)

    def test_uniform_scalar_broadcast(self) -> None:
        wave = SIMDWavefrontContext(wave_size=32)
        wave.salu.write_sgpr(1, 0x12345678)
        wave.salu.write_sgpr(2, 0x9ABCDEF0)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 5),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, _ = wave.step_salu(inst)
        for lane in range(32):
            self.assertEqual(wave.salu.sgprs[5], res)

class TestSaluOrn2B32RegisterAliasing(unittest.TestCase):
    """Unit tests for in-place register aliasing and operand reuse."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_alias_dst_equals_src0(self) -> None:
        self.interp.write_sgpr(0, 0x00FF0000)
        self.interp.write_sgpr(1, 0xFF00FF00)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 0),
            ssrc1=RegOperand(RegKind.SGPR, 1),
        )
        res, scc = self.interp.execute(inst)
        expected = 0x00FF0000 | (~0xFF00FF00 & MASK32)
        self.assertEqual(res, expected)
        self.assertEqual(self.interp.sgprs[0], expected)
        self.assertEqual(self.interp.sgprs[1], 0xFF00FF00)

    def test_alias_dst_equals_src1(self) -> None:
        self.interp.write_sgpr(0, 0x12340000)
        self.interp.write_sgpr(1, 0x00005678)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 1),
            ssrc0=RegOperand(RegKind.SGPR, 0),
            ssrc1=RegOperand(RegKind.SGPR, 1),
        )
        res, scc = self.interp.execute(inst)
        expected = 0x12340000 | (~0x00005678 & MASK32)
        self.assertEqual(res, expected)
        self.assertEqual(self.interp.sgprs[1], expected)
        self.assertEqual(self.interp.sgprs[0], 0x12340000)

    def test_alias_all_operands_same(self) -> None:
        for initial in (0x00000000, 0x12345678, 0xFFFFFFFF, 0x5555AAAA):
            self.interp.write_sgpr(0, initial)
            inst = SALUInstruction(
                Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.SGPR, 0),
                ssrc0=RegOperand(RegKind.SGPR, 0),
                ssrc1=RegOperand(RegKind.SGPR, 0),
            )
            res, scc = self.interp.execute(inst)
            self.assertEqual(res, 0xFFFFFFFF, f"Self-aliased OR-NOT failed for initial {initial:#x}")
            self.assertEqual(self.interp.sgprs[0], 0xFFFFFFFF)
            self.assertEqual(scc, 1)

    def test_alias_src0_equals_src1_distinct_dst(self) -> None:
        self.interp.write_sgpr(0, 0xCAFEBABE)
        self.interp.write_sgpr(2, 0)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 2),
            ssrc0=RegOperand(RegKind.SGPR, 0),
            ssrc1=RegOperand(RegKind.SGPR, 0),
        )
        res, scc = self.interp.execute(inst)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(self.interp.sgprs[2], 0xFFFFFFFF)
        self.assertEqual(self.interp.sgprs[0], 0xCAFEBABE)
        self.assertEqual(scc, 1)

@dataclasses.dataclass(frozen=True)
class ConditionCodeState:
    """Architectural SALU condition code flags."""
    scc: int
    is_zero: bool
    is_negative: bool
    parity_odd: bool


def evaluate_salu_flags(result_u32: int) -> ConditionCodeState:
    """Compute condition code evaluation for 32-bit SALU result."""
    res = u32(result_u32)
    scc = 1 if res != 0 else 0
    is_zero = (res == 0)
    is_negative = bool(res & 0x80000000)
    parity_odd = (bin(res).count("1") % 2 == 1)
    return ConditionCodeState(scc=scc, is_zero=is_zero, is_negative=is_negative, parity_odd=parity_odd)


class TestSaluOrn2B32ConditionCodes(unittest.TestCase):
    """Unit tests for Scalar Condition Code (SCC) semantics and transitions."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_scc_zero_result(self) -> None:
        # The ONLY input pair producing result 0 is s0 = 0 and s1 = 0xFFFFFFFF
        self.interp.write_sgpr(1, 0x00000000)
        self.interp.write_sgpr(2, 0xFFFFFFFF)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, scc = self.interp.execute(inst)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.scc, 0)
        flags = evaluate_salu_flags(res)
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertEqual(flags.scc, 0)

    def test_scc_nonzero_results(self) -> None:
        test_cases = [
            (0x00000001, 0xFFFFFFFF, 1),             # lowest bit set
            (0x80000000, 0xFFFFFFFF, 0x80000000),    # sign bit set
            (0x00000000, 0xFFFFFFFE, 1),             # inverted bit 0 set
            (0x12345678, 0x9ABCDEF0, 0x12345678 | (~0x9ABCDEF0 & MASK32)),
            (0xFFFFFFFF, 0x00000000, 0xFFFFFFFF),
        ]
        for s0, s1, expected_res in test_cases:
            self.interp.write_sgpr(1, s0)
            self.interp.write_sgpr(2, s1)
            inst = SALUInstruction(
                Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.SGPR, 0),
                ssrc0=RegOperand(RegKind.SGPR, 1),
                ssrc1=RegOperand(RegKind.SGPR, 2),
            )
            res, scc = self.interp.execute(inst)
            self.assertEqual(res, expected_res)
            self.assertEqual(scc, 1)
            self.assertEqual(self.interp.scc, 1)

    def test_scc_state_transitions(self) -> None:
        # Step 1: SCC becomes 1
        self.interp.write_sgpr(1, 0x10)
        self.interp.write_sgpr(2, 0x20)
        inst_nonzero = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.interp.execute(inst_nonzero)
        self.assertEqual(self.interp.scc, 1)

        # Step 2: SCC becomes 0 (0 | ~0xFFFFFFFF = 0)
        self.interp.write_sgpr(1, 0)
        self.interp.write_sgpr(2, 0xFFFFFFFF)
        self.interp.execute(inst_nonzero)
        self.assertEqual(self.interp.scc, 0)

        # Step 3: SCC becomes 1 again
        self.interp.write_sgpr(1, 1)
        self.interp.write_sgpr(2, 0xFFFFFFFF)
        self.interp.execute(inst_nonzero)
        self.assertEqual(self.interp.scc, 1)

        # Step 4: s_mov_b32 preserves SCC
        self.interp.write_sgpr(5, 0)
        inst_mov = SALUInstruction(
            Opcode.S_MOV_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 5),
        )
        self.interp.execute(inst_mov)
        self.assertEqual(self.interp.scc, 1, "s_mov_b32 must not corrupt existing SCC")

def f32_to_bits(val: float) -> int:
    """Convert 32-bit float to raw unsigned integer bit pattern."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    """Convert raw unsigned 32-bit integer bit pattern to float."""
    return struct.unpack("<f", struct.pack("<I", u32(bits)))[0]


class TestSaluOrn2B32IEEE754EdgeCases(unittest.TestCase):
    """Unit tests for bitwise operations over IEEE 754 float bit patterns."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_ieee754_zeros_and_infinities(self) -> None:
        pos_zero = f32_to_bits(0.0)             # 0x00000000
        neg_zero = f32_to_bits(-0.0)            # 0x80000000
        pos_inf = f32_to_bits(float("inf"))     # 0x7F800000
        neg_inf = f32_to_bits(float("-inf"))    # 0xFF800000

        # +0.0 | ~(-0.0) == 0x00000000 | 0x7FFFFFFF == 0x7FFFFFFF
        res, scc = ref_orn2_b32(pos_zero, neg_zero)
        self.assertEqual(res, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

        # +inf | ~(-inf) == 0x7F800000 | 0x007FFFFF == 0x7FFFFFFF
        res_inf, _ = ref_orn2_b32(pos_inf, neg_inf)
        self.assertEqual(res_inf, 0x7FFFFFFF)

        # Setting sign bit: (pos_zero | ~0x7FFFFFFF) = 0x80000000
        res_sign, _ = ref_orn2_b32(pos_zero, 0x7FFFFFFF)
        self.assertEqual(res_sign, 0x80000000)

    def test_ieee754_nan_payloads(self) -> None:
        qnan_bits = f32_to_bits(float("nan"))
        snan_bits = 0x7F800001

        res, scc = ref_orn2_b32(qnan_bits, 0)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

        res2, _ = ref_orn2_b32(snan_bits, 0xFFFFFFFF)
        self.assertEqual(res2, snan_bits)

    def test_ieee754_subnormals_and_denormals(self) -> None:
        subnormal_min = 0x00000001
        subnormal_max = 0x007FFFFF

        res, scc = ref_orn2_b32(subnormal_min, subnormal_max)
        self.assertEqual(res, 0xFF800001)
        self.assertEqual(scc, 1)

    def test_ieee754_exponent_masking(self) -> None:
        exp_mask = 0x7F800000
        float_one = f32_to_bits(1.0)  # 0x3F800000

        res, _ = ref_orn2_b32(float_one, exp_mask)
        self.assertEqual(res, float_one | (~exp_mask & MASK32))

if __name__ == "__main__":
    unittest.main()
