from __future__ import annotations

"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_add_u32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_add_u32 computes the 32-bit unsigned addition
  of two scalar source operands with carry-out generation:
      D.u = (S0.u + S1.u) mod 2^32
      SCC = (S0.u + S1.u >= 2^32) ? 1 : 0

  - Format: SOP2 (Scalar Operation with 2 inputs, 1 destination).
  - Operands:
      S0 and S1 can be scalar general-purpose registers (SGPR), inline constants
      (-16..64, float constants), or 32-bit literal constants.
      D is a 32-bit destination SGPR.
  - Condition code register (SCC):
      Updated with the 1-bit unsigned carry-out: 1 if unsigned sum overflows 2^32,
      0 otherwise. Prior SCC state does not affect the calculation.
  - SPIR-V translation (tools/rdna2spirv.py):
      Lowers to OpIAddCarry %umul %s0 %s1.
      Member 0: 32-bit wrapped sum stored to destination SGPR.
      Member 1: 32-bit carry-out flag stored to %scc variable.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

from dataclasses import dataclass
import enum
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


# --- Section 1: Mathematical Reference Oracle ---

def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_u32.

    Returns (wrapped_sum, scc_carry).
    """
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    wrapped = full & MASK32
    carry = 1 if full >= (1 << 32) else 0
    return wrapped, carry


def oracle_add_u32_analysis(s0: int, s1: int) -> dict[str, int | bool]:
    """Detailed architectural breakdown of s_add_u32 execution."""
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    has_carry = full >= (1 << 32)
    return {
        "s0": a,
        "s1": b,
        "full_sum": full,
        "carry": has_carry,
        "scc": 1 if has_carry else 0,
        "wrapped_result": full & MASK32,
    }


class SaluAddU32ReferenceOracleTests(unittest.TestCase):
    """Commit 1: Reference oracle tests for RDNA2 s_add_u32."""

    def test_oracle_zero_identity(self) -> None:
        """Adding zero to any 32-bit value preserves the value and sets SCC=0."""
        cases = [0, 1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                res0, scc0 = ref_s_add_u32(val, 0)
                self.assertEqual(res0, val)
                self.assertEqual(scc0, 0)

                res1, scc1 = ref_s_add_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)

    def test_oracle_basic_addition_no_carry(self) -> None:
        """Sums strictly below 2^32 must produce exact integer sum with SCC=0."""
        cases = [
            (10, 20, 30),
            (0x1000, 0x2000, 0x3000),
            (0x12345678, 0x01020304, 0x1336597C),
            (0x7FFFFFFE, 1, 0x7FFFFFFF),
            (0x40000000, 0x3FFFFFFF, 0x7FFFFFFF),
            (0x7FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFE),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
        ]
        for a, b, expected_sum in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, expected_sum)
                self.assertEqual(scc, 0)

    def test_oracle_carry_generation(self) -> None:
        """Sums at or above 2^32 must set SCC=1 and wrap modulo 2^32."""
        cases = [
            (0xFFFFFFFF, 1, 0, 1),
            (0xFFFFFFFF, 2, 1, 1),
            (0xFFFFFFFE, 2, 0, 1),
            (0x80000000, 0x80000000, 0, 1),
            (0x80000000, 0x80000001, 1, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 1),
            (0xC0000000, 0x50000000, 0x10000000, 1),
        ]
        for a, b, exp_res, exp_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, exp_res)
                self.assertEqual(scc, exp_scc)

    def test_oracle_commutativity(self) -> None:
        """s_add_u32 must be strictly commutative across all operand ranges."""
        pairs = [
            (0, 0),
            (0, 0xFFFFFFFF),
            (1, 0xFFFFFFFF),
            (0x12345678, 0x87654321),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xCAFEBABE, 0xDEADBEEF),
            (0xAAAAAAAA, 0x55555555),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                r1, s1 = ref_s_add_u32(a, b)
                r2, s2 = ref_s_add_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_oracle_analysis_breakdown(self) -> None:
        """Detailed analysis oracle must compute full sums and carry flags."""
        info = oracle_add_u32_analysis(0xFFFFFFFF, 1)
        self.assertEqual(info["s0"], 0xFFFFFFFF)
        self.assertEqual(info["s1"], 1)
        self.assertEqual(info["full_sum"], 0x100000000)
        self.assertTrue(info["carry"])
        self.assertEqual(info["scc"], 1)
        self.assertEqual(info["wrapped_result"], 0)

        info2 = oracle_add_u32_analysis(0x1234, 0x5678)
        self.assertFalse(info2["carry"])
        self.assertEqual(info2["scc"], 0)
        self.assertEqual(info2["wrapped_result"], 0x68AC)



# --- Section 2: Symbolic AST Opcode Definitions ---

class RegKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    IMM = "imm"
    LITERAL = "literal"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class InstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand


def parse_salu_instruction(line: str) -> InstructionAST:
    """Parse a single SALU SOP2 assembly line into an InstructionAST."""
    line = line.split(";")[0].strip()
    if not line:
        raise ValueError("Empty instruction line")
    parts = line.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"SOP2 instruction requires 3 operands, got: {line}")
    mnemonic, d_str, s0_str, s1_str = parts
    if mnemonic != "s_add_u32":
        raise ValueError(f"Expected s_add_u32 mnemonic, got {mnemonic}")

    def parse_op(tok: str) -> Operand:
        tok = tok.strip()
        if tok.startswith("s"):
            return Operand(RegKind.SGPR, index=int(tok[1:]))
        elif tok.startswith("v"):
            return Operand(RegKind.VGPR, index=int(tok[1:]))
        else:
            val = int(tok, 0)
            if -16 <= val <= 64:
                return Operand(RegKind.IMM, raw_val=val & MASK32)
            return Operand(RegKind.LITERAL, raw_val=val & MASK32)

    dst = parse_op(d_str)
    if dst.kind != RegKind.SGPR:
        raise ValueError(f"Destination must be an SGPR, got: {d_str}")

    return InstructionAST(mnemonic, dst, parse_op(s0_str), parse_op(s1_str))


class SaluAddU32AstValidationTests(unittest.TestCase):
    """Commit 2: AST definition and parsing tests for s_add_u32."""

    def test_ast_construction_sgpr_operands(self) -> None:
        ast = InstructionAST(
            mnemonic="s_add_u32",
            dst=Operand(RegKind.SGPR, index=4),
            src0=Operand(RegKind.SGPR, index=2),
            src1=Operand(RegKind.SGPR, index=3),
        )
        self.assertEqual(ast.mnemonic, "s_add_u32")
        self.assertEqual(ast.dst.kind, RegKind.SGPR)
        self.assertEqual(ast.dst.index, 4)
        self.assertEqual(ast.src0.index, 2)
        self.assertEqual(ast.src1.index, 3)

    def test_ast_imm_and_literal_operands(self) -> None:
        ast = InstructionAST(
            mnemonic="s_add_u32",
            dst=Operand(RegKind.SGPR, index=0),
            src0=Operand(RegKind.SGPR, index=1),
            src1=Operand(RegKind.IMM, raw_val=42),
        )
        self.assertEqual(ast.src1.kind, RegKind.IMM)
        self.assertEqual(ast.src1.raw_val, 42)

    def test_parser_valid_sop2_instructions(self) -> None:
        cases = [
            ("s_add_u32 s4, s2, s3", 4, RegKind.SGPR, 2, RegKind.SGPR, 3),
            ("s_add_u32 s0, s1, 1", 0, RegKind.SGPR, 1, RegKind.IMM, 1),
            ("s_add_u32 s10, 0x1000, s5", 10, RegKind.LITERAL, 0x1000, RegKind.SGPR, 5),
        ]
        for line, dst_idx, s0_k, s0_val, s1_k, s1_val in cases:
            ast = parse_salu_instruction(line)
            self.assertEqual(ast.mnemonic, "s_add_u32")
            self.assertEqual(ast.dst.index, dst_idx)
            self.assertEqual(ast.src0.kind, s0_k)
            self.assertEqual(ast.src1.kind, s1_k)

    def test_parser_syntax_rejection(self) -> None:
        invalid = [
            "s_add_u32",
            "s_add_u32 s0",
            "s_add_u32 s0, s1",
            "s_add_u32 s0, s1, s2, s3",
            "s_add_u32 v0, s1, s2",  # dst must be SGPR
            "v_add_u32 s0, s1, s2",  # wrong mnemonic
        ]
        for line in invalid:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_salu_instruction(line)


# --- Section 3: Deterministic Scalar Interpreter ---

class ScalarRegisterState:
    """Simulates scalar register file (SGPRs) and condition code (SCC)."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.scc: int = 0

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_operand_val(self, op: Operand) -> int:
        if op.kind == RegKind.SGPR:
            return self.get_sgpr(op.index)
        return op.raw_val & MASK32

    def step(self, ast: InstructionAST) -> tuple[int, int]:
        """Execute a single SALU instruction, returning (dst_val, new_scc)."""
        s0 = self.get_operand_val(ast.src0)
        s1 = self.get_operand_val(ast.src1)
        if ast.mnemonic == "s_add_u32":
            res, scc = ref_s_add_u32(s0, s1)
            self.scc = scc
            self.set_sgpr(ast.dst.index, res)
            return res, scc
        raise NotImplementedError(f"Unsupported mnemonic: {ast.mnemonic}")


class SaluAddU32ScalarInterpreterTests(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter tests."""

    def test_scalar_step_basic_add(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(2, 0x100)
        state.set_sgpr(3, 0x200)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 4), Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 3))
        res, scc = state.step(ast)
        self.assertEqual(res, 0x300)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(4), 0x300)
        self.assertEqual(state.scc, 0)

    def test_scalar_step_carry_update(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(0, 0xFFFFFFFF)
        state.set_sgpr(1, 5)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        res, scc = state.step(ast)
        self.assertEqual(res, 4)
        self.assertEqual(scc, 1)
        self.assertEqual(state.get_sgpr(2), 4)
        self.assertEqual(state.scc, 1)

    def test_scalar_chain_accumulations(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(0, 10)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 0), Operand(RegKind.IMM, raw_val=5))
        for expected in [15, 20, 25, 30]:
            res, scc = state.step(ast)
            self.assertEqual(res, expected)
            self.assertEqual(scc, 0)
            self.assertEqual(state.get_sgpr(0), expected)


# --- Section 4: SIMD Multi-Lane Divergence and Wave Execution ---

class SimdLaneState:
    """State of an individual vector lane."""

    def __init__(self, lane_id: int) -> None:
        self.lane_id = lane_id
        self.vgprs: dict[int, int] = {}

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0) & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


class SimdWaveState:
    """Simulates wave-level scalar execution and vector lane interactions."""

    def __init__(self, wave_size: int = 32) -> None:
        self.wave_size = wave_size
        self.scalar_state = ScalarRegisterState()
        self.lanes = [SimdLaneState(i) for i in range(wave_size)]
        self.exec_mask: int = (1 << wave_size) - 1

    def broadcast_scalar_to_lanes(self, sgpr_idx: int, vgpr_idx: int) -> None:
        """Broadcast a scalar register to a VGPR across active lanes."""
        val = self.scalar_state.get_sgpr(sgpr_idx)
        for i, lane in enumerate(self.lanes):
            if (self.exec_mask >> i) & 1:
                lane.set_vgpr(vgpr_idx, val)


class SaluAddU32SimdExecutionTests(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and EXEC masking tests."""

    def test_wave32_uniform_scalar_broadcast(self) -> None:
        wave = SimdWaveState(wave_size=32)
        wave.scalar_state.set_sgpr(1, 0x1234)
        wave.scalar_state.set_sgpr(2, 0x5678)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1), Operand(RegKind.SGPR, 2))
        wave.scalar_state.step(ast)
        wave.broadcast_scalar_to_lanes(sgpr_idx=0, vgpr_idx=5)
        for lane in wave.lanes:
            self.assertEqual(lane.get_vgpr(5), 0x68AC)

    def test_divergent_exec_mask_preservation(self) -> None:
        wave = SimdWaveState(wave_size=32)
        for lane in wave.lanes:
            lane.set_vgpr(0, 0xDEADBEEF)

        wave.exec_mask = 0x0000FFFF  # only low 16 lanes active
        wave.scalar_state.set_sgpr(0, 100)
        wave.scalar_state.set_sgpr(1, 200)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        wave.scalar_state.step(ast)
        wave.broadcast_scalar_to_lanes(sgpr_idx=2, vgpr_idx=0)

        for i, lane in enumerate(wave.lanes):
            if i < 16:
                self.assertEqual(lane.get_vgpr(0), 300)
            else:
                self.assertEqual(lane.get_vgpr(0), 0xDEADBEEF)

    def test_wave64_scalar_uniformity(self) -> None:
        wave = SimdWaveState(wave_size=64)
        wave.scalar_state.set_sgpr(10, 0xFFFFFFFF)
        wave.scalar_state.set_sgpr(11, 1)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 12), Operand(RegKind.SGPR, 10), Operand(RegKind.SGPR, 11))
        wave.scalar_state.step(ast)
        self.assertEqual(wave.scalar_state.get_sgpr(12), 0)
        self.assertEqual(wave.scalar_state.scc, 1)
        wave.broadcast_scalar_to_lanes(sgpr_idx=12, vgpr_idx=1)
        for lane in wave.lanes:
            self.assertEqual(lane.get_vgpr(1), 0)


# --- Section 5: Register Aliasing and In-Place Operands ---

class SaluAddU32RegisterAliasingTests(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation tests."""

    def test_dst_aliases_src0(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(4, 50)
        state.set_sgpr(5, 75)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 4), Operand(RegKind.SGPR, 4), Operand(RegKind.SGPR, 5))
        res, scc = state.step(ast)
        self.assertEqual(res, 125)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(4), 125)

    def test_dst_aliases_src1(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(4, 50)
        state.set_sgpr(5, 75)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 5), Operand(RegKind.SGPR, 4), Operand(RegKind.SGPR, 5))
        res, scc = state.step(ast)
        self.assertEqual(res, 125)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(5), 125)

    def test_all_operands_alias(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(8, 0x1000)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 8), Operand(RegKind.SGPR, 8), Operand(RegKind.SGPR, 8))
        res, scc = state.step(ast)
        self.assertEqual(res, 0x2000)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(8), 0x2000)

    def test_aliased_overflow_in_place(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x80000000)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 0))
        res, scc = state.step(ast)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)
        self.assertEqual(state.get_sgpr(0), 0)
        self.assertEqual(state.scc, 1)

    def test_src0_equals_src1_distinct_dst(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(1, 0x3FFFFFF)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 1), Operand(RegKind.SGPR, 1))
        res, scc = state.step(ast)
        self.assertEqual(res, 0x7FFFFFE)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(1), 0x3FFFFFF)
        self.assertEqual(state.get_sgpr(2), 0x7FFFFFE)


# --- Section 6: Condition Codes and Status Flags Generation ---

class SaluAddU32ConditionCodesTests(unittest.TestCase):
    """Commit 6: Condition codes (SCC) and status flag generation tests."""

    def test_scc_cleared_when_no_carry(self) -> None:
        state = ScalarRegisterState()
        state.scc = 1  # Preset SCC to 1
        state.set_sgpr(0, 10)
        state.set_sgpr(1, 20)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        _, scc = state.step(ast)
        self.assertEqual(scc, 0)
        self.assertEqual(state.scc, 0)

    def test_scc_set_when_carry(self) -> None:
        state = ScalarRegisterState()
        state.scc = 0  # Preset SCC to 0
        state.set_sgpr(0, 0xFFFFFFFF)
        state.set_sgpr(1, 1)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        _, scc = state.step(ast)
        self.assertEqual(scc, 1)
        self.assertEqual(state.scc, 1)

    def test_scc_prior_state_independence(self) -> None:
        """s_add_u32 does not use carry-in: result is identical whether SCC was 0 or 1."""
        for scc_preset in (0, 1):
            state = ScalarRegisterState()
            state.scc = scc_preset
            state.set_sgpr(0, 0x7FFFFFFF)
            state.set_sgpr(1, 1)
            ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
            res, scc = state.step(ast)
            self.assertEqual(res, 0x80000000)
            self.assertEqual(scc, 0)

    def test_boundary_carry_threshold(self) -> None:
        """0xFFFFFFFE + 1 does not carry (SCC=0), but 0xFFFFFFFF + 1 carries (SCC=1)."""
        state = ScalarRegisterState()
        state.set_sgpr(0, 0xFFFFFFFE)
        state.set_sgpr(1, 1)
        ast1 = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        res1, scc1 = state.step(ast1)
        self.assertEqual(res1, 0xFFFFFFFF)
        self.assertEqual(scc1, 0)

        state.set_sgpr(0, 0xFFFFFFFF)
        ast2 = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        res2, scc2 = state.step(ast2)
        self.assertEqual(res2, 0)
        self.assertEqual(scc2, 1)


# --- Section 7: IEEE 754 Boundary Extremes and NaN Handling ---

class SaluAddU32IEEE754BitPatternsTests(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN bit patterns."""

    def test_float_zero_bit_patterns(self) -> None:
        """+0.0f (0x0) and -0.0f (0x80000000) are treated strictly as unsigned integers."""
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        # +0.0 + -0.0 = 0x80000000 (no carry)
        res, scc = ref_s_add_u32(pos_zero, neg_zero)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 0)

        # -0.0 + -0.0 = 0x00000000 with carry-out
        res, scc = ref_s_add_u32(neg_zero, neg_zero)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

    def test_subnormal_float_preservation(self) -> None:
        """Subnormal bit patterns are preserved without denormal flushing."""
        min_subnorm = 0x00000001
        max_subnorm = 0x007FFFFF
        res, scc = ref_s_add_u32(min_subnorm, max_subnorm)
        self.assertEqual(res, 0x00800000)  # min normal float pattern
        self.assertEqual(scc, 0)

    def test_infinity_and_nan_bit_patterns(self) -> None:
        """Infinity and NaN bit patterns undergo standard modular integer arithmetic."""
        pos_inf = 0x7F800000
        qnan = 0x7FC00000
        # pos_inf + 1 becomes 0x7F800001 (signaling NaN pattern), no float exception trapped
        res, scc = ref_s_add_u32(pos_inf, 1)
        self.assertEqual(res, 0x7F800001)
        self.assertEqual(scc, 0)

        # qnan + qnan
        res, scc = ref_s_add_u32(qnan, qnan)
        self.assertEqual(res, (qnan * 2) & MASK32)
        self.assertEqual(scc, 0)

    def test_sign_bit_rollover(self) -> None:
        """0x7FFFFFFF (+max float exponent/mantissa edge) + 1 rolls over to 0x80000000."""
        res, scc = ref_s_add_u32(0x7FFFFFFF, 1)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 0)


# --- Section 8: Hardware Translator Integration Tests ---

def _run_ps_add_u32(lines: list[str]) -> tuple[int, int]:
    """Assemble pixel shader instructions, translate with rdna2spirv, execute via mini-interpreter.

    Returns (result, scc).
    """
    full_lines = list(lines) + [
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_add_u32", full_lines)
    words = mrt0_words(spv)
    return words[0], words[1]


def run_op_s_add_u32(s0: int, s1: int, preset_scc: int = 0) -> tuple[int, int]:
    """Execute s_add_u32 through rdna2spirv translation and return (sum, scc)."""
    lines = [
        f"s_mov_b32 s2, {s0 & MASK32:#x}",
        f"s_mov_b32 s3, {s1 & MASK32:#x}",
        f"s_cmp_eq_u32 {1 if preset_scc else 0}, 1",
        "s_add_u32 s0, s2, s3",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
    ]
    return _run_ps_add_u32(lines)


class SaluAddU32TranslatorIntegrationTests(unittest.TestCase):
    """Commit 8: Hardware translator integration tests."""

    def test_translator_basic_addition(self) -> None:
        res, scc = run_op_s_add_u32(100, 200)
        self.assertEqual(res, 300)
        self.assertEqual(scc, 0)

    def test_translator_carry_generation(self) -> None:
        res, scc = run_op_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

        res2, scc2 = run_op_s_add_u32(0x80000000, 0x80000000)
        self.assertEqual(res2, 0)
        self.assertEqual(scc2, 1)

    def test_translator_inline_immediates(self) -> None:
        lines = [
            "s_mov_b32 s2, 50",
            "s_add_u32 s0, s2, 15",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        res, scc = _run_ps_add_u32(lines)
        self.assertEqual(res, 65)
        self.assertEqual(scc, 0)

    def test_translator_literal_constants(self) -> None:
        lines = [
            "s_mov_b32 s2, 0x10000000",
            "s_add_u32 s0, s2, 0x20000000",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        res, scc = _run_ps_add_u32(lines)
        self.assertEqual(res, 0x30000000)
        self.assertEqual(scc, 0)

    def test_translator_register_aliasing(self) -> None:
        lines = [
            "s_mov_b32 s0, 0x40000000",
            "s_add_u32 s0, s0, s0",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        res, scc = _run_ps_add_u32(lines)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 0)

    def test_translator_matches_reference_oracle(self) -> None:
        sample_vectors = [
            (0, 0),
            (0, 1),
            (1, 0),
            (42, 58),
            (0x7FFFFFFF, 1),
            (0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xCAFEBABE, 0x12345678),
        ]
        for a, b in sample_vectors:
            with self.subTest(a=hex(a), b=hex(b)):
                exp_res, exp_scc = ref_s_add_u32(a, b)
                got_res, got_scc = run_op_s_add_u32(a, b)
                self.assertEqual(got_res, exp_res)
                self.assertEqual(got_scc, exp_scc)

if __name__ == "__main__":
    unittest.main()
