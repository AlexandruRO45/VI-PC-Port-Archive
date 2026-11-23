from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 s_xnor_b32.

AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP2 instruction:
  s_xnor_b32 sdst, ssrc0, ssrc1

Operation:
  sdst = ~(ssrc0 ^ ssrc1) & 0xFFFFFFFF
  scc  = 1 if sdst != 0 else 0

Architectural Characteristics:
  - Operates on 32-bit scalar general-purpose registers (SGPRs), inline constants,
    or literal constants.
  - Bitwise equivalence (NOT-XOR): Each bit in destination is set to 1 if the
    corresponding bits in ssrc0 and ssrc1 are equal, and 0 if they differ.
  - Scalar Condition Code (SCC): Reflects whether the resulting 32-bit dword is non-zero.
  - Scalar execution: Unlike vector instructions, SALU instructions execute uniformly
    once per wave across the scalar register file, unaffected by the vector EXEC mask.
  - Aliasing: Source and destination registers may alias arbitrarily (e.g. sdst == ssrc0).
  - Pure 32-bit bitwise operation: Preserves exact bit patterns of IEEE 754 representations
    without denormal flushing, quiet NaN canonicalization, or sign-zero collapsing.

Standard library Python only; zero third-party dependencies; zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32: int = 0xFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 0xFFFFFFFF]."""
    return val & MASK32


def ref_s_xnor_b32(src0: int, src1: int) -> Tuple[int, int]:
    """Pure mathematical and architectural reference oracle for RDNA2 s_xnor_b32.

    Returns:
        (result_u32, scc) where:
          result_u32 = ~(src0 ^ src1) & 0xFFFFFFFF
          scc = 1 if result_u32 != 0 else 0
    """
    s0 = u32(src0)
    s1 = u32(src1)
    res = u32(~(s0 ^ s1))
    scc = 1 if res != 0 else 0
    return res, scc


def run_translator_salu_xnor(
    src0: int,
    src1: int,
    name: str = "ps_salu_xnor",
) -> Tuple[int, int]:
    """Translate and run a minimal shader verifying s_xnor_b32 and SCC output.

    Uses s_cselect_b32 to inspect SCC flag:
      s_cselect_b32 s3, 1, 0
    Selects 1 if SCC != 0, else 0.
    Returns (result_dword, scc_code).
    """
    lines = [
        f"s_mov_b32 s0, {u32(src0):#x}",
        f"s_mov_b32 s1, {u32(src1):#x}",
        "s_xnor_b32 s2, s0, s1",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v0, s2",
        "v_mov_b32 v1, s3",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_truth_table_exhaustive(self) -> None:
        """Verify 1-bit XNOR logic: (0^0)->1, (0^1)->0, (1^0)->0, (1^1)->1."""
        self.assertEqual(ref_s_xnor_b32(0, 0), (0xFFFFFFFF, 1))
        self.assertEqual(ref_s_xnor_b32(0, 1), (0xFFFFFFFE, 1))
        self.assertEqual(ref_s_xnor_b32(1, 0), (0xFFFFFFFE, 1))
        self.assertEqual(ref_s_xnor_b32(1, 1), (0xFFFFFFFF, 1))

    def test_bitwise_identities(self) -> None:
        """Verify fundamental boolean algebra properties of XNOR."""
        vals = [0x00000000, 0xFFFFFFFF, 0x12345678, 0xAAAAAAAA, 0x55555555, 0x80000000]
        for v in vals:
            # Identity 1: X XNOR X == 0xFFFFFFFF (reflexive equivalence)
            res_id, scc_id = ref_s_xnor_b32(v, v)
            self.assertEqual(res_id, 0xFFFFFFFF)
            self.assertEqual(scc_id, 1)

            # Identity 2: X XNOR ~X == 0x00000000 (complementarity)
            res_comp, scc_comp = ref_s_xnor_b32(v, ~v)
            self.assertEqual(res_comp, 0x00000000)
            self.assertEqual(scc_comp, 0)

            # Identity 3: X XNOR 0 == ~X (bitwise NOT)
            res_not, _ = ref_s_xnor_b32(v, 0)
            self.assertEqual(res_not, u32(~v))

            # Identity 4: X XNOR 0xFFFFFFFF == X (bitwise identity)
            res_same, _ = ref_s_xnor_b32(v, 0xFFFFFFFF)
            self.assertEqual(res_same, u32(v))

    def test_commutativity(self) -> None:
        """Verify commutativity: ssrc0 XNOR ssrc1 == ssrc1 XNOR ssrc0."""
        a = 0xA5A55A5A
        b = 0x12345678
        self.assertEqual(ref_s_xnor_b32(a, b), ref_s_xnor_b32(b, a))

    def test_oracle_matches_translator(self) -> None:
        """Cross-check reference oracle with translator output on basic vectors."""
        cases = [
            (0x12345678, 0x12345678),
            (0x55555555, 0xAAAAAAAA),
            (0x00000000, 0xFFFFFFFF),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for idx, (s0, s1) in enumerate(cases):
            expected_res, expected_scc = ref_s_xnor_b32(s0, s1)
            trans_res, trans_scc = run_translator_salu_xnor(s0, s1, name=f"ps_oracle_xnor_{idx}")
            self.assertEqual(trans_res, expected_res)
            self.assertEqual(trans_scc, expected_scc)


# ==============================================================================
# SECTION 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(Enum):
    SGPR = auto()
    IMMEDIATE = auto()
    VGPR = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    value: int

    def __post_init__(self) -> None:
        if self.kind == OperandKind.SGPR:
            if not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of architectural range [0, 105]: {self.value}")
        elif self.kind == OperandKind.IMMEDIATE:
            if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
                raise ValueError(f"Immediate value out of 32-bit range: {self.value}")
        elif self.kind == OperandKind.VGPR:
            if not (0 <= self.value <= 255):
                raise ValueError(f"VGPR index out of architectural range [0, 255]: {self.value}")


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def Immediate(val: int) -> Operand:
    return Operand(OperandKind.IMMEDIATE, u32(val))


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


@dataclass(frozen=True)
class SXnorB32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.SGPR:
            raise TypeError(f"Destination for s_xnor_b32 must be an SGPR, got {self.dst.kind}")
        if self.src0.kind == OperandKind.VGPR or self.src1.kind == OperandKind.VGPR:
            raise TypeError("SOP2 scalar ALU instruction cannot accept VGPR source operands directly")

    def format_operand(self, op: Operand) -> str:
        if op.kind == OperandKind.SGPR:
            return f"s{op.value}"
        elif op.kind == OperandKind.IMMEDIATE:
            return f"{op.value:#x}"
        else:
            return f"v{op.value}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        return f"s_xnor_b32 {d}, {s0}, {s1}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_assembly(self) -> None:
        """Verify AST correctly validates and formats to standard RDNA2 disassembly."""
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        self.assertEqual(instr.to_assembly(), "s_xnor_b32 s2, s0, s1")

        instr_imm = SXnorB32Instruction(dst=SGPR(4), src0=SGPR(5), src1=Immediate(0x12345678))
        self.assertEqual(instr_imm.to_assembly(), "s_xnor_b32 s4, s5, 0x12345678")

    def test_ast_operand_type_guards(self) -> None:
        """Verify AST rejects malformed operands like VGPR destination or out-of-range registers."""
        with self.assertRaises(TypeError):
            SXnorB32Instruction(dst=VGPR(0), src0=SGPR(1), src1=SGPR(2))

        with self.assertRaises(TypeError):
            SXnorB32Instruction(dst=SGPR(0), src0=VGPR(1), src1=SGPR(2))

        with self.assertRaises(ValueError):
            SGPR(106)

        with self.assertRaises(ValueError):
            SGPR(-1)

    def test_ast_translation_validity(self) -> None:
        """Verify code generated from AST roundtrips through the translator cleanly."""
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        lines = [
            "s_mov_b32 s0, 0xAAAAAAAA",
            "s_mov_b32 s1, 0xAAAAAAAA",
            instr.to_assembly(),
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_xnor_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFFFFFF)


# ==============================================================================
# SECTION 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    sgprs: Dict[int, int] = field(default_factory=dict)
    scc: int = 0

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = u32(val)


class ScalarInterpreter:
    def resolve_operand(self, op: Operand, state: ScalarMachineState) -> int:
        if op.kind == OperandKind.SGPR:
            return state.get_sgpr(op.value)
        elif op.kind == OperandKind.IMMEDIATE:
            return u32(op.value)
        else:
            raise ValueError(f"Unsupported operand kind for scalar ALU: {op.kind}")

    def step(self, instr: SXnorB32Instruction, state: ScalarMachineState) -> int:
        val0 = self.resolve_operand(instr.src0, state)
        val1 = self.resolve_operand(instr.src1, state)
        res, scc = ref_s_xnor_b32(val0, val1)
        state.set_sgpr(instr.dst.value, res)
        state.scc = scc
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify interpreter state progression and register file updates."""
        state = ScalarMachineState()
        state.set_sgpr(0, 0x0F0F0F0F)
        state.set_sgpr(1, 0x00FF00FF)

        interp = ScalarInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        res = interp.step(instr, state)

        expected_res, expected_scc = ref_s_xnor_b32(0x0F0F0F0F, 0x00FF00FF)
        self.assertEqual(res, expected_res)
        self.assertEqual(state.get_sgpr(2), expected_res)
        self.assertEqual(state.scc, expected_scc)

    def test_scalar_interpreter_immediate_operand(self) -> None:
        """Verify scalar interpreter with inline immediate constant."""
        state = ScalarMachineState()
        state.set_sgpr(0, 0xF0F0F0F0)

        interp = ScalarInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(3), src0=SGPR(0), src1=Immediate(0x0F0F0F0F))
        res = interp.step(instr, state)

        # 0xF0F0F0F0 ^ 0x0F0F0F0F = 0xFFFFFFFF -> ~(0xFFFFFFFF) = 0x00000000
        self.assertEqual(res, 0x00000000)
        self.assertEqual(state.get_sgpr(3), 0x00000000)
        self.assertEqual(state.scc, 0)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Cross-validate interpreter step against translated SPIR-V shader."""
        state = ScalarMachineState()
        v0 = 0x13579BDF
        v1 = 0x2468ACE0
        state.set_sgpr(0, v0)
        state.set_sgpr(1, v1)

        interp = ScalarInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        interp_res = interp.step(instr, state)

        trans_res, trans_scc = run_translator_salu_xnor(v0, v1, name="ps_scalar_xnor_cross")
        self.assertEqual(interp_res, trans_res)
        self.assertEqual(state.scc, trans_scc)


# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    wave_size: int = 32
    exec_mask: int = MASK32
    vgprs: List[Dict[int, int]] = field(default_factory=list)
    sgprs: Dict[int, int] = field(default_factory=dict)
    scc: int = 0

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {self.wave_size}")
        if not self.vgprs:
            self.vgprs = [dict() for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, idx: int) -> int:
        return self.vgprs[lane].get(idx, 0)

    def set_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = u32(val)

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = u32(val)


class SIMDWaveInterpreter:
    """Interprets SALU instruction in the context of a SIMD Wave (Wave32 or Wave64).

    Architectural Rule (AMD RDNA2 ISA):
      Scalar ALU instructions execute ONCE per wave on the scalar execution unit.
      Unlike VALU instructions which only write active lanes according to EXEC,
      SALU instructions execute unconditionally across all lanes, reading SGPRs
      and writing their destination SGPR / SCC regardless of the vector EXEC mask.
    """

    def step_salu(self, instr: SXnorB32Instruction, wave: SIMDWaveState) -> int:
        val0 = wave.get_sgpr(instr.src0.value) if instr.src0.kind == OperandKind.SGPR else u32(instr.src0.value)
        val1 = wave.get_sgpr(instr.src1.value) if instr.src1.kind == OperandKind.SGPR else u32(instr.src1.value)

        res, scc = ref_s_xnor_b32(val0, val1)
        wave.set_sgpr(instr.dst.value, res)
        wave.scc = scc
        return res


class TestSIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_salu_uniformity_under_divergent_exec_mask(self) -> None:
        """Verify SALU s_xnor_b32 executes uniformly even when EXEC has inactive lanes."""
        # Wave32 with only lane 0 and lane 31 active
        mask = (1 << 0) | (1 << 31)
        wave = SIMDWaveState(wave_size=32, exec_mask=mask)
        wave.set_sgpr(0, 0x11111111)
        wave.set_sgpr(1, 0x11111111)

        interp = SIMDWaveInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        res = interp.step_salu(instr, wave)

        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(wave.get_sgpr(2), 0xFFFFFFFF)
        self.assertEqual(wave.scc, 1)

    def test_salu_under_zero_exec_mask(self) -> None:
        """Verify SALU s_xnor_b32 executes even when EXEC == 0 (all lanes divergent/inactive)."""
        wave = SIMDWaveState(wave_size=32, exec_mask=0x00000000)
        wave.set_sgpr(0, 0xAAAAAAAA)
        wave.set_sgpr(1, 0x55555555)

        interp = SIMDWaveInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(2), src0=SGPR(0), src1=SGPR(1))
        res = interp.step_salu(instr, wave)

        # 0xAAAAAAAA xnor 0x55555555 = 0
        self.assertEqual(res, 0x00000000)
        self.assertEqual(wave.get_sgpr(2), 0x00000000)
        self.assertEqual(wave.scc, 0)

    def test_wave32_lane_mask_equivalence_check(self) -> None:
        """Verify using s_xnor_b32 to compute bitwise equivalence between two lane ballot masks."""
        ballot_a = 0b11001100_10101010_11110000_00001111
        ballot_b = 0b11000000_10101111_11110000_00111111
        expected_agreement = ~(ballot_a ^ ballot_b) & MASK32

        wave = SIMDWaveState(wave_size=32, exec_mask=MASK32)
        wave.set_sgpr(4, ballot_a)
        wave.set_sgpr(5, ballot_b)

        interp = SIMDWaveInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(6), src0=SGPR(4), src1=SGPR(5))
        res = interp.step_salu(instr, wave)

        self.assertEqual(res, expected_agreement)
        self.assertEqual(wave.get_sgpr(6), expected_agreement)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0_in_place(self) -> None:
        """Verify s0 = s_xnor_b32(s0, s1) correctly reads prior s0 before overwriting."""
        state = ScalarMachineState()
        init_s0 = 0x12345678
        init_s1 = 0x87654321
        state.set_sgpr(0, init_s0)
        state.set_sgpr(1, init_s1)

        interp = ScalarInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(0), src0=SGPR(0), src1=SGPR(1))
        res = interp.step(instr, state)

        expected = ~(init_s0 ^ init_s1) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(state.get_sgpr(0), expected)

        lines = [
            f"s_mov_b32 s0, {init_s0:#x}",
            f"s_mov_b32 s1, {init_s1:#x}",
            "s_xnor_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_dst_src0", lines)
        self.assertEqual(support.mrt0_words(spv)[0], expected)

    def test_alias_dst_equals_src1_in_place(self) -> None:
        """Verify s1 = s_xnor_b32(s0, s1) correctly reads prior s1 before overwriting."""
        state = ScalarMachineState()
        init_s0 = 0xA5A5A5A5
        init_s1 = 0x5A5A5A5A
        state.set_sgpr(0, init_s0)
        state.set_sgpr(1, init_s1)

        interp = ScalarInterpreter()
        instr = SXnorB32Instruction(dst=SGPR(1), src0=SGPR(0), src1=SGPR(1))
        res = interp.step(instr, state)

        expected = ~(init_s0 ^ init_s1) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(state.get_sgpr(1), expected)

        lines = [
            f"s_mov_b32 s0, {init_s0:#x}",
            f"s_mov_b32 s1, {init_s1:#x}",
            "s_xnor_b32 s1, s0, s1",
            "v_mov_b32 v0, s1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_dst_src1", lines)
        self.assertEqual(support.mrt0_words(spv)[0], expected)

    def test_alias_all_operands_identical(self) -> None:
        """Verify s0 = s_xnor_b32(s0, s0) results in 0xFFFFFFFF and SCC = 1 for any s0."""
        for val in [0x00000000, 0x12345678, 0xFFFFFFFF, 0xDEADBEEF]:
            state = ScalarMachineState()
            state.set_sgpr(0, val)

            interp = ScalarInterpreter()
            instr = SXnorB32Instruction(dst=SGPR(0), src0=SGPR(0), src1=SGPR(0))
            res = interp.step(instr, state)

            self.assertEqual(res, 0xFFFFFFFF)
            self.assertEqual(state.get_sgpr(0), 0xFFFFFFFF)
            self.assertEqual(state.scc, 1)

            lines = [
                f"s_mov_b32 s0, {val:#x}",
                "s_xnor_b32 s0, s0, s0",
                "s_cselect_b32 s1, 1, 0",
                "v_mov_b32 v0, s0",
                "v_mov_b32 v1, s1",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
                "exp mrt0 v0, v1, v2, v3 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_alias_all_{val:x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0xFFFFFFFF)
            self.assertEqual(words[1], 1)


# ==============================================================================
# SECTION 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_scc_set_to_zero_when_result_zero(self) -> None:
        """Verify SCC is cleared to 0 when s_xnor_b32 produces 0."""
        res, scc = ref_s_xnor_b32(0x00000000, 0xFFFFFFFF)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        res_t, scc_t = run_translator_salu_xnor(0x00000000, 0xFFFFFFFF, name="ps_scc_zero")
        self.assertEqual(res_t, 0)
        self.assertEqual(scc_t, 0)

    def test_scc_set_to_one_when_result_nonzero(self) -> None:
        """Verify SCC is set to 1 when s_xnor_b32 produces non-zero (even single-bit)."""
        s0 = 0x00000001
        s1 = 0xFFFFFFFD
        res, scc = ref_s_xnor_b32(s0, s1)
        self.assertNotEqual(res, 0)
        self.assertEqual(scc, 1)

        res_t, scc_t = run_translator_salu_xnor(s0, s1, name="ps_scc_one")
        self.assertEqual(res_t, res)
        self.assertEqual(scc_t, 1)

    def test_scc_conditional_branch_selection(self) -> None:
        """Verify SCC interacts correctly with conditional selection s_cselect_b32."""
        lines_true = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x12345678",
            "s_xnor_b32 s2, s0, s1",            # Result 0xFFFFFFFF -> SCC = 1
            "s_cselect_b32 s3, 0xCAFE, 0xBABE", # Selects 0xCAFE
            "v_mov_b32 v0, s3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_true = support.translate_lines("ps_scc_cselect_true", lines_true)
        self.assertEqual(support.mrt0_words(spv_true)[0], 0xCAFE)

        lines_false = [
            "s_mov_b32 s0, 0x55555555",
            "s_mov_b32 s1, 0xAAAAAAAA",
            "s_xnor_b32 s2, s0, s1",            # Result 0x00000000 -> SCC = 0
            "s_cselect_b32 s3, 0xCAFE, 0xBABE", # Selects 0xBABE
            "v_mov_b32 v0, s3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_false = support.translate_lines("ps_scc_cselect_false", lines_false)
        self.assertEqual(support.mrt0_words(spv_false)[0], 0xBABE)


# ==============================================================================
# SECTION 7: IEEE 754 Boundary Extremes, Subnormals, and Bit Pattern Fidelity
# ==============================================================================

F32_POS_ZERO: int = 0x00000000
F32_NEG_ZERO: int = 0x80000000
F32_ONE: int = 0x3F800000
F32_NEG_ONE: int = 0xBF800000
F32_POS_INF: int = 0x7F800000
F32_NEG_INF: int = 0xFF800000
F32_QNAN: int = 0x7FC00000
F32_SNAN: int = 0x7F800001
F32_SUBNORMAL_MIN: int = 0x00000001
F32_SUBNORMAL_MAX: int = 0x007FFFFF


class TestIEEE754EdgeCases(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and bit pattern fidelity."""

    def test_signed_zeros_distinction(self) -> None:
        """Verify s_xnor_b32 operates purely bitwise and distinguishes +0.0 from -0.0."""
        res, scc = ref_s_xnor_b32(F32_POS_ZERO, F32_NEG_ZERO)
        self.assertEqual(res, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

        trans_res, trans_scc = run_translator_salu_xnor(F32_POS_ZERO, F32_NEG_ZERO, name="ps_f32_zeros")
        self.assertEqual(trans_res, 0x7FFFFFFF)
        self.assertEqual(trans_scc, 1)

    def test_nan_reflexivity_preservation(self) -> None:
        """Verify NaN XNOR NaN produces 0xFFFFFFFF without NaN quieting or canonicalization."""
        for nan_pattern in [F32_QNAN, F32_SNAN, 0x7FA5A5A5, 0xFFC00001]:
            res, scc = ref_s_xnor_b32(nan_pattern, nan_pattern)
            self.assertEqual(res, 0xFFFFFFFF)
            self.assertEqual(scc, 1)

            trans_res, trans_scc = run_translator_salu_xnor(nan_pattern, nan_pattern, name=f"ps_nan_{nan_pattern:x}")
            self.assertEqual(trans_res, 0xFFFFFFFF)
            self.assertEqual(trans_scc, 1)

    def test_infinities_and_subnormals(self) -> None:
        """Verify infinities and subnormal bit patterns undergo no FTZ (flush-to-zero)."""
        res_inf, _ = ref_s_xnor_b32(F32_POS_INF, F32_NEG_INF)
        self.assertEqual(res_inf, 0x7FFFFFFF)

        res_sub, _ = ref_s_xnor_b32(F32_SUBNORMAL_MIN, 0)
        self.assertEqual(res_sub, 0xFFFFFFFE)

        trans_res, _ = run_translator_salu_xnor(F32_SUBNORMAL_MIN, 0, name="ps_subnormal_no_ftz")
        self.assertEqual(trans_res, 0xFFFFFFFE)


# ==============================================================================
# SECTION 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestTranslatorExecutionAndMutations(unittest.TestCase):
    """Commit 8: Translator-backed execution tests, negative controls, and mutation detection."""

    def test_translator_constant_folding(self) -> None:
        """Verify translator correctly folds two immediate literals during translation."""
        lines = [
            "s_xnor_b32 s0, 0x12345678, 0x12345678",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_const_fold_xnor", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xFFFFFFFF)

    def test_mutation_detection_swap_xor(self) -> None:
        """Mutation detection: replacing XNOR with XOR produces inverted result and is detected."""
        s0 = 0xF0F0F0F0
        s1 = 0x0F0F0F0F
        expected_xnor, _ = ref_s_xnor_b32(s0, s1)
        mutated_xor = u32(s0 ^ s1)

        self.assertNotEqual(expected_xnor, mutated_xor)
        self.assertEqual(expected_xnor, u32(~mutated_xor))

        spv_xnor = support.translate_lines("ps_mut_xnor", [
            f"s_mov_b32 s0, {s0:#x}",
            f"s_mov_b32 s1, {s1:#x}",
            "s_xnor_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        spv_xor = support.translate_lines("ps_mut_xor", [
            f"s_mov_b32 s0, {s0:#x}",
            f"s_mov_b32 s1, {s1:#x}",
            "s_xor_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        res_xnor = support.mrt0_words(spv_xnor)[0]
        res_xor = support.mrt0_words(spv_xor)[0]
        self.assertEqual(res_xnor, expected_xnor)
        self.assertEqual(res_xor, mutated_xor)
        self.assertNotEqual(res_xnor, res_xor)

    def test_mutation_detection_swap_and_or(self) -> None:
        """Mutation detection: replacing XNOR with AND or OR produces distinct outputs."""
        s0 = 0x33333333
        s1 = 0x55555555
        xnor_val, _ = ref_s_xnor_b32(s0, s1)
        and_val = u32(s0 & s1)
        or_val = u32(s0 | s1)

        self.assertNotEqual(xnor_val, and_val)
        self.assertNotEqual(xnor_val, or_val)


if __name__ == "__main__":
    unittest.main()
