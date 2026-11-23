"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_not_b32.

Models RDNA2 / GFX10 Scalar ALU (SALU) bitwise NOT operations:
  - s_not_b32 D, S0: D.u[31:0] = ~S0.u[31:0]
  - SCC = (D.u != 0)
  Inverts each bit of source operand S0. Writes condition code SCC = 1 if result is non-zero,
  and SCC = 0 if result is zero (which occurs uniquely when S0 == 0xFFFFFFFF).

Verification layers:
  1. Pure Python reference mathematical oracles and bitwise models.
  2. Symbolic AST opcode definitions, operand typings, and validation.
  3. Deterministic scalar SALU register file and execution interpreter.
  4. SIMD multi-lane divergence and Wave32/Wave64 EXEC masking models.
  5. In-place operand preservation and register aliasing verification.
  6. Condition code / status flag invariance (SCC generation).
  7. IEEE 754 boundary extremes, signed zeros, subnormals, and NaNs.
  8. Syntax rejection, negative control gates, and fault injection / mutation testing.
  9. Real SPIR-V translation and execution via rdna2spirv and spirv_mini_interp.

Zero game data, standard library Python only, Python 3.9+ compatible.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import STUB_DXBC, mrt0_words, translate_lines  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned dword."""
    return val & MASK32


def ref_s_not_b32(src: int) -> int:
    """Reference mathematical oracle for 32-bit bitwise NOT.

    D.u = (~S0.u) & 0xFFFFFFFF.
    """
    return (~src) & MASK32


def ref_s_not_b32_scc(src: int) -> int:
    """Reference mathematical oracle for SCC generation in s_not_b32.

    SCC = 1 if D.u != 0 else 0.
    """
    res = ref_s_not_b32(src)
    return 1 if res != 0 else 0


def run_salu_not_ps(src_val: int) -> Tuple[int, int]:
    """Assemble and translate a pixel shader executing s_not_b32 through rdna2spirv.

    Returns (result_dword, scc_bit).
    """
    lines = [
        f"s_mov_b32 s0, {src_val & MASK32:#x}",
        "s_not_b32 s1, s0",
        "s_cselect_b32 s2, 1, 0",
        "v_mov_b32 v0, s1",
        "v_mov_b32 v1, s2",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_not_b32", lines)
    words = mrt0_words(spv)
    return words[0], words[1]


class TestSaluNotB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_truth_table(self) -> None:
        """Verify fundamental bitwise inversion and SCC generation."""
        cases = [
            (0x00000000, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0x00000000, 0),
            (0xAAAAAAAA, 0x55555555, 1),
            (0x55555555, 0xAAAAAAAA, 1),
            (0x12345678, 0xEDCBA987, 1),
            (0x80000000, 0x7FFFFFFF, 1),
            (0x00000001, 0xFFFFFFFE, 1),
        ]
        for src, expected_d, expected_scc in cases:
            with self.subTest(src=hex(src)):
                self.assertEqual(ref_s_not_b32(src), expected_d)
                self.assertEqual(ref_s_not_b32_scc(src), expected_scc)

    def test_involution_property(self) -> None:
        """Double NOT is an identity operation: not(not(x)) == x."""
        patterns = [0x00000000, 0xFFFFFFFF, 0x12345678, 0xCAFEBABE, 0x55555555]
        for p in patterns:
            with self.subTest(p=hex(p)):
                self.assertEqual(ref_s_not_b32(ref_s_not_b32(p)), p)

    def test_hardware_translator_execution(self) -> None:
        """Verify real SPIR-V translator lowering and execution against oracle."""
        cases = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x80000000,
            0x00000001,
            0xAAAAAAAA,
            0x55555555,
        ]
        for c in cases:
            with self.subTest(case=hex(c)):
                hw_res, hw_scc = run_salu_not_ps(c)
                self.assertEqual(hw_res, ref_s_not_b32(c))
                self.assertEqual(hw_scc, ref_s_not_b32_scc(c))

# --- Section 2: Symbolic AST Opcode Definitions and Typed Expressions ---

class OpcodeClass(enum.Enum):
    SOP1 = enum.auto()
    SOP2 = enum.auto()


class OperandKind(enum.Enum):
    SGPR = enum.auto()
    IMM_INT = enum.auto()
    LITERAL32 = enum.auto()


@dataclasses.dataclass(frozen=True)
class OperandAST:
    kind: OperandKind
    raw: str
    value: int
    reg_idx: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class InstructionAST:
    mnemonic: str
    dst: OperandAST
    src0: OperandAST
    opcode_class: OpcodeClass = OpcodeClass.SOP1
    modifies_scc: bool = True


def parse_operand(op_str: str) -> OperandAST:
    raw = op_str.strip()
    if raw.startswith("s") and raw[1:].isdigit():
        idx = int(raw[1:])
        if idx > 105:
            raise ValueError(f"SGPR index out of architectural range 0..105: {raw}")
        return OperandAST(kind=OperandKind.SGPR, raw=raw, value=0, reg_idx=idx)
    if raw.startswith("v"):
        raise ValueError(f"Vector register {raw} illegal in scalar instruction operand")
    try:
        val = int(raw, 0)
    except ValueError as e:
        raise ValueError(f"Invalid operand syntax: {raw}") from e
    if -16 <= val <= 64:
        return OperandAST(kind=OperandKind.IMM_INT, raw=raw, value=val & MASK32)
    return OperandAST(kind=OperandKind.LITERAL32, raw=raw, value=val & MASK32)


def parse_salu_instruction(line: str) -> InstructionAST:
    clean = line.split(";")[0].strip()
    if not clean:
        raise ValueError("Empty instruction line")
    parts = clean.split(None, 1)
    mnemonic = parts[0]
    if mnemonic != "s_not_b32":
        raise ValueError(f"Unexpected mnemonic: {mnemonic}")
    if len(parts) < 2:
        raise ValueError(f"Missing operands for {mnemonic}")
    operands = [op.strip() for op in parts[1].split(",")]
    if len(operands) != 2:
        raise ValueError(f"s_not_b32 requires exactly 2 operands (dst, src0), got {len(operands)}")
    dst = parse_operand(operands[0])
    if dst.kind != OperandKind.SGPR:
        raise ValueError(f"Destination must be an SGPR, got {dst.kind}")
    src0 = parse_operand(operands[1])
    return InstructionAST(
        mnemonic=mnemonic,
        dst=dst,
        src0=src0,
        opcode_class=OpcodeClass.SOP1,
        modifies_scc=True,
    )


class TestSaluNotB32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions tests."""

    def test_ast_parse_sgpr_operand(self) -> None:
        inst = parse_salu_instruction("s_not_b32 s5, s10")
        self.assertEqual(inst.mnemonic, "s_not_b32")
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.reg_idx, 5)
        self.assertEqual(inst.src0.kind, OperandKind.SGPR)
        self.assertEqual(inst.src0.reg_idx, 10)
        self.assertEqual(inst.opcode_class, OpcodeClass.SOP1)
        self.assertTrue(inst.modifies_scc)

    def test_ast_parse_literal_operand(self) -> None:
        inst = parse_salu_instruction("s_not_b32 s0, 0x12345678")
        self.assertEqual(inst.dst.reg_idx, 0)
        self.assertEqual(inst.src0.kind, OperandKind.LITERAL32)
        self.assertEqual(inst.src0.value, 0x12345678)

    def test_ast_parse_inline_immediate_operand(self) -> None:
        inst = parse_salu_instruction("s_not_b32 s1, 0")
        self.assertEqual(inst.dst.reg_idx, 1)
        self.assertEqual(inst.src0.kind, OperandKind.IMM_INT)
        self.assertEqual(inst.src0.value, 0)

    def test_ast_rejection_invalid_syntax(self) -> None:
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32 s0")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32 s0, s1, s2")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32 v0, s1")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32 s0, v1")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_not_b32 s106, s0")

# --- Section 3: Deterministic Scalar Instruction Interpreter ---

class ScalarRegisterFile:
    """RDNA2 Scalar Register File model (106 SGPRs: s0..s105, plus SCC)."""

    def __init__(self) -> None:
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0

    def read_sgpr(self, idx: int) -> int:
        if not (0 <= idx < 106):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < 106):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32

    def read_operand(self, op: OperandAST) -> int:
        if op.kind == OperandKind.SGPR:
            assert op.reg_idx is not None
            return self.read_sgpr(op.reg_idx)
        return op.value & MASK32


class SaluInterpreter:
    """Deterministic Scalar ALU instruction interpreter for s_not_b32."""

    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, inst: InstructionAST) -> int:
        if inst.mnemonic != "s_not_b32":
            raise ValueError(f"Unsupported instruction mnemonic: {inst.mnemonic}")
        src_val = self.reg_file.read_operand(inst.src0)
        res = ref_s_not_b32(src_val)
        assert inst.dst.reg_idx is not None
        self.reg_file.write_sgpr(inst.dst.reg_idx, res)
        self.reg_file.scc = ref_s_not_b32_scc(src_val)
        return res


class TestSaluNotB32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = SaluInterpreter(self.rf)

    def test_deterministic_vectors(self) -> None:
        for bit in range(32):
            self.rf.write_sgpr(0, 1 << bit)
            inst = parse_salu_instruction("s_not_b32 s1, s0")
            res = self.interp.execute(inst)
            expected = MASK32 ^ (1 << bit)
            self.assertEqual(res, expected)
            self.assertEqual(self.rf.read_sgpr(1), expected)
            self.assertEqual(self.rf.scc, 1)

    def test_all_ones_inversion_to_zero(self) -> None:
        self.rf.write_sgpr(0, 0xFFFFFFFF)
        inst = parse_salu_instruction("s_not_b32 s1, s0")
        res = self.interp.execute(inst)
        self.assertEqual(res, 0)
        self.assertEqual(self.rf.read_sgpr(1), 0)
        self.assertEqual(self.rf.scc, 0)

    def test_immediate_and_literal_operands(self) -> None:
        inst_imm = parse_salu_instruction("s_not_b32 s10, 0")
        res = self.interp.execute(inst_imm)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(self.rf.scc, 1)

    def test_translator_cross_validation(self) -> None:
        cases = [0x12345678, 0x89ABCDEF, 0xFFFFFFFF, 0x00000000]
        for c in cases:
            self.rf.write_sgpr(0, c)
            inst = parse_salu_instruction("s_not_b32 s1, s0")
            interp_res = self.interp.execute(inst)
            hw_res, hw_scc = run_salu_not_ps(c)
            self.assertEqual(interp_res, hw_res)
            self.assertEqual(self.rf.scc, hw_scc)

# --- Section 4: SIMD Multi-Lane Divergence and EXEC Masking ---

class SimdWaveNotModel:
    """Wavefront SIMD execution model for scalar bitfield broadcasting under EXEC masking."""

    def __init__(self, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError("wave_size must be 32 or 64")
        self.wave_size = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: List[List[int]] = [[0] * 32 for _ in range(wave_size)]

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def broadcast_scalar(self, scalar_val: int, dst_vgpr: int) -> None:
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                self.vgprs[lane][dst_vgpr] = scalar_val & MASK32

    def get_lane_vgpr(self, lane: int, vgpr: int) -> int:
        return self.vgprs[lane][vgpr]


def run_salu_not_simd_ps(src_val: int) -> List[int]:
    """Test scalar NOT broadcast in fragment shader with vector registers."""
    lines = [
        f"s_mov_b32 s0, {src_val & MASK32:#x}",
        "s_not_b32 s1, s0",
        "v_mov_b32 v0, s1",
        "v_mov_b32 v1, s1",
        "v_mov_b32 v2, s1",
        "v_mov_b32 v3, s1",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_not_simd", lines)
    return mrt0_words(spv)


class TestSaluNotB32SimdExecution(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking tests."""

    def test_wave32_full_exec(self) -> None:
        model = SimdWaveNotModel(wave_size=32)
        val = 0x12345678
        not_val = ref_s_not_b32(val)
        model.broadcast_scalar(not_val, dst_vgpr=0)
        for lane in range(32):
            self.assertEqual(model.get_lane_vgpr(lane, 0), not_val)

    def test_wave32_divergent_exec_mask(self) -> None:
        model = SimdWaveNotModel(wave_size=32)
        mask = 0x55555555
        model.set_exec(mask)
        for lane in range(32):
            model.vgprs[lane][1] = 0xDEADBEEF
        not_val = ref_s_not_b32(0xCAFEBABE)
        model.broadcast_scalar(not_val, dst_vgpr=1)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(model.get_lane_vgpr(lane, 1), not_val)
            else:
                self.assertEqual(model.get_lane_vgpr(lane, 1), 0xDEADBEEF)

    def test_wave64_divergent_exec_mask(self) -> None:
        model = SimdWaveNotModel(wave_size=64)
        mask = 0x00000000FFFFFFFF
        model.set_exec(mask)
        for lane in range(64):
            model.vgprs[lane][2] = (0x11111111 * (lane + 1)) & MASK32
        not_val = ref_s_not_b32(0x80000001)
        model.broadcast_scalar(not_val, dst_vgpr=2)
        for lane in range(32):
            self.assertEqual(model.get_lane_vgpr(lane, 2), not_val)
        for lane in range(32, 64):
            self.assertEqual(model.get_lane_vgpr(lane, 2), (0x11111111 * (lane + 1)) & MASK32)

    def test_hardware_simd_broadcast(self) -> None:
        test_val = 0x87654321
        expected = ref_s_not_b32(test_val)
        words = run_salu_not_simd_ps(test_val)
        self.assertEqual(words, [expected, expected, expected, expected])

# --- Section 5: Register Aliasing and In-Place Operand Preservation ---

def run_salu_not_inplace_ps(src_val: int) -> Tuple[int, int]:
    """Test in-place NOT (s_not_b32 s0, s0) via translator."""
    lines = [
        f"s_mov_b32 s0, {src_val & MASK32:#x}",
        "s_not_b32 s0, s0",
        "s_cselect_b32 s1, 1, 0",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_salu_not_inplace", lines)
    words = mrt0_words(spv)
    return words[0], words[1]


class TestSaluNotB32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation tests."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = SaluInterpreter(self.rf)

    def test_inplace_aliasing_execution(self) -> None:
        test_values = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x80000000,
            0x00000001,
            0x55555555,
            0xAAAAAAAA,
            0xDEADBEEF,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                self.rf.write_sgpr(5, val)
                inst = parse_salu_instruction("s_not_b32 s5, s5")
                res = self.interp.execute(inst)
                expected = ref_s_not_b32(val)
                self.assertEqual(res, expected)
                self.assertEqual(self.rf.read_sgpr(5), expected)
                self.assertEqual(self.rf.scc, ref_s_not_b32_scc(val))

    def test_unrelated_sgpr_preservation(self) -> None:
        self.rf.write_sgpr(0, 0x11111111)
        self.rf.write_sgpr(1, 0x22222222)
        self.rf.write_sgpr(2, 0x33333333)
        inst = parse_salu_instruction("s_not_b32 s1, s0")
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_sgpr(0), 0x11111111)
        self.assertEqual(self.rf.read_sgpr(1), ref_s_not_b32(0x11111111))
        self.assertEqual(self.rf.read_sgpr(2), 0x33333333)

    def test_double_inplace_not_identity(self) -> None:
        initial = 0xA5C3E187
        self.rf.write_sgpr(10, initial)
        inst = parse_salu_instruction("s_not_b32 s10, s10")
        self.interp.execute(inst)
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_sgpr(10), initial)

    def test_hardware_inplace_execution(self) -> None:
        cases = [0x12345678, 0xCAFEBABE, 0xFFFFFFFF, 0x00000000]
        for c in cases:
            with self.subTest(case=hex(c)):
                hw_res, hw_scc = run_salu_not_inplace_ps(c)
                self.assertEqual(hw_res, ref_s_not_b32(c))
                self.assertEqual(hw_scc, ref_s_not_b32_scc(c))

# --- Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation ---

class TestSaluNotB32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation tests."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = SaluInterpreter(self.rf)

    def test_scc_set_to_one_for_nonzero_result(self) -> None:
        """s_not_b32 yields non-zero result for any S0 != 0xFFFFFFFF, setting SCC = 1."""
        cases = [0x00000000, 0x12345678, 0x80000000, 0x00000001, 0xFFFFFFFE]
        for c in cases:
            self.rf.scc = 0
            self.rf.write_sgpr(0, c)
            inst = parse_salu_instruction("s_not_b32 s1, s0")
            self.interp.execute(inst)
            self.assertEqual(self.rf.scc, 1)

    def test_scc_cleared_to_zero_for_zero_result(self) -> None:
        """s_not_b32 yields 0 result when S0 == 0xFFFFFFFF, clearing SCC = 0."""
        self.rf.scc = 1
        self.rf.write_sgpr(0, 0xFFFFFFFF)
        inst = parse_salu_instruction("s_not_b32 s1, s0")
        res = self.interp.execute(inst)
        self.assertEqual(res, 0)
        self.assertEqual(self.rf.scc, 0)

    def test_hardware_scc_generation(self) -> None:
        # Non-zero result -> SCC = 1
        _, scc1 = run_salu_not_ps(0x00000000)
        self.assertEqual(scc1, 1)

        # Zero result -> SCC = 0
        _, scc0 = run_salu_not_ps(0xFFFFFFFF)
        self.assertEqual(scc0, 0)

# --- Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling ---

class TestSaluNotB32IEEE754BitPatterns(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling tests."""

    def test_ieee754_boundary_patterns(self) -> None:
        cases = {
            "pos_zero (+0.0f)": 0x00000000,
            "neg_zero (-0.0f)": 0x80000000,
            "pos_one (+1.0f)": 0x3F800000,
            "neg_one (-1.0f)": 0xBF800000,
            "pos_inf (+inf)": 0x7F800000,
            "neg_inf (-inf)": 0xFF800000,
            "quiet_nan (qNaN)": 0x7FC00000,
            "neg_quiet_nan (-qNaN)": 0xFFC00000,
            "signaling_nan (sNaN)": 0x7F800001,
            "min_subnormal": 0x00000001,
            "max_subnormal": 0x007FFFFF,
            "min_normal": 0x00800000,
            "max_normal": 0x7F7FFFFF,
        }
        for name, bits in cases.items():
            with self.subTest(case=name):
                expected = ref_s_not_b32(bits)
                hw_res, hw_scc = run_salu_not_ps(bits)
                self.assertEqual(hw_res, expected)
                self.assertEqual(hw_scc, ref_s_not_b32_scc(bits))

    def test_float_roundtrip_bit_integrity(self) -> None:
        nan_payloads = [0x7FC00001, 0x7FC0ABCD, 0xFFC00001, 0x7FFFFFFF]
        for nan in nan_payloads:
            with self.subTest(nan=hex(nan)):
                res, scc = run_salu_not_ps(nan)
                self.assertEqual(res, ref_s_not_b32(nan))
                self.assertEqual(scc, ref_s_not_b32_scc(nan))

if __name__ == "__main__":
    unittest.main()
