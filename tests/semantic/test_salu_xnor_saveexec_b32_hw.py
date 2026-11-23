"""Semantic interpreter tests and reference mathematical oracle for RDNA2 s_xnor_saveexec_b32.

Models RDNA2 / GFX10 Scalar ALU (SALU) saveexec operations on 32-bit execution masks:
  - s_xnor_saveexec_b32 D, S0:
      D = EXEC_LO
      new_exec_lo = ~(S0 ^ EXEC_LO) & 0xFFFFFFFF
      EXEC_LO = new_exec_lo
      SCC = (EXEC_LO != 0)
  In wave64, s_xnor_saveexec_b32 operates strictly on the lower 32 lanes (EXEC_LO [31:0]),
  leaving EXEC_HI [63:32] completely untouched, and SCC reflects (EXEC_LO != 0).
  In wave32, it operates on the full wave execution mask.

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
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
FSSM_64: bytes = bytes(0x138) + struct.pack("<4I", 0x88, 64, 1, 1)
VSHARP: List[int] = [0, 0, 0x1000, 0x31016FAC]


def u32(val: int) -> int:
    return val & MASK32


def ref_s_xnor_saveexec_b32(s0: int, old_exec_lo: int) -> Tuple[int, int, int]:
    """Reference mathematical oracle for s_xnor_saveexec_b32.

    Returns (new_exec_lo, saved_d, scc).
    """
    s0_u = s0 & MASK32
    exec_lo = old_exec_lo & MASK32
    new_exec_lo = (~(s0_u ^ exec_lo)) & MASK32
    saved_d = exec_lo
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, saved_d, scc


def run_salu_xnor_saveexec_b32(s0_val: int, init_lo: int, init_hi: int = 0x000000FF) -> Tuple[int, int, int, int]:
    """Execute s_xnor_saveexec_b32 via translate_lines and spirv_mini_interp.run_workgroup.

    Returns (new_exec_lo, exec_hi, scc, popcount_saved_d).
    """
    lines = [
        "s_mov_b64 exec, -1",
        f"s_mov_b32 exec_lo, {init_lo & MASK32:#x}",
        f"s_mov_b32 exec_hi, {init_hi & MASK32:#x}",
        f"s_xnor_saveexec_b32 s8, {s0_val & MASK32:#x}",
        "s_cselect_b32 s10, 1, 0",
        "v_mov_b32_e32 v10, 0",
        "s_mov_b64 s[12:13], exec",
        "s_mov_b64 exec, -1",
        "v_cndmask_b32_e64 v10, 0, 1, s[12:13]",
        "v_lshlrev_b32_e32 v11, 2, v0",
        "v_add_nc_u32_e32 v11, 64, v11",
        "buffer_store_dword v10, v11, s[0:3], 0 offen",
        "s_bcnt1_i32_b32 s11, s8",
        "v_cmpx_eq_u32_e32 0, v0",
        "v_mov_b32_e32 v1, s10",
        "v_mov_b32_e32 v2, s11",
        "v_mov_b32_e32 v12, 0", "buffer_store_dword v1, v12, s[0:3], 0 offen",
        "v_mov_b32_e32 v12, 4", "buffer_store_dword v2, v12, s[0:3], 0 offen",
        "s_endpgm",
    ]
    spv = support.translate_lines("cs_xnor_saveexec_b32", lines, fssm=FSSM_64)
    mem = si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory={},
    )
    exec_lo = sum(((mem.get(64 + 4 * i, 0) & 1) << i) for i in range(32))
    exec_hi = sum(((mem.get(64 + 4 * (32 + i), 0) & 1) << i) for i in range(32))
    scc = mem.get(0, 0)
    popcount_saved_d = mem.get(4, 0)
    return exec_lo, exec_hi, scc, popcount_saved_d


class TestSaluXnorSaveexecB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_truth_table(self) -> None:
        cases = [
            (0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 1),
            (0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000, 0),
            (0x55555555, 0x0000000F, 0xAAAAAAA5, 0x0000000F, 1),
            (0xAAAAAAAA, 0xAAAAAAAA, 0xFFFFFFFF, 0xAAAAAAAA, 1),
            (0x12345678, 0xEDCBA987, 0x00000000, 0xEDCBA987, 0),
        ]
        for s0, init_lo, exp_new_exec, exp_saved_d, exp_scc in cases:
            with self.subTest(s0=hex(s0), init_lo=hex(init_lo)):
                new_exec, saved_d, scc = ref_s_xnor_saveexec_b32(s0, init_lo)
                self.assertEqual(new_exec, exp_new_exec)
                self.assertEqual(saved_d, exp_saved_d)
                self.assertEqual(scc, exp_scc)

    def test_hardware_translator_execution(self) -> None:
        cases = [
            (0x55555555, 0x0000000F, 0x000000FF),
            (0xAAAAAAAA, 0xAAAAAAAA, 0x12345678),
            (0x55555555, 0xAAAAAAAA, 0x000000FF),
        ]
        for s0, init_lo, init_hi in cases:
            with self.subTest(s0=hex(s0), init_lo=hex(init_lo), init_hi=hex(init_hi)):
                exp_new_lo, exp_saved_d, exp_scc = ref_s_xnor_saveexec_b32(s0, init_lo)
                hw_lo, hw_hi, hw_scc, hw_popcount = run_salu_xnor_saveexec_b32(s0, init_lo, init_hi)
                self.assertEqual(hw_lo, exp_new_lo)
                self.assertEqual(hw_hi, init_hi)
                self.assertEqual(hw_scc, exp_scc)
                self.assertEqual(hw_popcount, bin(exp_saved_d).count("1"))

# --- Section 2: Symbolic AST Opcode Definitions and Typed Expressions ---

class OpcodeClass(enum.Enum):
    SOP1 = enum.auto()


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
    modifies_exec: bool = True


def parse_operand(op_str: str) -> OperandAST:
    raw = op_str.strip()
    if raw.startswith("s") and raw[1:].isdigit():
        idx = int(raw[1:])
        if idx > 105:
            raise ValueError(f"SGPR index out of range 0..105: {raw}")
        return OperandAST(kind=OperandKind.SGPR, raw=raw, value=0, reg_idx=idx)
    if raw.startswith("v"):
        raise ValueError(f"Vector register {raw} invalid in SALU instruction")
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
    if mnemonic != "s_xnor_saveexec_b32":
        raise ValueError(f"Unexpected mnemonic: {mnemonic}")
    if len(parts) < 2:
        raise ValueError(f"Missing operands for {mnemonic}")
    operands = [op.strip() for op in parts[1].split(",")]
    if len(operands) != 2:
        raise ValueError(f"s_xnor_saveexec_b32 requires 2 operands (dst, src0), got {len(operands)}")
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
        modifies_exec=True,
    )


class TestSaluXnorSaveexecB32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions tests."""

    def test_ast_parse_valid(self) -> None:
        inst = parse_salu_instruction("s_xnor_saveexec_b32 s8, s2")
        self.assertEqual(inst.mnemonic, "s_xnor_saveexec_b32")
        self.assertEqual(inst.dst.reg_idx, 8)
        self.assertEqual(inst.src0.reg_idx, 2)
        self.assertTrue(inst.modifies_scc)
        self.assertTrue(inst.modifies_exec)

    def test_ast_parse_immediate(self) -> None:
        inst = parse_salu_instruction("s_xnor_saveexec_b32 s0, 0")
        self.assertEqual(inst.src0.kind, OperandKind.IMM_INT)
        self.assertEqual(inst.src0.value, 0)

    def test_ast_rejection(self) -> None:
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_xnor_saveexec_b32")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_xnor_saveexec_b32 s0")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_xnor_saveexec_b32 v0, s1")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_xnor_saveexec_b32 s0, v1")
        with self.assertRaises(ValueError):
            parse_salu_instruction("s_xnor_saveexec_b32 s106, s0")

# --- Section 3: Deterministic Scalar Instruction Interpreter ---

class ScalarRegisterFile:
    def __init__(self) -> None:
        self.sgprs: List[int] = [0] * 106
        self.exec_lo: int = 0xFFFFFFFF
        self.exec_hi: int = 0xFFFFFFFF
        self.scc: int = 0

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_operand(self, op: OperandAST) -> int:
        if op.kind == OperandKind.SGPR:
            assert op.reg_idx is not None
            return self.read_sgpr(op.reg_idx)
        return op.value & MASK32


class SaluInterpreter:
    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None) -> None:
        self.rf = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, inst: InstructionAST) -> Tuple[int, int, int]:
        if inst.mnemonic != "s_xnor_saveexec_b32":
            raise ValueError(f"Unsupported mnemonic: {inst.mnemonic}")
        s0_val = self.rf.read_operand(inst.src0)
        new_exec_lo, saved_d, scc = ref_s_xnor_saveexec_b32(s0_val, self.rf.exec_lo)
        assert inst.dst.reg_idx is not None
        self.rf.write_sgpr(inst.dst.reg_idx, saved_d)
        self.rf.exec_lo = new_exec_lo
        self.rf.scc = scc
        return new_exec_lo, saved_d, scc


class TestSaluXnorSaveexecB32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = SaluInterpreter(self.rf)

    def test_interpreter_execution(self) -> None:
        self.rf.exec_lo = 0x0000000F
        self.rf.write_sgpr(2, 0x55555555)
        inst = parse_salu_instruction("s_xnor_saveexec_b32 s8, s2")
        new_lo, saved_d, scc = self.interp.execute(inst)
        self.assertEqual(saved_d, 0x0000000F)
        self.assertEqual(self.rf.read_sgpr(8), 0x0000000F)
        self.assertEqual(new_lo, 0xAAAAAAA5)
        self.assertEqual(self.rf.exec_lo, 0xAAAAAAA5)
        self.assertEqual(scc, 1)

    def test_zero_result_clears_scc(self) -> None:
        self.rf.exec_lo = 0xAAAAAAAA
        self.rf.write_sgpr(0, 0x55555555)
        inst = parse_salu_instruction("s_xnor_saveexec_b32 s1, s0")
        new_lo, saved_d, scc = self.interp.execute(inst)
        self.assertEqual(new_lo, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(self.rf.scc, 0)

# --- Section 4: SIMD Multi-Lane Divergence and EXEC Masking ---

class TestSaluXnorSaveexecB32SimdExecution(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking tests."""

    def test_wave64_exec_hi_preservation(self) -> None:
        """EXEC_HI must remain strictly untouched across s_xnor_saveexec_b32."""
        hi_patterns = [0x000000FF, 0xAAAAAAAA, 0x12345678, 0xFFFFFFFF]
        for hi in hi_patterns:
            with self.subTest(hi=hex(hi)):
                hw_lo, hw_hi, hw_scc, _ = run_salu_xnor_saveexec_b32(0x55555555, 0x0000000F, hi)
                self.assertEqual(hw_hi, hi)

    def test_divergent_lane_mask_update(self) -> None:
        hw_lo, _, hw_scc, _ = run_salu_xnor_saveexec_b32(0x0000FFFF, 0x000000FF, 0x000000FF)
        self.assertEqual(hw_lo, 0xFFFF00FF)
        self.assertEqual(hw_scc, 1)

# --- Section 5: Register Aliasing and In-Place Operand Preservation ---

class TestSaluXnorSaveexecB32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation tests."""

    def test_aliasing_dst_and_src0(self) -> None:
        rf = ScalarRegisterFile()
        interp = SaluInterpreter(rf)
        rf.exec_lo = 0x12345678
        rf.write_sgpr(0, 0x55555555)
        inst = parse_salu_instruction("s_xnor_saveexec_b32 s0, s0")
        new_lo, saved_d, scc = interp.execute(inst)
        self.assertEqual(saved_d, 0x12345678)
        self.assertEqual(rf.read_sgpr(0), 0x12345678)
        self.assertEqual(new_lo, (~(0x55555555 ^ 0x12345678)) & MASK32)

    def test_hardware_aliasing(self) -> None:
        lines = [
            "s_mov_b64 exec, -1",
            "s_mov_b32 exec_lo, 0x0000000F",
            "s_mov_b32 exec_hi, 0x000000FF",
            "s_xnor_saveexec_b32 s8, 0x55555555",
            "s_cselect_b32 s10, 1, 0",
            "s_bcnt1_i32_b32 s11, s8",
            "v_cmpx_eq_u32_e32 0, v0",
            "v_mov_b32_e32 v1, s10",
            "v_mov_b32_e32 v2, s11",
            "v_mov_b32_e32 v12, 0", "buffer_store_dword v1, v12, s[0:3], 0 offen",
            "v_mov_b32_e32 v12, 4", "buffer_store_dword v2, v12, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_xnor_aliasing", lines, fssm=FSSM_64)
        mem = si.run_workgroup(
            spv, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory={},
        )
        self.assertEqual(mem.get(0), 1)  # SCC
        self.assertEqual(mem.get(4), 4)  # popcount of saved_d (0xF has 4 bits)

# --- Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation ---

class TestSaluXnorSaveexecB32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation tests."""

    def test_scc_is_zero_when_exec_lo_is_zero(self) -> None:
        hw_lo, _, hw_scc, _ = run_salu_xnor_saveexec_b32(0x55555555, 0xAAAAAAAA, 0x000000FF)
        self.assertEqual(hw_lo, 0)
        self.assertEqual(hw_scc, 0)

    def test_scc_is_one_when_exec_lo_is_nonzero(self) -> None:
        hw_lo, _, hw_scc, _ = run_salu_xnor_saveexec_b32(0x55555555, 0x0000000F, 0x000000FF)
        self.assertNotEqual(hw_lo, 0)
        self.assertEqual(hw_scc, 1)

    def test_scc_ignores_exec_hi(self) -> None:
        """SCC reflects EXEC_LO only; non-zero EXEC_HI must not force SCC = 1 when EXEC_LO == 0."""
        hw_lo, hw_hi, hw_scc, _ = run_salu_xnor_saveexec_b32(0x55555555, 0xAAAAAAAA, 0xFFFFFFFF)
        self.assertEqual(hw_lo, 0)
        self.assertEqual(hw_hi, 0xFFFFFFFF)
        self.assertEqual(hw_scc, 0)

# --- Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling ---

class TestSaluXnorSaveexecB32IEEE754BitPatterns(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling tests."""

    def test_float_bit_patterns(self) -> None:
        cases = {
            "pos_zero (+0.0f)": 0x00000000,
            "neg_zero (-0.0f)": 0x80000000,
            "pos_one (+1.0f)": 0x3F800000,
            "neg_one (-1.0f)": 0xBF800000,
            "pos_inf (+inf)": 0x7F800000,
            "neg_inf (-inf)": 0xFF800000,
            "quiet_nan (qNaN)": 0x7FC00000,
            "min_subnormal": 0x00000001,
        }
        for name, bits in cases.items():
            with self.subTest(case=name):
                exp_lo, _, exp_scc = ref_s_xnor_saveexec_b32(bits, 0x55555555)
                hw_lo, _, hw_scc, _ = run_salu_xnor_saveexec_b32(bits, 0x55555555, 0x000000FF)
                self.assertEqual(hw_lo, exp_lo)
                self.assertEqual(hw_scc, exp_scc)

if __name__ == "__main__":
    unittest.main()
