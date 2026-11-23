from __future__ import annotations

"""Hardware semantic test and reference oracle for s_xnor_saveexec_b64.

AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP1 Instruction:
    s_xnor_saveexec_b64 D, S0

Hardware Semantics:
    D = EXEC;
    EXEC = ~(S0 ^ EXEC) & 0xFFFFFFFFFFFFFFFF;
    SCC = (EXEC != 0);

Key Architectural Characteristics:
  1. Operates on 64-bit scalar register pairs (SGPR pairs), VCC, or EXEC.
  2. D latches the prior EXEC mask before EXEC is modified.
  3. Bitwise XNOR allows previously inactive lanes (EXEC bit = 0) to become active
     if S0 bit = 0, because ~(0 ^ 0) = ~0 = 1.
  4. SCC reflects whether any lane remains active in the new EXEC mask.
  5. Translates through tools/rdna2spirv.py via support.translate_lines and
     executes on spirv_mini_interp.run_workgroup.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import enum
import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

# Architectural Bitmasks and Constants
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK32: int = 0xFFFF_FFFF
VSHARP: list[int] = [0, 0, 0x1000, 0x31016FAC]
SCC_OFFSET: int = 256
BCNT_OFFSET: int = 260
FF1_OFFSET: int = 264


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & ALL64


def ref_s_xnor_saveexec_b64(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Pure mathematical reference oracle for s_xnor_saveexec_b64.

    Args:
        s0: 64-bit source operand.
        old_exec: 64-bit current EXEC mask.

    Returns:
        Tuple[int, int, int]: (new_exec, d_saved, scc)
    """
    s0_clean = u64(s0)
    exec_clean = u64(old_exec)
    new_exec = u64(~(s0_clean ^ exec_clean))
    d_saved = exec_clean
    scc = 1 if (new_exec != 0) else 0
    return new_exec, d_saved, scc


def build_cs_fssm(threads: int = 64) -> bytes:
    """Build compute shader header blob declaring 64 threads per workgroup."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, 0x8088, threads, 1, 1)
    return bytes(header)


def run_hw_xnor_saveexec(s0: int, old_exec: int) -> Tuple[int, int, int, int]:
    """Translate and run s_xnor_saveexec_b64 through rdna2spirv and spirv_mini_interp.

    Returns:
        Tuple[int, int, int, int]: (new_exec, scc, d_popcount, d_ff1)
    """
    s0_clean = u64(s0)
    exec_clean = u64(old_exec)

    lines = [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "v_mbcnt_hi_u32_b32 v1, -1, v1",
        "v_mov_b32_e32 v2, 0",
        f"s_mov_b32 s12, {s0_clean & MASK32:#x}",
        f"s_mov_b32 s13, {s0_clean >> 32:#x}",
        f"s_mov_b32 exec_lo, {exec_clean & MASK32:#x}",
        f"s_mov_b32 exec_hi, {exec_clean >> 32:#x}",
        "s_xnor_saveexec_b64 s[6:7], s[12:13]",
        "s_cselect_b32 s8, 1, 0",
        "v_mov_b32_e32 v2, 1",
        "s_mov_b64 exec, -1",
        "s_bcnt1_i32_b64 s9, s[6:7]",
        "s_ff1_i32_b64 s10, s[6:7]",
        "v_lshlrev_b32_e32 v21, 2, v1",
        "buffer_store_dword v2, v21, s[0:3], 0 offen",
        "v_cmpx_eq_u32_e32 0, v1",
        f"v_mov_b32_e32 v20, s8",
        f"v_mov_b32_e32 v21, {SCC_OFFSET}",
        "buffer_store_dword v20, v21, s[0:3], 0 offen",
        f"v_mov_b32_e32 v20, s9",
        f"v_mov_b32_e32 v21, {BCNT_OFFSET}",
        "buffer_store_dword v20, v21, s[0:3], 0 offen",
        f"v_mov_b32_e32 v20, s10",
        f"v_mov_b32_e32 v21, {FF1_OFFSET}",
        "buffer_store_dword v20, v21, s[0:3], 0 offen",
        "s_endpgm"
    ]

    spv = support.translate_lines("cs_xnor_saveexec_hw", lines, fssm=build_cs_fssm(64))
    memory: Dict[int, int] = {}
    si.run_workgroup(
        spv, 64, 64,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=memory
    )

    new_exec = sum(memory.get(4 * lane, 0) << lane for lane in range(64))
    scc = memory.get(SCC_OFFSET, 0)
    bcnt = memory.get(BCNT_OFFSET, 0)
    ff1 = memory.get(FF1_OFFSET, 0)

    return new_exec, scc, bcnt, ff1


class TestSALUXnorSaveexecReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and hardware translation."""

    def test_oracle_all_ones(self) -> None:
        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(ALL64, ALL64)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(d_saved, ALL64)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros(self) -> None:
        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(0, 0)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(d_saved, 0)
        self.assertEqual(scc, 1)

    def test_oracle_complementary_masks_produce_zero_exec(self) -> None:
        s0 = 0x55555555_55555555
        old_exec = 0xAAAAAAAA_AAAAAAAA
        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d_saved, old_exec)
        self.assertEqual(scc, 0)

    def test_hardware_translation_against_reference(self) -> None:
        test_vectors = [
            (0x55555555_55555555, 0x00000000_FFFFFFFF),
            (0xAAAAAAAA_AAAAAAAA, 0xFFFFFFFF_00000000),
            (0x12345678_9ABCDEF0, 0x0FEDCBA9_87654321),
            (ALL64, 0x80000000_00000000),
        ]
        for s0, old_exec in test_vectors:
            with self.subTest(s0=hex(s0), old_exec=hex(old_exec)):
                expected_exec, expected_d, expected_scc = ref_s_xnor_saveexec_b64(s0, old_exec)
                hw_exec, hw_scc, hw_bcnt, hw_ff1 = run_hw_xnor_saveexec(s0, old_exec)
                self.assertEqual(hw_exec, expected_exec)
                self.assertEqual(hw_scc, expected_scc)
                self.assertEqual(hw_bcnt, bin(expected_d).count("1"))
                expected_ff1 = (expected_d & -expected_d).bit_length() - 1 if expected_d else 0xFFFFFFFF
                self.assertEqual(hw_ff1, expected_ff1)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class SALUOpcode(Enum):
    """SALU 64-bit SOP1 saveexec opcodes."""
    S_XNOR_SAVEEXEC_B64 = "s_xnor_saveexec_b64"
    S_XOR_SAVEEXEC_B64 = "s_xor_saveexec_b64"
    S_AND_SAVEEXEC_B64 = "s_and_saveexec_b64"
    S_OR_SAVEEXEC_B64 = "s_or_saveexec_b64"


class RegKind(Enum):
    """Register kinds for 64-bit operands."""
    SGPR_PAIR = "s"
    VCC = "vcc"
    EXEC = "exec"


@dataclass(frozen=True)
class Register64:
    """Represents a 64-bit register operand (e.g. s[6:7], vcc, exec)."""
    kind: RegKind
    index: int = 0

    def __post_init__(self) -> None:
        if self.kind == RegKind.SGPR_PAIR:
            if self.index < 0 or self.index > 104:
                raise ValueError(f"SGPR pair index out of range [0, 104]: {self.index}")
            if self.index % 2 != 0:
                raise ValueError(f"SGPR pair must be even aligned, got {self.index}")

    @property
    def name(self) -> str:
        if self.kind == RegKind.SGPR_PAIR:
            return f"s[{self.index}:{self.index + 1}]"
        return self.kind.value

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate64:
    """64-bit constant immediate operand."""
    val: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "val", u64(self.val))

    def __str__(self) -> str:
        return f"{self.val:#x}"


Operand64 = Union[Register64, Immediate64]


@dataclass(frozen=True)
class SXnorSaveexecInstruction:
    """Symbolic AST node for s_xnor_saveexec_b64."""
    dst: Register64
    src0: Operand64
    opcode: SALUOpcode = SALUOpcode.S_XNOR_SAVEEXEC_B64

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.SGPR_PAIR:
            raise ValueError(f"Destination must be an SGPR pair, got {self.dst}")

    def disassemble(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}"


class TestSymbolicAST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction_sgpr_src(self) -> None:
        dst = Register64(RegKind.SGPR_PAIR, 6)
        src = Register64(RegKind.SGPR_PAIR, 12)
        instr = SXnorSaveexecInstruction(dst=dst, src0=src)

        self.assertEqual(instr.opcode, SALUOpcode.S_XNOR_SAVEEXEC_B64)
        self.assertEqual(str(instr.dst), "s[6:7]")
        self.assertEqual(str(instr.src0), "s[12:13]")
        self.assertEqual(instr.disassemble(), "s_xnor_saveexec_b64 s[6:7], s[12:13]")

    def test_valid_ast_construction_special_regs(self) -> None:
        dst = Register64(RegKind.SGPR_PAIR, 8)
        instr_vcc = SXnorSaveexecInstruction(dst=dst, src0=Register64(RegKind.VCC))
        instr_exec = SXnorSaveexecInstruction(dst=dst, src0=Register64(RegKind.EXEC))

        self.assertEqual(instr_vcc.disassemble(), "s_xnor_saveexec_b64 s[8:9], vcc")
        self.assertEqual(instr_exec.disassemble(), "s_xnor_saveexec_b64 s[8:9], exec")

    def test_imm64_construction(self) -> None:
        dst = Register64(RegKind.SGPR_PAIR, 0)
        imm = Immediate64(0x12345678_9ABCDEF0)
        instr = SXnorSaveexecInstruction(dst=dst, src0=imm)

        self.assertEqual(instr.disassemble(), "s_xnor_saveexec_b64 s[0:1], 0x123456789abcdef0")


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Deterministic architectural scalar machine state."""

    def __init__(self) -> None:
        self.sgprs: List[int] = [0] * 106
        self.exec_mask: int = ALL64
        self.vcc: int = 0
        self.scc: int = 0

    def get_sgpr_pair(self, index: int) -> int:
        lo = self.sgprs[index] & MASK32
        hi = self.sgprs[index + 1] & MASK32
        return (hi << 32) | lo

    def set_sgpr_pair(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32
        self.sgprs[index + 1] = (val >> 32) & MASK32

    def get_operand_val(self, op: Operand64) -> int:
        if isinstance(op, Immediate64):
            return op.val
        if op.kind == RegKind.SGPR_PAIR:
            return self.get_sgpr_pair(op.index)
        if op.kind == RegKind.VCC:
            return self.vcc & ALL64
        if op.kind == RegKind.EXEC:
            return self.exec_mask & ALL64
        raise ValueError(f"Unknown operand: {op}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for s_xnor_saveexec_b64."""

    def execute(
        self,
        instr: SXnorSaveexecInstruction,
        state: ScalarMachineState,
    ) -> Tuple[int, int, int]:
        """Execute a single s_xnor_saveexec_b64 instruction on scalar machine state."""
        s0_val = state.get_operand_val(instr.src0)
        old_exec = state.exec_mask

        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(s0_val, old_exec)

        state.set_sgpr_pair(instr.dst.index, d_saved)
        state.exec_mask = new_exec
        state.scc = scc

        return new_exec, d_saved, scc


class TestScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()
        self.interp = ScalarInterpreter()

    def test_single_step_execution(self) -> None:
        self.state.set_sgpr_pair(12, 0x55555555_55555555)
        self.state.exec_mask = 0x00000000_FFFFFFFF

        instr = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 6),
            src0=Register64(RegKind.SGPR_PAIR, 12),
        )

        new_exec, d_saved, scc = self.interp.execute(instr, self.state)

        self.assertEqual(new_exec, 0xAAAAAAAA_55555555)
        self.assertEqual(d_saved, 0x00000000_FFFFFFFF)
        self.assertEqual(scc, 1)
        self.assertEqual(self.state.get_sgpr_pair(6), 0x00000000_FFFFFFFF)
        self.assertEqual(self.state.exec_mask, 0xAAAAAAAA_55555555)
        self.assertEqual(self.state.scc, 1)

    def test_multi_instruction_trace(self) -> None:
        self.state.exec_mask = 0x00000000_FFFFFFFF
        instr1 = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 2),
            src0=Immediate64(0),
        )
        self.interp.execute(instr1, self.state)
        self.assertEqual(self.state.exec_mask, 0xFFFFFFFF_00000000)

        instr2 = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 4),
            src0=Immediate64(ALL64),
        )
        self.interp.execute(instr2, self.state)
        self.assertEqual(self.state.exec_mask, 0xFFFFFFFF_00000000)
        self.assertEqual(self.state.get_sgpr_pair(4), 0xFFFFFFFF_00000000)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class TestSIMDDivergence(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_inactive_lanes_activated_by_xnor(self) -> None:
        old_exec = 0x00000000_00000000
        s0 = 0x00000000_00000000
        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(scc, 1)

        hw_exec, hw_scc, hw_bcnt, _ = run_hw_xnor_saveexec(s0, old_exec)
        self.assertEqual(hw_exec, ALL64)
        self.assertEqual(hw_scc, 1)
        self.assertEqual(hw_bcnt, 0)

    def test_active_lanes_deactivated_by_xnor(self) -> None:
        old_exec = ALL64
        s0 = 0x00000000_00000000
        new_exec, d_saved, scc = ref_s_xnor_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

        hw_exec, hw_scc, hw_bcnt, _ = run_hw_xnor_saveexec(s0, old_exec)
        self.assertEqual(hw_exec, 0)
        self.assertEqual(hw_scc, 0)
        self.assertEqual(hw_bcnt, 64)

    def test_checkerboard_divergence_patterns(self) -> None:
        mask_even = 0x55555555_55555555
        mask_odd = 0xAAAAAAAA_AAAAAAAA

        self.assertEqual(ref_s_xnor_saveexec_b64(mask_even, mask_even)[0], ALL64)
        self.assertEqual(ref_s_xnor_saveexec_b64(mask_even, mask_odd)[0], 0)

        hw_exec, hw_scc, hw_bcnt, _ = run_hw_xnor_saveexec(mask_even, mask_even)
        self.assertEqual(hw_exec, ALL64)
        self.assertEqual(hw_scc, 1)
        self.assertEqual(hw_bcnt, 32)


# ==============================================================================
# Register Aliasing and Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_src_with_exec(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.exec_mask = 0x12345678_9ABCDEF0
        instr = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 6),
            src0=Register64(RegKind.EXEC),
        )
        new_exec, d_saved, scc = interp.execute(instr, state)

        self.assertEqual(new_exec, ALL64)
        self.assertEqual(d_saved, 0x12345678_9ABCDEF0)
        self.assertEqual(scc, 1)

    def test_alias_dst_with_src(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_sgpr_pair(6, 0x55555555_55555555)
        state.exec_mask = 0x00000000_FFFFFFFF

        instr = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 6),
            src0=Register64(RegKind.SGPR_PAIR, 6),
        )
        new_exec, d_saved, scc = interp.execute(instr, state)

        self.assertEqual(new_exec, 0xAAAAAAAA_55555555)
        self.assertEqual(d_saved, 0x00000000_FFFFFFFF)
        self.assertEqual(state.get_sgpr_pair(6), 0x00000000_FFFFFFFF)

    def test_unrelated_registers_preserved(self) -> None:
        state = ScalarMachineState()
        interp = ScalarInterpreter()

        state.set_sgpr_pair(0, 0x11111111_22222222)
        state.set_sgpr_pair(10, 0x33333333_44444444)
        state.exec_mask = 0x00000000_FFFFFFFF

        instr = SXnorSaveexecInstruction(
            dst=Register64(RegKind.SGPR_PAIR, 6),
            src0=Register64(RegKind.SGPR_PAIR, 10),
        )
        interp.execute(instr, state)

        self.assertEqual(state.get_sgpr_pair(0), 0x11111111_22222222)
        self.assertEqual(state.get_sgpr_pair(10), 0x33333333_44444444)


if __name__ == "__main__":
    unittest.main()
