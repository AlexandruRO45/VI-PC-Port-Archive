from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_saveexec_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_saveexec_b64 instruction belongs to SOP1 format.
  Instruction:
    s_andn2_saveexec_b64 sdst, ssrc0

  Operands:
    sdst  = 64-bit destination SGPR pair (or EXEC) to receive the previous EXEC mask
    ssrc0 = 64-bit source operand: SGPR pair s[2k:2k+1], literal 64-bit constant, VCC, or EXEC

  Semantics:
    sdst = EXEC
    EXEC = ssrc0 & ~EXEC
    SCC = 1 if EXEC != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402
import saveexec_run  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_saveexec_b64(ssrc0: int, old_exec: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_saveexec_b64 instruction.

    Args:
        ssrc0: 64-bit source operand (uint64)
        old_exec: 64-bit current EXEC mask before instruction execution (uint64)

    Returns:
        tuple (new_exec, sdst, scc):
          - new_exec: updated 64-bit EXEC mask = ssrc0 & ~old_exec
          - sdst: saved previous EXEC mask = old_exec
          - scc: scalar condition code = 1 if new_exec != 0 else 0
    """
    s0 = u64(ssrc0)
    old_e = u64(old_exec)
    sdst = old_e
    new_exec = u64(s0 & (~old_e & MASK64))
    scc = 1 if new_exec != 0 else 0
    return new_exec, sdst, scc


class TestSaluAndn2SaveexecB64Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_saveexec_b64."""

    def test_oracle_all_ones_exec(self):
        # If old_exec = all 1s, ~old_exec = 0, so new_exec = ssrc0 & 0 = 0
        s0 = 0x123456789ABCDEF0
        old_exec = ALL_LANES_ACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 0)

    def test_oracle_all_zeros_exec(self):
        # If old_exec = 0, ~old_exec = all 1s, so new_exec = ssrc0 & all 1s = ssrc0
        s0 = 0x123456789ABCDEF0
        old_exec = ALL_LANES_INACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0_zero_exec(self):
        s0 = 0x0
        old_exec = ALL_LANES_INACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE)
        self.assertEqual(scc, 0)

    def test_oracle_self_exec(self):
        # S0 == old_exec -> new_exec = exec & ~exec = 0
        exec_val = 0xAAAAAAAAAAAAAAAA
        new_exec, sdst, scc = ref_andn2_saveexec_b64(exec_val, exec_val)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, exec_val)
        self.assertEqual(scc, 0)


class Opcode(Enum):
    S_ANDN2_SAVEEXEC_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_andn2_saveexec_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_ANDN2_SAVEEXEC_B64:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC):
            raise TypeError(f"Destination must be SGPR pair or EXEC, got {self.dst.kind}")
        if self.dst.count != 2:
            raise ValueError(f"Destination must be 64-bit (2 registers), got {self.dst.count}")
        if self.src0.count != 2:
            raise ValueError(f"Source must be 64-bit operand, got {self.src0.count}")


class TestSaluAndn2SaveexecB64AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(6, 2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ANDN2_SAVEEXEC_B64)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_unaligned_sgpr_pair(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(5, 2)

    def test_ast_rejects_single_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 1)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.literal64(0x123),
            src0=ASTOperand.sgpr(6, 2),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC, and SCC)."""

    def __init__(self, exec_mask: int = ALL_LANES_ACTIVE):
        self.sgpr = [0] * 106
        self.exec_mask = u64(exec_mask)
        self.scc = 0

    def read_operand_u64(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            lo = self.sgpr[op.index]
            hi = self.sgpr[op.index + 1]
            return (hi << 32) | lo
        elif op.kind == OperandKind.EXEC:
            return self.exec_mask
        elif op.kind in (OperandKind.LITERAL64, OperandKind.INLINE_CONST):
            return op.value
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_sgpr_u64(self, index: int, val: int) -> None:
        val = u64(val)
        self.sgpr[index] = val & MASK32
        self.sgpr[index + 1] = (val >> 32) & MASK32

    def execute(self, inst: ASTInstruction) -> tuple[int, int, int]:
        inst.validate()
        s0 = self.read_operand_u64(inst.src0)
        old_exec = self.exec_mask
        new_exec, sdst_val, scc = ref_andn2_saveexec_b64(s0, old_exec)

        if inst.dst.kind == OperandKind.SGPR:
            self.write_sgpr_u64(inst.dst.index, sdst_val)
        elif inst.dst.kind == OperandKind.EXEC:
            self.exec_mask = sdst_val

        self.exec_mask = new_exec
        self.scc = scc
        return new_exec, sdst_val, scc


class TestSaluAndn2SaveexecB64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = ScalarRegisterState(exec_mask=0x0000FFFFFFFF0000)
        state.write_sgpr_u64(2, 0x00000000FFFFFFFF)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(0, 2),
            src0=ASTOperand.sgpr(2, 2),
        )
        new_exec, sdst, scc = state.execute(inst)
        # old_exec = 0x0000FFFFFFFF0000 -> ~old_exec has bits 0..15 and 48..63 set
        # s0 = 0x00000000FFFFFFFF (bits 0..31 set)
        # new_exec = s0 & ~old_exec = bits 0..15 set = 0x000000000000FFFF
        self.assertEqual(new_exec, 0x000000000000FFFF)
        self.assertEqual(sdst, 0x0000FFFFFFFF0000)
        self.assertEqual(state.exec_mask, 0x000000000000FFFF)
        self.assertEqual(state.sgpr[0], 0xFFFF0000)
        self.assertEqual(state.sgpr[1], 0x0000FFFF)
        self.assertEqual(state.scc, 1)


class TestSaluAndn2SaveexecB64SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 64-bit EXEC masking."""

    def test_lane_suppression_and_reactivation(self):
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        # Initial active lanes: 0..31 (lower half)
        # Initial inactive lanes: 32..63 (upper half)
        old_exec = lanes(0, 32)
        # Source requests all lanes 0..63
        s0 = ALL_LANES_ACTIVE

        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        # All previously active lanes (0..31) are turned off (~old_exec = 0 in lower half)
        # All previously inactive lanes (32..63) are turned on (~old_exec = 1 in upper half)
        self.assertEqual(new_exec, lanes(32, 64))
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_divergent_mask_checkerboard(self):
        even_lanes = 0x5555555555555555
        odd_lanes = 0xAAAAAAAAAAAAAAAA
        # old_exec = even lanes active; s0 = ALL_LANES_ACTIVE
        # new_exec should be odd lanes active
        new_exec, sdst, scc = ref_andn2_saveexec_b64(ALL_LANES_ACTIVE, even_lanes)
        self.assertEqual(new_exec, odd_lanes)
        self.assertEqual(sdst, even_lanes)
        self.assertEqual(scc, 1)


class TestSaluAndn2SaveexecB64RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_src0_aliasing(self):
        # sdst and ssrc0 overlap in s[0:1]
        state = ScalarRegisterState(exec_mask=0x0000FFFFFFFF0000)
        initial_s0 = 0x00000000FFFFFFFF
        state.write_sgpr_u64(0, initial_s0)

        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(0, 2),
            src0=ASTOperand.sgpr(0, 2),
        )
        new_exec, sdst, scc = state.execute(inst)

        # Source is read BEFORE destination is overwritten with saved EXEC
        self.assertEqual(sdst, 0x0000FFFFFFFF0000)
        self.assertEqual(state.sgpr[0], 0xFFFF0000)
        self.assertEqual(state.sgpr[1], 0x0000FFFF)
        self.assertEqual(new_exec, 0x000000000000FFFF)
        self.assertEqual(state.exec_mask, 0x000000000000FFFF)
        self.assertEqual(scc, 1)

    def test_dst_is_exec(self):
        # When sdst is EXEC: sdst = EXEC, then EXEC = ssrc0 & ~EXEC
        state = ScalarRegisterState(exec_mask=ALL_LANES_ACTIVE)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.exec_mask(),
            src0=ASTOperand.literal64(0x123456789ABCDEF0),
        )
        new_exec, sdst, scc = state.execute(inst)
        # old_exec was ALL_LANES_ACTIVE, so new_exec is 0
        self.assertEqual(new_exec, 0)
        self.assertEqual(state.exec_mask, 0)
        self.assertEqual(scc, 0)


class TestSaluAndn2SaveexecB64ConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and SCC generation."""

    def test_scc_set_when_new_exec_nonzero(self):
        new_exec, _, scc = ref_andn2_saveexec_b64(0x1, ALL_LANES_INACTIVE)
        self.assertEqual(new_exec, 1)
        self.assertEqual(scc, 1)

    def test_scc_cleared_when_new_exec_zero(self):
        new_exec, _, scc = ref_andn2_saveexec_b64(ALL_LANES_ACTIVE, ALL_LANES_ACTIVE)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

    def test_scc_cleared_when_ssrc0_is_zero(self):
        new_exec, _, scc = ref_andn2_saveexec_b64(0, 0x123456789ABCDEF0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)


class TestSaluAndn2SaveexecB64BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 64-bit IEEE-754 bit representations and boundary bitmasks."""

    def test_ieee754_double_precision_bit_patterns(self):
        cases = [
            0x7FF0000000000000,  # +inf
            0xFFF0000000000000,  # -inf
            0x7FF8000000000000,  # Quiet NaN
            0x7FF0000000000001,  # Signaling NaN
            0x0000000000000000,  # +0.0
            0x8000000000000000,  # -0.0
            0x0000000000000001,  # Minimum subnormal
            0x000FFFFFFFFFFFFF,  # Maximum subnormal
        ]
        for pattern in cases:
            for old_exec in (0x0, ALL_LANES_ACTIVE, 0xAAAAAAAAAAAAAAAA):
                with self.subTest(pattern=hex(pattern), exec=hex(old_exec)):
                    new_e, d, scc = ref_andn2_saveexec_b64(pattern, old_exec)
                    self.assertEqual(d, old_exec)
                    self.assertEqual(new_e, pattern & (~old_exec & MASK64))
                    self.assertEqual(scc, int(new_e != 0))

    def test_boundary_alternating_bitmasks(self):
        checkerboards = [
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x3333333333333333,
            0xCCCCCCCCCCCCCCCC,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
            0x00FF00FF00FF00FF,
            0xFF00FF00FF00FF00,
        ]
        for cb1 in checkerboards:
            for cb2 in checkerboards:
                new_e, d, scc = ref_andn2_saveexec_b64(cb1, cb2)
                self.assertEqual(new_e, (cb1 & (~cb2 & MASK64)) & MASK64)


def parse_s_andn2_saveexec_b64_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for s_andn2_saveexec_b64."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    mnemonic = parts[0].lower()
    if mnemonic != "s_andn2_saveexec_b64":
        raise ValueError(f"Expected mnemonic s_andn2_saveexec_b64, got {mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")
    operands_str = [op.strip() for op in parts[1].split(",")]
    if len(operands_str) != 2:
        raise ValueError(f"Expected 2 operands (dst, src0), got {len(operands_str)}")

    dst_str, src_str = operands_str[0], operands_str[1]
    if dst_str.lower() == "exec":
        dst = ASTOperand.exec_mask()
    elif dst_str.lower().startswith("s[") and dst_str.endswith("]"):
        idx_pair = dst_str[2:-1].split(":")
        if len(idx_pair) != 2:
            raise ValueError(f"Invalid SGPR pair syntax: {dst_str}")
        start, end = int(idx_pair[0]), int(idx_pair[1])
        if end != start + 1 or start % 2 != 0:
            raise ValueError(f"SGPR pair must be even-aligned 2-register range: {dst_str}")
        dst = ASTOperand.sgpr(start, 2)
    else:
        raise ValueError(f"Invalid destination operand: {dst_str}")

    if src_str.lower() == "exec":
        src = ASTOperand.exec_mask()
    elif src_str.lower() == "vcc":
        src = ASTOperand.vcc()
    elif src_str.lower().startswith("s[") and src_str.endswith("]"):
        idx_pair = src_str[2:-1].split(":")
        if len(idx_pair) != 2:
            raise ValueError(f"Invalid SGPR pair syntax: {src_str}")
        start, end = int(idx_pair[0]), int(idx_pair[1])
        if end != start + 1 or start % 2 != 0:
            raise ValueError(f"SGPR pair must be even-aligned 2-register range: {src_str}")
        src = ASTOperand.sgpr(start, 2)
    elif src_str.startswith("0x") or src_str.startswith("-") or src_str.isdigit():
        val = int(src_str, 0)
        src = ASTOperand.literal64(val)
    else:
        raise ValueError(f"Invalid source operand: {src_str}")

    inst = ASTInstruction(Opcode.S_ANDN2_SAVEEXEC_B64, dst, src)
    inst.validate()
    return inst


# Mutant functions for mutation testing
def mutant_and(s0: int, old_exec: int) -> tuple[int, int, int]:
    """Mutant 1: uses AND instead of ANDN2 (omits ~ inversion)."""
    new_e = (s0 & old_exec) & MASK64
    return new_e, old_exec, 1 if new_e != 0 else 0


def mutant_orn2(s0: int, old_exec: int) -> tuple[int, int, int]:
    """Mutant 2: uses OR instead of AND."""
    new_e = (s0 | (~old_exec & MASK64)) & MASK64
    return new_e, old_exec, 1 if new_e != 0 else 0


def mutant_save_new_exec(s0: int, old_exec: int) -> tuple[int, int, int]:
    """Mutant 3: saves new EXEC to destination instead of old EXEC."""
    new_e = (s0 & (~old_exec & MASK64)) & MASK64
    return new_e, new_e, 1 if new_e != 0 else 0


def mutant_invert_scc(s0: int, old_exec: int) -> tuple[int, int, int]:
    """Mutant 4: inverts SCC logic (scc = 1 when EXEC == 0)."""
    new_e = (s0 & (~old_exec & MASK64)) & MASK64
    return new_e, old_exec, 1 if new_e == 0 else 0


class TestSaluAndn2SaveexecB64NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        with self.assertRaises(ValueError):
            parse_s_andn2_saveexec_b64_asm("")

        with self.assertRaises(ValueError):
            parse_s_andn2_saveexec_b64_asm("s_orn2_saveexec_b64 s[0:1], s[2:3]")

        with self.assertRaises(ValueError):
            parse_s_andn2_saveexec_b64_asm("s_andn2_saveexec_b64 s[0:1]")

        with self.assertRaises(ValueError):
            parse_s_andn2_saveexec_b64_asm("s_andn2_saveexec_b64 s[1:2], s[2:3]")

        with self.assertRaises(ValueError):
            parse_s_andn2_saveexec_b64_asm("s_andn2_saveexec_b64 s[0:2], s[2:3]")

    def test_mutation_detection_and(self):
        s0, e = 0x0F, 0x00
        correct = ref_andn2_saveexec_b64(s0, e)
        mutated = mutant_and(s0, e)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_orn2(self):
        s0, e = 0x00, 0x00
        correct = ref_andn2_saveexec_b64(s0, e)
        mutated = mutant_orn2(s0, e)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_save_new_exec(self):
        s0, e = 0xFF, 0x00
        correct = ref_andn2_saveexec_b64(s0, e)
        mutated = mutant_save_new_exec(s0, e)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_invert_scc(self):
        s0, e = 0x00, ALL_LANES_ACTIVE
        correct = ref_andn2_saveexec_b64(s0, e)
        mutated = mutant_invert_scc(s0, e)
        self.assertNotEqual(correct, mutated)

    def test_andn2_spirv_emission(self):
        """Verify that support.translate_lines emits expected SPIR-V logic for s_andn2_saveexec_b64."""
        lines = [
            "s_mov_b32 s0, 0x0000ffff",
            "s_mov_b32 s1, 0x00000000",
            "s_andn2_saveexec_b64 s[4:5], s[0:1]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_test_andn2_emit", lines, fssm=saveexec_run.fssm(threads=64))
        self.assertIn("OpNot", spv)
        self.assertIn("OpBitwiseAnd", spv)

    def test_andn2_every_lane_wave64_simulation(self):
        """Run through translator and spirv_mini_interp across all 4 sources and 14 test masks."""
        saveexec_run.check(self, "andn2")


if __name__ == "__main__":
    unittest.main()
