from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_sad_hi_u16.

AMD RDNA / GFX10 Vector ALU Instruction Set Architecture:
- v_sad_hi_u16 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Computes the 2-channel sum of unsigned 16-bit absolute differences between
    packed halfword lanes of src0 and src1, shifts the resulting 17-bit intermediate sum
    left by 16 bits (MSB halfword alignment), and accumulates into the 32-bit unsigned
    accumulator src2:
        diff0 = |src0[15:0]  - src1[15:0]|
        diff1 = |src0[31:16] - src1[31:16]|
        sum_diff = diff0 + diff1
        sum_shifted = sum_diff << 16
        dst = (sum_shifted + src2) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

- v_msad_hi_u16 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Masked sum of absolute differences across 2 halfword lanes with src1 serving as template/mask:
    If src1 halfword is 0, that lane difference is omitted:
        diff0 = |src0[15:0] - src1[15:0]| if src1[15:0] != 0 else 0
        diff1 = |src0[31:16] - src1[31:16]| if src1[31:16] != 0 else 0
        sum_diff = diff0 + diff1
        sum_shifted = sum_diff << 16
        dst = (sum_shifted + src2) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

Mathematical Properties and Bounds:
  - Operands src0, src1 are treated as two packed 16-bit unsigned integers in [0, 65535].
  - Each halfword absolute difference diff0, diff1 is in [0, 65535] (0x0000 to 0xFFFF).
  - Total sum of absolute differences sum_diff is in [0, 2 * 65535] = [0, 131070] (0x00000 to 0x1FFFE).
  - sum_diff strictly fits within 17 bits (< 2^17 = 131072).
  - Shifted sum (sum_diff << 16) is in [0, 131070 * 65536] = [0, 0x1_FFFE_0000].
  - Bits [15:0] of (sum_diff << 16) are strictly zero.
  - Adding unsigned 32-bit src2 produces a full sum in [0, 0x1_FFFE_0000 + 0xFFFF_FFFF] = [0, 0x2_FFFD_FFFF].
  - Destination wraps modulo 2^32: dst = ((sum_diff << 16) + src2) & 0xFFFFFFFF.
  - Carry-out into the upper carry word is: ((sum_diff << 16) + src2) >> 32 in {0, 1, 2}.
  - Symmetry / Commutativity:
    v_sad_hi_u16(src0, src1, src2) == v_sad_hi_u16(src1, src0, src2).

Standard library Python only: unittest, dataclasses, enum, math, struct.
Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Sequence, Tuple, Union

# Architectural bitmasks and saturation limits
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

CLAMP_MAX_U32: int = MASK32


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def unpack_halfwords_u32(val: int) -> Tuple[int, int]:
    """Unpack a 32-bit unsigned word into two 16-bit unsigned halfword lanes (h0, h1)."""
    val &= MASK32
    return (
        val & MASK16,
        (val >> 16) & MASK16,
    )


def pack_halfwords_u32(h0: int, h1: int) -> int:
    """Pack two 16-bit unsigned integers into a 32-bit unsigned dword."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def ref_halfword_abs_diff(h0: int, h1: int) -> int:
    """Calculate strictly unsigned 16-bit absolute difference |h0.u16 - h1.u16|."""
    u0 = h0 & MASK16
    u1 = h1 & MASK16
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u16_sum(s0: int, s1: int) -> int:
    """Compute 2-channel 16-bit sum of unsigned absolute differences.

    Result is bounded in [0, 2 * 65535] = [0, 131070] (0x1FFFE).
    """
    h0_0, h0_1 = unpack_halfwords_u32(s0)
    h1_0, h1_1 = unpack_halfwords_u32(s1)
    diff0 = ref_halfword_abs_diff(h0_0, h1_0)
    diff1 = ref_halfword_abs_diff(h0_1, h1_1)
    return diff0 + diff1


def ref_msad_u16_sum(s0: int, s1: int) -> int:
    """Compute masked 2-channel 16-bit sum of unsigned absolute differences.

    Halfword lanes where s1.halfword == 0 are masked out (difference is omitted / 0).
    """
    h0_0, h0_1 = unpack_halfwords_u32(s0)
    h1_0, h1_1 = unpack_halfwords_u32(s1)
    total = 0
    if h1_0 != 0:
        total += ref_halfword_abs_diff(h0_0, h1_0)
    if h1_1 != 0:
        total += ref_halfword_abs_diff(h0_1, h1_1)
    return total


def ref_sad_hi_u16(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_hi_u16.

    dst = ((sum_diff << 16) + s2) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_sad_u16_sum(s0, s1)
    acc = s2 & MASK32
    shifted_sad = sad << 16
    full_sum = acc + shifted_sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_msad_hi_u16(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_msad_hi_u16.

    dst = ((masked_sum_diff << 16) + s2) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_msad_u16_sum(s0, s1)
    acc = s2 & MASK32
    shifted_sad = sad << 16
    full_sum = acc + shifted_sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_sad_hi_carry_u16(s0: int, s1: int, s2: int) -> int:
    """Compute the high 32-bit carry/overflow word generated by v_sad_hi_u16."""
    sad = ref_sad_u16_sum(s0, s1)
    acc = s2 & MASK32
    shifted_sad = sad << 16
    full_sum = acc + shifted_sad
    return (full_sum >> 32) & MASK32


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a register operand or inline literal constant."""
    reg_type: RegType
    index: int = 0
    literal_value: int | None = None
    count: int = 1

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR range [{self.index}:{self.index + self.count - 1}] exceeds 255")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR range [{self.index}:{self.index + self.count - 1}] exceeds 105")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=u32(value), count=1)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.count == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.count - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.count == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.count - 1}]"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VSadHiU16Instruction:
    """Symbolic AST node for v_sad_hi_u16 dst, src0, src1, src2."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be single dword (count=1), got {self.dst.count}")
        for idx, src in enumerate([self.src0, self.src1, self.src2]):
            if src.count != 1:
                raise ValueError(f"src{idx} must be single dword (count=1), got {src.count}")

    def disassemble(self) -> str:
        parts = [
            "v_sad_hi_u16",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VMsadHiU16Instruction:
    """Symbolic AST node for v_msad_hi_u16 dst, src0, src1, src2."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be single dword (count=1), got {self.dst.count}")
        for idx, src in enumerate([self.src0, self.src1, self.src2]):
            if src.count != 1:
                raise ValueError(f"src{idx} must be single dword (count=1), got {src.count}")

    def disassemble(self) -> str:
        parts = [
            "v_msad_hi_u16",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


SadHiU16Instruction = Union[VSadHiU16Instruction, VMsadHiU16Instruction]


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for a single scalar/lane execution context."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read(self, op: Operand) -> int:
        """Read operand value as unsigned 32-bit dword."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write(self, op: Operand, value: int) -> None:
        """Write value to register operand (VGPR or SGPR)."""
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = u32(value)
        elif op.reg_type == RegType.SGPR:
            self.sgprs[op.index] = u32(value)
        else:
            raise TypeError("Cannot write to literal operand")


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for VALU SAD HI u16 instructions on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def execute_sad_hi_u16(self, instr: VSadHiU16Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        result = ref_sad_hi_u16(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def execute_msad_hi_u16(self, instr: VMsadHiU16Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        result = ref_msad_hi_u16(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def step(self, instr: SadHiU16Instruction) -> int:
        if isinstance(instr, VSadHiU16Instruction):
            return self.execute_sad_hi_u16(instr)
        if isinstance(instr, VMsadHiU16Instruction):
            return self.execute_msad_hi_u16(instr)
        raise TypeError(f"Unsupported instruction AST: {type(instr)}")

    def run_program(self, instructions: Sequence[SadHiU16Instruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# Unit Tests: Reference Mathematical Oracles
# ==============================================================================

class TestValuSadHiU16Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracles of v_sad_hi_u16 and v_msad_hi_u16."""

    def test_halfword_abs_diff_bounds_and_symmetry(self) -> None:
        pairs = [
            (0, 0),
            (0, 65535),
            (65535, 0),
            (32768, 32768),
            (100, 50000),
            (50000, 100),
            (32767, 32768),
        ]
        for h0, h1 in pairs:
            diff = ref_halfword_abs_diff(h0, h1)
            self.assertGreaterEqual(diff, 0)
            self.assertLessEqual(diff, 65535)
            self.assertEqual(diff, ref_halfword_abs_diff(h1, h0))

    def test_unsigned_vs_signed_halfword_difference(self) -> None:
        # 0x0000 vs 0xFFFF:
        # Unsigned 16-bit: |0 - 65535| = 65535
        # Signed 16-bit: 0xFFFF is -1, |0 - (-1)| = 1
        self.assertEqual(ref_halfword_abs_diff(0x0000, 0xFFFF), 65535)
        self.assertEqual(ref_halfword_abs_diff(0xFFFF, 0x0000), 65535)

    def test_sad_u16_sum_identity_and_maximum(self) -> None:
        # Identical halfwords produce sum 0
        self.assertEqual(ref_sad_u16_sum(0x1234_5678, 0x1234_5678), 0)
        self.assertEqual(ref_sad_u16_sum(0x0000_0000, 0x0000_0000), 0)
        self.assertEqual(ref_sad_u16_sum(0xFFFF_FFFF, 0xFFFF_FFFF), 0)

        # Max difference: each lane diff is 65535 -> sum = 2 * 65535 = 131070 (0x1FFFE)
        self.assertEqual(ref_sad_u16_sum(0x0000_0000, 0xFFFF_FFFF), 131070)
        self.assertEqual(ref_sad_u16_sum(0xFFFF_FFFF, 0x0000_0000), 131070)

    def test_sad_hi_u16_upper_word_shift_and_accumulation(self) -> None:
        # Low lane diff: |10 - 20| = 10. High lane diff: |30 - 50| = 20.
        # sum = 30. Shifted = 30 << 16 = 0x001E_0000 = 1966080.
        s0 = pack_halfwords_u32(10, 30)
        s1 = pack_halfwords_u32(20, 50)
        self.assertEqual(ref_sad_u16_sum(s0, s1), 30)
        self.assertEqual(ref_sad_hi_u16(s0, s1, 0), 0x001E_0000)
        self.assertEqual(ref_sad_hi_u16(s0, s1, 0x1234), 0x001E_1234)

    def test_sad_hi_u16_wraparound_modulo_2_32(self) -> None:
        # sum_shifted overflows 32-bit dword when added to src2
        s0 = pack_halfwords_u32(1, 0)
        s1 = pack_halfwords_u32(0, 0)  # sad = 1, sum_shifted = 0x0001_0000
        # src2 = 0xFFFF_FFFF -> 0xFFFF_FFFF + 0x0001_0000 = 0x1_0000_FFFF
        self.assertEqual(ref_sad_hi_u16(s0, s1, MASK32, clamp=False), 0x0000_FFFF)

    def test_sad_hi_u16_clamping_saturation(self) -> None:
        s0 = pack_halfwords_u32(1, 0)
        s1 = pack_halfwords_u32(0, 0)  # sum_shifted = 0x0001_0000
        self.assertEqual(ref_sad_hi_u16(s0, s1, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_sad_hi_u16(s0, s1, MASK32 - 0x0000_FFFF, clamp=True), MASK32)
        self.assertEqual(ref_sad_hi_u16(s0, s1, 50, clamp=True), 0x0001_0032)

    def test_sad_hi_carry_overflow_bounds(self) -> None:
        # Max sad = 131070 (0x1FFFE) -> sum_shifted = 0x1FFFE << 16 = 0x1_FFFE_0000
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000
        # If src2 = 0: full_sum = 0x1_FFFE_0000 -> carry = 1
        self.assertEqual(ref_sad_hi_carry_u16(s0, s1, 0), 1)

        # If src2 = 0xFFFFFFFF: full_sum = 0x1_FFFE_0000 + 0xFFFF_FFFF = 0x2_FFFD_FFFF -> carry = 2
        self.assertEqual(ref_sad_hi_carry_u16(s0, s1, MASK32), 2)

        # If sad is small and src2 is small: carry = 0
        self.assertEqual(ref_sad_hi_carry_u16(0x0001_0001, 0, 0x1000), 0)

    def test_msad_hi_u16_zero_masking(self) -> None:
        # s1 has low halfword zero, high halfword non-zero
        s0 = pack_halfwords_u32(500, 1000)
        s1 = pack_halfwords_u32(0, 800)
        # Low lane: s1=0 -> skipped (diff 0)
        # High lane: |1000 - 800| = 200
        # sad = 200 -> shifted = 200 << 16 = 0x00C8_0000
        self.assertEqual(ref_msad_u16_sum(s0, s1), 200)
        self.assertEqual(ref_msad_hi_u16(s0, s1, 0x55), 0x00C8_0055)

        # Contrast with unmasked sad: low diff = 500, high diff = 200 -> sum = 700
        self.assertEqual(ref_sad_u16_sum(s0, s1), 700)

    def test_msad_hi_u16_all_zeros_mask(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000  # all template halfwords zero
        self.assertEqual(ref_msad_u16_sum(s0, s1), 0)
        self.assertEqual(ref_msad_hi_u16(s0, s1, 0x1234_5678), 0x1234_5678)

    def test_msad_hi_u16_clamp_behavior(self) -> None:
        s0 = pack_halfwords_u32(65535, 65535)
        s1 = pack_halfwords_u32(1, 1)  # diff = 2 * 65534 = 131068
        self.assertEqual(ref_msad_hi_u16(s0, s1, MASK32, clamp=True), MASK32)
        self.assertNotEqual(ref_msad_hi_u16(s0, s1, MASK32, clamp=False), MASK32)

    def test_dual_precision_and_complementary_packing_with_v_sad_u16(self) -> None:
        s0 = pack_halfwords_u32(100, 200)
        s1 = pack_halfwords_u32(50, 120)
        sad = ref_sad_u16_sum(s0, s1)  # 50 + 80 = 130
        res_hi = ref_sad_hi_u16(s0, s1, 0)
        self.assertEqual(res_hi, 130 << 16)


class TestValuSadHiU16SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v0 = Operand.vgpr(0)
        self.assertEqual(v0.disassemble(), "v0")
        v_pair = Operand.vgpr(4, count=2)
        self.assertEqual(v_pair.disassemble(), "v[4:5]")

        s2 = Operand.sgpr(2)
        self.assertEqual(s2.disassemble(), "s2")
        s_pair = Operand.sgpr(10, count=2)
        self.assertEqual(s_pair.disassemble(), "s[10:11]")

        lit = Operand.literal(0x1234_5678)
        self.assertEqual(lit.disassemble(), "0x12345678")

    def test_operand_validation_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(255, count=2)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(105, count=2)

    def test_v_sad_hi_u16_ast_construction_and_disasm(self) -> None:
        instr = VSadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(100),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_sad_hi_u16 v0, v1, s2, 0x00000064")

    def test_v_sad_hi_u16_clamp_modifier_disasm(self) -> None:
        instr = VSadHiU16Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(6),
            src1=Operand.vgpr(7),
            src2=Operand.vgpr(8),
            clamp=True,
        )
        self.assertEqual(instr.disassemble(), "v_sad_hi_u16 v5, v6, v7, v8 clamp")

    def test_v_msad_hi_u16_ast_construction_and_disasm(self) -> None:
        instr = VMsadHiU16Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(11),
            src1=Operand.sgpr(12),
            src2=Operand.vgpr(13),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_msad_hi_u16 v10, v11, s12, v13")

        instr_clamped = VMsadHiU16Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(11),
            src1=Operand.sgpr(12),
            src2=Operand.vgpr(13),
            clamp=True,
        )
        self.assertEqual(instr_clamped.disassemble(), "v_msad_hi_u16 v10, v11, s12, v13 clamp")

    def test_instruction_validation_rejects_non_vgpr_dst(self) -> None:
        with self.assertRaises(TypeError):
            VSadHiU16Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadHiU16Instruction(
                dst=Operand.literal(42),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VMsadHiU16Instruction(
                dst=Operand.sgpr(5),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_instruction_validation_rejects_multi_dword_operands(self) -> None:
        with self.assertRaises(ValueError):
            VSadHiU16Instruction(
                dst=Operand.vgpr(0, count=2),
                src0=Operand.vgpr(2),
                src1=Operand.vgpr(3),
                src2=Operand.vgpr(4),
            )
        with self.assertRaises(ValueError):
            VSadHiU16Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(3),
                src2=Operand.vgpr(4),
            )


class TestValuSadHiU16ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.rf)

    def test_execute_sad_hi_u16_basic(self) -> None:
        self.rf.vgprs[1] = pack_halfwords_u32(100, 200)
        self.rf.vgprs[2] = pack_halfwords_u32(120, 250)
        self.rf.vgprs[3] = 0x0001_0000
        # diffs: 20 + 50 = 70. Shifted = 70 << 16 = 0x0046_0000. Total = 0x0047_0000
        instr = VSadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 0x0047_0000)
        self.assertEqual(self.rf.vgprs[0], 0x0047_0000)

    def test_execute_sad_hi_u16_with_literal_operands(self) -> None:
        self.rf.vgprs[1] = pack_halfwords_u32(10, 10)
        instr = VSadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.literal(pack_halfwords_u32(5, 5)),
            src2=Operand.literal(0x1000),
        )
        # diffs: 5 + 5 = 10. Shifted = 10 << 16 = 0x000A_0000. Total = 0x000A_1000
        res = self.interp.step(instr)
        self.assertEqual(res, 0x000A_1000)
        self.assertEqual(self.rf.vgprs[0], 0x000A_1000)

    def test_execute_sad_hi_u16_clamp_saturation(self) -> None:
        self.rf.vgprs[1] = 0xFFFF_FFFF
        self.rf.vgprs[2] = 0x0000_0000
        self.rf.vgprs[3] = MASK32
        instr = VSadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, MASK32)
        self.assertEqual(self.rf.vgprs[0], MASK32)

    def test_execute_msad_hi_u16_masked_execution(self) -> None:
        self.rf.vgprs[1] = pack_halfwords_u32(500, 1000)
        self.rf.sgprs[5] = pack_halfwords_u32(0, 800)  # low halfword masked out
        self.rf.vgprs[2] = 0x0010_0000
        # low skipped; high diff: 200. Shifted: 200 << 16 = 0x00C8_0000. Total: 0x00D8_0000
        instr = VMsadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(5),
            src2=Operand.vgpr(2),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 0x00D8_0000)
        self.assertEqual(self.rf.vgprs[0], 0x00D8_0000)

    def test_execute_msad_hi_u16_clamp(self) -> None:
        self.rf.vgprs[1] = pack_halfwords_u32(1000, 1000)
        self.rf.vgprs[2] = pack_halfwords_u32(500, 500)
        self.rf.vgprs[3] = MASK32
        instr = VMsadHiU16Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, MASK32)
        self.assertEqual(self.rf.vgprs[0], MASK32)

    def test_run_program_sequential_chaining(self) -> None:
        self.rf.vgprs[1] = pack_halfwords_u32(1, 1)
        self.rf.vgprs[2] = pack_halfwords_u32(2, 2)  # diff = 2 -> 2 << 16 = 0x0002_0000
        self.rf.vgprs[3] = pack_halfwords_u32(10, 10)
        self.rf.vgprs[4] = pack_halfwords_u32(12, 12)  # diff = 4 -> 4 << 16 = 0x0004_0000
        prog = [
            VSadHiU16Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.literal(0)),
            VSadHiU16Instruction(Operand.vgpr(0), Operand.vgpr(3), Operand.vgpr(4), Operand.vgpr(0)),
        ]
        self.interp.run_program(prog)
        self.assertEqual(self.rf.vgprs[0], 0x0006_0000)

    def test_unsupported_instruction_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            self.interp.step("invalid")  # type: ignore


if __name__ == "__main__":
    unittest.main()
