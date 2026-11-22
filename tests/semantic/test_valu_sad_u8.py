from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_sad_u8, v_sad_hi_u8, and v_msad_u8.

AMD RDNA / GFX10 Vector ALU Instruction Set Architecture:
- v_sad_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Computes the sum of absolute differences between four pairs of packed 8-bit unsigned
    integers in src0 and src1, and adds the result to the 32-bit accumulator src2:
        diff0 = |src0[7:0] - src1[7:0]|
        diff1 = |src0[15:8] - src1[15:8]|
        diff2 = |src0[23:16] - src1[23:16]|
        diff3 = |src0[31:24] - src1[31:24]|
        sad = diff0 + diff1 + diff2 + diff3
        dst = (src2 + sad) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

- v_sad_hi_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Computes the 4-byte SAD, shifts the intermediate sum left by 16 bits (MSB alignment),
    and accumulates into src2:
        sad = sum_{i=0..3} |src0.byte[i] - src1.byte[i]|
        dst = (src2 + (sad << 16)) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

- v_msad_u8 dst.u32, src0.u32, src1.u32, src2.u32 [clamp]:
    Masked sum of absolute differences across 4 byte lanes with src1 serving as template/mask:
    If src1.byte[i] == 0, that byte lane is masked out (difference is omitted / 0):
        sad = sum_{i=0..3, src1.byte[i] != 0} |src0.byte[i] - src1.byte[i]|
        dst = (src2 + sad) mod 2^32 (or clamped to 0xFFFFFFFF if clamp modifier is present)

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Sequence, Tuple, Union

# Architectural bitmasks and saturation limits
MASK8 = 0xFF
MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

CLAMP_MAX_U32 = MASK32


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


def extract_byte(word32: int, byte_idx: int) -> int:
    """Extract unsigned 8-bit byte at byte_idx (0..3) from a 32-bit word."""
    return (word32 >> (8 * (byte_idx & 3))) & MASK8


def pack_bytes(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit bytes into a 32-bit word [b3, b2, b1, b0]."""
    return ((b0 & MASK8) |
            ((b1 & MASK8) << 8) |
            ((b2 & MASK8) << 16) |
            ((b3 & MASK8) << 24))


def ref_byte_abs_diff(b0: int, b1: int) -> int:
    """Calculate strictly unsigned 8-bit absolute difference |b0.u8 - b1.u8|."""
    u0 = b0 & MASK8
    u1 = b1 & MASK8
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_sad_u8_sum(s0: int, s1: int) -> int:
    """Compute 4-element byte-wise sum of absolute differences between two 32-bit words.

    Result is bounded in [0, 4 * 255] = [0, 1020].
    """
    total = 0
    for i in range(4):
        total += ref_byte_abs_diff(extract_byte(s0, i), extract_byte(s1, i))
    return total


def ref_msad_u8_sum(s0: int, s1: int) -> int:
    """Compute masked 4-element byte-wise sum of absolute differences.

    Bytes where s1.byte[i] == 0 are masked out (difference is omitted).
    """
    total = 0
    for i in range(4):
        b1 = extract_byte(s1, i)
        if b1 == 0:
            continue
        total += ref_byte_abs_diff(extract_byte(s0, i), b1)
    return total


def ref_sad_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_u8.

    dst = (s2 + sum_{i=0..3} |s0.byte[i] - s1.byte[i]|) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    full_sum = acc + sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_sad_hi_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_sad_hi_u8.

    dst = (s2 + (sum_{i=0..3} |s0.byte[i] - s1.byte[i]| << 16)) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_sad_u8_sum(s0, s1)
    acc = s2 & MASK32
    shifted_sad = (sad << 16)
    full_sum = acc + shifted_sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


def ref_msad_u8(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    """Bit-accurate reference mathematical oracle for v_msad_u8.

    dst = (s2 + sum_{i=0..3, s1.byte[i] != 0} |s0.byte[i] - s1.byte[i]|) mod 2^32.
    If clamp is True, saturates to 0xFFFFFFFF on overflow.
    """
    sad = ref_msad_u8_sum(s0, s1)
    acc = s2 & MASK32
    full_sum = acc + sad
    if clamp:
        return min(CLAMP_MAX_U32, full_sum)
    return full_sum & MASK32


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
class VSadU8Instruction:
    """Symbolic AST node for v_sad_u8 dst, src0, src1, src2."""
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
            "v_sad_u8",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VSadHiU8Instruction:
    """Symbolic AST node for v_sad_hi_u8 dst, src0, src1, src2."""
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
            "v_sad_hi_u8",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VMsadU8Instruction:
    """Symbolic AST node for v_msad_u8 dst, src0, src1, src2."""
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
            "v_msad_u8",
            self.dst.disassemble() + ",",
            self.src0.disassemble() + ",",
            self.src1.disassemble() + ",",
            self.src2.disassemble(),
        ]
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


SadU8Instruction = Union[VSadU8Instruction, VSadHiU8Instruction, VMsadU8Instruction]


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
    """Deterministic instruction interpreter for VALU SAD u8 instructions on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def execute_sad_u8(self, instr: VSadU8Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        result = ref_sad_u8(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def execute_sad_hi_u8(self, instr: VSadHiU8Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        result = ref_sad_hi_u8(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def execute_msad_u8(self, instr: VMsadU8Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)
        result = ref_msad_u8(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
        return result

    def step(self, instr: SadU8Instruction) -> int:
        if isinstance(instr, VSadU8Instruction):
            return self.execute_sad_u8(instr)
        if isinstance(instr, VSadHiU8Instruction):
            return self.execute_sad_hi_u8(instr)
        if isinstance(instr, VMsadU8Instruction):
            return self.execute_msad_u8(instr)
        raise TypeError(f"Unsupported instruction AST: {type(instr)}")

    def run_program(self, instructions: Sequence[SadU8Instruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# Unit Tests: Reference Mathematical Oracles
# ==============================================================================

class TestValuSadU8Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracles of v_sad_u8, v_sad_hi_u8, and v_msad_u8."""

    def test_byte_abs_diff_bounds_and_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (0, 255),
            (255, 0),
            (128, 128),
            (10, 250),
            (250, 10),
            (127, 128),
        ]
        for b0, b1 in pairs:
            diff = ref_byte_abs_diff(b0, b1)
            self.assertGreaterEqual(diff, 0)
            self.assertLessEqual(diff, 255)
            self.assertEqual(diff, ref_byte_abs_diff(b1, b0))

    def test_unsigned_vs_signed_byte_difference(self) -> None:
        # 0x00 vs 0xFF:
        # In unsigned math: |0 - 255| = 255
        # In signed 8-bit math: 0xFF is -1, |0 - (-1)| = 1
        self.assertEqual(ref_byte_abs_diff(0x00, 0xFF), 255)
        self.assertEqual(ref_byte_abs_diff(0xFF, 0x00), 255)

    def test_sad_u8_sum_all_zero_and_all_max(self) -> None:
        # Identical words -> 0
        self.assertEqual(ref_sad_u8_sum(0x1234_5678, 0x1234_5678), 0)
        self.assertEqual(ref_sad_u8_sum(0x0000_0000, 0x0000_0000), 0)
        self.assertEqual(ref_sad_u8_sum(0xFFFF_FFFF, 0xFFFF_FFFF), 0)

        # Max possible difference: each byte diff is 255 -> 4 * 255 = 1020 (0x3FC)
        self.assertEqual(ref_sad_u8_sum(0x0000_0000, 0xFFFF_FFFF), 1020)
        self.assertEqual(ref_sad_u8_sum(0xFFFF_FFFF, 0x0000_0000), 1020)

    def test_sad_u8_sum_individual_byte_positions(self) -> None:
        # Test each byte lane independently
        s0 = pack_bytes(10, 20, 30, 40)
        s1 = pack_bytes(15, 20, 30, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 5)

        s1 = pack_bytes(10, 25, 30, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 5)

        s1 = pack_bytes(10, 20, 10, 40)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 20)

        s1 = pack_bytes(10, 20, 30, 0)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 40)

    def test_sad_u8_accumulation_and_identity(self) -> None:
        s0 = pack_bytes(1, 2, 3, 4)
        s1 = pack_bytes(5, 6, 7, 8)
        # diffs: |1-5|=4, |2-6|=4, |3-7|=4, |4-8|=4 -> sum = 16
        self.assertEqual(ref_sad_u8_sum(s0, s1), 16)
        self.assertEqual(ref_sad_u8(s0, s1, 100), 116)
        self.assertEqual(ref_sad_u8(s0, s1, 0), 16)

    def test_sad_u8_wraparound_modulo_2_32(self) -> None:
        # s2 = 0xFFFFFFFF, sad = 1 -> wraps to 0
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        self.assertEqual(ref_sad_u8_sum(s0, s1), 1)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32, clamp=False), 0)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32 - 1, clamp=False), MASK32)

    def test_sad_u8_clamp_saturation(self) -> None:
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_sad_u8(s0, s1, MASK32 - 1, clamp=True), MASK32)
        self.assertEqual(ref_sad_u8(s0, s1, 50, clamp=True), 51)

    def test_sad_hi_u8_upper_word_shift_and_accumulation(self) -> None:
        s0 = pack_bytes(1, 2, 3, 4)
        s1 = pack_bytes(2, 3, 4, 5)
        # diffs: 1+1+1+1 = 4. Shifted: 4 << 16 = 0x0004_0000 = 262144
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0), 0x0004_0000)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0x1234), 0x0004_1234)

    def test_sad_hi_u8_clamping_saturation(self) -> None:
        # Max sad = 1020 (0x3FC) -> shifted = 0x03FC_0000
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000
        # s2 = 0xFC05_0000 -> 0xFC05_0000 + 0x03FC_0000 = 0x1_0001_0000 (overflows 32-bit)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0xFC05_0000, clamp=False), 0x0001_0000)
        self.assertEqual(ref_sad_hi_u8(s0, s1, 0xFC05_0000, clamp=True), MASK32)

    def test_msad_u8_zero_masking_skips_inactive_bytes(self) -> None:
        # s1 has zeros in bytes 0 and 2
        s0 = pack_bytes(100, 20, 200, 40)
        s1 = pack_bytes(0, 25, 0, 50)
        # byte 0: s1=0 -> skipped (diff 0)
        # byte 1: s1=25 -> |20 - 25| = 5
        # byte 2: s1=0 -> skipped (diff 0)
        # byte 3: s1=50 -> |40 - 50| = 10
        # sum = 15
        self.assertEqual(ref_msad_u8_sum(s0, s1), 15)
        self.assertEqual(ref_msad_u8(s0, s1, 100), 115)

        # Contrast with unmasked sad_u8:
        # byte 0 diff = 100, byte 2 diff = 200 -> total = 100 + 5 + 200 + 10 = 315
        self.assertEqual(ref_sad_u8_sum(s0, s1), 315)

    def test_msad_u8_all_zeros_mask_yields_zero_sad(self) -> None:
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000  # all template bytes are 0
        self.assertEqual(ref_msad_u8_sum(s0, s1), 0)
        self.assertEqual(ref_msad_u8(s0, s1, 0x1234_5678), 0x1234_5678)

    def test_msad_u8_non_zero_reference_matching(self) -> None:
        # When no bytes in s1 are zero, msad equals sad
        s0 = pack_bytes(10, 20, 30, 40)
        s1 = pack_bytes(1, 2, 3, 4)
        self.assertEqual(ref_msad_u8_sum(s0, s1), ref_sad_u8_sum(s0, s1))
        self.assertEqual(ref_msad_u8(s0, s1, 50), ref_sad_u8(s0, s1, 50))

    def test_msad_u8_clamp_behavior(self) -> None:
        s0 = pack_bytes(255, 255, 255, 255)
        s1 = pack_bytes(1, 1, 1, 1)
        # diff = 4 * 254 = 1016
        self.assertEqual(ref_msad_u8(s0, s1, MASK32, clamp=True), MASK32)
        self.assertEqual(ref_msad_u8(s0, s1, MASK32, clamp=False), 1015)

    def test_sad_u8_and_sad_hi_u8_multi_precision_composition(self) -> None:
        # Combining sad_u8 and sad_hi_u8 enables dual-accumulator multi-precision tracking:
        s0 = pack_bytes(100, 150, 200, 250)
        s1 = pack_bytes(50, 75, 100, 125)
        sad = ref_sad_u8_sum(s0, s1)  # 50 + 75 + 100 + 125 = 350
        res_lo = ref_sad_u8(s0, s1, 0)
        res_hi = ref_sad_hi_u8(s0, s1, 0)
        self.assertEqual(res_lo, 350)
        self.assertEqual(res_hi, 350 << 16)
        self.assertEqual((res_lo << 16) & MASK32, res_hi)


class TestValuSadU8SymbolicAST(unittest.TestCase):
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

    def test_v_sad_u8_ast_construction_and_disasm(self) -> None:
        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(100),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_sad_u8 v0, v1, s2, 0x00000064")

    def test_v_sad_u8_clamp_modifier_disasm(self) -> None:
        instr = VSadU8Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(6),
            src1=Operand.vgpr(7),
            src2=Operand.vgpr(8),
            clamp=True,
        )
        self.assertEqual(instr.disassemble(), "v_sad_u8 v5, v6, v7, v8 clamp")

    def test_v_sad_hi_u8_ast_construction_and_disasm(self) -> None:
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(1),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(4),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_sad_hi_u8 v1, v2, v3, v4")

        instr_clamped = VSadHiU8Instruction(
            dst=Operand.vgpr(1),
            src0=Operand.vgpr(2),
            src1=Operand.vgpr(3),
            src2=Operand.vgpr(4),
            clamp=True,
        )
        self.assertEqual(instr_clamped.disassemble(), "v_sad_hi_u8 v1, v2, v3, v4 clamp")

    def test_v_msad_u8_ast_construction_and_disasm(self) -> None:
        instr = VMsadU8Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(11),
            src1=Operand.sgpr(12),
            src2=Operand.vgpr(13),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_msad_u8 v10, v11, s12, v13")

        instr_clamped = VMsadU8Instruction(
            dst=Operand.vgpr(10),
            src0=Operand.vgpr(11),
            src1=Operand.sgpr(12),
            src2=Operand.vgpr(13),
            clamp=True,
        )
        self.assertEqual(instr_clamped.disassemble(), "v_msad_u8 v10, v11, s12, v13 clamp")

    def test_instruction_validation_rejects_non_vgpr_dst(self) -> None:
        with self.assertRaises(TypeError):
            VSadU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.literal(42),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VMsadU8Instruction(
                dst=Operand.sgpr(5),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_instruction_validation_rejects_multi_dword_operands(self) -> None:
        with self.assertRaises(ValueError):
            VSadU8Instruction(
                dst=Operand.vgpr(0, count=2),
                src0=Operand.vgpr(2),
                src1=Operand.vgpr(3),
                src2=Operand.vgpr(4),
            )
        with self.assertRaises(ValueError):
            VSadU8Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(3),
                src2=Operand.vgpr(4),
            )


class TestValuSadU8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.rf)

    def test_execute_sad_u8_basic(self) -> None:
        self.rf.vgprs[1] = pack_bytes(10, 20, 30, 40)
        self.rf.vgprs[2] = pack_bytes(12, 24, 36, 48)
        self.rf.vgprs[3] = 100
        # diffs: 2 + 4 + 6 + 8 = 20. Result = 100 + 20 = 120
        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 120)
        self.assertEqual(self.rf.vgprs[0], 120)

    def test_execute_sad_u8_with_literal_operands(self) -> None:
        self.rf.vgprs[1] = pack_bytes(5, 5, 5, 5)
        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.literal(pack_bytes(1, 1, 1, 1)),
            src2=Operand.literal(50),
        )
        # diff = 4 * 4 = 16. Result = 50 + 16 = 66
        res = self.interp.step(instr)
        self.assertEqual(res, 66)
        self.assertEqual(self.rf.vgprs[0], 66)

    def test_execute_sad_u8_clamp_saturation(self) -> None:
        self.rf.vgprs[1] = 0xFFFF_FFFF
        self.rf.vgprs[2] = 0x0000_0000
        self.rf.vgprs[3] = MASK32 - 100
        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, MASK32)
        self.assertEqual(self.rf.vgprs[0], MASK32)

    def test_execute_sad_hi_u8_execution(self) -> None:
        self.rf.vgprs[1] = pack_bytes(10, 10, 10, 10)
        self.rf.vgprs[2] = pack_bytes(11, 11, 11, 11)
        self.rf.vgprs[3] = 0x0001_0000
        # diff = 4. Shifted = 4 << 16 = 0x0004_0000. Total = 0x0005_0000
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 0x0005_0000)
        self.assertEqual(self.rf.vgprs[0], 0x0005_0000)

    def test_execute_sad_hi_u8_clamp(self) -> None:
        self.rf.vgprs[1] = 0xFFFF_FFFF
        self.rf.vgprs[2] = 0x0000_0000
        self.rf.vgprs[3] = 0xFC05_0000  # Will overflow with 0x03FC_0000
        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
            clamp=True,
        )
        res = self.interp.step(instr)
        self.assertEqual(res, MASK32)
        self.assertEqual(self.rf.vgprs[0], MASK32)

    def test_execute_msad_u8_masked_execution(self) -> None:
        self.rf.vgprs[1] = pack_bytes(50, 100, 150, 200)
        self.rf.sgprs[5] = pack_bytes(0, 80, 0, 180)
        self.rf.vgprs[2] = 1000
        # bytes 0 and 2 masked. bytes 1 and 3: |100-80|=20, |200-180|=20 -> sum=40
        instr = VMsadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(5),
            src2=Operand.vgpr(2),
        )
        res = self.interp.step(instr)
        self.assertEqual(res, 1040)
        self.assertEqual(self.rf.vgprs[0], 1040)

    def test_execute_msad_u8_clamp(self) -> None:
        self.rf.vgprs[1] = pack_bytes(200, 200, 200, 200)
        self.rf.vgprs[2] = pack_bytes(10, 10, 10, 10)
        self.rf.vgprs[3] = MASK32
        instr = VMsadU8Instruction(
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
        # Multi-step accumulation:
        # v0 = sad_u8(v1, v2, 0)
        # v0 = sad_u8(v3, v4, v0)
        self.rf.vgprs[1] = pack_bytes(1, 2, 3, 4)
        self.rf.vgprs[2] = pack_bytes(2, 3, 4, 5)  # sad = 4
        self.rf.vgprs[3] = pack_bytes(10, 20, 30, 40)
        self.rf.vgprs[4] = pack_bytes(15, 25, 35, 45)  # sad = 20
        prog = [
            VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.literal(0)),
            VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(3), Operand.vgpr(4), Operand.vgpr(0)),
        ]
        self.interp.run_program(prog)
        self.assertEqual(self.rf.vgprs[0], 24)

    def test_unsupported_instruction_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            self.interp.step("not_an_instruction")  # type: ignore


if __name__ == "__main__":
    unittest.main()
