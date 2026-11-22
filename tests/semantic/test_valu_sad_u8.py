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
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """Represents the execution state of an AMD RDNA2 wavefront (Wave32 or Wave64)."""

    def __init__(self, wave_size: int = 32, exec_mask: int | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size {wave_size}; must be 32 or 64")
        self.wave_size: int = wave_size
        default_mask = MASK32 if wave_size == 32 else MASK64
        self.exec_mask: int = (exec_mask & default_mask) if exec_mask is not None else default_mask
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * 106

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane index {lane} out of range [0, {self.wave_size - 1}]")
        return bool((self.exec_mask >> lane) & 1)

    def read_lane(self, lane: int, op: Operand) -> int:
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write_lane(self, lane: int, op: Operand, value: int) -> None:
        if op.reg_type != RegType.VGPR:
            raise TypeError("VALU SIMD write must target VGPRs")
        self.vgprs[lane][op.index] = u32(value)


class SIMDVALUInterpreter:
    """SIMD wavefront instruction interpreter honoring 32/64-bit EXEC masking."""

    def __init__(self, wave: SIMDWaveState) -> None:
        self.wave: SIMDWaveState = wave

    def step(self, instr: SadU8Instruction) -> List[int | None]:
        results: List[int | None] = []
        for lane in range(self.wave.wave_size):
            if not self.wave.is_lane_active(lane):
                results.append(None)
                continue

            s0 = self.wave.read_lane(lane, instr.src0)
            s1 = self.wave.read_lane(lane, instr.src1)
            s2 = self.wave.read_lane(lane, instr.src2)

            if isinstance(instr, VSadU8Instruction):
                res = ref_sad_u8(s0, s1, s2, clamp=instr.clamp)
            elif isinstance(instr, VSadHiU8Instruction):
                res = ref_sad_hi_u8(s0, s1, s2, clamp=instr.clamp)
            elif isinstance(instr, VMsadU8Instruction):
                res = ref_msad_u8(s0, s1, s2, clamp=instr.clamp)
            else:
                raise TypeError(f"Unsupported instruction AST: {type(instr)}")

            self.wave.write_lane(lane, instr.dst, res)
            results.append(res)
        return results

    def run_program(self, instructions: Sequence[SadU8Instruction]) -> None:
        for instr in instructions:
            self.step(instr)


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class VALUStatusFlags:
    """Architectural condition codes and status flags for 32-bit ALU operations."""
    zero: bool
    sign: bool
    carry: bool
    overflow: bool

    @classmethod
    def from_sad_u8(cls, s0: int, s1: int, s2: int) -> VALUStatusFlags:
        sad = ref_sad_u8_sum(s0, s1)
        acc = s2 & MASK32
        full_sum = acc + sad
        result = full_sum & MASK32

        zero = (result == 0)
        sign = bool(result & 0x8000_0000)
        carry = (full_sum >= 0x1_0000_0000)
        s_acc = s32(acc)
        s_sad = sad
        s_res = s32(result)
        overflow = (s_acc > 0 and s_sad > 0 and s_res < 0)

        return cls(zero=zero, sign=sign, carry=carry, overflow=overflow)

    @classmethod
    def from_sad_hi_u8(cls, s0: int, s1: int, s2: int) -> VALUStatusFlags:
        sad = ref_sad_u8_sum(s0, s1)
        acc = s2 & MASK32
        shifted = (sad << 16)
        full_sum = acc + shifted
        result = full_sum & MASK32

        zero = (result == 0)
        sign = bool(result & 0x8000_0000)
        carry = (full_sum >= 0x1_0000_0000)
        s_acc = s32(acc)
        s_shifted = s32(shifted)
        s_res = s32(result)
        overflow = (s_acc > 0 and s_shifted > 0 and s_res < 0)

        return cls(zero=zero, sign=sign, carry=carry, overflow=overflow)

    @classmethod
    def from_msad_u8(cls, s0: int, s1: int, s2: int) -> VALUStatusFlags:
        sad = ref_msad_u8_sum(s0, s1)
        acc = s2 & MASK32
        full_sum = acc + sad
        result = full_sum & MASK32

        zero = (result == 0)
        sign = bool(result & 0x8000_0000)
        carry = (full_sum >= 0x1_0000_0000)
        s_acc = s32(acc)
        s_sad = sad
        s_res = s32(result)
        overflow = (s_acc > 0 and s_sad > 0 and s_res < 0)

        return cls(zero=zero, sign=sign, carry=carry, overflow=overflow)


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


class TestValuSadU8SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFF_FFFF)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][1] = pack_bytes(lane, lane, lane, lane)
            wave.vgprs[lane][2] = pack_bytes(0, 0, 0, 0)
            wave.vgprs[lane][3] = 100

        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            expected = 4 * lane + 100
            self.assertEqual(results[lane], expected)
            self.assertEqual(wave.vgprs[lane][0], expected)

    def test_wave32_divergent_mask_preserves_inactive_lanes(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        mask = 0x5555_5555
        wave = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDVALUInterpreter(wave)

        for lane in range(32):
            wave.vgprs[lane][0] = 0xCAFE_0000 + lane  # initial dst value
            wave.vgprs[lane][1] = pack_bytes(10, 20, 30, 40)
            wave.vgprs[lane][2] = pack_bytes(5, 10, 15, 20)  # diff = 5+10+15+20 = 50
            wave.vgprs[lane][3] = 1000

        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 1050)
                self.assertEqual(wave.vgprs[lane][0], 1050)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.vgprs[lane][0], 0xCAFE_0000 + lane)

    def test_wave32_all_inactive_lanes(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][0] = 0xDEAD_BEEF
        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.literal(1),
            src1=Operand.literal(2),
            src2=Operand.literal(3),
        )
        results = interp.step(instr)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0xDEAD_BEEF)

    def test_wave64_multi_lane_divergence(self) -> None:
        mask64 = 0x0123_4567_89AB_CDEF
        wave = SIMDWaveState(wave_size=64, exec_mask=mask64)
        interp = SIMDVALUInterpreter(wave)

        for lane in range(64):
            wave.vgprs[lane][0] = 0x7777 * (lane + 1)
            wave.vgprs[lane][1] = pack_bytes(lane % 50, (lane + 1) % 50, 0, 0)
            wave.vgprs[lane][2] = pack_bytes(10, 10, 0, 0)
            wave.vgprs[lane][3] = 500

        instr = VSadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(64):
            is_active = bool((mask64 >> lane) & 1)
            if is_active:
                expected = ref_sad_u8(wave.vgprs[lane][1], wave.vgprs[lane][2], 500)
                self.assertEqual(results[lane], expected)
                self.assertEqual(wave.vgprs[lane][0], expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.vgprs[lane][0], 0x7777 * (lane + 1))

    def test_wave32_msad_u8_simd_divergence(self) -> None:
        mask = 0x00FF_00FF
        wave = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][0] = 0
            wave.vgprs[lane][1] = pack_bytes(10, 20, 30, 40)
            wave.vgprs[lane][2] = pack_bytes(0, 10, 0, 20)  # bytes 0, 2 masked
            wave.vgprs[lane][3] = 100

        instr = VMsadU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            if bool((mask >> lane) & 1):
                self.assertEqual(results[lane], 130)  # |20-10| + |40-20| = 30; 100 + 30 = 130
                self.assertEqual(wave.vgprs[lane][0], 130)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.vgprs[lane][0], 0)

    def test_wave32_sad_hi_u8_simd_divergence(self) -> None:
        mask = 0x0000_000F  # first 4 lanes active
        wave = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDVALUInterpreter(wave)
        for lane in range(32):
            wave.vgprs[lane][0] = 0xAAAA_BBBB
            wave.vgprs[lane][1] = pack_bytes(1, 2, 3, 4)
            wave.vgprs[lane][2] = pack_bytes(2, 3, 4, 5)  # sad = 4, shifted = 0x0004_0000
            wave.vgprs[lane][3] = 0x0001_0000

        instr = VSadHiU8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = interp.step(instr)
        for lane in range(32):
            if lane < 4:
                self.assertEqual(results[lane], 0x0005_0000)
                self.assertEqual(wave.vgprs[lane][0], 0x0005_0000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(wave.vgprs[lane][0], 0xAAAA_BBBB)


class TestValuSadU8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.rf)

    def test_alias_dst_src0(self) -> None:
        # v0 = sad_u8(v0, v1, v2)
        self.rf.vgprs[0] = pack_bytes(50, 60, 70, 80)
        self.rf.vgprs[1] = pack_bytes(20, 20, 20, 20)
        self.rf.vgprs[2] = 10
        # diffs: 30 + 40 + 50 + 60 = 180. Result = 180 + 10 = 190
        instr = VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2))
        res = self.interp.step(instr)
        self.assertEqual(res, 190)
        self.assertEqual(self.rf.vgprs[0], 190)
        # Other registers preserved
        self.assertEqual(self.rf.vgprs[1], pack_bytes(20, 20, 20, 20))
        self.assertEqual(self.rf.vgprs[2], 10)

    def test_alias_dst_src1(self) -> None:
        # v0 = sad_u8(v1, v0, v2)
        self.rf.vgprs[0] = pack_bytes(10, 20, 30, 40)
        self.rf.vgprs[1] = pack_bytes(15, 25, 35, 45)
        self.rf.vgprs[2] = 500
        # diffs = 5 * 4 = 20. Result = 520
        instr = VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(0), Operand.vgpr(2))
        res = self.interp.step(instr)
        self.assertEqual(res, 520)
        self.assertEqual(self.rf.vgprs[0], 520)

    def test_alias_dst_src2_accumulator_loop(self) -> None:
        # Canonical loop accumulation pattern: v0 = sad_u8(v1, v2, v0)
        self.rf.vgprs[0] = 1000
        self.rf.vgprs[1] = pack_bytes(1, 2, 3, 4)
        self.rf.vgprs[2] = pack_bytes(2, 4, 6, 8)
        # diffs: 1 + 2 + 3 + 4 = 10. Result = 1000 + 10 = 1010
        instr = VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 1010)
        self.assertEqual(self.rf.vgprs[0], 1010)

    def test_alias_all_same_register(self) -> None:
        # v0 = sad_u8(v0, v0, v0)
        self.rf.vgprs[0] = 0x1234_5678
        instr = VSadU8Instruction(Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(0), Operand.vgpr(0))
        res = self.interp.step(instr)
        # |v0 - v0| = 0 -> 0 + v0 = v0
        self.assertEqual(res, 0x1234_5678)
        self.assertEqual(self.rf.vgprs[0], 0x1234_5678)

    def test_alias_sad_hi_dst_src2(self) -> None:
        # v0 = sad_hi_u8(v1, v2, v0)
        self.rf.vgprs[0] = 0x0001_0000
        self.rf.vgprs[1] = pack_bytes(5, 5, 5, 5)
        self.rf.vgprs[2] = pack_bytes(3, 3, 3, 3)
        # diff = 4 * 2 = 8. Shifted = 8 << 16 = 0x0008_0000. Total = 0x0009_0000
        instr = VSadHiU8Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 0x0009_0000)
        self.assertEqual(self.rf.vgprs[0], 0x0009_0000)

    def test_alias_msad_dst_src2(self) -> None:
        # v0 = msad_u8(v1, v2, v0)
        self.rf.vgprs[0] = 50
        self.rf.vgprs[1] = pack_bytes(10, 20, 30, 40)
        self.rf.vgprs[2] = pack_bytes(0, 10, 0, 20)  # masked bytes 0, 2
        # diffs: |20-10| + |40-20| = 30. Result = 50 + 30 = 80
        instr = VMsadU8Instruction(Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(0))
        res = self.interp.step(instr)
        self.assertEqual(res, 80)
        self.assertEqual(self.rf.vgprs[0], 80)

    def test_operand_preservation_unmodified_registers(self) -> None:
        # Ensure only the destination register is modified and all other 255 VGPRs are preserved
        for i in range(256):
            self.rf.vgprs[i] = 0x1000 + i
        instr = VSadU8Instruction(
            dst=Operand.vgpr(42),
            src0=Operand.vgpr(10),
            src1=Operand.vgpr(20),
            src2=Operand.vgpr(30),
        )
        self.interp.step(instr)
        for i in range(256):
            if i != 42:
                self.assertEqual(self.rf.vgprs[i], 0x1000 + i)


class TestValuSadU8StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_carry_generation_exact_boundary(self) -> None:
        # Sum = 0xFFFFFFFF -> no carry, sign=True (bit 31=1), zero=False
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)  # sad = 1
        flags_no_carry = VALUStatusFlags.from_sad_u8(s0, s1, 0xFFFF_FFFE)
        self.assertFalse(flags_no_carry.carry)
        self.assertFalse(flags_no_carry.zero)
        self.assertTrue(flags_no_carry.sign)

        # Sum = 0x1_0000_0000 -> carry = True, zero = True, sign = False
        flags_carry = VALUStatusFlags.from_sad_u8(s0, s1, MASK32)
        self.assertTrue(flags_carry.carry)
        self.assertTrue(flags_carry.zero)
        self.assertFalse(flags_carry.sign)

    def test_carry_generation_sad_hi_u8(self) -> None:
        # shifted sad = 0x03FC_0000. acc = 0xFC05_0000 -> full_sum = 0x1_0001_0000 -> carry = True
        s0 = 0xFFFF_FFFF
        s1 = 0x0000_0000
        flags = VALUStatusFlags.from_sad_hi_u8(s0, s1, 0xFC05_0000)
        self.assertTrue(flags.carry)
        self.assertFalse(flags.zero)

        # No carry case
        flags_no_carry = VALUStatusFlags.from_sad_hi_u8(s0, s1, 0x0001_0000)
        self.assertFalse(flags_no_carry.carry)

    def test_zero_flag_behaviors(self) -> None:
        # Diff = 0, acc = 0 -> zero = True, carry = False, sign = False
        flags_zero = VALUStatusFlags.from_sad_u8(0x1234_5678, 0x1234_5678, 0)
        self.assertTrue(flags_zero.zero)
        self.assertFalse(flags_zero.carry)
        self.assertFalse(flags_zero.sign)

        # Diff = 5, acc = 0 -> zero = False
        s0 = pack_bytes(5, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        flags_nonzero = VALUStatusFlags.from_sad_u8(s0, s1, 0)
        self.assertFalse(flags_nonzero.zero)

    def test_sign_flag_behaviors(self) -> None:
        # Result < 0x80000000 -> sign = False
        flags_pos = VALUStatusFlags.from_sad_u8(0, 0, 0x7FFF_FFFF)
        self.assertFalse(flags_pos.sign)

        # Result == 0x80000000 -> sign = True
        flags_neg = VALUStatusFlags.from_sad_u8(0, 0, 0x8000_0000)
        self.assertTrue(flags_neg.sign)

    def test_overflow_flag_behavior(self) -> None:
        # acc is positive signed (0x7FFF_FFFF), adding positive sad wraps into negative signed space
        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)  # sad = 1
        flags_ovf = VALUStatusFlags.from_sad_u8(s0, s1, 0x7FFF_FFFF)
        self.assertTrue(flags_ovf.overflow)
        self.assertTrue(flags_ovf.sign)

    def test_msad_u8_flags(self) -> None:
        # all template bytes zero -> sad = 0. acc = 0 -> zero = True
        flags = VALUStatusFlags.from_msad_u8(0xFFFF_FFFF, 0, 0)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.carry)


class TestValuSadU8IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit-pattern extremes, subnormals, and NaN handling."""

    POS_ZERO_F32: int = 0x0000_0000
    NEG_ZERO_F32: int = 0x8000_0000
    POS_INF_F32: int = 0x7F80_0000
    NEG_INF_F32: int = 0xFF80_0000
    QNAN_F32: int = 0x7FC0_0000
    SNAN_F32: int = 0x7F80_0001
    MIN_NORMAL_F32: int = 0x0080_0000
    MAX_SUBNORMAL_F32: int = 0x007F_FFFF
    MIN_SUBNORMAL_F32: int = 0x0000_0001

    @staticmethod
    def f32_to_bits(val: float) -> int:
        return struct.unpack("<I", struct.pack("<f", val))[0]

    def test_signed_zero_byte_distance(self) -> None:
        # Distance between +0.0 (0x00000000) and -0.0 (0x80000000)
        # Differ only in MSB of byte 3 (|0x00 - 0x80| = 128)
        dist = ref_sad_u8_sum(self.POS_ZERO_F32, self.NEG_ZERO_F32)
        self.assertEqual(dist, 128)
        self.assertEqual(ref_sad_u8(self.POS_ZERO_F32, self.NEG_ZERO_F32, 100), 228)

    def test_nan_payload_byte_isolation(self) -> None:
        # Distance between +infinity (0x7F800000) and quiet NaN (0x7FC00000)
        # Bytes: [0, 0, 0x80, 0x7F] vs [0, 0, 0xC0, 0x7F] -> diff is |0x80 - 0xC0| = 64
        dist_qnan = ref_sad_u8_sum(self.POS_INF_F32, self.QNAN_F32)
        self.assertEqual(dist_qnan, 64)

        # Distance between +infinity (0x7F800000) and signaling NaN (0x7F800001)
        # Differ only in byte 0: |0 - 1| = 1
        dist_snan = ref_sad_u8_sum(self.POS_INF_F32, self.SNAN_F32)
        self.assertEqual(dist_snan, 1)

    def test_subnormal_to_normal_boundary_step(self) -> None:
        # Between min normal (0x00800000) and max subnormal (0x007FFFFF)
        # Bytes of 0x00800000: [0, 0, 0x80, 0]
        # Bytes of 0x007FFFFF: [0xFF, 0xFF, 0x7F, 0]
        # Diffs: 255 + 255 + 1 + 0 = 511
        dist = ref_sad_u8_sum(self.MIN_NORMAL_F32, self.MAX_SUBNORMAL_F32)
        self.assertEqual(dist, 511)
        self.assertEqual(ref_sad_u8(self.MIN_NORMAL_F32, self.MAX_SUBNORMAL_F32, 0), 511)

    def test_inf_to_neg_inf_byte_distance(self) -> None:
        # +Inf (0x7F800000) vs -Inf (0xFF800000)
        # Differ only in MSB sign byte 3: |0x7F - 0xFF| = 128
        dist = ref_sad_u8_sum(self.POS_INF_F32, self.NEG_INF_F32)
        self.assertEqual(dist, 128)

    def test_nan_quiet_bit_msad(self) -> None:
        # Using QNaN as template in msad_u8:
        # Bytes of QNAN (0x7FC00000): byte 0=0, byte 1=0, byte 2=0xC0, byte 3=0x7F
        # Bytes 0 and 1 are masked out because they are zero!
        s0 = 0xFFFF_FFFF
        # byte 2 diff: |0xFF - 0xC0| = 63
        # byte 3 diff: |0xFF - 0x7F| = 128
        # total msad = 63 + 128 = 191
        self.assertEqual(ref_msad_u8_sum(s0, self.QNAN_F32), 191)

    def test_float_adjacent_ulp_distance(self) -> None:
        # Bit pattern for 1.0f is 0x3F800000; next adjacent float is 0x3F800001
        b_one = self.f32_to_bits(1.0)
        b_next = b_one + 1
        self.assertEqual(ref_sad_u8_sum(b_one, b_next), 1)


class TestValuSadU8NegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_negative_control_sgpr_as_destination(self) -> None:
        with self.assertRaises(TypeError):
            VSadU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VSadHiU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )
        with self.assertRaises(TypeError):
            VMsadU8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_negative_control_literal_as_destination(self) -> None:
        with self.assertRaises(TypeError):
            VSadU8Instruction(
                dst=Operand.literal(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr(3),
            )

    def test_negative_control_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)

    def test_negative_control_wave_state_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        wave = SIMDWaveState(wave_size=32)
        with self.assertRaises(IndexError):
            wave.is_lane_active(32)
        with self.assertRaises(IndexError):
            wave.is_lane_active(-1)

    def test_mutation_detection_signed_vs_unsigned_byte_diff(self) -> None:
        def mutant_signed_byte_sad(s0: int, s1: int) -> int:
            total = 0
            for i in range(4):
                b0 = extract_byte(s0, i)
                b1 = extract_byte(s1, i)
                # Faulty signed interpretation: 0xFF is -1
                s8_0 = b0 - 256 if (b0 & 0x80) else b0
                s8_1 = b1 - 256 if (b1 & 0x80) else b1
                total += abs(s8_0 - s8_1)
            return total

        s0 = pack_bytes(0, 0, 0, 0)
        s1 = pack_bytes(255, 255, 255, 255)
        # In correct unsigned SAD: |0 - 255| * 4 = 1020
        # In mutant signed SAD: |0 - (-1)| * 4 = 4
        correct = ref_sad_u8_sum(s0, s1)
        mutant = mutant_signed_byte_sad(s0, s1)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, 1020)
        self.assertEqual(mutant, 4)

    def test_mutation_detection_sad_hi_shift(self) -> None:
        def mutant_sad_hi_u8(s0: int, s1: int, s2: int) -> int:
            # Faulty: forgets << 16 shift
            sad = ref_sad_u8_sum(s0, s1)
            return (s2 + sad) & MASK32

        s0 = pack_bytes(1, 1, 1, 1)
        s1 = pack_bytes(0, 0, 0, 0)
        s2 = 0x0001_0000
        # Correct: sad=4, shifted = 4 << 16 = 0x0004_0000 -> result = 0x0005_0000
        # Mutant: 0x0001_0000 + 4 = 0x0001_0004
        correct = ref_sad_hi_u8(s0, s1, s2)
        mutant = mutant_sad_hi_u8(s0, s1, s2)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, 0x0005_0000)
        self.assertEqual(mutant, 0x0001_0004)

    def test_mutation_detection_msad_mask_check(self) -> None:
        def mutant_msad_u8_sum(s0: int, s1: int) -> int:
            # Faulty: fails to mask out when s1 byte is zero
            total = 0
            for i in range(4):
                total += ref_byte_abs_diff(extract_byte(s0, i), extract_byte(s1, i))
            return total

        s0 = pack_bytes(100, 100, 100, 100)
        s1 = pack_bytes(0, 100, 0, 100)
        # Correct msad: bytes 0 and 2 skipped; bytes 1 and 3 have diff 0 -> sum = 0
        # Mutant: bytes 0 and 2 have diff 100 each -> sum = 200
        correct = ref_msad_u8_sum(s0, s1)
        mutant = mutant_msad_u8_sum(s0, s1)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, 0)
        self.assertEqual(mutant, 200)

    def test_mutation_detection_clamping_wrap(self) -> None:
        def mutant_clamped_sad_u8(s0: int, s1: int, s2: int) -> int:
            # Faulty: wraps modulo 2^32 instead of saturating
            return (ref_sad_u8_sum(s0, s1) + (s2 & MASK32)) & MASK32

        s0 = pack_bytes(1, 0, 0, 0)
        s1 = pack_bytes(0, 0, 0, 0)
        s2 = MASK32
        # Correct: saturated to MASK32 (0xFFFFFFFF)
        # Mutant: wraps to 0
        correct = ref_sad_u8(s0, s1, s2, clamp=True)
        mutant = mutant_clamped_sad_u8(s0, s1, s2)
        self.assertNotEqual(correct, mutant)
        self.assertEqual(correct, MASK32)
        self.assertEqual(mutant, 0)


if __name__ == "__main__":
    unittest.main()
