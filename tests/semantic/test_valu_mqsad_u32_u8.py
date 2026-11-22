from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mqsad_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_mqsad_u32_u8 dst, src0, src1, src2
  Syntax: v_mqsad_u32_u8 v[d:d+3], src0[0:1], src1, v[s2:s2+3]

Format:
  VOP3 (3-operand vector ALU instruction with multi-register operands).

Operands:
  - dst:  128-bit vector destination spanning 4 consecutive VGPRs: v[d], v[d+1], v[d+2], v[d+3]
  - src0: 64-bit source spanning 2 registers (VGPR pair, SGPR pair, or 64-bit constant)
  - src1: 32-bit source (1 VGPR, SGPR, or 32-bit constant)
  - src2: 128-bit vector accumulator spanning 4 consecutive VGPRs: v[s2], v[s2+1], v[s2+2], v[s2+3]

Operation:
  Computes four masked byte-level sums of absolute differences (SAD) with four
  full 32-bit accumulators:
    For each dword k in 0..3:
      W_k = (src0 >> (8 * k)) & 0xFFFFFFFF  (sliding 32-bit window across 64-bit src0)
      sad_k = 0
      For each byte i in 0..3:
        b0 = (W_k >> (8 * i)) & 0xFF
        b1 = (src1 >> (8 * i)) & 0xFF
        If b1 != 0:
          sad_k += |b0 - b1|
        Else:
          (byte skipped, adds 0)
      dst[k] = (src2[k] + sad_k) & 0xFFFFFFFF

Mathematical Properties and Bounds:
  - Operands:
      src0 in [0, 2^64 - 1]
      src1 in [0, 2^32 - 1]
      src2[k] in [0, 2^32 - 1]
  - Each byte difference |b0 - b1| is in [0, 255].
  - The maximum sum of 4 byte differences is 4 * 255 = 1020 (0x3FC).
  - Accumulation sum64 = src2[k] + sad_k is in [0, 2^32 - 1 + 1020] = [0, 0x1000003FB].
  - The carry-out past 32 bits into bit 32 is strictly in {0, 1}.
  - Zero-byte mask rule: When byte i of src1 is 0, that byte position is masked out and
    contributes 0 to the sum of differences regardless of the value in W_k.
  - The 8th byte of src0 (bits 63:56) is unused by the 4 windows (window 3 uses bytes 3..6).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple, Union
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK128: int = (1 << 128) - 1

MAX_BYTE_SAD: int = 4 * 255  # 1020 (0x3FC)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def extract_byte(word: int, byte_idx: int) -> int:
    """Extract byte at byte_idx (0..3) from a 32-bit word."""
    return (word >> (8 * (byte_idx & 3))) & MASK8


def byte_window_u32(val64: int, byte_offset: int) -> int:
    """Extract 32-bit sliding window W_k at byte offset k from 64-bit val: bits [8*k+31 : 8*k]."""
    return (val64 >> (8 * (byte_offset & 7))) & MASK32


def ref_masked_byte_sad_u32(win32: int, mask32: int) -> int:
    """Compute masked sum of absolute differences of 4 bytes.

    A byte where mask32's byte is 0 adds nothing.
    """
    total = 0
    for i in range(4):
        b_win = extract_byte(win32, i)
        b_mask = extract_byte(mask32, i)
        if b_mask != 0:
            diff = (b_win - b_mask) if b_win >= b_mask else (b_mask - b_win)
            total += diff
    return total


def ref_mqsad_u32_u8(s0_64: int, s1_32: int, s2_quad: Sequence[int]) -> List[int]:
    """Pure-Python bit-accurate reference mathematical oracle for AMD RDNA2 v_mqsad_u32_u8.

    Args:
        s0_64: 64-bit integer source operand.
        s1_32: 32-bit integer source operand (mask and reference bytes).
        s2_quad: Sequence of 4 unsigned 32-bit accumulator dwords.

    Returns:
        List of 4 unsigned 32-bit dwords representing v[d:d+3].
    """
    if len(s2_quad) != 4:
        raise ValueError(f"src2 accumulator must have 4 elements, got {len(s2_quad)}")

    res = []
    for k in range(4):
        win = byte_window_u32(s0_64, k)
        sad = ref_masked_byte_sad_u32(win, s1_32)
        acc = s2_quad[k] & MASK32
        res.append((acc + sad) & MASK32)
    return res


def ref_mqsad_full_acc(s0_64: int, s1_32: int, s2_quad: Sequence[int]) -> List[Tuple[int, int]]:
    """Compute (lo32, carry_out) for each of the 4 accumulators."""
    if len(s2_quad) != 4:
        raise ValueError(f"src2 accumulator must have 4 elements, got {len(s2_quad)}")

    out = []
    for k in range(4):
        win = byte_window_u32(s0_64, k)
        sad = ref_masked_byte_sad_u32(win, s1_32)
        acc = s2_quad[k] & MASK32
        full_sum = acc + sad
        out.append((full_sum & MASK32, (full_sum >> 32) & 1))
    return out


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    """Represents a register operand or inline/literal constant."""
    reg_type: RegType
    index: int = 0
    count: int = 1
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"Operand count must be >= 1, got {self.count}")
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR span v[{self.index}:{self.index + self.count - 1}] exceeds 256")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR span s[{self.index}:{self.index + self.count - 1}] exceeds 106")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            max_val = (1 << (32 * self.count)) - 1
            if not (0 <= self.literal_value <= max_val):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, {max_val:#x}]")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int, count: int = 1) -> Operand:
        max_val = (1 << (32 * count)) - 1
        return cls(RegType.LITERAL, count=count, literal_value=value & max_val)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v[{self.index}:{self.index + self.count - 1}]" if self.count > 1 else f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s[{self.index}:{self.index + self.count - 1}]" if self.count > 1 else f"s{self.index}"
        assert self.literal_value is not None
        if self.count == 2:
            return f"0x{self.literal_value:016X}"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VMqsadU32U8Instruction:
    """Symbolic AST node for AMD RDNA2 v_mqsad_u32_u8 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR or self.dst.count != 4:
            raise TypeError(f"Destination must be a 4-VGPR span v[d:d+3], got {self.dst}")
        if self.src0.count != 2:
            raise ValueError(f"src0 must be a 64-bit 2-register span or 64-bit literal, got {self.src0}")
        if self.src1.count != 1:
            raise ValueError(f"src1 must be a 32-bit single register or literal, got {self.src1}")
        if self.src2.reg_type != RegType.VGPR or self.src2.count != 4:
            raise TypeError(f"src2 accumulator must be a 4-VGPR span v[s2:s2+3], got {self.src2}")

    def disassemble(self) -> str:
        return f"v_mqsad_u32_u8 {self.dst.disassemble()}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for a single execution thread."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Operand) -> Union[int, List[int]]:
        """Read operand value as unsigned 32-bit, 64-bit integer, or quad dwords."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value

        if op.count == 1:
            if op.reg_type == RegType.VGPR:
                return self.vgprs[op.index] & MASK32
            if op.reg_type == RegType.SGPR:
                return self.sgprs[op.index] & MASK32
        elif op.count == 2:
            if op.reg_type == RegType.VGPR:
                lo = self.vgprs[op.index] & MASK32
                hi = self.vgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
            if op.reg_type == RegType.SGPR:
                lo = self.sgprs[op.index] & MASK32
                hi = self.sgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
        elif op.count == 4:
            if op.reg_type == RegType.VGPR:
                return [self.vgprs[op.index + k] & MASK32 for k in range(4)]
            if op.reg_type == RegType.SGPR:
                return [self.sgprs[op.index + k] & MASK32 for k in range(4)]

        raise ValueError(f"Unsupported operand: {op}")

    def write_operand(self, op: Operand, values: Union[int, Sequence[int]]) -> None:
        """Write 32-bit dword(s) to destination register operand."""
        if op.reg_type == RegType.LITERAL:
            raise TypeError("Cannot write to literal operand")

        val_list: Sequence[int] = [values] if isinstance(values, int) else values
        if len(val_list) != op.count:
            raise ValueError(f"Value count {len(val_list)} does not match operand count {op.count}")

        if op.reg_type == RegType.VGPR:
            for k in range(op.count):
                self.vgprs[op.index + k] = u32(val_list[k])
        elif op.reg_type == RegType.SGPR:
            for k in range(op.count):
                self.sgprs[op.index + k] = u32(val_list[k])


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for v_mqsad_u32_u8 on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def step(self, instr: VMqsadU32U8Instruction) -> List[int]:
        # Atomic read phase
        s0_val = self.regs.read_operand(instr.src0)
        s1_val = self.regs.read_operand(instr.src1)
        s2_vals = self.regs.read_operand(instr.src2)

        assert isinstance(s0_val, int)
        assert isinstance(s1_val, int)
        assert isinstance(s2_vals, list)

        # Computation phase
        result = ref_mqsad_u32_u8(s0_val, s1_val, s2_vals)

        # Write-back phase
        self.regs.write_operand(instr.dst, result)
        return result


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD register state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def read_lane(self, op: Operand, lane: int) -> Union[int, List[int]]:
        """Read operand for a specific SIMD lane."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value

        if op.count == 1:
            if op.reg_type == RegType.VGPR:
                return self.vgprs[lane][op.index] & MASK32
            if op.reg_type == RegType.SGPR:
                return self.sgprs[op.index] & MASK32
        elif op.count == 2:
            if op.reg_type == RegType.VGPR:
                lo = self.vgprs[lane][op.index] & MASK32
                hi = self.vgprs[lane][op.index + 1] & MASK32
                return (hi << 32) | lo
            if op.reg_type == RegType.SGPR:
                lo = self.sgprs[op.index] & MASK32
                hi = self.sgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
        elif op.count == 4:
            if op.reg_type == RegType.VGPR:
                return [self.vgprs[lane][op.index + k] & MASK32 for k in range(4)]
            if op.reg_type == RegType.SGPR:
                return [self.sgprs[op.index + k] & MASK32 for k in range(4)]

        raise ValueError(f"Unsupported operand: {op}")

    def write_lane(self, op: Operand, lane: int, values: Union[int, Sequence[int]]) -> None:
        """Write value(s) to a register operand for a specific SIMD lane."""
        if op.reg_type == RegType.LITERAL:
            raise TypeError("Cannot write to literal operand")

        val_list: Sequence[int] = [values] if isinstance(values, int) else values
        if len(val_list) != op.count:
            raise ValueError(f"Value count {len(val_list)} does not match operand count {op.count}")

        if op.reg_type == RegType.VGPR:
            for k in range(op.count):
                self.vgprs[lane][op.index + k] = u32(val_list[k])
        elif op.reg_type == RegType.SGPR:
            for k in range(op.count):
                self.sgprs[op.index + k] = u32(val_list[k])


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def step(self, instr: VMqsadU32U8Instruction, state: SIMDState) -> List[Optional[List[int]]]:
        results: List[Optional[List[int]]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [state.read_lane(instr.src0, lane) for lane in range(state.num_lanes)]
        s1_vals = [state.read_lane(instr.src1, lane) for lane in range(state.num_lanes)]
        s2_vals = [state.read_lane(instr.src2, lane) for lane in range(state.num_lanes)]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                s0 = s0_vals[lane]
                s1 = s1_vals[lane]
                s2 = s2_vals[lane]
                assert isinstance(s0, int)
                assert isinstance(s1, int)
                assert isinstance(s2, list)

                res = ref_mqsad_u32_u8(s0, s1, s2)
                state.write_lane(instr.dst, lane, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class MqsadDwordStatus:
    """Per-dword status flags for v_mqsad_u32_u8 accumulator."""
    carry: bool
    zero: bool
    sad_zero: bool
    max_sad: bool
    byte_sad: int
    full_sum: int


@dataclass(frozen=True)
class MqsadStatusFlags:
    """Aggregate status flags and carry generation for v_mqsad_u32_u8."""
    dword_status: Tuple[MqsadDwordStatus, MqsadDwordStatus, MqsadDwordStatus, MqsadDwordStatus]
    any_carry: bool
    all_carry: bool
    any_zero: bool
    all_zero: bool
    all_sad_zero: bool
    active_mask_bytes: int


def compute_mqsad_dword_status(win: int, mask: int, acc: int) -> MqsadDwordStatus:
    """Compute status and carry for a single accumulator dword."""
    sad = ref_masked_byte_sad_u32(win, mask)
    acc_u32 = acc & MASK32
    full_sum = acc_u32 + sad
    carry = bool(full_sum >= (1 << 32))
    res = full_sum & MASK32
    return MqsadDwordStatus(
        carry=carry,
        zero=(res == 0),
        sad_zero=(sad == 0),
        max_sad=(sad == MAX_BYTE_SAD),
        byte_sad=sad,
        full_sum=full_sum,
    )


def evaluate_mqsad_flags(s0_64: int, s1_32: int, s2_quad: Sequence[int]) -> MqsadStatusFlags:
    """Evaluate architectural condition codes and status flags for v_mqsad_u32_u8."""
    if len(s2_quad) != 4:
        raise ValueError(f"src2 accumulator must have 4 elements, got {len(s2_quad)}")

    statuses = []
    for k in range(4):
        win = byte_window_u32(s0_64, k)
        statuses.append(compute_mqsad_dword_status(win, s1_32, s2_quad[k]))

    d0, d1, d2, d3 = statuses
    tuple_statuses = (d0, d1, d2, d3)

    active_bytes = sum(1 for i in range(4) if extract_byte(s1_32, i) != 0)

    return MqsadStatusFlags(
        dword_status=tuple_statuses,
        any_carry=any(s.carry for s in statuses),
        all_carry=all(s.carry for s in statuses),
        any_zero=any(s.zero for s in statuses),
        all_zero=all(s.zero for s in statuses),
        all_sad_zero=all(s.sad_zero for s in statuses),
        active_mask_bytes=active_bytes,
    )


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMqsadU32U8Oracle(unittest.TestCase):
    """Unit tests for AMD RDNA2 v_mqsad_u32_u8 architectural bit model and reference oracle."""

    def test_known_vectors_basic(self) -> None:
        # Zero inputs -> zero outputs
        self.assertEqual(ref_mqsad_u32_u8(0, 0, [0, 0, 0, 0]), [0, 0, 0, 0])

        # All-zero src1: all mask bytes are 0, so SAD is 0 regardless of src0
        self.assertEqual(
            ref_mqsad_u32_u8(0xDEAD_BEEF_CAFE_BABE, 0x0000_0000, [10, 20, 30, 40]),
            [10, 20, 30, 40],
        )

    def test_sliding_window_byte_shifts(self) -> None:
        # src0 has distinct bytes: byte0=0x01, byte1=0x02, byte2=0x03, byte3=0x04,
        # byte4=0x05, byte5=0x06, byte6=0x07, byte7=0x08.
        src0 = 0x0807060504030201
        self.assertEqual(byte_window_u32(src0, 0), 0x04030201)
        self.assertEqual(byte_window_u32(src0, 1), 0x05040302)
        self.assertEqual(byte_window_u32(src0, 2), 0x06050403)
        self.assertEqual(byte_window_u32(src0, 3), 0x07060504)

    def test_isa_known_vector_sads(self) -> None:
        # Standard ISA known vector:
        # src0 = 0x0807060504030201, src1 = 0x04030201, accs = [0, 0, 0, 0]
        # Window 0: 0x04030201 vs 0x04030201 -> diffs (0, 0, 0, 0) -> SAD = 0
        # Window 1: 0x05040302 vs 0x04030201 -> diffs (1, 1, 1, 1) -> SAD = 4
        # Window 2: 0x06050403 vs 0x04030201 -> diffs (2, 2, 2, 2) -> SAD = 8
        # Window 3: 0x07060504 vs 0x04030201 -> diffs (3, 3, 3, 3) -> SAD = 12
        got = ref_mqsad_u32_u8(0x0807060504030201, 0x04030201, [0, 0, 0, 0])
        self.assertEqual(got, [0, 4, 8, 12])

        # With non-zero accumulators
        got_acc = ref_mqsad_u32_u8(0x0807060504030201, 0x04030201, [100, 200, 300, 400])
        self.assertEqual(got_acc, [100, 204, 308, 412])

    def test_masking_zero_byte_skips_accumulation(self) -> None:
        # src1 has only byte 0 non-zero: 0x00000005
        # Bytes 1, 2, 3 are 0 and MUST NOT contribute to the SAD
        win = 0xFFFFFFFF
        mask = 0x00000005
        # Only byte 0: |0xFF - 0x05| = 255 - 5 = 250
        self.assertEqual(ref_masked_byte_sad_u32(win, mask), 250)

        # Byte 1 only non-zero: 0x00000500
        mask_b1 = 0x00000500
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b1), 250)

        # Byte 2 only non-zero: 0x00050000
        mask_b2 = 0x00050000
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b2), 250)

        # Byte 3 only non-zero: 0x05000000
        mask_b3 = 0x05000000
        self.assertEqual(ref_masked_byte_sad_u32(win, mask_b3), 250)

    def test_accumulator_wrapping_u32(self) -> None:
        # src2 has accumulator 0xFFFFFFFF.
        # Adding SAD of 4 should wrap to 3: (0xFFFFFFFF + 4) mod 2^32 = 3.
        src0 = 0x0807060504030201
        src1 = 0x04030201
        accs = [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF]
        got = ref_mqsad_u32_u8(src0, src1, accs)
        # SADs are [0, 4, 8, 12]
        self.assertEqual(got, [
            0xFFFF_FFFF,
            3,
            7,
            11,
        ])

        full = ref_mqsad_full_acc(src0, src1, accs)
        self.assertEqual(full, [
            (0xFFFF_FFFF, 0),
            (3, 1),
            (7, 1),
            (11, 1),
        ])

    def test_maximum_possible_byte_sad(self) -> None:
        # Max difference occurs when win has 0xFF for all 4 bytes and mask has 0x01 (or vice versa with mask non-zero)
        # diff per byte = |0xFF - 0x01| = 254.
        # If win has 0x00 and mask has 0xFF, diff per byte = |0x00 - 0xFF| = 255.
        win_0 = 0x00000000
        mask_ff = 0xFFFFFFFF
        self.assertEqual(ref_masked_byte_sad_u32(win_0, mask_ff), 4 * 255)
        self.assertEqual(ref_masked_byte_sad_u32(win_0, mask_ff), MAX_BYTE_SAD)

    def test_upper_byte_of_src0_is_unused(self) -> None:
        # Byte 7 of src0 (bits 63:56) is never in window 0, 1, 2, or 3
        src0_a = 0x0007060504030201
        src0_b = 0xFF07060504030201
        src1 = 0x04030201
        accs = [1, 2, 3, 4]
        self.assertEqual(ref_mqsad_u32_u8(src0_a, src1, accs), ref_mqsad_u32_u8(src0_b, src1, accs))

    def test_invalid_accumulator_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            ref_mqsad_u32_u8(0, 0, [0, 0, 0])
        with self.assertRaises(ValueError):
            ref_mqsad_full_acc(0, 0, [0, 0, 0, 0, 0])


class TestValuMqsadU32U8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v_quad = Operand.vgpr(8, count=4)
        v_pair = Operand.vgpr(2, count=2)
        v_single = Operand.vgpr(1)
        s_pair = Operand.sgpr(4, count=2)
        s_single = Operand.sgpr(10)
        lit64 = Operand.literal(0x0807060504030201, count=2)
        lit32 = Operand.literal(0x04030201)

        self.assertEqual(v_quad.disassemble(), "v[8:11]")
        self.assertEqual(v_pair.disassemble(), "v[2:3]")
        self.assertEqual(v_single.disassemble(), "v1")
        self.assertEqual(s_pair.disassemble(), "s[4:5]")
        self.assertEqual(s_single.disassemble(), "s10")
        self.assertEqual(lit64.disassemble(), "0x0807060504030201")
        self.assertEqual(lit32.disassemble(), "0x04030201")

    def test_operand_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr(254, count=4)  # 254+4=258 > 256
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr(105, count=2)  # 105+2=107 > 106
        with self.assertRaises(ValueError):
            Operand.vgpr(0, count=0)

    def test_instruction_construction_and_disassembly(self) -> None:
        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )
        self.assertEqual(instr.disassemble(), "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7]")

        # With SGPR pair for src0 and literal for src1
        instr2 = VMqsadU32U8Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.sgpr(2, count=2),
            src1=Operand.literal(0x04030201),
            src2=Operand.vgpr(4, count=4),
        )
        self.assertEqual(instr2.disassemble(), "v_mqsad_u32_u8 v[0:3], s[2:3], 0x04030201, v[4:7]")

    def test_instruction_validation_rejections(self) -> None:
        # dst cannot be SGPR
        with self.assertRaises(TypeError):
            VMqsadU32U8Instruction(
                dst=Operand.sgpr(0, count=4),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(4, count=4),
            )
        # dst cannot have count != 4
        with self.assertRaises(TypeError):
            VMqsadU32U8Instruction(
                dst=Operand.vgpr(0, count=2),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(4, count=4),
            )
        # src0 must have count 2
        with self.assertRaises(ValueError):
            VMqsadU32U8Instruction(
                dst=Operand.vgpr(0, count=4),
                src0=Operand.vgpr(2, count=1),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(4, count=4),
            )
        # src1 must have count 1
        with self.assertRaises(ValueError):
            VMqsadU32U8Instruction(
                dst=Operand.vgpr(0, count=4),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(1, count=2),
                src2=Operand.vgpr(4, count=4),
            )
        # src2 must be VGPR quad
        with self.assertRaises(TypeError):
            VMqsadU32U8Instruction(
                dst=Operand.vgpr(0, count=4),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(1),
                src2=Operand.sgpr(4, count=4),
            )
        with self.assertRaises(TypeError):
            VMqsadU32U8Instruction(
                dst=Operand.vgpr(0, count=4),
                src0=Operand.vgpr(2, count=2),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(4, count=2),
            )


class TestValuMqsadU32U8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.regs = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.regs)

    def test_scalar_register_file_read_write(self) -> None:
        # Single 32-bit dword
        self.regs.write_operand(Operand.vgpr(5), 0x12345678)
        self.assertEqual(self.regs.read_operand(Operand.vgpr(5)), 0x12345678)

        # 64-bit pair
        self.regs.write_operand(Operand.vgpr(2, count=2), [0x04030201, 0x08070605])
        self.assertEqual(self.regs.read_operand(Operand.vgpr(2, count=2)), 0x0807060504030201)

        # 128-bit quad
        self.regs.write_operand(Operand.vgpr(4, count=4), [10, 20, 30, 40])
        self.assertEqual(self.regs.read_operand(Operand.vgpr(4, count=4)), [10, 20, 30, 40])

        # SGPR 64-bit pair
        self.regs.write_operand(Operand.sgpr(0, count=2), [0xAAAAAAAA, 0x55555555])
        self.assertEqual(self.regs.read_operand(Operand.sgpr(0, count=2)), 0x55555555AAAAAAAA)

    def test_scalar_execution_all_vgprs(self) -> None:
        # Load src0 (v[2:3]) = 0x0807060504030201
        self.regs.write_operand(Operand.vgpr(2, count=2), [0x04030201, 0x08070605])
        # Load src1 (v1) = 0x04030201
        self.regs.write_operand(Operand.vgpr(1), 0x04030201)
        # Load src2 (v[4:7]) = [0, 0, 0, 0]
        self.regs.write_operand(Operand.vgpr(4, count=4), [0, 0, 0, 0])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        res = self.interp.step(instr)
        self.assertEqual(res, [0, 4, 8, 12])
        self.assertEqual(self.regs.read_operand(Operand.vgpr(8, count=4)), [0, 4, 8, 12])

    def test_scalar_execution_sgpr_and_literal(self) -> None:
        # src0 from s[2:3], src1 as literal constant
        self.regs.write_operand(Operand.sgpr(2, count=2), [0x04030201, 0x08070605])
        self.regs.write_operand(Operand.vgpr(4, count=4), [100, 200, 300, 400])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.sgpr(2, count=2),
            src1=Operand.literal(0x04030201),
            src2=Operand.vgpr(4, count=4),
        )

        res = self.interp.step(instr)
        self.assertEqual(res, [100, 204, 308, 412])
        self.assertEqual(self.regs.read_operand(Operand.vgpr(8, count=4)), [100, 204, 308, 412])

    def test_scalar_sequential_program(self) -> None:
        # Step 1: compute initial SADs into v[8:11]
        self.regs.write_operand(Operand.vgpr(2, count=2), [0x04030201, 0x08070605])
        self.regs.write_operand(Operand.vgpr(1), 0x04030201)
        self.regs.write_operand(Operand.vgpr(4, count=4), [0, 0, 0, 0])

        instr1 = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )
        self.interp.step(instr1)
        self.assertEqual(self.regs.read_operand(Operand.vgpr(8, count=4)), [0, 4, 8, 12])

        # Step 2: feed v[8:11] as src2 accumulator for next step
        instr2 = VMqsadU32U8Instruction(
            dst=Operand.vgpr(12, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(8, count=4),
        )
        self.interp.step(instr2)
        # Should double the SADs: [0, 8, 16, 24]
        self.assertEqual(self.regs.read_operand(Operand.vgpr(12, count=4)), [0, 8, 16, 24])


class TestValuMqsadU32U8SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        # Each lane has distinct accumulator offset
        for lane in range(32):
            state.write_lane(Operand.vgpr(2, count=2), lane, [0x04030201, 0x08070605])
            state.write_lane(Operand.vgpr(1), lane, 0x04030201)
            state.write_lane(Operand.vgpr(4, count=4), lane, [lane, lane * 2, lane * 3, lane * 4])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = [lane + 0, lane * 2 + 4, lane * 3 + 8, lane * 4 + 12]
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.read_lane(Operand.vgpr(8, count=4), lane), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            # Lower 32 lanes use one pattern, upper 32 lanes use another
            src0_hi = 0x08070605 if lane < 32 else 0x18171615
            state.write_lane(Operand.vgpr(2, count=2), lane, [0x04030201, src0_hi])
            state.write_lane(Operand.vgpr(1), lane, 0x04030201)
            state.write_lane(Operand.vgpr(4, count=4), lane, [0, 0, 0, 0])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(32):
            self.assertEqual(results[lane], [0, 4, 8, 12])
        for lane in range(32, 64):
            self.assertIsNotNone(results[lane])
            assert results[lane] is not None
            self.assertEqual(results[lane][0], 0)

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel = [0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF]
        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # only even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr(8, count=4), lane, sentinel)
            state.write_lane(Operand.vgpr(2, count=2), lane, [0x04030201, 0x08070605])
            state.write_lane(Operand.vgpr(1), lane, 0x04030201)
            state.write_lane(Operand.vgpr(4, count=4), lane, [0, 0, 0, 0])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], [0, 4, 8, 12])
                self.assertEqual(state.read_lane(Operand.vgpr(8, count=4), lane), [0, 4, 8, 12])
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr(8, count=4), lane), sentinel)

    def test_exec_mask_zero_no_lanes_modified(self) -> None:
        sentinel = [0xCAFEBABE, 0xCAFEBABE, 0xCAFEBABE, 0xCAFEBABE]
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr(8, count=4), lane, sentinel)

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        results = interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.read_lane(Operand.vgpr(8, count=4), lane), sentinel)

    def test_shared_sgpr_src0_across_divergent_vgpr_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        # Shared SGPR pair for src0: s[2:3] = 0x0807060504030201
        state.write_lane(Operand.sgpr(2, count=2), 0, [0x04030201, 0x08070605])

        for lane in range(32):
            # Divergent src1 per lane: lane 0 has 0x04030201, lane 1 has 0 (all masked out)
            src1_val = 0x04030201 if (lane % 2 == 0) else 0x00000000
            state.write_lane(Operand.vgpr(1), lane, src1_val)
            state.write_lane(Operand.vgpr(4, count=4), lane, [10, 20, 30, 40])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(8, count=4),
            src0=Operand.sgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], [10, 24, 38, 52])
            else:
                self.assertEqual(results[lane], [10, 20, 30, 40])


class TestValuMqsadU32U8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_exact_alias_with_src2_accumulator(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # src0 = 0x0807060504030201, src1 = 0x04030201, src2 = [10, 20, 30, 40]
        regs.write_operand(Operand.vgpr(2, count=2), [0x04030201, 0x08070605])
        regs.write_operand(Operand.vgpr(1), 0x04030201)
        regs.write_operand(Operand.vgpr(4, count=4), [10, 20, 30, 40])

        # dst = v[4:7], src2 = v[4:7] (exact in-place accumulation)
        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(4, count=4),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(4, count=4),
        )

        res = interp.step(instr)
        self.assertEqual(res, [10, 24, 38, 52])
        self.assertEqual(regs.read_operand(Operand.vgpr(4, count=4)), [10, 24, 38, 52])
        # src0 and src1 remain uncorrupted
        self.assertEqual(regs.read_operand(Operand.vgpr(2, count=2)), 0x0807060504030201)
        self.assertEqual(regs.read_operand(Operand.vgpr(1)), 0x04030201)

    def test_dst_partially_overlaps_src2(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst is v[3:6], src2 is v[4:7]. Overlap on v4, v5, v6.
        regs.write_operand(Operand.vgpr(0, count=2), [0x04030201, 0x08070605])
        regs.write_operand(Operand.vgpr(2), 0x04030201)
        regs.write_operand(Operand.vgpr(4, count=4), [100, 200, 300, 400])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(3, count=4),  # writes v3, v4, v5, v6
            src0=Operand.vgpr(0, count=2),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(4, count=4),  # reads v4, v5, v6, v7
        )

        res = interp.step(instr)
        self.assertEqual(res, [100, 204, 308, 412])
        # v3 gets res[0], v4 gets res[1], v5 gets res[2], v6 gets res[3]
        self.assertEqual(regs.read_operand(Operand.vgpr(3)), 100)
        self.assertEqual(regs.read_operand(Operand.vgpr(4)), 204)
        self.assertEqual(regs.read_operand(Operand.vgpr(5)), 308)
        self.assertEqual(regs.read_operand(Operand.vgpr(6)), 412)
        # v7 (not in dst) should still hold original src2[3] = 400
        self.assertEqual(regs.read_operand(Operand.vgpr(7)), 400)

    def test_dst_aliases_src0_pair(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst = v[0:3], src0 = v[0:1]
        regs.write_operand(Operand.vgpr(0, count=2), [0x04030201, 0x08070605])
        regs.write_operand(Operand.vgpr(4), 0x04030201)
        regs.write_operand(Operand.vgpr(5, count=4), [50, 60, 70, 80])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(0, count=2),  # aliased with dst[0:1]
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(5, count=4),
        )

        res = interp.step(instr)
        self.assertEqual(res, [50, 64, 78, 92])
        self.assertEqual(regs.read_operand(Operand.vgpr(0, count=4)), [50, 64, 78, 92])

    def test_dst_aliases_src1_single(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst = v[0:3], src1 = v0
        regs.write_operand(Operand.vgpr(4, count=2), [0x04030201, 0x08070605])
        regs.write_operand(Operand.vgpr(0), 0x04030201)
        regs.write_operand(Operand.vgpr(6, count=4), [1, 2, 3, 4])

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(4, count=2),
            src1=Operand.vgpr(0),  # aliased with dst[0]
            src2=Operand.vgpr(6, count=4),
        )

        res = interp.step(instr)
        self.assertEqual(res, [1, 6, 11, 16])
        self.assertEqual(regs.read_operand(Operand.vgpr(0, count=4)), [1, 6, 11, 16])

    def test_dst_aliases_src0_and_src2_simultaneously(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst = v[0:3], src0 = v[0:1], src2 = v[0:3]
        regs.write_operand(Operand.vgpr(0, count=4), [0x04030201, 0x08070605, 100, 200])
        regs.write_operand(Operand.vgpr(4), 0x04030201)

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(0, count=4),
            src0=Operand.vgpr(0, count=2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(0, count=4),
        )

        res = interp.step(instr)
        expected = [
            (0x04030201 + 0) & MASK32,
            (0x08070605 + 4) & MASK32,
            100 + 8,
            200 + 12,
        ]
        self.assertEqual(res, expected)
        self.assertEqual(regs.read_operand(Operand.vgpr(0, count=4)), expected)

    def test_simd_aliased_in_place_preservation_with_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active
        interp = SIMDInterpreter()

        initial_acc = [1000, 2000, 3000, 4000]
        for lane in range(32):
            state.write_lane(Operand.vgpr(0, count=2), lane, [0x04030201, 0x08070605])
            state.write_lane(Operand.vgpr(2), lane, 0x04030201)
            state.write_lane(Operand.vgpr(4, count=4), lane, initial_acc)

        instr = VMqsadU32U8Instruction(
            dst=Operand.vgpr(4, count=4),  # aliased with src2
            src0=Operand.vgpr(0, count=2),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(4, count=4),
        )

        interp.step(instr, state)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(
                    state.read_lane(Operand.vgpr(4, count=4), lane),
                    [1000, 2004, 3008, 4012],
                )
            else:
                self.assertEqual(
                    state.read_lane(Operand.vgpr(4, count=4), lane),
                    initial_acc,
                )


class TestValuMqsadU32U8StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_dword_carry_exact_boundary(self) -> None:
        # acc = 0xFFFFFFFF, win = 0x05, mask = 0x04 -> diff = 1 -> full_sum = 0x100000000
        # carry = True, zero = True (wrapped to 0)
        status = compute_mqsad_dword_status(win=0x05, mask=0x04, acc=0xFFFF_FFFF)
        self.assertTrue(status.carry)
        self.assertTrue(status.zero)
        self.assertEqual(status.full_sum, 0x1_0000_0000)
        self.assertEqual(status.byte_sad, 1)

        # acc = 0xFFFFFFFF, but sad = 0 -> full_sum = 0xFFFFFFFF -> carry = False, zero = False
        status_no_carry = compute_mqsad_dword_status(win=0x04, mask=0x04, acc=0xFFFF_FFFF)
        self.assertFalse(status_no_carry.carry)
        self.assertFalse(status_no_carry.zero)
        self.assertEqual(status_no_carry.full_sum, 0xFFFF_FFFF)
        self.assertEqual(status_no_carry.byte_sad, 0)

    def test_all_carry_generation(self) -> None:
        # Each window produces SAD >= 1, and all accumulators are 0xFFFFFFFF
        src0 = 0x0807060504030201
        src1 = 0x04030201  # SADs are [0, 4, 8, 12] -> window 0 has SAD 0, so no carry on d0
        flags = evaluate_mqsad_flags(src0, src1, [0xFFFF_FFFF] * 4)
        self.assertTrue(flags.any_carry)
        self.assertFalse(flags.all_carry)
        self.assertFalse(flags.dword_status[0].carry)
        self.assertTrue(flags.dword_status[1].carry)
        self.assertTrue(flags.dword_status[2].carry)
        self.assertTrue(flags.dword_status[3].carry)

        # When all windows produce SAD >= 1
        src0_all_diff = 0xFFFFFFFFFFFFFFFF
        src1_all_diff = 0x01010101
        flags_all = evaluate_mqsad_flags(src0_all_diff, src1_all_diff, [0xFFFF_FFFF] * 4)
        self.assertTrue(flags_all.all_carry)

    def test_all_zero_condition(self) -> None:
        flags = evaluate_mqsad_flags(0, 0, [0, 0, 0, 0])
        self.assertTrue(flags.all_zero)
        self.assertTrue(flags.all_sad_zero)
        self.assertFalse(flags.any_carry)
        self.assertEqual(flags.active_mask_bytes, 0)

    def test_active_mask_byte_counting(self) -> None:
        patterns = [
            (0x00000000, 0),
            (0x000000FF, 1),
            (0x0000FFFF, 2),
            (0x00FFFFFF, 3),
            (0xFFFFFFFF, 4),
            (0xAA005500, 2),
        ]
        for mask, expected_count in patterns:
            with self.subTest(mask=hex(mask)):
                flags = evaluate_mqsad_flags(0, mask, [0, 0, 0, 0])
                self.assertEqual(flags.active_mask_bytes, expected_count)

    def test_max_byte_sad_flag(self) -> None:
        # win has 4 bytes of 0x00, mask has 4 bytes of 0xFF -> SAD = 4 * 255 = 1020
        status = compute_mqsad_dword_status(win=0x00000000, mask=0xFFFFFFFF, acc=0)
        self.assertTrue(status.max_sad)
        self.assertEqual(status.byte_sad, 1020)
        self.assertFalse(status.carry)
        self.assertFalse(status.zero)


if __name__ == "__main__":
    unittest.main()
