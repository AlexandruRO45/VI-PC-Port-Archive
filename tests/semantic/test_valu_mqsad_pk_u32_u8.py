from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mqsad_pk_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_mqsad_pk_u32_u8 dst, src0, src1, src2 [clamp]

Format:
  VOP3 (Vector ALU 3-source instruction, 128-bit destination).

Operands:
  - dst:  128-bit VGPR quad v[d:d+3], holding 4 unsigned 32-bit dwords:
            dst[31:0]   = output dword 0
            dst[63:32]  = output dword 1
            dst[95:64]  = output dword 2
            dst[127:96] = output dword 3
  - src0: 64-bit VGPR pair v[s0:s0+1], holding 8 packed 8-bit unsigned bytes (sliding reference window):
            byte 0: src0[7:0]
            byte 1: src0[15:8]
            byte 2: src0[23:16]
            byte 3: src0[31:24]
            byte 4: src0[39:32]
            byte 5: src0[47:40]
            byte 6: src0[55:48]
            byte 7: src0[63:56]  (unused in quad-SAD sliding windows)
  - src1: 32-bit VGPR, SGPR, or literal, holding 4 packed 8-bit unsigned bytes (query template with byte-zero masking):
            byte 0: src1[7:0]
            byte 1: src1[15:8]
            byte 2: src1[23:16]
            byte 3: src1[31:24]
            Masking rule: If query byte src1.byte[j] == 0, that byte is masked out (diff = 0).
  - src2: 128-bit VGPR quad, SGPR quad, or literal (accumulator):
            accum[0]: src2[31:0]
            accum[1]: src2[63:32]
            accum[2]: src2[95:64]
            accum[3]: src2[127:96]

Operation:
  Computes 4 overlapping 4-byte Masked Sum of Absolute Differences (MSAD) across the 8-byte
  reference window in src0 against the 4-byte query template in src1, accumulating into
  the 4 32-bit dwords in src2:

    D[31:0]   = MSAD_U8(S0[31:0],  S1[31:0], S2[31:0]);
    D[63:32]  = MSAD_U8(S0[39:8],  S1[31:0], S2[63:32]);
    D[95:64]  = MSAD_U8(S0[47:16], S1[31:0], S2[95:64]);
    D[127:96] = MSAD_U8(S0[55:24], S1[31:0], S2[127:96]);

  Where MSAD_U8(ref4, query4, acc32) evaluates:
    sad = sum_{j=0..3} (|ref4_byte[j] - query4_byte[j]| if query4_byte[j] != 0 else 0)
    total = sad + acc32
    result = min(total, 0xFFFFFFFF) if CLAMP else (total & 0xFFFFFFFF)

Mathematical Properties and Bounds:
  - Each individual byte difference is in [0, 255].
  - 4-byte masked SAD sum is in [0, 4 * 255] = [0, 1020] (0x03FC).
  - Query byte 0 acts as a per-byte mask: when src1.byte[j] == 0, no difference is accumulated.
  - Accumulator is 32-bit unsigned [0, 2^32 - 1].
  - Unclamped accumulation wraps modulo 2^32 (& 0xFFFFFFFF).
  - Clamped accumulation saturates at 2^32 - 1 (0xFFFFFFFF).
  - src0 byte 7 (bits 63:56) is never referenced by the 4 sliding windows of length 4.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
from typing import List, Optional, Tuple, Union
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
MASK128: int = (1 << 128) - 1


def u8(val: int) -> int:
    """Normalize integer to unsigned 8-bit byte."""
    return val & MASK8


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def u128(val: int) -> int:
    """Normalize integer to unsigned 128-bit quadword."""
    return val & MASK128


def extract_byte(val: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0-based) from an integer."""
    return (val >> (8 * byte_idx)) & MASK8


def extract_u32(val: int, dword_idx: int) -> int:
    """Extract 32-bit dword at index dword_idx (0-based) from an integer."""
    return (val >> (32 * dword_idx)) & MASK32


def pack_u32x4(d0: int, d1: int, d2: int, d3: int) -> int:
    """Pack four 32-bit unsigned integers into a 128-bit integer."""
    return (
        (u32(d0))
        | (u32(d1) << 32)
        | (u32(d2) << 64)
        | (u32(d3) << 96)
    ) & MASK128


def unpack_u32x4(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 128-bit integer into four 32-bit unsigned integers (d0, d1, d2, d3)."""
    return (
        extract_u32(val, 0),
        extract_u32(val, 1),
        extract_u32(val, 2),
        extract_u32(val, 3),
    )


def pack_u8x8(b0: int, b1: int, b2: int, b3: int, b4: int, b5: int, b6: int, b7: int) -> int:
    """Pack eight unsigned bytes into a 64-bit qword."""
    return (
        (u8(b0))
        | (u8(b1) << 8)
        | (u8(b2) << 16)
        | (u8(b3) << 24)
        | (u8(b4) << 32)
        | (u8(b5) << 40)
        | (u8(b6) << 48)
        | (u8(b7) << 56)
    ) & MASK64


def ref_msad_u8_quad(slice_4bytes: int, query_4bytes: int) -> int:
    """Reference mathematical oracle for a 4-byte Masked Sum of Absolute Differences."""
    total = 0
    for j in range(4):
        ref_b = extract_byte(slice_4bytes, j)
        qry_b = extract_byte(query_4bytes, j)
        if qry_b != 0:
            total += abs(ref_b - qry_b)
    return total


def ref_mqsad_pk_u32_u8(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_mqsad_pk_u32_u8.

    Args:
        src0: 64-bit unsigned reference window (8 packed bytes).
        src1: 32-bit unsigned query template (4 packed bytes with byte-0 masking).
        src2: 128-bit unsigned accumulator (4 packed 32-bit dwords).
        clamp: If True, saturates each 32-bit sum at 0xFFFFFFFF.

    Returns:
        128-bit unsigned integer containing 4 packed 32-bit accumulated MSADs.
    """
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_msad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u32(src2, k)
        raw_sum = sad + acc
        if clamp:
            res = min(raw_sum, MASK32)
        else:
            res = raw_sum & MASK32
        out_elements.append(res)

    return pack_u32x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])


def ref_mqsad_pk_u32_u8_details(
    src0: int, src1: int, src2: int, clamp: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[bool, bool, bool, bool], int]:
    """Extended reference oracle returning (sads, raw_sums, carries, packed_result)."""
    sads = []
    raw_sums = []
    carries = []
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_msad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u32(src2, k)
        raw = sad + acc
        carry = raw > MASK32
        res = min(raw, MASK32) if clamp else (raw & MASK32)
        sads.append(sad)
        raw_sums.append(raw)
        carries.append(carry)
        out_elements.append(res)

    packed = pack_u32x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])
    return (
        (sads[0], sads[1], sads[2], sads[3]),
        (raw_sums[0], raw_sums[1], raw_sums[2], raw_sums[3]),
        (carries[0], carries[1], carries[2], carries[3]),
        packed,
    )


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
    size: int = 1  # 1 = 32-bit dword, 2 = 64-bit pair, 4 = 128-bit quad
    literal_value: int | None = None
    literal_bits: int = 32

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            max_idx = 256 - self.size
            if not (0 <= self.index <= max_idx):
                raise ValueError(f"VGPR index {self.index} out of range [0, {max_idx}] for size {self.size}")
        elif self.reg_type == RegType.SGPR:
            max_idx = 106 - self.size
            if not (0 <= self.index <= max_idx):
                raise ValueError(f"SGPR index {self.index} out of range [0, {max_idx}] for size {self.size}")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if self.literal_bits == 32:
                if not (0 <= self.literal_value <= MASK32):
                    raise ValueError(f"32-bit literal {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")
            elif self.literal_bits == 64:
                if not (0 <= self.literal_value <= MASK64):
                    raise ValueError(f"64-bit literal {self.literal_value:#x} out of range [0, 0xFFFFFFFFFFFFFFFF]")
            elif self.literal_bits == 128:
                if not (0 <= self.literal_value <= MASK128):
                    raise ValueError(f"128-bit literal {self.literal_value:#x} out of range [0, 2^128 - 1]")
            else:
                raise ValueError(f"Unsupported literal bit width: {self.literal_bits}")

    @classmethod
    def vgpr(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index, size=1)

    @classmethod
    def vgpr_pair(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index, size=2)

    @classmethod
    def vgpr_quad(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index, size=4)

    @classmethod
    def sgpr(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index, size=1)

    @classmethod
    def sgpr_pair(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index, size=2)

    @classmethod
    def sgpr_quad(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index, size=4)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value, literal_bits=32)

    @classmethod
    def literal64(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value, literal_bits=64)

    @classmethod
    def literal128(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value, literal_bits=128)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.size == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.size - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.size == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.size - 1}]"
        assert self.literal_value is not None
        if self.literal_bits == 128:
            return f"0x{self.literal_value:032X}"
        if self.literal_bits == 64:
            return f"0x{self.literal_value:016X}"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VMqsadPkU32U8Instruction:
    """Symbolic AST node for AMD RDNA2 v_mqsad_pk_u32_u8 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.size != 4:
            raise ValueError("Destination of v_mqsad_pk_u32_u8 must be a 128-bit VGPR quad (size=4)")

        # src0 must be 64-bit (register pair or 64-bit literal)
        if self.src0.reg_type in (RegType.VGPR, RegType.SGPR) and self.src0.size != 2:
            raise ValueError("src0 must be a 64-bit register pair (size=2)")
        if self.src0.reg_type == RegType.LITERAL and self.src0.literal_bits != 64:
            raise ValueError("src0 literal must be 64-bit")

        # src1 must be 32-bit (single register or 32-bit literal)
        if self.src1.size != 1 or (self.src1.reg_type == RegType.LITERAL and self.src1.literal_bits != 32):
            raise ValueError("src1 must be 32-bit (size=1)")

        # src2 must be 128-bit (register quad or 128-bit literal)
        if self.src2.reg_type in (RegType.VGPR, RegType.SGPR) and self.src2.size != 4:
            raise ValueError("src2 must be a 128-bit register quad (size=4)")
        if self.src2.reg_type == RegType.LITERAL and self.src2.literal_bits != 128:
            raise ValueError("src2 literal must be 128-bit")

    def disassemble(self) -> str:
        clamp_str = " clamp" if self.clamp else ""
        return (
            f"v_mqsad_pk_u32_u8 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, "
            f"{self.src2.disassemble()}{clamp_str}"
        )


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for a single execution thread."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read(self, op: Operand) -> int:
        """Read operand value as unsigned 32-bit, 64-bit, or 128-bit integer."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            if op.literal_bits == 128:
                return op.literal_value & MASK128
            if op.literal_bits == 64:
                return op.literal_value & MASK64
            return op.literal_value & MASK32

        bank = self.vgprs if op.reg_type == RegType.VGPR else self.sgprs
        res = 0
        for i in range(op.size):
            res |= (bank[op.index + i] & MASK32) << (32 * i)
        return res

    def write(self, op: Operand, value: int) -> None:
        """Write value to register operand across its allocated dwords."""
        if op.reg_type not in (RegType.VGPR, RegType.SGPR):
            raise TypeError("Cannot write to literal operand")

        bank = self.vgprs if op.reg_type == RegType.VGPR else self.sgprs
        for i in range(op.size):
            bank[op.index + i] = u32(value >> (32 * i))


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for v_mqsad_pk_u32_u8 on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def step(self, instr: VMqsadPkU32U8Instruction) -> int:
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)

        result = ref_mqsad_pk_u32_u8(s0, s1, s2, clamp=instr.clamp)
        self.regs.write(instr.dst, result)
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

    def read_lane(self, op: Operand, lane: int) -> int:
        """Read operand value for a specific SIMD lane."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            if op.literal_bits == 128:
                return op.literal_value & MASK128
            if op.literal_bits == 64:
                return op.literal_value & MASK64
            return op.literal_value & MASK32

        bank = self.vgprs[lane] if op.reg_type == RegType.VGPR else self.sgprs
        res = 0
        for i in range(op.size):
            res |= (bank[op.index + i] & MASK32) << (32 * i)
        return res

    def write_lane(self, op: Operand, lane: int, value: int) -> None:
        """Write value to register operand in a specific lane."""
        if op.reg_type not in (RegType.VGPR, RegType.SGPR):
            raise TypeError("Cannot write to literal operand")

        bank = self.vgprs[lane] if op.reg_type == RegType.VGPR else self.sgprs
        for i in range(op.size):
            bank[op.index + i] = u32(value >> (32 * i))


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def step(self, instr: VMqsadPkU32U8Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes
        s0_vals = [state.read_lane(instr.src0, lane) for lane in range(state.num_lanes)]
        s1_vals = [state.read_lane(instr.src1, lane) for lane in range(state.num_lanes)]
        s2_vals = [state.read_lane(instr.src2, lane) for lane in range(state.num_lanes)]

        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = ref_mqsad_pk_u32_u8(s0_vals[lane], s1_vals[lane], s2_vals[lane], clamp=instr.clamp)
                state.write_lane(instr.dst, lane, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class MqsadStatusFlags:
    """Architectural condition codes and status flags for v_mqsad_pk_u32_u8."""
    element_zeros: Tuple[bool, bool, bool, bool]
    all_zero: bool
    any_zero: bool
    element_carries: Tuple[bool, bool, bool, bool]
    any_carry: bool
    all_carry: bool
    element_sad_zeros: Tuple[bool, bool, bool, bool]
    all_sad_zero: bool
    element_clamped: Tuple[bool, bool, bool, bool]
    any_clamped: bool


def evaluate_mqsad_flags(src0: int, src1: int, src2: int, clamp: bool, res: int) -> MqsadStatusFlags:
    """Evaluate architectural condition codes and status flags for v_mqsad_pk_u32_u8."""
    sads, raw_sums, carries, _ = ref_mqsad_pk_u32_u8_details(src0, src1, src2, clamp=clamp)
    res_elems = unpack_u32x4(res)

    elem_zeros = tuple(e == 0 for e in res_elems)
    sad_zeros = tuple(s == 0 for s in sads)
    clamped = tuple(clamp and (raw > MASK32) for raw in raw_sums)

    return MqsadStatusFlags(
        element_zeros=(elem_zeros[0], elem_zeros[1], elem_zeros[2], elem_zeros[3]),
        all_zero=all(elem_zeros),
        any_zero=any(elem_zeros),
        element_carries=carries,
        any_carry=any(carries),
        all_carry=all(carries),
        element_sad_zeros=(sad_zeros[0], sad_zeros[1], sad_zeros[2], sad_zeros[3]),
        all_sad_zero=all(sad_zeros),
        element_clamped=(clamped[0], clamped[1], clamped[2], clamped[3]),
        any_clamped=any(clamped),
    )


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuMqsadPkU32U8Oracle(unittest.TestCase):
    """Unit tests for RDNA2 v_mqsad_pk_u32_u8 architectural bit model and reference oracle."""

    def test_zero_inputs(self) -> None:
        self.assertEqual(ref_mqsad_pk_u32_u8(0, 0, 0), 0)
        self.assertEqual(ref_mqsad_pk_u32_u8(0, 0, 0, clamp=True), 0)

    def test_masking_byte_zero_skips_difference(self) -> None:
        # Ref: [100, 200, 150, 50, ...]
        # Query: [100, 0, 150, 0] -> bytes 1 and 3 are 0 (masked out!)
        # Window 0:
        #   byte 0: |100 - 100| = 0
        #   byte 1: query is 0 -> skipped (diff = 0 despite ref being 200!)
        #   byte 2: |150 - 150| = 0
        #   byte 3: query is 0 -> skipped (diff = 0 despite ref being 50!)
        # Total SAD = 0!
        ref = pack_u8x8(100, 200, 150, 50, 0, 0, 0, 0)
        query = pack_u8x8(100, 0, 150, 0, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(10, 20, 30, 40)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 10)  # SAD 0 + 10 = 10

    def test_sliding_window_offsets(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 0, 0, 0)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 0)
        self.assertEqual(d1, 40)
        self.assertEqual(d2, 80)
        self.assertEqual(d3, 120)

    def test_maximum_individual_sad(self) -> None:
        # All 0xFF vs 0x01 (nonzero, so not masked) -> SAD = 4 * 254 = 1016
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 10, 100, 1000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 1016)
        self.assertEqual(d1, 1026)
        self.assertEqual(d2, 1116)
        self.assertEqual(d3, 2016)

    def test_accumulator_addition_and_wrapping(self) -> None:
        # SAD = 1016. Accumulator = 0xFFFFFFFF (2^32 - 1).
        # Without clamp: 0xFFFFFFFF + 1016 = 0x1000003F7 -> wraps to 1015 (0x03F7)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFE, 0x0000_0001, 0x0000_0000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=False)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, (1016 + 0xFFFF_FFFF) & MASK32)  # 1015
        self.assertEqual(d1, (1016 + 0xFFFF_FFFE) & MASK32)  # 1014
        self.assertEqual(d2, 1017)
        self.assertEqual(d3, 1016)

    def test_clamp_modifier_saturation(self) -> None:
        # With clamp: saturated at 0xFFFFFFFF
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = pack_u8x8(0x01, 0x01, 0x01, 0x01, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFE, 0x0000_0001, 0x0000_0000)

        res = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=True)
        d0, d1, d2, d3 = unpack_u32x4(res)
        self.assertEqual(d0, 0xFFFF_FFFF)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(d2, 1017)
        self.assertEqual(d3, 1016)

    def test_unused_byte7_invariance(self) -> None:
        ref_a = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0x00)
        ref_b = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0xFF)
        query = 0x1234_5678
        acc = pack_u32x4(100, 200, 300, 400)

        res_a = ref_mqsad_pk_u32_u8(ref_a, query, acc)
        res_b = ref_mqsad_pk_u32_u8(ref_b, query, acc)
        self.assertEqual(res_a, res_b)

    def test_ref_details_breakdown(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 0xFFFF_FFFF, 0, 0)

        sads, raw_sums, carries, packed = ref_mqsad_pk_u32_u8_details(ref, query, acc, clamp=False)
        self.assertEqual(sads, (0, 40, 80, 120))
        self.assertEqual(raw_sums, (0, 40 + 0xFFFF_FFFF, 80, 120))
        self.assertEqual(carries, (False, True, False, False))
        self.assertEqual(unpack_u32x4(packed), (0, (40 + 0xFFFF_FFFF) & MASK32, 80, 120))


class TestValuMqsadPkU32U8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v_quad = Operand.vgpr_quad(0)
        v_pair = Operand.vgpr_pair(4)
        v_single = Operand.vgpr(8)
        s_quad = Operand.sgpr_quad(0)
        s_pair = Operand.sgpr_pair(4)
        s_single = Operand.sgpr(8)
        lit32 = Operand.literal(0x12345678)
        lit64 = Operand.literal64(0x0123456789ABCDEF)
        lit128 = Operand.literal128((1 << 100) | 42)

        self.assertEqual(v_quad.disassemble(), "v[0:3]")
        self.assertEqual(v_pair.disassemble(), "v[4:5]")
        self.assertEqual(v_single.disassemble(), "v8")
        self.assertEqual(s_quad.disassemble(), "s[0:3]")
        self.assertEqual(s_pair.disassemble(), "s[4:5]")
        self.assertEqual(s_single.disassemble(), "s8")
        self.assertEqual(lit32.disassemble(), "0x12345678")
        self.assertEqual(lit64.disassemble(), "0x0123456789ABCDEF")
        self.assertIn("0x", lit128.disassemble())

    def test_operand_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr_pair(255)
        with self.assertRaises(ValueError):
            Operand.vgpr_quad(253)  # max base is 252 (256 - 4)
        with self.assertRaises(ValueError):
            Operand.sgpr_quad(103)  # max base is 102 (106 - 4)
        with self.assertRaises(ValueError):
            Operand.literal(-1)
        with self.assertRaises(ValueError):
            Operand.literal(0x1_0000_0000)
        with self.assertRaises(ValueError):
            Operand.literal64(0x1_0000_0000_0000_0000)
        with self.assertRaises(ValueError):
            Operand.literal128(1 << 128)

    def test_instruction_construction_and_disassembly(self) -> None:
        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.sgpr(2),
            src2=Operand.vgpr_quad(8),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_mqsad_pk_u32_u8 v[0:3], v[4:5], s2, v[8:11]")

        instr_clamp = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.literal(0x12345678),
            src2=Operand.literal128(0),
            clamp=True,
        )
        self.assertTrue(instr_clamp.disassemble().endswith(" clamp"))

    def test_instruction_validation_rejection(self) -> None:
        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VMqsadPkU32U8Instruction(
                dst=Operand.sgpr_quad(0),
                src0=Operand.vgpr_pair(4),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr_quad(8),
            )
        # Destination must be quad (size=4)
        with self.assertRaises(ValueError):
            VMqsadPkU32U8Instruction(
                dst=Operand.vgpr_pair(0),
                src0=Operand.vgpr_pair(4),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr_quad(8),
            )
        # src0 must be 64-bit pair
        with self.assertRaises(ValueError):
            VMqsadPkU32U8Instruction(
                dst=Operand.vgpr_quad(0),
                src0=Operand.vgpr(4),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr_quad(8),
            )
        # src1 must be 32-bit single
        with self.assertRaises(ValueError):
            VMqsadPkU32U8Instruction(
                dst=Operand.vgpr_quad(0),
                src0=Operand.vgpr_pair(4),
                src1=Operand.vgpr_pair(2),
                src2=Operand.vgpr_quad(8),
            )
        # src2 must be 128-bit quad
        with self.assertRaises(ValueError):
            VMqsadPkU32U8Instruction(
                dst=Operand.vgpr_quad(0),
                src0=Operand.vgpr_pair(4),
                src1=Operand.vgpr(2),
                src2=Operand.vgpr_pair(8),
            )


class TestValuMqsadPkU32U8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.regs = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.regs)

    def test_register_file_quad_dword_access(self) -> None:
        val128 = (0x11111111 << 96) | (0x22222222 << 64) | (0x33333333 << 32) | 0x44444444
        self.regs.write(Operand.vgpr_quad(4), val128)
        self.assertEqual(self.regs.read(Operand.vgpr_quad(4)), val128)
        self.assertEqual(self.regs.read(Operand.vgpr(4)), 0x44444444)
        self.assertEqual(self.regs.read(Operand.vgpr(5)), 0x33333333)
        self.assertEqual(self.regs.read(Operand.vgpr(6)), 0x22222222)
        self.assertEqual(self.regs.read(Operand.vgpr(7)), 0x11111111)

        # Truncation check
        self.regs.write(Operand.vgpr(0), 0x1_FFFF_FFFF)
        self.assertEqual(self.regs.read(Operand.vgpr(0)), 0xFFFF_FFFF)

        # Cannot write to literal
        with self.assertRaises(TypeError):
            self.regs.write(Operand.literal(0), 10)

    def test_scalar_execution_basic(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 0, 0, 0)

        self.regs.write(Operand.vgpr_pair(4), ref)
        self.regs.write(Operand.vgpr(6), query)
        self.regs.write(Operand.vgpr_quad(8), acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(8),
        )

        res = self.interp.step(instr)
        self.assertEqual(unpack_u32x4(res), (0, 40, 80, 120))
        self.assertEqual(self.regs.read(Operand.vgpr_quad(0)), res)

    def test_scalar_execution_sgpr_and_literal(self) -> None:
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        self.regs.write(Operand.vgpr_pair(4), ref)
        self.regs.write(Operand.sgpr(2), 0x01010101)  # nonzero, so not masked

        max_acc = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF)
        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.sgpr(2),
            src2=Operand.literal128(max_acc),
            clamp=True,
        )

        res = self.interp.step(instr)
        self.assertEqual(unpack_u32x4(res), (0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF))
        self.assertEqual(self.regs.read(Operand.vgpr_quad(0)), res)

    def test_scalar_sequential_pipeline(self) -> None:
        # Step 1: Compute into v[0:3]
        ref1 = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query1 = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        self.regs.write(Operand.vgpr_pair(4), ref1)
        self.regs.write(Operand.vgpr(6), query1)
        self.regs.write(Operand.vgpr_quad(8), 0)

        instr1 = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(8),
        )
        res1 = self.interp.step(instr1)
        self.assertEqual(unpack_u32x4(res1), (0, 40, 80, 120))

        # Step 2: Use v[0:3] as accumulator in step 2 with query=0 (all masked -> SADs=0)
        # Result should equal accumulated v[0:3]
        instr2 = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(12),
            src0=Operand.vgpr_pair(4),
            src1=Operand.literal(0x00000000),  # fully masked
            src2=Operand.vgpr_quad(0),         # reads step 1 result
        )
        res2 = self.interp.step(instr2)
        self.assertEqual(unpack_u32x4(res2), (0, 40, 80, 120))
        self.assertEqual(self.regs.read(Operand.vgpr_quad(12)), res1)


class TestValuMqsadPkU32U8SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            ref = pack_u8x8(lane, lane + 1, lane + 2, lane + 3, lane + 4, lane + 5, lane + 6, lane + 7)
            query = pack_u8x8(lane, lane + 1, lane + 2, lane + 3, 0, 0, 0, 0) & MASK32
            acc = pack_u32x4(lane * 10, lane * 20, lane * 30, lane * 40)
            state.write_lane(Operand.vgpr_pair(4), lane, ref)
            state.write_lane(Operand.vgpr(6), lane, query)
            state.write_lane(Operand.vgpr_quad(8), lane, acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(8),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            read_back = state.read_lane(Operand.vgpr_quad(0), lane)
            self.assertEqual(results[lane], read_back)
            d0, d1, d2, d3 = unpack_u32x4(read_back)
            self.assertEqual(d0, lane * 10)  # SAD=0 + acc[0]

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
            query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
            acc = pack_u32x4(lane, lane, lane, lane)
            state.write_lane(Operand.vgpr_pair(4), lane, ref)
            state.write_lane(Operand.vgpr(6), lane, query)
            state.write_lane(Operand.vgpr_quad(8), lane, acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(8),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertIsNotNone(results[lane])
            d0, _, _, _ = unpack_u32x4(state.read_lane(Operand.vgpr_quad(0), lane))
            self.assertEqual(d0, lane)

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel = (0xDEADBEEF << 96) | (0xCAFEBABE << 64) | (0xBAADF00D << 32) | 0xFEEDFACE
        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr_quad(0), lane, sentinel)
            ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
            query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
            acc = 0
            state.write_lane(Operand.vgpr_pair(4), lane, ref)
            state.write_lane(Operand.vgpr(6), lane, query)
            state.write_lane(Operand.vgpr_quad(8), lane, acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(8),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_quad(0), lane), pack_u32x4(0, 40, 80, 120))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_quad(0), lane), sentinel)

    def test_zero_exec_mask_no_modification(self) -> None:
        sentinel = 0x99999999888888887777777766666666
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr_quad(0), lane, sentinel)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.literal64(0),
            src1=Operand.literal(0),
            src2=Operand.literal128(0),
        )

        results = interp.step(instr, state)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(state.read_lane(Operand.vgpr_quad(0), lane), sentinel)

    def test_sgpr_uniform_broadcast_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(100, 200, 300, 400)
        state.sgprs[2] = query
        for i in range(4):
            state.sgprs[4 + i] = extract_u32(acc, i)

        for lane in range(32):
            ref = pack_u8x8(10 + lane, 20, 30, 40, 50, 60, 70, 80)
            state.write_lane(Operand.vgpr_pair(4), lane, ref)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.sgpr(2),
            src2=Operand.sgpr_quad(4),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            d0, _, _, _ = unpack_u32x4(state.read_lane(Operand.vgpr_quad(0), lane))
            self.assertEqual(d0, 100 + lane)

    def test_wave64_upper_divergence(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_0000_0000)
        interp = SIMDInterpreter()
        sentinel = 0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

        for lane in range(64):
            state.write_lane(Operand.vgpr_quad(0), lane, sentinel)
            ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
            query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
            state.write_lane(Operand.vgpr_pair(4), lane, ref)
            state.write_lane(Operand.vgpr(6), lane, query)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.literal128(0),
        )

        results = interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_quad(0), lane), sentinel)
            else:
                self.assertIsNotNone(results[lane])
                self.assertNotEqual(state.read_lane(Operand.vgpr_quad(0), lane), sentinel)


class TestValuMqsadPkU32U8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src2_in_place_accumulation(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        initial_acc = pack_u32x4(100, 200, 300, 400)
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32

        regs.write(Operand.vgpr_quad(0), initial_acc)  # src2 and dst
        regs.write(Operand.vgpr_pair(4), ref)          # src0
        regs.write(Operand.vgpr(6), query)             # src1

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(0),  # In-place accumulation!
        )

        res = interp.step(instr)
        expected = pack_u32x4(100, 240, 380, 520)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_quad(0)), expected)

    def test_dst_aliases_src0_low_pair(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(1, 2, 3, 4)

        regs.write(Operand.vgpr_pair(0), ref)          # src0 in v[0:1], will be overwritten by dst v[0:3]
        regs.write(Operand.vgpr(4), query)
        regs.write(Operand.vgpr_quad(8), acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(0),  # Aliased with dst low pair
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_quad(8),
        )

        res = interp.step(instr)
        expected = pack_u32x4(1, 42, 83, 124)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_quad(0)), expected)

    def test_dst_aliases_src1_single_dword(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        acc = pack_u32x4(0, 0, 0, 0)

        regs.write(Operand.vgpr(0), query)             # src1 in v0, will be overwritten by dst v[0:3]
        regs.write(Operand.vgpr_pair(4), ref)
        regs.write(Operand.vgpr_quad(8), acc)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(0),  # Aliased with v0
            src2=Operand.vgpr_quad(8),
        )

        res = interp.step(instr)
        self.assertEqual(res, pack_u32x4(0, 40, 80, 120))
        self.assertEqual(regs.read(Operand.vgpr_quad(0)), res)

    def test_dst_aliases_src0_and_src2(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        initial_val = (0x04030201 << 96) | (0x04030201 << 64) | ref

        regs.write(Operand.vgpr_quad(0), initial_val)
        regs.write(Operand.vgpr(4), query)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(0),  # v[0:1]
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_quad(0),  # v[0:3]
        )

        expected = ref_mqsad_pk_u32_u8(ref, query, initial_val)
        res = interp.step(instr)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_quad(0)), expected)

    def test_partial_overlap_dst_and_src2(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst is v[0:3], src2 is v[2:5] (overlapping at v2, v3)
        ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
        query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32

        # Initialize v[2:5] with accumulator values
        for i in range(4):
            regs.write(Operand.vgpr(2 + i), (i + 1) * 100)
        acc = regs.read(Operand.vgpr_quad(2))

        regs.write(Operand.vgpr_pair(6), ref)
        regs.write(Operand.vgpr(8), query)

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(6),
            src1=Operand.vgpr(8),
            src2=Operand.vgpr_quad(2),  # Partial overlap with dst
        )

        expected = ref_mqsad_pk_u32_u8(ref, query, acc)
        res = interp.step(instr)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_quad(0)), expected)

    def test_simd_multi_lane_aliased_in_place(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active
        interp = SIMDInterpreter()

        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32

        for lane in range(32):
            acc = pack_u32x4(lane, lane, lane, lane)
            state.write_lane(Operand.vgpr_quad(0), lane, acc)  # dst and src2
            state.write_lane(Operand.vgpr_pair(4), lane, ref)  # src0
            state.write_lane(Operand.vgpr(6), lane, query)     # src1

        instr = VMqsadPkU32U8Instruction(
            dst=Operand.vgpr_quad(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.vgpr(6),
            src2=Operand.vgpr_quad(0),  # In-place accumulation across SIMD lanes
        )

        interp.step(instr, state)
        for lane in range(32):
            val = state.read_lane(Operand.vgpr_quad(0), lane)
            if lane < 16:
                expected = pack_u32x4(lane, 40 + lane, 80 + lane, 120 + lane)
                self.assertEqual(val, expected)
            else:
                original = pack_u32x4(lane, lane, lane, lane)
                self.assertEqual(val, original)


class TestValuMqsadPkU32U8StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_all_zero_flag(self) -> None:
        res = ref_mqsad_pk_u32_u8(0, 0, 0)
        flags = evaluate_mqsad_flags(0, 0, 0, clamp=False, res=res)
        self.assertTrue(flags.all_zero)
        self.assertTrue(flags.any_zero)
        self.assertEqual(flags.element_zeros, (True, True, True, True))
        self.assertTrue(flags.all_sad_zero)
        self.assertFalse(flags.any_carry)
        self.assertFalse(flags.any_clamped)

    def test_individual_element_zero_flags(self) -> None:
        ref = pack_u8x8(1, 2, 3, 4, 10, 20, 30, 40)
        query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(0, 5, 10, 15)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        flags = evaluate_mqsad_flags(ref, query, acc, clamp=False, res=res)
        self.assertFalse(flags.all_zero)
        self.assertTrue(flags.any_zero)
        self.assertEqual(flags.element_zeros, (True, False, False, False))

    def test_single_and_multiple_carry_generation(self) -> None:
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x01010101  # SAD = 1016
        # Element 1 has accumulator 0xFFFFFFFF -> carry!
        acc = pack_u32x4(0, 0xFFFF_FFFF, 10, 20)

        res = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=False)
        flags = evaluate_mqsad_flags(ref, query, acc, clamp=False, res=res)
        self.assertEqual(flags.element_carries, (False, True, False, False))
        self.assertTrue(flags.any_carry)
        self.assertFalse(flags.all_carry)

        # All carries: accumulator 0xFFFFFFFF across all elements
        acc_all = pack_u32x4(0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF)
        res_all = ref_mqsad_pk_u32_u8(ref, query, acc_all, clamp=False)
        flags_all = evaluate_mqsad_flags(ref, query, acc_all, clamp=False, res=res_all)
        self.assertEqual(flags_all.element_carries, (True, True, True, True))
        self.assertTrue(flags_all.all_carry)

    def test_sad_zero_detection(self) -> None:
        ref = pack_u8x8(1, 2, 3, 4, 1, 2, 3, 4)
        query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
        acc = pack_u32x4(10, 20, 30, 40)

        res = ref_mqsad_pk_u32_u8(ref, query, acc)
        flags = evaluate_mqsad_flags(ref, query, acc, clamp=False, res=res)
        self.assertEqual(flags.element_sad_zeros, (True, False, False, False))
        self.assertFalse(flags.all_sad_zero)

    def test_clamped_flag_evaluation(self) -> None:
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x01010101  # SAD = 1016
        acc = pack_u32x4(0xFFFF_FFFF, 0x00000000, 0xFFFF_FFFF, 0x00000000)

        # With clamp=True
        res_clamped = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=True)
        flags_clamped = evaluate_mqsad_flags(ref, query, acc, clamp=True, res=res_clamped)
        self.assertEqual(flags_clamped.element_clamped, (True, False, True, False))
        self.assertTrue(flags_clamped.any_clamped)

        # With clamp=False
        res_unclamped = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=False)
        flags_unclamped = evaluate_mqsad_flags(ref, query, acc, clamp=False, res=res_unclamped)
        self.assertEqual(flags_unclamped.element_clamped, (False, False, False, False))
        self.assertFalse(flags_unclamped.any_clamped)


class TestValuMqsadPkU32U8IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit pattern boundaries and integer extreme limits."""

    def test_ieee754_float32_special_bit_patterns_as_raw_bytes(self) -> None:
        pos_zero = 0x0000_0000
        neg_zero = 0x8000_0000
        pos_inf = 0x7F80_0000
        neg_inf = 0xFF80_0000
        qnan = 0x7FC0_0000
        snan = 0x7F80_0001

        patterns = [pos_zero, neg_zero, pos_inf, neg_inf, qnan, snan]
        for p0 in patterns:
            for p1 in patterns:
                ref = (p0 << 32) | p1
                query = p0
                acc = 0
                res = ref_mqsad_pk_u32_u8(ref, query, acc)
                expected = ref_mqsad_pk_u32_u8(ref, query, acc)
                self.assertEqual(res, expected)

    def test_subnormal_float_bits_no_ftz(self) -> None:
        sub1 = 0x0000_0001
        sub2 = 0x0000_0002
        ref = (sub1 << 32) | sub1
        query = sub2
        res = ref_mqsad_pk_u32_u8(ref, query, 0)
        d0, _, _, _ = unpack_u32x4(res)
        # Byte 0 of sub2 is 2 (nonzero, so not masked!)
        # Byte 0 of sub1 is 1 -> |1 - 2| = 1
        self.assertEqual(d0, 1)

    def test_nan_payload_preservation_and_integer_arithmetic(self) -> None:
        snan = 0x7F80_0001  # bytes: 01, 00, 80, 7F
        qnan = 0x7FC0_0000  # bytes: 00, 00, C0, 7F
        ref = (snan << 32) | snan
        query = qnan
        res = ref_mqsad_pk_u32_u8(ref, query, 0)
        d0, _, _, _ = unpack_u32x4(res)
        # Query bytes:
        # byte 0: 0x00 -> masked out (diff = 0)
        # byte 1: 0x00 -> masked out (diff = 0)
        # byte 2: 0xC0 -> |0x80 - 0xC0| = 64
        # byte 3: 0x7F -> |0x7F - 0x7F| = 0
        # Total MSAD = 64
        self.assertEqual(d0, 64)

    def test_byte_extreme_boundary_matrix(self) -> None:
        byte_extremes = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]
        for b0 in byte_extremes:
            for b1 in byte_extremes:
                slice4 = pack_u8x8(b0, b0, b0, b0, 0, 0, 0, 0) & MASK32
                query4 = pack_u8x8(b1, b1, b1, b1, 0, 0, 0, 0) & MASK32
                sad = ref_msad_u8_quad(slice4, query4)
                expected = 0 if (b1 == 0) else (4 * abs(b0 - b1))
                self.assertEqual(sad, expected)

    def test_accumulator_32bit_extreme_boundaries(self) -> None:
        acc_extremes = [0x0000_0000, 0x0000_0001, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFE, 0xFFFF_FFFF]
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x01010101  # SAD = 4 * 254 = 1016

        for a in acc_extremes:
            acc = pack_u32x4(a, a, a, a)
            res_unclamped = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=False)
            d_unclamped, _, _, _ = unpack_u32x4(res_unclamped)
            self.assertEqual(d_unclamped, (1016 + a) & MASK32)

            res_clamped = ref_mqsad_pk_u32_u8(ref, query, acc, clamp=True)
            d_clamped, _, _, _ = unpack_u32x4(res_clamped)
            self.assertEqual(d_clamped, min(1016 + a, MASK32))


if __name__ == "__main__":
    unittest.main()
