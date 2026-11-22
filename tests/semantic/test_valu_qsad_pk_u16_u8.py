from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_qsad_pk_u16_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_qsad_pk_u16_u8 dst, src0, src1, src2 [clamp]

Format:
  VOP3 (Vector ALU 3-source instruction, 64-bit destination).

Operands:
  - dst:  64-bit VGPR pair v[d:d+1], holding 4 packed 16-bit unsigned integers:
            dst[15:0]  = packed halfword 0
            dst[31:16] = packed halfword 1
            dst[47:32] = packed halfword 2
            dst[63:48] = packed halfword 3
  - src0: 64-bit VGPR pair v[s0:s0+1], holding 8 packed 8-bit unsigned bytes (sliding reference window):
            byte 0: src0[7:0]
            byte 1: src0[15:8]
            byte 2: src0[23:16]
            byte 3: src0[31:24]
            byte 4: src0[39:32]
            byte 5: src0[47:40]
            byte 6: src0[55:48]
            byte 7: src0[63:56]  (unused in quad-SAD sliding windows)
  - src1: 32-bit VGPR, SGPR, or literal, holding 4 packed 8-bit unsigned bytes (query template):
            byte 0: src1[7:0]
            byte 1: src1[15:8]
            byte 2: src1[23:16]
            byte 3: src1[31:24]
  - src2: 64-bit VGPR pair, SGPR pair, or literal/constant (accumulator):
            accum[0]: src2[15:0]
            accum[1]: src2[31:16]
            accum[2]: src2[47:32]
            accum[3]: src2[63:48]

Operation:
  Computes 4 overlapping 4-byte Sum of Absolute Differences (SAD) across the 8-byte
  reference window in src0 against the 4-byte query template in src1, accumulating into
  the 4 packed 16-bit integers in src2:

    D[15:0]  = SAD_U8(S0[31:0],  S1[31:0], S2[15:0]);
    D[31:16] = SAD_U8(S0[39:8],  S1[31:0], S2[31:16]);
    D[47:32] = SAD_U8(S0[47:16], S1[31:0], S2[47:32]);
    D[63:48] = SAD_U8(S0[55:24], S1[31:0], S2[63:48]);

  Where SAD_U8(ref4, query4, acc16) evaluates:
    sad = sum_{j=0..3} |ref4_byte[j] - query4_byte[j]|
    total = sad + acc16
    result = min(total, 0xFFFF) if CLAMP else (total & 0xFFFF)

Mathematical Properties and Bounds:
  - Each individual byte difference |ref - query| is in [0, 255].
  - 4-byte SAD sum is in [0, 4 * 255] = [0, 1020] (0x03FC).
  - Accumulator is 16-bit unsigned [0, 65535].
  - Unclamped accumulation wraps modulo 65536 (& 0xFFFF).
  - Clamped accumulation saturates at 65535 (0xFFFF).
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


def extract_byte(val: int, byte_idx: int) -> int:
    """Extract byte at index byte_idx (0-based) from an integer."""
    return (val >> (8 * byte_idx)) & MASK8


def extract_u16(val: int, half_idx: int) -> int:
    """Extract 16-bit halfword at index half_idx (0-based) from an integer."""
    return (val >> (16 * half_idx)) & MASK16


def pack_u16x4(e0: int, e1: int, e2: int, e3: int) -> int:
    """Pack four 16-bit unsigned integers into a 64-bit qword."""
    return (
        (u16(e0))
        | (u16(e1) << 16)
        | (u16(e2) << 32)
        | (u16(e3) << 48)
    ) & MASK64


def unpack_u16x4(val: int) -> Tuple[int, int, int, int]:
    """Unpack a 64-bit integer into four 16-bit unsigned integers (e0, e1, e2, e3)."""
    return (
        extract_u16(val, 0),
        extract_u16(val, 1),
        extract_u16(val, 2),
        extract_u16(val, 3),
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


def unpack_u8x8(val: int) -> Tuple[int, int, int, int, int, int, int, int]:
    """Unpack a 64-bit integer into eight unsigned bytes (b0..b7)."""
    return tuple(extract_byte(val, i) for i in range(8))  # type: ignore[return-value]


def ref_sad_u8_quad(slice_4bytes: int, query_4bytes: int) -> int:
    """Reference mathematical oracle for a 4-byte Sum of Absolute Differences."""
    total = 0
    for j in range(4):
        ref_b = extract_byte(slice_4bytes, j)
        qry_b = extract_byte(query_4bytes, j)
        total += abs(ref_b - qry_b)
    return total


def ref_qsad_pk_u16_u8(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_qsad_pk_u16_u8.

    Args:
        src0: 64-bit unsigned reference window (8 packed bytes).
        src1: 32-bit unsigned query template (4 packed bytes).
        src2: 64-bit unsigned accumulator (4 packed 16-bit halfwords).
        clamp: If True, saturates each 16-bit sum at 0xFFFF.

    Returns:
        64-bit unsigned qword containing 4 packed 16-bit accumulated SADs.
    """
    out_elements = []
    for k in range(4):
        # 4 bytes starting at byte offset k
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_sad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u16(src2, k)
        raw_sum = sad + acc
        if clamp:
            res = min(raw_sum, MASK16)
        else:
            res = raw_sum & MASK16
        out_elements.append(res)

    return pack_u16x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])


def ref_qsad_pk_u16_u8_details(
    src0: int, src1: int, src2: int, clamp: bool = False
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[bool, bool, bool, bool], int]:
    """Extended reference oracle returning (sads, raw_sums, carries, packed_result)."""
    sads = []
    raw_sums = []
    carries = []
    out_elements = []
    for k in range(4):
        ref_slice = (src0 >> (8 * k)) & MASK32
        sad = ref_sad_u8_quad(ref_slice, src1 & MASK32)
        acc = extract_u16(src2, k)
        raw = sad + acc
        carry = raw > MASK16
        res = min(raw, MASK16) if clamp else (raw & MASK16)
        sads.append(sad)
        raw_sums.append(raw)
        carries.append(carry)
        out_elements.append(res)

    packed = pack_u16x4(out_elements[0], out_elements[1], out_elements[2], out_elements[3])
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
    is_pair: bool = False
    literal_value: int | None = None
    literal_bits: int = 32

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            max_idx = 254 if self.is_pair else 255
            if not (0 <= self.index <= max_idx):
                raise ValueError(f"VGPR index {self.index} out of range [0, {max_idx}] (pair={self.is_pair})")
        elif self.reg_type == RegType.SGPR:
            max_idx = 104 if self.is_pair else 105
            if not (0 <= self.index <= max_idx):
                raise ValueError(f"SGPR index {self.index} out of range [0, {max_idx}] (pair={self.is_pair})")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if self.literal_bits == 32:
                if not (0 <= self.literal_value <= MASK32):
                    raise ValueError(f"32-bit literal {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")
            elif self.literal_bits == 64:
                if not (0 <= self.literal_value <= MASK64):
                    raise ValueError(f"64-bit literal {self.literal_value:#x} out of range [0, 0xFFFFFFFFFFFFFFFF]")
            else:
                raise ValueError(f"Unsupported literal bit width: {self.literal_bits}")

    @classmethod
    def vgpr(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index, is_pair=False)

    @classmethod
    def vgpr_pair(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index, is_pair=True)

    @classmethod
    def sgpr(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index, is_pair=False)

    @classmethod
    def sgpr_pair(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index, is_pair=True)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value, literal_bits=32)

    @classmethod
    def literal64(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value, literal_bits=64)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v[{self.index}:{self.index + 1}]" if self.is_pair else f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s[{self.index}:{self.index + 1}]" if self.is_pair else f"s{self.index}"
        assert self.literal_value is not None
        if self.literal_bits == 64:
            return f"0x{self.literal_value:016X}"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VQsadPkU16U8Instruction:
    """Symbolic AST node for AMD RDNA2 v_qsad_pk_u16_u8 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if not self.dst.is_pair:
            raise ValueError("Destination of v_qsad_pk_u16_u8 must be a 64-bit VGPR pair")

        # src0 must be 64-bit (VGPR pair or 64-bit literal)
        if self.src0.reg_type in (RegType.VGPR, RegType.SGPR) and not self.src0.is_pair:
            raise ValueError("src0 must be a 64-bit register pair")
        if self.src0.reg_type == RegType.LITERAL and self.src0.literal_bits != 64:
            raise ValueError("src0 literal must be 64-bit")

        # src1 must be 32-bit
        if self.src1.is_pair or (self.src1.reg_type == RegType.LITERAL and self.src1.literal_bits != 32):
            raise ValueError("src1 must be 32-bit")

        # src2 must be 64-bit
        if self.src2.reg_type in (RegType.VGPR, RegType.SGPR) and not self.src2.is_pair:
            raise ValueError("src2 must be a 64-bit register pair")
        if self.src2.reg_type == RegType.LITERAL and self.src2.literal_bits != 64:
            raise ValueError("src2 literal must be 64-bit")

    def disassemble(self) -> str:
        clamp_str = " clamp" if self.clamp else ""
        return (
            f"v_qsad_pk_u16_u8 {self.dst.disassemble()}, "
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
        """Read operand value as unsigned 32-bit dword or 64-bit qword."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            mask = MASK64 if op.literal_bits == 64 else MASK32
            return op.literal_value & mask
        if op.reg_type == RegType.VGPR:
            if op.is_pair:
                lo = self.vgprs[op.index] & MASK32
                hi = self.vgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            if op.is_pair:
                lo = self.sgprs[op.index] & MASK32
                hi = self.sgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write(self, op: Operand, value: int) -> None:
        """Write value to register operand (single or pair)."""
        if op.reg_type == RegType.VGPR:
            if op.is_pair:
                self.vgprs[op.index] = u32(value)
                self.vgprs[op.index + 1] = u32(value >> 32)
            else:
                self.vgprs[op.index] = u32(value)
        elif op.reg_type == RegType.SGPR:
            if op.is_pair:
                self.sgprs[op.index] = u32(value)
                self.sgprs[op.index + 1] = u32(value >> 32)
            else:
                self.sgprs[op.index] = u32(value)
        else:
            raise TypeError("Cannot write to literal operand")


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for v_qsad_pk_u16_u8 on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def step(self, instr: VQsadPkU16U8Instruction) -> int:
        # Atomic read phase
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)

        # Computation phase
        result = ref_qsad_pk_u16_u8(s0, s1, s2, clamp=instr.clamp)

        # Write-back phase
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
            mask = MASK64 if op.literal_bits == 64 else MASK32
            return op.literal_value & mask
        if op.reg_type == RegType.VGPR:
            if op.is_pair:
                lo = self.vgprs[lane][op.index] & MASK32
                hi = self.vgprs[lane][op.index + 1] & MASK32
                return (hi << 32) | lo
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            if op.is_pair:
                lo = self.sgprs[op.index] & MASK32
                hi = self.sgprs[op.index + 1] & MASK32
                return (hi << 32) | lo
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write_lane(self, op: Operand, lane: int, value: int) -> None:
        """Write value to register operand in a specific lane."""
        if op.reg_type == RegType.VGPR:
            if op.is_pair:
                self.vgprs[lane][op.index] = u32(value)
                self.vgprs[lane][op.index + 1] = u32(value >> 32)
            else:
                self.vgprs[lane][op.index] = u32(value)
        elif op.reg_type == RegType.SGPR:
            if op.is_pair:
                self.sgprs[op.index] = u32(value)
                self.sgprs[op.index + 1] = u32(value >> 32)
            else:
                self.sgprs[op.index] = u32(value)
        else:
            raise TypeError("Cannot write to literal operand")


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def step(self, instr: VQsadPkU16U8Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes
        # Atomic read phase across all lanes
        s0_vals = [state.read_lane(instr.src0, lane) for lane in range(state.num_lanes)]
        s1_vals = [state.read_lane(instr.src1, lane) for lane in range(state.num_lanes)]
        s2_vals = [state.read_lane(instr.src2, lane) for lane in range(state.num_lanes)]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = ref_qsad_pk_u16_u8(s0_vals[lane], s1_vals[lane], s2_vals[lane], clamp=instr.clamp)
                state.write_lane(instr.dst, lane, res)
                results[lane] = res

        return results


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuQsadPkU16U8Oracle(unittest.TestCase):
    """Unit tests for RDNA2 v_qsad_pk_u16_u8 architectural bit model and reference oracle."""

    def test_zero_inputs(self) -> None:
        self.assertEqual(ref_qsad_pk_u16_u8(0, 0, 0), 0)
        self.assertEqual(ref_qsad_pk_u16_u8(0, 0, 0, clamp=True), 0)

    def test_identical_reference_and_query_zero_sad(self) -> None:
        # Repeating pattern 0x04030201 in query and reference
        query = 0x0403_0201
        ref = (query << 32) | query  # 8 bytes: [1,2,3,4, 1,2,3,4]
        # At window offset 0: bytes 1,2,3,4 vs 1,2,3,4 -> SAD = 0
        # At window offset 4: bytes 1,2,3,4 vs 1,2,3,4 -> SAD = 0
        # Accumulator values
        acc = pack_u16x4(10, 20, 30, 40)
        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 10)  # SAD 0 + 10 = 10

    def test_sliding_window_offsets(self) -> None:
        # Construct reference: [10, 20, 30, 40, 50, 60, 70, 80]
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        # Query: [10, 20, 30, 40]
        # Window 0: [10,20,30,40] vs [10,20,30,40] -> SAD = 0
        # Window 1: [20,30,40,50] vs [10,20,30,40] -> diffs (10+10+10+10) = 40
        # Window 2: [30,40,50,60] vs [10,20,30,40] -> diffs (20+20+20+20) = 80
        # Window 3: [40,50,60,70] vs [10,20,30,40] -> diffs (30+30+30+30) = 120
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(0, 0, 0, 0)

        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 0)
        self.assertEqual(e1, 40)
        self.assertEqual(e2, 80)
        self.assertEqual(e3, 120)

    def test_maximum_individual_sad(self) -> None:
        # All 0xFF vs 0x00 -> SAD = 4 * 255 = 1020 (0x03FC)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0, 10, 100, 1000)

        res = ref_qsad_pk_u16_u8(ref, query, acc)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 1020)
        self.assertEqual(e1, 1030)
        self.assertEqual(e2, 1120)
        self.assertEqual(e3, 2020)

    def test_accumulator_addition_and_wrapping(self) -> None:
        # SAD = 1020. Accumulator = 0xFFFF (65535).
        # Without clamp: 65535 + 1020 = 66555 = 0x103FB -> wraps to 0x03FB (1019)
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0xFFFF, 0xFFFE, 0x0001, 0x0000)

        res = ref_qsad_pk_u16_u8(ref, query, acc, clamp=False)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, (1020 + 0xFFFF) & 0xFFFF)  # 1019
        self.assertEqual(e1, (1020 + 0xFFFE) & 0xFFFF)  # 1018
        self.assertEqual(e2, 1021)
        self.assertEqual(e3, 1020)

    def test_clamp_modifier_saturation(self) -> None:
        # With clamp: saturated at 0xFFFF
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        query = 0x0000_0000
        acc = pack_u16x4(0xFFFF, 0xFFFE, 0x0001, 0x0000)

        res = ref_qsad_pk_u16_u8(ref, query, acc, clamp=True)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 0xFFFF)
        self.assertEqual(e1, 0xFFFF)
        self.assertEqual(e2, 1021)
        self.assertEqual(e3, 1020)

    def test_unused_byte7_invariance(self) -> None:
        # Modifying byte 7 of src0 must have zero effect on any output element
        ref_a = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0x00)
        ref_b = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 0xFF)
        query = 0x1234_5678
        acc = pack_u16x4(100, 200, 300, 400)

        res_a = ref_qsad_pk_u16_u8(ref_a, query, acc)
        res_b = ref_qsad_pk_u16_u8(ref_b, query, acc)
        self.assertEqual(res_a, res_b)

    def test_ref_details_breakdown(self) -> None:
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(0, 0xFFFF, 0, 0)

        sads, raw_sums, carries, packed = ref_qsad_pk_u16_u8_details(ref, query, acc, clamp=False)
        self.assertEqual(sads, (0, 40, 80, 120))
        self.assertEqual(raw_sums, (0, 40 + 0xFFFF, 80, 120))
        self.assertEqual(carries, (False, True, False, False))
        self.assertEqual(unpack_u16x4(packed), (0, (40 + 0xFFFF) & 0xFFFF, 80, 120))


class TestValuQsadPkU16U8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v_pair = Operand.vgpr_pair(0)
        v_single = Operand.vgpr(5)
        s_pair = Operand.sgpr_pair(4)
        s_single = Operand.sgpr(10)
        lit32 = Operand.literal(0x12345678)
        lit64 = Operand.literal64(0x0123456789ABCDEF)

        self.assertEqual(v_pair.disassemble(), "v[0:1]")
        self.assertEqual(v_single.disassemble(), "v5")
        self.assertEqual(s_pair.disassemble(), "s[4:5]")
        self.assertEqual(s_single.disassemble(), "s10")
        self.assertEqual(lit32.disassemble(), "0x12345678")
        self.assertEqual(lit64.disassemble(), "0x0123456789ABCDEF")

    def test_operand_bounds_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.vgpr_pair(255)  # 255 cannot form pair [255:256]
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)
        with self.assertRaises(ValueError):
            Operand.sgpr_pair(105)  # 105 cannot form pair [105:106]
        with self.assertRaises(ValueError):
            Operand.literal(-1)
        with self.assertRaises(ValueError):
            Operand.literal(0x1_0000_0000)
        with self.assertRaises(ValueError):
            Operand.literal64(-1)
        with self.assertRaises(ValueError):
            Operand.literal64(0x1_0000_0000_0000_0000)

    def test_instruction_construction_and_disassembly(self) -> None:
        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.sgpr(4),
            src2=Operand.vgpr_pair(6),
            clamp=False,
        )
        self.assertEqual(instr.disassemble(), "v_qsad_pk_u16_u8 v[0:1], v[2:3], s4, v[6:7]")

        instr_clamped = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.literal(0x12345678),
            src2=Operand.literal64(0x0000000100000002),
            clamp=True,
        )
        self.assertEqual(
            instr_clamped.disassemble(),
            "v_qsad_pk_u16_u8 v[0:1], v[2:3], 0x12345678, 0x0000000100000002 clamp",
        )

    def test_instruction_validation_rejection(self) -> None:
        # Destination must be VGPR
        with self.assertRaises(TypeError):
            VQsadPkU16U8Instruction(
                dst=Operand.sgpr_pair(0),
                src0=Operand.vgpr_pair(2),
                src1=Operand.vgpr(4),
                src2=Operand.vgpr_pair(6),
            )
        # Destination must be a pair
        with self.assertRaises(ValueError):
            VQsadPkU16U8Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr_pair(2),
                src1=Operand.vgpr(4),
                src2=Operand.vgpr_pair(6),
            )
        # src0 must be 64-bit
        with self.assertRaises(ValueError):
            VQsadPkU16U8Instruction(
                dst=Operand.vgpr_pair(0),
                src0=Operand.vgpr(2),
                src1=Operand.vgpr(4),
                src2=Operand.vgpr_pair(6),
            )
        # src1 must be 32-bit (not pair)
        with self.assertRaises(ValueError):
            VQsadPkU16U8Instruction(
                dst=Operand.vgpr_pair(0),
                src0=Operand.vgpr_pair(2),
                src1=Operand.vgpr_pair(4),
                src2=Operand.vgpr_pair(6),
            )
        # src2 must be 64-bit
        with self.assertRaises(ValueError):
            VQsadPkU16U8Instruction(
                dst=Operand.vgpr_pair(0),
                src0=Operand.vgpr_pair(2),
                src1=Operand.vgpr(4),
                src2=Operand.vgpr(6),
            )


class TestValuQsadPkU16U8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.regs = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.regs)

    def test_register_file_32bit_and_64bit_access(self) -> None:
        # 32-bit single write and read
        self.regs.write(Operand.vgpr(10), 0x12345678)
        self.assertEqual(self.regs.read(Operand.vgpr(10)), 0x12345678)

        # 64-bit pair write and read
        self.regs.write(Operand.vgpr_pair(0), 0x0123456789ABCDEF)
        self.assertEqual(self.regs.read(Operand.vgpr_pair(0)), 0x0123456789ABCDEF)
        self.assertEqual(self.regs.read(Operand.vgpr(0)), 0x89ABCDEF)
        self.assertEqual(self.regs.read(Operand.vgpr(1)), 0x01234567)

        # SGPR pair write and read
        self.regs.write(Operand.sgpr_pair(2), 0xFEDCBA9876543210)
        self.assertEqual(self.regs.read(Operand.sgpr_pair(2)), 0xFEDCBA9876543210)
        self.assertEqual(self.regs.read(Operand.sgpr(2)), 0x76543210)
        self.assertEqual(self.regs.read(Operand.sgpr(3)), 0xFEDCBA98)

        # Truncation check
        self.regs.write(Operand.vgpr(20), 0x1_0000_0001)
        self.assertEqual(self.regs.read(Operand.vgpr(20)), 1)

        # Cannot write to literal
        with self.assertRaises(TypeError):
            self.regs.write(Operand.literal(5), 10)

    def test_scalar_execution_basic(self) -> None:
        # Ref: [10, 20, 30, 40, 50, 60, 70, 80]
        # Query: [10, 20, 30, 40]
        # Acc: [0, 0, 0, 0]
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(0, 0, 0, 0)

        self.regs.write(Operand.vgpr_pair(2), ref)
        self.regs.write(Operand.vgpr(4), query)
        self.regs.write(Operand.vgpr_pair(6), acc)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(6),
        )

        res = self.interp.step(instr)
        self.assertEqual(unpack_u16x4(res), (0, 40, 80, 120))
        self.assertEqual(self.regs.read(Operand.vgpr_pair(0)), res)

    def test_scalar_execution_sgpr_and_literal(self) -> None:
        # Maximum difference 1020, accumulator 0xFFFF, clamp=True -> 0xFFFF
        ref = pack_u8x8(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF)
        self.regs.write(Operand.vgpr_pair(4), ref)
        self.regs.write(Operand.sgpr(2), 0x00000000)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(4),
            src1=Operand.sgpr(2),
            src2=Operand.literal64(pack_u16x4(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF)),
            clamp=True,
        )

        res = self.interp.step(instr)
        self.assertEqual(unpack_u16x4(res), (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF))
        self.assertEqual(self.regs.read(Operand.vgpr_pair(0)), res)

    def test_scalar_sequential_pipeline(self) -> None:
        # Step 1: Compute SAD into v[0:1]
        ref1 = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query1 = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        self.regs.write(Operand.vgpr_pair(2), ref1)
        self.regs.write(Operand.vgpr(4), query1)
        self.regs.write(Operand.vgpr_pair(6), 0)

        instr1 = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(6),
        )
        res1 = self.interp.step(instr1)
        self.assertEqual(unpack_u16x4(res1), (0, 40, 80, 120))

        # Step 2: Use v[0:1] as accumulator src2 in a second instruction
        # Identical ref2 and query2 -> SADs = 0, so result equals accumulated v[0:1]
        ref2 = pack_u8x8(5, 5, 5, 5, 5, 5, 5, 5)
        query2 = pack_u8x8(5, 5, 5, 5, 0, 0, 0, 0) & MASK32
        self.regs.write(Operand.vgpr_pair(8), ref2)
        self.regs.write(Operand.vgpr(10), query2)

        instr2 = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(12),
            src0=Operand.vgpr_pair(8),
            src1=Operand.vgpr(10),
            src2=Operand.vgpr_pair(0),  # Reads result of step 1
        )
        res2 = self.interp.step(instr2)
        self.assertEqual(unpack_u16x4(res2), (0, 40, 80, 120))
        self.assertEqual(self.regs.read(Operand.vgpr_pair(12)), res1)


class TestValuQsadPkU16U8SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        # Each lane has distinct reference slice and query
        for lane in range(32):
            ref = pack_u8x8(lane, lane + 1, lane + 2, lane + 3, lane + 4, lane + 5, lane + 6, lane + 7)
            query = pack_u8x8(lane, lane + 1, lane + 2, lane + 3, 0, 0, 0, 0) & MASK32
            acc = pack_u16x4(lane * 10, lane * 20, lane * 30, lane * 40)
            state.write_lane(Operand.vgpr_pair(2), lane, ref)
            state.write_lane(Operand.vgpr(4), lane, query)
            state.write_lane(Operand.vgpr_pair(6), lane, acc)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(6),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            read_back = state.read_lane(Operand.vgpr_pair(0), lane)
            self.assertEqual(results[lane], read_back)
            # Window 0 matches exactly -> SAD=0, output is acc[0] = lane * 10
            e0, e1, e2, e3 = unpack_u16x4(read_back)
            self.assertEqual(e0, lane * 10)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
            query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
            acc = pack_u16x4(lane, lane, lane, lane)
            state.write_lane(Operand.vgpr_pair(2), lane, ref)
            state.write_lane(Operand.vgpr(4), lane, query)
            state.write_lane(Operand.vgpr_pair(6), lane, acc)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(6),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertIsNotNone(results[lane])
            e0, e1, e2, e3 = unpack_u16x4(state.read_lane(Operand.vgpr_pair(0), lane))
            self.assertEqual(e0, lane)  # 0 SAD + lane

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel_lo = 0xDEADBEEF
        sentinel_hi = 0xCAFEBABE
        sentinel_pair = (sentinel_hi << 32) | sentinel_lo

        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr_pair(0), lane, sentinel_pair)
            ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
            query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
            acc = 0
            state.write_lane(Operand.vgpr_pair(2), lane, ref)
            state.write_lane(Operand.vgpr(4), lane, query)
            state.write_lane(Operand.vgpr_pair(6), lane, acc)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(6),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_pair(0), lane), pack_u16x4(0, 40, 80, 120))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_pair(0), lane), sentinel_pair)

    def test_zero_exec_mask_no_modification(self) -> None:
        sentinel = 0x1122334455667788
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr_pair(0), lane, sentinel)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.literal64(0),
            src1=Operand.literal(0),
            src2=Operand.literal64(0),
        )

        results = interp.step(instr, state)
        self.assertEqual(results, [None] * 32)
        for lane in range(32):
            self.assertEqual(state.read_lane(Operand.vgpr_pair(0), lane), sentinel)

    def test_sgpr_uniform_broadcast_across_divergent_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        # Uniform query in s4, uniform accumulator in s[6:7]
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(100, 200, 300, 400)
        state.sgprs[4] = query
        state.sgprs[6] = acc & MASK32
        state.sgprs[7] = (acc >> 32) & MASK32

        for lane in range(32):
            # Divergent reference per lane
            ref = pack_u8x8(10 + lane, 20, 30, 40, 50, 60, 70, 80)
            state.write_lane(Operand.vgpr_pair(2), lane, ref)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.sgpr(4),
            src2=Operand.sgpr_pair(6),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            e0, e1, e2, e3 = unpack_u16x4(state.read_lane(Operand.vgpr_pair(0), lane))
            # Window 0 difference: |(10 + lane) - 10| = lane. Result = lane + 100
            self.assertEqual(e0, 100 + lane)

    def test_wave64_upper_divergence(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_0000_0000)  # only lanes 32..63 active
        interp = SIMDInterpreter()
        sentinel = 0x9999999999999999

        for lane in range(64):
            state.write_lane(Operand.vgpr_pair(0), lane, sentinel)
            ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
            query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
            state.write_lane(Operand.vgpr_pair(2), lane, ref)
            state.write_lane(Operand.vgpr(4), lane, query)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.literal64(0),
        )

        results = interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr_pair(0), lane), sentinel)
            else:
                self.assertIsNotNone(results[lane])
                self.assertNotEqual(state.read_lane(Operand.vgpr_pair(0), lane), sentinel)


class TestValuQsadPkU16U8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # Ref in v[0:1], will be overwritten by dst v[0:1]
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32
        acc = pack_u16x4(1, 2, 3, 4)

        regs.write(Operand.vgpr_pair(0), ref)  # src0 and dst
        regs.write(Operand.vgpr(2), query)     # src1
        regs.write(Operand.vgpr_pair(4), acc)  # src2

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(0),  # Aliased with dst
            src1=Operand.vgpr(2),
            src2=Operand.vgpr_pair(4),
        )

        res = interp.step(instr)
        expected = pack_u16x4(1, 42, 83, 124)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), expected)
        # src1 and src2 remain intact
        self.assertEqual(regs.read(Operand.vgpr(2)), query)
        self.assertEqual(regs.read(Operand.vgpr_pair(4)), acc)

    def test_dst_aliases_src1_low_dword(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # src1 is v0 (low dword of dst v[0:1])
        query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32
        ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
        acc = pack_u16x4(0, 0, 0, 0)

        regs.write(Operand.vgpr(0), query)     # src1 and dst lo
        regs.write(Operand.vgpr(1), 0xCAFEBABE)  # dst hi
        regs.write(Operand.vgpr_pair(2), ref)  # src0
        regs.write(Operand.vgpr_pair(4), acc)  # src2

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(0),  # Aliased with dst lo
            src2=Operand.vgpr_pair(4),
        )

        res = interp.step(instr)
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e0, 0)  # [1,2,3,4] vs [1,2,3,4] -> SAD=0
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), res)

    def test_dst_aliases_src1_high_dword(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # src1 is v1 (high dword of dst v[0:1])
        query = pack_u8x8(5, 6, 7, 8, 0, 0, 0, 0) & MASK32
        ref = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)

        regs.write(Operand.vgpr(0), 0x11111111)  # dst lo
        regs.write(Operand.vgpr(1), query)       # src1 and dst hi
        regs.write(Operand.vgpr_pair(2), ref)    # src0
        regs.write(Operand.vgpr_pair(4), 0)      # src2

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(1),  # Aliased with dst hi
            src2=Operand.vgpr_pair(4),
        )

        res = interp.step(instr)
        # Window 3 is [4,5,6,7] vs [5,6,7,8] -> SAD = 4
        e0, e1, e2, e3 = unpack_u16x4(res)
        self.assertEqual(e3, 4)
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), res)

    def test_dst_aliases_src2_in_place_accumulation(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # Accumulator initially in v[0:1]
        initial_acc = pack_u16x4(100, 200, 300, 400)
        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32

        regs.write(Operand.vgpr_pair(0), initial_acc)  # src2 and dst
        regs.write(Operand.vgpr_pair(2), ref)          # src0
        regs.write(Operand.vgpr(4), query)             # src1

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(0),  # In-place accumulation!
        )

        res = interp.step(instr)
        expected = pack_u16x4(100, 240, 380, 520)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), expected)

    def test_dst_aliases_src0_and_src2(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # v[0:1] is used for both reference window and accumulator!
        val = pack_u8x8(1, 2, 3, 4, 5, 6, 7, 8)
        query = pack_u8x8(1, 2, 3, 4, 0, 0, 0, 0) & MASK32

        regs.write(Operand.vgpr_pair(0), val)
        regs.write(Operand.vgpr(2), query)

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(0),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr_pair(0),
        )

        # Expected calculation:
        # src0 = val, src1 = query, src2 = val
        expected = ref_qsad_pk_u16_u8(val, query, val)
        res = interp.step(instr)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), expected)

    def test_partial_overlap_dst_and_src0(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # dst is v[0:1], src0 is v[1:2] (overlapping at v1)
        # v1 is src0 lo, v2 is src0 hi
        src0_lo = 0x04030201
        src0_hi = 0x08070605
        src0 = (src0_hi << 32) | src0_lo

        regs.write(Operand.vgpr(0), 0xAAAAAAAA)
        regs.write(Operand.vgpr(1), src0_lo)
        regs.write(Operand.vgpr(2), src0_hi)
        regs.write(Operand.vgpr(3), 0x04030201)  # query
        regs.write(Operand.vgpr_pair(4), 0)      # acc

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(1),  # Overlaps with dst at v1
            src1=Operand.vgpr(3),
            src2=Operand.vgpr_pair(4),
        )

        expected = ref_qsad_pk_u16_u8(src0, 0x04030201, 0)
        res = interp.step(instr)
        self.assertEqual(res, expected)
        self.assertEqual(regs.read(Operand.vgpr_pair(0)), expected)

    def test_simd_multi_lane_aliased_in_place(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active
        interp = SIMDInterpreter()

        ref = pack_u8x8(10, 20, 30, 40, 50, 60, 70, 80)
        query = pack_u8x8(10, 20, 30, 40, 0, 0, 0, 0) & MASK32

        for lane in range(32):
            acc = pack_u16x4(lane, lane, lane, lane)
            state.write_lane(Operand.vgpr_pair(0), lane, acc)  # dst and src2
            state.write_lane(Operand.vgpr_pair(2), lane, ref)  # src0
            state.write_lane(Operand.vgpr(4), lane, query)     # src1

        instr = VQsadPkU16U8Instruction(
            dst=Operand.vgpr_pair(0),
            src0=Operand.vgpr_pair(2),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr_pair(0),  # In-place accumulation across SIMD lanes
        )

        interp.step(instr, state)
        for lane in range(32):
            val = state.read_lane(Operand.vgpr_pair(0), lane)
            if lane < 16:
                expected = pack_u16x4(lane, 40 + lane, 80 + lane, 120 + lane)
                self.assertEqual(val, expected)
            else:
                original = pack_u16x4(lane, lane, lane, lane)
                self.assertEqual(val, original)


if __name__ == "__main__":
    unittest.main()
