from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_mqsad_pk_u16_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_mqsad_pk_u16_u8 dst, src0, src1, src2

Format:
  VOP3 (3-operand vector ALU instruction, 64-bit opcode encoding).

Operands:
  dst:  64-bit vector register pair (v[d:d+1]) containing 4x packed unsigned 16-bit integers.
  src0: 64-bit vector register pair (v[s0:s0+1]) containing 8x packed unsigned 8-bit integers (sliding reference window).
  src1: 32-bit register (s[s1] or v[s1]) containing 4x packed unsigned 8-bit integers (query/template with byte-zero masking).
  src2: 64-bit vector register pair (v[s2:s2+1]) containing 4x packed unsigned 16-bit accumulator values.

Operation:
  Computes four overlapping 4-byte Sum of Absolute Differences (SAD) between sliding 4-byte
  windows in src0 and the 4-byte query pattern in src1, accumulating into 16-bit packed results:
    D[15:0]  = MSAD_U8(S0[31:0],  S1[31:0], S2[15:0])
    D[31:16] = MSAD_U8(S0[39:8],  S1[31:0], S2[31:16])
    D[47:32] = MSAD_U8(S0[47:16], S1[31:0], S2[47:32])
    D[63:48] = MSAD_U8(S0[55:24], S1[31:0], S2[63:48])

Masking Semantics (MSAD_U8):
  For each byte k in {0, 1, 2, 3}:
    If query byte S1.byte[k] == 0:
      diff_k = 0 (byte comparison is skipped / masked out)
    Else:
      diff_k = |S0_window.byte[k] - S1.byte[k]|
  Accumulation:
    out_16 = (S2_acc_16 + diff_0 + diff_1 + diff_2 + diff_3) & 0xFFFF

Sliding Window Byte Layout in S0:
  Window 0: bytes [0, 1, 2, 3] = S0[31:0]
  Window 1: bytes [1, 2, 3, 4] = S0[39:8]
  Window 2: bytes [2, 3, 4, 5] = S0[47:16]
  Window 3: bytes [3, 4, 5, 6] = S0[55:24]
  Note: S0 byte 7 (bits [63:56]) is not consumed by the 4-window quad-SAD, but S0 is aligned to a 64-bit register pair.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Sequence, Tuple, Union
import unittest

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


def extract_byte(val: int, idx: int) -> int:
    """Extract byte at index `idx` (0-indexed from LSB)."""
    return (val >> (8 * idx)) & MASK8


def extract_u16(val: int, idx: int) -> int:
    """Extract 16-bit word at index `idx` (0-indexed from LSB)."""
    return (val >> (16 * idx)) & MASK16


def pack_u16_to_u64(w0: int, w1: int, w2: int, w3: int) -> int:
    """Pack four unsigned 16-bit halfwords into an unsigned 64-bit qword."""
    return (
        (w0 & MASK16)
        | ((w1 & MASK16) << 16)
        | ((w2 & MASK16) << 32)
        | ((w3 & MASK16) << 48)
    )


def ref_abs_diff_u8(b0: int, b1: int) -> int:
    """Pure-Python reference oracle for 8-bit unsigned absolute difference: |b0 - b1|."""
    u0 = b0 & MASK8
    u1 = b1 & MASK8
    return (u0 - u1) if u0 >= u1 else (u1 - u0)


def ref_msad_u8(s0_chunk32: int, s1_q32: int, acc16: int) -> int:
    """Reference mathematical oracle for a single 4-byte masked SAD lane.

    Computes:
      acc = acc16 & 0xFFFF
      for k in 0..3:
        b_s0 = (s0_chunk32 >> (8*k)) & 0xFF
        b_s1 = (s1_q32 >> (8*k)) & 0xFF
        if b_s1 != 0:
          acc += |b_s0 - b_s1|
      return acc & 0xFFFF
    """
    acc = acc16 & MASK16
    for k in range(4):
        b_s0 = extract_byte(s0_chunk32, k)
        b_s1 = extract_byte(s1_q32, k)
        if b_s1 != 0:
            acc += ref_abs_diff_u8(b_s0, b_s1)
    return acc & MASK16


def ref_mqsad_pk_u16_u8(s0_64: int, s1_32: int, s2_64: int) -> Tuple[int, int, int, int]:
    """Reference mathematical oracle returning 4x packed 16-bit words (w0, w1, w2, w3)."""
    win0 = (s0_64 >> 0) & MASK32
    win1 = (s0_64 >> 8) & MASK32
    win2 = (s0_64 >> 16) & MASK32
    win3 = (s0_64 >> 24) & MASK32

    acc0 = extract_u16(s2_64, 0)
    acc1 = extract_u16(s2_64, 1)
    acc2 = extract_u16(s2_64, 2)
    acc3 = extract_u16(s2_64, 3)

    out0 = ref_msad_u8(win0, s1_32, acc0)
    out1 = ref_msad_u8(win1, s1_32, acc1)
    out2 = ref_msad_u8(win2, s1_32, acc2)
    out3 = ref_msad_u8(win3, s1_32, acc3)

    return out0, out1, out2, out3


def ref_mqsad_pk_u16_u8_u64(s0_64: int, s1_32: int, s2_64: int) -> int:
    """Reference mathematical oracle returning full packed 64-bit result."""
    w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(s0_64, s1_32, s2_64)
    return pack_u16_to_u64(w0, w1, w2, w3)


class TestValuMqsadPkU16U8ReferenceOracle(unittest.TestCase):
    """Architectural reference tests validating mathematical model for v_mqsad_pk_u16_u8."""

    def test_pure_zero_operands(self) -> None:
        """All zero operands produce zero output across all 4 packed lanes."""
        w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(0, 0, 0)
        self.assertEqual((w0, w1, w2, w3), (0, 0, 0, 0))
        self.assertEqual(ref_mqsad_pk_u16_u8_u64(0, 0, 0), 0)

    def test_accumulator_pass_through_on_all_zero_mask(self) -> None:
        """When query s1 is 0, all 4 byte diffs are masked out and accumulator passes through unchanged."""
        s2 = pack_u16_to_u64(0x1234, 0x5678, 0x9ABC, 0xDEF0)
        s0 = 0xFFFF_FFFF_FFFF_FFFF
        s1 = 0x0000_0000  # All query bytes are 0 -> all masked out
        w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(s0, s1, s2)
        self.assertEqual((w0, w1, w2, w3), (0x1234, 0x5678, 0x9ABC, 0xDEF0))
        self.assertEqual(ref_mqsad_pk_u16_u8_u64(s0, s1, s2), s2)

    def test_exact_match_zero_difference(self) -> None:
        """Identical reference and query bytes produce zero difference, preserving accumulator."""
        # s1 query is [10, 20, 30, 40]
        s1 = 10 | (20 << 8) | (30 << 16) | (40 << 24)
        # s0 has [10, 20, 30, 40] repeated at every sliding window offset
        s0_bytes = [10, 20, 30, 40, 10, 20, 30, 40]
        s0 = sum(b << (8 * i) for i, b in enumerate(s0_bytes))
        # For window 0 (bytes 0..3): exact match -> diff = 0
        w0, _, _, _ = ref_mqsad_pk_u16_u8(s0, s1, 0)
        self.assertEqual(w0, 0)

    def test_sliding_window_offsets(self) -> None:
        """Verify that sliding windows 0, 1, 2, 3 correctly sample bytes 0..3, 1..4, 2..5, 3..6."""
        # Reference bytes: 1, 2, 3, 4, 5, 6, 7, 8
        s0 = sum((i + 1) << (8 * i) for i in range(8))
        # Query matches bytes 1..4 exactly: [2, 3, 4, 5]
        s1 = 2 | (3 << 8) | (4 << 16) | (5 << 24)
        w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(s0, s1, 0)
        # Window 0 [1, 2, 3, 4] vs [2, 3, 4, 5]: diffs |1-2|=1, |2-3|=1, |3-4|=1, |4-5|=1 -> sum = 4
        self.assertEqual(w0, 4)
        # Window 1 [2, 3, 4, 5] vs [2, 3, 4, 5]: exact match -> sum = 0
        self.assertEqual(w1, 0)
        # Window 2 [3, 4, 5, 6] vs [2, 3, 4, 5]: diffs 1, 1, 1, 1 -> sum = 4
        self.assertEqual(w2, 4)
        # Window 3 [4, 5, 6, 7] vs [2, 3, 4, 5]: diffs 2, 2, 2, 2 -> sum = 8
        self.assertEqual(w3, 8)

    def test_masked_byte_zero_skipping(self) -> None:
        """Verify that only query bytes == 0 are masked, non-zero query bytes are evaluated."""
        # Query: byte 0 is 10, byte 1 is 0 (masked), byte 2 is 30, byte 3 is 0 (masked)
        s1 = 10 | (0 << 8) | (30 << 16) | (0 << 24)
        # S0 window 0: [20, 99, 10, 88]
        s0 = 20 | (99 << 8) | (10 << 16) | (88 << 24)
        w0, _, _, _ = ref_mqsad_pk_u16_u8(s0, s1, 0)
        # Byte 0: |20 - 10| = 10
        # Byte 1: masked out (0)
        # Byte 2: |10 - 30| = 20
        # Byte 3: masked out (0)
        # Total = 10 + 20 = 30
        self.assertEqual(w0, 30)

    def test_maximum_byte_difference_and_accumulation(self) -> None:
        """Max difference per byte is 255. 4 active bytes = 1020 max SAD per step."""
        # Query: [255, 255, 255, 255]
        s1 = 0xFFFF_FFFF
        # S0: all zeros
        s0 = 0
        w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(s0, s1, 0)
        self.assertEqual((w0, w1, w2, w3), (1020, 1020, 1020, 1020))

    def test_accumulator_16bit_modulo_wrapping(self) -> None:
        """16-bit packed lanes wrap modulo 2^16 (65536) on arithmetic overflow."""
        # Query diff is 1020
        s1 = 0xFFFF_FFFF
        s0 = 0
        # Accumulator is 0xFFFF (65535). 65535 + 1020 = 66555 = 0x103FB -> wraps to 0x03FB (1019)
        s2 = pack_u16_to_u64(0xFFFF, 0xFFFE, 0x0000, 0x8000)
        w0, w1, w2, w3 = ref_mqsad_pk_u16_u8(s0, s1, s2)
        self.assertEqual(w0, (0xFFFF + 1020) & 0xFFFF)
        self.assertEqual(w0, 1019)
        self.assertEqual(w1, (0xFFFE + 1020) & 0xFFFF)
        self.assertEqual(w1, 1018)
        self.assertEqual(w2, 1020)
        self.assertEqual(w3, (0x8000 + 1020) & 0xFFFF)


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
    count: int = 1
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"Operand register count must be >= 1, got {self.count}")
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
            if self.index + self.count > 256:
                raise ValueError(f"VGPR range [v{self.index}:v{self.index + self.count - 1}] exceeds 255")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
            if self.index + self.count > 106:
                raise ValueError(f"SGPR range [s{self.index}:s{self.index + self.count - 1}] exceeds 105")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if not (0 <= self.literal_value <= MASK64):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, 0xFFFFFFFFFFFFFFFF]")

    @classmethod
    def vgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.VGPR, index=index, count=count)

    @classmethod
    def sgpr(cls, index: int, count: int = 1) -> Operand:
        return cls(RegType.SGPR, index=index, count=count)

    @classmethod
    def literal(cls, value: int, count: int = 1) -> Operand:
        return cls(RegType.LITERAL, count=count, literal_value=u64(value))

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            if self.count == 1:
                return f"v{self.index}"
            return f"v[{self.index}:{self.index + self.count - 1}]"
        if self.reg_type == RegType.SGPR:
            if self.count == 1:
                return f"s{self.index}"
            return f"s[{self.index}:{self.index + self.count - 1}]"
        assert self.literal_value is not None
        if self.literal_value > MASK32:
            return f"0x{self.literal_value:016X}"
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VMqsadPkU16U8Instruction:
    """Symbolic AST node for AMD RDNA2 v_mqsad_pk_u16_u8 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")
        if self.dst.count != 2:
            raise ValueError(f"Destination must be a 64-bit VGPR pair (count=2), got {self.dst.count}")
        if self.src0.count != 2:
            raise ValueError(f"src0 must be a 64-bit operand (count=2), got {self.src0.count}")
        if self.src1.count != 1:
            raise ValueError(f"src1 must be a 32-bit operand (count=1), got {self.src1.count}")
        if self.src2.count != 2:
            raise ValueError(f"src2 must be a 64-bit operand (count=2), got {self.src2.count}")

    def disassemble(self) -> str:
        return (
            f"v_mqsad_pk_u16_u8 {self.dst.disassemble()}, "
            f"{self.src0.disassemble()}, {self.src1.disassemble()}, "
            f"{self.src2.disassemble()}"
        )


class TestValuMqsadPkU16U8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self) -> None:
        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(4, count=1),
            src2=Operand.vgpr(6, count=2),
        )
        self.assertEqual(instr.dst.index, 0)
        self.assertEqual(instr.dst.count, 2)
        self.assertEqual(instr.src0.index, 2)
        self.assertEqual(instr.src1.index, 4)
        self.assertEqual(instr.src2.index, 6)
        self.assertEqual(
            instr.disassemble(),
            "v_mqsad_pk_u16_u8 v[0:1], v[2:3], s4, v[6:7]",
        )

    def test_ast_disassembly_with_literal(self) -> None:
        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(10, count=2),
            src0=Operand.vgpr(12, count=2),
            src1=Operand.literal(0x11223344),
            src2=Operand.vgpr(14, count=2),
        )
        self.assertEqual(
            instr.disassemble(),
            "v_mqsad_pk_u16_u8 v[10:11], v[12:13], 0x11223344, v[14:15]",
        )

    def test_ast_immutability(self) -> None:
        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(4, count=1),
            src2=Operand.vgpr(6, count=2),
        )
        with self.assertRaises(Exception):
            instr.dst = Operand.vgpr(8, count=2)  # type: ignore[misc]

    def test_ast_operand_range_bounds(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(255, count=2)  # 255 + 2 = 257 > 256
        with self.assertRaises(ValueError):
            Operand.sgpr(105, count=2)  # 105 + 2 = 107 > 106
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarRegisterState:
    """Architectural register state for a single thread/scalar core."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def read_operand_32(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def read_operand_64(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[op.index] & MASK32
            hi = self.vgprs[op.index + 1] & MASK32
            return lo | (hi << 32)
        if op.reg_type == RegType.SGPR:
            lo = self.sgprs[op.index] & MASK32
            hi = self.sgprs[op.index + 1] & MASK32
            return lo | (hi << 32)
        assert op.literal_value is not None
        return op.literal_value & MASK64

    def write_vgpr_64(self, index: int, val64: int) -> None:
        self.vgprs[index] = val64 & MASK32
        self.vgprs[index + 1] = (val64 >> 32) & MASK32


class ScalarInterpreter:
    """Deterministic scalar interpreter executing single VALU instructions."""

    def step(self, instr: VMqsadPkU16U8Instruction, state: ScalarRegisterState) -> int:
        s0 = state.read_operand_64(instr.src0)
        s1 = state.read_operand_32(instr.src1)
        s2 = state.read_operand_64(instr.src2)
        res = ref_mqsad_pk_u16_u8_u64(s0, s1, s2)
        state.write_vgpr_64(instr.dst.index, res)
        return res


class TestValuMqsadPkU16U8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.state = ScalarRegisterState()
        self.interp = ScalarInterpreter()

    def test_scalar_execution_from_vgprs(self) -> None:
        # Load src0 (64-bit) into v[2:3]: bytes 1..8
        s0_val = sum((i + 1) << (8 * i) for i in range(8))
        self.state.write_vgpr_64(2, s0_val)

        # Load src1 (32-bit query) into v4: [2, 3, 4, 5]
        s1_val = 2 | (3 << 8) | (4 << 16) | (5 << 24)
        self.state.vgprs[4] = s1_val

        # Load src2 (64-bit accumulator) into v[6:7]: all zero
        self.state.write_vgpr_64(6, 0)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.vgpr(4, count=1),
            src2=Operand.vgpr(6, count=2),
        )

        res = self.interp.step(instr, self.state)
        # S0 windows: win0=4, win1=0, win2=4, win3=8
        expected = pack_u16_to_u64(4, 0, 4, 8)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(0, count=2)), expected)

    def test_scalar_execution_with_sgpr_and_literal(self) -> None:
        # src0 in v[4:5]
        s0_val = 0x0102030405060708
        self.state.write_vgpr_64(4, s0_val)

        # src1 query in s10 (SGPR)
        self.state.sgprs[10] = 0x01020304

        # src2 accumulator as literal 0x0010002000300040
        acc_lit = pack_u16_to_u64(0x40, 0x30, 0x20, 0x10)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(4, count=2),
            src1=Operand.sgpr(10, count=1),
            src2=Operand.literal(acc_lit, count=2),
        )

        res = self.interp.step(instr, self.state)
        expected = ref_mqsad_pk_u16_u8_u64(s0_val, 0x01020304, acc_lit)
        self.assertEqual(res, expected)

    def test_multi_step_chained_accumulation(self) -> None:
        """Sequential chained accumulation into the same destination register pair."""
        # Initial accumulation: v[0:1] = mqsad(v[2:3], s4, 0)
        s0_1 = 0x11111111_22222222
        self.state.write_vgpr_64(2, s0_1)
        self.state.sgprs[4] = 0x11111111
        self.state.write_vgpr_64(6, 0)

        instr1 = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(4, count=1),
            src2=Operand.vgpr(6, count=2),
        )
        res1 = self.interp.step(instr1, self.state)

        # Second step: feed v[0:1] as accumulator src2
        s0_2 = 0x33333333_44444444
        self.state.write_vgpr_64(8, s0_2)
        instr2 = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(8, count=2),
            src1=Operand.sgpr(4, count=1),
            src2=Operand.vgpr(0, count=2),
        )
        res2 = self.interp.step(instr2, self.state)

        expected2 = ref_mqsad_pk_u16_u8_u64(s0_2, 0x11111111, res1)
        self.assertEqual(res2, expected2)
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(0, count=2)), expected2)

    def test_scalar_interpreter_determinism(self) -> None:
        """Executing identical instruction sequences from identical state yields deterministic results."""
        state_a = ScalarRegisterState()
        state_b = ScalarRegisterState()
        val = 0xAA55AA55_55AA55AA
        state_a.write_vgpr_64(2, val)
        state_b.write_vgpr_64(2, val)
        state_a.sgprs[0] = 0x12345678
        state_b.sgprs[0] = 0x12345678

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(0, count=1),
            src2=Operand.literal(0x1000200030004000, count=2),
        )
        res_a = self.interp.step(instr, state_a)
        res_b = self.interp.step(instr, state_b)
        self.assertEqual(res_a, res_b)
        self.assertEqual(state_a.vgprs, state_b.vgprs)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    """Architectural SIMD state for Wave32 and Wave64 execution modes."""
    wave_size: int = 32
    exec_mask: int = MASK32
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size {self.wave_size}, must be 32 or 64")
        if self.wave_size == 32:
            self.exec_mask &= MASK32
        else:
            self.exec_mask &= MASK64
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size - 1}]")
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_operand_32(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        assert op.literal_value is not None
        return op.literal_value & MASK32

    def read_lane_operand_64(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[lane][op.index] & MASK32
            hi = self.vgprs[lane][op.index + 1] & MASK32
            return lo | (hi << 32)
        if op.reg_type == RegType.SGPR:
            lo = self.sgprs[op.index] & MASK32
            hi = self.sgprs[op.index + 1] & MASK32
            return lo | (hi << 32)
        assert op.literal_value is not None
        return op.literal_value & MASK64

    def write_lane_vgpr_64(self, index: int, lane: int, val64: int) -> None:
        self.vgprs[lane][index] = val64 & MASK32
        self.vgprs[lane][index + 1] = (val64 >> 32) & MASK32


class SIMDInterpreter:
    """SIMD vector interpreter executing instructions with lane-divergence and EXEC masking."""

    def step(self, instr: VMqsadPkU16U8Instruction, state: SIMDWaveState) -> List[int | None]:
        results: List[int | None] = []
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                results.append(None)
                continue
            s0 = state.read_lane_operand_64(instr.src0, lane)
            s1 = state.read_lane_operand_32(instr.src1, lane)
            s2 = state.read_lane_operand_64(instr.src2, lane)
            res = ref_mqsad_pk_u16_u8_u64(s0, s1, s2)
            state.write_lane_vgpr_64(instr.dst.index, lane, res)
            results.append(res)
        return results


class TestValuMqsadPkU16U8SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=32, exec_mask=MASK32)
        interp = SIMDInterpreter()

        # Shared query in s4
        state.sgprs[4] = 0x05040302
        # Distinct reference per lane in v[2:3]
        for lane in range(32):
            s0_val = sum(((lane + i) & 0xFF) << (8 * i) for i in range(8))
            state.write_lane_vgpr_64(2, lane, s0_val)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(4, count=1),
            src2=Operand.literal(0, count=2),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            s0_val = sum(((lane + i) & 0xFF) << (8 * i) for i in range(8))
            expected = ref_mqsad_pk_u16_u8_u64(s0_val, 0x05040302, 0)
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.read_lane_operand_64(Operand.vgpr(0, count=2), lane), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDWaveState(wave_size=64, exec_mask=MASK64)
        interp = SIMDInterpreter()
        state.sgprs[0] = 0x11223344

        for lane in range(64):
            val = (lane << 32) | lane
            state.write_lane_vgpr_64(2, lane, val)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(0, count=1),
            src2=Operand.literal(0x1000, count=2),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            val = (lane << 32) | lane
            expected = ref_mqsad_pk_u16_u8_u64(val, 0x11223344, 0x1000)
            self.assertEqual(results[lane], expected)

    def test_wave32_partial_exec_mask_inactive_lanes_unmodified(self) -> None:
        """Inactive lanes indicated by 0 in EXEC mask must not be written or modified."""
        # Alternating mask: lanes 0, 2, 4, ... active (0x55555555)
        mask = 0x5555_5555
        state = SIMDWaveState(wave_size=32, exec_mask=mask)
        interp = SIMDInterpreter()

        sentinel_lo = 0xDEAD_BEEF
        sentinel_hi = 0xCAFE_BABE
        sentinel64 = sentinel_lo | (sentinel_hi << 32)
        for lane in range(32):
            state.write_lane_vgpr_64(0, lane, sentinel64)
            state.write_lane_vgpr_64(2, lane, 0x01020304_05060708)
        state.sgprs[2] = 0x01010101

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(2, count=1),
            src2=Operand.literal(0, count=2),
        )

        results = interp.step(instr, state)
        expected_active = ref_mqsad_pk_u16_u8_u64(0x01020304_05060708, 0x01010101, 0)
        for lane in range(32):
            if (mask >> lane) & 1:
                self.assertEqual(results[lane], expected_active)
                self.assertEqual(state.read_lane_operand_64(Operand.vgpr(0, count=2), lane), expected_active)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane_operand_64(Operand.vgpr(0, count=2), lane), sentinel64)

    def test_wave64_isolated_lane_divergence(self) -> None:
        """Only lane 42 active in Wave64."""
        mask = 1 << 42
        state = SIMDWaveState(wave_size=64, exec_mask=mask)
        interp = SIMDInterpreter()

        sentinel = 0xBAAD_F00D_BAAD_F00D
        for lane in range(64):
            state.write_lane_vgpr_64(0, lane, sentinel)
            state.write_lane_vgpr_64(2, lane, 0x12345678_9ABCDEF0)
        state.sgprs[0] = 0x0F0F0F0F

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(0, count=1),
            src2=Operand.literal(0, count=2),
        )

        results = interp.step(instr, state)
        expected = ref_mqsad_pk_u16_u8_u64(0x12345678_9ABCDEF0, 0x0F0F0F0F, 0)
        for lane in range(64):
            if lane == 42:
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.read_lane_operand_64(Operand.vgpr(0, count=2), lane), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane_operand_64(Operand.vgpr(0, count=2), lane), sentinel)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestValuMqsadPkU16U8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing hazards and in-place operand preservation."""

    def setUp(self) -> None:
        self.state = ScalarRegisterState()
        self.interp = ScalarInterpreter()

    def test_aliasing_dst_equals_src2_inplace_accumulator(self) -> None:
        """v[0:1] = v_mqsad_pk_u16_u8 v[0:1], v[2:3], s4, v[0:1]"""
        # Initial accumulator in v[0:1]
        acc_initial = pack_u16_to_u64(10, 20, 30, 40)
        self.state.write_vgpr_64(0, acc_initial)

        # src0 in v[2:3]
        s0_val = sum((i + 1) << (8 * i) for i in range(8))
        self.state.write_vgpr_64(2, s0_val)

        # src1 query in s4
        s1_val = 1 | (2 << 8) | (3 << 16) | (4 << 24)
        self.state.sgprs[4] = s1_val

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(4, count=1),
            src2=Operand.vgpr(0, count=2),
        )

        expected = ref_mqsad_pk_u16_u8_u64(s0_val, s1_val, acc_initial)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(0, count=2)), expected)

    def test_aliasing_dst_equals_src0_reference_window(self) -> None:
        """v[0:1] = v_mqsad_pk_u16_u8 v[0:1], v[0:1], s2, v[4:5]"""
        s0_initial = 0xAABBCCDD_11223344
        self.state.write_vgpr_64(0, s0_initial)

        s1_val = 0x11223344
        self.state.sgprs[2] = s1_val

        acc_val = pack_u16_to_u64(100, 200, 300, 400)
        self.state.write_vgpr_64(4, acc_val)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(0, count=2),
            src1=Operand.sgpr(2, count=1),
            src2=Operand.vgpr(4, count=2),
        )

        expected = ref_mqsad_pk_u16_u8_u64(s0_initial, s1_val, acc_val)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(0, count=2)), expected)

    def test_aliasing_src0_equals_src2_shared_operands(self) -> None:
        """dst v[0:1] distinct, src0 and src2 both read from v[2:3]."""
        shared_val = 0x01020304_05060708
        self.state.write_vgpr_64(2, shared_val)
        self.state.sgprs[0] = 0x05060708

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(2, count=2),
            src1=Operand.sgpr(0, count=1),
            src2=Operand.vgpr(2, count=2),
        )

        expected = ref_mqsad_pk_u16_u8_u64(shared_val, 0x05060708, shared_val)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        # Verify source v[2:3] is uncorrupted
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(2, count=2)), shared_val)

    def test_aliasing_triple_collision(self) -> None:
        """dst == src0 == src2 (v[0:1] all aliased)."""
        initial = 0x55AA55AA_AA55AA55
        self.state.write_vgpr_64(0, initial)
        self.state.sgprs[0] = 0xAA55AA55

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(0, count=2),
            src0=Operand.vgpr(0, count=2),
            src1=Operand.sgpr(0, count=1),
            src2=Operand.vgpr(0, count=2),
        )

        expected = ref_mqsad_pk_u16_u8_u64(initial, 0xAA55AA55, initial)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(0, count=2)), expected)

    def test_aliasing_partial_overlap_low_and_high(self) -> None:
        """dst v[1:2] partially overlaps with src0 v[0:1]."""
        # Set v0 = 0x11223344, v1 = 0x55667788, v2 = 0x99AABBCC
        self.state.vgprs[0] = 0x11223344
        self.state.vgprs[1] = 0x55667788
        self.state.vgprs[2] = 0x99AABBCC

        src0_val = 0x11223344 | (0x55667788 << 32)
        s1_val = 0x12345678
        acc_val = pack_u16_to_u64(1, 2, 3, 4)
        self.state.write_vgpr_64(4, acc_val)

        instr = VMqsadPkU16U8Instruction(
            dst=Operand.vgpr(1, count=2),
            src0=Operand.vgpr(0, count=2),
            src1=Operand.literal(s1_val),
            src2=Operand.vgpr(4, count=2),
        )

        expected = ref_mqsad_pk_u16_u8_u64(src0_val, s1_val, acc_val)
        res = self.interp.step(instr, self.state)
        self.assertEqual(res, expected)
        # v0 must remain intact
        self.assertEqual(self.state.vgprs[0], 0x11223344)
        # v[1:2] must hold result
        self.assertEqual(self.state.read_operand_64(Operand.vgpr(1, count=2)), expected)


if __name__ == "__main__":
    unittest.main()
