from __future__ import annotations

"""Semantic reference oracle and architectural model for AMD RDNA2 v_dot4_u32_u8.

AMD RDNA VALU Instruction Set Architecture:
  v_dot4_u32_u8 dst, src0, src1, src2

Format:
  VOP3 (3-operand vector ALU instruction with 32-bit operands).

Operands:
  - dst:  32-bit vector destination register (VGPR)
  - src0: 32-bit vector or scalar register, or 32-bit inline/literal constant
  - src1: 32-bit vector or scalar register, or 32-bit inline/literal constant
  - src2: 32-bit accumulator register (VGPR, SGPR, or constant)

Operation:
  Computes the 4-element unsigned 8-bit dot product with 32-bit unsigned accumulation:
    b0 = (src0 >> 0) & 0xFF
    b1 = (src0 >> 8) & 0xFF
    b2 = (src0 >> 16) & 0xFF
    b3 = (src0 >> 24) & 0xFF

    c0 = (src1 >> 0) & 0xFF
    c1 = (src1 >> 8) & 0xFF
    c2 = (src1 >> 16) & 0xFF
    c3 = (src1 >> 24) & 0xFF

    dot = (b0 * c0) + (b1 * c1) + (b2 * c2) + (b3 * c3)
    dst = (src2 + dot) mod 2^32

Mathematical Properties and Bounds:
  - Operands:
      src0 in [0, 2^32 - 1]
      src1 in [0, 2^32 - 1]
      src2 in [0, 2^32 - 1]
  - Each byte product (b_i * c_i) is in [0, 255 * 255] = [0, 65025].
  - The sum of four products dot is in [0, 4 * 65025] = [0, 260100] (0x0003F804).
  - Accumulation sum64 = src2 + dot is in [0, 2^32 - 1 + 260100] = [0, 0x10003F803].
  - Result wraps modulo 2^32 into [0, 2^32 - 1].
  - Carry-out into bit 32 is strictly bounded in {0, 1}.

Standard library Python only: unittest, dataclasses, enum, math, struct.
Single module scope, zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple, Union
import unittest

# Architectural bitmasks
MASK8: int = 0xFF
MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
MASK64: int = (1 << 64) - 1

MAX_BYTE_PRODUCT: int = 255 * 255  # 65025
MAX_DOT4_SUM: int = 4 * MAX_BYTE_PRODUCT  # 260100 (0x3F804)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def unpack_u8(dword: int) -> Tuple[int, int, int, int]:
    """Unpack 32-bit dword into four unsigned 8-bit integers (byte0, byte1, byte2, byte3)."""
    dword &= MASK32
    return (
        dword & MASK8,
        (dword >> 8) & MASK8,
        (dword >> 16) & MASK8,
        (dword >> 24) & MASK8,
    )


def pack_u8(b0: int, b1: int, b2: int, b3: int) -> int:
    """Pack four unsigned 8-bit integers into a 32-bit dword."""
    return (
        (b0 & MASK8)
        | ((b1 & MASK8) << 8)
        | ((b2 & MASK8) << 16)
        | ((b3 & MASK8) << 24)
    ) & MASK32


def ref_dot4_u32_u8(s0: int, s1: int, s2: int = 0) -> int:
    """Pure-Python bit-accurate reference oracle for AMD RDNA2 v_dot4_u32_u8.

    dst = (s2 + sum(s0[i] * s1[i] for i in 0..3)) mod 2^32
    """
    b = unpack_u8(s0)
    c = unpack_u8(s1)
    dot = (b[0] * c[0]) + (b[1] * c[1]) + (b[2] * c[2]) + (b[3] * c[3])
    acc = s2 & MASK32
    return (acc + dot) & MASK32


def ref_dot4_u32_u8_full(s0: int, s1: int, s2: int = 0) -> Tuple[int, int]:
    """Compute (lo32, carry_out) for (src2 + dot)."""
    b = unpack_u8(s0)
    c = unpack_u8(s1)
    dot = (b[0] * c[0]) + (b[1] * c[1]) + (b[2] * c[2]) + (b[3] * c[3])
    acc = s2 & MASK32
    full_sum = acc + dot
    return full_sum & MASK32, (full_sum >> 32) & 1


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
    literal_value: int | None = None

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 105):
                raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
        elif self.reg_type == RegType.LITERAL:
            if self.literal_value is None:
                raise ValueError("Literal operand requires literal_value")
            if not (0 <= self.literal_value <= MASK32):
                raise ValueError(f"Literal value {self.literal_value:#x} out of range [0, 0xFFFFFFFF]")

    @classmethod
    def vgpr(cls, index: int) -> Operand:
        return cls(RegType.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> Operand:
        return cls(RegType.SGPR, index=index)

    @classmethod
    def literal(cls, value: int) -> Operand:
        return cls(RegType.LITERAL, literal_value=value & MASK32)

    def disassemble(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        if self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        assert self.literal_value is not None
        return f"0x{self.literal_value:08X}"


@dataclass(frozen=True)
class VDot4U32U8Instruction:
    """Symbolic AST node for AMD RDNA2 v_dot4_u32_u8 instruction."""
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst}")
        if self.clamp:
            raise ValueError("clamp modifier is unsupported on v_dot4_u32_u8")

    def disassemble(self) -> str:
        return f"v_dot4_u32_u8 {self.dst.disassemble()}, {self.src0.disassemble()}, {self.src1.disassemble()}, {self.src2.disassemble()}"


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Architectural register file for a single execution thread."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read(self, op: Operand) -> int:
        """Read operand value as unsigned 32-bit dword."""
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value & MASK32
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write(self, op: Operand, value: int) -> None:
        """Write 32-bit dword to register operand (VGPR or SGPR)."""
        if op.reg_type == RegType.VGPR:
            self.vgprs[op.index] = u32(value)
        elif op.reg_type == RegType.SGPR:
            self.sgprs[op.index] = u32(value)
        else:
            raise TypeError("Cannot write to literal operand")


class ScalarVALUInterpreter:
    """Deterministic instruction interpreter for v_dot4_u32_u8 on a scalar register file."""

    def __init__(self, regs: ScalarRegisterFile | None = None) -> None:
        self.regs: ScalarRegisterFile = regs if regs is not None else ScalarRegisterFile()

    def step(self, instr: VDot4U32U8Instruction) -> int:
        # Atomic read phase
        s0 = self.regs.read(instr.src0)
        s1 = self.regs.read(instr.src1)
        s2 = self.regs.read(instr.src2)

        # Computation phase
        result = ref_dot4_u32_u8(s0, s1, s2)

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
        if op.reg_type == RegType.LITERAL:
            assert op.literal_value is not None
            return op.literal_value & MASK32
        if op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index] & MASK32
        if op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write_lane(self, op: Operand, lane: int, value: int) -> None:
        if op.reg_type == RegType.VGPR:
            self.vgprs[lane][op.index] = u32(value)
        elif op.reg_type == RegType.SGPR:
            self.sgprs[op.index] = u32(value)
        else:
            raise TypeError("Cannot write to literal operand")


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def step(self, instr: VDot4U32U8Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes

        # Atomic read phase across all lanes
        s0_vals = [state.read_lane(instr.src0, lane) for lane in range(state.num_lanes)]
        s1_vals = [state.read_lane(instr.src1, lane) for lane in range(state.num_lanes)]
        s2_vals = [state.read_lane(instr.src2, lane) for lane in range(state.num_lanes)]

        # Execution and selective write-back phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = ref_dot4_u32_u8(s0_vals[lane], s1_vals[lane], s2_vals[lane])
                state.write_lane(instr.dst, lane, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry Generation
# ==============================================================================

@dataclass(frozen=True)
class Dot4StatusFlags:
    """Architectural condition codes and status flags for v_dot4_u32_u8."""
    carry: bool
    zero: bool
    dot_zero: bool
    max_dot: bool
    active_terms: int
    dot_sum: int
    full_sum: int


def evaluate_dot4_flags(s0: int, s1: int, s2: int = 0) -> Dot4StatusFlags:
    """Evaluate architectural condition codes and status flags for v_dot4_u32_u8."""
    b = unpack_u8(s0)
    c = unpack_u8(s1)
    prods = [(b[i] * c[i]) for i in range(4)]
    dot = sum(prods)
    acc = s2 & MASK32
    full_sum = acc + dot
    carry = bool(full_sum >= (1 << 32))
    res = full_sum & MASK32
    active = sum(1 for p in prods if p > 0)

    return Dot4StatusFlags(
        carry=carry,
        zero=(res == 0),
        dot_zero=(dot == 0),
        max_dot=(dot == MAX_DOT4_SUM),
        active_terms=active,
        dot_sum=dot,
        full_sum=full_sum,
    )


# ==============================================================================
# Unit Tests: Model and Reference Oracle
# ==============================================================================

class TestValuDot4U32U8Oracle(unittest.TestCase):
    """Unit tests for AMD RDNA2 v_dot4_u32_u8 architectural bit model and reference oracle."""

    def test_known_vectors_basic(self) -> None:
        # 4*8 + 3*7 + 2*6 + 1*5 = 32 + 21 + 12 + 5 = 70
        self.assertEqual(ref_dot4_u32_u8(0x01020304, 0x05060708, 0), 70)
        # With accumulator: 70 + 1000 = 1070
        self.assertEqual(ref_dot4_u32_u8(0x01020304, 0x05060708, 1000), 1070)

    def test_zero_identities(self) -> None:
        # Zero operand in s0 or s1 results in the accumulator value
        test_words = [0x01020304, 0xFFFFFFFF, 0x80808080, 0x12345678]
        for w in test_words:
            for acc in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
                with self.subTest(w=hex(w), acc=hex(acc)):
                    self.assertEqual(ref_dot4_u32_u8(0, w, acc), acc & MASK32)
                    self.assertEqual(ref_dot4_u32_u8(w, 0, acc), acc & MASK32)

    def test_commutativity_of_multipliers(self) -> None:
        # s0 . s1 == s1 . s0
        test_pairs = [
            (0x01020304, 0x05060708),
            (0xFFFFFFFF, 0x12345678),
            (0x80808080, 0x7F7F7F7F),
            (0xA1B2C3D4, 0x1E2D3C4B),
            (0x00FF00FF, 0xFF00FF00),
        ]
        for s0, s1 in test_pairs:
            for s2 in (0, 100, 0x12345678, 0xFFFFFFFF):
                with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                    self.assertEqual(
                        ref_dot4_u32_u8(s0, s1, s2),
                        ref_dot4_u32_u8(s1, s0, s2),
                    )

    def test_identity_broadcast_sums_bytes(self) -> None:
        # Multiplying with 0x01010101 computes the sum of the four unsigned bytes
        test_words = [
            (0x01020304, 1 + 2 + 3 + 4),
            (0xFFFFFFFF, 255 * 4),
            (0x10203040, 0x10 + 0x20 + 0x30 + 0x40),
            (0x000000FF, 255),
            (0x0000FF00, 255),
            (0x00FF0000, 255),
            (0xFF000000, 255),
        ]
        for word, expected_sum in test_words:
            with self.subTest(word=hex(word)):
                self.assertEqual(ref_dot4_u32_u8(word, 0x01010101, 0), expected_sum)
                self.assertEqual(ref_dot4_u32_u8(0x01010101, word, 0), expected_sum)

    def test_byte_lane_isolation(self) -> None:
        # Each byte position operates independently without carry bleed into neighbor bytes
        for byte_idx in range(4):
            for val_a in (1, 15, 128, 255):
                for val_b in (1, 10, 100, 255):
                    s0 = val_a << (8 * byte_idx)
                    s1 = val_b << (8 * byte_idx)
                    expected = val_a * val_b
                    with self.subTest(lane=byte_idx, a=val_a, b=val_b):
                        self.assertEqual(ref_dot4_u32_u8(s0, s1, 0), expected)

    def test_maximum_unsigned_product(self) -> None:
        # 4 * (255 * 255) = 260100 (0x3F804)
        self.assertEqual(ref_dot4_u32_u8(0xFFFFFFFF, 0xFFFFFFFF, 0), MAX_DOT4_SUM)
        self.assertEqual(MAX_DOT4_SUM, 260100)

    def test_accumulator_wrapping_modulo_32(self) -> None:
        # Wrap boundary: 260100 + (0xFFFFFFFF - 260100 + 1) = 0 mod 2^32
        acc = (MASK32 - MAX_DOT4_SUM + 1) & MASK32
        lo, carry = ref_dot4_u32_u8_full(0xFFFFFFFF, 0xFFFFFFFF, acc)
        self.assertEqual(lo, 0)
        self.assertEqual(carry, 1)

        # Non-zero remainder wrap
        lo2, carry2 = ref_dot4_u32_u8_full(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(lo2, 260099)
        self.assertEqual(carry2, 1)

    def test_unsigned_vs_signed_divergence(self) -> None:
        # When high bit is set (e.g. 0xFF), unsigned treats it as 255, whereas signed would treat it as -1
        # 0x000000FF * 0x00000001: unsigned is 255, signed would be -1 (0xFFFFFFFF)
        self.assertEqual(ref_dot4_u32_u8(0x000000FF, 0x00000001, 0), 255)
        # 0x000000FF * 0x000000FF: unsigned is 65025, signed would be (-1)*(-1) = 1
        self.assertEqual(ref_dot4_u32_u8(0x000000FF, 0x000000FF, 0), 65025)


class TestValuDot4U32U8SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_operand_creation_and_disassembly(self) -> None:
        v0 = Operand.vgpr(0)
        v255 = Operand.vgpr(255)
        s10 = Operand.sgpr(10)
        lit = Operand.literal(0x12345678)

        self.assertEqual(v0.disassemble(), "v0")
        self.assertEqual(v255.disassemble(), "v255")
        self.assertEqual(s10.disassemble(), "s10")
        self.assertEqual(lit.disassemble(), "0x12345678")

    def test_operand_bounds_validation(self) -> None:
        with self.assertRaises(ValueError):
            Operand.vgpr(-1)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(-1)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

    def test_instruction_construction_and_disassembly(self) -> None:
        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(5),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(2),
            src2=Operand.literal(100),
        )
        self.assertEqual(instr.disassemble(), "v_dot4_u32_u8 v5, v1, s2, 0x00000064")

    def test_instruction_validation_rejections(self) -> None:
        # dst cannot be SGPR or literal
        with self.assertRaises(TypeError):
            VDot4U32U8Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.literal(0),
            )
        with self.assertRaises(TypeError):
            VDot4U32U8Instruction(
                dst=Operand.literal(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.literal(0),
            )
        # clamp is unsupported
        with self.assertRaises(ValueError):
            VDot4U32U8Instruction(
                dst=Operand.vgpr(0),
                src0=Operand.vgpr(1),
                src1=Operand.vgpr(2),
                src2=Operand.literal(0),
                clamp=True,
            )


class TestValuDot4U32U8ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.regs = ScalarRegisterFile()
        self.interp = ScalarVALUInterpreter(self.regs)

    def test_scalar_register_file_read_write(self) -> None:
        self.regs.write(Operand.vgpr(5), 0x12345678)
        self.assertEqual(self.regs.read(Operand.vgpr(5)), 0x12345678)

        self.regs.write(Operand.sgpr(2), 0x9ABCDEF0)
        self.assertEqual(self.regs.read(Operand.sgpr(2)), 0x9ABCDEF0)

        # 32-bit truncation
        self.regs.write(Operand.vgpr(0), 0x1_FFFF_FFFF)
        self.assertEqual(self.regs.read(Operand.vgpr(0)), 0xFFFF_FFFF)

        # Literal read
        self.assertEqual(self.regs.read(Operand.literal(42)), 42)

    def test_scalar_execution_all_vgprs(self) -> None:
        # 0x01020304 . 0x05060708 + 1000 = 1070
        self.regs.write(Operand.vgpr(1), 0x01020304)
        self.regs.write(Operand.vgpr(2), 0x05060708)
        self.regs.write(Operand.vgpr(3), 1000)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        res = self.interp.step(instr)
        self.assertEqual(res, 1070)
        self.assertEqual(self.regs.read(Operand.vgpr(0)), 1070)

    def test_scalar_execution_sgpr_and_literal(self) -> None:
        self.regs.write(Operand.vgpr(1), 0x01020304)
        self.regs.write(Operand.sgpr(4), 0x05060708)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.sgpr(4),
            src2=Operand.literal(2000),
        )

        res = self.interp.step(instr)
        self.assertEqual(res, 2070)
        self.assertEqual(self.regs.read(Operand.vgpr(0)), 2070)

    def test_scalar_sequential_program(self) -> None:
        # Step 1: compute dot product into v0
        self.regs.write(Operand.vgpr(1), 0x01020304)
        self.regs.write(Operand.vgpr(2), 0x05060708)
        self.regs.write(Operand.vgpr(3), 0)

        instr1 = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res1 = self.interp.step(instr1)
        self.assertEqual(res1, 70)

        # Step 2: accumulate v0 into v4
        instr2 = VDot4U32U8Instruction(
            dst=Operand.vgpr(4),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        res2 = self.interp.step(instr2)
        self.assertEqual(res2, 140)
        self.assertEqual(self.regs.read(Operand.vgpr(4)), 140)


class TestValuDot4U32U8SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr(1), lane, 0x01020304)
            state.write_lane(Operand.vgpr(2), lane, 0x05060708)
            state.write_lane(Operand.vgpr(3), lane, lane * 10)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = 70 + lane * 10
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.read_lane(Operand.vgpr(0), lane), expected)

    def test_wave64_all_lanes_active(self) -> None:
        state = SIMDState(num_lanes=64)
        interp = SIMDInterpreter()

        for lane in range(64):
            # Lower 32 lanes use 0x01020304 . 0x05060708 (70), upper 32 lanes use 0xFFFFFFFF . 0x01010101 (1020)
            s0 = 0x01020304 if lane < 32 else 0xFFFFFFFF
            s1 = 0x05060708 if lane < 32 else 0x01010101
            state.write_lane(Operand.vgpr(1), lane, s0)
            state.write_lane(Operand.vgpr(2), lane, s1)
            state.write_lane(Operand.vgpr(3), lane, 0)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = interp.step(instr, state)
        self.assertEqual(len(results), 64)
        for lane in range(32):
            self.assertEqual(results[lane], 70)
        for lane in range(32, 64):
            self.assertEqual(results[lane], 1020)

    def test_partial_exec_mask_even_lanes(self) -> None:
        sentinel = 0xDEADBEEF
        state = SIMDState(num_lanes=32, exec_mask=0x55555555)  # only even lanes active
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr(0), lane, sentinel)
            state.write_lane(Operand.vgpr(1), lane, 0x01020304)
            state.write_lane(Operand.vgpr(2), lane, 0x05060708)
            state.write_lane(Operand.vgpr(3), lane, 0)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 70)
                self.assertEqual(state.read_lane(Operand.vgpr(0), lane), 70)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.read_lane(Operand.vgpr(0), lane), sentinel)

    def test_exec_mask_zero_no_lanes_modified(self) -> None:
        sentinel = 0xCAFEBABE
        state = SIMDState(num_lanes=32, exec_mask=0)
        interp = SIMDInterpreter()

        for lane in range(32):
            state.write_lane(Operand.vgpr(0), lane, sentinel)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )

        results = interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.read_lane(Operand.vgpr(0), lane), sentinel)

    def test_divergent_operands_across_lanes(self) -> None:
        state = SIMDState(num_lanes=32)
        interp = SIMDInterpreter()

        # Shared SGPR src0, lane-private VGPR src1
        state.write_lane(Operand.sgpr(2), 0, 0x01010101)
        for lane in range(32):
            val = pack_u8(lane, lane + 1, lane + 2, lane + 3)
            state.write_lane(Operand.vgpr(1), lane, val)
            state.write_lane(Operand.vgpr(2), lane, 5)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.sgpr(2),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        results = interp.step(instr, state)
        for lane in range(32):
            expected_sum = (lane + (lane + 1) + (lane + 2) + (lane + 3)) + 5
            self.assertEqual(results[lane], expected_sum)


class TestValuDot4U32U8RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src2_accumulator(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # In-place accumulation: v0 = v1 . v2 + v0
        regs.write(Operand.vgpr(0), 1000)
        regs.write(Operand.vgpr(1), 0x01020304)
        regs.write(Operand.vgpr(2), 0x05060708)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),  # aliased
        )

        res = interp.step(instr)
        self.assertEqual(res, 1070)
        self.assertEqual(regs.read(Operand.vgpr(0)), 1070)
        # src0 and src1 remain uncorrupted
        self.assertEqual(regs.read(Operand.vgpr(1)), 0x01020304)
        self.assertEqual(regs.read(Operand.vgpr(2)), 0x05060708)

    def test_dst_aliases_src0(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # v0 = v0 . v1 + v2
        regs.write(Operand.vgpr(0), 0x01020304)
        regs.write(Operand.vgpr(1), 0x05060708)
        regs.write(Operand.vgpr(2), 100)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),  # aliased with dst
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )

        res = interp.step(instr)
        self.assertEqual(res, 170)
        self.assertEqual(regs.read(Operand.vgpr(0)), 170)

    def test_dst_aliases_src1(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # v0 = v1 . v0 + v2
        regs.write(Operand.vgpr(1), 0x01020304)
        regs.write(Operand.vgpr(0), 0x05060708)
        regs.write(Operand.vgpr(2), 50)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(0),  # aliased with dst
            src2=Operand.vgpr(2),
        )

        res = interp.step(instr)
        self.assertEqual(res, 120)
        self.assertEqual(regs.read(Operand.vgpr(0)), 120)

    def test_dst_aliases_all_operands(self) -> None:
        regs = ScalarRegisterFile()
        interp = ScalarVALUInterpreter(regs)

        # v0 = v0 . v0 + v0
        # For v0 = 0x01010101: 4 * (1 * 1) = 4. 4 + 0x01010101 = 0x01010105
        regs.write(Operand.vgpr(0), 0x01010101)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )

        res = interp.step(instr)
        self.assertEqual(res, 0x01010105)
        self.assertEqual(regs.read(Operand.vgpr(0)), 0x01010105)

    def test_simd_aliased_in_place_preservation_with_exec_mask(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x0000FFFF)  # lanes 0..15 active
        interp = SIMDInterpreter()

        initial_acc = 5000
        for lane in range(32):
            state.write_lane(Operand.vgpr(0), lane, initial_acc)
            state.write_lane(Operand.vgpr(1), lane, 0x01020304)
            state.write_lane(Operand.vgpr(2), lane, 0x05060708)

        instr = VDot4U32U8Instruction(
            dst=Operand.vgpr(0),  # aliased with src2 accumulator
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )

        interp.step(instr, state)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(state.read_lane(Operand.vgpr(0), lane), initial_acc + 70)
            else:
                self.assertEqual(state.read_lane(Operand.vgpr(0), lane), initial_acc)


class TestValuDot4U32U8StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry generation."""

    def test_carry_flag_exact_boundary(self) -> None:
        # acc = 0xFFFFFFFF, dot = 1 -> full_sum = 0x100000000 -> carry = True, zero = True
        flags = evaluate_dot4_flags(s0=0x00000001, s1=0x00000001, s2=0xFFFF_FFFF)
        self.assertTrue(flags.carry)
        self.assertTrue(flags.zero)
        self.assertEqual(flags.full_sum, 0x1_0000_0000)
        self.assertEqual(flags.dot_sum, 1)

        # acc = 0xFFFFFFFF, dot = 0 -> full_sum = 0xFFFFFFFF -> carry = False, zero = False
        flags_no_carry = evaluate_dot4_flags(s0=0, s1=0, s2=0xFFFF_FFFF)
        self.assertFalse(flags_no_carry.carry)
        self.assertFalse(flags_no_carry.zero)
        self.assertEqual(flags_no_carry.full_sum, 0xFFFF_FFFF)
        self.assertEqual(flags_no_carry.dot_sum, 0)

    def test_max_dot_condition(self) -> None:
        flags = evaluate_dot4_flags(0xFFFFFFFF, 0xFFFFFFFF, 0)
        self.assertTrue(flags.max_dot)
        self.assertEqual(flags.dot_sum, 260100)
        self.assertEqual(flags.active_terms, 4)
        self.assertFalse(flags.carry)
        self.assertFalse(flags.zero)

    def test_all_zero_condition(self) -> None:
        flags = evaluate_dot4_flags(0, 0, 0)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.dot_zero)
        self.assertFalse(flags.carry)
        self.assertFalse(flags.max_dot)
        self.assertEqual(flags.active_terms, 0)

    def test_active_terms_counting(self) -> None:
        # Term 0 only
        f1 = evaluate_dot4_flags(0x00000001, 0x00000001, 0)
        self.assertEqual(f1.active_terms, 1)

        # Terms 0 and 2
        f2 = evaluate_dot4_flags(0x00010001, 0x00010001, 0)
        self.assertEqual(f2.active_terms, 2)

        # Terms 0, 1, 2
        f3 = evaluate_dot4_flags(0x00010101, 0x00010101, 0)
        self.assertEqual(f3.active_terms, 3)

        # All 4 terms
        f4 = evaluate_dot4_flags(0x01010101, 0x01010101, 0)
        self.assertEqual(f4.active_terms, 4)

    def test_carry_without_zero(self) -> None:
        # acc = 0xFFFFFFFF, dot = 2 -> full_sum = 0x100000001 -> carry = True, zero = False (res = 1)
        flags = evaluate_dot4_flags(s0=0x00000002, s1=0x00000001, s2=0xFFFF_FFFF)
        self.assertTrue(flags.carry)
        self.assertFalse(flags.zero)
        self.assertEqual(flags.full_sum, 0x1_0000_0001)


class TestValuDot4U32U8IEEE754Boundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit pattern boundaries, subnormals, and NaNs as raw integer bytes."""

    def test_float_signed_zeros_distinction(self) -> None:
        # +0.0 is 0x00000000, -0.0 is 0x80000000 (byte 3 is 0x80 = 128)
        # Dot with 0x01000000: byte 3 is 1
        # For +0.0: 0 * 1 = 0
        # For -0.0: 128 * 1 = 128
        self.assertEqual(ref_dot4_u32_u8(0x00000000, 0x01000000, 0), 0)
        self.assertEqual(ref_dot4_u32_u8(0x80000000, 0x01000000, 0), 128)

        # Dot of -0.0 with -0.0: 128 * 128 = 16384 (0x4000)
        self.assertEqual(ref_dot4_u32_u8(0x80000000, 0x80000000, 0), 16384)

    def test_subnormal_floats_no_flush_to_zero(self) -> None:
        # Subnormal float 1: 0x00000001 (byte 0 = 1)
        # Subnormal float 2: 0x00000002 (byte 0 = 2)
        # If hardware erroneously flushed subnormals to zero (FTZ),
        # 1 * 2 would evaluate to 0 instead of 2!
        self.assertEqual(ref_dot4_u32_u8(0x00000001, 0x00000002, 0), 2)

        # Subnormal float with byte 1: 0x00000100 (byte 1 = 1)
        self.assertEqual(ref_dot4_u32_u8(0x00000100, 0x00000200, 0), 2)

    def test_nan_payloads_as_raw_bytes(self) -> None:
        # qNaN: 0x7FC00000 -> bytes [0, 0, 0xC0 (192), 0x7F (127)]
        # sNaN: 0x7F800001 -> bytes [1, 0, 0x80 (128), 0x7F (127)]
        # Product:
        # Byte 0: 0 * 1 = 0
        # Byte 1: 0 * 0 = 0
        # Byte 2: 192 * 128 = 24576
        # Byte 3: 127 * 127 = 16129
        # Sum = 24576 + 16129 = 40705
        expected = 24576 + 16129
        self.assertEqual(ref_dot4_u32_u8(0x7FC00000, 0x7F800001, 0), expected)

    def test_float_infinities(self) -> None:
        # +inf: 0x7F800000 -> bytes [0, 0, 128, 127]
        # -inf: 0xFF800000 -> bytes [0, 0, 128, 255]
        # Byte 2: 128 * 128 = 16384
        # Byte 3: 127 * 255 = 32385
        # Sum = 16384 + 32385 = 48769
        expected = 16384 + 32385
        self.assertEqual(ref_dot4_u32_u8(0x7F800000, 0xFF800000, 0), expected)

    def test_byte_extremes_boundary_matrix(self) -> None:
        extremes = [0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF]
        for b0 in extremes:
            for b1 in extremes:
                w0 = pack_u8(b0, b0, b0, b0)
                w1 = pack_u8(b1, b1, b1, b1)
                expected = 4 * (b0 * b1)
                self.assertEqual(ref_dot4_u32_u8(w0, w1, 0), expected)


if __name__ == "__main__":
    unittest.main()
