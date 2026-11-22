from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_src2_f64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Float Atomic Decrement:
Performs a 64-bit IEEE 754 double-precision atomic wrapping decrement on LDS
memory location A using the secondary LDS memory location B as the wrap threshold.

Syntax:
    ds_dec_src2_f64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    if isnan(val_a) or val_a <= 0.0 or val_a > val_b:
        new_val_a = val_b
    else:
        new_val_a = val_a - 1.0
    LDS64[qword_idx_a] = new_val_a

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Addressing: Base byte address in VGPR vaddr + 8-bit unsigned immediate offsets (* 8 bytes).
  - Alignment: 64-bit qword aligned; low 3 bits are dropped (eff_addr >> 3).
  - Wrap Condition:
      When float val_a <= 0.0 or val_a > val_b (or NaN), memory location A wraps to val_b.
      Otherwise, memory location A is decremented by 1.0 (binary64).
  - Non-returning:
      Vector general-purpose registers (VGPRs) are not modified.
"""

import math
import struct
import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit unsigned mask
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET8_MASK: int = 0xFF            # 8-bit immediate byte offset mask
DEFAULT_LDS_QWORDS: int = 8192      # 64 KiB capacity in 64-bit qwords (8192 * 8 bytes)
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256        # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106        # Standard GFX10 scalar register file size
DEFAULT_NUM_LANES: int = 64         # Wave64 default width


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def bits_to_f64(bits: int) -> float:
    """Interpret 64-bit unsigned integer as IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Pack IEEE 754 double-precision float into 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision NaN."""
    return (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    """Test if 64-bit bit pattern is an IEEE 754 double-precision subnormal float."""
    return (bits & 0x7FF0_0000_0000_0000) == 0 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def calc_lds_src2_f64_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_qwords: int = DEFAULT_LDS_QWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and qword indices for location A and location B."""
    byte_addr_a = u32(vaddr + ((offset0 & OFFSET8_MASK) * 8))
    byte_addr_b = u32(vaddr + ((offset1 & OFFSET8_MASK) * 8))
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    return byte_addr_a, byte_addr_b, qword_idx_a, qword_idx_b


def ref_ds_dec_src2_f64(bits_a: int, bits_b: int) -> int:
    """Pure mathematical reference oracle for ds_dec_src2_f64.

    If float at A is NaN, <= 0.0, or > threshold at B: reset to threshold bits_b.
    Otherwise: decrement float by 1.0.
    """
    u_a = u64(bits_a)
    u_b = u64(bits_b)

    if is_f64_nan(u_a) or is_f64_nan(u_b):
        return u_b

    f_a = bits_to_f64(u_a)
    f_b = bits_to_f64(u_b)

    if f_a <= 0.0 or f_a > f_b:
        return u_b

    f_res = f_a - 1.0
    return f64_to_bits(f_res)


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, qword_idx: int) -> int:
        idx = qword_idx & (self.capacity_qwords - 1)
        return self.memory[idx]

    def write_qword(self, qword_idx: int, val: int) -> None:
        idx = qword_idx & (self.capacity_qwords - 1)
        self.memory[idx] = u64(val)

    def atomic_dec_src2_f64(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        _, _, idx_a, idx_b = calc_lds_src2_f64_addresses(
            vaddr=vaddr,
            offset0=offset0,
            offset1=offset1,
            lds_qwords=self.capacity_qwords,
        )
        val_a = self.read_qword(idx_a)
        val_b = self.read_qword(idx_b)
        new_val_a = ref_ds_dec_src2_f64(val_a, val_b)
        self.write_qword(idx_a, new_val_a)
        return new_val_a


class TestReferenceOracleAndLDSModel64(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_decrement(self) -> None:
        b5 = f64_to_bits(5.0)
        b4 = f64_to_bits(4.0)
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(b5, b10), b4)

        b1 = f64_to_bits(1.0)
        b0 = f64_to_bits(0.0)
        self.assertEqual(ref_ds_dec_src2_f64(b1, b10), b0)

    def test_oracle_wrap_at_zero(self) -> None:
        b0 = f64_to_bits(0.0)
        b_neg0 = f64_to_bits(-0.0)
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(b0, b10), b10)
        self.assertEqual(ref_ds_dec_src2_f64(b_neg0, b10), b10)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        b15 = f64_to_bits(15.0)
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(b15, b10), b10)

    def test_oracle_negative_numbers_wrap(self) -> None:
        b_neg5 = f64_to_bits(-5.0)
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(b_neg5, b10), b10)

    def test_oracle_nan_handling(self) -> None:
        qnan = 0x7FF8_0000_0000_0000
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(qnan, b10), b10)

    def test_address_calculation_and_stride(self) -> None:
        base = 0x1000
        off0 = 3  # 3 * 8 = 24 = 0x18
        off1 = 6  # 6 * 8 = 48 = 0x30
        addr_a, addr_b, idx_a, idx_b = calc_lds_src2_f64_addresses(base, off0, off1)
        self.assertEqual(addr_a, 0x1018)
        self.assertEqual(addr_b, 0x1030)
        self.assertEqual(idx_a, 0x1018 >> 3)
        self.assertEqual(idx_b, 0x1030 >> 3)

    def test_lds_model_atomic_dec_src2_f64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        lds.write_qword(idx_a, f64_to_bits(3.0))
        lds.write_qword(idx_b, f64_to_bits(10.0))

        res = lds.atomic_dec_src2_f64(vaddr, offset0=0, offset1=1)
        self.assertEqual(res, f64_to_bits(2.0))
        self.assertEqual(lds.read_qword(idx_a), f64_to_bits(2.0))
        self.assertEqual(lds.read_qword(idx_b), f64_to_bits(10.0))


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

import dataclasses
import enum


class RegKind(enum.Enum):
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= DEFAULT_NUM_VGPRS:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > DEFAULT_NUM_VGPRS:
            raise ValueError(f"Register range exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count > 1:
            return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"
        return f"{self.kind.value}{self.index}"


class DsSrc2F64Opcode(enum.Enum):
    DS_DEC_SRC2_F64 = "ds_dec_src2_f64"
    DS_INC_SRC2_F64 = "ds_inc_src2_f64"


@dataclasses.dataclass(frozen=True)
class DsSrc2F64Instruction:
    opcode: DsSrc2F64Opcode
    vaddr: Register
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR: {self.vaddr}")
        if self.offset0 < 0 or self.offset0 > 255:
            raise ValueError(f"offset0 must be 8-bit unsigned [0, 255], got {self.offset0}")
        if self.offset1 < 0 or self.offset1 > 255:
            raise ValueError(f"offset1 must be 8-bit unsigned [0, 255], got {self.offset1}")

    def to_assembly(self) -> str:
        parts: list[str] = [self.opcode.value, self.vaddr.format_asm()]
        if self.offset0 != 0:
            parts.append(f"offset0:{self.offset0}")
        if self.offset1 != 0:
            parts.append(f"offset1:{self.offset1}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


class DsDecSrc2F64Expr:
    def __init__(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> None:
        self.vaddr = vaddr & MASK32
        self.offset0 = offset0 & OFFSET8_MASK
        self.offset1 = offset1 & OFFSET8_MASK

    def evaluate(self, lds: LDSModel64) -> int:
        return lds.atomic_dec_src2_f64(self.vaddr, self.offset0, self.offset1)


class TestDsDecSrc2F64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_values(self) -> None:
        self.assertEqual(DsSrc2F64Opcode.DS_DEC_SRC2_F64.value, "ds_dec_src2_f64")
        self.assertEqual(DsSrc2F64Opcode.DS_INC_SRC2_F64.value, "ds_inc_src2_f64")

    def test_register_operand_validation(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        self.assertEqual(v0.format_asm(), "v0")
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)

    def test_instruction_creation_and_assembly(self) -> None:
        v1 = Register(RegKind.VGPR, 1)
        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=v1,
            offset0=1,
            offset1=2,
            gds=True,
        )
        self.assertEqual(inst.to_assembly(), "ds_dec_src2_f64 v1 offset0:1 offset1:2 gds")

    def test_instruction_validation_errors(self) -> None:
        s0 = Register(RegKind.SGPR, 0)
        v0 = Register(RegKind.VGPR, 0)
        with self.assertRaises(ValueError):
            DsSrc2F64Instruction(opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64, vaddr=s0)
        with self.assertRaises(ValueError):
            DsSrc2F64Instruction(opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64, vaddr=v0, offset0=300)

    def test_typed_expression_evaluation(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x200
        idx_a = 0x200 >> 3
        idx_b = (0x200 + 8) >> 3

        lds.write_qword(idx_a, f64_to_bits(8.0))
        lds.write_qword(idx_b, f64_to_bits(12.0))

        expr = DsDecSrc2F64Expr(vaddr=vaddr, offset0=0, offset1=1)
        res = expr.evaluate(lds)
        self.assertEqual(res, f64_to_bits(7.0))


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    def __init__(self) -> None:
        self.vgprs = [0] * DEFAULT_NUM_VGPRS
        self.sgprs = [0] * DEFAULT_NUM_SGPRS

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarLDS64Interpreter:
    def __init__(self, lds: LDSModel64, rf: ScalarRegisterFile | None = None) -> None:
        self.lds = lds
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def step(self, inst: DsSrc2F64Instruction) -> int:
        vaddr = self.rf.read_vgpr(inst.vaddr.index)
        new_val_a = self.lds.atomic_dec_src2_f64(
            vaddr=vaddr,
            offset0=inst.offset0,
            offset1=inst.offset1,
        )
        return new_val_a


class TestDsDecSrc2F64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSModel64(capacity_qwords=512)
        self.rf = ScalarRegisterFile()
        self.interp = ScalarLDS64Interpreter(self.lds, self.rf)

    def test_scalar_step_normal_decrement(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, f64_to_bits(10.0))
        self.lds.write_qword(idx_b, f64_to_bits(20.0))
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, f64_to_bits(9.0))
        self.assertEqual(self.lds.read_qword(idx_a), f64_to_bits(9.0))
        self.assertEqual(self.lds.read_qword(idx_b), f64_to_bits(20.0))

    def test_scalar_step_wrap_at_zero(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, f64_to_bits(0.0))
        self.lds.write_qword(idx_b, f64_to_bits(15.0))
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, f64_to_bits(15.0))
        self.assertEqual(self.lds.read_qword(idx_a), f64_to_bits(15.0))

    def test_scalar_step_reset_on_greater_threshold(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, f64_to_bits(50.0))
        self.lds.write_qword(idx_b, f64_to_bits(10.0))
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, f64_to_bits(10.0))

    def test_scalar_step_with_offsets(self) -> None:
        vaddr = 0x100
        idx_a = (0x100 + 16) >> 3
        idx_b = (0x100 + 32) >> 3

        self.lds.write_qword(idx_a, f64_to_bits(2.0))
        self.lds.write_qword(idx_b, f64_to_bits(5.0))
        self.rf.write_vgpr(1, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 1),
            offset0=2,
            offset1=4,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, f64_to_bits(1.0))
        self.assertEqual(self.lds.read_qword(idx_a), f64_to_bits(1.0))

    def test_scalar_step_non_returning_preserves_rf(self) -> None:
        vaddr = 0x40
        idx_a = 0x40 >> 3
        idx_b = (0x40 + 8) >> 3

        self.lds.write_qword(idx_a, f64_to_bits(3.0))
        self.lds.write_qword(idx_b, f64_to_bits(10.0))

        for i in range(10):
            self.rf.write_vgpr(i, 0x500 + i)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        self.interp.step(inst)
        self.assertEqual(self.rf.read_vgpr(0), vaddr)
        for i in range(1, 10):
            self.assertEqual(self.rf.read_vgpr(i), 0x500 + i)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDLDS64WavefrontInterpreter:
    def __init__(self, wave_size: int = 64, lds: LDSModel64 | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.lds = lds if lds is not None else LDSModel64()
        self.vgprs = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self._exec_mask = mask & limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    def set_exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    def set_exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def read_lane_vgpr(self, lane: int, idx: int) -> int:
        return self.vgprs[lane][idx]

    def write_lane_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK32

    def step(self, inst: DsSrc2F64Instruction) -> list[int | None]:
        results: list[int | None] = []
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                vaddr = self.read_lane_vgpr(lane, inst.vaddr.index)
                new_val_a = self.lds.atomic_dec_src2_f64(
                    vaddr=vaddr,
                    offset0=inst.offset0,
                    offset1=inst.offset1,
                )
                results.append(new_val_a)
            else:
                results.append(None)
        return results


class TestDsDecSrc2F64SIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave64_full_exec_independent_lanes(self) -> None:
        lds = LDSModel64(capacity_qwords=2048)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)

        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, f64_to_bits(float(10 + lane)))
            lds.write_qword(idx_b, f64_to_bits(100.0))
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            expected = f64_to_bits(float(9 + lane))
            self.assertEqual(results[lane], expected)
            self.assertEqual(lds.read_qword(idx_a), expected)

    def test_wave32_full_exec(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)

        for lane in range(32):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, f64_to_bits(5.0))
            lds.write_qword(idx_b, f64_to_bits(10.0))
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], f64_to_bits(4.0))

    def test_exec_mask_disabled_lanes_unmodified(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        simd.set_exec_mask(0x5555_5555_5555_5555)

        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, f64_to_bits(50.0))
            lds.write_qword(idx_b, f64_to_bits(100.0))
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            if lane % 2 == 0:
                self.assertEqual(results[lane], f64_to_bits(49.0))
                self.assertEqual(lds.read_qword(idx_a), f64_to_bits(49.0))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(lds.read_qword(idx_a), f64_to_bits(50.0))

    def test_multi_lane_serialization_same_target_a(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0xF)

        common_addr = 0x80
        idx_a = (common_addr + 0) >> 3
        idx_b = (common_addr + 8) >> 3
        lds.write_qword(idx_a, f64_to_bits(10.0))
        lds.write_qword(idx_b, f64_to_bits(20.0))

        for lane in range(4):
            simd.write_lane_vgpr(lane, 0, common_addr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(results[0], f64_to_bits(9.0))
        self.assertEqual(results[1], f64_to_bits(8.0))
        self.assertEqual(results[2], f64_to_bits(7.0))
        self.assertEqual(results[3], f64_to_bits(6.0))
        self.assertEqual(lds.read_qword(idx_a), f64_to_bits(6.0))


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestDsDecSrc2F64RegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_vaddr_register_preservation_scalar(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        vaddr = 0x100
        idx_a = 0x100 >> 3
        idx_b = (0x100 + 8) >> 3
        lds.write_qword(idx_a, f64_to_bits(5.0))
        lds.write_qword(idx_b, f64_to_bits(10.0))

        rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = interp.step(inst)
        self.assertEqual(res, f64_to_bits(4.0))
        self.assertEqual(rf.read_vgpr(0), vaddr)

    def test_identical_offsets_location_a_equals_location_b(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        vaddr = 0x200
        target_idx = (vaddr + 2 * 8) >> 3
        lds.write_qword(target_idx, f64_to_bits(42.0))
        rf.write_vgpr(0, vaddr)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=2,
            offset1=2,
        )

        res = interp.step(inst)
        self.assertEqual(res, f64_to_bits(41.0))
        self.assertEqual(lds.read_qword(target_idx), f64_to_bits(41.0))

        lds.write_qword(target_idx, f64_to_bits(0.0))
        res_zero = interp.step(inst)
        self.assertEqual(res_zero, f64_to_bits(0.0))

    def test_interlaced_memory_locations_across_divergent_lanes(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0x3)

        idx_0 = 0x100 >> 3
        idx_1 = 0x108 >> 3
        idx_2 = 0x110 >> 3

        lds.write_qword(idx_0, f64_to_bits(10.0))
        lds.write_qword(idx_1, f64_to_bits(20.0))
        lds.write_qword(idx_2, f64_to_bits(30.0))

        simd.write_lane_vgpr(0, 0, 0x100)
        simd.write_lane_vgpr(1, 0, 0x108)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(results[0], f64_to_bits(9.0))
        self.assertEqual(results[1], f64_to_bits(19.0))
        self.assertEqual(simd.read_lane_vgpr(0, 0), 0x100)
        self.assertEqual(simd.read_lane_vgpr(1, 0), 0x108)


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class AtomicDecSrc2F64StatusFlags:
    """Architectural condition codes and status flags for ds_dec_src2_f64."""
    wrapped: bool
    zero: bool
    negative: bool
    borrow: bool
    threshold_reset: bool
    nan_flag: bool
    unaligned_addr: bool


def evaluate_ds_dec_src2_f64_flags(
    val_a: int,
    val_b: int,
    eff_addr: int,
) -> AtomicDecSrc2F64StatusFlags:
    u_a = u64(val_a)
    u_b = u64(val_b)

    nan_flag = is_f64_nan(u_a) or is_f64_nan(u_b)
    if nan_flag:
        return AtomicDecSrc2F64StatusFlags(
            wrapped=True,
            zero=False,
            negative=False,
            borrow=False,
            threshold_reset=False,
            nan_flag=True,
            unaligned_addr=bool(eff_addr & 0x7),
        )

    f_a = bits_to_f64(u_a)
    f_b = bits_to_f64(u_b)

    wrapped = (f_a <= 0.0 or f_a > f_b)
    threshold_reset = (f_a > f_b)
    borrow = (f_a <= 0.0)

    new_val_bits = u_b if wrapped else f64_to_bits(f_a - 1.0)
    zero = (bits_to_f64(new_val_bits) == 0.0)
    negative = bool(new_val_bits & 0x8000_0000_0000_0000)
    unaligned_addr = bool(eff_addr & 0x7)

    return AtomicDecSrc2F64StatusFlags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        borrow=borrow,
        threshold_reset=threshold_reset,
        nan_flag=False,
        unaligned_addr=unaligned_addr,
    )


class TestDsDecSrc2F64ConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_decrement_flags(self) -> None:
        flags = evaluate_ds_dec_src2_f64_flags(
            val_a=f64_to_bits(10.0),
            val_b=f64_to_bits(20.0),
            eff_addr=0x100,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.threshold_reset)
        self.assertFalse(flags.nan_flag)
        self.assertFalse(flags.unaligned_addr)

    def test_zero_wrap_flags(self) -> None:
        flags = evaluate_ds_dec_src2_f64_flags(
            val_a=f64_to_bits(0.0),
            val_b=f64_to_bits(15.0),
            eff_addr=0x200,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.threshold_reset)

    def test_greater_threshold_reset_flags(self) -> None:
        flags = evaluate_ds_dec_src2_f64_flags(
            val_a=f64_to_bits(100.0),
            val_b=f64_to_bits(50.0),
            eff_addr=0x300,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.threshold_reset)
        self.assertFalse(flags.borrow)

    def test_nan_flag_detection(self) -> None:
        qnan = 0x7FF8_0000_0000_0000
        flags = evaluate_ds_dec_src2_f64_flags(qnan, f64_to_bits(10.0), 0x400)
        self.assertTrue(flags.nan_flag)
        self.assertTrue(flags.wrapped)

    def test_unaligned_qword_address_detection(self) -> None:
        for offset in range(1, 8):
            flags = evaluate_ds_dec_src2_f64_flags(f64_to_bits(5.0), f64_to_bits(10.0), 0x100 + offset)
            self.assertTrue(flags.unaligned_addr)
        flags_aligned = evaluate_ds_dec_src2_f64_flags(f64_to_bits(5.0), f64_to_bits(10.0), 0x108)
        self.assertFalse(flags_aligned.unaligned_addr)


# ==============================================================================
# Phase 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

POS_ZERO_F64 = 0x0000_0000_0000_0000
NEG_ZERO_F64 = 0x8000_0000_0000_0000
POS_INF_F64 = 0x7FF0_0000_0000_0000
NEG_INF_F64 = 0xFFF0_0000_0000_0000
QNAN_F64 = 0x7FF8_0000_0000_0000
SNAN_F64 = 0x7FF0_0000_0000_0001
MIN_NORMAL_F64 = 0x0010_0000_0000_0000
MIN_SUBNORMAL_F64 = 0x0000_0000_0000_0001
MAX_SUBNORMAL_F64 = 0x000F_FFFF_FFFF_FFFF
MAX_NORMAL_F64 = 0x7FEF_FFFF_FFFF_FFFF


class TestDsDecSrc2F64IEEE754EdgeCases(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_decrement_wraps_at_zero(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x100
        idx_a = 0x100 >> 3
        idx_b = (0x100 + 8) >> 3

        # Subnormal float < 1.0 decrements into negative float, which triggers wrap to threshold
        lds.write_qword(idx_a, MIN_SUBNORMAL_F64)
        lds.write_qword(idx_b, f64_to_bits(10.0))

        new_val = lds.atomic_dec_src2_f64(vaddr, offset0=0, offset1=1)
        # MIN_SUBNORMAL_F64 - 1.0 is ~ -1.0 < 0 -> next wrap condition triggers
        f_res = bits_to_f64(new_val)
        self.assertTrue(math.isclose(f_res, -1.0, rel_tol=1e-9))

    def test_positive_and_negative_zero_asymmetry(self) -> None:
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(POS_ZERO_F64, b10), b10)
        self.assertEqual(ref_ds_dec_src2_f64(NEG_ZERO_F64, b10), b10)

    def test_infinities_bit_patterns(self) -> None:
        b10 = f64_to_bits(10.0)
        # +Inf > 10.0 resets to 10.0
        self.assertEqual(ref_ds_dec_src2_f64(POS_INF_F64, b10), b10)
        # -Inf <= 0.0 resets to 10.0
        self.assertEqual(ref_ds_dec_src2_f64(NEG_INF_F64, b10), b10)

    def test_quiet_and_signaling_nan_resets(self) -> None:
        b10 = f64_to_bits(10.0)
        self.assertEqual(ref_ds_dec_src2_f64(QNAN_F64, b10), b10)
        self.assertEqual(ref_ds_dec_src2_f64(SNAN_F64, b10), b10)

    def test_counter_cyclic_trajectory(self) -> None:
        limit_val = 5.0
        b_limit = f64_to_bits(limit_val)
        curr = b_limit
        for expected in [4.0, 3.0, 2.0, 1.0, 0.0]:
            curr = ref_ds_dec_src2_f64(curr, b_limit)
            self.assertEqual(bits_to_f64(curr), expected)

        # At 0.0, wraps to 5.0
        curr = ref_ds_dec_src2_f64(curr, b_limit)
        self.assertEqual(bits_to_f64(curr), 5.0)


# ==============================================================================
# Phase 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

import re


_ASM_F64_SRC2_RE = re.compile(
    r"^(ds_dec_src2_f64|ds_inc_src2_f64)\s+"
    r"(v\d+)"                        # vaddr
    r"(?:\s+offset0:(\d+))?"         # optional offset0
    r"(?:\s+offset1:(\d+))?"         # optional offset1
    r"(?:\s+(gds))?$"                # optional gds
)


def parse_ds_dec_src2_f64_asm(asm_line: str) -> DsSrc2F64Instruction:
    line = asm_line.strip()
    if not line:
        raise ValueError("Empty assembly instruction line")

    m = _ASM_F64_SRC2_RE.match(line)
    if not m:
        raise ValueError(f"Malformed ds_dec_src2_f64 instruction syntax: {line!r}")

    op_str, addr_str, off0_str, off1_str, gds_str = m.groups()
    opcode = DsSrc2F64Opcode(op_str)

    if not (addr_str.startswith("v") and addr_str[1:].isdigit()):
        raise ValueError(f"Invalid VGPR address operand: {addr_str}")
    vaddr = Register(RegKind.VGPR, int(addr_str[1:]), count=1)

    offset0 = int(off0_str) if off0_str else 0
    offset1 = int(off1_str) if off1_str else 0
    gds = (gds_str == "gds")

    return DsSrc2F64Instruction(
        opcode=opcode,
        vaddr=vaddr,
        offset0=offset0,
        offset1=offset1,
        gds=gds,
    )


class TestDsDecSrc2F64NegativeRejectionAndMutations(unittest.TestCase):
    """Phase 8: Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_malformed_tokens(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_dec_src2_f64_asm("")
        with self.assertRaises(ValueError):
            parse_ds_dec_src2_f64_asm("ds_dec_src2_f64")
        with self.assertRaises(ValueError):
            parse_ds_dec_src2_f64_asm("ds_dec_f64 v0")
        with self.assertRaises(ValueError):
            parse_ds_dec_src2_f64_asm("ds_dec_src2_f64 v0, v1")
        with self.assertRaises(ValueError):
            parse_ds_dec_src2_f64_asm("ds_dec_src2_f64 v0 bad_mod:1")

    def test_syntax_rejection_register_classes(self) -> None:
        s0 = Register(RegKind.SGPR, 0)
        with self.assertRaises(ValueError):
            DsSrc2F64Instruction(opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64, vaddr=s0)

    def test_syntax_rejection_offset_ranges(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        with self.assertRaises(ValueError):
            DsSrc2F64Instruction(opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64, vaddr=v0, offset0=-1)
        with self.assertRaises(ValueError):
            DsSrc2F64Instruction(opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64, vaddr=v0, offset0=256)

    def test_mutation_detection_zero_wrap_omission(self) -> None:
        def mutant_dec(bits_a: int, bits_b: int) -> int:
            f_a = bits_to_f64(bits_a)
            f_b = bits_to_f64(bits_b)
            if f_a > f_b:  # Missing f_a <= 0.0 check!
                return bits_b
            return f64_to_bits(f_a - 1.0)

        b0 = f64_to_bits(0.0)
        b10 = f64_to_bits(10.0)
        correct_res = ref_ds_dec_src2_f64(b0, b10)
        mutant_res = mutant_dec(b0, b10)
        self.assertEqual(correct_res, b10)
        self.assertEqual(bits_to_f64(mutant_res), -1.0)
        self.assertNotEqual(correct_res, mutant_res)

    def test_mutation_detection_strict_greater_threshold(self) -> None:
        def mutant_dec(bits_a: int, bits_b: int) -> int:
            f_a = bits_to_f64(bits_a)
            f_b = bits_to_f64(bits_b)
            if f_a <= 0.0 or f_a >= f_b:  # Bug: >= instead of >
                return bits_b
            return f64_to_bits(f_a - 1.0)

        b5 = f64_to_bits(5.0)
        correct_res = ref_ds_dec_src2_f64(b5, b5)
        mutant_res = mutant_dec(b5, b5)
        self.assertEqual(bits_to_f64(correct_res), 4.0)
        self.assertEqual(bits_to_f64(mutant_res), 5.0)
        self.assertNotEqual(correct_res, mutant_res)

    def test_mutation_detection_exec_mask_leak(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0x1)

        addr_0 = 0x100
        addr_1 = 0x200

        lds.write_qword(addr_0 >> 3, f64_to_bits(10.0))
        lds.write_qword((addr_0 + 8) >> 3, f64_to_bits(20.0))
        lds.write_qword(addr_1 >> 3, f64_to_bits(30.0))
        lds.write_qword((addr_1 + 8) >> 3, f64_to_bits(20.0))

        simd.write_lane_vgpr(0, 0, addr_0)
        simd.write_lane_vgpr(1, 0, addr_1)

        inst = DsSrc2F64Instruction(
            opcode=DsSrc2F64Opcode.DS_DEC_SRC2_F64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(results[0], f64_to_bits(9.0))
        self.assertIsNone(results[1])
        self.assertEqual(lds.read_qword(addr_1 >> 3), f64_to_bits(30.0))


if __name__ == "__main__":
    unittest.main()
