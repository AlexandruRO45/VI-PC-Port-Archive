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


if __name__ == "__main__":
    unittest.main()
