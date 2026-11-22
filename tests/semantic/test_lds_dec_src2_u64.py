from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_dec_src2_u64.

RDNA2 / GFX10 Local Data Share (LDS) 64-bit Dual-Source Atomic Decrement:
Performs a 64-bit atomic wrapping decrement on LDS memory location A using
the secondary LDS memory location B as the wrap threshold.

Syntax:
    ds_dec_src2_u64 vaddr [offset0:offset8] [offset1:offset8] [gds]

Semantics:
    byte_addr_a = (vaddr + (offset0 & 0xFF) * 8) & 0xFFFFFFFF
    byte_addr_b = (vaddr + (offset1 & 0xFF) * 8) & 0xFFFFFFFF
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    val_a = LDS64[qword_idx_a]
    val_b = LDS64[qword_idx_b]
    if val_a == 0 or val_a > val_b:
        new_val_a = val_b
    else:
        new_val_a = (val_a - 1) & 0xFFFFFFFFFFFFFFFF
    LDS64[qword_idx_a] = new_val_a

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing:
      Base byte address provided in VGPR vaddr.
      offset0 and offset1 are 8-bit unsigned immediates multiplied by 8 bytes (qword stride):
        byte_addr = (vaddr + offset * 8) mod 2^32.
      Addresses are qword-aligned: low 3 bits are dropped (byte_addr >> 3).
      Qword indices wrap modulo the allocated power-of-two capacity.
  - Wrap Condition:
      Unsigned 64-bit integer comparison between current memory value at A (val_a)
      and threshold value at B (val_b).
      When val_a == 0 or val_a > val_b, memory location A resets to val_b.
      Otherwise, memory location A is decremented by 1 mod 2^64.
  - Non-returning:
      Vector general-purpose registers (VGPRs) are not modified.
"""

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


def calc_lds_src2_u64_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_qwords: int = DEFAULT_LDS_QWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and qword indices for location A and location B.

    Address A = (vaddr + (offset0 & 0xFF) * 8) & MASK32
    Address B = (vaddr + (offset1 & 0xFF) * 8) & MASK32
    Qword index = (addr >> 3) & (lds_qwords - 1)

    Returns:
        Tuple[int, int, int, int]: (byte_addr_a, byte_addr_b, qword_idx_a, qword_idx_b)
    """
    byte_addr_a = u32(vaddr + ((offset0 & OFFSET8_MASK) * 8))
    byte_addr_b = u32(vaddr + ((offset1 & OFFSET8_MASK) * 8))
    qword_idx_a = (byte_addr_a >> 3) & (lds_qwords - 1)
    qword_idx_b = (byte_addr_b >> 3) & (lds_qwords - 1)
    return byte_addr_a, byte_addr_b, qword_idx_a, qword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_dec_src2_u64(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_dec_src2_u64.

    If current memory value at A (val_a) is 0 or strictly greater than threshold
    at B (val_b), reset to val_b. Otherwise, decrement by 1 mod 2^64.
    """
    a = u64(val_a)
    b = u64(val_b)
    if a == 0 or a > b:
        return b
    return u64(a - 1)


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = DEFAULT_LDS_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, qword_idx: int) -> int:
        """Read a 64-bit qword given an aligned qword index."""
        idx = qword_idx & (self.capacity_qwords - 1)
        return self.memory[idx]

    def write_qword(self, qword_idx: int, val: int) -> None:
        """Write a 64-bit qword given an aligned qword index."""
        idx = qword_idx & (self.capacity_qwords - 1)
        self.memory[idx] = u64(val)

    def atomic_dec_src2_u64(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic 64-bit dual-source decrement on LDS location A.

        Returns:
            int: The new 64-bit value written to LDS location A.
        """
        _, _, idx_a, idx_b = calc_lds_src2_u64_addresses(
            vaddr=vaddr,
            offset0=offset0,
            offset1=offset1,
            lds_qwords=self.capacity_qwords,
        )
        val_a = self.read_qword(idx_a)
        val_b = self.read_qword(idx_b)
        new_val_a = ref_ds_dec_src2_u64(val_a, val_b)
        self.write_qword(idx_a, new_val_a)
        return new_val_a


class TestReferenceOracleAndLDSModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_normal_decrement(self) -> None:
        # 5 decrements to 4 with threshold 10
        self.assertEqual(ref_ds_dec_src2_u64(5, 10), 4)
        self.assertEqual(ref_ds_dec_src2_u64(10, 10), 9)
        self.assertEqual(ref_ds_dec_src2_u64(1, 10), 0)

    def test_oracle_wrap_on_zero(self) -> None:
        # At 0, wraps to threshold val_b
        self.assertEqual(ref_ds_dec_src2_u64(0, 10), 10)
        self.assertEqual(ref_ds_dec_src2_u64(0, 0x1_0000_0000), 0x1_0000_0000)

    def test_oracle_wrap_on_greater_threshold(self) -> None:
        # If val_a > val_b, resets to val_b
        self.assertEqual(ref_ds_dec_src2_u64(11, 10), 10)
        self.assertEqual(ref_ds_dec_src2_u64(100, 5), 5)
        self.assertEqual(ref_ds_dec_src2_u64(MASK64, 50), 50)

    def test_oracle_zero_threshold(self) -> None:
        # Threshold 0: if val_a == 0 -> 0; if val_a > 0 -> 0
        self.assertEqual(ref_ds_dec_src2_u64(0, 0), 0)
        self.assertEqual(ref_ds_dec_src2_u64(5, 0), 0)

    def test_oracle_max_u64_boundary(self) -> None:
        # Max u64 threshold
        self.assertEqual(ref_ds_dec_src2_u64(MASK64, MASK64), MASK64 - 1)
        self.assertEqual(ref_ds_dec_src2_u64(0, MASK64), MASK64)

    def test_address_calculation_and_stride(self) -> None:
        base = 0x1000
        off0 = 2  # 2 * 8 = 16 bytes
        off1 = 5  # 5 * 8 = 40 bytes
        addr_a, addr_b, idx_a, idx_b = calc_lds_src2_u64_addresses(base, off0, off1)

        self.assertEqual(addr_a, 0x1010)
        self.assertEqual(addr_b, 0x1028)
        self.assertEqual(idx_a, 0x1010 >> 3)
        self.assertEqual(idx_b, 0x1028 >> 3)

    def test_offset8_masking(self) -> None:
        # Offsets are 8-bit unsigned
        base = 0x2000
        off = 0x102  # Masked to 2 -> 16 bytes
        addr_a, _, idx_a, _ = calc_lds_src2_u64_addresses(base, off, 0)
        self.assertEqual(addr_a, 0x2010)
        self.assertEqual(idx_a, 0x2010 >> 3)

    def test_lds_model_atomic_dec_src2_u64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x100

        # Location A: vaddr + 0*8 = idx 32
        # Location B: vaddr + 1*8 = idx 33
        idx_a = 0x100 >> 3
        idx_b = (0x100 + 8) >> 3

        lds.write_qword(idx_a, 5)
        lds.write_qword(idx_b, 10)  # threshold 10

        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 4)
        self.assertEqual(lds.read_qword(idx_a), 4)
        self.assertEqual(lds.read_qword(idx_b), 10)  # Threshold untouched!

        # Decrement down to 0
        lds.write_qword(idx_a, 1)
        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 0)

        # Wrap at 0
        new_val = lds.atomic_dec_src2_u64(vaddr, offset0=0, offset1=1)
        self.assertEqual(new_val, 10)
        self.assertEqual(lds.read_qword(idx_a), 10)


# ==============================================================================
# Phase 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

import dataclasses
import enum


class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
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


class DsSrc2Opcode(enum.Enum):
    """LDS dual-source instruction opcodes."""
    DS_DEC_SRC2_U64 = "ds_dec_src2_u64"
    DS_INC_SRC2_U64 = "ds_inc_src2_u64"


@dataclasses.dataclass(frozen=True)
class DsSrc2Instruction:
    """Symbolic AST node representing a dual-source DS atomic instruction."""
    opcode: DsSrc2Opcode
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


class DsDecSrc2Expr:
    """Typed symbolic expression for ds_dec_src2_u64."""

    def __init__(self, vaddr: int, offset0: int = 0, offset1: int = 0) -> None:
        self.vaddr = vaddr & MASK32
        self.offset0 = offset0 & OFFSET8_MASK
        self.offset1 = offset1 & OFFSET8_MASK

    def evaluate(self, lds: LDSModel64) -> int:
        return lds.atomic_dec_src2_u64(self.vaddr, self.offset0, self.offset1)


class TestDsDecSrc2U64AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_values(self) -> None:
        self.assertEqual(DsSrc2Opcode.DS_DEC_SRC2_U64.value, "ds_dec_src2_u64")
        self.assertEqual(DsSrc2Opcode.DS_INC_SRC2_U64.value, "ds_inc_src2_u64")

    def test_register_operand_validation(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        self.assertEqual(v0.format_asm(), "v0")
        s0 = Register(RegKind.SGPR, 4, count=2)
        self.assertEqual(s0.format_asm(), "s[4:5]")

        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)

    def test_instruction_creation_and_assembly(self) -> None:
        v1 = Register(RegKind.VGPR, 1)
        inst_simple = DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v1)
        self.assertEqual(inst_simple.to_assembly(), "ds_dec_src2_u64 v1")

        inst_offsets = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=v1,
            offset0=2,
            offset1=4,
            gds=True,
        )
        self.assertEqual(inst_offsets.to_assembly(), "ds_dec_src2_u64 v1 offset0:2 offset1:4 gds")

    def test_instruction_validation_errors(self) -> None:
        s0 = Register(RegKind.SGPR, 0)
        v_pair = Register(RegKind.VGPR, 0, count=2)
        v0 = Register(RegKind.VGPR, 0)

        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=s0)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v_pair)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset0=-1)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset0=256)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset1=-5)
        with self.assertRaises(ValueError):
            DsSrc2Instruction(opcode=DsSrc2Opcode.DS_DEC_SRC2_U64, vaddr=v0, offset1=300)

    def test_typed_expression_evaluation(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        vaddr = 0x200
        idx_a = 0x200 >> 3
        idx_b = (0x200 + 8) >> 3

        lds.write_qword(idx_a, 10)
        lds.write_qword(idx_b, 20)

        expr = DsDecSrc2Expr(vaddr=vaddr, offset0=0, offset1=1)
        res = expr.evaluate(lds)
        self.assertEqual(res, 9)
        self.assertEqual(lds.read_qword(idx_a), 9)


# ==============================================================================
# Phase 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarRegisterFile:
    """Scalar register file maintaining 256 VGPRs and 106 SGPRs."""

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
    """Deterministic scalar interpreter executing dual-source 64-bit LDS atomic instructions."""

    def __init__(self, lds: LDSModel64, rf: ScalarRegisterFile | None = None) -> None:
        self.lds = lds
        self.rf = rf if rf is not None else ScalarRegisterFile()

    def step(self, inst: DsSrc2Instruction) -> int:
        vaddr = self.rf.read_vgpr(inst.vaddr.index)
        new_val_a = self.lds.atomic_dec_src2_u64(
            vaddr=vaddr,
            offset0=inst.offset0,
            offset1=inst.offset1,
        )
        return new_val_a


class TestDsDecSrc2U64ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSModel64(capacity_qwords=512)
        self.rf = ScalarRegisterFile()
        self.interp = ScalarLDS64Interpreter(self.lds, self.rf)

    def test_scalar_step_normal_decrement(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 50)
        self.lds.write_qword(idx_b, 100)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 49)
        self.assertEqual(self.lds.read_qword(idx_a), 49)
        self.assertEqual(self.lds.read_qword(idx_b), 100)

    def test_scalar_step_wrap_at_zero(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 0)
        self.lds.write_qword(idx_b, 0x1_0000_0000)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 0x1_0000_0000)
        self.assertEqual(self.lds.read_qword(idx_a), 0x1_0000_0000)

    def test_scalar_step_reset_on_greater_threshold(self) -> None:
        vaddr = 0x80
        idx_a = 0x80 >> 3
        idx_b = (0x80 + 8) >> 3

        self.lds.write_qword(idx_a, 500)
        self.lds.write_qword(idx_b, 100)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 100)
        self.assertEqual(self.lds.read_qword(idx_a), 100)

    def test_scalar_step_with_offsets(self) -> None:
        vaddr = 0x100
        # offset0 = 2 -> 16 bytes -> idx (0x100 + 16) >> 3 = 34
        # offset1 = 4 -> 32 bytes -> idx (0x100 + 32) >> 3 = 36
        idx_a = (0x100 + 16) >> 3
        idx_b = (0x100 + 32) >> 3

        self.lds.write_qword(idx_a, 10)
        self.lds.write_qword(idx_b, 20)
        self.rf.write_vgpr(1, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 1),
            offset0=2,
            offset1=4,
        )

        res = self.interp.step(inst)
        self.assertEqual(res, 9)
        self.assertEqual(self.lds.read_qword(idx_a), 9)

    def test_scalar_step_non_returning_preserves_rf(self) -> None:
        vaddr = 0x40
        idx_a = 0x40 >> 3
        idx_b = (0x40 + 8) >> 3

        self.lds.write_qword(idx_a, 1)
        self.lds.write_qword(idx_b, 5)

        # Pre-seed registers
        for i in range(10):
            self.rf.write_vgpr(i, 0x1000 + i)
        self.rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        self.interp.step(inst)
        # Verify VGPRs completely untouched except v0 remaining vaddr
        self.assertEqual(self.rf.read_vgpr(0), vaddr)
        for i in range(1, 10):
            self.assertEqual(self.rf.read_vgpr(i), 0x1000 + i)


# ==============================================================================
# Phase 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDLDS64WavefrontInterpreter:
    """Multi-lane SIMD wavefront interpreter for ds_dec_src2_u64 with EXEC masking."""

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

    def step(self, inst: DsSrc2Instruction) -> list[int | None]:
        results: list[int | None] = []
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                vaddr = self.read_lane_vgpr(lane, inst.vaddr.index)
                new_val_a = self.lds.atomic_dec_src2_u64(
                    vaddr=vaddr,
                    offset0=inst.offset0,
                    offset1=inst.offset1,
                )
                results.append(new_val_a)
            else:
                results.append(None)
        return results


class TestDsDecSrc2U64SIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave64_full_exec_independent_lanes(self) -> None:
        lds = LDSModel64(capacity_qwords=2048)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)

        for lane in range(64):
            addr = lane * 16  # Each lane has 2 qwords: A at +0, B at +8
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, 20 + lane)
            lds.write_qword(idx_b, 100)
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 64)
        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            expected = (20 + lane) - 1
            self.assertEqual(results[lane], expected)
            self.assertEqual(lds.read_qword(idx_a), expected)
            self.assertEqual(lds.read_qword(idx_b), 100)

    def test_wave32_full_exec(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)

        for lane in range(32):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, 5)
            lds.write_qword(idx_b, 10)
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], 4)

    def test_exec_mask_disabled_lanes_unmodified(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        simd.set_exec_mask(0x5555_5555_5555_5555)  # Even lanes active

        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, 50)
            lds.write_qword(idx_b, 100)
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        for lane in range(64):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            if lane % 2 == 0:
                self.assertEqual(results[lane], 49)
                self.assertEqual(lds.read_qword(idx_a), 49)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(lds.read_qword(idx_a), 50)

    def test_exec_lo_hi_partial_wavefront_masking(self) -> None:
        lds = LDSModel64(capacity_qwords=1024)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=64, lds=lds)
        simd.set_exec_lo(0x1)        # Lane 0 only
        simd.set_exec_hi(0x80000000) # Lane 63 only

        for lane in (0, 63):
            addr = lane * 16
            idx_a = (addr + 0) >> 3
            idx_b = (addr + 8) >> 3
            lds.write_qword(idx_a, 10)
            lds.write_qword(idx_b, 20)
            simd.write_lane_vgpr(lane, 0, addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        self.assertEqual(results[0], 9)
        self.assertEqual(results[63], 9)
        for lane in range(1, 63):
            self.assertIsNone(results[lane])

    def test_multi_lane_serialization_same_target_a(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0xF)  # First 4 lanes active

        # All 4 lanes target same location A and threshold B
        common_addr = 0x80
        idx_a = (common_addr + 0) >> 3
        idx_b = (common_addr + 8) >> 3
        lds.write_qword(idx_a, 10)
        lds.write_qword(idx_b, 20)

        for lane in range(4):
            simd.write_lane_vgpr(lane, 0, common_addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        # Sequential decrement across lanes: 10 -> 9 -> 8 -> 7 -> 6
        self.assertEqual(results[0], 9)
        self.assertEqual(results[1], 8)
        self.assertEqual(results[2], 7)
        self.assertEqual(results[3], 6)
        self.assertEqual(lds.read_qword(idx_a), 6)

    def test_multi_lane_serialization_wrapping_at_zero(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0xF)  # 4 lanes active

        common_addr = 0x80
        idx_a = (common_addr + 0) >> 3
        idx_b = (common_addr + 8) >> 3
        lds.write_qword(idx_a, 1)  # Starts at 1
        lds.write_qword(idx_b, 5)  # Wrap threshold 5

        for lane in range(4):
            simd.write_lane_vgpr(lane, 0, common_addr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        # Lane 0: 1 -> 0
        self.assertEqual(results[0], 0)
        # Lane 1: 0 -> wraps to threshold 5
        self.assertEqual(results[1], 5)
        # Lane 2: 5 -> 4
        self.assertEqual(results[2], 4)
        # Lane 3: 4 -> 3
        self.assertEqual(results[3], 3)
        self.assertEqual(lds.read_qword(idx_a), 3)


# ==============================================================================
# Phase 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestDsDecSrc2U64RegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def test_vaddr_register_preservation_scalar(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        vaddr = 0x100
        idx_a = 0x100 >> 3
        idx_b = (0x100 + 8) >> 3
        lds.write_qword(idx_a, 50)
        lds.write_qword(idx_b, 100)

        rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        res = interp.step(inst)
        self.assertEqual(res, 49)
        # vaddr in v0 must remain intact
        self.assertEqual(rf.read_vgpr(0), vaddr)

    def test_identical_offsets_location_a_equals_location_b(self) -> None:
        # offset0 == offset1: location A is also location B (self-threshold aliasing)
        lds = LDSModel64(capacity_qwords=512)
        rf = ScalarRegisterFile()
        interp = ScalarLDS64Interpreter(lds, rf)

        vaddr = 0x200
        target_idx = (vaddr + 2 * 8) >> 3
        lds.write_qword(target_idx, 42)
        rf.write_vgpr(0, vaddr)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=2,
            offset1=2,  # A == B
        )

        # val_a = 42, val_b = 42. Since 42 != 0 and 42 is not > 42, it decrements to 41!
        res = interp.step(inst)
        self.assertEqual(res, 41)
        self.assertEqual(lds.read_qword(target_idx), 41)

        # When val_a == 0 and val_b == 0: wraps to val_b (0)
        lds.write_qword(target_idx, 0)
        res_zero = interp.step(inst)
        self.assertEqual(res_zero, 0)

    def test_interlaced_memory_locations_across_divergent_lanes(self) -> None:
        # Lane 0: A at 0x100, B at 0x108
        # Lane 1: A at 0x108, B at 0x110
        lds = LDSModel64(capacity_qwords=512)
        simd = SIMDLDS64WavefrontInterpreter(wave_size=32, lds=lds)
        simd.set_exec_mask(0x3)  # Lanes 0 and 1 active

        idx_0 = 0x100 >> 3
        idx_1 = 0x108 >> 3
        idx_2 = 0x110 >> 3

        lds.write_qword(idx_0, 10) # Lane 0's A
        lds.write_qword(idx_1, 20) # Lane 0's B, Lane 1's A
        lds.write_qword(idx_2, 30) # Lane 1's B

        simd.write_lane_vgpr(0, 0, 0x100)
        simd.write_lane_vgpr(1, 0, 0x108)

        inst = DsSrc2Instruction(
            opcode=DsSrc2Opcode.DS_DEC_SRC2_U64,
            vaddr=Register(RegKind.VGPR, 0),
            offset0=0,
            offset1=1,
        )

        results = simd.step(inst)
        # Lane 0: A was 10, threshold 20 -> 9
        self.assertEqual(results[0], 9)
        # Lane 1: A was 20, threshold 30 -> 19
        self.assertEqual(results[1], 19)

        # Registers preserved
        self.assertEqual(simd.read_lane_vgpr(0, 0), 0x100)
        self.assertEqual(simd.read_lane_vgpr(1, 0), 0x108)


# ==============================================================================
# Phase 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass(frozen=True)
class AtomicDecSrc2StatusFlags:
    """Architectural condition codes and status flags for ds_dec_src2_u64."""
    wrapped: bool
    zero: bool
    negative: bool
    borrow: bool
    threshold_reset: bool
    unaligned_addr: bool


def evaluate_ds_dec_src2_flags(
    val_a: int,
    val_b: int,
    eff_addr: int,
) -> AtomicDecSrc2StatusFlags:
    """Evaluate architectural flags for a 64-bit dual-source decrement with wrap operation."""
    u_a = u64(val_a)
    u_b = u64(val_b)

    wrapped = (u_a == 0 or u_a > u_b)
    threshold_reset = (u_a > u_b)
    borrow = (u_a == 0)

    new_val_a = u_b if wrapped else u64(u_a - 1)
    zero = (new_val_a == 0)
    negative = bool(new_val_a & (1 << 63))
    unaligned_addr = bool(eff_addr & 0x7)

    return AtomicDecSrc2StatusFlags(
        wrapped=wrapped,
        zero=zero,
        negative=negative,
        borrow=borrow,
        threshold_reset=threshold_reset,
        unaligned_addr=unaligned_addr,
    )


class TestDsDecSrc2U64ConditionCodesAndStatusFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_normal_decrement_flags(self) -> None:
        flags = evaluate_ds_dec_src2_flags(
            val_a=10,
            val_b=20,
            eff_addr=0x100,
        )
        self.assertFalse(flags.wrapped)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.borrow)
        self.assertFalse(flags.threshold_reset)
        self.assertFalse(flags.unaligned_addr)

    def test_zero_wrap_flags(self) -> None:
        flags = evaluate_ds_dec_src2_flags(
            val_a=0,
            val_b=15,
            eff_addr=0x200,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.borrow)
        self.assertFalse(flags.threshold_reset)
        self.assertFalse(flags.zero)

    def test_greater_threshold_reset_flags(self) -> None:
        flags = evaluate_ds_dec_src2_flags(
            val_a=100,
            val_b=50,
            eff_addr=0x300,
        )
        self.assertTrue(flags.wrapped)
        self.assertTrue(flags.threshold_reset)
        self.assertFalse(flags.borrow)

    def test_negative_flag_on_msb_set(self) -> None:
        # High bit set in threshold val_b (0x8000_0000_0000_0000)
        flags = evaluate_ds_dec_src2_flags(
            val_a=0,
            val_b=0x8000_0000_0000_0000,
            eff_addr=0x400,
        )
        self.assertTrue(flags.negative)
        self.assertTrue(flags.wrapped)

    def test_unaligned_qword_address_detection(self) -> None:
        for offset in range(1, 8):
            flags = evaluate_ds_dec_src2_flags(5, 10, 0x100 + offset)
            self.assertTrue(flags.unaligned_addr)

        flags_aligned = evaluate_ds_dec_src2_flags(5, 10, 0x108)
        self.assertFalse(flags_aligned.unaligned_addr)


if __name__ == "__main__":
    unittest.main()
