"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_condxchg32_f64.

Models double-precision floating-point (IEEE 754 binary64) conditional exchange
and AMD RDNA2 / GFX10 Local Data Share (LDS) architectural semantics:
  - ds_condxchg32_f64 (also known as ds_cmpst_f64):
    Atomic conditional exchange on 64-bit LDS storage without return using IEEE 754
    double-precision floating-point comparison.

RDNA2 LDS / Non-Returning 64-Bit Floating-Point Conditional Exchange ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     A 64-bit element occupies two consecutive 32-bit dwords: low dword at dword_idx,
     high dword at (dword_idx + 1).
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Floating-Point Comparison Semantics:
     The memory value and comparison value are interpreted as IEEE 754 binary64 floats.
     - Equality (+0.0 == -0.0): In IEEE 754, positive and negative zero compare equal.
       If memory contains -0.0 and cmp_val is +0.0, the exchange succeeds.
     - Not-a-Number (NaN != NaN): NaNs compare unordered; NaN == NaN evaluates to False.
       If cmp_val or memory is NaN, the exchange never occurs.
     - Infinities: +INF == +INF evaluates to True, -INF == -INF evaluates to True.
     - Subnormals: Compared according to IEEE 754 binary64 float values.
  3. Non-Returning Semantics:
     The non-returning variant discards previous memory contents (vdst is None).

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Set

MASK64: int = 0xFFFFFFFFFFFFFFFF
MASK32: int = 0xFFFFFFFF
SIGN_MASK_F64: int = 0x80000000_00000000
EXP_MASK_F64: int = 0x7FF00000_00000000
FRAC_MASK_F64: int = 0x000FFFFF_FFFFFFFF
QNAN_BIT_F64: int = 0x00080000_00000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_RAW_F64: int = 0x00000000_00000000
NEG_ZERO_RAW_F64: int = 0x80000000_00000000
POS_INF_RAW_F64: int = 0x7FF00000_00000000
NEG_INF_RAW_F64: int = 0xFFF00000_00000000
QNAN_RAW_F64: int = 0x7FF80000_00000000
SNAN_RAW_F64: int = 0x7FF00000_00000001
ONE_RAW_F64: int = 0x3FF00000_00000000      # 1.0
TWO_RAW_F64: int = 0x40000000_00000000      # 2.0
THREE_RAW_F64: int = 0x40080000_00000000    # 3.0

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def split64(val: int) -> Tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32)."""
    return val & MASK32, (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    """Combine two 32-bit integers into unsigned 64-bit integer."""
    return ((lo & MASK32) | ((hi & MASK32) << 32)) & MASK64


def raw_u64_to_f64(raw: int) -> float:
    """Convert raw 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", raw & MASK64))[0]


def f64_to_raw_u64(val: float) -> int:
    """Convert Python float to 64-bit IEEE 754 double-precision bit pattern."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_nan_raw_f64(raw: int) -> bool:
    """Check if 64-bit bit pattern represents an IEEE 754 NaN."""
    return (raw & EXP_MASK_F64) == EXP_MASK_F64 and (raw & FRAC_MASK_F64) != 0


def float_equal_f64(a_raw: int, b_raw: int) -> bool:
    """Evaluate IEEE 754 binary64 equality between two raw bit patterns."""
    if is_nan_raw_f64(a_raw) or is_nan_raw_f64(b_raw):
        return False
    fa = raw_u64_to_f64(a_raw)
    fb = raw_u64_to_f64(b_raw)
    return fa == fb


def oracle_condxchg64_f64(
    mem_raw: int,
    cmp_raw: int,
    new_raw: int,
) -> Tuple[int, int, bool]:
    """Reference mathematical oracle for ds_condxchg32_f64.
    
    Returns:
        (old_val, new_mem_val, exchanged)
    """
    m = mem_raw & MASK64
    c = cmp_raw & MASK64
    n = new_raw & MASK64

    if float_equal_f64(m, c):
        return m, n, True
    return m, m, False


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_indices(self, byte_addr: int, offset: int = 0) -> Tuple[int, int]:
        """Compute effective wrapped dword indices for a 64-bit (2 dword) access."""
        effective_byte = (byte_addr + offset) & MASK32
        base_idx = (effective_byte >> 2) & (self.num_dwords - 1)
        hi_idx = (base_idx + 1) & (self.num_dwords - 1)
        return base_idx, hi_idx

    def get_banks(self, idx0: int, idx1: int) -> Tuple[int, int]:
        """Compute memory bank indices for the given dword pair."""
        return idx0 % NUM_LDS_BANKS, idx1 % NUM_LDS_BANKS

    def read_qword(self, byte_addr: int, offset: int = 0) -> int:
        """Read a 64-bit qword from LDS."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo = self.dwords[idx0]
        hi = self.dwords[idx1]
        return join64(lo, hi)

    def write_qword(self, byte_addr: int, val: int, offset: int = 0) -> None:
        """Write a 64-bit qword into LDS."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo, hi = split64(val)
        self.dwords[idx0] = lo
        self.dwords[idx1] = hi

    def atomic_condxchg64_f64(
        self,
        byte_addr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 64-bit floating-point conditional exchange."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo = self.dwords[idx0]
        hi = self.dwords[idx1]
        old_val = join64(lo, hi)

        old, new_mem, exchanged = oracle_condxchg64_f64(old_val, cmp_val, new_val)
        if exchanged:
            new_lo, new_hi = split64(new_mem)
            self.dwords[idx0] = new_lo
            self.dwords[idx1] = new_hi

        return old, new_mem, exchanged


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for reference 64-bit float conditional exchange oracle and LDS model."""

    def test_oracle_finite_matches_and_mismatches(self) -> None:
        old, mem, swapped = oracle_condxchg64_f64(ONE_RAW_F64, ONE_RAW_F64, TWO_RAW_F64)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW_F64)
        self.assertEqual(mem, TWO_RAW_F64)

        old, mem, swapped = oracle_condxchg64_f64(ONE_RAW_F64, TWO_RAW_F64, THREE_RAW_F64)
        self.assertFalse(swapped)
        self.assertEqual(old, ONE_RAW_F64)
        self.assertEqual(mem, ONE_RAW_F64)

    def test_oracle_signed_zero_equality(self) -> None:
        old, mem, swapped = oracle_condxchg64_f64(NEG_ZERO_RAW_F64, POS_ZERO_RAW_F64, ONE_RAW_F64)
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_RAW_F64)
        self.assertEqual(mem, ONE_RAW_F64)

        old2, mem2, swapped2 = oracle_condxchg64_f64(POS_ZERO_RAW_F64, NEG_ZERO_RAW_F64, TWO_RAW_F64)
        self.assertTrue(swapped2)
        self.assertEqual(old2, POS_ZERO_RAW_F64)
        self.assertEqual(mem2, TWO_RAW_F64)

    def test_oracle_nan_unordered_rejection(self) -> None:
        old, mem, swapped = oracle_condxchg64_f64(QNAN_RAW_F64, QNAN_RAW_F64, ONE_RAW_F64)
        self.assertFalse(swapped)
        self.assertEqual(old, QNAN_RAW_F64)
        self.assertEqual(mem, QNAN_RAW_F64)

        old_s, mem_s, swapped_s = oracle_condxchg64_f64(SNAN_RAW_F64, SNAN_RAW_F64, TWO_RAW_F64)
        self.assertFalse(swapped_s)
        self.assertEqual(old_s, SNAN_RAW_F64)
        self.assertEqual(mem_s, SNAN_RAW_F64)

    def test_oracle_infinities(self) -> None:
        old, mem, swapped = oracle_condxchg64_f64(POS_INF_RAW_F64, POS_INF_RAW_F64, ONE_RAW_F64)
        self.assertTrue(swapped)
        self.assertEqual(mem, ONE_RAW_F64)

        old, mem, swapped = oracle_condxchg64_f64(POS_INF_RAW_F64, NEG_INF_RAW_F64, ONE_RAW_F64)
        self.assertFalse(swapped)
        self.assertEqual(mem, POS_INF_RAW_F64)

    def test_lds_dword_indices_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        for b in range(4):
            idx0, idx1 = lds.get_dword_indices(b)
            self.assertEqual(idx0, 0)
            self.assertEqual(idx1, 1)

        idx0, idx1 = lds.get_dword_indices(4)
        self.assertEqual(idx0, 1)
        self.assertEqual(idx1, 2)

    def test_lds_read_write_qword(self) -> None:
        lds = LDSMemory(65536)
        val = 0x3FF00000_00000000  # 1.0f64
        lds.write_qword(0x20, val)
        self.assertEqual(lds.read_qword(0x20), val)

    def test_lds_atomic_condxchg64_f64_execution(self) -> None:
        lds = LDSMemory(65536)
        lds.write_qword(40, ONE_RAW_F64)

        old, mem, swapped = lds.atomic_condxchg64_f64(40, ONE_RAW_F64, TWO_RAW_F64)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW_F64)
        self.assertEqual(lds.read_qword(40), TWO_RAW_F64)

class DSOpcode(str, enum.Enum):
    """Opcode enumerations for 64-bit floating-point non-returning conditional exchange."""
    DS_CONDXCHG32_F64 = "ds_condxchg32_f64"
    DS_CMPST_F64 = "ds_cmpst_f64"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return False

    @property
    def bit_width(self) -> int:
        """Operand bit width."""
        return 64


@dataclasses.dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register operand v[0] .. v[255]."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class VGPRPair:
    """Consecutive pair of Vector General Purpose Registers v[i:i+1]."""
    base: int

    def __post_init__(self) -> None:
        if not (0 <= self.base <= 254):
            raise ValueError(f"VGPRPair base index out of range [0, 254]: {self.base}")

    @property
    def lo(self) -> VGPR:
        return VGPR(self.base)

    @property
    def hi(self) -> VGPR:
        return VGPR(self.base + 1)

    def __str__(self) -> str:
        return f"v[{self.base}:{self.base + 1}]"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """Unsigned 16-bit byte offset immediate [0, 65535]."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"Offset immediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSCondXchg32F64Instruction:
    """Symbolic AST node representing an RDNA2 non-returning DS 64-bit float conditional exchange."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata0: VGPRPair
    vdata1: VGPRPair
    vdst: Optional[VGPRPair] = None
    offset: OffsetImmediate = OffsetImmediate(0)

    def __post_init__(self) -> None:
        if self.vdst is not None:
            raise ValueError(f"Non-returning opcode {self.opcode.value} must not have destination vdst")

    def disasm(self) -> str:
        """Format symbolic instruction into canonical RDNA2 assembly text."""
        parts = [self.opcode.value, f"{self.vaddr}, {self.vdata0}, {self.vdata1}"]
        if self.offset.value != 0:
            parts.append(str(self.offset))
        return " ".join(parts)

    @classmethod
    def parse(cls, text: str) -> DSCondXchg32F64Instruction:
        """Parse canonical RDNA2 assembly string into a DSCondXchg32F64Instruction AST."""
        tokens = text.strip().split(None, 1)
        if not tokens:
            raise ValueError("Empty assembly string")
        op_str = tokens[0]
        try:
            opcode = DSOpcode(op_str)
        except ValueError:
            raise ValueError(f"Unknown or unsupported DS opcode: {op_str}")

        if len(tokens) == 1:
            raise ValueError(f"Missing operands for {op_str}")

        rest = tokens[1].strip()
        offset_val = 0
        if "offset:" in rest:
            m = rest.split("offset:")
            rest = m[0].strip()
            offset_part = m[1].split()
            if not offset_part:
                raise ValueError("Malformed offset immediate")
            offset_tok = offset_part[0].rstrip(",")
            try:
                offset_val = int(offset_tok, 0)
            except ValueError:
                raise ValueError(f"Invalid offset integer: {offset_tok}")

        reg_tokens = [tok.strip().rstrip(",") for tok in rest.split(",") if tok.strip()]

        def parse_vgpr(s: str) -> VGPR:
            s = s.strip()
            if not s.startswith("v") or not s[1:].isdigit():
                raise ValueError(f"Invalid VGPR syntax: {s}")
            return VGPR(int(s[1:]))

        def parse_vgpr_pair(s: str) -> VGPRPair:
            s = s.strip()
            if s.startswith("v[") and s.endswith("]"):
                inner = s[2:-1]
                parts = inner.split(":")
                if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                    raise ValueError(f"Invalid VGPRPair range syntax: {s}")
                base = int(parts[0])
                hi = int(parts[1])
                if hi != base + 1:
                    raise ValueError(f"VGPRPair must be consecutive [i:i+1], got {s}")
                return VGPRPair(base)
            elif s.startswith("v") and s[1:].isdigit():
                return VGPRPair(int(s[1:]))
            raise ValueError(f"Invalid VGPRPair syntax: {s}")

        if len(reg_tokens) != 3:
            raise ValueError(f"{opcode.value} expects 3 operands (vaddr, vdata0, vdata1), got {reg_tokens}")

        vaddr = parse_vgpr(reg_tokens[0])
        vdata0 = parse_vgpr_pair(reg_tokens[1])
        vdata1 = parse_vgpr_pair(reg_tokens[2])

        return cls(
            opcode=opcode,
            vaddr=vaddr,
            vdata0=vdata0,
            vdata1=vdata1,
            vdst=None,
            offset=OffsetImmediate(offset_val),
        )


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions, registers, and assembly parsing."""

    def test_opcode_properties(self) -> None:
        self.assertFalse(DSOpcode.DS_CONDXCHG32_F64.is_return)
        self.assertEqual(DSOpcode.DS_CONDXCHG32_F64.bit_width, 64)
        self.assertFalse(DSOpcode.DS_CMPST_F64.is_return)
        self.assertEqual(DSOpcode.DS_CMPST_F64.bit_width, 64)

    def test_vgpr_and_vgpr_pair_validation(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")

        pair0 = VGPRPair(0)
        self.assertEqual(str(pair0), "v[0:1]")

        with self.assertRaises(ValueError):
            VGPRPair(-1)
        with self.assertRaises(ValueError):
            VGPRPair(255)

    def test_offset_immediate_validation(self) -> None:
        off0 = OffsetImmediate(0)
        off64 = OffsetImmediate(64)
        self.assertEqual(str(off0), "")
        self.assertEqual(str(off64), "offset:64")

        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

    def test_instruction_post_init_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSCondXchg32F64Instruction(
                opcode=DSOpcode.DS_CONDXCHG32_F64,
                vaddr=VGPR(0),
                vdata0=VGPRPair(2),
                vdata1=VGPRPair(4),
                vdst=VGPRPair(6),
            )

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSCondXchg32F64Instruction(
                DSOpcode.DS_CONDXCHG32_F64,
                vaddr=VGPR(1),
                vdata0=VGPRPair(4),
                vdata1=VGPRPair(6),
            ),
            DSCondXchg32F64Instruction(
                DSOpcode.DS_CONDXCHG32_F64,
                vaddr=VGPR(1),
                vdata0=VGPRPair(4),
                vdata1=VGPRPair(6),
                offset=OffsetImmediate(16),
            ),
            DSCondXchg32F64Instruction(
                DSOpcode.DS_CMPST_F64,
                vaddr=VGPR(10),
                vdata0=VGPRPair(12),
                vdata1=VGPRPair(14),
                offset=OffsetImmediate(64),
            ),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSCondXchg32F64Instruction.parse(asm)
                self.assertEqual(inst, parsed)


if __name__ == "__main__":
    unittest.main()
