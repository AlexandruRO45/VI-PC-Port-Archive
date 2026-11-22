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

class ScalarRegisterFile:
    """Architectural register file for a single wavefront lane (256 32-bit VGPRs)."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * 256

    def get_u32(self, reg: VGPR) -> int:
        return self.regs[reg.index] & MASK32

    def set_u32(self, reg: VGPR, val: int) -> None:
        self.regs[reg.index] = val & MASK32

    def get_u64(self, pair: VGPRPair) -> int:
        lo = self.get_u32(pair.lo)
        hi = self.get_u32(pair.hi)
        return join64(lo, hi)

    def set_u64(self, pair: VGPRPair, val: int) -> None:
        lo, hi = split64(val)
        self.set_u32(pair.lo, lo)
        self.set_u32(pair.hi, hi)


class ScalarThreadState:
    """Execution state of a single scalar thread executing LDS instructions."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for non-returning RDNA2 DS 64-bit float conditional exchange."""

    def execute(self, state: ScalarThreadState, inst: DSCondXchg32F64Instruction) -> bool:
        """Execute a single non-returning DS 64-bit float conditional exchange instruction."""
        addr = state.rf.get_u32(inst.vaddr)
        cmp_val = state.rf.get_u64(inst.vdata0)
        new_val = state.rf.get_u64(inst.vdata1)

        old_val, new_mem, exchanged = state.lds.atomic_condxchg64_f64(
            addr, cmp_val, new_val, offset=inst.offset.value
        )
        return exchanged


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution of ds_condxchg32_f64."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_exchange_success_on_match(self) -> None:
        self.lds.write_qword(32, ONE_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 32)
        self.state.rf.set_u64(VGPRPair(4), ONE_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)

        self.assertTrue(swapped)
        self.assertEqual(self.lds.read_qword(32), TWO_RAW_F64)

    def test_scalar_exchange_failure_on_mismatch(self) -> None:
        self.lds.write_qword(40, ONE_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 40)
        self.state.rf.set_u64(VGPRPair(4), THREE_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)

        self.assertFalse(swapped)
        self.assertEqual(self.lds.read_qword(40), ONE_RAW_F64)

    def test_scalar_exchange_with_offset(self) -> None:
        self.lds.write_qword(48, TWO_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 32)
        self.state.rf.set_u64(VGPRPair(4), TWO_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), THREE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            offset=OffsetImmediate(16),  # 32 + 16 = 48
        )
        swapped = self.interp.execute(self.state, inst)

        self.assertTrue(swapped)
        self.assertEqual(self.lds.read_qword(48), THREE_RAW_F64)

    def test_scalar_signed_zero_exchange(self) -> None:
        self.lds.write_qword(24, NEG_ZERO_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 24)
        self.state.rf.set_u64(VGPRPair(4), POS_ZERO_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), ONE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)

        self.assertTrue(swapped)
        self.assertEqual(self.lds.read_qword(24), ONE_RAW_F64)

class SIMDWaveState:
    """Architectural wavefront execution state for 32 or 64 lanes."""

    def __init__(self, wave_size: int = 32, exec_mask: int = 0xFFFFFFFF, lds: Optional[LDSMemory] = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wavefront size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = exec_mask & ((1 << wave_size) - 1)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_u32(self, lane: int, reg: VGPR) -> int:
        return self.vgprs[lane][reg.index] & MASK32

    def set_u32(self, lane: int, reg: VGPR, val: int) -> None:
        self.vgprs[lane][reg.index] = val & MASK32

    def get_u64(self, lane: int, pair: VGPRPair) -> int:
        lo = self.get_u32(lane, pair.lo)
        hi = self.get_u32(lane, pair.hi)
        return join64(lo, hi)

    def set_u64(self, lane: int, pair: VGPRPair, val: int) -> None:
        lo, hi = split64(val)
        self.set_u32(lane, pair.lo, lo)
        self.set_u32(lane, pair.hi, hi)


class SIMDDSInterpreter:
    """Multi-lane SIMD instruction interpreter modeling RDNA2 lockstep execution and 64-bit float atomics."""

    def execute(self, state: SIMDWaveState, inst: DSCondXchg32F64Instruction) -> Dict[int, bool]:
        active_lanes = [lane for lane in range(state.wave_size) if state.is_lane_active(lane)]
        latched_inputs = [
            (
                lane,
                state.get_u32(lane, inst.vaddr),
                state.get_u64(lane, inst.vdata0),
                state.get_u64(lane, inst.vdata1),
            )
            for lane in active_lanes
        ]

        returns: Dict[int, bool] = {}

        for lane, addr, cmp_val, new_val in latched_inputs:
            old_val, new_mem, exchanged = state.lds.atomic_condxchg64_f64(
                addr, cmp_val, new_val, offset=inst.offset.value
            )
            returns[lane] = exchanged

        return returns


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD lockstep execution, 32/64-bit EXEC masking, and address convergence."""

    def test_wave32_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_qword(i * 8, ONE_RAW_F64)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), ONE_RAW_F64)
            wave.set_u64(lane, VGPRPair(6), f64_to_raw_u64(float(lane + 10)))

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        for i in range(32):
            self.assertTrue(returns[i])
            self.assertEqual(lds.read_qword(i * 8), f64_to_raw_u64(float(i + 10)))

    def test_wave64_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(64):
            lds.write_qword(i * 8, TWO_RAW_F64)

        for lane in range(64):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), TWO_RAW_F64)
            wave.set_u64(lane, VGPRPair(6), THREE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        for i in range(64):
            self.assertTrue(returns[i])
            self.assertEqual(lds.read_qword(i * 8), THREE_RAW_F64)

    def test_wave32_partial_exec_masking(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_qword(i * 8, ONE_RAW_F64)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), ONE_RAW_F64)
            wave.set_u64(lane, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(lds.read_qword(lane * 8), TWO_RAW_F64)
                self.assertTrue(returns[lane])
            else:
                self.assertEqual(lds.read_qword(lane * 8), ONE_RAW_F64)
                self.assertNotIn(lane, returns)

    def test_zero_exec_mask_no_mutation(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_qword(0, ONE_RAW_F64)
        for lane in range(32):
            wave.set_u32(lane, VGPR(1), 0)
            wave.set_u64(lane, VGPRPair(4), ONE_RAW_F64)
            wave.set_u64(lane, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        self.assertEqual(len(returns), 0)
        self.assertEqual(lds.read_qword(0), ONE_RAW_F64)

    def test_convergent_collision_cas_daisy_chain(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_qword(16, POS_ZERO_RAW_F64)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), 16)
            wave.set_u64(lane, VGPRPair(4), f64_to_raw_u64(float(lane)))
            wave.set_u64(lane, VGPRPair(6), f64_to_raw_u64(float(lane + 1)))

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        self.assertEqual(lds.read_qword(16), f64_to_raw_u64(32.0))
        for lane in range(32):
            self.assertTrue(returns[lane])

class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_addr_aliases_data0_lo(self) -> None:
        val = f64_to_raw_u64(32.0)
        self.lds.write_qword(32, val)
        self.state.rf.set_u32(VGPR(4), 32)
        self.state.rf.set_u32(VGPR(5), 0)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(4),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertFalse(swapped)
        self.assertEqual(self.state.rf.get_u32(VGPR(4)), 32)
        self.assertEqual(self.lds.read_qword(32), val)

    def test_scalar_data0_aliases_data1(self) -> None:
        self.lds.write_qword(40, ONE_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 40)
        self.state.rf.set_u64(VGPRPair(4), ONE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(4),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertTrue(swapped)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(4)), ONE_RAW_F64)
        self.assertEqual(self.lds.read_qword(40), ONE_RAW_F64)

    def test_operand_preservation_non_aliased(self) -> None:
        self.lds.write_qword(48, ONE_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 48)
        self.state.rf.set_u64(VGPRPair(4), ONE_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)
        self.state.rf.set_u32(VGPR(100), 0xCAFEBABE)
        self.state.rf.set_u32(VGPR(255), 0x12345678)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.rf.get_u32(VGPR(1)), 48)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(4)), ONE_RAW_F64)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(6)), TWO_RAW_F64)
        self.assertEqual(self.state.rf.get_u32(VGPR(100)), 0xCAFEBABE)
        self.assertEqual(self.state.rf.get_u32(VGPR(255)), 0x12345678)

    def test_simd_aliasing_under_divergence(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xAAAAAAAA, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_qword(i * 8, ONE_RAW_F64)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), ONE_RAW_F64)
            wave.set_u64(lane, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 != 0:
                self.assertEqual(lds.read_qword(lane * 8), TWO_RAW_F64)
                self.assertTrue(returns[lane])
            else:
                self.assertEqual(lds.read_qword(lane * 8), ONE_RAW_F64)
                self.assertNotIn(lane, returns)

@dataclasses.dataclass
class LDSStatusFlags:
    """Architectural status flags, wrapping conditions, and bank metrics for 64-bit float LDS operations."""
    carry_out: bool = False
    unaligned_address: bool = False
    oob_wrapped: bool = False
    bank_conflicts: int = 0
    address_collisions: int = 0
    exchange_occurred: bool = False
    exchange_failed: bool = False
    result_is_zero: bool = False
    result_has_sign_bit: bool = False


def evaluate_status_flags(
    wave: SIMDWaveState,
    inst: DSCondXchg32F64Instruction,
) -> LDSStatusFlags:
    flags = LDSStatusFlags()
    active_lanes = [l for l in range(wave.wave_size) if wave.is_lane_active(l)]

    bank_to_dwords: Dict[int, Set[int]] = {}
    dword_to_lanes: Dict[int, List[int]] = {}

    for lane in active_lanes:
        addr = wave.get_u32(lane, inst.vaddr)
        off = inst.offset.value
        cmp_val = wave.get_u64(lane, inst.vdata0)
        new_val = wave.get_u64(lane, inst.vdata1)

        if (addr + off) > MASK32:
            flags.carry_out = True

        effective_byte = (addr + off) & MASK32
        if (effective_byte & 7) != 0:
            flags.unaligned_address = True

        if (effective_byte + 8) > wave.lds.size_bytes:
            flags.oob_wrapped = True

        idx0, idx1 = wave.lds.get_dword_indices(addr, off)
        bank0, bank1 = wave.lds.get_banks(idx0, idx1)

        bank_to_dwords.setdefault(bank0, set()).add(idx0)
        bank_to_dwords.setdefault(bank1, set()).add(idx1)

        dword_to_lanes.setdefault(idx0, []).append(lane)

        old_val = wave.lds.read_qword(addr, off)
        _, new_mem, swapped = oracle_condxchg64_f64(old_val, cmp_val, new_val)
        if swapped:
            flags.exchange_occurred = True
        else:
            flags.exchange_failed = True

        if (new_mem & 0x7FFFFFFF_FFFFFFFF) == 0:
            flags.result_is_zero = True
        if bool(new_mem & 0x80000000_00000000):
            flags.result_has_sign_bit = True

    for bank, dword_set in bank_to_dwords.items():
        if len(dword_set) > 1:
            flags.bank_conflicts += (len(dword_set) - 1)

    for dword, lanes in dword_to_lanes.items():
        if len(lanes) > 1:
            flags.address_collisions += (len(lanes) - 1)

    return flags


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_address_carry_out_generation(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_u32(0, VGPR(1), 0xFFFFFFFF)
        wave.set_u64(0, VGPRPair(4), ONE_RAW_F64)
        wave.set_u64(0, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            offset=OffsetImmediate(16),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.carry_out)

    def test_unaligned_byte_address_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_u32(0, VGPR(1), 4)
        wave.set_u64(0, VGPRPair(4), ONE_RAW_F64)
        wave.set_u64(0, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.unaligned_address)

    def test_oob_address_wrapping_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_u32(0, VGPR(1), 65532)
        wave.set_u64(0, VGPRPair(4), ONE_RAW_F64)
        wave.set_u64(0, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.oob_wrapped)

    def test_bank_conflicts_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x3)
        wave.set_u32(0, VGPR(1), 0 * 4)
        wave.set_u32(1, VGPR(1), 32 * 4)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertEqual(flags.bank_conflicts, 2)
        self.assertEqual(flags.address_collisions, 0)

    def test_address_collisions_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xF)
        for l in range(4):
            wave.set_u32(l, VGPR(1), 64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertEqual(flags.address_collisions, 3)

    def test_exchange_outcome_and_result_flags(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x3)
        wave.lds.write_qword(0, ONE_RAW_F64)
        wave.lds.write_qword(8, TWO_RAW_F64)

        wave.set_u32(0, VGPR(1), 0)
        wave.set_u64(0, VGPRPair(4), ONE_RAW_F64)
        wave.set_u64(0, VGPRPair(6), NEG_ZERO_RAW_F64)

        wave.set_u32(1, VGPR(1), 8)
        wave.set_u64(1, VGPRPair(4), THREE_RAW_F64)
        wave.set_u64(1, VGPRPair(6), ONE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.exchange_occurred)
        self.assertTrue(flags.exchange_failed)
        self.assertTrue(flags.result_is_zero)
        self.assertTrue(flags.result_has_sign_bit)

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_snan_preservation_without_quietening(self) -> None:
        snan64 = 0x7FF00000_00000001
        self.assertEqual(snan64 & QNAN_BIT_F64, 0)

        self.lds.write_qword(16, snan64)
        self.state.rf.set_u32(VGPR(1), 16)
        self.state.rf.set_u64(VGPRPair(4), snan64)
        self.state.rf.set_u64(VGPRPair(6), ONE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertFalse(swapped)
        self.assertEqual(self.lds.read_qword(16), snan64)

    def test_qnan_payload_preservation_and_comparison_failure(self) -> None:
        qnan64 = 0x7FF8DEAD_BEEFCAFE
        self.assertNotEqual(qnan64 & QNAN_BIT_F64, 0)

        self.lds.write_qword(24, qnan64)
        self.state.rf.set_u32(VGPR(1), 24)
        self.state.rf.set_u64(VGPRPair(4), qnan64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertFalse(swapped)
        self.assertEqual(self.lds.read_qword(24), qnan64)

    def test_signed_zeros_both_directions(self) -> None:
        self.lds.write_qword(32, NEG_ZERO_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 32)
        self.state.rf.set_u64(VGPRPair(4), POS_ZERO_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), THREE_RAW_F64)

        inst1 = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped1 = self.interp.execute(self.state, inst1)
        self.assertTrue(swapped1)
        self.assertEqual(self.lds.read_qword(32), THREE_RAW_F64)

        self.lds.write_qword(40, POS_ZERO_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 40)
        self.state.rf.set_u64(VGPRPair(4), NEG_ZERO_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst2 = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped2 = self.interp.execute(self.state, inst2)
        self.assertTrue(swapped2)
        self.assertEqual(self.lds.read_qword(40), TWO_RAW_F64)

    def test_infinities_matching_and_sign_distinction(self) -> None:
        self.lds.write_qword(48, POS_INF_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 48)
        self.state.rf.set_u64(VGPRPair(4), POS_INF_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), ONE_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertTrue(swapped)
        self.assertEqual(self.lds.read_qword(48), ONE_RAW_F64)

        self.lds.write_qword(56, NEG_INF_RAW_F64)
        self.state.rf.set_u32(VGPR(1), 56)
        self.state.rf.set_u64(VGPRPair(4), POS_INF_RAW_F64)
        self.state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst_neg = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped_neg = self.interp.execute(self.state, inst_neg)
        self.assertFalse(swapped_neg)
        self.assertEqual(self.lds.read_qword(56), NEG_INF_RAW_F64)

    def test_subnormals_preservation_and_exact_match(self) -> None:
        subnorm_min = 0x00000000_00000001
        subnorm_max = 0x000FFFFF_FFFFFFFF

        self.lds.write_qword(64, subnorm_min)
        self.state.rf.set_u32(VGPR(1), 64)
        self.state.rf.set_u64(VGPRPair(4), subnorm_min)
        self.state.rf.set_u64(VGPRPair(6), subnorm_max)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        swapped = self.interp.execute(self.state, inst)
        self.assertTrue(swapped)
        self.assertEqual(self.lds.read_qword(64), subnorm_max)

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_syntax(self) -> None:
        invalid_assemblies = [
            "",
            "   ",
            "v_condxchg_f64 v1, v[2:3], v[4:5]",
            "ds_condxchg64_f64 v1, v[2:3], v[4:5]",
            "ds_min_f64 v1, v[2:3]",
            "ds_condxchg32_f64 v1, v[2:3]",  # Missing 3rd operand
            "ds_condxchg32_f64 v[0:1], v1, v[2:3], v[4:5]",  # 4 operands (only valid for returning variant!)
            "ds_condxchg32_f64 r1, v[2:3], v[4:5]",
            "ds_condxchg32_f64 s1, v[2:3], v[4:5]",
            "ds_condxchg32_f64 v256, v[2:3], v[4:5]",
            "ds_condxchg32_f64 v1, v[2:3], v[4:5] offset:70000",
            "ds_condxchg32_f64 v1, v[2:3], v[4:5] offset:-1",
            "ds_condxchg32_f64 v1, v[2:3], v[4:5] offset:",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, IndexError)):
                    DSCondXchg32F64Instruction.parse(asm)

    def test_ast_and_model_negative_controls(self) -> None:
        with self.assertRaises(ValueError):
            LDSMemory(1000)
        with self.assertRaises(ValueError):
            LDSMemory(-64)
        with self.assertRaises(ValueError):
            LDSMemory(0)

        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=128)

        with self.assertRaises(ValueError):
            DSCondXchg32F64Instruction(
                opcode=DSOpcode.DS_CONDXCHG32_F64,
                vaddr=VGPR(0),
                vdata0=VGPRPair(2),
                vdata1=VGPRPair(4),
                vdst=VGPRPair(6),
            )

    def test_mutation_detection_raw_integer_equality_mutant(self) -> None:
        def mutant_condxchg(mem: int, cmp_val: int, new_val: int) -> Tuple[int, int, bool]:
            if (mem & MASK64) == (cmp_val & MASK64):
                return mem & MASK64, new_val & MASK64, True
            return mem & MASK64, mem & MASK64, False

        _, _, correct_swap = oracle_condxchg64_f64(NEG_ZERO_RAW_F64, POS_ZERO_RAW_F64, ONE_RAW_F64)
        _, _, mutant_swap = mutant_condxchg(NEG_ZERO_RAW_F64, POS_ZERO_RAW_F64, ONE_RAW_F64)
        self.assertTrue(correct_swap)
        self.assertFalse(mutant_swap)
        self.assertNotEqual(correct_swap, mutant_swap)

        _, _, correct_nan_swap = oracle_condxchg64_f64(QNAN_RAW_F64, QNAN_RAW_F64, TWO_RAW_F64)
        _, _, mutant_nan_swap = mutant_condxchg(QNAN_RAW_F64, QNAN_RAW_F64, TWO_RAW_F64)
        self.assertFalse(correct_nan_swap)
        self.assertTrue(mutant_nan_swap)
        self.assertNotEqual(correct_nan_swap, mutant_nan_swap)

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        class MutantSIMDInterpreter(SIMDDSInterpreter):
            def execute(self, state: SIMDWaveState, inst: DSCondXchg32F64Instruction) -> Dict[int, bool]:
                saved = state.exec_mask
                state.exec_mask = (1 << state.wave_size) - 1
                try:
                    return super().execute(state, inst)
                finally:
                    state.exec_mask = saved

        lds = LDSMemory(65536)
        lds.write_qword(40, ONE_RAW_F64)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.set_u32(l, VGPR(1), 40)
            wave.set_u64(l, VGPRPair(4), ONE_RAW_F64)
            wave.set_u64(l, VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
        )
        mutant = MutantSIMDInterpreter()
        mutant.execute(wave, inst)

        self.assertEqual(lds.read_qword(40), TWO_RAW_F64)

        lds.write_qword(40, ONE_RAW_F64)
        SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(lds.read_qword(40), ONE_RAW_F64)

    def test_mutation_detection_dropped_offset(self) -> None:
        class DroppedOffsetInterpreter(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSCondXchg32F64Instruction) -> bool:
                no_off = DSCondXchg32F64Instruction(
                    opcode=inst.opcode,
                    vaddr=inst.vaddr,
                    vdata0=inst.vdata0,
                    vdata1=inst.vdata1,
                    vdst=None,
                    offset=OffsetImmediate(0),
                )
                return super().execute(state, no_off)

        lds = LDSMemory(65536)
        lds.write_qword(16, ONE_RAW_F64)
        lds.write_qword(32, ONE_RAW_F64)
        state = ScalarThreadState(lds=lds)
        state.rf.set_u32(VGPR(1), 16)
        state.rf.set_u64(VGPRPair(4), ONE_RAW_F64)
        state.rf.set_u64(VGPRPair(6), TWO_RAW_F64)

        inst = DSCondXchg32F64Instruction(
            DSOpcode.DS_CONDXCHG32_F64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            offset=OffsetImmediate(16),  # 16 + 16 = 32
        )
        mutant = DroppedOffsetInterpreter()
        mutant.execute(state, inst)

        self.assertEqual(lds.read_qword(16), TWO_RAW_F64)
        self.assertEqual(lds.read_qword(32), ONE_RAW_F64)


if __name__ == "__main__":
    unittest.main()
