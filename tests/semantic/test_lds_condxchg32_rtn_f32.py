"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_condxchg32_rtn_f32.

Models single-precision floating-point (IEEE 754 binary32) conditional exchange
and AMD RDNA2 / GFX10 Local Data Share (LDS) architectural semantics:
  - ds_condxchg32_rtn_f32 (also known as ds_cmpst_rtn_f32):
    Atomic conditional exchange on 32-bit LDS storage using IEEE 754 single-precision
    floating-point comparison.

RDNA2 LDS / Floating-Point Conditional Exchange ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Floating-Point Comparison Semantics:
     The memory value and comparison value are interpreted as IEEE 754 binary32 floats.
     - Equality (+0.0 == -0.0): In IEEE 754, positive and negative zero compare equal.
       If memory contains -0.0 and cmp_val is +0.0, the exchange succeeds.
     - Not-a-Number (NaN != NaN): NaNs compare unordered; NaN == NaN evaluates to False.
       If cmp_val or memory is NaN, the exchange never occurs.
     - Infinities: +INF == +INF evaluates to True, -INF == -INF evaluates to True.
     - Subnormals: Compared according to IEEE 754 binary32 float values.
  3. Return Semantics:
     The instruction atomically reads the existing 32-bit dword from LDS and writes
     it into destination VGPR vdst regardless of whether the comparison succeeded.

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

MASK32: int = 0xFFFFFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000

# Canonical IEEE 754 bit representations
POS_ZERO_RAW: int = 0x00000000
NEG_ZERO_RAW: int = 0x80000000
POS_INF_RAW: int = 0x7F800000
NEG_INF_RAW: int = 0xFF800000
QNAN_RAW: int = 0x7FC00000
SNAN_RAW: int = 0x7F800001
ONE_RAW: int = 0x3F800000      # 1.0f
TWO_RAW: int = 0x40000000      # 2.0f
THREE_RAW: int = 0x40400000    # 3.0f

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def raw_u32_to_f32(raw: int) -> float:
    """Convert raw 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", raw & MASK32))[0]


def f32_to_raw_u32(val: float) -> int:
    """Convert Python float to 32-bit IEEE 754 single-precision bit pattern."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_nan_raw(raw: int) -> bool:
    """Check if 32-bit bit pattern represents an IEEE 754 NaN."""
    return (raw & EXP_MASK) == EXP_MASK and (raw & FRAC_MASK) != 0


def float_equal_f32(a_raw: int, b_raw: int) -> bool:
    """Evaluate IEEE 754 binary32 equality between two raw bit patterns.
    
    Returns True if fa == fb according to IEEE 754 rules.
    Specifically:
      - Any NaN operand yields False (including NaN == NaN).
      - +0.0 == -0.0 yields True.
      - Exact value equality for finite numbers and infinities.
    """
    if is_nan_raw(a_raw) or is_nan_raw(b_raw):
        return False
    fa = raw_u32_to_f32(a_raw)
    fb = raw_u32_to_f32(b_raw)
    return fa == fb


def oracle_condxchg32_f32(
    mem_raw: int,
    cmp_raw: int,
    new_raw: int,
) -> Tuple[int, int, bool]:
    """Reference mathematical oracle for ds_condxchg32_rtn_f32.
    
    Returns:
        (old_val, new_mem_val, exchanged)
    """
    old_val = mem_raw & MASK32
    c_raw = cmp_raw & MASK32
    n_raw = new_raw & MASK32

    if float_equal_f32(old_val, c_raw):
        return old_val, n_raw, True
    return old_val, old_val, False


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        effective_byte = (byte_addr + offset) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def atomic_condxchg32_f32(
        self,
        byte_addr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic conditional exchange at (byte_addr + offset)."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        old_val, new_mem, exchanged = oracle_condxchg32_f32(old_val, cmp_val, new_val)
        self.dwords[idx] = new_mem
        return old_val, new_mem, exchanged


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for reference floating-point conditional exchange oracle and LDS model."""

    def test_oracle_finite_matches_and_mismatches(self) -> None:
        # Match: 1.0 == 1.0 -> exchange to 2.0
        old, mem, swapped = oracle_condxchg32_f32(ONE_RAW, ONE_RAW, TWO_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(mem, TWO_RAW)

        # Mismatch: 1.0 != 2.0 -> no exchange
        old, mem, swapped = oracle_condxchg32_f32(ONE_RAW, TWO_RAW, THREE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(mem, ONE_RAW)

    def test_oracle_signed_zero_equality(self) -> None:
        # IEEE 754: +0.0 == -0.0 is True
        old, mem, swapped = oracle_condxchg32_f32(NEG_ZERO_RAW, POS_ZERO_RAW, ONE_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_RAW)
        self.assertEqual(mem, ONE_RAW)

        old2, mem2, swapped2 = oracle_condxchg32_f32(POS_ZERO_RAW, NEG_ZERO_RAW, TWO_RAW)
        self.assertTrue(swapped2)
        self.assertEqual(old2, POS_ZERO_RAW)
        self.assertEqual(mem2, TWO_RAW)

    def test_oracle_nan_unordered_rejection(self) -> None:
        # NaN == NaN is always False in IEEE 754
        old, mem, swapped = oracle_condxchg32_f32(QNAN_RAW, QNAN_RAW, ONE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(old, QNAN_RAW)
        self.assertEqual(mem, QNAN_RAW)

        # Signaling NaN rejection
        old_s, mem_s, swapped_s = oracle_condxchg32_f32(SNAN_RAW, SNAN_RAW, TWO_RAW)
        self.assertFalse(swapped_s)
        self.assertEqual(old_s, SNAN_RAW)
        self.assertEqual(mem_s, SNAN_RAW)

    def test_oracle_infinities(self) -> None:
        # +inf == +inf
        old, mem, swapped = oracle_condxchg32_f32(POS_INF_RAW, POS_INF_RAW, ONE_RAW)
        self.assertTrue(swapped)
        self.assertEqual(mem, ONE_RAW)

        # +inf != -inf
        old, mem, swapped = oracle_condxchg32_f32(POS_INF_RAW, NEG_INF_RAW, ONE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(mem, POS_INF_RAW)

    def test_lds_dword_indexing_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.get_dword_index(0), 0)
        self.assertEqual(lds.get_dword_index(1), 0)
        self.assertEqual(lds.get_dword_index(2), 0)
        self.assertEqual(lds.get_dword_index(3), 0)
        self.assertEqual(lds.get_dword_index(4), 1)

    def test_lds_offset_and_wrapping(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.get_dword_index(0, offset=8), 2)
        self.assertEqual(lds.get_dword_index(65536), 0)

    def test_lds_bank_distribution(self) -> None:
        lds = LDSMemory(65536)
        for i in range(32):
            self.assertEqual(lds.get_bank(i), i)
        self.assertEqual(lds.get_bank(32), 0)

    def test_lds_atomic_condxchg32_f32_execution(self) -> None:
        lds = LDSMemory(65536)
        lds.write_dword(5, ONE_RAW)

        # Match at dword 5
        old, mem, swapped = lds.atomic_condxchg32_f32(byte_addr=20, cmp_val=ONE_RAW, new_val=TWO_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(lds.read_dword(5), TWO_RAW)

        # Mismatch at dword 5
        old2, mem2, swapped2 = lds.atomic_condxchg32_f32(byte_addr=20, cmp_val=THREE_RAW, new_val=ONE_RAW)
        self.assertFalse(swapped2)
        self.assertEqual(old2, TWO_RAW)
        self.assertEqual(lds.read_dword(5), TWO_RAW)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for 32-bit floating-point conditional exchange."""
    DS_CONDXCHG32_RTN_F32 = "ds_condxchg32_rtn_f32"
    DS_CMPST_RTN_F32 = "ds_cmpst_rtn_f32"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR."""
        return True

    @property
    def bit_width(self) -> int:
        """Operand bit width."""
        return 32


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
class OffsetImmediate:
    """Unsigned 16-bit byte offset immediate [0, 65535]."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"Offset immediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSCondXchg32F32Instruction:
    """Symbolic AST node representing an RDNA2 DS conditional exchange float instruction."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata0: VGPR
    vdata1: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = OffsetImmediate(0)

    def __post_init__(self) -> None:
        if self.vdst is None:
            raise ValueError(f"Returning opcode {self.opcode.value} requires destination vdst")

    def disasm(self) -> str:
        """Format symbolic instruction into canonical RDNA2 assembly text."""
        parts = [self.opcode.value, f"{self.vdst}, {self.vaddr}, {self.vdata0}, {self.vdata1}"]
        if self.offset.value != 0:
            parts.append(str(self.offset))
        return " ".join(parts)

    @classmethod
    def parse(cls, text: str) -> DSCondXchg32F32Instruction:
        """Parse canonical RDNA2 assembly string into a DSCondXchg32F32Instruction AST."""
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

        if len(reg_tokens) != 4:
            raise ValueError(f"{opcode.value} expects 4 registers (vdst, vaddr, vdata0, vdata1), got {reg_tokens}")

        vdst = parse_vgpr(reg_tokens[0])
        vaddr = parse_vgpr(reg_tokens[1])
        vdata0 = parse_vgpr(reg_tokens[2])
        vdata1 = parse_vgpr(reg_tokens[3])
        return cls(
            opcode=opcode,
            vaddr=vaddr,
            vdata0=vdata0,
            vdata1=vdata1,
            vdst=vdst,
            offset=OffsetImmediate(offset_val),
        )


class TestSymbolicASTAndTypedExpressions(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions, registers, and assembly parsing."""

    def test_opcode_properties(self) -> None:
        self.assertTrue(DSOpcode.DS_CONDXCHG32_RTN_F32.is_return)
        self.assertEqual(DSOpcode.DS_CONDXCHG32_RTN_F32.bit_width, 32)
        self.assertTrue(DSOpcode.DS_CMPST_RTN_F32.is_return)
        self.assertEqual(DSOpcode.DS_CMPST_RTN_F32.bit_width, 32)

    def test_vgpr_validation(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")

        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

    def test_offset_immediate_validation(self) -> None:
        off0 = OffsetImmediate(0)
        off64 = OffsetImmediate(64)
        off_max = OffsetImmediate(65535)
        self.assertEqual(str(off0), "")
        self.assertEqual(str(off64), "offset:64")
        self.assertEqual(str(off_max), "offset:65535")

        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

    def test_instruction_post_init_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSCondXchg32F32Instruction(
                opcode=DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(0),
                vdata0=VGPR(1),
                vdata1=VGPR(2),
                vdst=None,
            )

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSCondXchg32F32Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(1),
                vdata0=VGPR(2),
                vdata1=VGPR(3),
                vdst=VGPR(0),
            ),
            DSCondXchg32F32Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(1),
                vdata0=VGPR(2),
                vdata1=VGPR(3),
                vdst=VGPR(0),
                offset=OffsetImmediate(16),
            ),
            DSCondXchg32F32Instruction(
                DSOpcode.DS_CMPST_RTN_F32,
                vaddr=VGPR(10),
                vdata0=VGPR(11),
                vdata1=VGPR(12),
                vdst=VGPR(9),
                offset=OffsetImmediate(64),
            ),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSCondXchg32F32Instruction.parse(asm)
                self.assertEqual(inst, parsed)


class ScalarRegisterFile:
    """Architectural register file for a single wavefront lane (256 32-bit VGPRs)."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * 256

    def get(self, reg: VGPR) -> int:
        """Read 32-bit VGPR value."""
        return self.regs[reg.index] & MASK32

    def set(self, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR value."""
        self.regs[reg.index] = val & MASK32


class ScalarThreadState:
    """Execution state of a single scalar thread executing LDS instructions."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def get_vgpr(self, reg: VGPR) -> int:
        return self.rf.get(reg)

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.rf.set(reg, val)


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 DS conditional exchange f32."""

    def execute(self, state: ScalarThreadState, inst: DSCondXchg32F32Instruction) -> int:
        """Execute a single DS conditional exchange f32 instruction."""
        addr = state.get_vgpr(inst.vaddr)
        cmp_val = state.get_vgpr(inst.vdata0)
        new_val = state.get_vgpr(inst.vdata1)

        old_val, new_mem, exchanged = state.lds.atomic_condxchg32_f32(
            addr, cmp_val, new_val, offset=inst.offset.value
        )

        assert inst.vdst is not None
        state.set_vgpr(inst.vdst, old_val)
        return old_val


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution of ds_condxchg32_rtn_f32."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_exchange_success_on_match(self) -> None:
        self.lds.write_dword(5, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 20)      # dword 5
        self.state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val = 1.0f
        self.state.set_vgpr(VGPR(3), TWO_RAW)  # new_val = 2.0f

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), ONE_RAW)
        self.assertEqual(self.lds.read_dword(5), TWO_RAW)

    def test_scalar_exchange_failure_on_mismatch(self) -> None:
        self.lds.write_dword(8, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 32)        # dword 8
        self.state.set_vgpr(VGPR(2), THREE_RAW)  # cmp_val = 3.0f (mismatch)
        self.state.set_vgpr(VGPR(3), TWO_RAW)    # new_val = 2.0f

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(4),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(4)), ONE_RAW)
        self.assertEqual(self.lds.read_dword(8), ONE_RAW)  # Unchanged

    def test_scalar_exchange_with_offset(self) -> None:
        self.lds.write_dword(10, TWO_RAW)
        self.state.set_vgpr(VGPR(1), 36)        # dword 9
        self.state.set_vgpr(VGPR(2), TWO_RAW)   # cmp_val = 2.0f
        self.state.set_vgpr(VGPR(3), THREE_RAW) # new_val = 3.0f

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(5),
            offset=OffsetImmediate(4),  # offset 4 -> dword 10
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, TWO_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(5)), TWO_RAW)
        self.assertEqual(self.lds.read_dword(10), THREE_RAW)

    def test_scalar_signed_zero_exchange(self) -> None:
        # Memory contains -0.0, comparison is +0.0
        self.lds.write_dword(2, NEG_ZERO_RAW)
        self.state.set_vgpr(VGPR(1), 8)
        self.state.set_vgpr(VGPR(2), POS_ZERO_RAW)
        self.state.set_vgpr(VGPR(3), ONE_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(6),
        )
        old = self.interp.execute(self.state, inst)

        # Exchange succeeds; vdst gets -0.0 exact bit pattern
        self.assertEqual(old, NEG_ZERO_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(6)), NEG_ZERO_RAW)
        self.assertEqual(self.lds.read_dword(2), ONE_RAW)

    def test_scalar_sequential_cas_loop(self) -> None:
        self.lds.write_dword(0, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 0)

        # Step 1: cas(expected=1.0, desired=2.0) -> succeeds
        self.state.set_vgpr(VGPR(2), ONE_RAW)
        self.state.set_vgpr(VGPR(3), TWO_RAW)
        r1 = self.interp.execute(
            self.state,
            DSCondXchg32F32Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(1), vdata0=VGPR(2), vdata1=VGPR(3), vdst=VGPR(10)
            )
        )
        self.assertEqual(r1, ONE_RAW)
        self.assertEqual(self.lds.read_dword(0), TWO_RAW)

        # Step 2: cas(expected=1.0, desired=3.0) -> fails because current is 2.0
        r2 = self.interp.execute(
            self.state,
            DSCondXchg32F32Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(1), vdata0=VGPR(2), vdata1=VGPR(3), vdst=VGPR(11)
            )
        )
        self.assertEqual(r2, TWO_RAW)
        self.assertEqual(self.lds.read_dword(0), TWO_RAW)


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
        """Return True if the given lane is enabled in the current EXEC mask."""
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg: VGPR) -> int:
        """Read 32-bit VGPR for a specific lane."""
        return self.vgprs[lane][reg.index] & MASK32

    def set_vgpr(self, lane: int, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR for a specific lane."""
        self.vgprs[lane][reg.index] = val & MASK32


class SIMDDSInterpreter:
    """Multi-lane SIMD instruction interpreter modeling RDNA2 lockstep execution and LDS atomics."""

    def execute(self, state: SIMDWaveState, inst: DSCondXchg32F32Instruction) -> Dict[int, int]:
        """Execute DS conditional exchange f32 across all active lanes of the wave."""
        active_lanes = [lane for lane in range(state.wave_size) if state.is_lane_active(lane)]
        latched_inputs = [
            (
                lane,
                state.get_vgpr(lane, inst.vaddr),
                state.get_vgpr(lane, inst.vdata0),
                state.get_vgpr(lane, inst.vdata1),
            )
            for lane in active_lanes
        ]

        returns: Dict[int, int] = {}

        for lane, addr, cmp_val, new_val in latched_inputs:
            old_val, new_mem, exchanged = state.lds.atomic_condxchg32_f32(
                addr, cmp_val, new_val, offset=inst.offset.value
            )
            assert inst.vdst is not None
            state.set_vgpr(lane, inst.vdst, old_val)
            returns[lane] = old_val

        return returns


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD lockstep execution, 32/64-bit EXEC masking, and address convergence."""

    def test_wave32_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Initialize LDS dwords 0..31 with 1.0f
        for i in range(32):
            lds.write_dword(i, ONE_RAW)

        # Lane i targets dword i, expects 1.0f, swaps to float(i + 10)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), ONE_RAW)
            wave.set_vgpr(lane, VGPR(3), f32_to_raw_u32(float(lane + 10)))

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        returns = interp.execute(wave, inst)

        for i in range(32):
            self.assertEqual(returns[i], ONE_RAW)
            self.assertEqual(wave.get_vgpr(i, VGPR(0)), ONE_RAW)
            self.assertEqual(lds.read_dword(i), f32_to_raw_u32(float(i + 10)))

    def test_wave64_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(64):
            lds.write_dword(i, TWO_RAW)

        for lane in range(64):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), TWO_RAW)
            wave.set_vgpr(lane, VGPR(3), THREE_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(4),
        )
        returns = interp.execute(wave, inst)

        for i in range(64):
            self.assertEqual(returns[i], TWO_RAW)
            self.assertEqual(lds.read_dword(i), THREE_RAW)

    def test_wave32_partial_exec_masking(self) -> None:
        lds = LDSMemory(65536)
        # Even lanes only: 0x55555555
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_dword(i, ONE_RAW)

        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), ONE_RAW)
            wave.set_vgpr(lane, VGPR(3), TWO_RAW)
            wave.set_vgpr(lane, VGPR(0), 0xCAFEBABE)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(lds.read_dword(lane), TWO_RAW)
                self.assertEqual(wave.get_vgpr(lane, VGPR(0)), ONE_RAW)
                self.assertIn(lane, returns)
            else:
                self.assertEqual(lds.read_dword(lane), ONE_RAW)
                self.assertEqual(wave.get_vgpr(lane, VGPR(0)), 0xCAFEBABE)
                self.assertNotIn(lane, returns)

    def test_zero_exec_mask_no_mutation(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_dword(0, ONE_RAW)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), 0)
            wave.set_vgpr(lane, VGPR(2), ONE_RAW)
            wave.set_vgpr(lane, VGPR(3), TWO_RAW)
            wave.set_vgpr(lane, VGPR(5), 0x12345678)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(5),
        )
        returns = interp.execute(wave, inst)

        self.assertEqual(len(returns), 0)
        self.assertEqual(lds.read_dword(0), ONE_RAW)
        for lane in range(32):
            self.assertEqual(wave.get_vgpr(lane, VGPR(5)), 0x12345678)

    def test_convergent_collision_cas_daisy_chain(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Initial value at dword 4 is 0.0f
        lds.write_dword(4, POS_ZERO_RAW)

        # Lane i expects float(i) and exchanges to float(i + 1)
        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), 16)  # dword 4
            wave.set_vgpr(lane, VGPR(2), f32_to_raw_u32(float(lane)))
            wave.set_vgpr(lane, VGPR(3), f32_to_raw_u32(float(lane + 1)))

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        returns = interp.execute(wave, inst)

        # Daisy chain: lane 0 saw 0.0, wrote 1.0; lane 1 saw 1.0, wrote 2.0; ...
        # Final value at dword 4 must be 32.0f
        self.assertEqual(lds.read_dword(4), f32_to_raw_u32(32.0))
        for lane in range(32):
            self.assertEqual(returns[lane], f32_to_raw_u32(float(lane)))


if __name__ == "__main__":
    unittest.main()
