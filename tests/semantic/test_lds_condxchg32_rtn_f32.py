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


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_dst_aliases_addr(self) -> None:
        self.lds.write_dword(3, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 12)       # byte 12 -> dword 3
        self.state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val
        self.state.set_vgpr(VGPR(3), TWO_RAW)  # new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(1),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(1)), ONE_RAW)
        self.assertEqual(self.lds.read_dword(3), TWO_RAW)

    def test_scalar_dst_aliases_data0(self) -> None:
        self.lds.write_dword(6, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 24)       # byte 24 -> dword 6
        self.state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val & vdst
        self.state.set_vgpr(VGPR(3), THREE_RAW)# new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), ONE_RAW)
        self.assertEqual(self.lds.read_dword(6), THREE_RAW)

    def test_scalar_dst_aliases_data1(self) -> None:
        self.lds.write_dword(7, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 28)       # byte 28 -> dword 7
        self.state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val
        self.state.set_vgpr(VGPR(3), TWO_RAW)  # new_val & vdst

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(3),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(3)), ONE_RAW)
        self.assertEqual(self.lds.read_dword(7), TWO_RAW)

    def test_scalar_data0_aliases_data1(self) -> None:
        self.lds.write_dword(4, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 16)
        self.state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val == new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(2),
            vdst=VGPR(0),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, ONE_RAW)
        self.assertEqual(self.lds.read_dword(4), ONE_RAW)

    def test_scalar_quad_alias(self) -> None:
        raw_val = f32_to_raw_u32(32.0)
        self.lds.write_dword(8, raw_val)
        self.state.set_vgpr(VGPR(0), 32)  # byte 32 -> dword 8

        # In this test, memory dword 8 has raw_val = 32.0f
        # v0 has byte address 32. But 32 as a raw float is subnormal != 32.0f
        # So comparison between memory (32.0f) and v0 (32 as float) fails
        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(0),
            vdata0=VGPR(0),
            vdata1=VGPR(0),
            vdst=VGPR(0),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, raw_val)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), raw_val)
        # Memory unchanged
        self.assertEqual(self.lds.read_dword(8), raw_val)

    def test_operand_preservation_non_aliased(self) -> None:
        self.lds.write_dword(10, ONE_RAW)
        self.state.set_vgpr(VGPR(1), 40)
        self.state.set_vgpr(VGPR(2), ONE_RAW)
        self.state.set_vgpr(VGPR(3), TWO_RAW)
        self.state.set_vgpr(VGPR(4), 0xCAFEBABE)
        self.state.set_vgpr(VGPR(255), 0x12344321)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(4),
        )
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.get_vgpr(VGPR(1)), 40)
        self.assertEqual(self.state.get_vgpr(VGPR(2)), ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(3)), TWO_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(4)), ONE_RAW)
        self.assertEqual(self.state.get_vgpr(VGPR(255)), 0x12344321)

    def test_simd_aliasing_under_divergence(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xAAAAAAAA, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_dword(i, ONE_RAW)

        for lane in range(32):
            wave.set_vgpr(lane, VGPR(1), lane * 4)
            wave.set_vgpr(lane, VGPR(2), ONE_RAW)
            wave.set_vgpr(lane, VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(1),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 != 0:
                self.assertEqual(wave.get_vgpr(lane, VGPR(1)), ONE_RAW)
                self.assertEqual(lds.read_dword(lane), TWO_RAW)
                self.assertEqual(returns[lane], ONE_RAW)
            else:
                self.assertEqual(wave.get_vgpr(lane, VGPR(1)), lane * 4)
                self.assertEqual(lds.read_dword(lane), ONE_RAW)
                self.assertNotIn(lane, returns)


@dataclasses.dataclass
class LDSStatusFlags:
    """Architectural status flags, wrapping conditions, and bank metrics for LDS operations."""
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
    inst: DSCondXchg32F32Instruction,
) -> LDSStatusFlags:
    """Evaluate architectural flags, address wrap conditions, bank conflicts, and ALU status."""
    flags = LDSStatusFlags()
    active_lanes = [l for l in range(wave.wave_size) if wave.is_lane_active(l)]

    bank_to_dwords: Dict[int, Set[int]] = {}
    dword_to_lanes: Dict[int, List[int]] = {}

    for lane in active_lanes:
        addr = wave.get_vgpr(lane, inst.vaddr)
        off = inst.offset.value
        cmp_val = wave.get_vgpr(lane, inst.vdata0)
        new_val = wave.get_vgpr(lane, inst.vdata1)

        if (addr + off) > MASK32:
            flags.carry_out = True

        effective_byte = (addr + off) & MASK32
        if (effective_byte & 3) != 0:
            flags.unaligned_address = True

        if effective_byte >= wave.lds.size_bytes:
            flags.oob_wrapped = True

        dword_idx = wave.lds.get_dword_index(addr, off)
        bank_idx = wave.lds.get_bank(dword_idx)

        bank_to_dwords.setdefault(bank_idx, set()).add(dword_idx)
        dword_to_lanes.setdefault(dword_idx, []).append(lane)

        old_val = wave.lds.read_dword(dword_idx)
        _, new_mem, swapped = oracle_condxchg32_f32(old_val, cmp_val, new_val)
        if swapped:
            flags.exchange_occurred = True
        else:
            flags.exchange_failed = True

        if (new_mem & 0x7FFFFFFF) == 0:
            flags.result_is_zero = True
        if bool(new_mem & 0x80000000):
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
        wave.set_vgpr(0, VGPR(1), 0xFFFFFFFF)
        wave.set_vgpr(0, VGPR(2), ONE_RAW)
        wave.set_vgpr(0, VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
            offset=OffsetImmediate(16),
        )
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.carry_out)

    def test_unaligned_byte_address_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 7)
        wave.set_vgpr(0, VGPR(2), ONE_RAW)
        wave.set_vgpr(0, VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.unaligned_address)

    def test_oob_address_wrapping_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 70000)
        wave.set_vgpr(0, VGPR(2), ONE_RAW)
        wave.set_vgpr(0, VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        flags = evaluate_status_flags(wave, inst)

        self.assertTrue(flags.oob_wrapped)

    def test_bank_conflicts_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x7)
        wave.set_vgpr(0, VGPR(1), 0 * 4)   # bank 0, dword 0
        wave.set_vgpr(1, VGPR(1), 32 * 4)  # bank 0, dword 32
        wave.set_vgpr(2, VGPR(1), 64 * 4)  # bank 0, dword 64

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        flags = evaluate_status_flags(wave, inst)

        self.assertEqual(flags.bank_conflicts, 2)
        self.assertEqual(flags.address_collisions, 0)

    def test_address_collisions_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xF)
        for l in range(4):
            wave.set_vgpr(l, VGPR(1), 40)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        flags = evaluate_status_flags(wave, inst)

        self.assertEqual(flags.address_collisions, 3)
        self.assertEqual(flags.bank_conflicts, 0)

    def test_exchange_outcome_and_result_flags(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x3)
        wave.lds.write_dword(0, ONE_RAW)
        wave.lds.write_dword(1, TWO_RAW)

        # Lane 0: match (1.0 == 1.0) -> exchanges to -0.0 (zero and sign bit set!)
        wave.set_vgpr(0, VGPR(1), 0)
        wave.set_vgpr(0, VGPR(2), ONE_RAW)
        wave.set_vgpr(0, VGPR(3), NEG_ZERO_RAW)

        # Lane 1: mismatch (2.0 != 3.0) -> exchange fails
        wave.set_vgpr(1, VGPR(1), 4)
        wave.set_vgpr(1, VGPR(2), THREE_RAW)
        wave.set_vgpr(1, VGPR(3), ONE_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
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
        # Signaling NaN: exponent=0xFF, fraction != 0, quiet bit 22 is 0
        snan32 = 0x7F800001
        self.assertEqual(snan32 & QNAN_BIT, 0)

        self.lds.write_dword(1, snan32)
        self.state.set_vgpr(VGPR(1), 4)        # dword 1
        self.state.set_vgpr(VGPR(2), snan32)   # cmp_val
        self.state.set_vgpr(VGPR(3), ONE_RAW)  # new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(4),
        )
        old = self.interp.execute(self.state, inst)

        # NaN == NaN is False in IEEE 754, so exchange must not happen
        self.assertEqual(old, snan32)
        self.assertEqual(old & QNAN_BIT, 0)  # Quiet bit NOT set
        self.assertEqual(self.state.get_vgpr(VGPR(4)), snan32)
        self.assertEqual(self.lds.read_dword(1), snan32)

    def test_qnan_payload_preservation_and_comparison_failure(self) -> None:
        # Quiet NaN with payload
        qnan32 = 0x7FC01234
        self.assertNotEqual(qnan32 & QNAN_BIT, 0)

        self.lds.write_dword(2, qnan32)
        self.state.set_vgpr(VGPR(1), 8)
        self.state.set_vgpr(VGPR(2), qnan32)
        self.state.set_vgpr(VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(5),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, qnan32)
        self.assertEqual(self.state.get_vgpr(VGPR(5)), qnan32)
        self.assertEqual(self.lds.read_dword(2), qnan32)

    def test_signed_zeros_both_directions(self) -> None:
        # Direction 1: memory is -0.0, cmp is +0.0
        self.lds.write_dword(3, NEG_ZERO_RAW)
        self.state.set_vgpr(VGPR(1), 12)
        self.state.set_vgpr(VGPR(2), POS_ZERO_RAW)
        self.state.set_vgpr(VGPR(3), THREE_RAW)

        inst1 = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(6),
        )
        old1 = self.interp.execute(self.state, inst1)

        self.assertEqual(old1, NEG_ZERO_RAW)
        self.assertEqual(self.lds.read_dword(3), THREE_RAW)

        # Direction 2: memory is +0.0, cmp is -0.0
        self.lds.write_dword(4, POS_ZERO_RAW)
        self.state.set_vgpr(VGPR(1), 16)
        self.state.set_vgpr(VGPR(2), NEG_ZERO_RAW)
        self.state.set_vgpr(VGPR(3), TWO_RAW)

        inst2 = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(7),
        )
        old2 = self.interp.execute(self.state, inst2)

        self.assertEqual(old2, POS_ZERO_RAW)
        self.assertEqual(self.lds.read_dword(4), TWO_RAW)

    def test_infinities_matching_and_sign_distinction(self) -> None:
        # +inf == +inf
        self.lds.write_dword(5, POS_INF_RAW)
        self.state.set_vgpr(VGPR(1), 20)
        self.state.set_vgpr(VGPR(2), POS_INF_RAW)
        self.state.set_vgpr(VGPR(3), ONE_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(8),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, POS_INF_RAW)
        self.assertEqual(self.lds.read_dword(5), ONE_RAW)

        # -inf != +inf
        self.lds.write_dword(6, NEG_INF_RAW)
        self.state.set_vgpr(VGPR(1), 24)
        self.state.set_vgpr(VGPR(2), POS_INF_RAW)
        self.state.set_vgpr(VGPR(3), TWO_RAW)

        inst_neg = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(9),
        )
        old_neg = self.interp.execute(self.state, inst_neg)

        self.assertEqual(old_neg, NEG_INF_RAW)
        self.assertEqual(self.lds.read_dword(6), NEG_INF_RAW)

    def test_subnormals_preservation_and_exact_match(self) -> None:
        subnorm_min = 0x00000001
        subnorm_max = 0x007FFFFF

        # Match on subnormal min
        self.lds.write_dword(7, subnorm_min)
        self.state.set_vgpr(VGPR(1), 28)
        self.state.set_vgpr(VGPR(2), subnorm_min)
        self.state.set_vgpr(VGPR(3), subnorm_max)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(10),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, subnorm_min)
        self.assertEqual(self.lds.read_dword(7), subnorm_max)

        # Subnormal != 0.0
        self.lds.write_dword(8, subnorm_min)
        self.state.set_vgpr(VGPR(1), 32)
        self.state.set_vgpr(VGPR(2), POS_ZERO_RAW)  # cmp with 0.0
        self.state.set_vgpr(VGPR(3), ONE_RAW)

        inst_zero = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(11),
        )
        old_zero = self.interp.execute(self.state, inst_zero)

        self.assertEqual(old_zero, subnorm_min)
        self.assertEqual(self.lds.read_dword(8), subnorm_min)

    def test_finite_boundary_extremes(self) -> None:
        flt_max = 0x7F7FFFFF
        flt_min = 0x00800000

        # Exact match at flt_max
        self.lds.write_dword(9, flt_max)
        self.state.set_vgpr(VGPR(1), 36)
        self.state.set_vgpr(VGPR(2), flt_max)
        self.state.set_vgpr(VGPR(3), flt_min)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(12),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, flt_max)
        self.assertEqual(self.lds.read_dword(9), flt_min)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_syntax(self) -> None:
        invalid_assemblies = [
            "",
            "   ",
            "v_condxchg_f32 v0, v1, v2, v3",
            "ds_condxchg64_rtn_f32 v0, v1, v2, v3",
            "ds_min_f32 v0, v1",
            "ds_condxchg32_rtn_f32 v0, v1, v2",  # Missing 4th register
            "ds_condxchg32_rtn_f32 v0, v1, v2, v3, v4",  # Too many registers
            "ds_condxchg32_rtn_f32 r0, v1, v2, v3",
            "ds_condxchg32_rtn_f32 v0, s1, v2, v3",
            "ds_condxchg32_rtn_f32 v256, v1, v2, v3",
            "ds_condxchg32_rtn_f32 v0, v1, v2, v3 offset:70000",
            "ds_condxchg32_rtn_f32 v0, v1, v2, v3 offset:-1",
            "ds_condxchg32_rtn_f32 v0, v1, v2, v3 offset:",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, IndexError)):
                    DSCondXchg32F32Instruction.parse(asm)

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
            DSCondXchg32F32Instruction(
                opcode=DSOpcode.DS_CONDXCHG32_RTN_F32,
                vaddr=VGPR(0),
                vdata0=VGPR(1),
                vdata1=VGPR(2),
                vdst=None,
            )

    def test_mutation_detection_raw_integer_equality_mutant(self) -> None:
        # Mutant: uses raw integer equality (a_raw == b_raw) instead of IEEE 754 float_equal_f32
        def mutant_condxchg(mem: int, cmp_val: int, new_val: int) -> Tuple[int, int, bool]:
            if (mem & MASK32) == (cmp_val & MASK32):
                return mem & MASK32, new_val & MASK32, True
            return mem & MASK32, mem & MASK32, False

        # Mutation bug 1: +0.0 vs -0.0
        # True float equality: +0.0 == -0.0 (swapped is True)
        # Mutant integer equality: 0x0 != 0x80000000 (swapped is False)
        _, _, correct_swap = oracle_condxchg32_f32(NEG_ZERO_RAW, POS_ZERO_RAW, ONE_RAW)
        _, _, mutant_swap = mutant_condxchg(NEG_ZERO_RAW, POS_ZERO_RAW, ONE_RAW)
        self.assertTrue(correct_swap)
        self.assertFalse(mutant_swap)
        self.assertNotEqual(correct_swap, mutant_swap)

        # Mutation bug 2: NaN vs NaN
        # True float equality: NaN == NaN is False (swapped is False)
        # Mutant integer equality: NaN == NaN is True (swapped is True)
        _, _, correct_nan_swap = oracle_condxchg32_f32(QNAN_RAW, QNAN_RAW, TWO_RAW)
        _, _, mutant_nan_swap = mutant_condxchg(QNAN_RAW, QNAN_RAW, TWO_RAW)
        self.assertFalse(correct_nan_swap)
        self.assertTrue(mutant_nan_swap)
        self.assertNotEqual(correct_nan_swap, mutant_nan_swap)

    def test_mutation_detection_destructive_aliasing(self) -> None:
        class DestructiveMutantInterpreter(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSCondXchg32F32Instruction) -> int:
                # Clobber destination register before memory access
                state.set_vgpr(inst.vdst, 0xBAD00000)
                return super().execute(state, inst)

        lds = LDSMemory(65536)
        lds.write_dword(2, ONE_RAW)
        state = ScalarThreadState(lds=lds)
        state.set_vgpr(VGPR(1), 8)        # dword 2
        state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val
        state.set_vgpr(VGPR(3), TWO_RAW)  # new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(1),
        )
        mutant = DestructiveMutantInterpreter()
        mutant.execute(state, inst)

        # Destructive mutant clobbers vaddr (VGPR 1) before memory access in super().execute
        self.assertNotEqual(lds.read_dword(2), TWO_RAW)

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        class MutantSIMDInterpreter(SIMDDSInterpreter):
            def execute(self, state: SIMDWaveState, inst: DSCondXchg32F32Instruction) -> Dict[int, int]:
                saved = state.exec_mask
                state.exec_mask = (1 << state.wave_size) - 1
                try:
                    return super().execute(state, inst)
                finally:
                    state.exec_mask = saved

        lds = LDSMemory(65536)
        lds.write_dword(5, ONE_RAW)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.set_vgpr(l, VGPR(1), 20)
            wave.set_vgpr(l, VGPR(2), ONE_RAW)
            wave.set_vgpr(l, VGPR(3), TWO_RAW)

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(0),
        )
        mutant = MutantSIMDInterpreter()
        mutant.execute(wave, inst)

        # Mutant executed on inactive lanes
        self.assertEqual(lds.read_dword(5), TWO_RAW)

        # Proper interpreter preserves memory
        lds.write_dword(5, ONE_RAW)
        SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(lds.read_dword(5), ONE_RAW)

    def test_mutation_detection_dropped_offset(self) -> None:
        class DroppedOffsetInterpreter(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSCondXchg32F32Instruction) -> int:
                no_off = DSCondXchg32F32Instruction(
                    opcode=inst.opcode,
                    vaddr=inst.vaddr,
                    vdata0=inst.vdata0,
                    vdata1=inst.vdata1,
                    vdst=inst.vdst,
                    offset=OffsetImmediate(0),
                )
                return super().execute(state, no_off)

        lds = LDSMemory(65536)
        lds.write_dword(1, ONE_RAW)
        lds.write_dword(2, ONE_RAW)
        state = ScalarThreadState(lds=lds)
        state.set_vgpr(VGPR(1), 4)        # dword 1
        state.set_vgpr(VGPR(2), ONE_RAW)  # cmp_val
        state.set_vgpr(VGPR(3), TWO_RAW)  # new_val

        inst = DSCondXchg32F32Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_F32,
            vaddr=VGPR(1),
            vdata0=VGPR(2),
            vdata1=VGPR(3),
            vdst=VGPR(4),
            offset=OffsetImmediate(4),  # offset 4 -> dword 2
        )
        mutant = DroppedOffsetInterpreter()
        mutant.execute(state, inst)

        # Mutant targeted dword 1 instead of dword 2
        self.assertEqual(lds.read_dword(1), TWO_RAW)
        self.assertEqual(lds.read_dword(2), ONE_RAW)


if __name__ == "__main__":
    unittest.main()
