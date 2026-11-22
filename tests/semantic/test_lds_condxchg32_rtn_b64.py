"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_condxchg32_rtn_b64.

Models 64-bit Local Data Share (LDS) conditional exchange (compare-and-swap)
and AMD RDNA2 / GFX10 Local Data Share (LDS) architectural semantics:
  - ds_condxchg32_rtn_b64 (also known as ds_cmpst_rtn_b64):
    Atomic 64-bit conditional exchange on LDS storage returning previous 64-bit value.

RDNA2 LDS / 64-Bit Conditional Exchange ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     A 64-bit element occupies two consecutive 32-bit dwords: low dword at dword_idx,
     high dword at (dword_idx + 1).
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. 64-Bit Conditional Exchange Semantics:
     Memory value M[addr] is compared with 64-bit cmp_val.
     If M[addr] == cmp_val:
         M[addr] = new_val
         exchanged = True
     Else:
         exchanged = False
  3. Return Semantics:
     The instruction atomically reads the existing 64-bit qword from LDS and writes
     it into destination VGPR pair vdst[0:1] regardless of whether the comparison succeeded.

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
ALL_ONES_B64: int = 0xFFFFFFFFFFFFFFFF
ALL_ZEROS_B64: int = 0x0000000000000000

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def split64(val: int) -> Tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32)."""
    return val & MASK32, (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    """Combine two 32-bit integers into unsigned 64-bit integer."""
    return ((lo & MASK32) | ((hi & MASK32) << 32)) & MASK64


def oracle_condxchg64(
    mem_val: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Reference mathematical oracle for 64-bit conditional exchange.
    
    Returns:
        (old_val, new_mem_val, exchanged)
    """
    m = mem_val & MASK64
    c = cmp_val & MASK64
    n = new_val & MASK64

    if m == c:
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

    def atomic_condxchg64(
        self,
        byte_addr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 64-bit conditional exchange at (byte_addr + offset)."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo = self.dwords[idx0]
        hi = self.dwords[idx1]
        old_val = join64(lo, hi)

        old, new_mem, exchanged = oracle_condxchg64(old_val, cmp_val, new_val)
        if exchanged:
            new_lo, new_hi = split64(new_mem)
            self.dwords[idx0] = new_lo
            self.dwords[idx1] = new_hi

        return old, new_mem, exchanged


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for reference 64-bit conditional exchange oracle and LDS model."""

    def test_split_and_join_roundtrip(self) -> None:
        values = [0x0, 0x1, 0xFFFFFFFF, 0x123456789ABCDEF0, 0xFFFFFFFFFFFFFFFF]
        for v in values:
            with self.subTest(v=hex(v)):
                lo, hi = split64(v)
                self.assertEqual(join64(lo, hi), v)

    def test_oracle_match_and_mismatch(self) -> None:
        val_a = 0x1111222233334444
        val_b = 0x5555666677778888
        val_c = 0x9999AAAABBBBCCCC

        # Match -> exchanges to val_c
        old, mem, swapped = oracle_condxchg64(val_a, val_a, val_c)
        self.assertTrue(swapped)
        self.assertEqual(old, val_a)
        self.assertEqual(mem, val_c)

        # Mismatch -> preserves val_a
        old2, mem2, swapped2 = oracle_condxchg64(val_a, val_b, val_c)
        self.assertFalse(swapped2)
        self.assertEqual(old2, val_a)
        self.assertEqual(mem2, val_a)

    def test_oracle_partial_mismatch_low_vs_high(self) -> None:
        # Match only in low 32 bits
        val1 = 0xAAAAAAAAAAAAAAAA_12345678
        val2 = 0xBBBBBBBBBBBBBBBB_12345678
        old, mem, swapped = oracle_condxchg64(val1, val2, 0)
        self.assertFalse(swapped)

        # Match only in high 32 bits
        val3 = 0x12345678_AAAAAAAAAAAAAAAA
        val4 = 0x12345678_BBBBBBBBBBBBBBBB
        old2, mem2, swapped2 = oracle_condxchg64(val3, val4, 0)
        self.assertFalse(swapped2)

    def test_lds_dword_indices_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        # Byte addresses 0..3 map to dwords 0 and 1
        for b in range(4):
            idx0, idx1 = lds.get_dword_indices(b)
            self.assertEqual(idx0, 0)
            self.assertEqual(idx1, 1)

        # Byte address 4 maps to dwords 1 and 2
        idx0, idx1 = lds.get_dword_indices(4)
        self.assertEqual(idx0, 1)
        self.assertEqual(idx1, 2)

    def test_lds_offset_addition(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(0, offset=8)
        self.assertEqual(idx0, 2)
        self.assertEqual(idx1, 3)

    def test_lds_out_of_bounds_wrapping(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(65536)
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)

        # Wrapping at highest dword boundary
        idx0_end, idx1_end = lds.get_dword_indices(65532)
        self.assertEqual(idx0_end, 16383)
        self.assertEqual(idx1_end, 0)  # Wraps around to 0

    def test_lds_bank_calculation(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(0)
        bank0, bank1 = lds.get_banks(idx0, idx1)
        self.assertEqual(bank0, 0)
        self.assertEqual(bank1, 1)

    def test_lds_read_write_qword(self) -> None:
        lds = LDSMemory(65536)
        test_val = 0xCAFEBABE_DEADBEEF
        lds.write_qword(0x20, test_val)
        self.assertEqual(lds.read_qword(0x20), test_val)
        # Check raw dwords
        self.assertEqual(lds.dwords[8], 0xDEADBEEF)
        self.assertEqual(lds.dwords[9], 0xCAFEBABE)

    def test_lds_atomic_condxchg64_execution(self) -> None:
        lds = LDSMemory(65536)
        initial_val = 0x12345678_9ABCDEF0
        new_val = 0xFEDCBA98_76543210
        lds.write_qword(40, initial_val)

        # Successful CAS
        old, mem, swapped = lds.atomic_condxchg64(40, initial_val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, initial_val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(40), new_val)

        # Failed CAS
        old2, mem2, swapped2 = lds.atomic_condxchg64(40, initial_val, 0)
        self.assertFalse(swapped2)
        self.assertEqual(old2, new_val)
        self.assertEqual(mem2, new_val)
        self.assertEqual(lds.read_qword(40), new_val)


class DSOpcode(str, enum.Enum):
    """Opcode enumerations for 64-bit LDS conditional exchange."""
    DS_CONDXCHG32_RTN_B64 = "ds_condxchg32_rtn_b64"
    DS_CMPST_RTN_B64 = "ds_cmpst_rtn_b64"

    @property
    def is_return(self) -> bool:
        """True if instruction returns prior LDS value to destination VGPR pair."""
        return True

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
class DSCondXchg32B64Instruction:
    """Symbolic AST node representing an RDNA2 DS 64-bit conditional exchange instruction."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata0: VGPRPair
    vdata1: VGPRPair
    vdst: Optional[VGPRPair] = None
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
    def parse(cls, text: str) -> DSCondXchg32B64Instruction:
        """Parse canonical RDNA2 assembly string into a DSCondXchg32B64Instruction AST."""
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

        if len(reg_tokens) != 4:
            raise ValueError(f"{opcode.value} expects 4 operands (vdst, vaddr, vdata0, vdata1), got {reg_tokens}")

        vdst = parse_vgpr_pair(reg_tokens[0])
        vaddr = parse_vgpr(reg_tokens[1])
        vdata0 = parse_vgpr_pair(reg_tokens[2])
        vdata1 = parse_vgpr_pair(reg_tokens[3])

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
        self.assertTrue(DSOpcode.DS_CONDXCHG32_RTN_B64.is_return)
        self.assertEqual(DSOpcode.DS_CONDXCHG32_RTN_B64.bit_width, 64)
        self.assertTrue(DSOpcode.DS_CMPST_RTN_B64.is_return)
        self.assertEqual(DSOpcode.DS_CMPST_RTN_B64.bit_width, 64)

    def test_vgpr_and_vgpr_pair_validation(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

        pair0 = VGPRPair(0)
        pair254 = VGPRPair(254)
        self.assertEqual(str(pair0), "v[0:1]")
        self.assertEqual(str(pair254), "v[254:255]")
        self.assertEqual(pair0.lo.index, 0)
        self.assertEqual(pair0.hi.index, 1)

        with self.assertRaises(ValueError):
            VGPRPair(-1)
        with self.assertRaises(ValueError):
            VGPRPair(255)

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
            DSCondXchg32B64Instruction(
                opcode=DSOpcode.DS_CONDXCHG32_RTN_B64,
                vaddr=VGPR(0),
                vdata0=VGPRPair(2),
                vdata1=VGPRPair(4),
                vdst=None,
            )

    def test_disasm_and_parse_roundtrip(self) -> None:
        test_cases = [
            DSCondXchg32B64Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_B64,
                vaddr=VGPR(1),
                vdata0=VGPRPair(4),
                vdata1=VGPRPair(6),
                vdst=VGPRPair(2),
            ),
            DSCondXchg32B64Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_B64,
                vaddr=VGPR(1),
                vdata0=VGPRPair(4),
                vdata1=VGPRPair(6),
                vdst=VGPRPair(2),
                offset=OffsetImmediate(16),
            ),
            DSCondXchg32B64Instruction(
                DSOpcode.DS_CMPST_RTN_B64,
                vaddr=VGPR(10),
                vdata0=VGPRPair(12),
                vdata1=VGPRPair(14),
                vdst=VGPRPair(8),
                offset=OffsetImmediate(64),
            ),
        ]
        for inst in test_cases:
            with self.subTest(inst=inst):
                asm = inst.disasm()
                parsed = DSCondXchg32B64Instruction.parse(asm)
                self.assertEqual(inst, parsed)


class ScalarRegisterFile:
    """Architectural register file for a single wavefront lane (256 32-bit VGPRs)."""

    def __init__(self) -> None:
        self.regs: List[int] = [0] * 256

    def get_u32(self, reg: VGPR) -> int:
        """Read 32-bit VGPR value."""
        return self.regs[reg.index] & MASK32

    def set_u32(self, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR value."""
        self.regs[reg.index] = val & MASK32

    def get_u64(self, pair: VGPRPair) -> int:
        """Read 64-bit value from consecutive VGPR pair."""
        lo = self.get_u32(pair.lo)
        hi = self.get_u32(pair.hi)
        return join64(lo, hi)

    def set_u64(self, pair: VGPRPair, val: int) -> None:
        """Write 64-bit value into consecutive VGPR pair."""
        lo, hi = split64(val)
        self.set_u32(pair.lo, lo)
        self.set_u32(pair.hi, hi)


class ScalarThreadState:
    """Execution state of a single scalar thread executing LDS instructions."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.rf: ScalarRegisterFile = ScalarRegisterFile()
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()


class ScalarDSInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 DS 64-bit conditional exchange."""

    def execute(self, state: ScalarThreadState, inst: DSCondXchg32B64Instruction) -> int:
        """Execute a single DS 64-bit conditional exchange instruction."""
        addr = state.rf.get_u32(inst.vaddr)
        cmp_val = state.rf.get_u64(inst.vdata0)
        new_val = state.rf.get_u64(inst.vdata1)

        old_val, new_mem, exchanged = state.lds.atomic_condxchg64(
            addr, cmp_val, new_val, offset=inst.offset.value
        )

        assert inst.vdst is not None
        state.rf.set_u64(inst.vdst, old_val)
        return old_val


class TestDeterministicScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution of ds_condxchg32_rtn_b64."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_exchange_success_on_match(self) -> None:
        val_init = 0x11112222_33334444
        val_new = 0x55556666_77778888
        self.lds.write_qword(32, val_init)

        self.state.rf.set_u32(VGPR(1), 32)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(2)), val_init)
        self.assertEqual(self.lds.read_qword(32), val_new)

    def test_scalar_exchange_failure_on_mismatch(self) -> None:
        val_init = 0xAAAAAAAA_BBBBBBBB
        val_cmp = 0xCCCCCCCC_DDDDDDDD
        val_new = 0xEEEEEEEE_FFFFFFFF
        self.lds.write_qword(64, val_init)

        self.state.rf.set_u32(VGPR(1), 64)
        self.state.rf.set_u64(VGPRPair(4), val_cmp)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(8),
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(8)), val_init)
        self.assertEqual(self.lds.read_qword(64), val_init)

    def test_scalar_exchange_with_offset(self) -> None:
        val_init = 0x12345678_9ABCDEF0
        val_new = 0xFEDCBA98_76543210
        self.lds.write_qword(48, val_init)

        self.state.rf.set_u32(VGPR(1), 32)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(10),
            offset=OffsetImmediate(16),  # 32 + 16 = 48
        )
        old = self.interp.execute(self.state, inst)

        self.assertEqual(old, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(10)), val_init)
        self.assertEqual(self.lds.read_qword(48), val_new)

    def test_scalar_sequential_cas_steps(self) -> None:
        val0 = 0x00000000_00000001
        val1 = 0x00000000_00000002
        val2 = 0x00000000_00000003
        self.lds.write_qword(0, val0)
        self.state.rf.set_u32(VGPR(1), 0)

        # Step 1: cas(1 -> 2) succeeds
        self.state.rf.set_u64(VGPRPair(4), val0)
        self.state.rf.set_u64(VGPRPair(6), val1)
        r1 = self.interp.execute(
            self.state,
            DSCondXchg32B64Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_B64,
                vaddr=VGPR(1), vdata0=VGPRPair(4), vdata1=VGPRPair(6), vdst=VGPRPair(2)
            )
        )
        self.assertEqual(r1, val0)
        self.assertEqual(self.lds.read_qword(0), val1)

        # Step 2: cas(1 -> 3) fails because current is 2
        r2 = self.interp.execute(
            self.state,
            DSCondXchg32B64Instruction(
                DSOpcode.DS_CONDXCHG32_RTN_B64,
                vaddr=VGPR(1), vdata0=VGPRPair(4), vdata1=VGPRPair(6), vdst=VGPRPair(2)
            )
        )
        self.assertEqual(r2, val1)
        self.assertEqual(self.lds.read_qword(0), val1)


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

    def get_u32(self, lane: int, reg: VGPR) -> int:
        """Read 32-bit VGPR for a specific lane."""
        return self.vgprs[lane][reg.index] & MASK32

    def set_u32(self, lane: int, reg: VGPR, val: int) -> None:
        """Write 32-bit VGPR for a specific lane."""
        self.vgprs[lane][reg.index] = val & MASK32

    def get_u64(self, lane: int, pair: VGPRPair) -> int:
        """Read 64-bit value from VGPR pair for a specific lane."""
        lo = self.get_u32(lane, pair.lo)
        hi = self.get_u32(lane, pair.hi)
        return join64(lo, hi)

    def set_u64(self, lane: int, pair: VGPRPair, val: int) -> None:
        """Write 64-bit value into VGPR pair for a specific lane."""
        lo, hi = split64(val)
        self.set_u32(lane, pair.lo, lo)
        self.set_u32(lane, pair.hi, hi)


class SIMDDSInterpreter:
    """Multi-lane SIMD instruction interpreter modeling RDNA2 lockstep execution and LDS 64-bit atomics."""

    def execute(self, state: SIMDWaveState, inst: DSCondXchg32B64Instruction) -> Dict[int, int]:
        """Execute DS 64-bit conditional exchange across all active lanes of the wave."""
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

        returns: Dict[int, int] = {}

        for lane, addr, cmp_val, new_val in latched_inputs:
            old_val, new_mem, exchanged = state.lds.atomic_condxchg64(
                addr, cmp_val, new_val, offset=inst.offset.value
            )
            assert inst.vdst is not None
            state.set_u64(lane, inst.vdst, old_val)
            returns[lane] = old_val

        return returns


class TestSIMDMultiLaneDivergenceAndExecMasking(unittest.TestCase):
    """Unit tests for SIMD lockstep execution, 32/64-bit EXEC masking, and address convergence."""

    def test_wave32_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Seed each 8-byte element with (i << 32 | i)
        for i in range(32):
            lds.write_qword(i * 8, (i << 32) | i)

        # Lane i targets element i, expects (i << 32 | i), exchanges to ((i + 100) << 32 | (i + 100))
        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            exp_val = (lane << 32) | lane
            new_val = ((lane + 100) << 32) | (lane + 100)
            wave.set_u64(lane, VGPRPair(4), exp_val)
            wave.set_u64(lane, VGPRPair(6), new_val)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        returns = interp.execute(wave, inst)

        for i in range(32):
            exp_old = (i << 32) | i
            exp_new = ((i + 100) << 32) | (i + 100)
            self.assertEqual(returns[i], exp_old)
            self.assertEqual(wave.get_u64(i, VGPRPair(2)), exp_old)
            self.assertEqual(lds.read_qword(i * 8), exp_new)

    def test_wave64_full_mask_divergent_addresses(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(64):
            lds.write_qword(i * 8, 0x11111111_22222222)

        for lane in range(64):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), 0x11111111_22222222)
            wave.set_u64(lane, VGPRPair(6), 0x33333333_44444444)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        returns = interp.execute(wave, inst)

        for i in range(64):
            self.assertEqual(returns[i], 0x11111111_22222222)
            self.assertEqual(lds.read_qword(i * 8), 0x33333333_44444444)

    def test_wave32_partial_exec_masking(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        interp = SIMDDSInterpreter()

        for i in range(32):
            lds.write_qword(i * 8, 0xAAAAAAAA_BBBBBBBB)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), 0xAAAAAAAA_BBBBBBBB)
            wave.set_u64(lane, VGPRPair(6), 0xCCCCCCCC_DDDDDDDD)
            wave.set_u64(lane, VGPRPair(2), 0xCAFEBABE_CAFEBABE)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(lds.read_qword(lane * 8), 0xCCCCCCCC_DDDDDDDD)
                self.assertEqual(wave.get_u64(lane, VGPRPair(2)), 0xAAAAAAAA_BBBBBBBB)
                self.assertIn(lane, returns)
            else:
                self.assertEqual(lds.read_qword(lane * 8), 0xAAAAAAAA_BBBBBBBB)
                self.assertEqual(wave.get_u64(lane, VGPRPair(2)), 0xCAFEBABE_CAFEBABE)
                self.assertNotIn(lane, returns)

    def test_zero_exec_mask_no_mutation(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        interp = SIMDDSInterpreter()

        init_val = 0x12345678_9ABCDEF0
        lds.write_qword(0, init_val)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), 0)
            wave.set_u64(lane, VGPRPair(4), init_val)
            wave.set_u64(lane, VGPRPair(6), 0)
            wave.set_u64(lane, VGPRPair(2), 0xDEADBEEF_DEADBEEF)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        returns = interp.execute(wave, inst)

        self.assertEqual(len(returns), 0)
        self.assertEqual(lds.read_qword(0), init_val)
        for lane in range(32):
            self.assertEqual(wave.get_u64(lane, VGPRPair(2)), 0xDEADBEEF_DEADBEEF)

    def test_convergent_collision_cas_daisy_chain_64bit(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Target address 0x20
        target_addr = 0x20
        lds.write_qword(target_addr, 0)

        # Lane i expects i and exchanges to i + 1
        for lane in range(32):
            wave.set_u32(lane, VGPR(1), target_addr)
            wave.set_u64(lane, VGPRPair(4), lane)
            wave.set_u64(lane, VGPRPair(6), lane + 1)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        returns = interp.execute(wave, inst)

        # Final memory value is 32
        self.assertEqual(lds.read_qword(target_addr), 32)
        for lane in range(32):
            self.assertEqual(returns[lane], lane)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(65536)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_dst_aliases_vaddr(self) -> None:
        val_init = 0x11112222_33334444
        val_new = 0x55556666_77778888
        self.lds.write_qword(32, val_init)

        # v2 holds address 32. vdst is v[2:3].
        self.state.rf.set_u32(VGPR(2), 32)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(2),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(2)), val_init)
        self.assertEqual(self.lds.read_qword(32), val_new)

    def test_scalar_dst_aliases_data0(self) -> None:
        val_init = 0xAAAA0000_BBBB0000
        val_new = 0xCCCC0000_DDDD0000
        self.lds.write_qword(40, val_init)

        self.state.rf.set_u32(VGPR(1), 40)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        # vdst aliases vdata0 (both v[4:5])
        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(4),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(4)), val_init)
        self.assertEqual(self.lds.read_qword(40), val_new)

    def test_scalar_dst_aliases_data1(self) -> None:
        val_init = 0x12340000_56780000
        val_new = 0x9ABC0000_DEF00000
        self.lds.write_qword(48, val_init)

        self.state.rf.set_u32(VGPR(1), 48)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)

        # vdst aliases vdata1 (both v[6:7])
        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(6),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(6)), val_init)
        self.assertEqual(self.lds.read_qword(48), val_new)

    def test_scalar_data0_aliases_data1(self) -> None:
        val_init = 0xFEEDFACE_CAFEBABE
        self.lds.write_qword(56, val_init)

        self.state.rf.set_u32(VGPR(1), 56)
        self.state.rf.set_u64(VGPRPair(4), val_init)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(4),
            vdst=VGPRPair(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(2)), val_init)
        self.assertEqual(self.lds.read_qword(56), val_init)

    def test_operand_preservation_non_aliased(self) -> None:
        val_init = 0x11111111_11111111
        val_new = 0x22222222_22222222
        self.lds.write_qword(64, val_init)

        self.state.rf.set_u32(VGPR(1), 64)
        self.state.rf.set_u64(VGPRPair(4), val_init)
        self.state.rf.set_u64(VGPRPair(6), val_new)
        self.state.rf.set_u32(VGPR(100), 0xCAFEBABE)
        self.state.rf.set_u32(VGPR(255), 0x12345678)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(8),
        )
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.rf.get_u32(VGPR(1)), 64)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(4)), val_init)
        self.assertEqual(self.state.rf.get_u64(VGPRPair(6)), val_new)
        self.assertEqual(self.state.rf.get_u32(VGPR(100)), 0xCAFEBABE)
        self.assertEqual(self.state.rf.get_u32(VGPR(255)), 0x12345678)

    def test_simd_aliasing_under_divergence(self) -> None:
        lds = LDSMemory(65536)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xAAAAAAAA, lds=lds)
        interp = SIMDDSInterpreter()

        val_init = 0x55555555_55555555
        val_new = 0x77777777_77777777
        for i in range(32):
            lds.write_qword(i * 8, val_init)

        for lane in range(32):
            wave.set_u32(lane, VGPR(1), lane * 8)
            wave.set_u64(lane, VGPRPair(4), val_init)
            wave.set_u64(lane, VGPRPair(6), val_new)

        inst = DSCondXchg32B64Instruction(
            DSOpcode.DS_CONDXCHG32_RTN_B64,
            vaddr=VGPR(1),
            vdata0=VGPRPair(4),
            vdata1=VGPRPair(6),
            vdst=VGPRPair(4),
        )
        returns = interp.execute(wave, inst)

        for lane in range(32):
            if lane % 2 != 0:
                self.assertEqual(wave.get_u64(lane, VGPRPair(4)), val_init)
                self.assertEqual(lds.read_qword(lane * 8), val_new)
                self.assertEqual(returns[lane], val_init)
            else:
                self.assertEqual(wave.get_u64(lane, VGPRPair(4)), val_init)
                self.assertEqual(lds.read_qword(lane * 8), val_init)
                self.assertNotIn(lane, returns)


if __name__ == "__main__":
    unittest.main()
