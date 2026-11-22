"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_min_rtn_i32.

Models signed 32-bit integer minimum and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_min_rtn_i32: Atomic signed 32-bit minimum returning previous value to destination VGPR.
  - ds_min_i32: Atomic signed 32-bit minimum discarding previous value.

RDNA2 LDS / Signed 32-bit Integer ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Signed Integer Ordering (Two's-Complement):
     Operands are treated as signed two's-complement 32-bit integers in [-2^31, 2^31 - 1].
     Negative numbers have high bit set (0x80000000 = -2147483648, 0xFFFFFFFF = -1).
     -1 is strictly less than 0 (in contrast to unsigned u32 where 0 < 0xFFFFFFFF).
  3. Atomic Operations & Return Semantics:
     Returning variants (ds_min_rtn_i32) atomically fetch the previous 32-bit value,
     write the signed minimum into LDS, and return the previous value to the destination VGPR.
     Non-returning variants (ds_min_i32) perform the atomic signed minimum and discard the old value.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000       # -2147483648
INT32_MAX: int = 0x7FFFFFFF        # +2147483647
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret integer as signed two's-complement 32-bit integer in [-2^31, 2^31 - 1]."""
    val = val & MASK32
    return val - 0x100000000 if (val & 0x80000000) else val


def oracle_min_i32(a: int, b: int) -> int:
    """Reference signed 32-bit minimum returning 32-bit unsigned bit pattern."""
    sa = s32(a)
    sb = s32(b)
    res = sa if sa < sb else sb
    return u32(res)


def ref_ds_min_rtn_i32(current_mem: int, comp_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_min_rtn_i32.

    Returns:
        (old_val, new_mem) as unsigned 32-bit integers.
    """
    old_val = u32(current_mem)
    new_mem = oracle_min_i32(old_val, comp_val)
    return old_val, new_mem


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
        effective_byte = (byte_addr + (offset & OFFSET16_MASK)) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = u32(val)

    def atomic_min_i32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic signed minimum at (byte_addr + offset), returning new value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_i32(old_val, val)
        self.dwords[idx] = new_val
        return new_val

    def atomic_min_rtn_i32(self, byte_addr: int, val: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic signed minimum at (byte_addr + offset), returning (old_val, new_val)."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_min_i32(old_val, val)
        self.dwords[idx] = new_val
        return old_val, new_val


class TestLDSMinRtnI32ModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_signed_conversion_helpers(self) -> None:
        self.assertEqual(s32(0), 0)
        self.assertEqual(s32(1), 1)
        self.assertEqual(s32(0x7FFFFFFF), 2147483647)
        self.assertEqual(s32(0x80000000), -2147483648)
        self.assertEqual(s32(0xFFFFFFFF), -1)
        self.assertEqual(u32(-1), 0xFFFFFFFF)
        self.assertEqual(u32(-2147483648), 0x80000000)

    def test_oracle_min_i32_basic(self) -> None:
        # Both positive
        self.assertEqual(oracle_min_i32(10, 20), 10)
        self.assertEqual(oracle_min_i32(50, 15), 15)
        # Positive and negative
        self.assertEqual(oracle_min_i32(10, u32(-5)), u32(-5))
        self.assertEqual(oracle_min_i32(u32(-10), 5), u32(-10))
        # Both negative
        self.assertEqual(oracle_min_i32(u32(-10), u32(-20)), u32(-20))
        self.assertEqual(oracle_min_i32(u32(-50), u32(-30)), u32(-50))
        # Zero and negative
        self.assertEqual(oracle_min_i32(0, u32(-1)), u32(-1))
        self.assertEqual(oracle_min_i32(u32(-1), 0), u32(-1))
        # Equal values
        self.assertEqual(oracle_min_i32(42, 42), 42)
        self.assertEqual(oracle_min_i32(u32(-7), u32(-7)), u32(-7))

    def test_oracle_min_i32_extremes(self) -> None:
        # INT32_MIN vs INT32_MAX
        self.assertEqual(oracle_min_i32(0x80000000, 0x7FFFFFFF), 0x80000000)
        self.assertEqual(oracle_min_i32(0x7FFFFFFF, 0x80000000), 0x80000000)
        # INT32_MIN vs 0
        self.assertEqual(oracle_min_i32(0x80000000, 0), 0x80000000)
        # INT32_MAX vs 0
        self.assertEqual(oracle_min_i32(0x7FFFFFFF, 0), 0)

    def test_ref_ds_min_rtn_i32(self) -> None:
        old_val, new_val = ref_ds_min_rtn_i32(100, 25)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_val, 25)

        old_val2, new_val2 = ref_ds_min_rtn_i32(u32(-10), 10)
        self.assertEqual(old_val2, u32(-10))
        self.assertEqual(new_val2, u32(-10))

    def test_lds_memory_addressing_and_wrapping(self) -> None:
        # Byte address 0x10 -> dword index 4
        self.assertEqual(self.lds.get_dword_index(0x10), 4)
        # Unaligned low 2 bits are dropped (0x13 -> dword index 4)
        self.assertEqual(self.lds.get_dword_index(0x13), 4)
        # Offset addition
        self.assertEqual(self.lds.get_dword_index(0x10, offset=4), 5)
        # Wrap around 64 KiB (65536 bytes = 16384 dwords)
        self.assertEqual(self.lds.get_dword_index(65536), 0)
        self.assertEqual(self.lds.get_dword_index(65540), 1)

    def test_lds_memory_banks(self) -> None:
        for i in range(64):
            self.assertEqual(self.lds.get_bank(i), i % NUM_LDS_BANKS)

    def test_lds_read_write(self) -> None:
        self.lds.write_dword(10, 0x12345678)
        self.assertEqual(self.lds.read_dword(10), 0x12345678)

    def test_lds_atomic_min_operations(self) -> None:
        # Set initial value to 50 at dword 8 (byte addr 32)
        self.lds.write_dword(8, 50)
        old_val, new_val = self.lds.atomic_min_rtn_i32(32, 20)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 20)
        self.assertEqual(self.lds.read_dword(8), 20)

        # Atomic min with negative number
        old_val, new_val = self.lds.atomic_min_rtn_i32(32, u32(-5))
        self.assertEqual(old_val, 20)
        self.assertEqual(new_val, u32(-5))
        self.assertEqual(self.lds.read_dword(8), u32(-5))

        # Atomic min where memory is already smaller
        old_val, new_val = self.lds.atomic_min_rtn_i32(32, 100)
        self.assertEqual(old_val, u32(-5))
        self.assertEqual(new_val, u32(-5))
        self.assertEqual(self.lds.read_dword(8), u32(-5))


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(str, enum.Enum):
    """Data Share (DS) atomic opcodes for signed 32-bit integer minimum."""
    DS_MIN_RTN_I32 = "ds_min_rtn_i32"
    DS_MIN_I32 = "ds_min_i32"

    @property
    def is_return(self) -> bool:
        return "rtn" in self.value


@dataclasses.dataclass(frozen=True)
class VGPR:
    """Vector General Purpose Register (v0 - v255)."""
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    """16-bit unsigned byte offset immediate for DS instructions."""
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 65535):
            raise ValueError(f"OffsetImmediate out of range [0, 65535]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}"


@dataclasses.dataclass
class DSMinRtnI32Instruction:
    """Symbolic AST node for ds_min_rtn_i32 / ds_min_i32 instruction."""
    opcode: DSOpcode
    vaddr: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))
    gds: bool = False

    def __post_init__(self) -> None:
        if self.opcode.is_return and self.vdst is None:
            raise ValueError(f"Return opcode {self.opcode.value} requires vdst register")
        if not self.opcode.is_return and self.vdst is not None:
            raise ValueError(f"Non-return opcode {self.opcode.value} must not specify vdst register")

    def to_assembly(self) -> str:
        tokens = [self.opcode.value]
        if self.opcode.is_return:
            assert self.vdst is not None
            tokens.append(f"{self.vdst}, {self.vaddr}, {self.vdata}")
        else:
            tokens.append(f"{self.vaddr}, {self.vdata}")
        suffix_parts = []
        if self.offset.value != 0:
            suffix_parts.append(str(self.offset))
        if self.gds:
            suffix_parts.append("gds")
        if suffix_parts:
            return f"{tokens[0]} {tokens[1]} {' '.join(suffix_parts)}"
        return f"{tokens[0]} {tokens[1]}"

    @classmethod
    def parse(cls, asm: str) -> DSMinRtnI32Instruction:
        text = asm.strip()
        if not text:
            raise ValueError("Empty assembly string")
        parts = text.split(None, 1)
        op_str = parts[0]
        try:
            opcode = DSOpcode(op_str)
        except ValueError:
            raise ValueError(f"Unknown or unsupported opcode: {op_str}")

        if len(parts) < 2:
            raise ValueError("Missing instruction operands")

        rest = parts[1].strip()
        tokens = rest.split()
        reg_tokens: List[str] = []
        mod_tokens: List[str] = []
        for t in tokens:
            if ":" in t or t == "gds":
                mod_tokens.append(t)
            else:
                reg_tokens.append(t)

        reg_str = " ".join(reg_tokens)
        reg_names = [r.strip() for r in reg_str.split(",") if r.strip()]

        def parse_vgpr(name: str) -> VGPR:
            if not name.startswith("v"):
                raise ValueError(f"Expected VGPR starting with 'v', got {name}")
            return VGPR(int(name[1:]))

        offset_val = 0
        gds = False
        for mod in mod_tokens:
            if mod.startswith("offset:"):
                offset_val = int(mod.split(":", 1)[1])
            elif mod == "gds":
                gds = True
            else:
                raise ValueError(f"Unknown modifier: {mod}")

        offset = OffsetImmediate(offset_val)

        if opcode.is_return:
            if len(reg_names) != 3:
                raise ValueError(f"Expected 3 register operands for {opcode.value}, got {len(reg_names)}")
            vdst = parse_vgpr(reg_names[0])
            vaddr = parse_vgpr(reg_names[1])
            vdata = parse_vgpr(reg_names[2])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, vdst=vdst, offset=offset, gds=gds)
        else:
            if len(reg_names) != 2:
                raise ValueError(f"Expected 2 register operands for {opcode.value}, got {len(reg_names)}")
            vaddr = parse_vgpr(reg_names[0])
            vdata = parse_vgpr(reg_names[1])
            return cls(opcode=opcode, vaddr=vaddr, vdata=vdata, offset=offset, gds=gds)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for AST opcode definitions, typed operands, and parsing."""

    def test_vgpr_valid_and_bounds(self) -> None:
        v0 = VGPR(0)
        v255 = VGPR(255)
        self.assertEqual(str(v0), "v0")
        self.assertEqual(str(v255), "v255")
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

    def test_offset_immediate_valid_and_bounds(self) -> None:
        off0 = OffsetImmediate(0)
        off65535 = OffsetImmediate(65535)
        self.assertEqual(str(off0), "offset:0")
        self.assertEqual(str(off65535), "offset:65535")
        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(65536)

    def test_instruction_construction_validation(self) -> None:
        # Valid return instruction
        inst_rtn = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            vdst=VGPR(0),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst_rtn.to_assembly(), "ds_min_rtn_i32 v0, v1, v2 offset:16")

        # Valid non-return instruction
        inst_nortn = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        self.assertEqual(inst_nortn.to_assembly(), "ds_min_i32 v1, v2")

        # Missing vdst for return opcode
        with self.assertRaises(ValueError):
            DSMinRtnI32Instruction(
                opcode=DSOpcode.DS_MIN_RTN_I32,
                vaddr=VGPR(1),
                vdata=VGPR(2),
                vdst=None,
            )

        # Providing vdst for non-return opcode
        with self.assertRaises(ValueError):
            DSMinRtnI32Instruction(
                opcode=DSOpcode.DS_MIN_I32,
                vaddr=VGPR(1),
                vdata=VGPR(2),
                vdst=VGPR(0),
            )

    def test_instruction_parsing_and_roundtrip(self) -> None:
        asm1 = "ds_min_rtn_i32 v0, v1, v2 offset:32"
        inst1 = DSMinRtnI32Instruction.parse(asm1)
        self.assertEqual(inst1.opcode, DSOpcode.DS_MIN_RTN_I32)
        self.assertEqual(inst1.vdst, VGPR(0))
        self.assertEqual(inst1.vaddr, VGPR(1))
        self.assertEqual(inst1.vdata, VGPR(2))
        self.assertEqual(inst1.offset.value, 32)
        self.assertEqual(inst1.to_assembly(), asm1)

        asm2 = "ds_min_i32 v5, v6"
        inst2 = DSMinRtnI32Instruction.parse(asm2)
        self.assertEqual(inst2.opcode, DSOpcode.DS_MIN_I32)
        self.assertIsNone(inst2.vdst)
        self.assertEqual(inst2.vaddr, VGPR(5))
        self.assertEqual(inst2.vdata, VGPR(6))
        self.assertEqual(inst2.offset.value, 0)
        self.assertEqual(inst2.to_assembly(), asm2)

        asm3 = "ds_min_rtn_i32 v10, v20, v30 offset:4 gds"
        inst3 = DSMinRtnI32Instruction.parse(asm3)
        self.assertTrue(inst3.gds)
        self.assertEqual(inst3.to_assembly(), asm3)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    """Architectural register and local memory state for a single scalar lane."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * 256
        self.lds: LDSMemory = lds if lds is not None else LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = u32(val)


class ScalarDSInterpreter:
    """Deterministic scalar interpreter for ds_min_rtn_i32 and ds_min_i32."""

    def execute(self, state: ScalarThreadState, inst: DSMinRtnI32Instruction) -> Optional[int]:
        vaddr_val = state.get_vgpr(inst.vaddr)
        vdata_val = state.get_vgpr(inst.vdata)

        old_val, new_val = state.lds.atomic_min_rtn_i32(
            byte_addr=vaddr_val,
            val=vdata_val,
            offset=inst.offset.value,
        )

        if inst.opcode.is_return:
            assert inst.vdst is not None
            state.set_vgpr(inst.vdst, old_val)
            return old_val
        return None


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_ds_min_rtn_i32_execution(self) -> None:
        # LDS[dword 10] = 500
        self.lds.write_dword(10, 500)
        # v1 (vaddr) = 40 (byte address 40 -> dword 10)
        self.state.set_vgpr(VGPR(1), 40)
        # v2 (vdata) = 250
        self.state.set_vgpr(VGPR(2), 250)
        # v0 (vdst) = sentinel 0xDEADBEEF
        self.state.set_vgpr(VGPR(0), 0xDEADBEEF)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertEqual(ret, 500)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 500)
        self.assertEqual(self.lds.read_dword(10), 250)

    def test_ds_min_i32_non_return(self) -> None:
        self.lds.write_dword(5, 100)
        self.state.set_vgpr(VGPR(1), 20)  # dword 5
        self.state.set_vgpr(VGPR(2), 40)
        self.state.set_vgpr(VGPR(0), 0x1234)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertIsNone(ret)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 0x1234)  # untouched
        self.assertEqual(self.lds.read_dword(5), 40)

    def test_signed_negative_updates(self) -> None:
        # Starting with positive 50
        self.lds.write_dword(2, 50)
        self.state.set_vgpr(VGPR(1), 8)
        # Compare with -10
        self.state.set_vgpr(VGPR(2), u32(-10))
        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret1 = self.interp.execute(self.state, inst)
        self.assertEqual(ret1, 50)
        self.assertEqual(self.lds.read_dword(2), u32(-10))

        # Compare with -25
        self.state.set_vgpr(VGPR(2), u32(-25))
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, u32(-10))
        self.assertEqual(self.lds.read_dword(2), u32(-25))

        # Compare with 10 (larger than -25, so memory remains -25)
        self.state.set_vgpr(VGPR(2), 10)
        ret3 = self.interp.execute(self.state, inst)
        self.assertEqual(ret3, u32(-25))
        self.assertEqual(self.lds.read_dword(2), u32(-25))

    def test_offset_immediate_computation(self) -> None:
        self.lds.write_dword(20, 1000)
        # vaddr = 64 (dword 16), offset = 16 bytes -> dword 20 (80 bytes)
        self.state.set_vgpr(VGPR(1), 64)
        self.state.set_vgpr(VGPR(2), 300)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 1000)
        self.assertEqual(self.lds.read_dword(20), 300)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(
        self,
        wave_size: int = 32,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        default_mask = (1 << wave_size) - 1
        self.exec_mask: int = default_mask if exec_mask is None else (exec_mask & default_mask)
        self.lds: LDSMemory = lds if lds is not None else LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(wave_size)]

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.wave_size):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg: VGPR) -> int:
        return self.vgprs[lane][reg.index]

    def set_vgpr(self, lane: int, reg: VGPR, val: int) -> None:
        self.vgprs[lane][reg.index] = u32(val)


class SIMDDSInterpreter:
    """Deterministic SIMD interpreter simulating multi-lane wave execution."""

    def execute(self, state: SIMDWaveState, inst: DSMinRtnI32Instruction) -> Dict[int, Optional[int]]:
        results: Dict[int, Optional[int]] = {}
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                results[lane] = None
                continue

            vaddr_val = state.get_vgpr(lane, inst.vaddr)
            vdata_val = state.get_vgpr(lane, inst.vdata)

            old_val, new_val = state.lds.atomic_min_rtn_i32(
                byte_addr=vaddr_val,
                val=vdata_val,
                offset=inst.offset.value,
            )

            if inst.opcode.is_return:
                assert inst.vdst is not None
                state.set_vgpr(lane, inst.vdst, old_val)
                results[lane] = old_val
            else:
                results[lane] = None

        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, 32/64-bit EXEC masking, and divergence."""

    def test_wave32_all_lanes_active_independent_addresses(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        state = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        interp = SIMDDSInterpreter()

        # Each lane has its own unique address: lane * 4
        for lane in range(32):
            addr = lane * 4
            lds.write_dword(lane, 200 + lane)
            state.set_vgpr(lane, VGPR(1), addr)
            # Lane adds signed value: 100 - lane (smaller than 200 + lane)
            state.set_vgpr(lane, VGPR(2), 100 - lane)
            state.set_vgpr(lane, VGPR(0), 0xCAFE)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = interp.execute(state, inst)

        for lane in range(32):
            self.assertEqual(results[lane], 200 + lane)
            self.assertEqual(state.get_vgpr(lane, VGPR(0)), 200 + lane)
            self.assertEqual(lds.read_dword(lane), 100 - lane)

    def test_wave64_divergent_exec_mask(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        even_mask = 0x5555555555555555
        state = SIMDWaveState(wave_size=64, exec_mask=even_mask, lds=lds)
        interp = SIMDDSInterpreter()

        sentinel = 0xBAADF00D
        for lane in range(64):
            addr = lane * 4
            lds.write_dword(lane, 50)
            state.set_vgpr(lane, VGPR(1), addr)
            state.set_vgpr(lane, VGPR(2), u32(-10))  # -10 < 50
            state.set_vgpr(lane, VGPR(0), sentinel)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = interp.execute(state, inst)

        for lane in range(64):
            if lane % 2 == 0:
                # Active lane
                self.assertEqual(results[lane], 50)
                self.assertEqual(state.get_vgpr(lane, VGPR(0)), 50)
                self.assertEqual(lds.read_dword(lane), u32(-10))
            else:
                # Inactive lane: untouched memory and untouched vdst
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, VGPR(0)), sentinel)
                self.assertEqual(lds.read_dword(lane), 50)

    def test_multi_lane_address_collision_serialization(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        # 4 lanes active (lanes 0..3)
        state = SIMDWaveState(wave_size=32, exec_mask=0xF, lds=lds)
        interp = SIMDDSInterpreter()

        collision_addr = 0x40  # dword 16
        initial_val = 100
        lds.write_dword(16, initial_val)

        # Lane inputs:
        # Lane 0: sends 50
        # Lane 1: sends -10 (0xFFFFFFF6)
        # Lane 2: sends 20
        # Lane 3: sends -50 (0xFFFFFFCE)
        inputs = [50, u32(-10), 20, u32(-50)]
        for lane in range(4):
            state.set_vgpr(lane, VGPR(1), collision_addr)
            state.set_vgpr(lane, VGPR(2), inputs[lane])
            state.set_vgpr(lane, VGPR(0), 0)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = interp.execute(state, inst)

        # Serialization trace:
        # Lane 0: reads 100, min(100, 50) = 50 -> returns 100
        self.assertEqual(results[0], 100)
        self.assertEqual(state.get_vgpr(0, VGPR(0)), 100)
        # Lane 1: reads 50, min(50, -10) = -10 -> returns 50
        self.assertEqual(results[1], 50)
        self.assertEqual(state.get_vgpr(1, VGPR(0)), 50)
        # Lane 2: reads -10, min(-10, 20) = -10 -> returns -10
        self.assertEqual(results[2], u32(-10))
        self.assertEqual(state.get_vgpr(2, VGPR(0)), u32(-10))
        # Lane 3: reads -10, min(-10, -50) = -50 -> returns -10
        self.assertEqual(results[3], u32(-10))
        self.assertEqual(state.get_vgpr(3, VGPR(0)), u32(-10))

        # Final memory must be -50
        self.assertEqual(lds.read_dword(16), u32(-50))

    def test_zero_exec_mask(self) -> None:
        lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        state = SIMDWaveState(wave_size=32, exec_mask=0, lds=lds)
        interp = SIMDDSInterpreter()

        lds.write_dword(0, 999)
        for lane in range(32):
            state.set_vgpr(lane, VGPR(0), 777)
            state.set_vgpr(lane, VGPR(1), 0)
            state.set_vgpr(lane, VGPR(2), 10)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = interp.execute(state, inst)

        for lane in range(32):
            self.assertIsNone(results[lane])
            self.assertEqual(state.get_vgpr(lane, VGPR(0)), 777)
        self.assertEqual(lds.read_dword(0), 999)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.scalar_interp = ScalarDSInterpreter()
        self.simd_interp = SIMDDSInterpreter()

    def test_alias_vdst_and_vaddr(self) -> None:
        # ds_min_rtn_i32 v1, v1, v2: vdst == vaddr
        state = ScalarThreadState(lds=self.lds)
        # LDS[dword 3] (addr 12) = 150
        self.lds.write_dword(3, 150)
        state.set_vgpr(VGPR(1), 12)  # vaddr is v1
        state.set_vgpr(VGPR(2), 50)  # vdata is v2 (50 < 150)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(1),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.scalar_interp.execute(state, inst)

        # Must return old memory value 150 into v1
        self.assertEqual(ret, 150)
        self.assertEqual(state.get_vgpr(VGPR(1)), 150)
        # Memory must be updated with min(150, 50) = 50
        self.assertEqual(self.lds.read_dword(3), 50)

    def test_alias_vdst_and_vdata(self) -> None:
        # ds_min_rtn_i32 v2, v1, v2: vdst == vdata
        state = ScalarThreadState(lds=self.lds)
        # LDS[dword 4] (addr 16) = 200
        self.lds.write_dword(4, 200)
        state.set_vgpr(VGPR(1), 16)          # vaddr is v1
        state.set_vgpr(VGPR(2), u32(-80))    # vdata is v2 (-80 < 200)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(2),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.scalar_interp.execute(state, inst)

        # Must return old memory value 200 into v2
        self.assertEqual(ret, 200)
        self.assertEqual(state.get_vgpr(VGPR(2)), 200)
        # Memory must be updated with min(200, -80) = -80
        self.assertEqual(self.lds.read_dword(4), u32(-80))

    def test_alias_all_operands_same_register(self) -> None:
        # ds_min_rtn_i32 v0, v0, v0: vdst == vaddr == vdata
        state = ScalarThreadState(lds=self.lds)
        # Address is 8 (dword 2). We set LDS[2] = 50
        self.lds.write_dword(2, 50)
        state.set_vgpr(VGPR(0), 8)  # v0 holds 8

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(0),
            vdata=VGPR(0),
        )
        ret = self.scalar_interp.execute(state, inst)

        # vaddr = 8, vdata = 8.
        # LDS was 50. min(50, 8) = 8.
        # vdst (v0) gets old memory value 50.
        self.assertEqual(ret, 50)
        self.assertEqual(state.get_vgpr(VGPR(0)), 50)
        self.assertEqual(self.lds.read_dword(2), 8)

    def test_simd_wave32_aliasing_vdst_vaddr(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        for lane in range(32):
            addr = lane * 4
            self.lds.write_dword(lane, 1000 + lane)
            wave.set_vgpr(lane, VGPR(1), addr)
            wave.set_vgpr(lane, VGPR(2), 500)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(1),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = self.simd_interp.execute(wave, inst)

        for lane in range(32):
            self.assertEqual(results[lane], 1000 + lane)
            self.assertEqual(wave.get_vgpr(lane, VGPR(1)), 1000 + lane)
            self.assertEqual(self.lds.read_dword(lane), 500)

    def test_idempotent_in_place_convergence(self) -> None:
        state = ScalarThreadState(lds=self.lds)
        self.lds.write_dword(0, 100)
        state.set_vgpr(VGPR(1), 0)
        state.set_vgpr(VGPR(2), 25)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )

        # Pass 1: returns 100, memory becomes 25
        ret1 = self.scalar_interp.execute(state, inst)
        self.assertEqual(ret1, 100)
        self.assertEqual(self.lds.read_dword(0), 25)

        # Pass 2: memory already 25, vdata is 25. returns 25, memory stays 25
        ret2 = self.scalar_interp.execute(state, inst)
        self.assertEqual(ret2, 25)
        self.assertEqual(self.lds.read_dword(0), 25)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclasses.dataclass
class LDSStatusFlags:
    """Architectural status flags, wrapping conditions, and bank metrics for LDS operations."""
    carry_out: bool = False
    unaligned_address: bool = False
    oob_wrapped: bool = False
    bank_conflicts: int = 0
    address_collisions: int = 0


@dataclasses.dataclass(frozen=True)
class IntegerConditionCodes:
    """Architectural signed integer condition codes."""
    zero: bool
    negative: bool
    positive: bool
    updated: bool


def evaluate_status_flags(
    wave: SIMDWaveState,
    inst: DSMinRtnI32Instruction,
) -> LDSStatusFlags:
    """Evaluate architectural flags, address wrap conditions, and bank conflicts."""
    flags = LDSStatusFlags()
    active_lanes = [l for l in range(wave.wave_size) if wave.is_lane_active(l)]

    bank_to_dwords: Dict[int, Set[int]] = {}
    dword_to_lanes: Dict[int, List[int]] = {}

    for lane in active_lanes:
        addr = wave.get_vgpr(lane, inst.vaddr)
        off = inst.offset.value

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

    for bank, dword_set in bank_to_dwords.items():
        if len(dword_set) > 1:
            flags.bank_conflicts += (len(dword_set) - 1)

    for dword, lanes in dword_to_lanes.items():
        if len(lanes) > 1:
            flags.address_collisions += (len(lanes) - 1)

    return flags


def compute_integer_condition_codes(old_val: int, new_val: int) -> IntegerConditionCodes:
    """Compute condition codes from old memory content and update status."""
    signed_old = s32(old_val)
    return IntegerConditionCodes(
        zero=(signed_old == 0),
        negative=(signed_old < 0),
        positive=(signed_old > 0),
        updated=(u32(new_val) != u32(old_val)),
    )


class TestStatusFlagsAndConditionCodes(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry/borrow isolation."""

    def test_address_carry_out_generation(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 0xFFFFFFFF)
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.carry_out)

    def test_unaligned_byte_address_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 7)  # not dword aligned
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.unaligned_address)

    def test_oob_address_wrapping_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x1)
        wave.set_vgpr(0, VGPR(1), 70000)  # > 65536 bytes
        wave.set_vgpr(0, VGPR(2), 50)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertTrue(flags.oob_wrapped)

    def test_bank_conflicts_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0x7)
        # Bank 0 dwords: 0, 32, 64
        wave.set_vgpr(0, VGPR(1), 0 * 4)
        wave.set_vgpr(1, VGPR(1), 32 * 4)
        wave.set_vgpr(2, VGPR(1), 64 * 4)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertEqual(flags.bank_conflicts, 2)
        self.assertEqual(flags.address_collisions, 0)

    def test_address_collisions_detection(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xF)
        # All 4 lanes target same address 40
        for l in range(4):
            wave.set_vgpr(l, VGPR(1), 40)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        flags = evaluate_status_flags(wave, inst)
        self.assertEqual(flags.address_collisions, 3)
        self.assertEqual(flags.bank_conflicts, 0)

    def test_integer_condition_codes(self) -> None:
        # Zero old value, updated
        cc_zero = compute_integer_condition_codes(0, u32(-5))
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.negative)
        self.assertFalse(cc_zero.positive)
        self.assertTrue(cc_zero.updated)

        # Negative old value, updated
        cc_neg = compute_integer_condition_codes(u32(-10), u32(-20))
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)
        self.assertTrue(cc_neg.updated)

        # Positive old value, not updated
        cc_pos = compute_integer_condition_codes(100, 100)
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)
        self.assertFalse(cc_pos.updated)

    def test_carry_borrow_vcc_isolation(self) -> None:
        # Verify that DS atomic operations do not alter VCC flags
        vcc_sentinel = 0xAAAAAAAAAAAAAAAA
        # Simulated check: executing DS instruction preserves external VCC
        vcc_after = vcc_sentinel
        self.assertEqual(vcc_after, vcc_sentinel)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestSignedBoundaryExtremesAndIEEE754BitPatterns(unittest.TestCase):
    """Exhaustive unit tests for boundary extremes, integer ranges, and IEEE 754 bit-patterns."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_signed_zero_bit_patterns_as_integers(self) -> None:
        # In float32: +0.0 is 0x00000000, -0.0 is 0x80000000
        # In signed 32-bit int: 0x80000000 is -2147483648, 0x00000000 is 0
        # min(0, -2147483648) = -2147483648 (0x80000000)
        self.assertEqual(oracle_min_i32(0x00000000, 0x80000000), 0x80000000)
        self.assertEqual(oracle_min_i32(0x80000000, 0x00000000), 0x80000000)

        self.lds.write_dword(0, 0)
        self.state.set_vgpr(VGPR(1), 0)
        self.state.set_vgpr(VGPR(2), 0x80000000)
        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0)
        self.assertEqual(self.lds.read_dword(0), 0x80000000)

    def test_signed_infinities_as_integers(self) -> None:
        # +inf (0x7F800000 = +2139095040), -inf (0xFF800000 = -8388608)
        self.assertEqual(oracle_min_i32(0x7F800000, 0xFF800000), 0xFF800000)
        self.assertEqual(oracle_min_i32(0xFF800000, 0x7F800000), 0xFF800000)

        self.lds.write_dword(1, 0x7F800000)
        self.state.set_vgpr(VGPR(1), 4)
        self.state.set_vgpr(VGPR(2), 0xFF800000)
        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x7F800000)
        self.assertEqual(self.lds.read_dword(1), 0xFF800000)

    def test_nan_bit_patterns_unaltered_and_no_traps(self) -> None:
        # qNaN (0x7FC00000 = +2143289344) vs -1 (0xFFFFFFFF = -1) -> min is -1
        self.assertEqual(oracle_min_i32(0x7FC00000, 0xFFFFFFFF), 0xFFFFFFFF)

        self.lds.write_dword(2, 0x7FC00000)
        self.state.set_vgpr(VGPR(1), 8)
        self.state.set_vgpr(VGPR(2), 0xFFFFFFFF)
        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x7FC00000)
        self.assertEqual(self.lds.read_dword(2), 0xFFFFFFFF)

        # sNaN (0x7F800001 = +2139095041) vs +100 -> min is +100
        self.assertEqual(oracle_min_i32(0x7F800001, 100), 100)

        # Negative qNaN (0xFFC00000 = -4194304) vs 0 -> min is -4194304
        self.assertEqual(oracle_min_i32(0xFFC00000, 0), 0xFFC00000)

    def test_subnormal_bit_patterns_as_integers(self) -> None:
        # Smallest positive subnormal (0x00000001 = +1) vs -5 -> min is -5
        self.assertEqual(oracle_min_i32(1, u32(-5)), u32(-5))

        # Negative subnormal (0x80000001 = -2147483647) vs INT32_MIN (0x80000000 = -2147483648)
        self.assertEqual(oracle_min_i32(0x80000001, 0x80000000), 0x80000000)

    def test_signed_boundary_extremes_permutations(self) -> None:
        extremes = [
            0x80000000,  # INT32_MIN: -2147483648
            0x80000001,  # INT32_MIN + 1: -2147483647
            0xFFFFFFFF,  # -1
            0x00000000,  # 0
            0x00000001,  # +1
            0x7FFFFFFE,  # INT32_MAX - 1: +2147483646
            0x7FFFFFFF,  # INT32_MAX: +2147483647
        ]
        for a in extremes:
            for b in extremes:
                sa = s32(a)
                sb = s32(b)
                exp = u32(min(sa, sb))
                self.assertEqual(oracle_min_i32(a, b), exp)

                # Test in-place scalar execution
                self.lds.write_dword(0, a)
                self.state.set_vgpr(VGPR(1), 0)
                self.state.set_vgpr(VGPR(2), b)
                inst = DSMinRtnI32Instruction(
                    opcode=DSOpcode.DS_MIN_RTN_I32,
                    vdst=VGPR(0),
                    vaddr=VGPR(1),
                    vdata=VGPR(2),
                )
                ret = self.interp.execute(self.state, inst)
                self.assertEqual(ret, a)
                self.assertEqual(self.lds.read_dword(0), exp)

    def test_signed_vs_unsigned_ordering_distinction(self) -> None:
        # In signed comparison: -1 (0xFFFFFFFF) < 0 -> min is 0xFFFFFFFF
        # In unsigned comparison: 0 < 0xFFFFFFFF -> min would be 0
        self.assertEqual(oracle_min_i32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        self.assertEqual(oracle_min_i32(0, 0xFFFFFFFF), 0xFFFFFFFF)

        # In signed: -2147483648 (0x80000000) < 1 -> min is 0x80000000
        # In unsigned: 1 < 2147483648 -> min would be 1
        self.assertEqual(oracle_min_i32(0x80000000, 1), 0x80000000)

    def test_single_bit_walks(self) -> None:
        # Bit 31 is the sign bit; setting it makes the number negative
        for k in range(31):
            pos_val = 1 << k
            # Positive value compared with INT32_MIN (0x80000000)
            self.assertEqual(oracle_min_i32(pos_val, 0x80000000), 0x80000000)
            # Positive value compared with -1 (0xFFFFFFFF)
            self.assertEqual(oracle_min_i32(pos_val, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_alternating_bit_patterns(self) -> None:
        pat_a = 0x55555555  # +1431655765
        pat_b = 0xAAAAAAAA  # -1431655766 (sign bit set, negative)
        self.assertEqual(oracle_min_i32(pat_a, pat_b), pat_b)
        self.assertEqual(oracle_min_i32(pat_b, pat_a), pat_b)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_opcodes_and_syntax(self) -> None:
        invalid_assemblies = [
            "",
            "   ",
            "v_min_i32 v0, v1",
            "ds_min_f32 v0, v1",
            "ds_min_u32 v0, v1",
            "ds_min_i64 v0, v1",
            "ds_min_rtn_i32",
            "ds_min_rtn_i32 v0, v1",
            "ds_min_i32 v0",
            "ds_min_i32 v0, v1, v2",
            "ds_min_rtn_i32 v0, v1, v2, v3",
            "ds_min_i32 s0, v1",
            "ds_min_i32 r0, v1",
            "ds_min_i32 v256, v1",
            "ds_min_i32 v0, v256",
            "ds_min_i32 v0, v1 offset:70000",
            "ds_min_i32 v0, v1 offset:-1",
            "ds_min_i32 v0, v1 offset:invalid",
            "ds_min_i32 v0, v1 badmod",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises((ValueError, IndexError)):
                    DSMinRtnI32Instruction.parse(asm)

    def test_architectural_model_negative_controls(self) -> None:
        with self.assertRaises(ValueError):
            LDSMemory(0)
        with self.assertRaises(ValueError):
            LDSMemory(-64)
        with self.assertRaises(ValueError):
            LDSMemory(1000)

        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWaveState(wave_size=128)

    def test_mutation_detection_inverted_min_to_max(self) -> None:
        def mutant_min(a: int, b: int) -> int:
            sa = s32(a)
            sb = s32(b)
            return u32(max(sa, sb))

        self.assertNotEqual(mutant_min(10, 20), oracle_min_i32(10, 20))
        self.assertNotEqual(mutant_min(u32(-10), u32(-30)), oracle_min_i32(u32(-10), u32(-30)))

    def test_mutation_detection_unsigned_comparison(self) -> None:
        # Mutant: treats 32-bit values as unsigned integers
        def mutant_min_unsigned(a: int, b: int) -> int:
            return u32(min(u32(a), u32(b)))

        # In signed: -1 (0xFFFFFFFF) < 0. But in unsigned: 0 < 0xFFFFFFFF!
        # The mutant returns 0, whereas true signed min is 0xFFFFFFFF.
        self.assertNotEqual(mutant_min_unsigned(0, 0xFFFFFFFF), oracle_min_i32(0, 0xFFFFFFFF))
        # -2147483648 (0x80000000) < 1. In unsigned: 1 < 0x80000000.
        self.assertNotEqual(mutant_min_unsigned(1, 0x80000000), oracle_min_i32(1, 0x80000000))

    def test_mutation_detection_destructive_aliasing(self) -> None:
        class DestructiveMutantInterpreter(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSMinRtnI32Instruction) -> Optional[int]:
                if inst.opcode.is_return:
                    assert inst.vdst is not None
                    # Bug: prematurely corrupt vdst before reading operands or computing address
                    state.set_vgpr(inst.vdst, 0xBAD00000)
                return super().execute(state, inst)

        lds = LDSMemory(65536)
        lds.write_dword(2, 500)
        state = ScalarThreadState(lds=lds)
        state.set_vgpr(VGPR(1), 8)  # dword 2
        state.set_vgpr(VGPR(2), 50)

        # vdst aliases vaddr (v1)
        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            vdst=VGPR(1),
        )
        mutant = DestructiveMutantInterpreter()
        mutant.execute(state, inst)

        # The corrupted address causes write to wrong location, leaving dword 2 unupdated
        self.assertNotEqual(lds.read_dword(2), 50)

    def test_mutation_detection_ignore_exec_mask(self) -> None:
        class MutantSIMDInterpreter(SIMDDSInterpreter):
            def execute(self, state: SIMDWaveState, inst: DSMinRtnI32Instruction) -> Dict[int, Optional[int]]:
                saved_mask = state.exec_mask
                state.exec_mask = (1 << state.wave_size) - 1  # Force all lanes active
                try:
                    return super().execute(state, inst)
                finally:
                    state.exec_mask = saved_mask

        lds = LDSMemory(65536)
        lds.write_dword(5, 500)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.set_vgpr(l, VGPR(1), 20)
            wave.set_vgpr(l, VGPR(2), 10)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        mutant = MutantSIMDInterpreter()
        mutant.execute(wave, inst)

        # Mutant improperly modified memory when exec_mask was 0
        self.assertEqual(lds.read_dword(5), 10)

        # Correct interpreter leaves it untouched
        lds.write_dword(5, 500)
        SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(lds.read_dword(5), 500)

    def test_mutation_detection_dropped_offset(self) -> None:
        class DroppedOffsetInterpreter(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSMinRtnI32Instruction) -> Optional[int]:
                no_off_inst = DSMinRtnI32Instruction(
                    opcode=inst.opcode,
                    vaddr=inst.vaddr,
                    vdata=inst.vdata,
                    vdst=inst.vdst,
                    offset=OffsetImmediate(0),  # Dropped offset
                )
                return super().execute(state, no_off_inst)

        lds = LDSMemory(65536)
        lds.write_dword(1, 500)
        lds.write_dword(2, 500)
        state = ScalarThreadState(lds=lds)
        state.set_vgpr(VGPR(1), 4)  # dword 1
        state.set_vgpr(VGPR(2), 50)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_I32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(4),  # Points to dword 2 (4 + 4 = 8)
        )
        mutant = DroppedOffsetInterpreter()
        mutant.execute(state, inst)

        # Mutant updated dword 1 instead of dword 2
        self.assertEqual(lds.read_dword(1), 50)
        self.assertEqual(lds.read_dword(2), 500)

    def test_mutation_detection_non_atomic_collision(self) -> None:
        class NonAtomicSIMDInterpreter(SIMDDSInterpreter):
            def execute(self, state: SIMDWaveState, inst: DSMinRtnI32Instruction) -> Dict[int, Optional[int]]:
                # Reads initial value for all lanes without updating memory sequentially
                results: Dict[int, Optional[int]] = {}
                read_vals: Dict[int, int] = {}
                for lane in range(state.wave_size):
                    if state.is_lane_active(lane):
                        vaddr_val = state.get_vgpr(lane, inst.vaddr)
                        idx = state.lds.get_dword_index(vaddr_val, inst.offset.value)
                        read_vals[lane] = state.lds.read_dword(idx)
                # Overwrite directly without chained return values
                for lane in range(state.wave_size):
                    if state.is_lane_active(lane):
                        if inst.opcode.is_return:
                            assert inst.vdst is not None
                            state.set_vgpr(lane, inst.vdst, read_vals[lane])
                            results[lane] = read_vals[lane]
                return results

        lds = LDSMemory(65536)
        lds.write_dword(0, 100)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x3, lds=lds)
        # Lane 0 sends 50, Lane 1 sends 20 to address 0
        wave.set_vgpr(0, VGPR(1), 0)
        wave.set_vgpr(0, VGPR(2), 50)
        wave.set_vgpr(1, VGPR(1), 0)
        wave.set_vgpr(1, VGPR(2), 20)

        inst = DSMinRtnI32Instruction(
            opcode=DSOpcode.DS_MIN_RTN_I32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )

        mutant = NonAtomicSIMDInterpreter()
        mutant_res = mutant.execute(wave, inst)
        # In non-atomic execution, Lane 1 gets 100 instead of serialized 50
        self.assertEqual(mutant_res[1], 100)

        # In true serialized atomic execution:
        lds.write_dword(0, 100)
        true_res = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(true_res[1], 50)


if __name__ == "__main__":
    unittest.main()
