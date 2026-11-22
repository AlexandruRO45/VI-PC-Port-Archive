"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_nor_rtn_b32.

Models 32-bit bitwise NOR and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_nor_rtn_b32: Atomic 32-bit bitwise NOR returning previous value to destination VGPR.
  - ds_nor_b32: Atomic 32-bit bitwise NOR discarding previous value.

RDNA2 LDS / Bitwise NOR ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Bitwise NOR Logic:
     Operands are treated as 32-bit bitmasks.
     nor(A, B) = ~(A | B) & 0xFFFFFFFF.
     Bitwise truth table:
       0 NOR 0 = 1
       0 NOR 1 = 0
       1 NOR 0 = 0
       1 NOR 1 = 0
     0 NOR 0 = 0xFFFFFFFF; 0xFFFFFFFF NOR any = 0; 0 NOR x = ~x.
  3. Atomic Operations & Return Semantics:
     Returning variants (ds_nor_rtn_b32) atomically fetch the previous 32-bit value,
     write the bitwise NOR into LDS, and return the previous value to the destination VGPR.
     Non-returning variants (ds_nor_b32) perform the atomic NOR and discard the old value.

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
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def oracle_nor_b32(a: int, b: int) -> int:
    """Reference 32-bit bitwise NOR returning 32-bit unsigned bit pattern."""
    ua = a & MASK32
    ub = b & MASK32
    return (~(ua | ub)) & MASK32


def ref_ds_nor_rtn_b32(current_mem: int, operand: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nor_rtn_b32.

    Returns:
        (old_val, new_mem) as unsigned 32-bit integers.
    """
    old_val = u32(current_mem)
    new_mem = oracle_nor_b32(old_val, operand)
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

    def atomic_nor_b32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic bitwise NOR at (byte_addr + offset), returning new value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_nor_b32(old_val, val)
        self.dwords[idx] = new_val
        return new_val

    def atomic_nor_rtn_b32(self, byte_addr: int, val: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic bitwise NOR at (byte_addr + offset), returning (old_val, new_val)."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_nor_b32(old_val, val)
        self.dwords[idx] = new_val
        return old_val, new_val


class TestLDSNorRtnB32ModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_oracle_nor_b32_truth_table(self) -> None:
        # 0 NOR 0 = 0xFFFFFFFF
        self.assertEqual(oracle_nor_b32(0, 0), 0xFFFFFFFF)
        # 0xFFFFFFFF NOR 0 = 0
        self.assertEqual(oracle_nor_b32(0xFFFFFFFF, 0), 0)
        # 0 NOR 0xFFFFFFFF = 0
        self.assertEqual(oracle_nor_b32(0, 0xFFFFFFFF), 0)
        # 0xFFFFFFFF NOR 0xFFFFFFFF = 0
        self.assertEqual(oracle_nor_b32(0xFFFFFFFF, 0xFFFFFFFF), 0)

    def test_oracle_nor_b32_bit_patterns(self) -> None:
        # Alternating patterns: 0x55555555 | 0xAAAAAAAA = 0xFFFFFFFF -> ~(0xFFFFFFFF) = 0
        self.assertEqual(oracle_nor_b32(0x55555555, 0xAAAAAAAA), 0)
        # 0x55555555 NOR 0 = ~0x55555555 = 0xAAAAAAAA
        self.assertEqual(oracle_nor_b32(0x55555555, 0), 0xAAAAAAAA)
        # 0x0F0F0F0F NOR 0x00FF00FF = ~(0x0FFF0FFF) = 0xF000F000
        self.assertEqual(oracle_nor_b32(0x0F0F0F0F, 0x00FF00FF), 0xF000F000)

    def test_ref_ds_nor_rtn_b32(self) -> None:
        old_val, new_val = ref_ds_nor_rtn_b32(0x00000000, 0x00000000)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0xFFFFFFFF)

        old_val2, new_val2 = ref_ds_nor_rtn_b32(0x12345678, 0x87654321)
        expected = (~(0x12345678 | 0x87654321)) & MASK32
        self.assertEqual(old_val2, 0x12345678)
        self.assertEqual(new_val2, expected)

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
        self.lds.write_dword(10, 0xAABBCCDD)
        self.assertEqual(self.lds.read_dword(10), 0xAABBCCDD)

    def test_lds_atomic_nor_operations(self) -> None:
        # Set initial value to 0x55555555 at dword 8 (byte addr 32)
        self.lds.write_dword(8, 0x55555555)
        old_val, new_val = self.lds.atomic_nor_rtn_b32(32, 0)
        self.assertEqual(old_val, 0x55555555)
        self.assertEqual(new_val, 0xAAAAAAAA)
        self.assertEqual(self.lds.read_dword(8), 0xAAAAAAAA)

        # Atomic nor with 0xAAAAAAAA: ~(0xAAAAAAAA | 0xAAAAAAAA) = ~0xAAAAAAAA = 0x55555555
        old_val2, new_val2 = self.lds.atomic_nor_rtn_b32(32, 0xAAAAAAAA)
        self.assertEqual(old_val2, 0xAAAAAAAA)
        self.assertEqual(new_val2, 0x55555555)
        self.assertEqual(self.lds.read_dword(8), 0x55555555)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(str, enum.Enum):
    """Data Share (DS) atomic opcodes for 32-bit bitwise NOR."""
    DS_NOR_RTN_B32 = "ds_nor_rtn_b32"
    DS_NOR_B32 = "ds_nor_b32"

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
class DSNorRtnB32Instruction:
    """Symbolic AST node for ds_nor_rtn_b32 / ds_nor_b32 instruction."""
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
    def parse(cls, asm: str) -> DSNorRtnB32Instruction:
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
        inst_rtn = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
            vdst=VGPR(0),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst_rtn.to_assembly(), "ds_nor_rtn_b32 v0, v1, v2 offset:16")

        inst_nortn = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        self.assertEqual(inst_nortn.to_assembly(), "ds_nor_b32 v1, v2")

        with self.assertRaises(ValueError):
            DSNorRtnB32Instruction(
                opcode=DSOpcode.DS_NOR_RTN_B32,
                vaddr=VGPR(1),
                vdata=VGPR(2),
                vdst=None,
            )

        with self.assertRaises(ValueError):
            DSNorRtnB32Instruction(
                opcode=DSOpcode.DS_NOR_B32,
                vaddr=VGPR(1),
                vdata=VGPR(2),
                vdst=VGPR(0),
            )

    def test_instruction_parsing_and_roundtrip(self) -> None:
        asm1 = "ds_nor_rtn_b32 v0, v1, v2 offset:32"
        inst1 = DSNorRtnB32Instruction.parse(asm1)
        self.assertEqual(inst1.opcode, DSOpcode.DS_NOR_RTN_B32)
        self.assertEqual(inst1.vdst, VGPR(0))
        self.assertEqual(inst1.vaddr, VGPR(1))
        self.assertEqual(inst1.vdata, VGPR(2))
        self.assertEqual(inst1.offset.value, 32)
        self.assertEqual(inst1.to_assembly(), asm1)

        asm2 = "ds_nor_b32 v5, v6"
        inst2 = DSNorRtnB32Instruction.parse(asm2)
        self.assertEqual(inst2.opcode, DSOpcode.DS_NOR_B32)
        self.assertIsNone(inst2.vdst)
        self.assertEqual(inst2.vaddr, VGPR(5))
        self.assertEqual(inst2.vdata, VGPR(6))
        self.assertEqual(inst2.offset.value, 0)
        self.assertEqual(inst2.to_assembly(), asm2)

        asm3 = "ds_nor_rtn_b32 v10, v20, v30 offset:4 gds"
        inst3 = DSNorRtnB32Instruction.parse(asm3)
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
    """Deterministic scalar interpreter for ds_nor_rtn_b32 and ds_nor_b32."""

    def execute(self, state: ScalarThreadState, inst: DSNorRtnB32Instruction) -> Optional[int]:
        vaddr_val = state.get_vgpr(inst.vaddr)
        vdata_val = state.get_vgpr(inst.vdata)

        old_val, new_val = state.lds.atomic_nor_rtn_b32(
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

    def test_ds_nor_rtn_b32_execution(self) -> None:
        # LDS[dword 10] = 0x00000000
        self.lds.write_dword(10, 0x00000000)
        # v1 (vaddr) = 40 (byte address 40 -> dword 10)
        self.state.set_vgpr(VGPR(1), 40)
        # v2 (vdata) = 0x00000000
        self.state.set_vgpr(VGPR(2), 0x00000000)
        # v0 (vdst) = sentinel 0xDEADBEEF
        self.state.set_vgpr(VGPR(0), 0xDEADBEEF)

        inst = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)

        # Old value was 0
        self.assertEqual(ret, 0)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 0)
        # ~(0 | 0) = 0xFFFFFFFF
        self.assertEqual(self.lds.read_dword(10), 0xFFFFFFFF)

    def test_ds_nor_b32_non_return(self) -> None:
        self.lds.write_dword(5, 0x0000FFFF)
        self.state.set_vgpr(VGPR(1), 20)  # dword 5
        self.state.set_vgpr(VGPR(2), 0xFFFF0000)
        self.state.set_vgpr(VGPR(0), 0xCAFEBABE)

        inst = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_B32,
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        ret = self.interp.execute(self.state, inst)

        self.assertIsNone(ret)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 0xCAFEBABE)  # untouched
        # ~(0x0000FFFF | 0xFFFF0000) = ~0xFFFFFFFF = 0
        self.assertEqual(self.lds.read_dword(5), 0x00000000)

    def test_double_nor_not_identity(self) -> None:
        # Starting with 0x12345678
        initial = 0x12345678
        self.lds.write_dword(2, initial)
        self.state.set_vgpr(VGPR(1), 8)
        self.state.set_vgpr(VGPR(2), 0)

        inst = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )

        # First NOR with 0: ~(initial | 0) = ~initial
        ret1 = self.interp.execute(self.state, inst)
        self.assertEqual(ret1, initial)
        inverted = (~initial) & MASK32
        self.assertEqual(self.lds.read_dword(2), inverted)

        # Second NOR with 0: ~(~initial | 0) = ~(~initial) = initial
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, inverted)
        self.assertEqual(self.lds.read_dword(2), initial)

    def test_offset_immediate_computation(self) -> None:
        self.lds.write_dword(20, 0x55555555)
        # vaddr = 64 (dword 16), offset = 16 bytes -> dword 20 (80 bytes)
        self.state.set_vgpr(VGPR(1), 64)
        self.state.set_vgpr(VGPR(2), 0xAAAAAAAA)

        inst = DSNorRtnB32Instruction(
            opcode=DSOpcode.DS_NOR_RTN_B32,
            vdst=VGPR(0),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x55555555)
        self.assertEqual(self.lds.read_dword(20), 0)


if __name__ == "__main__":
    unittest.main()
