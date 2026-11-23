"""Semantic interpreter tests and reference hardware-matching oracle for RDNA2 ds_xor_b32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) 32-bit atomic bitwise XOR operations:
  ds_xor_b32 vaddr, vdata [offset:N]
  ds_xor_rtn_b32 vdst, vaddr, vdata [offset:N]

RDNA2 LDS Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vaddr: VGPR containing byte offset into LDS allocation.
     - vdata: VGPR containing 32-bit bitmask to XOR atomically into LDS storage.
     - vdst: (Returning form only) VGPR receiving pre-atomic dword from memory.
     - offset: 16-bit unsigned immediate byte offset [0, 65535].
  2. Addressing and Alignment:
     effective_address = (vaddr + offset) & 0xFFFFFFFF
     dword_index = (effective_address >> 2) & (lds_dwords - 1)
     Unaligned byte offsets [1:0] are dropped by hardware.
  3. Atomic Operation:
     old_val = LDS[dword_index]
     new_val = (old_val ^ vdata) & 0xFFFFFFFF
     LDS[dword_index] = new_val
     if returning: vdst = old_val
  4. Execution Masking (EXEC):
     Lanes disabled by EXEC do not perform atomic updates and do not alter vdst.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import random
import re
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import support  # noqa: E402
    import lds_run as L  # noqa: E402
    TRANSLATOR_ERROR = None
except Exception as _err:
    support = None  # type: ignore
    L = None  # type: ignore
    TRANSLATOR_ERROR = str(_err)

MASK32: int = 0xFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default capacity (16K dwords)

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def lds_effective_dword(vaddr: int, offset: int = 0, lds_dwords: int = DEFAULT_LDS_DWORDS) -> int:
    """Computes wrapped dword index in LDS storage."""
    eff_byte = (u32(vaddr) + (offset & 0xFFFF)) & MASK32
    return (eff_byte >> 2) & (lds_dwords - 1)


def ref_atomic_xor_b32(current_val: int, xor_mask: int) -> tuple[int, int]:
    """Pure bitwise reference oracle for 32-bit atomic XOR: returns (old_val, new_val)."""
    old = u32(current_val)
    new = u32(old ^ u32(xor_mask))
    return old, new


class LDSStorage:
    """Simulates on-chip Local Data Share (LDS) dword-array memory."""

    def __init__(self, dword_count: int = DEFAULT_LDS_DWORDS) -> None:
        self.capacity: int = dword_count
        self.dwords: list[int] = [0] * dword_count

    def read_dword(self, index: int) -> int:
        return self.dwords[index & (self.capacity - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.dwords[index & (self.capacity - 1)] = u32(val)

    def atomic_xor(self, vaddr: int, vdata: int, offset: int = 0) -> tuple[int, int]:
        idx = lds_effective_dword(vaddr, offset, self.capacity)
        old_val = self.read_dword(idx)
        _, new_val = ref_atomic_xor_b32(old_val, vdata)
        self.write_dword(idx, new_val)
        return old_val, new_val

class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    LDS = "lds"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register representation."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count < 1 or self.index + self.count > 256:
            raise ValueError(f"Register count exceeds bounds: {self.count} at {self.index}")

    def format_asm(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        if self.count == 1:
            return f"{prefix}{self.index}"
        return f"{prefix}[{self.index}:{self.index + self.count - 1}]"


class Opcode(enum.Enum):
    DS_XOR_B32 = "ds_xor_b32"
    DS_XOR_RTN_B32 = "ds_xor_rtn_b32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST for Local Data Share XOR instructions."""
    opcode: Opcode
    vaddr: Register
    vdata: Register
    vdst: Register | None = None
    offset: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if self.vaddr.kind != RegKind.VGPR:
            raise ValueError("vaddr must be a VGPR")
        if self.vdata.kind != RegKind.VGPR:
            raise ValueError("vdata must be a VGPR")
        if self.opcode == Opcode.DS_XOR_RTN_B32:
            if self.vdst is None or self.vdst.kind != RegKind.VGPR:
                raise ValueError("ds_xor_rtn_b32 requires a destination VGPR (vdst)")
        elif self.vdst is not None:
            raise ValueError("ds_xor_b32 does not accept a destination register")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"LDS immediate offset must be in range [0, 65535]: {self.offset}")

    def format_asm(self) -> str:
        parts = [self.opcode.value]
        if self.vdst is not None:
            parts.append(self.vdst.format_asm() + ",")
        parts.append(self.vaddr.format_asm() + ",")
        parts.append(self.vdata.format_asm())
        if self.offset > 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)

class ScalarLDSInterpreter:
    """Deterministic scalar instruction interpreter executing LDS instructions."""

    def __init__(self, capacity_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        self.storage: LDSStorage = LDSStorage(capacity_dwords)
        self.vgprs: list[int] = [0] * 256

    def read_vgpr(self, reg_idx: int) -> int:
        return self.vgprs[reg_idx & 0xFF]

    def write_vgpr(self, reg_idx: int, val: int) -> None:
        self.vgprs[reg_idx & 0xFF] = u32(val)

    def execute(self, inst: Instruction) -> tuple[int, int]:
        addr = self.read_vgpr(inst.vaddr.index)
        data = self.read_vgpr(inst.vdata.index)
        old_val, new_val = self.storage.atomic_xor(addr, data, inst.offset)
        if inst.opcode == Opcode.DS_XOR_RTN_B32 and inst.vdst is not None:
            self.write_vgpr(inst.vdst.index, old_val)
        return old_val, new_val

class SIMDWavefrontInterpreter:
    """SIMD wavefront interpreter simulating multi-lane execution under EXEC masking."""

    def __init__(self, wave_size: int = 32, capacity_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64: {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.storage: LDSStorage = LDSStorage(capacity_dwords)
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]

    def set_exec_mask(self, mask: int) -> None:
        max_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & max_mask

    def set_lane_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx & 0xFF] = u32(val)

    def get_lane_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx & 0xFF]

    def step(self, inst: Instruction) -> list[tuple[int, int] | None]:
        results: list[tuple[int, int] | None] = []
        for lane in range(self.wave_size):
            if not bool(self.exec_mask & (1 << lane)):
                results.append(None)
                continue
            addr = self.get_lane_vgpr(lane, inst.vaddr.index)
            data = self.get_lane_vgpr(lane, inst.vdata.index)
            old_val, new_val = self.storage.atomic_xor(addr, data, inst.offset)
            if inst.opcode == Opcode.DS_XOR_RTN_B32 and inst.vdst is not None:
                self.set_lane_vgpr(lane, inst.vdst.index, old_val)
            results.append((old_val, new_val))
        return results

class TestLdsXorB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.lds = LDSStorage(DEFAULT_LDS_DWORDS)

    def test_basic_xor_truth_table(self) -> None:
        # Identity: X ^ 0 == X
        old, new = ref_atomic_xor_b32(0x12345678, 0x00000000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12345678)

        # Inversion: X ^ 0xFFFFFFFF == ~X
        old, new = ref_atomic_xor_b32(0x12345678, 0xFFFFFFFF)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0xEDCBA987)

        # Involution: (X ^ Y) ^ Y == X
        k = 0xA5A5A5A5
        old1, new1 = ref_atomic_xor_b32(0x12345678, k)
        old2, new2 = ref_atomic_xor_b32(new1, k)
        self.assertEqual(old2, new1)
        self.assertEqual(new2, 0x12345678)

    def test_lds_storage_atomic_operation(self) -> None:
        self.lds.write_dword(16, 0xAAAAAAAA)
        old, new = self.lds.atomic_xor(vaddr=64, vdata=0x55555555, offset=0)
        self.assertEqual(old, 0xAAAAAAAA)
        self.assertEqual(new, 0xFFFFFFFF)
        self.assertEqual(self.lds.read_dword(16), 0xFFFFFFFF)

    def test_unaligned_address_bits_ignored(self) -> None:
        # Bytes 64, 65, 66, 67 all index dword 16
        self.lds.write_dword(16, 0x00000000)
        self.lds.atomic_xor(vaddr=65, vdata=0x00000001)
        self.lds.atomic_xor(vaddr=66, vdata=0x00000002)
        self.lds.atomic_xor(vaddr=67, vdata=0x00000004)
        self.assertEqual(self.lds.read_dword(16), 0x00000007)

class TestLdsXorB32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_formatting_non_returning(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_XOR_B32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=16,
        )
        self.assertEqual(inst.format_asm(), "ds_xor_b32 v1, v2 offset:16")

    def test_instruction_formatting_returning(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_XOR_RTN_B32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=32,
        )
        self.assertEqual(inst.format_asm(), "ds_xor_rtn_b32 v5, v1, v2 offset:32")

class TestLdsXorB32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter()

    def test_scalar_execution_non_returning(self) -> None:
        self.interp.storage.write_dword(10, 0x11111111)
        self.interp.write_vgpr(1, 40)  # byte address 40 -> dword 10
        self.interp.write_vgpr(2, 0x22222222)
        inst = Instruction(Opcode.DS_XOR_B32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        old, new = self.interp.execute(inst)
        self.assertEqual(old, 0x11111111)
        self.assertEqual(new, 0x33333333)
        self.assertEqual(self.interp.storage.read_dword(10), 0x33333333)

    def test_scalar_execution_returning(self) -> None:
        self.interp.storage.write_dword(10, 0xF0F0F0F0)
        self.interp.write_vgpr(1, 40)
        self.interp.write_vgpr(2, 0x0F0F0F0F)
        self.interp.write_vgpr(5, 0xCAFEFEED)
        inst = Instruction(
            Opcode.DS_XOR_RTN_B32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        old, new = self.interp.execute(inst)
        self.assertEqual(old, 0xF0F0F0F0)
        self.assertEqual(new, 0xFFFFFFFF)
        self.assertEqual(self.interp.read_vgpr(5), 0xF0F0F0F0)
        self.assertEqual(self.interp.storage.read_dword(10), 0xFFFFFFFF)

class TestLdsXorB32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_lanes(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        # Each lane writes to its own dword: lane * 4
        for lane in range(32):
            simd.storage.write_dword(lane, 0x00000000)
            simd.set_lane_vgpr(lane, 1, lane * 4)
            simd.set_lane_vgpr(lane, 2, 1 << lane)

        inst = Instruction(Opcode.DS_XOR_B32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(simd.storage.read_dword(lane), 1 << lane)

    def test_exec_mask_isolation(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.storage.write_dword(0, 0x55555555)
        for lane in range(32):
            simd.set_lane_vgpr(lane, 1, 0)
            simd.set_lane_vgpr(lane, 2, 0xFFFFFFFF)
            simd.set_lane_vgpr(lane, 5, 0xDEADBEEF)

        # Only lane 0 is active
        simd.set_exec_mask(0x00000001)
        inst = Instruction(
            Opcode.DS_XOR_RTN_B32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        simd.step(inst)

        # Lane 0 received old value
        self.assertEqual(simd.get_lane_vgpr(0, 5), 0x55555555)
        # Disabled lanes must retain original register contents
        for lane in range(1, 32):
            self.assertEqual(simd.get_lane_vgpr(lane, 5), 0xDEADBEEF)

    def test_multi_lane_contention_same_address(self) -> None:
        # All 32 lanes atomic XOR to address 0
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.storage.write_dword(0, 0x00000000)
        for lane in range(32):
            simd.set_lane_vgpr(lane, 1, 0)
            simd.set_lane_vgpr(lane, 2, 1 << lane)

        inst = Instruction(Opcode.DS_XOR_B32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        simd.step(inst)
        # Final result must be bitwise OR/XOR of all 1 << lane = 0xFFFFFFFF
        self.assertEqual(simd.storage.read_dword(0), 0xFFFFFFFF)

class TestLdsXorB32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter()

    def test_vdst_aliases_vaddr(self) -> None:
        # vdst and vaddr both use v1
        self.interp.storage.write_dword(10, 0xCAFEBABE)
        self.interp.write_vgpr(1, 40)            # vaddr = 40 (dword 10)
        self.interp.write_vgpr(2, 0x0000000F)    # vdata = 0xF
        inst = Instruction(
            Opcode.DS_XOR_RTN_B32,
            vdst=Register(RegKind.VGPR, 1),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        self.interp.execute(inst)
        # v1 should now contain old memory value 0xCAFEBABE
        self.assertEqual(self.interp.read_vgpr(1), 0xCAFEBABE)
        self.assertEqual(self.interp.storage.read_dword(10), 0xCAFEBAB1)

    def test_vdst_aliases_vdata(self) -> None:
        # vdst and vdata both use v2
        self.interp.storage.write_dword(10, 0x12345678)
        self.interp.write_vgpr(1, 40)
        self.interp.write_vgpr(2, 0x00FF00FF)
        inst = Instruction(
            Opcode.DS_XOR_RTN_B32,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        self.interp.execute(inst)
        # v2 should now contain old memory value 0x12345678
        self.assertEqual(self.interp.read_vgpr(2), 0x12345678)
        self.assertEqual(self.interp.storage.read_dword(10), 0x12CB5687)

if __name__ == "__main__":
    unittest.main()
