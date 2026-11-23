"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_mskor_b32 and ds_mskor_rtn_b32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - 32-bit Atomic Masked OR (Bitfield Insert / Mask-and-OR):
    - RDNA2 ISA: tmp = M[A]; M[A] = (tmp & ~MASK) | DATA
    - ds_mskor_b32 vaddr, vmask, vdata [offset:N]:
      Non-returning form:
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = ((old_val & ~vmask) | vdata) & 0xFFFFFFFF
      The prior value at LDS[dword_idx] is discarded.
    - ds_mskor_rtn_b32 vdst, vaddr, vmask, vdata [offset:N]:
      Returning form:
        Atomically updates memory according to the same formula and returns pre-atomic
        value old_val in destination VGPR vdst.
  - Active lane masking:
    - Governed by EXEC mask (Wave32 / Wave64). Inactive lanes do not mutate LDS storage
      and leave destination VGPRs unmodified.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


def ds_mskor_b32_oracle(old_val: int, mask: int, data: int) -> int:
    """Pure reference oracle for RDNA2 ds_mskor_b32: (old & ~mask) | data."""
    old = old_val & MASK32
    m = mask & MASK32
    d = data & MASK32
    return ((old & (~m & MASK32)) | d) & MASK32


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords

    def read_dword(self, byte_addr: int) -> int:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_mskor(self, byte_addr: int, mask: int, data: int) -> int:
        """Atomic masked OR. Returns pre-atomic dword value."""
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = ds_mskor_b32_oracle(old, mask, data)
        return old


def ds_mskor_b32_execute_model(
    lds: LDSMemory, vaddr: int, vmask: int, vdata: int, offset: int = 0
) -> Tuple[Optional[int], int]:
    """Executes non-returning ds_mskor_b32. Returns (None, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_mskor(eff_byte, vmask, vdata)
    new_val = lds.read_dword(eff_byte)
    return None, new_val


def ds_mskor_rtn_b32_execute_model(
    lds: LDSMemory, vaddr: int, vmask: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_mskor_rtn_b32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_mskor(eff_byte, vmask, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSMskorB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_byte_replacement(self) -> None:
        # Replace second byte (bits [23:16]) of 0x12345678 with 0xAA -> 0x12AA5678
        res = ds_mskor_b32_oracle(0x12345678, 0x00FF0000, 0x00AA0000)
        self.assertEqual(res, 0x12AA5678)

    def test_oracle_clear_bits_when_data_is_zero(self) -> None:
        # mask = 0x0F0F0F0F, data = 0 -> clears masked nibbles
        res = ds_mskor_b32_oracle(0xFFFFFFFF, 0x0F0F0F0F, 0x00000000)
        self.assertEqual(res, 0xF0F0F0F0)

    def test_oracle_set_bits_when_mask_matches_data(self) -> None:
        # mask = 0x0F0F0F0F, data = 0x0F0F0F0F -> sets masked nibbles
        res = ds_mskor_b32_oracle(0x00000000, 0x0F0F0F0F, 0x0F0F0F0F)
        self.assertEqual(res, 0x0F0F0F0F)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 0x12345678)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 0x12345678)
        self.assertEqual(lds.read_dword(0x40 + 4096), 0x12345678)

    def test_model_non_returning_and_returning(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 0x12345678)

        old, new = ds_mskor_rtn_b32_execute_model(lds, 0x100, 0x00FF0000, 0x00AA0000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12AA5678)

        rtn, new2 = ds_mskor_b32_execute_model(lds, 0x100, 0x000000FF, 0x00000099)
        self.assertIsNone(rtn)
        self.assertEqual(new2, 0x12AA5699)


class DSOpcode(str, enum.Enum):
    DS_MSKOR_B32 = "ds_mskor_b32"
    DS_MSKOR_RTN_B32 = "ds_mskor_rtn_b32"


@dataclasses.dataclass(frozen=True)
class VGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")

    def __str__(self) -> str:
        return f"v{self.index}"


@dataclasses.dataclass(frozen=True)
class SGPR:
    index: int

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    def __str__(self) -> str:
        return f"s{self.index}"


@dataclasses.dataclass(frozen=True)
class OffsetImmediate:
    value: int

    def __post_init__(self) -> None:
        if not (0 <= self.value <= 0xFFFF):
            raise ValueError(f"OffsetImmediate out of range [0, 0xFFFF]: {self.value}")

    def __str__(self) -> str:
        return f"offset:{self.value}" if self.value != 0 else ""


@dataclasses.dataclass(frozen=True)
class DSMskorInstruction:
    opcode: DSOpcode
    vaddr: VGPR
    vmask: VGPR
    vdata: VGPR
    vdst: Optional[VGPR] = None
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))

    def __post_init__(self) -> None:
        if self.opcode == DSOpcode.DS_MSKOR_B32 and self.vdst is not None:
            raise ValueError(f"{self.opcode} must not have a destination register")
        if self.opcode == DSOpcode.DS_MSKOR_RTN_B32 and self.vdst is None:
            raise ValueError(f"{self.opcode} requires a destination register")

    def to_asm(self) -> str:
        off_str = f" {self.offset}" if self.offset.value != 0 else ""
        if self.opcode == DSOpcode.DS_MSKOR_RTN_B32:
            return f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vmask}, {self.vdata}{off_str}"
        return f"{self.opcode.value} {self.vaddr}, {self.vmask}, {self.vdata}{off_str}"


class TestDSMskorB32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_non_returning(self) -> None:
        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_B32,
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_MSKOR_B32)
        self.assertEqual(inst.vaddr.index, 1)
        self.assertEqual(inst.vmask.index, 2)
        self.assertEqual(inst.vdata.index, 3)
        self.assertIsNone(inst.vdst)
        self.assertEqual(inst.offset.value, 16)
        self.assertEqual(inst.to_asm(), "ds_mskor_b32 v1, v2, v3 offset:16")

    def test_ast_construction_returning(self) -> None:
        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
            offset=OffsetImmediate(0),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_MSKOR_RTN_B32)
        self.assertEqual(inst.vdst.index, 5)
        self.assertEqual(inst.to_asm(), "ds_mskor_rtn_b32 v5, v1, v2, v3")


class RegisterFile:
    """Architectural VGPR and SGPR register storage."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32


class ScalarThreadState:
    """Thread state for scalar LDS execution."""

    def __init__(self, lds: LDSMemory) -> None:
        self.rf: RegisterFile = RegisterFile()
        self.lds: LDSMemory = lds


class ScalarDSInterpreter:
    """Deterministic scalar interpreter for DS MSKOR instructions."""

    def execute(self, state: ScalarThreadState, inst: DSMskorInstruction) -> Tuple[Optional[int], int]:
        addr = state.rf.get_vgpr(inst.vaddr)
        mask = state.rf.get_vgpr(inst.vmask)
        data = state.rf.get_vgpr(inst.vdata)
        eff_addr = (addr + inst.offset.value) & MASK32

        if inst.opcode == DSOpcode.DS_MSKOR_RTN_B32:
            old_val = state.lds.atomic_mskor(eff_addr, mask, data)
            new_val = state.lds.read_dword(eff_addr)
            assert inst.vdst is not None
            state.rf.set_vgpr(inst.vdst, old_val)
            return old_val, new_val
        else:
            old_val = state.lds.atomic_mskor(eff_addr, mask, data)
            new_val = state.lds.read_dword(eff_addr)
            return None, new_val


class TestDSMskorB32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution(self) -> None:
        self.lds.write_dword(0x40, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0x00FF0000)
        self.state.rf.set_vgpr(VGPR(3), 0x00AA0000)
        self.state.rf.set_vgpr(VGPR(5), 0xDEADBEEF)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12AA5678)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(5)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x40), 0x12AA5678)


class SIMDWaveState:
    """Wavefront state for SIMD execution (Wave32 or Wave64)."""

    def __init__(
        self,
        wave_size: int = 32,
        exec_mask: int = 0xFFFFFFFF,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError("wave_size must be 32 or 64")
        self.wave_size: int = wave_size
        self.exec_mask: int = exec_mask & ((1 << wave_size) - 1) if wave_size == 64 else exec_mask & MASK32
        self.lds: LDSMemory = lds or LDSMemory()
        self.lanes: List[RegisterFile] = [RegisterFile() for _ in range(wave_size)]

    def is_lane_active(self, lane_id: int) -> bool:
        return bool((self.exec_mask >> lane_id) & 1)


class SIMDDSInterpreter:
    """SIMD interpreter executing LDS operations with EXEC masking."""

    def execute(
        self, wave: SIMDWaveState, inst: DSMskorInstruction
    ) -> List[Tuple[int, Optional[int], int]]:
        """Returns list of (lane_id, old_val, new_val) for executed active lanes."""
        results = []
        for lane_id in range(wave.wave_size):
            if not wave.is_lane_active(lane_id):
                continue
            rf = wave.lanes[lane_id]
            addr = rf.get_vgpr(inst.vaddr)
            mask = rf.get_vgpr(inst.vmask)
            data = rf.get_vgpr(inst.vdata)
            eff_addr = (addr + inst.offset.value) & MASK32

            if inst.opcode == DSOpcode.DS_MSKOR_RTN_B32:
                old_val = wave.lds.atomic_mskor(eff_addr, mask, data)
                new_val = wave.lds.read_dword(eff_addr)
                assert inst.vdst is not None
                rf.set_vgpr(inst.vdst, old_val)
                results.append((lane_id, old_val, new_val))
            else:
                old_val = wave.lds.atomic_mskor(eff_addr, mask, data)
                new_val = wave.lds.read_dword(eff_addr)
                results.append((lane_id, None, new_val))
        return results


class TestDSMskorB32SIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_full_wave32(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0x12345678)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 0x000000FF)
            wave.lanes[l].set_vgpr(VGPR(3), l)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 32)
        for l in range(32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0x12345678)
            self.assertEqual(lds.read_dword(l * 4), 0x12345600 | l)

    def test_simd_divergent_lanes(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0x11111111)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 0xFFFFFFFF)
            wave.lanes[l].set_vgpr(VGPR(3), 0x22222222)
            wave.lanes[l].set_vgpr(VGPR(5), 0xCAFEBABE)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        SIMDDSInterpreter().execute(wave, inst)
        for l in range(32):
            if l % 2 == 0:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0x11111111)
                self.assertEqual(lds.read_dword(l * 4), 0x22222222)
            else:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xCAFEBABE)
                self.assertEqual(lds.read_dword(l * 4), 0x11111111)


class TestDSMskorB32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_vdst_aliases_vaddr(self) -> None:
        self.lds.write_dword(0x40, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0x00FF0000)
        self.state.rf.set_vgpr(VGPR(3), 0x00AA0000)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(1),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12AA5678)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(1)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x40), 0x12AA5678)

    def test_vdst_aliases_vmask(self) -> None:
        self.lds.write_dword(0x40, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0x00FF0000)
        self.state.rf.set_vgpr(VGPR(3), 0x00AA0000)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(2),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12AA5678)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(2)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x40), 0x12AA5678)


@dataclasses.dataclass
class StatusFlags:
    """Hardware status flags: SCC (Scalar Condition Code) and VCC (Vector Condition Code)."""
    scc: int = 0
    vcc: int = 0


class TestDSMskorB32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation neutrality."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()
        self.flags = StatusFlags(scc=1, vcc=0xFFFFFFFF_FFFFFFFF)

    def test_condition_codes_scc_vcc_unmodified(self) -> None:
        self.lds.write_dword(0x40, 0xFFFFFFFF)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0xFFFFFFFF)
        self.state.rf.set_vgpr(VGPR(3), 0x00000000)

        inst = DSMskorInstruction(
            opcode=DSOpcode.DS_MSKOR_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vmask=VGPR(2),
            vdata=VGPR(3),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0xFFFFFFFF)
        self.assertEqual(new, 0)
        self.assertEqual(self.flags.scc, 1)
        self.assertEqual(self.flags.vcc, 0xFFFFFFFF_FFFFFFFF)


if __name__ == "__main__":
    unittest.main()
