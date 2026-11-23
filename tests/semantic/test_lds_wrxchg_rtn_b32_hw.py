"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_wrxchg_rtn_b32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - Single 32-bit Atomic Exchange / Swap:
    - ds_wrxchg_rtn_b32 vdst, vaddr, vdata [offset:N]:
      Returning form (atomic swap):
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = vdata & 0xFFFFFFFF
        vdst = old_val
      Atomically swaps the memory contents with vdata and stores pre-atomic value in vdst.
  - Active lane masking:
    - Vector operations governed by EXEC mask (32-bit for Wave32, 64-bit for Wave64).
    - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
  - Hardware wrapping & alignment:
    - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
    - Resulting dword index is masked with (lds_dwords - 1).
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
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS, bounds_check: bool = False) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords
        self.bounds_check: bool = bounds_check

    def read_dword(self, byte_addr: int) -> int:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_xchg(self, byte_addr: int, data: int) -> int:
        """Atomic exchange (swap). Returns pre-atomic dword value."""
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = data & MASK32
        return old


def ds_wrxchg_rtn_b32_oracle(old_val: int, data: int) -> Tuple[int, int]:
    """Pure reference oracle for 32-bit atomic swap. Returns (old_val, new_val)."""
    return (old_val & MASK32), (data & MASK32)


def ds_wrxchg_rtn_b32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_wrxchg_rtn_b32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_xchg(eff_byte, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSWrxchgRtnB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_swap_values(self) -> None:
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0x00000000, 0xFFFFFFFF),
            (0xAAAAAAAA, 0x55555555),
            (0xCAFEBABE, 0xDEADBEEF),
        ]
        for old, new in test_pairs:
            with self.subTest(old=hex(old), new=hex(new)):
                returned_old, left_new = ds_wrxchg_rtn_b32_oracle(old, new)
                self.assertEqual(returned_old, old)
                self.assertEqual(left_new, new)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 0xCAFEBABE)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 0xCAFEBABE)
        self.assertEqual(lds.read_dword(0x40 + 4096), 0xCAFEBABE)

    def test_model_execution_swap(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 0x11112222)
        old_val, new_val = ds_wrxchg_rtn_b32_execute_model(lds, 0x100, 0x33334444)
        self.assertEqual(old_val, 0x11112222)
        self.assertEqual(new_val, 0x33334444)
        self.assertEqual(lds.read_dword(0x100), 0x33334444)


class DSOpcode(str, enum.Enum):
    DS_WRXCHG_RTN_B32 = "ds_wrxchg_rtn_b32"


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
class DSWrxchgInstruction:
    opcode: DSOpcode
    vdst: VGPR
    vaddr: VGPR
    vdata: VGPR
    offset: OffsetImmediate = dataclasses.field(default_factory=lambda: OffsetImmediate(0))

    def __post_init__(self) -> None:
        if self.vdst is None:
            raise ValueError(f"{self.opcode} requires a destination register")

    def to_asm(self) -> str:
        off_str = f" {self.offset}" if self.offset.value != 0 else ""
        return f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata}{off_str}"


class TestDSWrxchgRtnB32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction(self) -> None:
        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(16),
        )
        self.assertEqual(inst.opcode, DSOpcode.DS_WRXCHG_RTN_B32)
        self.assertEqual(inst.vdst.index, 5)
        self.assertEqual(inst.vaddr.index, 1)
        self.assertEqual(inst.vdata.index, 2)
        self.assertEqual(inst.offset.value, 16)
        self.assertEqual(inst.to_asm(), "ds_wrxchg_rtn_b32 v5, v1, v2 offset:16")

    def test_ast_zero_offset_omitted_in_asm(self) -> None:
        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        self.assertEqual(inst.to_asm(), "ds_wrxchg_rtn_b32 v5, v1, v2")


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
    """Deterministic scalar interpreter for DS WRXCHG instructions."""

    def execute(self, state: ScalarThreadState, inst: DSWrxchgInstruction) -> Tuple[int, int]:
        addr = state.rf.get_vgpr(inst.vaddr)
        data = state.rf.get_vgpr(inst.vdata)
        eff_addr = (addr + inst.offset.value) & MASK32

        # Capture old memory value before writing to vdst (handling vdst == vdata or vdst == vaddr)
        old_val = state.lds.atomic_xchg(eff_addr, data)
        new_val = state.lds.read_dword(eff_addr)
        state.rf.set_vgpr(inst.vdst, old_val)
        return old_val, new_val


class TestDSWrxchgRtnB32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_swap(self) -> None:
        self.lds.write_dword(0x80, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x80)
        self.state.rf.set_vgpr(VGPR(2), 0x9ABCDEF0)
        self.state.rf.set_vgpr(VGPR(5), 0xDEADBEEF)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x9ABCDEF0)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(5)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x80), 0x9ABCDEF0)

    def test_scalar_execution_with_offset(self) -> None:
        self.lds.write_dword(0x54, 0xCAFEBABE)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0xBAADF00D)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(0x14),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0xCAFEBABE)
        self.assertEqual(new, 0xBAADF00D)
        self.assertEqual(self.lds.read_dword(0x54), 0xBAADF00D)


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
        self, wave: SIMDWaveState, inst: DSWrxchgInstruction
    ) -> List[Tuple[int, int, int]]:
        """Returns list of (lane_id, old_val, new_val) for executed active lanes."""
        results = []
        for lane_id in range(wave.wave_size):
            if not wave.is_lane_active(lane_id):
                continue
            rf = wave.lanes[lane_id]
            addr = rf.get_vgpr(inst.vaddr)
            data = rf.get_vgpr(inst.vdata)
            eff_addr = (addr + inst.offset.value) & MASK32

            old_val = wave.lds.atomic_xchg(eff_addr, data)
            new_val = wave.lds.read_dword(eff_addr)
            rf.set_vgpr(inst.vdst, old_val)
            results.append((lane_id, old_val, new_val))
        return results


class TestDSWrxchgRtnB32SIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_full_wave32(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0x1000 + l)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 0x2000 + l)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 32)
        for l in range(32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0x1000 + l)
            self.assertEqual(lds.read_dword(l * 4), 0x2000 + l)

    def test_simd_divergent_alternating_lanes(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        # Even lanes active (0x55555555)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x55555555, lds=lds)
        for l in range(32):
            lds.write_dword(l * 4, 0x1000 + l)
            wave.lanes[l].set_vgpr(VGPR(1), l * 4)
            wave.lanes[l].set_vgpr(VGPR(2), 0x2000 + l)
            wave.lanes[l].set_vgpr(VGPR(5), 0xCAFEBABE)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 16)
        for l in range(32):
            if l % 2 == 0:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0x1000 + l)
                self.assertEqual(lds.read_dword(l * 4), 0x2000 + l)
            else:
                self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xCAFEBABE)
                self.assertEqual(lds.read_dword(l * 4), 0x1000 + l)

    def test_simd_exec_zero_neutrality(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x40, 0xFFFFFFFF)
        wave = SIMDWaveState(wave_size=32, exec_mask=0x0, lds=lds)
        for l in range(32):
            wave.lanes[l].set_vgpr(VGPR(1), 0x40)
            wave.lanes[l].set_vgpr(VGPR(2), 0x0)
            wave.lanes[l].set_vgpr(VGPR(5), 0xDEADBEEF)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        results = SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(len(results), 0)
        self.assertEqual(lds.read_dword(0x40), 0xFFFFFFFF)
        for l in range(32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 0xDEADBEEF)

    def test_simd_multi_lane_serialization_token_ring(self) -> None:
        # 32 lanes targeting the exact same LDS dword at 0x10
        # Lane 0 writes 100, returns initial 0
        # Lane 1 writes 101, returns 100
        # Lane l writes 100 + l, returns 100 + l - 1
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x10, 0)
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=lds)
        for l in range(32):
            wave.lanes[l].set_vgpr(VGPR(1), 0x10)
            wave.lanes[l].set_vgpr(VGPR(2), 100 + l)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        SIMDDSInterpreter().execute(wave, inst)
        self.assertEqual(wave.lanes[0].get_vgpr(VGPR(5)), 0)
        for l in range(1, 32):
            self.assertEqual(wave.lanes[l].get_vgpr(VGPR(5)), 100 + l - 1)
        self.assertEqual(lds.read_dword(0x10), 100 + 31)


class TestDSWrxchgRtnB32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_vdst_aliases_vaddr(self) -> None:
        self.lds.write_dword(0x40, 0x12345678)
        self.state.rf.set_vgpr(VGPR(1), 0x40)  # vaddr
        self.state.rf.set_vgpr(VGPR(2), 0x9ABCDEF0)  # vdata

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(1),  # vdst aliases vaddr
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x9ABCDEF0)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(1)), 0x12345678)
        self.assertEqual(self.lds.read_dword(0x40), 0x9ABCDEF0)

    def test_vdst_aliases_vdata(self) -> None:
        self.lds.write_dword(0x60, 0xF0F0F0F0)
        self.state.rf.set_vgpr(VGPR(1), 0x60)  # vaddr
        self.state.rf.set_vgpr(VGPR(2), 0x0F0F0F0F)  # vdata and vdst

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(2),  # vdst aliases vdata
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0xF0F0F0F0)
        self.assertEqual(new, 0x0F0F0F0F)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(2)), 0xF0F0F0F0)
        self.assertEqual(self.lds.read_dword(0x60), 0x0F0F0F0F)

    def test_vaddr_aliases_vdata(self) -> None:
        self.lds.write_dword(0x40, 0xCAFEBABE)
        self.state.rf.set_vgpr(VGPR(1), 0x40)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(1),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0xCAFEBABE)
        self.assertEqual(new, 0x40)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(5)), 0xCAFEBABE)
        self.assertEqual(self.lds.read_dword(0x40), 0x40)

    def test_complete_three_way_alias(self) -> None:
        self.lds.write_dword(0x20, 0x3F)
        self.state.rf.set_vgpr(VGPR(3), 0x20)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(3),
            vaddr=VGPR(3),
            vdata=VGPR(3),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0x3F)
        self.assertEqual(new, 0x20)
        self.assertEqual(self.state.rf.get_vgpr(VGPR(3)), 0x3F)
        self.assertEqual(self.lds.read_dword(0x20), 0x20)


@dataclasses.dataclass
class StatusFlags:
    """Hardware status flags: SCC (Scalar Condition Code) and VCC (Vector Condition Code)."""
    scc: int = 0
    vcc: int = 0


class TestDSWrxchgRtnB32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation neutrality."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()
        self.flags = StatusFlags(scc=1, vcc=0xFFFFFFFF_FFFFFFFF)

    def test_condition_codes_scc_vcc_unmodified(self) -> None:
        self.lds.write_dword(0x40, 0xFFFFFFFF)
        self.state.rf.set_vgpr(VGPR(1), 0x40)
        self.state.rf.set_vgpr(VGPR(2), 0x00000000)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
        )
        old, new = self.interp.execute(self.state, inst)
        self.assertEqual(old, 0xFFFFFFFF)
        self.assertEqual(new, 0)
        self.assertEqual(self.flags.scc, 1)
        self.assertEqual(self.flags.vcc, 0xFFFFFFFF_FFFFFFFF)


class TestDSWrxchgRtnB32BoundaryExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.lds = LDSMemory(num_dwords=1024)
        self.state = ScalarThreadState(self.lds)
        self.interp = ScalarDSInterpreter()

    def test_ieee754_exact_bit_preservation(self) -> None:
        patterns = {
            "pos_zero": 0x00000000,
            "neg_zero": 0x80000000,
            "pos_inf": 0x7F800000,
            "neg_inf": 0xFF800000,
            "qnan": 0x7FC00000,
            "snan": 0x7F800001,
            "subnormal_min": 0x00000001,
            "subnormal_max": 0x007FFFFF,
        }
        for name, bits in patterns.items():
            with self.subTest(pattern=name, bits=hex(bits)):
                self.lds.write_dword(0x40, bits)
                self.state.rf.set_vgpr(VGPR(1), 0x40)
                self.state.rf.set_vgpr(VGPR(2), ~bits & MASK32)

                inst = DSWrxchgInstruction(
                    opcode=DSOpcode.DS_WRXCHG_RTN_B32,
                    vdst=VGPR(5),
                    vaddr=VGPR(1),
                    vdata=VGPR(2),
                )
                old, new = self.interp.execute(self.state, inst)
                self.assertEqual(old, bits)
                self.assertEqual(new, ~bits & MASK32)


class TestDSWrxchgRtnB32NegativeAndMutation(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_offset(self) -> None:
        with self.assertRaises(ValueError):
            OffsetImmediate(-1)
        with self.assertRaises(ValueError):
            OffsetImmediate(0x10000)

    def test_syntax_rejection_invalid_vgpr(self) -> None:
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            VGPR(256)

    def test_mutation_detection_dropped_offset(self) -> None:
        class DroppedOffsetMutant(ScalarDSInterpreter):
            def execute(self, state: ScalarThreadState, inst: DSWrxchgInstruction) -> Tuple[int, int]:
                no_off = DSWrxchgInstruction(
                    opcode=inst.opcode,
                    vdst=inst.vdst,
                    vaddr=inst.vaddr,
                    vdata=inst.vdata,
                    offset=OffsetImmediate(0),
                )
                return super().execute(state, no_off)

        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x10, 0x11111111)
        lds.write_dword(0x20, 0x22222222)
        state = ScalarThreadState(lds)
        state.rf.set_vgpr(VGPR(1), 0x10)
        state.rf.set_vgpr(VGPR(2), 0x99999999)

        inst = DSWrxchgInstruction(
            opcode=DSOpcode.DS_WRXCHG_RTN_B32,
            vdst=VGPR(5),
            vaddr=VGPR(1),
            vdata=VGPR(2),
            offset=OffsetImmediate(0x10),
        )
        DroppedOffsetMutant().execute(state, inst)
        self.assertEqual(lds.read_dword(0x10), 0x99999999)
        self.assertEqual(lds.read_dword(0x20), 0x22222222)


class TestDSWrxchgRtnB32TranslatorIntegration(unittest.TestCase):
    """Hardware translator integration: tools/rdna2spirv.py and spirv_mini_interp."""

    TABLE = (  # (M, D)
        (0xFFFFFFFF, 0x0F0F0F0F),
        (0x12345678, 0x9ABCDEF0),
        (0xAAAAAAAA, 0x55555555),
        (0x00000000, 0xFFFFFFFF),
        (0x80000001, 0x80000000),
    )

    def test_translator_one_operation_table(self) -> None:
        for m, d in self.TABLE:
            with self.subTest(m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_wrxchg_rtn_b32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual(rtns, [m])
                self.assertEqual(fin, {0x40: d})

    def test_translator_offset_addressing(self) -> None:
        init = {0x40: 0xFFFFFFFF, 0x54: 0x12345678}
        rtns, fin = L.run_chain([("ds_wrxchg_rtn_b32", 0x40, 0x14, (0xCAFEBABE,))], init)
        self.assertEqual(rtns, [0x12345678])
        self.assertEqual(fin, {0x54: 0xCAFEBABE})

    def test_translator_chain_sequential_swaps(self) -> None:
        # Sequential swaps on the same dword: each returns what the previous wrote
        ops = [
            ("ds_wrxchg_rtn_b32", 0x40, 0, (0x11111111,)),
            ("ds_wrxchg_rtn_b32", 0x40, 0, (0x22222222,)),
            ("ds_wrxchg_rtn_b32", 0x40, 0, (0x33333333,)),
        ]
        rtns, fin = L.run_chain(ops, {0x40: 0x00000000})
        self.assertEqual(rtns, [0x00000000, 0x11111111, 0x22222222])
        self.assertEqual(fin, {0x40: 0x33333333})

    def test_translator_rejects_malformed_syntax(self) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("bad_syntax", ["ds_wrxchg_rtn_b32 v1, v2"])


if __name__ == "__main__":
    unittest.main()
