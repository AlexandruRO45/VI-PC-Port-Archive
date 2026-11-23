"""Reference mathematical oracle and hardware translator tests for ds_sub_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Subtraction:
Performs an atomic 32-bit subtraction of operand from LDS memory location:
    LDS[address + offset] = (LDS[address + offset] - data) mod 2^32

Syntax:
    ds_sub_u32 vaddr, vdata [offset:N]

Semantics:
    Effective Address = (vaddr + offset) & 0xFFFFFFFF
    idx = (Effective Address >> 2) & (lds_dwords - 1)
    old_val = LDS[idx]
    LDS[idx] = (old_val - data) & 0xFFFFFFFF

Non-returning:
    Does not write back to vector general-purpose registers (VGPRs).

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
from typing import Any
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

# Architectural Constants for RDNA2 Local Data Share (LDS)
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK32: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
SIGN_BIT32: int = 0x80000000

# Compute Shader RSRC2 metadata: 4 user SGPRs, workgroup id in s4, LDS_SIZE = 4 KiB
RSRC2_4K: int = 0x40088
VSHARP: list[int] = [0, 0, 0x1000, 0x31016FAC]


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def calc_lds_address(
    vaddr: int,
    offset: int = 0,
    capacity_dwords: int = LDS_SIZE_DWORDS,
) -> tuple[int, int]:
    """Calculate effective byte address and dword index for LDS single-address operand."""
    eff_byte = (vaddr + (offset & OFFSET16_MASK)) & MASK32
    idx = (eff_byte >> 2) & (capacity_dwords - 1)
    return eff_byte, idx


def oracle_ds_sub_u32(orig_val: int, sub_val: int) -> int:
    """Pure mathematical reference oracle for ds_sub_u32.
    
    Subtraction: res = (orig_val - sub_val) mod 2^32.
    """
    orig = orig_val & MASK32
    sub = sub_val & MASK32
    return (orig - sub) & MASK32


class LDSModel:
    """Architectural memory model representing the RDNA2 Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords: int = capacity_dwords
        self.memory: list[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        self.memory[idx] = val & MASK32

    def atomic_sub_u32(self, vaddr: int, sub_val: int, offset: int = 0) -> int:
        """Execute atomic subtraction on LDS: LDS[addr] = (LDS[addr] - sub_val) mod 2^32."""
        eff_byte, idx = calc_lds_address(vaddr, offset, self.capacity_dwords)
        old_val = self.memory[idx]
        new_val = oracle_ds_sub_u32(old_val, sub_val)
        self.memory[idx] = new_val
        return new_val


def make_cs_fssm(rsrc2: int = RSRC2_4K, size: tuple[int, int, int] = (32, 1, 1)) -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def translate_compute(lines: list[str], name: str = "cs_ds_sub_u32") -> str:
    body = list(lines) + ["s_endpgm"]
    return support.translate_lines(name, body, fssm=make_cs_fssm())


def run_compute_kernel(
    lines: list[str],
    lanes: int = 32,
    inputs_for: Any | None = None,
    memory: dict[int, int] | None = None,
) -> dict[int, int]:
    spv = translate_compute(lines)
    if memory is None:
        memory = {}
    if inputs_for is None:
        inputs_for = lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i}
    return si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=inputs_for,
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )




# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representation."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2, 4):
            raise ValueError(f"Unsupported register count: {self.count}")
        if self.index + self.count > 256:
            raise ValueError(f"Register range exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count > 1:
            return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class DSOffset:
    """Immediate byte offset modifier for DS instructions."""
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0 or self.offset > 65535:
            raise ValueError(f"offset must be in unsigned 16-bit range [0, 65535]: {self.offset}")


class Opcode(enum.Enum):
    """LDS instruction opcodes."""
    DS_SUB_U32 = "ds_sub_u32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_sub_u32 instruction."""
    opcode: Opcode
    vaddr: Register
    vdata: Register
    offset: DSOffset = dataclasses.field(default_factory=DSOffset)

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR (count=1): {self.vaddr}")
        if not self.vdata.is_vgpr or self.vdata.count != 1:
            raise ValueError(f"vdata must be a single 32-bit VGPR (count=1): {self.vdata}")

    def to_assembly(self) -> str:
        va_str = self.vaddr.format_asm()
        vd_str = self.vdata.format_asm()
        off_str = f" offset:{self.offset.offset}" if self.offset.offset != 0 else ""
        return f"{self.opcode.value} {va_str}, {vd_str}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)



# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_sub_u32."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.lds: LDSModel = LDSModel(capacity_dwords)
        self.vcc: int = 0
        self.last_condition_codes: Any | None = None
        self.last_status_flags: Any | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= len(self.vgprs)) or (kind == RegKind.SGPR and idx >= len(self.sgprs)):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def step(self, inst: Instruction) -> int:
        """Executes a single ds_sub_u32 instruction."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        res = self.lds.atomic_sub_u32(
            vaddr=vaddr,
            sub_val=vdata,
            offset=inst.offset.offset,
        )
        self.trace.append(f"{inst.to_assembly()} => new_val={res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]



# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDLDSWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for LDS atomics."""

    def __init__(self, wave_size: int = 64, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106
        self.lds: LDSModel = LDSModel(capacity_dwords)

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def step(self, inst: Instruction) -> list[int | None]:
        """Executes a ds_sub_u32 instruction across all lanes in the wavefront."""
        lane_vaddrs: list[int] = []
        lane_vdatas: list[int] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            vd = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            lane_vaddrs.append(va)
            lane_vdatas.append(vd)

        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            res = self.lds.atomic_sub_u32(
                vaddr=lane_vaddrs[lane],
                sub_val=lane_vdatas[lane],
                offset=inst.offset.offset,
            )
            lane_results.append(res)

        return lane_results


class TestLdsSubU32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and hardware translator execution."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_oracle_basic_subtraction(self) -> None:
        # 100 - 30 = 70
        res = oracle_ds_sub_u32(orig_val=100, sub_val=30)
        self.assertEqual(res, 70)

    def test_oracle_underflow_wrapping(self) -> None:
        # 30 - 100 = -70 mod 2^32 = 0xFFFFFFBA
        res = oracle_ds_sub_u32(orig_val=30, sub_val=100)
        self.assertEqual(res, (30 - 100) & MASK32)
        self.assertEqual(res, 0xFFFFFFBA)

    def test_oracle_zero_identity(self) -> None:
        for val in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            self.assertEqual(oracle_ds_sub_u32(val, 0), val)
            self.assertEqual(oracle_ds_sub_u32(val, val), 0)

    def test_lds_model_execution_with_offset(self) -> None:
        # Seed address 0x10 with 50
        self.lds.write_dword(0x10, 50)
        # vaddr = 0x00, offset = 0x10 (byte 16 = dword 4), sub_val = 15
        res = self.lds.atomic_sub_u32(vaddr=0x00, sub_val=15, offset=0x10)
        self.assertEqual(res, 35)
        self.assertEqual(self.lds.read_dword(0x10), 35)

    def test_translator_hardware_execution(self) -> None:
        # Translates a compute kernel with ds_sub_u32 and executes on spirv_mini_interp
        lines = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 15",
            "ds_sub_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicISub", spv)
        mem = run_compute_kernel(lines, lanes=1)
        # Verify result in output buffer is 50 - 15 = 35
        self.assertEqual(mem.get(0), 35)



class TestLdsSubU32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1 = Register(RegKind.VGPR, 1)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")

    def test_instruction_formatting(self) -> None:
        inst_no_off = Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        self.assertEqual(inst_no_off.to_assembly(), "ds_sub_u32 v1, v2")

        inst_with_off = Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=16),
        )
        self.assertEqual(inst_with_off.to_assembly(), "ds_sub_u32 v1, v2 offset:16")

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_SUB_U32,
                vaddr=Register(RegKind.SGPR, 1),
                vdata=Register(RegKind.VGPR, 2),
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_SUB_U32,
                vaddr=Register(RegKind.VGPR, 1, count=2),
                vdata=Register(RegKind.VGPR, 2),
            )

    def test_offset_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSOffset(offset=-1)
        with self.assertRaises(ValueError):
            DSOffset(offset=65536)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=4),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 3),
            vdata=Register(RegKind.VGPR, 4),
        ))
        expected = "ds_sub_u32 v1, v2 offset:4\nds_sub_u32 v3, v4"
        self.assertEqual(prog.to_assembly(), expected)



class TestLdsSubU32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_scalar_execution_basic(self) -> None:
        # Seed address 0x10 with 80
        self.interp.lds.write_dword(0x10, 80)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 25)

        inst = Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 55)
        self.assertEqual(self.interp.lds.read_dword(0x10), 55)
        # Verify operands are preserved (non-returning)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 0x10)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 2), 25)

    def test_scalar_chained_execution(self) -> None:
        self.interp.lds.write_dword(0x20, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 20)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 30)

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 3),
        ))

        results = self.interp.run_program(prog)
        self.assertEqual(results, [80, 50])
        self.assertEqual(self.interp.lds.read_dword(0x20), 50)



class TestLdsSubU32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        for lane in range(32):
            simd.lds.write_dword(lane * 4, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane)

        inst = Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], 100 - lane)
            self.assertEqual(simd.lds.read_dword(lane * 4), 100 - lane)

    def test_wave64_exec_masking_partial(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, capacity_dwords=1024)
        simd.set_exec_mask(0x00000000FFFFFFFF)  # Only lower 32 lanes active
        for lane in range(64):
            simd.lds.write_dword(lane * 4, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 10)

        inst = Instruction(
            opcode=Opcode.DS_SUB_U32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 40)
                self.assertEqual(simd.lds.read_dword(lane * 4), 40)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.lds.read_dword(lane * 4), 50)

    def test_translator_simd_divergent_offsets(self) -> None:
        # Translates a multi-lane compute shader where each lane subtracts lane_id from its own dword
        # Prologue computes lane * 4 in v1, writes 100, then subtracts v0 (lane_id)
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # v1 = lane * 4
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_sub_u32 v1, v0",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v1, s[0:3], 0 offen",
        ]
        mem = run_compute_kernel(lines, lanes=8)
        for lane in range(8):
            self.assertEqual(mem.get(lane * 4), 100 - lane)


if __name__ == "__main__":
    unittest.main()
