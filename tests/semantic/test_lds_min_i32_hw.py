"""Reference mathematical oracle and hardware translator tests for ds_min_i32.

RDNA2 / GFX10 Local Data Share (LDS) Signed 32-Bit Atomic Minimum:
Performs an atomic signed 32-bit integer minimum of operand and LDS memory location:
    LDS[address + offset] = min_i32(LDS[address + offset], data)

Syntax:
    ds_min_i32 vaddr, vdata [offset:N]
    ds_min_rtn_i32 vdst, vaddr, vdata [offset:N]

Semantics:
    Effective Address = (vaddr + offset) & 0xFFFFFFFF
    idx = (Effective Address >> 2) & (lds_dwords - 1)
    old_val = LDS[idx]
    LDS[idx] = min_i32(old_val, data)
    if rtn: vdst = old_val

Two's-Complement Signed 32-bit Ordering:
    Values are interpreted as signed integers in [-2^31, 2^31 - 1].
    Bit 31 is the sign bit: negative values compare strictly less than positive values.
    For example:
        -1 (0xFFFFFFFF) < 0 (0x00000000)
        INT32_MIN (0x80000000) < INT32_MAX (0x7FFFFFFF)
    In contrast to ds_min_u32 where 0x00000000 < 0xFFFFFFFF and 1 < 0x80000000.

Translator Mapping:
    Mapped to SPIR-V OpAtomicSMin in the Workgroup storage class.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import random
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
INT32_MIN: int = 0x80000000
INT32_MAX: int = 0x7FFFFFFF

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


def oracle_ds_min_i32(m: int, d: int) -> int:
    """Pure mathematical reference oracle for ds_min_i32.
    
    Signed 32-bit integer minimum: returns the operand with smaller signed value.
    """
    m32 = u32(m)
    d32 = u32(d)
    return m32 if s32(m32) <= s32(d32) else d32


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

    def atomic_min_i32(self, vaddr: int, data: int, offset: int = 0) -> tuple[int, int]:
        """Execute atomic signed minimum on LDS: returns (old_val, new_val)."""
        eff_byte, idx = calc_lds_address(vaddr, offset, self.capacity_dwords)
        old_val = self.memory[idx]
        new_val = oracle_ds_min_i32(old_val, data)
        self.memory[idx] = new_val
        return old_val, new_val


def make_cs_fssm(rsrc2: int = RSRC2_4K, size: tuple[int, int, int] = (32, 1, 1)) -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def translate_compute(lines: list[str], name: str = "cs_ds_min_i32") -> str:
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


class TestLdsMinI32ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and hardware translator execution."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_oracle_basic_min(self) -> None:
        self.assertEqual(oracle_ds_min_i32(100, 30), 30)
        self.assertEqual(oracle_ds_min_i32(30, 100), 30)
        self.assertEqual(oracle_ds_min_i32(-5, -3), u32(-5))
        self.assertEqual(oracle_ds_min_i32(-3, -5), u32(-5))
        self.assertEqual(oracle_ds_min_i32(0, 0), 0)

    def test_oracle_signed_boundary_extremes(self) -> None:
        self.assertEqual(oracle_ds_min_i32(INT32_MIN, INT32_MAX), INT32_MIN)
        self.assertEqual(oracle_ds_min_i32(INT32_MAX, INT32_MIN), INT32_MIN)
        self.assertEqual(oracle_ds_min_i32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        self.assertEqual(oracle_ds_min_i32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(oracle_ds_min_i32(INT32_MIN, 0xFFFFFFFF), INT32_MIN)

    def test_signed_vs_unsigned_divergence(self) -> None:
        self.assertEqual(oracle_ds_min_i32(1, 0x80000000), 0x80000000)
        self.assertEqual(oracle_ds_min_i32(0x80000000, 1), 0x80000000)

    def test_lds_model_execution_with_offset(self) -> None:
        self.lds.write_dword(0x10, 50)
        old_val, new_val = self.lds.atomic_min_i32(vaddr=0x00, data=15, offset=0x10)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 15)
        self.assertEqual(self.lds.read_dword(0x10), 15)

    def test_translator_hardware_execution_non_returning(self) -> None:
        lines = [
            "v_mov_b32_e32 v1, 0x10",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 15",
            "ds_min_i32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicSMin", spv)
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 15)

    def test_translator_hardware_execution_returning(self) -> None:
        lines = [
            "v_mov_b32_e32 v1, 0x20",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, -20",
            "ds_min_rtn_i32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpAtomicSMin", spv)
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 100)
        self.assertEqual(mem.get(4), u32(-20))


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
    DS_MIN_I32 = "ds_min_i32"
    DS_MIN_RTN_I32 = "ds_min_rtn_i32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_min_i32 or ds_min_rtn_i32 instruction."""
    opcode: Opcode
    vaddr: Register
    vdata: Register
    vdst: Register | None = None
    offset: DSOffset = dataclasses.field(default_factory=DSOffset)

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR (count=1): {self.vaddr}")
        if not self.vdata.is_vgpr or self.vdata.count != 1:
            raise ValueError(f"vdata must be a single 32-bit VGPR (count=1): {self.vdata}")
        if self.opcode == Opcode.DS_MIN_RTN_I32:
            if self.vdst is None or not self.vdst.is_vgpr or self.vdst.count != 1:
                raise ValueError("ds_min_rtn_i32 requires single 32-bit VGPR destination vdst")
        elif self.vdst is not None:
            raise ValueError("Non-returning ds_min_i32 cannot have vdst destination")

    def to_assembly(self) -> str:
        off_str = f" offset:{self.offset.offset}" if self.offset.offset != 0 else ""
        if self.opcode == Opcode.DS_MIN_RTN_I32:
            assert self.vdst is not None
            return f"{self.opcode.value} {self.vdst.format_asm()}, {self.vaddr.format_asm()}, {self.vdata.format_asm()}{off_str}"
        return f"{self.opcode.value} {self.vaddr.format_asm()}, {self.vdata.format_asm()}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestLdsMinI32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_formatting(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v5 = Register(RegKind.VGPR, 5)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v5.format_asm(), "v5")

    def test_instruction_formatting_non_returning(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=16),
        )
        self.assertEqual(inst.to_assembly(), "ds_min_i32 v1, v2 offset:16")

    def test_instruction_formatting_returning(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_MIN_RTN_I32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=4),
        )
        self.assertEqual(inst.to_assembly(), "ds_min_rtn_i32 v5, v1, v2 offset:4")

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_MIN_I32,
                vaddr=Register(RegKind.SGPR, 1),
                vdata=Register(RegKind.VGPR, 2),
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_MIN_RTN_I32,
                vdst=None,
                vaddr=Register(RegKind.VGPR, 1),
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
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=8),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_MIN_RTN_I32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 3),
            vdata=Register(RegKind.VGPR, 4),
        ))
        expected = "ds_min_i32 v1, v2 offset:8\nds_min_rtn_i32 v5, v3, v4"
        self.assertEqual(prog.to_assembly(), expected)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LdsMinI32ConditionCodes:
    """Condition codes resulting from atomic signed minimum."""
    zero: bool
    negative: bool
    positive: bool
    operand_clamped: bool


@dataclasses.dataclass(frozen=True)
class LdsMinI32StatusFlags:
    """Execution status and address resolution metadata."""
    effective_byte_address: int
    dword_index: int
    old_value: int
    operand_value: int
    new_value: int


class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_min_i32 and ds_min_rtn_i32."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.lds: LDSModel = LDSModel(capacity_dwords)
        self.last_condition_codes: LdsMinI32ConditionCodes | None = None
        self.last_status_flags: LdsMinI32StatusFlags | None = None
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

    def step(self, inst: Instruction) -> tuple[int, int]:
        """Executes a single ds_min_i32 or ds_min_rtn_i32 instruction: returns (old_val, new_val)."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        vdata = self.read_reg_u32(RegKind.VGPR, inst.vdata.index)
        eff_byte, idx = calc_lds_address(vaddr, inst.offset.offset, self.lds.capacity_dwords)
        old_val, new_val = self.lds.atomic_min_i32(
            vaddr=vaddr,
            data=vdata,
            offset=inst.offset.offset,
        )
        if inst.opcode == Opcode.DS_MIN_RTN_I32:
            assert inst.vdst is not None
            self.write_reg_u32(RegKind.VGPR, inst.vdst.index, old_val)

        is_neg = bool(new_val & SIGN_BIT32)
        self.last_condition_codes = LdsMinI32ConditionCodes(
            zero=(new_val == 0),
            negative=is_neg,
            positive=(new_val > 0 and not is_neg),
            operand_clamped=(new_val != old_val),
        )
        self.last_status_flags = LdsMinI32StatusFlags(
            effective_byte_address=eff_byte,
            dword_index=idx,
            old_value=old_val,
            operand_value=vdata,
            new_value=new_val,
        )
        self.trace.append(f"{inst.to_assembly()} => old={old_val:#010x}, new={new_val:#010x}")
        return old_val, new_val

    def run_program(self, prog: Program) -> list[tuple[int, int]]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestLdsMinI32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_scalar_execution_non_returning(self) -> None:
        self.interp.lds.write_dword(0x10, 80)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 25)

        inst = Instruction(
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        old_val, new_val = self.interp.step(inst)
        self.assertEqual(old_val, 80)
        self.assertEqual(new_val, 25)
        self.assertEqual(self.interp.lds.read_dword(0x10), 25)
        # Verify operands are preserved (non-returning)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 0x10)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 2), 25)

    def test_scalar_execution_returning(self) -> None:
        self.interp.lds.write_dword(0x20, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, -10)
        self.interp.write_reg_u32(RegKind.VGPR, 5, 0xDEADBEEF)

        inst = Instruction(
            opcode=Opcode.DS_MIN_RTN_I32,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        old_val, new_val = self.interp.step(inst)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, u32(-10))
        # vdst receives old_val
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 5), 50)
        self.assertEqual(self.interp.lds.read_dword(0x20), u32(-10))

    def test_scalar_chained_reduction(self) -> None:
        self.interp.lds.write_dword(0x30, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x30)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 70)
        self.interp.write_reg_u32(RegKind.VGPR, 3, 20)
        self.interp.write_reg_u32(RegKind.VGPR, 4, -5)

        prog = Program()
        prog.append(Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2)))
        prog.append(Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 3)))
        prog.append(Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 4)))

        results = self.interp.run_program(prog)
        self.assertEqual(results, [(100, 70), (70, 20), (20, u32(-5))])
        self.assertEqual(self.interp.lds.read_dword(0x30), u32(-5))


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

    def step(self, inst: Instruction) -> list[tuple[int, int] | None]:
        """Executes a ds_min_i32 instruction across all lanes in the wavefront."""
        lane_vaddrs: list[int] = []
        lane_vdatas: list[int] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            vd = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vdata.index)
            lane_vaddrs.append(va)
            lane_vdatas.append(vd)

        lane_results: list[tuple[int, int] | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            old_val, new_val = self.lds.atomic_min_i32(
                vaddr=lane_vaddrs[lane],
                data=lane_vdatas[lane],
                offset=inst.offset.offset,
            )
            if inst.opcode == Opcode.DS_MIN_RTN_I32:
                assert inst.vdst is not None
                self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdst.index, old_val)
            lane_results.append((old_val, new_val))

        return lane_results


class TestLdsMinI32SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_lanes(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        for lane in range(32):
            simd.lds.write_dword(lane * 4, 100)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, lane)

        inst = Instruction(
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(results[lane], (100, lane))
            self.assertEqual(simd.lds.read_dword(lane * 4), lane)

    def test_wave64_exec_masking(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, capacity_dwords=1024)
        simd.set_exec_mask(0x00000000FFFFFFFF)  # Only lower 32 lanes active
        for lane in range(64):
            simd.lds.write_dword(lane * 4, 50)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 10)

        inst = Instruction(
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        results = simd.step(inst)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], (50, 10))
                self.assertEqual(simd.lds.read_dword(lane * 4), 10)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.lds.read_dword(lane * 4), 50)

    def test_translator_simd_multi_lane_kernel(self) -> None:
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # v1 = lane * 4
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_min_i32 v1, v0",  # min(50, lane) = lane
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v30, 2, v0",  # v30 = lane * 4
            "buffer_store_dword v3, v30, s[0:3], 0 offen",
        ]
        mem = run_compute_kernel(lines, lanes=16)
        for lane in range(16):
            self.assertEqual(mem.get(lane * 4), lane)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestLdsMinI32RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and operand preservation contracts."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_vaddr_equals_vdata_aliasing(self) -> None:
        # Address 0x10 (16) stores value 50. Instruction ds_min_i32 v1, v1 where v1=0x10 (16).
        # LDS[16] = min(50, 16) = 16.
        self.interp.lds.write_dword(0x10, 50)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)

        inst = Instruction(
            opcode=Opcode.DS_MIN_I32,
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 1),
        )
        old_val, new_val = self.interp.step(inst)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 0x10)
        self.assertEqual(self.interp.lds.read_dword(0x10), 0x10)
        # v1 remains preserved as 0x10
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 0x10)

    def test_vdst_equals_vaddr_aliasing(self) -> None:
        # In ds_min_rtn_i32 v1, v1, v2: v1 is both address and return destination.
        # Address is 0x20, LDS has 100, v2 is 25.
        # Address must be evaluated before vdst overwrite: old_val (100) written into v1.
        self.interp.lds.write_dword(0x20, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 25)

        inst = Instruction(
            opcode=Opcode.DS_MIN_RTN_I32,
            vdst=Register(RegKind.VGPR, 1),
            vaddr=Register(RegKind.VGPR, 1),
            vdata=Register(RegKind.VGPR, 2),
        )
        old_val, new_val = self.interp.step(inst)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_val, 25)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 1), 100)
        self.assertEqual(self.interp.lds.read_dword(0x20), 25)

    def test_translator_aliased_kernel(self) -> None:
        # Translates kernel with vaddr == vdata
        lines = [
            "v_mov_b32_e32 v1, 0x10",  # v1 = 16
            "v_mov_b32_e32 v2, 40",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_min_i32 v1, v1",  # min(40, 16) = 16
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 16)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and VCC Isolation
# ---------------------------------------------------------------------------

class TestLdsMinI32ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and VCC isolation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_zero_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 5)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 0)
        inst = Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        assert cc is not None
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)
        self.assertTrue(cc.operand_clamped)

    def test_negative_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 10)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, -50)
        inst = Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.positive)

    def test_positive_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 100)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 45)
        inst = Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2))
        self.interp.step(inst)
        cc = self.interp.last_condition_codes
        assert cc is not None
        self.assertFalse(cc.zero)
        self.assertFalse(cc.negative)
        self.assertTrue(cc.positive)

    def test_status_flags_metadata(self) -> None:
        self.interp.lds.write_dword(0x28, 90)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        self.interp.write_reg_u32(RegKind.VGPR, 2, 30)
        inst = Instruction(opcode=Opcode.DS_MIN_I32, vaddr=Register(RegKind.VGPR, 1), vdata=Register(RegKind.VGPR, 2), offset=DSOffset(offset=8))
        self.interp.step(inst)
        flags = self.interp.last_status_flags
        assert flags is not None
        self.assertEqual(flags.effective_byte_address, 0x28)
        self.assertEqual(flags.dword_index, 10)
        self.assertEqual(flags.old_value, 90)
        self.assertEqual(flags.operand_value, 30)
        self.assertEqual(flags.new_value, 30)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

class TestLdsMinI32IEEE754AndBoundaries(unittest.TestCase):
    """Unit tests for IEEE 754 bit-patterns and boundary two's-complement extremes."""

    def test_ieee754_signed_zeros(self) -> None:
        # +0.0 is 0x00000000 (signed value 0)
        # -0.0 is 0x80000000 (signed value -2147483648 = INT32_MIN)
        # In signed integer comparison, -0.0 bit pattern compares strictly less than +0.0!
        res = oracle_ds_min_i32(0x00000000, 0x80000000)
        self.assertEqual(res, 0x80000000)
        res_rev = oracle_ds_min_i32(0x80000000, 0x00000000)
        self.assertEqual(res_rev, 0x80000000)

    def test_ieee754_nan_and_infinity_patterns(self) -> None:
        # +Infinity is 0x7F800000 (signed value 2139095040)
        # -Infinity is 0xFF800000 (signed value -8388608)
        # -Infinity < +Infinity in signed two's complement
        self.assertEqual(oracle_ds_min_i32(0x7F800000, 0xFF800000), 0xFF800000)
        # NaN 0x7FC00000 vs 0: 0 < 0x7FC00000
        self.assertEqual(oracle_ds_min_i32(0x7FC00000, 0), 0)

    def test_signed_two_complement_extremes(self) -> None:
        table = [
            (INT32_MIN, INT32_MAX, INT32_MIN),
            (INT32_MIN, 0, INT32_MIN),
            (INT32_MIN, 0xFFFFFFFF, INT32_MIN),
            (0xFFFFFFFF, 0, 0xFFFFFFFF),  # -1 < 0
            (0xFFFFFFFF, 1, 0xFFFFFFFF),  # -1 < 1
            (u32(-10), u32(-5), u32(-10)),
            (u32(-5), u32(-10), u32(-10)),
        ]
        for m, d, expected in table:
            with self.subTest(m=hex(m), d=hex(d)):
                self.assertEqual(oracle_ds_min_i32(m, d), expected)

    def test_differential_signed_vs_unsigned(self) -> None:
        # When operands have different signs (bit 31 differs), signed and unsigned min diverge
        cases = [
            (1, 0x80000000, 0x80000000),  # signed min is 0x80000000 (unsigned min is 1)
            (10, 0xFFFFFFFF, 0xFFFFFFFF), # signed min is -1 (unsigned min is 10)
            (0x7FFFFFFF, 0x80000000, 0x80000000),
        ]
        for m, d, expected_signed in cases:
            self.assertEqual(oracle_ds_min_i32(m, d), expected_signed)


# ---------------------------------------------------------------------------
# Negative Controls, Syntax Rejection, and Mutation Detection
# ---------------------------------------------------------------------------

class TestLdsMinI32NegativeAndSyntax(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)

    def test_invalid_offset_ranges(self) -> None:
        with self.assertRaises(ValueError):
            DSOffset(offset=-1)
        with self.assertRaises(ValueError):
            DSOffset(offset=65536)

    def test_instruction_syntax_rejection(self) -> None:
        # Non-returning ds_min_i32 cannot have vdst
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_MIN_I32,
                vdst=Register(RegKind.VGPR, 5),
                vaddr=Register(RegKind.VGPR, 1),
                vdata=Register(RegKind.VGPR, 2),
            )
        # Returning ds_min_rtn_i32 requires vdst
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_MIN_RTN_I32,
                vdst=None,
                vaddr=Register(RegKind.VGPR, 1),
                vdata=Register(RegKind.VGPR, 2),
            )

    def test_mutation_detection_oracle(self) -> None:
        # Mutant: using unsigned min instead of signed min
        def mutant_unsigned_min(m: int, d: int) -> int:
            return min(m & MASK32, d & MASK32)

        # 0x80000000 vs 1: signed min is 0x80000000, unsigned min is 1
        m, d = 0x80000000, 1
        self.assertNotEqual(oracle_ds_min_i32(m, d), mutant_unsigned_min(m, d))

        # Mutant: using max instead of min
        def mutant_max_i32(m: int, d: int) -> int:
            return m if s32(m) >= s32(d) else d
        self.assertNotEqual(oracle_ds_min_i32(10, 20), mutant_max_i32(10, 20))


if __name__ == "__main__":
    unittest.main()
