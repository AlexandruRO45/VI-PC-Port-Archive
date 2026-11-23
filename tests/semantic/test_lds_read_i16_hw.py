"""Reference mathematical oracle and hardware translator tests for ds_read_i16.

RDNA2 / GFX10 Local Data Share (LDS) 16-Bit Signed Memory Read:
Reads a 16-bit signed integer from LDS memory and sign-extends it to 32 bits into vdst:
    halfword = LDS_read_u16(address + offset)
    vdst = sign_extend_16_to_32(halfword)

Syntax:
    ds_read_i16 vdst, vaddr [offset:N]

Semantics:
    Effective Address = (vaddr + offset) & 0xFFFFFFFF
    word0 = LDS[(Effective Address) >> 2]
    shift = (Effective Address & 3) * 8
    if (Effective Address & 3) == 3:  # Cross-dword unaligned read
        word1 = LDS[((Effective Address) >> 2) + 1]
        raw16 = ((word0 >> 24) & 0xFF) | ((word1 & 0xFF) << 8)
    else:
        raw16 = (word0 >> shift) & 0xFFFF
    vdst = (raw16 | 0xFFFF0000) if (raw16 & 0x8000) else raw16

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
SIGN_BIT16: int = 0x8000
SIGN_EXTEND_MASK: int = 0xFFFF0000

# Compute Shader RSRC2 metadata: 4 user SGPRs, workgroup id in s4, LDS_SIZE = 4 KiB
RSRC2_4K: int = 0x40088
VSHARP: list[int] = [0, 0, 0x1000, 0x31016FAC]


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def oracle_ds_read_i16(raw_16bit: int) -> int:
    """Pure mathematical reference oracle for ds_read_i16.
    
    Sign-extends a 16-bit halfword to 32-bit two's-complement integer.
    """
    val16 = raw_16bit & 0xFFFF
    if val16 & SIGN_BIT16:
        return (val16 | SIGN_EXTEND_MASK) & MASK32
    return val16


class LDSModel:
    """Architectural memory model representing the RDNA2 Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords: int = capacity_dwords
        self.memory: list[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = (eff_addr >> 2) & (self.capacity_dwords - 1)
        self.memory[idx] = val & MASK32

    def read_i16(self, vaddr: int, offset: int = 0) -> int:
        """Read a 16-bit signed integer from LDS with support for cross-dword unaligned access."""
        eff_addr = (vaddr + (offset & OFFSET16_MASK)) & MASK32
        idx0 = (eff_addr >> 2) & (self.capacity_dwords - 1)
        sh = (eff_addr & 3) * 8
        w0 = self.memory[idx0]
        if (eff_addr & 3) == 3:
            idx1 = ((eff_addr >> 2) + 1) & (self.capacity_dwords - 1)
            w1 = self.memory[idx1]
            raw16 = ((w0 >> 24) & 0xFF) | ((w1 & 0xFF) << 8)
        else:
            raw16 = (w0 >> sh) & 0xFFFF
        return oracle_ds_read_i16(raw16)


def make_cs_fssm(rsrc2: int = RSRC2_4K, size: tuple[int, int, int] = (32, 1, 1)) -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, rsrc2, *size)
    return bytes(blob)


def translate_compute(lines: list[str], name: str = "cs_ds_read_i16") -> str:
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
    DS_READ_I16 = "ds_read_i16"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_read_i16 instruction."""
    opcode: Opcode
    vdst: Register
    vaddr: Register
    offset: DSOffset = dataclasses.field(default_factory=DSOffset)

    def __post_init__(self) -> None:
        if not self.vdst.is_vgpr or self.vdst.count != 1:
            raise ValueError(f"vdst must be a single 32-bit VGPR (count=1): {self.vdst}")
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single 32-bit VGPR (count=1): {self.vaddr}")

    def to_assembly(self) -> str:
        vd_str = self.vdst.format_asm()
        va_str = self.vaddr.format_asm()
        off_str = f" offset:{self.offset.offset}" if self.offset.offset != 0 else ""
        return f"{self.opcode.value} {vd_str}, {va_str}{off_str}"


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


@dataclasses.dataclass(frozen=True)
class LdsReadI16ConditionCodes:
    """Condition codes resulting from signed read interpretation."""
    zero: bool
    negative: bool
    positive: bool


@dataclasses.dataclass(frozen=True)
class LdsReadI16StatusFlags:
    """Execution status and address resolution metadata."""
    effective_byte_address: int
    raw_16bit: int
    sign_extended_32bit: int

class ScalarLDSInterpreter:
    """Deterministic scalar interpreter for ds_read_i16."""

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
        """Executes a single ds_read_i16 instruction."""
        vaddr = self.read_reg_u32(RegKind.VGPR, inst.vaddr.index)
        eff_addr = (vaddr + (inst.offset.offset & OFFSET16_MASK)) & MASK32
        res = self.lds.read_i16(vaddr, inst.offset.offset)
        self.write_reg_u32(RegKind.VGPR, inst.vdst.index, res)
        self.last_condition_codes = LdsReadI16ConditionCodes(
            zero=(res == 0),
            negative=bool(res & 0x80000000),
            positive=(res > 0 and not bool(res & 0x80000000)),
        )
        self.last_status_flags = LdsReadI16StatusFlags(
            effective_byte_address=eff_addr,
            raw_16bit=res & 0xFFFF,
            sign_extended_32bit=res,
        )
        self.trace.append(f"{inst.to_assembly()} => {inst.vdst.format_asm()}={res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]



# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDLDSWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution for ds_read_i16."""

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
        """Executes a ds_read_i16 instruction across all lanes in the wavefront."""
        lane_vaddrs: list[int] = []
        for lane in range(self.wave_size):
            va = self.read_lane_reg_u32(lane, RegKind.VGPR, inst.vaddr.index)
            lane_vaddrs.append(va)

        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            res = self.lds.read_i16(lane_vaddrs[lane], inst.offset.offset)
            self.write_lane_reg_u32(lane, RegKind.VGPR, inst.vdst.index, res)
            lane_results.append(res)

        return lane_results


class TestLdsReadI16ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and hardware translator execution."""

    def setUp(self) -> None:
        self.lds = LDSModel(LDS_SIZE_DWORDS)

    def test_oracle_positive_values(self) -> None:
        self.assertEqual(oracle_ds_read_i16(0x0001), 1)
        self.assertEqual(oracle_ds_read_i16(0x0042), 66)
        self.assertEqual(oracle_ds_read_i16(0x7FFF), 32767)

    def test_oracle_zero(self) -> None:
        self.assertEqual(oracle_ds_read_i16(0x0000), 0)

    def test_oracle_negative_values(self) -> None:
        # 0x8000 = -32768 -> 0xFFFF8000
        self.assertEqual(oracle_ds_read_i16(0x8000), 0xFFFF8000)
        # 0xFFFF = -1 -> 0xFFFFFFFF
        self.assertEqual(oracle_ds_read_i16(0xFFFF), 0xFFFFFFFF)
        # 0xFFFE = -2 -> 0xFFFFFFFE
        self.assertEqual(oracle_ds_read_i16(0xFFFE), 0xFFFFFFFE)

    def test_lds_model_halfword_reads(self) -> None:
        # Dword at address 0 contains low 16-bit 0x8000, high 16-bit 0x0020
        self.lds.write_dword(0, 0x00208000)
        # Read low halfword at byte 0
        self.assertEqual(self.lds.read_i16(vaddr=0, offset=0), 0xFFFF8000)
        # Read high halfword at byte 2
        self.assertEqual(self.lds.read_i16(vaddr=0, offset=2), 0x00000020)

    def test_translator_hardware_execution(self) -> None:
        # Translates a compute kernel with ds_read_i16 and executes on spirv_mini_interp
        lines = [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0xFFFF8000",  # low half is 0x8000 (-32768)
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        spv = translate_compute(lines)
        self.assertIn("OpBitFieldSExtract", spv)
        mem = run_compute_kernel(lines, lanes=1)
        self.assertEqual(mem.get(0), 0xFFFF8000)



class TestLdsReadI16AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_construction(self) -> None:
        v0 = Register(RegKind.VGPR, 0)
        v1 = Register(RegKind.VGPR, 1)
        self.assertEqual(v0.format_asm(), "v0")
        self.assertEqual(v1.format_asm(), "v1")

    def test_instruction_formatting(self) -> None:
        inst_no_off = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 3),
            vaddr=Register(RegKind.VGPR, 1),
        )
        self.assertEqual(inst_no_off.to_assembly(), "ds_read_i16 v3, v1")

        inst_with_off = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 4),
            vaddr=Register(RegKind.VGPR, 2),
            offset=DSOffset(offset=12),
        )
        self.assertEqual(inst_with_off.to_assembly(), "ds_read_i16 v4, v2 offset:12")

    def test_instruction_validation(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_READ_I16,
                vdst=Register(RegKind.SGPR, 1),
                vaddr=Register(RegKind.VGPR, 2),
            )
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.DS_READ_I16,
                vdst=Register(RegKind.VGPR, 1),
                vaddr=Register(RegKind.SGPR, 2),
            )

    def test_offset_validation(self) -> None:
        with self.assertRaises(ValueError):
            DSOffset(offset=-1)
        with self.assertRaises(ValueError):
            DSOffset(offset=65536)

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSOffset(offset=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 6),
            vaddr=Register(RegKind.VGPR, 1),
        ))
        expected = "ds_read_i16 v5, v1 offset:2\nds_read_i16 v6, v1"
        self.assertEqual(prog.to_assembly(), expected)



class TestLdsReadI16ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_scalar_execution_basic(self) -> None:
        # Seed address 0x10 with 0xFFFE (low 16) and 0x0005 (high 16)
        self.interp.lds.write_dword(0x10, 0x0005FFFE)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)

        inst_lo = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 3),
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSOffset(offset=0),
        )
        res_lo = self.interp.step(inst_lo)
        self.assertEqual(res_lo, 0xFFFFFFFE)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 3), 0xFFFFFFFE)

        inst_hi = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 4),
            vaddr=Register(RegKind.VGPR, 1),
            offset=DSOffset(offset=2),
        )
        res_hi = self.interp.step(inst_hi)
        self.assertEqual(res_hi, 5)
        self.assertEqual(self.interp.read_reg_u32(RegKind.VGPR, 4), 5)



class TestLdsReadI16SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergent_addresses(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=32, capacity_dwords=1024)
        for lane in range(32):
            val16 = (0x8000 | lane) if (lane % 2 == 1) else (100 + lane)
            simd.lds.write_dword(lane * 4, val16)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)

        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            expected = oracle_ds_read_i16((0x8000 | lane) if (lane % 2 == 1) else (100 + lane))
            self.assertEqual(results[lane], expected)
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 2), expected)

    def test_wave64_exec_masking_partial(self) -> None:
        simd = SIMDLDSWavefrontInterpreter(wave_size=64, capacity_dwords=1024)
        simd.set_exec_mask(0x00000000FFFFFFFF)  # Only lower 32 active
        for lane in range(64):
            simd.lds.write_dword(lane * 4, 0xFFFF)  # -1 in i16
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 1, lane * 4)
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 2, 0x12345678)

        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        results = simd.step(inst)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(results[lane], 0xFFFFFFFF)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 2), 0xFFFFFFFF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 2), 0x12345678)

    def test_translator_simd_divergent_reads(self) -> None:
        # Multi-lane compute shader where each lane reads 16-bit word from lane * 4
        # Lane 0 writes 0x000A, Lane 1 writes 0xFFFF (-1), Lane 2 writes 0x8000 (-32768)
        lines = [
            "v_lshlrev_b32_e32 v1, 2, v0",  # v1 = lane * 4
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_saveexec_b64 s[4:5], vcc",
            "v_mov_b32_e32 v2, 0x000A",
            "ds_write_b32 v1, v2",
            "s_mov_b64 exec, s[4:5]",
            "v_cmp_eq_u32_e32 vcc, 1, v0",
            "s_and_saveexec_b64 s[4:5], vcc",
            "v_mov_b32_e32 v2, 0xFFFF",
            "ds_write_b32 v1, v2",
            "s_mov_b64 exec, s[4:5]",
            "v_cmp_eq_u32_e32 vcc, 2, v0",
            "s_and_saveexec_b64 s[4:5], vcc",
            "v_mov_b32_e32 v2, 0x8000",
            "ds_write_b32 v1, v2",
            "s_mov_b64 exec, s[4:5]",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_i16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v3, v1, s[0:3], 0 offen",
        ]
        mem = run_compute_kernel(lines, lanes=3)
        self.assertEqual(mem.get(0), 10)
        self.assertEqual(mem.get(4), 0xFFFFFFFF)
        self.assertEqual(mem.get(8), 0xFFFF8000)



class TestLdsReadI16RegisterAliasing(unittest.TestCase):
    """Unit tests for register preservation and operand aliasing."""

    def test_unrelated_vgprs_preserved(self) -> None:
        interp = ScalarLDSInterpreter(capacity_dwords=1024)
        for r in range(256):
            interp.write_reg_u32(RegKind.VGPR, r, 0xCAFE0000 | r)
        interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        interp.lds.write_dword(0x10, 0x7FFF)

        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 5),
            vaddr=Register(RegKind.VGPR, 1),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0x7FFF)
        for r in range(256):
            if r == 5:
                self.assertEqual(interp.read_reg_u32(RegKind.VGPR, r), 0x7FFF)
            elif r == 1:
                self.assertEqual(interp.read_reg_u32(RegKind.VGPR, r), 0x10)
            else:
                self.assertEqual(interp.read_reg_u32(RegKind.VGPR, r), 0xCAFE0000 | r)

    def test_vdst_vaddr_aliased(self) -> None:
        # ds_read_i16 v1, v1: v1 is read as address, then overwritten with sign-extended result
        interp = ScalarLDSInterpreter(capacity_dwords=1024)
        interp.write_reg_u32(RegKind.VGPR, 1, 0x20)
        interp.lds.write_dword(0x20, 0xFFFF)  # -1

        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 1),
            vaddr=Register(RegKind.VGPR, 1),
        )
        res = interp.step(inst)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0xFFFFFFFF)



class TestLdsReadI16ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and VCC isolation."""

    def setUp(self) -> None:
        self.interp = ScalarLDSInterpreter(capacity_dwords=1024)

    def test_zero_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 0x0000)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

    def test_positive_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 0x0042)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 66)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.negative)
        self.assertTrue(cc.positive)

    def test_negative_flag(self) -> None:
        self.interp.lds.write_dword(0x10, 0x8001)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        res = self.interp.step(inst)
        self.assertEqual(res, 0xFFFF8001)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.positive)

    def test_vcc_isolation(self) -> None:
        self.interp.vcc = 0x12345678DEADBEEF
        self.interp.lds.write_dword(0x10, 0x1234)
        self.interp.write_reg_u32(RegKind.VGPR, 1, 0x10)
        inst = Instruction(
            opcode=Opcode.DS_READ_I16,
            vdst=Register(RegKind.VGPR, 2),
            vaddr=Register(RegKind.VGPR, 1),
        )
        self.interp.step(inst)
        self.assertEqual(self.interp.vcc, 0x12345678DEADBEEF)



class TestLdsReadI16EdgeCases(unittest.TestCase):
    """Unit tests for boundary extremes, halfword integer limits, and IEEE 754 bit-pattern invariance."""

    def test_16bit_integer_boundaries(self) -> None:
        # Max positive int16: 0x7FFF = +32767
        self.assertEqual(oracle_ds_read_i16(0x7FFF), 32767)
        # Min negative int16: 0x8000 = -32768 -> 0xFFFF8000
        self.assertEqual(oracle_ds_read_i16(0x8000), 0xFFFF8000)
        # -1 int16: 0xFFFF -> 0xFFFFFFFF
        self.assertEqual(oracle_ds_read_i16(0xFFFF), 0xFFFFFFFF)

    def test_max_offset_addressing(self) -> None:
        lds = LDSModel(capacity_dwords=16384)
        lds.write_dword(65532, 0x8000)
        res = lds.read_i16(vaddr=0, offset=65532)
        self.assertEqual(res, 0xFFFF8000)

    def test_cross_dword_unaligned_read(self) -> None:
        # Unaligned read across dword boundary at byte offset 3
        # Dword 0: high byte (byte 3) is 0x34
        # Dword 1: low byte (byte 0) is 0x12
        # Result halfword is 0x1234 -> sign-extended = 0x00001234
        lds = LDSModel(capacity_dwords=1024)
        lds.write_dword(0, 0x34000000)
        lds.write_dword(4, 0x00000012)
        res = lds.read_i16(vaddr=0, offset=3)
        self.assertEqual(res, 0x1234)

        # Cross-dword read with sign bit set (0x80 in byte 1 -> 0x8034 -> 0xFFFF8034)
        lds.write_dword(4, 0x00000080)
        res_neg = lds.read_i16(vaddr=0, offset=3)
        self.assertEqual(res_neg, 0xFFFF8034)

    def test_fp16_bitpatterns_treated_as_integers(self) -> None:
        # FP16 +inf: 0x7C00 -> positive int16: 0x00007C00
        self.assertEqual(oracle_ds_read_i16(0x7C00), 0x7C00)
        # FP16 -inf: 0xFC00 -> sign-extended: 0xFFFFFC00
        self.assertEqual(oracle_ds_read_i16(0xFC00), 0xFFFFFC00)
        # FP16 qNaN: 0x7E00 -> positive int16: 0x00007E00
        self.assertEqual(oracle_ds_read_i16(0x7E00), 0x7E00)
        # FP16 -0.0: 0x8000 -> sign-extended: 0xFFFF8000
        self.assertEqual(oracle_ds_read_i16(0x8000), 0xFFFF8000)


if __name__ == "__main__":
    unittest.main()
