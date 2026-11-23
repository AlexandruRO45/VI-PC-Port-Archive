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
        res = self.lds.read_i16(vaddr, inst.offset.offset)
        self.write_reg_u32(RegKind.VGPR, inst.vdst.index, res)
        self.trace.append(f"{inst.to_assembly()} => {inst.vdst.format_asm()}={res:#010x}")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


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


if __name__ == "__main__":
    unittest.main()
