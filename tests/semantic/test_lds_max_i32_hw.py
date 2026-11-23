"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_i32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) ds_max_i32 / ds_max_rtn_i32 atomic signed maximum semantics:
  - Atomically computes:
      addr = (vaddr + offset) & 0xFFFFFFFF
      idx = (addr >> 2) & (lds_dwords - 1)
      old_val = LDS[idx]
      LDS[idx] = max(to_signed_i32(old_val), to_signed_i32(vdata))
  - In returning form (ds_max_rtn_i32): writes old_val to vdst VGPR.
  - In non-returning form (ds_max_i32): prior value is discarded.
  - Signed 32-bit two's complement integer arithmetic:
      * 0xFFFFFFFF is -1 (strictly less than 0).
      * 0x80000000 is INT_MIN (-2^31).
      * 0x7FFFFFFF is INT_MAX (2^31 - 1).
      * Crucially diverges from unsigned maximum (ds_max_u32).
  - Pure memory atomic: condition codes (SCC / VCC) are not modified.

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares output buffers.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
SIGN_BIT = 0x80000000
INT32_MIN = 0x80000000
INT32_MAX = 0x7FFFFFFF

# Compute FSSM header: RSRC2 0x8088 (4 user SGPRs, 64x1x1 threads)
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 64, 1, 1)
CS_FSSM_BYTES = bytes(CS_FSSM)

OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28


def to_signed_i32(val: int) -> int:
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT) else val


def to_unsigned_u32(val: int) -> int:
    return val & MASK32


def ref_max_i32(current_val: int, new_val: int) -> int:
    """Architectural reference oracle for RDNA2 signed 32-bit atomic max."""
    c = to_signed_i32(current_val)
    n = to_signed_i32(new_val)
    return to_unsigned_u32(max(c, n))


def run_cs_lds(instructions: Sequence[str]) -> dict[int, int]:
    """Translate and execute a synthetic compute shader, returning output memory."""
    lines = list(instructions) + ["s_endpgm"]
    spvasm = support.translate_lines("cs_lds_max_i32_test", lines, fssm=CS_FSSM_BYTES)
    memory: dict[int, int] = {}
    spirv_mini_interp.run_fragment(
        spvasm,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=OUT_VSHARP,
        memory=memory,
    )
    return memory

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_signed_two_complement_max(self):
        # max(-10, 5) -> 5
        curr = to_unsigned_u32(-10)
        val = to_unsigned_u32(5)
        self.assertEqual(ref_max_i32(curr, val), 5)

        # max(-5, -1) -> -1 (0xFFFFFFFF)
        c2 = to_unsigned_u32(-5)
        v2 = to_unsigned_u32(-1)
        self.assertEqual(ref_max_i32(c2, v2), 0xFFFFFFFF)

        # max(INT32_MIN, 0) -> 0
        self.assertEqual(ref_max_i32(INT32_MIN, 0), 0)

    def test_translator_basic_execution(self):
        # Seed LDS[0] with -10 (0xFFFFFFF6), apply ds_max_i32 with 5, read back to mem[0]
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0xFFFFFFF6",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 5",
            "ds_max_i32 v0, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 5)

    def test_translator_with_byte_offset(self):
        # Seed LDS[4] with -50, apply ds_max_i32 at offset 4 with -20
        instructions = [
            "v_mov_b32_e32 v0, 0",
            f"v_mov_b32_e32 v1, {to_unsigned_u32(-50):#x}",
            "ds_write_b32 v0, v1 offset:4",
            "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v2, {to_unsigned_u32(-20):#x}",
            "ds_max_i32 v0, v2 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0 offset:4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), to_unsigned_u32(-20))

class RegType(Enum):
    VGPR = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0

    def __post_init__(self):
        if not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, index=idx)

    def to_asm(self) -> str:
        return f"v{self.index}"


@dataclass(frozen=True)
class DSMaxI32Instruction:
    vaddr: Operand
    vdata: Operand
    offset: int = 0
    rtn: bool = False
    vdst: Optional[Operand] = None

    def __post_init__(self):
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"Offset {self.offset} out of bounds [0, 65535]")
        if self.rtn and self.vdst is None:
            raise ValueError("Returning form requires vdst operand")
        if not self.rtn and self.vdst is not None:
            raise ValueError("Non-returning form must not have vdst operand")

    def to_asm(self) -> str:
        off_str = f" offset:{self.offset}" if self.offset != 0 else ""
        if self.rtn:
            return f"ds_max_rtn_i32 {self.vdst.to_asm()}, {self.vaddr.to_asm()}, {self.vdata.to_asm()}{off_str}"
        return f"ds_max_i32 {self.vaddr.to_asm()}, {self.vdata.to_asm()}{off_str}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn1 = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
            offset=16,
        )
        self.assertEqual(insn1.to_asm(), "ds_max_i32 v0, v1 offset:16")

        insn2 = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
            offset=8,
            rtn=True,
            vdst=Operand.vgpr(2),
        )
        self.assertEqual(insn2.to_asm(), "ds_max_rtn_i32 v2, v0, v1 offset:8")

    def test_ast_immutability(self):
        insn = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
        )
        with self.assertRaises(Exception):
            insn.offset = 4  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(ValueError):
            DSMaxI32Instruction(
                vaddr=Operand.vgpr(0),
                vdata=Operand.vgpr(1),
                offset=70000,
            )
        with self.assertRaises(ValueError):
            DSMaxI32Instruction(
                vaddr=Operand.vgpr(0),
                vdata=Operand.vgpr(1),
                rtn=True,
                vdst=None,
            )

class LDSState:
    def __init__(self, size_dwords: int = 1024):
        self.size_dwords = size_dwords
        self.data = [0] * size_dwords

    def atomic_max_i32(self, byte_addr: int, val: int) -> int:
        idx = ((byte_addr & MASK32) >> 2) & (self.size_dwords - 1)
        old = self.data[idx]
        self.data[idx] = ref_max_i32(old, val)
        return old


class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.scc = 0
        self.vcc = 0


class ScalarLDSInterpreter:
    def __init__(self, lds: Optional[LDSState] = None, regs: Optional[ScalarRegisterState] = None):
        self.lds = lds if lds is not None else LDSState()
        self.regs = regs if regs is not None else ScalarRegisterState()

    def step(self, insn: DSMaxI32Instruction) -> int:
        base_addr = self.regs.vgprs[insn.vaddr.index]
        eff_addr = (base_addr + insn.offset) & MASK32
        vdata = self.regs.vgprs[insn.vdata.index]
        old = self.lds.atomic_max_i32(eff_addr, vdata)
        if insn.rtn and insn.vdst is not None:
            self.regs.vgprs[insn.vdst.index] = old
        return old


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarLDSInterpreter()
        # Seed LDS[0] with -100
        interp.lds.data[0] = to_unsigned_u32(-100)

        interp.regs.vgprs[0] = 0  # addr
        interp.regs.vgprs[1] = to_unsigned_u32(-50)
        interp.regs.vgprs[2] = to_unsigned_u32(10)

        # Step 1: ds_max_rtn_i32 v3, v0, v1 -> old=-100, new=-50
        insn1 = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
            rtn=True,
            vdst=Operand.vgpr(3),
        )
        old1 = interp.step(insn1)
        self.assertEqual(old1, to_unsigned_u32(-100))
        self.assertEqual(interp.regs.vgprs[3], to_unsigned_u32(-100))
        self.assertEqual(interp.lds.data[0], to_unsigned_u32(-50))

        # Step 2: ds_max_rtn_i32 v4, v0, v2 -> old=-50, new=10
        insn2 = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(2),
            rtn=True,
            vdst=Operand.vgpr(4),
        )
        old2 = interp.step(insn2)
        self.assertEqual(old2, to_unsigned_u32(-50))
        self.assertEqual(interp.regs.vgprs[4], to_unsigned_u32(-50))
        self.assertEqual(interp.lds.data[0], 10)

class WaveSize(Enum):
    WAVE32 = 32
    WAVE64 = 64


class SIMDWaveState:
    def __init__(self, size: WaveSize = WaveSize.WAVE32):
        self.size = size
        self.num_lanes = size.value
        self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        self.exec_mask = (1 << self.num_lanes) - 1

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)


class SIMDLDSInterpreter:
    @staticmethod
    def step(insn: DSMaxI32Instruction, wave: SIMDWaveState, lds: LDSState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            base_addr = wave.vgprs[lane][insn.vaddr.index]
            eff_addr = (base_addr + insn.offset) & MASK32
            vdata = wave.vgprs[lane][insn.vdata.index]
            old = lds.atomic_max_i32(eff_addr, vdata)
            if insn.rtn and insn.vdst is not None:
                wave.vgprs[lane][insn.vdst.index] = old


class TestSIMDDivergence(unittest.TestCase):
    def test_wave32_divergence_and_exec_masking(self):
        lds = LDSState()
        wave = SIMDWaveState(WaveSize.WAVE32)
        wave.exec_mask = 0x55555555  # Alternating lanes

        # Initialize LDS with -100 at all slots
        for i in range(32):
            lds.data[i] = to_unsigned_u32(-100)

        sentinel = 0xDEADBEEF
        for lane in range(32):
            wave.vgprs[lane][0] = lane * 4  # byte address
            wave.vgprs[lane][1] = lane       # new value
            wave.vgprs[lane][2] = sentinel   # vdst

        insn = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
            rtn=True,
            vdst=Operand.vgpr(2),
        )
        SIMDLDSInterpreter.step(insn, wave, lds)

        for lane in range(32):
            if lane % 2 == 0:  # Active
                self.assertEqual(wave.vgprs[lane][2], to_unsigned_u32(-100))
                self.assertEqual(lds.data[lane], lane)
            else:  # Inactive
                self.assertEqual(wave.vgprs[lane][2], sentinel)
                self.assertEqual(lds.data[lane], to_unsigned_u32(-100))

    def test_wave64_divergence(self):
        lds = LDSState()
        wave = SIMDWaveState(WaveSize.WAVE64)
        wave.exec_mask = 0x00000000FFFFFFFF  # Lower 32 lanes active

        for i in range(64):
            lds.data[i] = to_unsigned_u32(-50)

        sentinel = 0xCAFEBABE
        for lane in range(64):
            wave.vgprs[lane][0] = lane * 4
            wave.vgprs[lane][1] = 100
            wave.vgprs[lane][2] = sentinel

        insn = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
            rtn=True,
            vdst=Operand.vgpr(2),
        )
        SIMDLDSInterpreter.step(insn, wave, lds)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][2], to_unsigned_u32(-50))
            self.assertEqual(lds.data[lane], 100)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][2], sentinel)
            self.assertEqual(lds.data[lane], to_unsigned_u32(-50))

class TestRegisterAliasing(unittest.TestCase):
    def test_alias_vaddr_equals_vdata(self):
        # Address and data share the same register v0
        # If v0 is 0, address is 0 and data is 0.
        # Seed LDS[0] with -5. After max with 0, LDS[0] should be 0.
        instructions = [
            "v_mov_b32_e32 v0, 0",
            f"v_mov_b32_e32 v1, {to_unsigned_u32(-5):#x}",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_max_i32 v0, v0",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v0",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 0)

    def test_alias_vdst_equals_vaddr(self):
        # In returning form: vdst and vaddr share v0
        # Seed LDS[0] with 42. vaddr=0, vdata=10.
        # ds_max_rtn_i32 v0, v0, v1 -> returns old 42 into v0.
        instructions = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 42",
            "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 10",
            "ds_max_rtn_i32 v0, v0, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v0, v31, s[0:3], 0 offen",
        ]
        mem = run_cs_lds(instructions)
        self.assertEqual(mem.get(0), 42)

@dataclass
class IntegerStatusFlags:
    zero: bool = False
    negative: bool = False


def classify_integer_status(val: int) -> IntegerStatusFlags:
    val &= MASK32
    return IntegerStatusFlags(
        zero=(val == 0),
        negative=bool(val & SIGN_BIT),
    )


class TestConditionCodesAndStatus(unittest.TestCase):
    def test_vcc_and_scc_preservation(self):
        # LDS atomic max does not modify VCC or SCC
        interp = ScalarLDSInterpreter()
        interp.regs.scc = 1
        interp.regs.vcc = 0x123456789ABCDEF0
        interp.regs.vgprs[0] = 0
        interp.regs.vgprs[1] = 10

        insn = DSMaxI32Instruction(
            vaddr=Operand.vgpr(0),
            vdata=Operand.vgpr(1),
        )
        interp.step(insn)

        self.assertEqual(interp.regs.scc, 1)
        self.assertEqual(interp.regs.vcc, 0x123456789ABCDEF0)

    def test_status_flag_classification(self):
        s_zero = classify_integer_status(0)
        self.assertTrue(s_zero.zero)
        self.assertFalse(s_zero.negative)

        s_neg = classify_integer_status(0xFFFFFFFF)
        self.assertFalse(s_neg.zero)
        self.assertTrue(s_neg.negative)

        s_pos = classify_integer_status(42)
        self.assertFalse(s_pos.zero)
        self.assertFalse(s_pos.negative)


if __name__ == "__main__":
    unittest.main()
