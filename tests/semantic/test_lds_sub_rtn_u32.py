from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_sub_rtn_u32.

Opcode:
  - ds_sub_rtn_u32 vdst, vaddr, vdata [offset:N]
      Atomic 32-bit subtraction on LDS memory with destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        LDS[idx] = (old_val - vdata) & 0xFFFFFFFF
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + optional unsigned immediate byte offset.
  - Modulo arithmetic: Subtraction wraps strictly modulo 2^32 (e.g. 0 - 1 = 0xFFFFFFFF).
  - Return value: Prior value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum
import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB LDS capacity in dwords
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106

# IEEE 754 Boundary Constants
POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000
SNAN_F32 = 0x7F800001
MIN_NORMAL_F32 = 0x00800000
MIN_SUBNORMAL_F32 = 0x00000001
MAX_SUBNORMAL_F32 = 0x007FFFFF
MAX_NORMAL_F32 = 0x7F7FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


class LDSMemory:
    """Model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)



def compute_sub_borrow_u32(a: int, b: int) -> int:
    """Calculate whether 32-bit subtraction (a - b) produces an unsigned borrow."""
    return 1 if (u32(a) < u32(b)) else 0


def compute_sub_zero_u32(a: int, b: int) -> int:
    """Calculate whether 32-bit subtraction (a - b) produces zero."""
    return 1 if (u32(a) == u32(b)) else 0


def compute_sub_sign_u32(a: int, b: int) -> int:
    """Calculate the sign bit (bit 31) of 32-bit subtraction result."""
    return 1 if bool(u32(a - b) & 0x80000000) else 0


def ref_ds_sub_rtn_u32(
    mem: LDSMemory,
    vaddr: int,
    vdata: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_sub_rtn_u32.

    Atomically computes:
      old_val = LDS[idx]
      LDS[idx] = (old_val - vdata) mod 2^32
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    idx = (effective_byte >> 2) & (mem.size_dwords - 1)
    old_val = mem.read_dword(idx)
    new_val = u32(old_val - u32(vdata))
    mem.write_dword(idx, new_val)
    return old_val


class TestLdsSubRtnU32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)

    def test_basic_sub_and_return(self) -> None:
        self.mem.write_dword(10, 100)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=40, vdata=30, offset=0)
        self.assertEqual(ret, 100)
        self.assertEqual(self.mem.read_dword(10), 70)

    def test_sub_underflow_wrap(self) -> None:
        self.mem.write_dword(5, 5)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=20, vdata=10, offset=0)
        self.assertEqual(ret, 5)
        self.assertEqual(self.mem.read_dword(5), MASK32 - 4)

    def test_sub_zero_preserves_val_and_returns(self) -> None:
        self.mem.write_dword(2, 42)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=8, vdata=0, offset=0)
        self.assertEqual(ret, 42)
        self.assertEqual(self.mem.read_dword(2), 42)

    def test_address_offset_and_wrapping(self) -> None:
        self.mem.write_dword(6, 0x12345678)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=16, vdata=0x11111111, offset=8)
        self.assertEqual(ret, 0x12345678)
        self.assertEqual(self.mem.read_dword(6), 0x01234567)



# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class DSOpcode(Enum):
    """Supported DS family opcodes."""
    DS_SUB_RTN_U32 = "ds_sub_rtn_u32"


class RegType(Enum):
    """Register file class."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.index < DEFAULT_NUM_VGPRS):
            raise ValueError(
                f"Register index out of bounds: {self.reg_type.value}{self.index} "
                f"(expected 0 <= index < {DEFAULT_NUM_VGPRS})"
            )

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class DSSubRtnInstruction:
    """Symbolic representation of ds_sub_rtn_u32 instruction."""
    dst: RegisterOperand
    addr: RegisterOperand
    data: RegisterOperand
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_SUB_RTN_U32

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination operand must be a VGPR")
        if self.addr.reg_type != RegType.VGPR:
            raise TypeError("Address operand must be a VGPR")
        if self.data.reg_type != RegType.VGPR:
            raise TypeError("Data operand must be a VGPR")
        if not (0 <= self.offset <= 0xFFFF):
            raise ValueError(f"Offset out of 16-bit range: {self.offset} (expected 0 <= offset <= 65535)")

    def disasm(self) -> str:
        """Render disassembled assembly syntax."""
        parts = [f"{self.opcode.value} {self.dst.name}, {self.addr.name}, {self.data.name}"]
        if self.offset != 0:
            parts.append(f"offset:{self.offset}")
        if self.gds:
            parts.append("gds")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState:
    """Architectural state for scalar execution context."""

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_dwords)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar interpreter for ds_sub_rtn_u32."""

    def execute(self, instr: DSSubRtnInstruction, state: ScalarMachineState) -> int:
        """Executes a scalar ds_sub_rtn_u32 instruction."""
        if instr.opcode != DSOpcode.DS_SUB_RTN_U32:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        vaddr = state.get_vgpr(instr.addr.index)
        vdata = state.get_vgpr(instr.data.index)
        old_val = ref_ds_sub_rtn_u32(state.lds, vaddr=vaddr, vdata=vdata, offset=instr.offset)
        state.set_vgpr(instr.dst.index, old_val)
        return old_val


# ---------------------------------------------------------------------------
# SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ---------------------------------------------------------------------------

class SIMDState:
    """Architectural state for SIMD Wave32 / Wave64 execution context."""

    def __init__(
        self,
        num_lanes: int = DEFAULT_NUM_LANES,
        exec_mask: int = MASK64,
        lds_dwords: int = DEFAULT_LDS_DWORDS,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Supported lane counts are 32 or 64, got: {num_lanes}")
        self.num_lanes: int = num_lanes
        mask_limit = MASK32 if num_lanes == 32 else MASK64
        self.exec_mask: int = exec_mask & mask_limit
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_dwords=lds_dwords)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx]

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = u32(val)

    def get_sgpr(self, reg_idx: int) -> int:
        return self.sgprs[reg_idx]

    def set_sgpr(self, reg_idx: int, val: int) -> None:
        self.sgprs[reg_idx] = u32(val)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane execution and EXEC masking."""

    def execute(self, instr: DSSubRtnInstruction, state: SIMDState) -> list[tuple[int, int]]:
        """Executes ds_sub_rtn_u32 across SIMD lanes."""
        if instr.opcode != DSOpcode.DS_SUB_RTN_U32:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        latched_lanes: list[tuple[int, int, int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                vaddr = state.get_vgpr(lane, instr.addr.index)
                vdata = state.get_vgpr(lane, instr.data.index)
                latched_lanes.append((lane, vaddr, vdata))

        results: list[tuple[int, int]] = []
        for lane, vaddr, vdata in latched_lanes:
            old_val = ref_ds_sub_rtn_u32(state.lds, vaddr=vaddr, vdata=vdata, offset=instr.offset)
            state.set_vgpr(lane, instr.dst.index, old_val)
            results.append((lane, old_val))

        return results


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_SUB_RTN_U32.value, "ds_sub_rtn_u32")

    def test_register_operand_validation(self) -> None:
        v0 = RegisterOperand(index=0, reg_type=RegType.VGPR)
        self.assertEqual(v0.name, "v0")

        v255 = RegisterOperand(index=255, reg_type=RegType.VGPR)
        self.assertEqual(v255.name, "v255")

        with self.assertRaises(ValueError):
            RegisterOperand(index=-1, reg_type=RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(index=256, reg_type=RegType.VGPR)

    def test_instruction_validation(self) -> None:
        v0 = RegisterOperand(0)
        v1 = RegisterOperand(1)
        v2 = RegisterOperand(2)

        instr = DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=64)
        self.assertEqual(instr.offset, 64)
        self.assertFalse(instr.gds)

        with self.assertRaises(ValueError):
            DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=-1)
        with self.assertRaises(ValueError):
            DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=65536)

        s0 = RegisterOperand(0, reg_type=RegType.SGPR)
        with self.assertRaises(TypeError):
            DSSubRtnInstruction(dst=s0, addr=v1, data=v2)
        with self.assertRaises(TypeError):
            DSSubRtnInstruction(dst=v0, addr=s0, data=v2)
        with self.assertRaises(TypeError):
            DSSubRtnInstruction(dst=v0, addr=v1, data=s0)

    def test_disasm_rendering(self) -> None:
        v0 = RegisterOperand(0)
        v1 = RegisterOperand(1)
        v2 = RegisterOperand(2)

        instr_basic = DSSubRtnInstruction(dst=v0, addr=v1, data=v2)
        self.assertEqual(instr_basic.disasm(), "ds_sub_rtn_u32 v0, v1, v2")

        instr_offset = DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=16)
        self.assertEqual(instr_offset.disasm(), "ds_sub_rtn_u32 v0, v1, v2 offset:16")

        instr_gds = DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=32, gds=True)
        self.assertEqual(instr_gds.disasm(), "ds_sub_rtn_u32 v0, v1, v2 offset:32 gds")

    def test_structural_equality_and_immutability(self) -> None:
        v0 = RegisterOperand(0)
        v1 = RegisterOperand(1)
        v2 = RegisterOperand(2)

        i1 = DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=10)
        i2 = DSSubRtnInstruction(dst=v0, addr=v1, data=v2, offset=10)
        self.assertEqual(i1, i2)
        self.assertEqual(hash(i1), hash(i2))

        with self.assertRaises(Exception):
            i1.offset = 20  # type: ignore


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState(lds_dwords=1024)
        self.interp = ScalarInterpreter()

    def test_scalar_sub_and_return(self) -> None:
        self.state.lds.write_dword(4, 500)
        self.state.set_vgpr(1, 16)
        self.state.set_vgpr(2, 200)
        self.state.set_vgpr(0, 0)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
            offset=0,
        )
        old_val = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 500)
        self.assertEqual(self.state.get_vgpr(0), 500)
        self.assertEqual(self.state.lds.read_dword(4), 300)

    def test_scalar_underflow_wrapping(self) -> None:
        self.state.lds.write_dword(8, 10)
        self.state.set_vgpr(1, 32)
        self.state.set_vgpr(2, 25)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        old_val = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 10)
        self.assertEqual(self.state.get_vgpr(0), 10)
        self.assertEqual(self.state.lds.read_dword(8), MASK32 - 14)

    def test_scalar_offset_handling(self) -> None:
        self.state.lds.write_dword(31, 1000)
        self.state.set_vgpr(1, 100)
        self.state.set_vgpr(2, 100)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
            offset=24,
        )
        old_val = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 1000)
        self.assertEqual(self.state.get_vgpr(0), 1000)
        self.assertEqual(self.state.lds.read_dword(31), 900)

    def test_scalar_determinism(self) -> None:
        runs: list[tuple[int, int]] = []
        for _ in range(50):
            st = ScalarMachineState(lds_dwords=512)
            st.lds.write_dword(20, 0x12345678)
            st.set_vgpr(1, 80)
            st.set_vgpr(2, 0x00000078)
            instr = DSSubRtnInstruction(
                dst=RegisterOperand(0),
                addr=RegisterOperand(1),
                data=RegisterOperand(2),
            )
            ret = self.interp.execute(instr, st)
            runs.append((ret, st.lds.read_dword(20)))

        self.assertTrue(all(r == (0x12345678, 0x12345600) for r in runs))


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_execution(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=MASK32, lds_dwords=256)
        for i in range(32):
            state.lds.write_dword(i, (i + 1) * 100)
            state.set_vgpr(i, 1, i * 4)
            state.set_vgpr(i, 2, 50)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        results = self.interp.execute(instr, state)

        self.assertEqual(len(results), 32)
        for i in range(32):
            expected_old = (i + 1) * 100
            self.assertEqual(state.get_vgpr(i, 0), expected_old)
            self.assertEqual(state.lds.read_dword(i), expected_old - 50)

    def test_wave64_divergence_and_exec_masking(self) -> None:
        alt_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=alt_mask, lds_dwords=512)

        for lane in range(64):
            state.lds.write_dword(lane, 1000 + lane)
            state.set_vgpr(lane, 1, lane * 4)
            state.set_vgpr(lane, 2, 200)
            state.set_vgpr(lane, 0, 0xDEADBEEF)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        results = self.interp.execute(instr, state)

        self.assertEqual(len(results), 32)

        for lane in range(64):
            if (lane % 2) == 0:
                self.assertEqual(state.get_vgpr(lane, 0), 1000 + lane)
                self.assertEqual(state.lds.read_dword(lane), 1000 + lane - 200)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), 0xDEADBEEF)
                self.assertEqual(state.lds.read_dword(lane), 1000 + lane)

    def test_simd_lane_collision_serialization(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x7, lds_dwords=256)
        target_dword = 10
        initial_val = 100
        state.lds.write_dword(target_dword, initial_val)

        for lane in range(3):
            state.set_vgpr(lane, 1, target_dword * 4)
            state.set_vgpr(lane, 2, (lane + 1) * 10)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        results = self.interp.execute(instr, state)

        self.assertEqual(results, [(0, 100), (1, 90), (2, 70)])
        self.assertEqual(state.get_vgpr(0, 0), 100)
        self.assertEqual(state.get_vgpr(1, 0), 90)
        self.assertEqual(state.get_vgpr(2, 0), 70)
        self.assertEqual(state.lds.read_dword(target_dword), 40)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_dst_aliases_addr(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(5, 777)
        state.set_vgpr(0, 20)
        state.set_vgpr(1, 77)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(0),
            data=RegisterOperand(1),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 777)
        self.assertEqual(state.get_vgpr(0), 777)
        self.assertEqual(state.lds.read_dword(5), 700)

    def test_dst_aliases_data(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(8, 250)
        state.set_vgpr(0, 32)
        state.set_vgpr(1, 50)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(1),
            addr=RegisterOperand(0),
            data=RegisterOperand(1),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 250)
        self.assertEqual(state.get_vgpr(1), 250)
        self.assertEqual(state.lds.read_dword(8), 200)

    def test_addr_aliases_data(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(12, 100)
        state.set_vgpr(0, 48)
        state.set_vgpr(2, 0)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(2),
            addr=RegisterOperand(0),
            data=RegisterOperand(0),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 100)
        self.assertEqual(state.get_vgpr(2), 100)
        self.assertEqual(state.get_vgpr(0), 48)
        self.assertEqual(state.lds.read_dword(12), 100 - 48)

    def test_dst_aliases_addr_and_data(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(4, 50)
        state.set_vgpr(0, 16)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(0),
            data=RegisterOperand(0),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 50)
        self.assertEqual(state.get_vgpr(0), 50)
        self.assertEqual(state.lds.read_dword(4), 34)

    def test_unmodified_registers_preserved(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(2, 123)
        state.set_vgpr(1, 8)
        state.set_vgpr(2, 23)
        for r in range(3, 10):
            state.set_vgpr(r, 0xA000 + r)
        for s in range(5):
            state.set_sgpr(s, 0xB000 + s)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        self.interp.execute(instr, state)

        for r in range(3, 10):
            self.assertEqual(state.get_vgpr(r), 0xA000 + r)
        for s in range(5):
            self.assertEqual(state.get_sgpr(s), 0xB000 + s)


class TestConditionCodesAndBorrow(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_flags_preserved_on_scalar_underflow_borrow(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(1, 5)
        state.set_vgpr(1, 4)
        state.set_vgpr(2, 10)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        interp = ScalarInterpreter()
        old_val = interp.execute(instr, state)

        self.assertEqual(old_val, 5)
        self.assertEqual(state.lds.read_dword(1), MASK32 - 4)

    def test_flags_preserved_on_zero_result(self) -> None:
        state = ScalarMachineState(lds_dwords=256)
        state.lds.write_dword(2, 42)
        state.set_vgpr(1, 8)
        state.set_vgpr(2, 42)

        instr = DSSubRtnInstruction(
            dst=RegisterOperand(0),
            addr=RegisterOperand(1),
            data=RegisterOperand(2),
        )
        interp = ScalarInterpreter()
        interp.execute(instr, state)

        self.assertEqual(state.lds.read_dword(2), 0)

    def test_mathematical_borrow_and_status_oracles(self) -> None:
        self.assertEqual(compute_sub_borrow_u32(0, 1), 1)
        self.assertEqual(compute_sub_borrow_u32(5, 10), 1)
        self.assertEqual(compute_sub_borrow_u32(10, 5), 0)
        self.assertEqual(compute_sub_borrow_u32(0, 0), 0)
        self.assertEqual(compute_sub_borrow_u32(MASK32, 1), 0)
        self.assertEqual(compute_sub_borrow_u32(0, MASK32), 1)

        self.assertEqual(compute_sub_zero_u32(50, 50), 1)
        self.assertEqual(compute_sub_zero_u32(50, 49), 0)

        self.assertEqual(compute_sub_sign_u32(10, 20), 1)
        self.assertEqual(compute_sub_sign_u32(20, 10), 0)


if __name__ == '__main__':
    unittest.main()
