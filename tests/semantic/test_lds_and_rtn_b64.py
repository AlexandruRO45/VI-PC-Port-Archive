from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_and_rtn_b64.

Opcode:
  - ds_and_rtn_b64 vdst, vaddr, vdata [offset:N]
      Atomic 64-bit bitwise AND on LDS memory with 64-bit destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte & (lds_bytes - 1)) & ~7
        old_val = LDS_qword[idx]
        LDS_qword[idx] = old_val & vdata
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in 32-bit VGPR vaddr + optional unsigned immediate byte offset.
  - Alignment: 64-bit access is 8-byte aligned (low 3 bits of byte address masked out).
  - Return value: Prior 64-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR pair v[dst:dst+1].
  - Flags: LDS operations do not alter VCC or SCC condition flags.
  - Untyped bitvector: Operates on raw 64-bit qwords without IEEE 754 interpretation or NaN quietening.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB LDS capacity in bytes
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001
MIN_NORMAL_F64 = 0x0010000000000000
MIN_SUBNORMAL_F64 = 0x0000000000000001
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MAX_NORMAL_F64 = 0x7FEFFFFFFFFFFFFF
ALL_ONES_F64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000000000000000) else val


class LDSMemory:
    """Model of Local Data Share memory storage with 64-bit qword access."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_bytes}")
        self.size_bytes = size_bytes
        self.mem: bytearray = bytearray(size_bytes)

    def read_qword(self, byte_addr: int) -> int:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        return struct.unpack_from("<Q", self.mem, idx)[0]

    def write_qword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        struct.pack_into("<Q", self.mem, idx, u64(val))



def compute_and_zero_u64(a: int, b: int) -> int:
    """Calculate whether 64-bit bitwise AND produces zero."""
    return 1 if ((u64(a) & u64(b)) == 0) else 0


def compute_and_sign_u64(a: int, b: int) -> int:
    """Calculate sign bit (bit 63) of 64-bit bitwise AND result."""
    return 1 if bool((u64(a) & u64(b)) & 0x8000000000000000) else 0


def compute_and_popcount_u64(a: int, b: int) -> int:
    """Calculate population count of 64-bit bitwise AND result."""
    return bin(u64(a) & u64(b)).count("1")


def bits_to_f64(bits: int) -> float:
    """Interpret 64-bit unsigned integer as IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Pack IEEE 754 double-precision float into 64-bit unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    """Test if 64-bit bit pattern represents an IEEE 754 double NaN."""
    return (bits & 0x7FF0000000000000) == 0x7FF0000000000000 and (bits & 0x000FFFFFFFFFFFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    """Test if 64-bit bit pattern represents an IEEE 754 double subnormal float."""
    return (bits & 0x7FF0000000000000) == 0 and (bits & 0x000FFFFFFFFFFFFF) != 0


def ref_ds_and_rtn_b64(
    mem: LDSMemory,
    vaddr: int,
    vdata: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_and_rtn_b64.

    Atomically computes:
      effective_byte = (vaddr + offset) mod 2^32
      old_val = LDS[effective_byte]
      LDS[effective_byte] = old_val & vdata
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    old_val = mem.read_qword(effective_byte)
    new_val = u64(old_val & u64(vdata))
    mem.write_qword(effective_byte, new_val)
    return old_val


class TestLdsAndRtnB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_bytes=4096)

    def test_basic_and_and_return(self) -> None:
        # Pre-seed memory at byte 16 with 0xFF00FF00FF00FF00
        self.mem.write_qword(16, 0xFF00FF00FF00FF00)
        # vaddr = 16, AND with 0x0F0F0F0F0F0F0F0F
        ret = ref_ds_and_rtn_b64(self.mem, vaddr=16, vdata=0x0F0F0F0F0F0F0F0F, offset=0)
        # Old value returned
        self.assertEqual(ret, 0xFF00FF00FF00FF00)
        # Memory updated to 0x0F000F000F000F00
        self.assertEqual(self.mem.read_qword(16), 0x0F000F000F000F00)

    def test_and_with_zero_clears_memory_and_returns_old(self) -> None:
        self.mem.write_qword(32, 0x123456789ABCDEF0)
        ret = ref_ds_and_rtn_b64(self.mem, vaddr=32, vdata=0, offset=0)
        self.assertEqual(ret, 0x123456789ABCDEF0)
        self.assertEqual(self.mem.read_qword(32), 0)

    def test_and_with_all_ones_preserves_memory_and_returns_old(self) -> None:
        self.mem.write_qword(40, 0xDEADBEEFCAFEBABE)
        ret = ref_ds_and_rtn_b64(self.mem, vaddr=40, vdata=ALL_ONES_F64, offset=0)
        self.assertEqual(ret, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.mem.read_qword(40), 0xDEADBEEFCAFEBABE)

    def test_address_offset_and_8byte_alignment(self) -> None:
        # vaddr = 24, offset = 16 -> effective = 40 (aligned)
        self.mem.write_qword(40, 0xAAAA5555AAAA5555)
        # unaligned byte 43 with ~7 masking resolves to 40
        ret = ref_ds_and_rtn_b64(self.mem, vaddr=27, vdata=0x5555555555555555, offset=16)
        self.assertEqual(ret, 0xAAAA5555AAAA5555)
        self.assertEqual(self.mem.read_qword(40), 0x0000555500005555)



# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class DSOpcode(Enum):
    """Supported DS family opcodes."""
    DS_AND_RTN_B64 = "ds_and_rtn_b64"


class RegType(Enum):
    """Register file class."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegSingle:
    """Symbolic 32-bit register operand representation."""
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
class RegPair:
    """Symbolic 64-bit register pair operand representation v[N:N+1]."""
    base_index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if not (0 <= self.base_index < DEFAULT_NUM_VGPRS - 1):
            raise ValueError(
                f"Register pair base index out of bounds: {self.reg_type.value}[{self.base_index}:{self.base_index+1}] "
                f"(expected 0 <= index < {DEFAULT_NUM_VGPRS - 1})"
            )

    @property
    def lo(self) -> int:
        return self.base_index

    @property
    def hi(self) -> int:
        return self.base_index + 1

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}[{self.lo}:{self.hi}]"


@dataclass(frozen=True)
class DSAndRtnB64Instruction:
    """Symbolic representation of ds_and_rtn_b64 instruction."""
    dst: RegPair
    addr: RegSingle
    data: RegPair
    offset: int = 0
    gds: bool = False
    opcode: DSOpcode = DSOpcode.DS_AND_RTN_B64

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination operand must be a VGPR pair")
        if self.addr.reg_type != RegType.VGPR:
            raise TypeError("Address operand must be a VGPR")
        if self.data.reg_type != RegType.VGPR:
            raise TypeError("Data operand must be a VGPR pair")
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
    """Architectural state for scalar execution context with 64-bit VGPR pair access."""

    def __init__(self, lds_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        self.vgprs: list[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_bytes=lds_bytes)

    def get_vgpr(self, index: int) -> int:
        return self.vgprs[index]

    def set_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = u32(val)

    def get_vgpr64(self, reg: RegPair) -> int:
        lo = self.vgprs[reg.lo]
        hi = self.vgprs[reg.hi]
        return u64(lo | (hi << 32))

    def set_vgpr64(self, reg: RegPair, val: int) -> None:
        val = u64(val)
        self.vgprs[reg.lo] = u32(val)
        self.vgprs[reg.hi] = u32(val >> 32)

    def get_sgpr(self, index: int) -> int:
        return self.sgprs[index]

    def set_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = u32(val)


class ScalarInterpreter:
    """Deterministic scalar interpreter for ds_and_rtn_b64."""

    def execute(self, instr: DSAndRtnB64Instruction, state: ScalarMachineState) -> int:
        """Executes a scalar ds_and_rtn_b64 instruction."""
        if instr.opcode != DSOpcode.DS_AND_RTN_B64:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        vaddr = state.get_vgpr(instr.addr.index)
        vdata = state.get_vgpr64(instr.data)
        old_val = ref_ds_and_rtn_b64(state.lds, vaddr=vaddr, vdata=vdata, offset=instr.offset)
        state.set_vgpr64(instr.dst, old_val)
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
        lds_bytes: int = DEFAULT_LDS_SIZE_BYTES,
    ) -> None:
        if num_lanes not in (32, 64):
            raise ValueError(f"Supported lane counts are 32 or 64, got: {num_lanes}")
        self.num_lanes: int = num_lanes
        mask_limit = MASK32 if num_lanes == 32 else MASK64
        self.exec_mask: int = exec_mask & mask_limit
        self.vgprs: list[list[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = LDSMemory(size_bytes=lds_bytes)

    def is_lane_active(self, lane: int) -> bool:
        if not (0 <= lane < self.num_lanes):
            return False
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx]

    def set_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.vgprs[lane][reg_idx] = u32(val)

    def get_vgpr64(self, lane: int, reg: RegPair) -> int:
        lo = self.vgprs[lane][reg.lo]
        hi = self.vgprs[lane][reg.hi]
        return u64(lo | (hi << 32))

    def set_vgpr64(self, lane: int, reg: RegPair, val: int) -> None:
        val = u64(val)
        self.vgprs[lane][reg.lo] = u32(val)
        self.vgprs[lane][reg.hi] = u32(val >> 32)


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane execution and EXEC masking."""

    def execute(self, instr: DSAndRtnB64Instruction, state: SIMDState) -> list[tuple[int, int]]:
        """Executes ds_and_rtn_b64 across SIMD lanes."""
        if instr.opcode != DSOpcode.DS_AND_RTN_B64:
            raise NotImplementedError(f"Unsupported opcode: {instr.opcode}")

        latched_lanes: list[tuple[int, int, int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                vaddr = state.get_vgpr(lane, instr.addr.index)
                vdata = state.get_vgpr64(lane, instr.data)
                latched_lanes.append((lane, vaddr, vdata))

        results: list[tuple[int, int]] = []
        for lane, vaddr, vdata in latched_lanes:
            old_val = ref_ds_and_rtn_b64(state.lds, vaddr=vaddr, vdata=vdata, offset=instr.offset)
            state.set_vgpr64(lane, instr.dst, old_val)
            results.append((lane, old_val))

        return results


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_opcode_enumeration(self) -> None:
        self.assertEqual(DSOpcode.DS_AND_RTN_B64.value, "ds_and_rtn_b64")

    def test_register_operand_validation(self) -> None:
        v0 = RegSingle(index=0, reg_type=RegType.VGPR)
        self.assertEqual(v0.name, "v0")

        vpair = RegPair(base_index=0, reg_type=RegType.VGPR)
        self.assertEqual(vpair.name, "v[0:1]")
        self.assertEqual(vpair.lo, 0)
        self.assertEqual(vpair.hi, 1)

        with self.assertRaises(ValueError):
            RegSingle(index=-1)
        with self.assertRaises(ValueError):
            RegSingle(index=256)
        with self.assertRaises(ValueError):
            RegPair(base_index=255)  # 255 has no valid 256 hi pair

    def test_instruction_validation(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        instr = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=64)
        self.assertEqual(instr.offset, 64)
        self.assertFalse(instr.gds)

        with self.assertRaises(ValueError):
            DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=-1)
        with self.assertRaises(ValueError):
            DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=65536)

        s_single = RegSingle(0, reg_type=RegType.SGPR)
        s_pair = RegPair(0, reg_type=RegType.SGPR)
        with self.assertRaises(TypeError):
            DSAndRtnB64Instruction(dst=s_pair, addr=addr, data=data)
        with self.assertRaises(TypeError):
            DSAndRtnB64Instruction(dst=dst, addr=s_single, data=data)
        with self.assertRaises(TypeError):
            DSAndRtnB64Instruction(dst=dst, addr=addr, data=s_pair)

    def test_disasm_rendering(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        instr_basic = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data)
        self.assertEqual(instr_basic.disasm(), "ds_and_rtn_b64 v[0:1], v2, v[4:5]")

        instr_offset = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=32)
        self.assertEqual(instr_offset.disasm(), "ds_and_rtn_b64 v[0:1], v2, v[4:5] offset:32")

        instr_gds = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=64, gds=True)
        self.assertEqual(instr_gds.disasm(), "ds_and_rtn_b64 v[0:1], v2, v[4:5] offset:64 gds")

    def test_structural_equality_and_immutability(self) -> None:
        dst = RegPair(0)
        addr = RegSingle(2)
        data = RegPair(4)

        i1 = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=16)
        i2 = DSAndRtnB64Instruction(dst=dst, addr=addr, data=data, offset=16)
        self.assertEqual(i1, i2)
        self.assertEqual(hash(i1), hash(i2))

        with self.assertRaises(Exception):
            i1.offset = 32  # type: ignore


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.state = ScalarMachineState(lds_bytes=4096)
        self.interp = ScalarInterpreter()

    def test_scalar_and_and_return(self) -> None:
        # Pre-seed LDS qword at byte 32 with 0x0123456789ABCDEF
        self.state.lds.write_qword(32, 0x0123456789ABCDEF)
        self.state.set_vgpr(2, 32)  # vaddr = 32
        self.state.set_vgpr64(RegPair(4), 0x00FF00FF00FF00FF)  # vdata

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
            offset=0,
        )
        old_val = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 0x0123456789ABCDEF)
        self.assertEqual(self.state.get_vgpr64(RegPair(0)), 0x0123456789ABCDEF)
        self.assertEqual(self.state.lds.read_qword(32), 0x0123456789ABCDEF & 0x00FF00FF00FF00FF)

    def test_scalar_offset_handling(self) -> None:
        self.state.lds.write_qword(64, 0xAAAAAAAAAAAAAAAA)
        self.state.set_vgpr(1, 48)  # vaddr = 48, offset = 16 -> 64
        self.state.set_vgpr64(RegPair(6), 0x5555555555555555)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(4),
            addr=RegSingle(1),
            data=RegPair(6),
            offset=16,
        )
        old_val = self.interp.execute(instr, self.state)

        self.assertEqual(old_val, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(self.state.get_vgpr64(RegPair(4)), 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(self.state.lds.read_qword(64), 0)

    def test_scalar_determinism(self) -> None:
        runs: list[tuple[int, int]] = []
        for _ in range(50):
            st = ScalarMachineState(lds_bytes=2048)
            st.lds.write_qword(16, 0xFEDCBA9876543210)
            st.set_vgpr(0, 16)
            st.set_vgpr64(RegPair(2), 0x0F0F0F0F0F0F0F0F)
            instr = DSAndRtnB64Instruction(
                dst=RegPair(4),
                addr=RegSingle(0),
                data=RegPair(2),
            )
            ret = self.interp.execute(instr, st)
            runs.append((ret, st.lds.read_qword(16)))

        self.assertTrue(all(r == (0xFEDCBA9876543210, 0x0E0C0A0806040200) for r in runs))


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_execution(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=MASK32, lds_bytes=1024)
        for i in range(32):
            qword_addr = i * 8
            state.lds.write_qword(qword_addr, 0xF0F0F0F0F0F0F0F0 | i)
            state.set_vgpr(i, 2, qword_addr)
            state.set_vgpr64(i, RegPair(4), 0x0F0F0F0F0F0F0F0F)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        results = self.interp.execute(instr, state)

        self.assertEqual(len(results), 32)
        for i in range(32):
            expected_old = 0xF0F0F0F0F0F0F0F0 | i
            self.assertEqual(state.get_vgpr64(i, RegPair(0)), expected_old)
            self.assertEqual(state.lds.read_qword(i * 8), expected_old & 0x0F0F0F0F0F0F0F0F)

    def test_wave64_divergence_and_exec_masking(self) -> None:
        alt_mask = 0x5555555555555555
        state = SIMDState(num_lanes=64, exec_mask=alt_mask, lds_bytes=2048)

        for lane in range(64):
            qword_addr = lane * 8
            state.lds.write_qword(qword_addr, 0x1111222233334444 + lane)
            state.set_vgpr(lane, 2, qword_addr)
            state.set_vgpr64(lane, RegPair(4), 0xFFFFFFFF00000000)
            state.set_vgpr64(lane, RegPair(0), 0xDEADBEEFCAFEBABE)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        results = self.interp.execute(instr, state)

        self.assertEqual(len(results), 32)

        for lane in range(64):
            qword_addr = lane * 8
            orig_val = 0x1111222233334444 + lane
            if (lane % 2) == 0:
                self.assertEqual(state.get_vgpr64(lane, RegPair(0)), orig_val)
                self.assertEqual(state.lds.read_qword(qword_addr), orig_val & 0xFFFFFFFF00000000)
            else:
                self.assertEqual(state.get_vgpr64(lane, RegPair(0)), 0xDEADBEEFCAFEBABE)
                self.assertEqual(state.lds.read_qword(qword_addr), orig_val)

    def test_simd_lane_collision_serialization(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0x7, lds_bytes=1024)
        target_addr = 64
        initial_val = 0xFFFFFFFFFFFFFFFF
        state.lds.write_qword(target_addr, initial_val)

        # Lane 0: and with 0xFF00FF00FF00FF00
        # Lane 1: and with 0x0FF00FF00FF00FF0
        # Lane 2: and with 0x00FF00FF00FF00FF
        masks = [0xFF00FF00FF00FF00, 0x0FF00FF00FF00FF0, 0x00FF00FF00FF00FF]
        for lane in range(3):
            state.set_vgpr(lane, 2, target_addr)
            state.set_vgpr64(lane, RegPair(4), masks[lane])

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        results = self.interp.execute(instr, state)

        # Expected sequential results:
        # Lane 0 sees initial_val
        # Lane 1 sees initial_val & masks[0]
        # Lane 2 sees initial_val & masks[0] & masks[1]
        v0 = initial_val
        v1 = v0 & masks[0]
        v2 = v1 & masks[1]
        final_mem = v2 & masks[2]

        self.assertEqual(results, [(0, v0), (1, v1), (2, v2)])
        self.assertEqual(state.get_vgpr64(0, RegPair(0)), v0)
        self.assertEqual(state.get_vgpr64(1, RegPair(0)), v1)
        self.assertEqual(state.get_vgpr64(2, RegPair(0)), v2)
        self.assertEqual(state.lds.read_qword(target_addr), final_mem)


class TestRegisterAliasingAndInPlacePreservation(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_dst_aliases_data(self) -> None:
        # vdst == vdata: v[0:1] is both destination and data operand
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(16, 0x123456789ABCDEF0)
        state.set_vgpr(2, 16)  # vaddr = 16
        state.set_vgpr64(RegPair(0), 0x0000FFFF0000FFFF)  # vdata in v[0:1]

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(0),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 0x123456789ABCDEF0)
        self.assertEqual(state.get_vgpr64(RegPair(0)), 0x123456789ABCDEF0)
        self.assertEqual(state.lds.read_qword(16), 0x123456789ABCDEF0 & 0x0000FFFF0000FFFF)

    def test_dst_aliases_addr(self) -> None:
        # vdst low register aliases vaddr (e.g. vaddr is v0, vdst is v[0:1])
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(32, 0xFEDCBA9876543210)
        state.set_vgpr(0, 32)  # vaddr = 32
        state.set_vgpr64(RegPair(2), 0x0F0F0F0F0F0F0F0F)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(0),
            data=RegPair(2),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 0xFEDCBA9876543210)
        self.assertEqual(state.get_vgpr64(RegPair(0)), 0xFEDCBA9876543210)
        self.assertEqual(state.lds.read_qword(32), 0xFEDCBA9876543210 & 0x0F0F0F0F0F0F0F0F)

    def test_dst_aliases_addr_high(self) -> None:
        # vaddr is v1, vdst is v[0:1]
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(40, 0x777788889999AAAA)
        state.set_vgpr(1, 40)
        state.set_vgpr64(RegPair(2), 0x1111111111111111)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(1),
            data=RegPair(2),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 0x777788889999AAAA)
        self.assertEqual(state.get_vgpr64(RegPair(0)), 0x777788889999AAAA)
        self.assertEqual(state.lds.read_qword(40), 0x1111000011110000)

    def test_all_operands_overlapping(self) -> None:
        # vdst is v[0:1], vdata is v[0:1], vaddr is v0
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(0, 0x5555AAAA5555AAAA)
        state.set_vgpr64(RegPair(0), 0)  # address 0, mask 0

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(0),
            data=RegPair(0),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, 0x5555AAAA5555AAAA)
        self.assertEqual(state.get_vgpr64(RegPair(0)), 0x5555AAAA5555AAAA)
        self.assertEqual(state.lds.read_qword(0), 0)

    def test_unmodified_registers_preserved(self) -> None:
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(16, 0x1111)
        state.set_vgpr(2, 16)
        state.set_vgpr64(RegPair(4), 0x2222)
        for r in range(6, 16):
            state.set_vgpr(r, 0xE000 + r)
        for s in range(5):
            state.set_sgpr(s, 0xF000 + s)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        self.interp.execute(instr, state)

        for r in range(6, 16):
            self.assertEqual(state.get_vgpr(r), 0xE000 + r)
        for s in range(5):
            self.assertEqual(state.get_sgpr(s), 0xF000 + s)


class TestConditionCodesAndFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and bitwise status generation."""

    def test_flags_preserved_on_zero_and_negative_results(self) -> None:
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(8, 0x8000000000000000)
        state.set_vgpr(2, 8)
        state.set_vgpr64(RegPair(4), 0x8000000000000000)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        interp = ScalarInterpreter()
        old_val = interp.execute(instr, state)

        self.assertEqual(old_val, 0x8000000000000000)
        self.assertEqual(state.lds.read_qword(8), 0x8000000000000000)

    def test_mathematical_bitwise_status_oracles(self) -> None:
        self.assertEqual(compute_and_zero_u64(0xAAAA, 0x5555), 1)
        self.assertEqual(compute_and_zero_u64(0xAAAA, 0xAAAA), 0)

        self.assertEqual(compute_and_sign_u64(0x8000000000000000, 0xFFFFFFFFFFFFFFFF), 1)
        self.assertEqual(compute_and_sign_u64(0x7FFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), 0)

        self.assertEqual(compute_and_popcount_u64(0x0F, 0x07), 3)
        self.assertEqual(compute_and_popcount_u64(ALL_ONES_F64, ALL_ONES_F64), 64)


class TestIEEE754BitPatternHandling(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()

    def test_float_helper_utilities(self) -> None:
        self.assertTrue(is_f64_nan(QNAN_F64))
        self.assertTrue(is_f64_nan(SNAN_F64))
        self.assertFalse(is_f64_nan(POS_INF_F64))
        self.assertFalse(is_f64_nan(POS_ZERO_F64))

        self.assertTrue(is_f64_subnormal(MIN_SUBNORMAL_F64))
        self.assertTrue(is_f64_subnormal(MAX_SUBNORMAL_F64))
        self.assertFalse(is_f64_subnormal(MIN_NORMAL_F64))
        self.assertFalse(is_f64_subnormal(POS_ZERO_F64))

    def test_signed_zeros_handling(self) -> None:
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(16, NEG_ZERO_F64)
        state.set_vgpr(2, 16)
        state.set_vgpr64(RegPair(4), ALL_ONES_F64)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, NEG_ZERO_F64)
        self.assertEqual(state.get_vgpr64(RegPair(0)), NEG_ZERO_F64)
        self.assertEqual(state.lds.read_qword(16), NEG_ZERO_F64)

    def test_infinities_and_nans_without_quietening(self) -> None:
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(24, SNAN_F64)
        state.set_vgpr(2, 24)
        state.set_vgpr64(RegPair(4), 0x7FFFFFFFFFFFFFFF)  # Clear sign bit

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, SNAN_F64)
        self.assertEqual(state.lds.read_qword(24), SNAN_F64 & 0x7FFFFFFFFFFFFFFF)

    def test_subnormals_without_flushing_to_zero(self) -> None:
        state = ScalarMachineState(lds_bytes=1024)
        state.lds.write_qword(32, MIN_SUBNORMAL_F64)
        state.set_vgpr(2, 32)
        state.set_vgpr64(RegPair(4), ALL_ONES_F64)

        instr = DSAndRtnB64Instruction(
            dst=RegPair(0),
            addr=RegSingle(2),
            data=RegPair(4),
        )
        old_val = self.interp.execute(instr, state)

        self.assertEqual(old_val, MIN_SUBNORMAL_F64)
        self.assertEqual(state.lds.read_qword(32), MIN_SUBNORMAL_F64)


if __name__ == '__main__':
    unittest.main()
