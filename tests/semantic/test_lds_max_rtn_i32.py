from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_rtn_i32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Signed 32-bit Integer Maximum with Return:
Atomically compares a signed 32-bit integer operand with the value in LDS memory,
computes the signed maximum, updates LDS with the maximum, and returns the original
32-bit value in memory to the destination VGPR.

Syntax:
    ds_max_rtn_i32 vdst, vaddr, vdata0 [offset:offset16]

Architectural Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    s_old = s32(old_val)
    s_data = s32(vdata0)
    s_max = max(s_old, s_data)
    LDS[dword_index] = u32(s_max)
    vdst = old_val

Signed 32-bit Integer Properties:
    - Range: [-2^31, 2^31 - 1] ([-2147483648, 2147483647])
    - INT32_MIN = -2147483648 (0x80000000)
    - INT32_MAX = 2147483647 (0x7FFFFFFF)
    - Signed comparison: 0 > -1 (0 > 0xFFFFFFFF), 10 > -20 (10 > 0xFFFFFFEC)
"""

import math
import re
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

INT32_MIN: int = -2147483648
INT32_MAX: int = 2147483647


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer [-2^31, 2^31 - 1]."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_max_rtn_i32(
    current_mem: int,
    data_val: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_i32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_val = max(s32(current_mem), s32(data_val)) & 0xFFFFFFFF

    Returns:
        (old_val, new_val)
    """
    old_val = u32(current_mem)
    s_curr = s32(current_mem)
    s_data = s32(data_val)
    s_max = max(s_curr, s_data)
    new_val = u32(s_max)
    return old_val, new_val


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_max_rtn_i32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic signed 32-bit integer maximum with return on LDS.

        Returns:
            (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_val = ref_ds_max_rtn_i32(curr, vdata0)
        self.memory[idx] = new_val
        return old_val, new_val


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) atomic opcodes."""
    DS_MAX_RTN_I32 = "ds_max_rtn_i32"


class RegType(Enum):
    """Register file types."""
    VGPR = "v"
    SGPR = "s"


@dataclass(frozen=True)
class RegisterOperand:
    """Typed symbolic register operand representation."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        max_limit = DEFAULT_NUM_VGPRS if self.reg_type == RegType.VGPR else DEFAULT_NUM_SGPRS
        if not (0 <= self.index < max_limit):
            raise ValueError(
                f"Register index out of range for {self.reg_type.name}: {self.index} (limit: {max_limit})"
            )

    def __str__(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class DSMaxRtnI32Instruction:
    """Symbolic AST node representing a ds_max_rtn_i32 instruction.

    Format:
        ds_max_rtn_i32 vdst, vaddr, vdata0 [offset:offset16]
    """
    vdst: RegisterOperand
    vaddr: RegisterOperand
    vdata0: RegisterOperand
    offset: int = 0
    opcode: DSOpcode = DSOpcode.DS_MAX_RTN_I32

    def __post_init__(self) -> None:
        if self.vdst.reg_type != RegType.VGPR:
            raise ValueError(f"vdst must be a VGPR, got {self.vdst.reg_type}")
        if self.vaddr.reg_type != RegType.VGPR:
            raise ValueError(f"vaddr must be a VGPR, got {self.vaddr.reg_type}")
        if self.vdata0.reg_type != RegType.VGPR:
            raise ValueError(f"vdata0 must be a VGPR, got {self.vdata0.reg_type}")
        if not (0 <= self.offset <= 65535):
            raise ValueError(f"offset out of range [0, 65535]: {self.offset}")

    def to_assembly(self) -> str:
        """Serialize AST node to standard assembly string."""
        asm = f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata0}"
        if self.offset != 0:
            asm += f" offset:{self.offset}"
        return asm

    @classmethod
    def parse(cls, asm: str) -> DSMaxRtnI32Instruction:
        """Parse an assembly string into a DSMaxRtnI32Instruction AST node."""
        cleaned = re.sub(r"[;#].*$", "", asm).strip()
        if not cleaned:
            raise ValueError("Empty assembly string")

        tokens = cleaned.split()
        if not tokens:
            raise ValueError("Empty instruction")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_MAX_RTN_I32.value:
            raise ValueError(f"Unsupported opcode '{tokens[0]}', expected '{DSOpcode.DS_MAX_RTN_I32.value}'")

        rest = " ".join(tokens[1:])
        parts = [p.strip() for p in rest.split(",") if p.strip()]

        if len(parts) < 3:
            raise ValueError(f"Insufficient operands for {op_str}: expected at least 3, got {len(parts)}")

        vdst_str = parts[0]
        vaddr_str = parts[1]

        # Third part might contain vdata0 and optionally offset:offset separated by space or attached
        vdata_and_mods = parts[2].split()
        vdata_str = vdata_and_mods[0]

        offset_val = 0
        mod_tokens = vdata_and_mods[1:] + parts[3:]
        for m in mod_tokens:
            m_lower = m.lower()
            if m_lower.startswith("offset:"):
                raw_offset = m[len("offset:"):]
                try:
                    offset_val = int(raw_offset, 0)
                except ValueError:
                    raise ValueError(f"Invalid integer offset value: '{raw_offset}'")
            else:
                raise ValueError(f"Unknown modifier or unexpected extra token: '{m}'")

        def parse_vgpr(reg_s: str) -> RegisterOperand:
            reg_s = reg_s.strip()
            if not (reg_s.startswith("v") or reg_s.startswith("V")):
                raise ValueError(f"Expected VGPR starting with 'v', got '{reg_s}'")
            try:
                idx = int(reg_s[1:])
            except ValueError:
                raise ValueError(f"Invalid VGPR index: '{reg_s}'")
            return RegisterOperand(idx, RegType.VGPR)

        vdst = parse_vgpr(vdst_str)
        vaddr = parse_vgpr(vaddr_str)
        vdata0 = parse_vgpr(vdata_str)

        return cls(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=offset_val)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    """Scalar thread architectural execution state (VGPRs, SGPRs, and LDS reference)."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.m0: int = 0
        self.vcc: int = 0
        self.scc: int = 0
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = u32(val)


@dataclass(frozen=True)
class DSAtomicConditionCodes:
    """Condition codes evaluating the returned old value and signed max transition."""
    zero: bool
    negative: bool
    positive: bool
    updated: bool
    is_int32_max: bool


@dataclass(frozen=True)
class DSMemoryStatusFlags:
    """Status flags for LDS address wrapping, bank mapping, and alignment."""
    address_wrapped: bool
    bank_index: int
    unaligned_access: bool


def evaluate_atomic_max_condition_codes(old_val: int, data_val: int, new_val: int) -> DSAtomicConditionCodes:
    """Evaluate architectural condition codes for a signed 32-bit maximum operation."""
    s_old = s32(old_val)
    return DSAtomicConditionCodes(
        zero=(s_old == 0),
        negative=(s_old < 0),
        positive=(s_old > 0),
        updated=(u32(new_val) != u32(old_val)),
        is_int32_max=(s_old == INT32_MAX or s32(new_val) == INT32_MAX),
    )


def evaluate_lds_status_flags(vaddr: int, offset: int, capacity_bytes: int = LDS_SIZE_BYTES) -> DSMemoryStatusFlags:
    """Evaluate memory subsystem status flags for LDS address computation."""
    raw_addr = vaddr + (offset & OFFSET16_MASK)
    eff_addr = raw_addr & MASK32
    return DSMemoryStatusFlags(
        address_wrapped=(raw_addr >= capacity_bytes),
        bank_index=calc_lds_bank(eff_addr),
        unaligned_access=bool(eff_addr & 0x3),
    )


class ScalarDSInterpreter:
    """Deterministic scalar execution interpreter for Data Share (DS) instructions."""

    def __init__(self) -> None:
        self.last_condition_codes: Optional[DSAtomicConditionCodes] = None
        self.last_status_flags: Optional[DSMemoryStatusFlags] = None

    def execute(self, state: ScalarThreadState, inst: DSMaxRtnI32Instruction) -> int:
        """Execute ds_max_rtn_i32 on the scalar thread state.

        1. Read address from inst.vaddr VGPR.
        2. Read data from inst.vdata0 VGPR.
        3. Perform atomic signed maximum on LDS at eff_addr = vaddr + offset.
        4. Write original memory value into inst.vdst VGPR.
        5. Evaluate condition codes and memory status flags.
        6. Returns the returned old memory value.
        """
        vaddr = state.get_vgpr(inst.vaddr.index)
        vdata = state.get_vgpr(inst.vdata0.index)
        old_val, new_val = state.lds.atomic_max_rtn_i32(vaddr, vdata, inst.offset)
        state.set_vgpr(inst.vdst.index, old_val)

        self.last_condition_codes = evaluate_atomic_max_condition_codes(old_val, vdata, new_val)
        self.last_status_flags = evaluate_lds_status_flags(vaddr, inst.offset, state.lds.size_dwords * 4)
        return old_val


# ==============================================================================
# SIMD Wave Execution and EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """SIMD wave architectural execution state supporting Wave32 and Wave64 execution modes."""

    def __init__(
        self,
        wave_size: int = 32,
        exec_mask: Optional[int] = None,
        lds: Optional[LDSMemory] = None,
    ) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"wave_size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.full_mask: int = (1 << wave_size) - 1
        self.exec_mask: int = exec_mask if exec_mask is not None else self.full_mask
        self.lane_vgprs: List[List[int]] = [[0] * DEFAULT_NUM_VGPRS for _ in range(wave_size)]
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_lane_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.lane_vgprs[lane][reg_idx]

    def set_lane_vgpr(self, lane: int, reg_idx: int, val: int) -> None:
        self.lane_vgprs[lane][reg_idx] = u32(val)


def detect_lds_bank_conflicts(eff_addrs: List[int]) -> Dict[int, List[int]]:
    """Analyze a list of effective byte addresses from active lanes to detect bank conflicts.

    In RDNA2 LDS architecture:
    - LDS has 32 banks (bank = (eff_addr >> 2) % 32).
    - If multiple active lanes access the same bank with different dword addresses,
      hardware serializes the accesses (bank conflict).

    Returns:
        Mapping from bank index (0..31) to list of conflicting lane indices.
    """
    bank_to_dwords: Dict[int, Dict[int, List[int]]] = {}
    for lane, addr in enumerate(eff_addrs):
        idx = (addr >> 2) & (LDS_SIZE_DWORDS - 1)
        bank = idx % LDS_BANK_COUNT
        if bank not in bank_to_dwords:
            bank_to_dwords[bank] = {}
        if idx not in bank_to_dwords[bank]:
            bank_to_dwords[bank][idx] = []
        bank_to_dwords[bank][idx].append(lane)

    conflicts: Dict[int, List[int]] = {}
    for bank, dword_map in bank_to_dwords.items():
        if len(dword_map) > 1:
            lanes: List[int] = []
            for lane_list in dword_map.values():
                lanes.extend(lane_list)
            conflicts[bank] = sorted(lanes)
    return conflicts


class SIMDDSInterpreter:
    """SIMD wave execution interpreter handling lane divergence, EXEC masking, and atomics."""

    def execute(
        self,
        state: SIMDWaveState,
        inst: DSMaxRtnI32Instruction,
    ) -> Dict[int, int]:
        """Execute ds_max_rtn_i32 across all active SIMD lanes.

        In AMD RDNA2:
        - Only active lanes (where (exec_mask >> lane) & 1 == 1) perform memory operations and update VGPRs.
        - Inactive lanes do not touch LDS and their vdst VGPRs remain unchanged.
        - If multiple active lanes target the exact same LDS dword, the atomic operations
          are serialized deterministically in lane order.

        Returns:
            Dictionary mapping active lane index to the returned old 32-bit value.
        """
        results: Dict[int, int] = {}
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                continue
            vaddr = state.get_lane_vgpr(lane, inst.vaddr.index)
            vdata0 = state.get_lane_vgpr(lane, inst.vdata0.index)
            old_val, _ = state.lds.atomic_max_rtn_i32(vaddr, vdata0, inst.offset)
            state.set_lane_vgpr(lane, inst.vdst.index, old_val)
            results[lane] = old_val
        return results


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestLDSMaxRtnI32OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for signed 32-bit integer oracle and LDS architectural model."""

    def test_oracle_signed_positive_vs_positive(self) -> None:
        old_val, new_val = ref_ds_max_rtn_i32(10, 20)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 20)

        old_val2, new_val2 = ref_ds_max_rtn_i32(100, 50)
        self.assertEqual(old_val2, 100)
        self.assertEqual(new_val2, 100)

    def test_oracle_signed_positive_vs_negative(self) -> None:
        # In signed: 10 > -20. In unsigned: 10 < 0xFFFFFFEC.
        old_val, new_val = ref_ds_max_rtn_i32(10, u32(-20))
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 10)

        # Negative in memory, positive in operand
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-50), 5)
        self.assertEqual(old_val2, u32(-50))
        self.assertEqual(new_val2, 5)

    def test_oracle_signed_negative_vs_negative(self) -> None:
        # -10 > -30
        old_val, new_val = ref_ds_max_rtn_i32(u32(-30), u32(-10))
        self.assertEqual(old_val, u32(-30))
        self.assertEqual(s32(new_val), -10)

        # -5 > -100
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-5), u32(-100))
        self.assertEqual(old_val2, u32(-5))
        self.assertEqual(s32(new_val2), -5)

    def test_oracle_zero_vs_negative(self) -> None:
        # 0 > -1 (0x00000000 > 0xFFFFFFFF in signed i32)
        old_val, new_val = ref_ds_max_rtn_i32(0, u32(-1))
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        # -1 in memory, 0 in data
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(-1), 0)
        self.assertEqual(old_val2, u32(-1))
        self.assertEqual(new_val2, 0)

    def test_oracle_int32_extremes(self) -> None:
        # INT32_MIN (-2147483648) vs INT32_MAX (2147483647)
        old_val, new_val = ref_ds_max_rtn_i32(u32(INT32_MIN), u32(INT32_MAX))
        self.assertEqual(old_val, u32(INT32_MIN))
        self.assertEqual(new_val, u32(INT32_MAX))

        # INT32_MIN vs 0
        old_val2, new_val2 = ref_ds_max_rtn_i32(u32(INT32_MIN), 0)
        self.assertEqual(old_val2, u32(INT32_MIN))
        self.assertEqual(new_val2, 0)

    def test_oracle_equal_operands(self) -> None:
        old_val, new_val = ref_ds_max_rtn_i32(42, 42)
        self.assertEqual(old_val, 42)
        self.assertEqual(new_val, 42)

    def test_lds_memory_basic_operations(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0x100, 0x12345678)
        self.assertEqual(lds.read_dword(0x100), 0x12345678)

    def test_lds_dword_alignment(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0x10, 0xDEADBEEF)
        # Low 2 bits (bytes 0, 1, 2, 3) must access the exact same dword
        self.assertEqual(lds.read_dword(0x10), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x11), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x12), 0xDEADBEEF)
        self.assertEqual(lds.read_dword(0x13), 0xDEADBEEF)

    def test_lds_memory_capacity_wrapping(self) -> None:
        # 1024 dwords = 4096 bytes capacity
        lds = LDSMemory(size_dwords=1024)
        lds.write_dword(0, 0xCAFEBABE)
        self.assertEqual(lds.read_dword(4096), 0xCAFEBABE)
        self.assertEqual(lds.read_dword(8192), 0xCAFEBABE)

    def test_lds_atomic_max_rtn_i32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        # Initialize memory with -10
        lds.write_dword(0x40, u32(-10))

        # Atomic max with 5 -> old should be -10, new memory should be 5
        old_val, new_val = lds.atomic_max_rtn_i32(0x40, 5)
        self.assertEqual(old_val, u32(-10))
        self.assertEqual(new_val, 5)
        self.assertEqual(lds.read_dword(0x40), 5)

        # Atomic max with -20 -> old should be 5, new memory should remain 5
        old_val2, new_val2 = lds.atomic_max_rtn_i32(0x40, u32(-20))
        self.assertEqual(old_val2, 5)
        self.assertEqual(new_val2, 5)
        self.assertEqual(lds.read_dword(0x40), 5)

    def test_lds_bank_distribution(self) -> None:
        # Banks cycle through 0..31 for consecutive dwords
        for i in range(64):
            expected_bank = i % 32
            eff_addr = i * 4
            self.assertEqual(calc_lds_bank(eff_addr), expected_bank)


class TestDSMaxRtnI32AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_operand_valid(self) -> None:
        vgpr = RegisterOperand(0, RegType.VGPR)
        self.assertEqual(str(vgpr), "v0")
        self.assertEqual(vgpr.index, 0)
        self.assertEqual(vgpr.reg_type, RegType.VGPR)

        vgpr_max = RegisterOperand(255, RegType.VGPR)
        self.assertEqual(str(vgpr_max), "v255")

    def test_register_operand_out_of_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(106, RegType.SGPR)

    def test_instruction_construction_and_to_assembly(self) -> None:
        vdst = RegisterOperand(0, RegType.VGPR)
        vaddr = RegisterOperand(1, RegType.VGPR)
        vdata0 = RegisterOperand(2, RegType.VGPR)

        inst1 = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0)
        self.assertEqual(inst1.to_assembly(), "ds_max_rtn_i32 v0, v1, v2")

        inst2 = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=128)
        self.assertEqual(inst2.to_assembly(), "ds_max_rtn_i32 v0, v1, v2 offset:128")

    def test_instruction_parsing_valid(self) -> None:
        inst1 = DSMaxRtnI32Instruction.parse("ds_max_rtn_i32 v0, v1, v2")
        self.assertEqual(inst1.vdst.index, 0)
        self.assertEqual(inst1.vaddr.index, 1)
        self.assertEqual(inst1.vdata0.index, 2)
        self.assertEqual(inst1.offset, 0)

        inst2 = DSMaxRtnI32Instruction.parse("ds_max_rtn_i32 v10, v20, v30 offset:256")
        self.assertEqual(inst2.vdst.index, 10)
        self.assertEqual(inst2.vaddr.index, 20)
        self.assertEqual(inst2.vdata0.index, 30)
        self.assertEqual(inst2.offset, 256)

        # Hex offset and case insensitivity
        inst3 = DSMaxRtnI32Instruction.parse("DS_MAX_RTN_I32 v5, v6, v7 offset:0x40")
        self.assertEqual(inst3.offset, 64)

    def test_instruction_roundtrip(self) -> None:
        vdst = RegisterOperand(4, RegType.VGPR)
        vaddr = RegisterOperand(8, RegType.VGPR)
        vdata0 = RegisterOperand(12, RegType.VGPR)
        inst = DSMaxRtnI32Instruction(vdst=vdst, vaddr=vaddr, vdata0=vdata0, offset=60)
        parsed = DSMaxRtnI32Instruction.parse(inst.to_assembly())
        self.assertEqual(inst, parsed)


class TestDSMaxRtnI32ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_signed_positive_update(self) -> None:
        # Memory initially -50, operand 10 -> memory becomes 10, vdst receives -50
        addr = 0x20
        self.lds.write_dword(addr, u32(-50))
        self.state.set_vgpr(1, addr)         # vaddr
        self.state.set_vgpr(2, 10)           # vdata0
        self.state.set_vgpr(0, 0xCAFEBABE)   # vdst initial canary

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, u32(-50))
        self.assertEqual(self.state.get_vgpr(0), u32(-50))
        self.assertEqual(self.lds.read_dword(addr), 10)

    def test_scalar_execution_signed_no_update(self) -> None:
        # Memory initially 100, operand -200 -> memory remains 100, vdst receives 100
        addr = 0x80
        self.lds.write_dword(addr, 100)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, u32(-200))

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 100)
        self.assertEqual(self.state.get_vgpr(0), 100)
        self.assertEqual(self.lds.read_dword(addr), 100)

    def test_scalar_execution_with_offset(self) -> None:
        # Address in VGPR is 0x40, offset is 0x10 -> effective address is 0x50
        base_addr = 0x40
        offset = 0x10
        eff_addr = base_addr + offset
        self.lds.write_dword(eff_addr, u32(-10))
        self.state.set_vgpr(1, base_addr)
        self.state.set_vgpr(2, 5)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=offset,
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, u32(-10))
        self.assertEqual(self.state.get_vgpr(0), u32(-10))
        self.assertEqual(self.lds.read_dword(eff_addr), 5)
        # Verify base address was untouched
        self.assertEqual(self.lds.read_dword(base_addr), 0)

    def test_scalar_execution_sequential_running_maximum(self) -> None:
        addr = 0x100
        self.lds.write_dword(addr, u32(-100))
        self.state.set_vgpr(1, addr)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        steps = [
            (-50, -100, -50),
            (-75, -50, -50),
            (0, -50, 0),
            (25, 0, 25),
            (10, 25, 25),
            (INT32_MAX, 25, INT32_MAX),
            (1000, INT32_MAX, INT32_MAX),
        ]

        for op_val, expected_old, expected_new in steps:
            self.state.set_vgpr(2, u32(op_val))
            ret = self.interp.execute(self.state, inst)
            self.assertEqual(s32(ret), expected_old)
            self.assertEqual(s32(self.state.get_vgpr(0)), expected_old)
            self.assertEqual(s32(self.lds.read_dword(addr)), expected_new)

    def test_scalar_unaligned_address_drop_low_bits(self) -> None:
        self.lds.write_dword(0x20, 77)
        # Low 2 bits of 0x23 are 0b11; should map to aligned 0x20
        self.state.set_vgpr(1, 0x23)
        self.state.set_vgpr(2, 99)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 77)
        self.assertEqual(self.lds.read_dword(0x20), 99)


class TestDSMaxRtnI32SIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=2048)
        self.interp = SIMDDSInterpreter()

    def test_wave32_independent_lanes(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        for l in range(32):
            addr = l * 4
            self.lds.write_dword(addr, u32(-100 + l))
            wave.set_lane_vgpr(l, 1, addr)
            wave.set_lane_vgpr(l, 2, u32(-50 + l))

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 32)

        for l in range(32):
            addr = l * 4
            expected_old = -100 + l
            expected_new = -50 + l  # since -50 + l > -100 + l
            self.assertEqual(s32(results[l]), expected_old)
            self.assertEqual(s32(wave.get_lane_vgpr(l, 0)), expected_old)
            self.assertEqual(s32(self.lds.read_dword(addr)), expected_new)

    def test_wave64_full_divergence(self) -> None:
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=self.lds)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        for l in range(64):
            addr = l * 4
            self.lds.write_dword(addr, u32(l))
            wave.set_lane_vgpr(l, 1, addr)
            wave.set_lane_vgpr(l, 2, u32(l + 10))

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 64)

        for l in range(64):
            addr = l * 4
            self.assertEqual(results[l], l)
            self.assertEqual(wave.get_lane_vgpr(l, 0), l)
            self.assertEqual(self.lds.read_dword(addr), l + 10)

    def test_wave32_exec_masking_partial(self) -> None:
        # Lanes 0..15 active (0x0000FFFF), lanes 16..31 inactive
        mask = 0x0000FFFF
        wave = SIMDWaveState(wave_size=32, exec_mask=mask, lds=self.lds)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        canary_vgpr = 0xDEADBEEF
        canary_mem = 0xCAFEBABE

        for l in range(32):
            addr = l * 4
            self.lds.write_dword(addr, canary_mem)
            wave.set_lane_vgpr(l, 0, canary_vgpr)
            wave.set_lane_vgpr(l, 1, addr)
            wave.set_lane_vgpr(l, 2, 42)

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 16)

        # Active lanes 0..15
        for l in range(16):
            addr = l * 4
            self.assertIn(l, results)
            self.assertEqual(results[l], canary_mem)
            self.assertEqual(wave.get_lane_vgpr(l, 0), canary_mem)
            # 42 vs canary_mem (0xCAFEBABE is negative in signed i32: -889275714), so max is 42
            self.assertEqual(self.lds.read_dword(addr), 42)

        # Inactive lanes 16..31: must remain untouched!
        for l in range(16, 32):
            addr = l * 4
            self.assertNotIn(l, results)
            self.assertEqual(wave.get_lane_vgpr(l, 0), canary_vgpr)
            self.assertEqual(self.lds.read_dword(addr), canary_mem)

    def test_wave32_exec_mask_zero(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0, lds=self.lds)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        for l in range(32):
            addr = l * 4
            self.lds.write_dword(addr, 100)
            wave.set_lane_vgpr(l, 0, 0x11111111)
            wave.set_lane_vgpr(l, 1, addr)
            wave.set_lane_vgpr(l, 2, 200)

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 0)

        for l in range(32):
            addr = l * 4
            self.assertEqual(wave.get_lane_vgpr(l, 0), 0x11111111)
            self.assertEqual(self.lds.read_dword(addr), 100)

    def test_simd_atomic_race_reduction_same_address(self) -> None:
        # All 32 lanes atomically target address 0x100
        target_addr = 0x100
        self.lds.write_dword(target_addr, u32(-1000))
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=self.lds)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        # Values written by lanes: varying signed values
        test_values = [
            -500, 10, -200, 50, -1, 100, -80, 200,
            -300, 0, -50, 150, -400, 75, -25, 250,
            -10, 5, -20, 15, -30, 25, -40, 35,
            -60, 45, -70, 55, -90, 65, -110, 70,
        ]
        expected_global_max = max([-1000] + test_values)  # should be 250

        for l in range(32):
            wave.set_lane_vgpr(l, 1, target_addr)
            wave.set_lane_vgpr(l, 2, u32(test_values[l]))

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 32)

        # Memory must end up with the overall signed maximum
        final_mem = s32(self.lds.read_dword(target_addr))
        self.assertEqual(final_mem, expected_global_max)

    def test_lds_bank_conflict_detection(self) -> None:
        # Bank conflict: Lane 0 targets dword 0 (bank 0), Lane 1 targets dword 32 (bank 0)
        eff_addrs_conflict = [0 * 4, 32 * 4]
        conflicts = detect_lds_bank_conflicts(eff_addrs_conflict)
        self.assertIn(0, conflicts)
        self.assertEqual(conflicts[0], [0, 1])

        # No conflict: Lane 0 and Lane 1 target identical dword (broadcast / atomics to same word)
        eff_addrs_same_dword = [16 * 4, 16 * 4]
        self.assertEqual(detect_lds_bank_conflicts(eff_addrs_same_dword), {})

        # No conflict: 32 lanes targeting consecutive dwords 0..31 (banks 0..31)
        eff_addrs_no_conflict = [l * 4 for l in range(32)]
        self.assertEqual(detect_lds_bank_conflicts(eff_addrs_no_conflict), {})


class TestDSMaxRtnI32RegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_alias_vdst_equals_vaddr(self) -> None:
        # ds_max_rtn_i32 v0, v0, v1 (vdst == vaddr)
        addr = 0x40
        self.lds.write_dword(addr, u32(-20))
        self.state.set_vgpr(0, addr)  # v0 is vaddr and vdst
        self.state.set_vgpr(1, 15)    # v1 is vdata0

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        # v0 must receive the old memory value (-20)
        self.assertEqual(ret, u32(-20))
        self.assertEqual(self.state.get_vgpr(0), u32(-20))
        # v1 must be preserved
        self.assertEqual(self.state.get_vgpr(1), 15)
        # Memory at 0x40 must be updated with max(-20, 15) = 15
        self.assertEqual(self.lds.read_dword(addr), 15)

    def test_alias_vdst_equals_vdata(self) -> None:
        # ds_max_rtn_i32 v1, v0, v1 (vdst == vdata0)
        addr = 0x50
        self.lds.write_dword(addr, 100)
        self.state.set_vgpr(0, addr)  # v0 is vaddr
        self.state.set_vgpr(1, 50)    # v1 is vdata0 and vdst

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(1, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        # v1 must receive the old memory value (100)
        self.assertEqual(ret, 100)
        self.assertEqual(self.state.get_vgpr(1), 100)
        # v0 must be preserved
        self.assertEqual(self.state.get_vgpr(0), addr)
        # Memory remains 100 (max(100, 50))
        self.assertEqual(self.lds.read_dword(addr), 100)

    def test_alias_vdst_equals_vaddr_equals_vdata(self) -> None:
        # ds_max_rtn_i32 v0, v0, v0 (vdst == vaddr == vdata0)
        addr = 0x10  # 16
        self.lds.write_dword(addr, u32(-5))
        self.state.set_vgpr(0, addr)  # v0 is vdst, vaddr, and vdata0

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(0, RegType.VGPR),
        )

        ret = self.interp.execute(self.state, inst)
        # Old memory value is -5
        self.assertEqual(ret, u32(-5))
        self.assertEqual(self.state.get_vgpr(0), u32(-5))
        # max(-5, 16) = 16
        self.assertEqual(self.lds.read_dword(addr), 16)

    def test_unreferenced_registers_preserved(self) -> None:
        addr = 0x20
        self.lds.write_dword(addr, 50)
        self.state.set_vgpr(0, 0)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 60)

        # Fill background registers with canaries
        for r in range(3, DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(r, 0x10000000 + r)
        for s in range(DEFAULT_NUM_SGPRS):
            self.state.set_sgpr(s, 0x20000000 + s)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        self.interp.execute(self.state, inst)

        # vdst received old value 50
        self.assertEqual(self.state.get_vgpr(0), 50)
        # vaddr and vdata0 preserved
        self.assertEqual(self.state.get_vgpr(1), addr)
        self.assertEqual(self.state.get_vgpr(2), 60)

        # All other registers untouched
        for r in range(3, DEFAULT_NUM_VGPRS):
            self.assertEqual(self.state.get_vgpr(r), 0x10000000 + r)
        for s in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.get_sgpr(s), 0x20000000 + s)

    def test_simd_multi_lane_register_aliasing(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        simd_interp = SIMDDSInterpreter()

        # Each lane has vdst == vaddr: v0 is dst and addr, v1 is vdata0
        for l in range(32):
            addr = l * 4
            self.lds.write_dword(addr, 50 + l)
            wave.set_lane_vgpr(l, 0, addr)
            wave.set_lane_vgpr(l, 1, 100)

        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(0, RegType.VGPR),
            vdata0=RegisterOperand(1, RegType.VGPR),
        )

        results = simd_interp.execute(wave, inst)

        for l in range(32):
            self.assertEqual(results[l], 50 + l)
            self.assertEqual(wave.get_lane_vgpr(l, 0), 50 + l)  # v0 received old value
            self.assertEqual(wave.get_lane_vgpr(l, 1), 100)     # v1 preserved
            self.assertEqual(self.lds.read_dword(l * 4), 100)   # Memory updated with max(50+l, 100) = 100


class TestDSMaxRtnI32ConditionCodesAndFlags(unittest.TestCase):
    """Phase 6: Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_condition_codes_zero_pos_neg(self) -> None:
        # Zero condition
        self.lds.write_dword(0, 0)
        self.state.set_vgpr(0, 0)
        self.state.set_vgpr(1, 0)
        self.state.set_vgpr(2, 5)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        self.interp.execute(self.state, inst)
        cc0 = self.interp.last_condition_codes
        self.assertIsNotNone(cc0)
        assert cc0 is not None
        self.assertTrue(cc0.zero)
        self.assertFalse(cc0.negative)
        self.assertFalse(cc0.positive)
        self.assertTrue(cc0.updated)  # memory changed from 0 to 5

        # Negative condition
        self.lds.write_dword(4, u32(-15))
        self.state.set_vgpr(1, 4)
        self.state.set_vgpr(2, u32(-20))
        self.interp.execute(self.state, inst)
        cc_neg = self.interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)
        self.assertFalse(cc_neg.updated)  # max(-15, -20) == -15, not updated

        # Positive condition
        self.lds.write_dword(8, 42)
        self.state.set_vgpr(1, 8)
        self.state.set_vgpr(2, 100)
        self.interp.execute(self.state, inst)
        cc_pos = self.interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)
        self.assertTrue(cc_pos.updated)

    def test_condition_codes_int32_max_saturation(self) -> None:
        self.lds.write_dword(0x10, 10)
        self.state.set_vgpr(1, 0x10)
        self.state.set_vgpr(2, u32(INT32_MAX))
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        self.interp.execute(self.state, inst)
        cc = self.interp.last_condition_codes
        self.assertIsNotNone(cc)
        assert cc is not None
        self.assertTrue(cc.is_int32_max)
        self.assertTrue(cc.updated)

    def test_status_flags_address_wrapping_and_alignment(self) -> None:
        # In-bounds aligned access: addr=0x20, offset=0x10 -> 0x30 < 4096 (1024 dwords * 4)
        self.state.set_vgpr(1, 0x20)
        self.state.set_vgpr(2, 5)
        inst_in = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0x10,
        )
        self.interp.execute(self.state, inst_in)
        flags_in = self.interp.last_status_flags
        self.assertIsNotNone(flags_in)
        assert flags_in is not None
        self.assertFalse(flags_in.address_wrapped)
        self.assertFalse(flags_in.unaligned_access)
        self.assertEqual(flags_in.bank_index, (0x30 >> 2) % 32)

        # Unaligned access: addr=0x21
        self.state.set_vgpr(1, 0x21)
        inst_unaligned = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0,
        )
        self.interp.execute(self.state, inst_unaligned)
        flags_unaligned = self.interp.last_status_flags
        self.assertIsNotNone(flags_unaligned)
        assert flags_unaligned is not None
        self.assertTrue(flags_unaligned.unaligned_access)

        # Address wrapped: addr=4096 (exceeds capacity of 1024 dwords * 4 = 4096 bytes)
        self.state.set_vgpr(1, 4096)
        self.interp.execute(self.state, inst_unaligned)
        flags_wrapped = self.interp.last_status_flags
        self.assertIsNotNone(flags_wrapped)
        assert flags_wrapped is not None
        self.assertTrue(flags_wrapped.address_wrapped)

    def test_carry_borrow_vcc_and_scc_isolation(self) -> None:
        # VCC and SCC must not be clobbered by LDS atomic instructions
        canary_vcc = 0x1234567887654321
        canary_scc = 1
        self.state.vcc = canary_vcc
        self.state.scc = canary_scc

        self.state.set_vgpr(1, 0x40)
        self.state.set_vgpr(2, u32(INT32_MIN))
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        self.interp.execute(self.state, inst)

        self.assertEqual(self.state.vcc, canary_vcc)
        self.assertEqual(self.state.scc, canary_scc)


class TestDSMaxRtnI32IEEE754BoundaryExtremes(unittest.TestCase):
    """Phase 7: Unit tests for IEEE 754 boundary extremes, subnormals, and NaN bit patterns."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=16384)  # full 64 KiB
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_signed_zero_bit_patterns_as_integers(self) -> None:
        # In IEEE 754: +0.0 is 0x00000000, -0.0 is 0x80000000.
        # In signed 32-bit int: 0x00000000 is 0, 0x80000000 is INT32_MIN (-2147483648).
        # max(0, -2147483648) = 0 (+0.0 bit pattern)
        addr = 0x100
        self.lds.write_dword(addr, 0x00000000)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x80000000)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 0x00000000)
        self.assertEqual(self.lds.read_dword(addr), 0x00000000)

        # Reverse: memory has -0.0 (INT32_MIN), operand is +0.0 (0)
        self.lds.write_dword(addr, 0x80000000)
        self.state.set_vgpr(2, 0x00000000)
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, 0x80000000)
        self.assertEqual(self.lds.read_dword(addr), 0x00000000)

    def test_signed_infinities_as_integers(self) -> None:
        # Single precision float:
        # +Inf: 0x7F800000 (+2139095040 in i32)
        # -Inf: 0xFF800000 (-8388608 in i32)
        pos_inf = 0x7F800000
        neg_inf = 0xFF800000

        # max(+Inf, -Inf) -> +Inf (since +2139095040 > -8388608)
        addr = 0x200
        self.lds.write_dword(addr, pos_inf)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, neg_inf)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, pos_inf)
        self.assertEqual(self.lds.read_dword(addr), pos_inf)

        # max(-Inf, 0) -> 0 (since 0 > -8388608)
        self.lds.write_dword(addr, neg_inf)
        self.state.set_vgpr(2, 0)
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, neg_inf)
        self.assertEqual(self.lds.read_dword(addr), 0)

        # max(-Inf, -1) -> -1 (since -1 > -8388608)
        self.lds.write_dword(addr, neg_inf)
        self.state.set_vgpr(2, 0xFFFFFFFF)
        ret3 = self.interp.execute(self.state, inst)
        self.assertEqual(ret3, neg_inf)
        self.assertEqual(self.lds.read_dword(addr), 0xFFFFFFFF)

    def test_nan_bit_patterns_unaltered_and_no_traps(self) -> None:
        # Float NaNs:
        # qNaN: 0x7FC00000 (+2143289344 in i32)
        # sNaN: 0x7F800001 (+2139095041 in i32)
        # neg_NaN: 0xFFC00000 (-4194304 in i32)
        qnan = 0x7FC00000
        snan = 0x7F800001
        neg_nan = 0xFFC00000
        pos_inf = 0x7F800000

        addr = 0x300
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        # In signed i32: 2143289344 (qNaN) > 2139095040 (+Inf)
        self.lds.write_dword(addr, pos_inf)
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, qnan)
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, pos_inf)
        self.assertEqual(self.lds.read_dword(addr), qnan)

        # sNaN vs +100 -> sNaN (+2139095041 > 100)
        self.lds.write_dword(addr, 100)
        self.state.set_vgpr(2, snan)
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, 100)
        self.assertEqual(self.lds.read_dword(addr), snan)

        # neg_NaN (-4194304) vs -Inf (-8388608): -4194304 > -8388608 -> neg_NaN
        self.lds.write_dword(addr, 0xFF800000)
        self.state.set_vgpr(2, neg_nan)
        ret3 = self.interp.execute(self.state, inst)
        self.assertEqual(ret3, 0xFF800000)
        self.assertEqual(self.lds.read_dword(addr), neg_nan)

    def test_subnormal_bit_patterns_as_integers(self) -> None:
        # Smallest positive subnormal: 0x00000001 (+1 in i32)
        # Largest positive subnormal: 0x007FFFFF (+8388607 in i32)
        # Smallest negative subnormal: 0x80000001 (-2147483647 in i32)
        addr = 0x400
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
        )

        # +1 vs -10 -> max is +1 (0x00000001)
        self.lds.write_dword(addr, u32(-10))
        self.state.set_vgpr(1, addr)
        self.state.set_vgpr(2, 0x00000001)
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, u32(-10))
        self.assertEqual(self.lds.read_dword(addr), 0x00000001)

        # -2147483647 (0x80000001) vs INT32_MIN (-2147483648) -> max is 0x80000001
        self.lds.write_dword(addr, u32(INT32_MIN))
        self.state.set_vgpr(2, 0x80000001)
        ret2 = self.interp.execute(self.state, inst)
        self.assertEqual(ret2, u32(INT32_MIN))
        self.assertEqual(self.lds.read_dword(addr), 0x80000001)

    def test_max_offset_and_last_dword_boundary(self) -> None:
        # Maximum 16-bit offset = 0xFFFF (65535)
        # Last valid 4-byte dword boundary in 64 KiB LDS starts at byte 65532 (0xFFFC)
        last_dword_addr = 65532
        self.lds.write_dword(last_dword_addr, 50)

        # Use base 0 and offset 65532
        self.state.set_vgpr(1, 0)
        self.state.set_vgpr(2, 99)
        inst = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=last_dword_addr,
        )
        ret = self.interp.execute(self.state, inst)
        self.assertEqual(ret, 50)
        self.assertEqual(self.lds.read_dword(last_dword_addr), 99)

        # Test offset 65535 (0xFFFF): effective address is 65535 -> dword index (65535 >> 2) & 16383 = 16383
        # Which is byte 65532!
        self.state.set_vgpr(1, 0)
        self.state.set_vgpr(2, 150)
        inst_max_off = DSMaxRtnI32Instruction(
            vdst=RegisterOperand(0, RegType.VGPR),
            vaddr=RegisterOperand(1, RegType.VGPR),
            vdata0=RegisterOperand(2, RegType.VGPR),
            offset=0xFFFF,
        )
        ret2 = self.interp.execute(self.state, inst_max_off)
        self.assertEqual(ret2, 99)
        self.assertEqual(self.lds.read_dword(last_dword_addr), 150)


if __name__ == "__main__":
    unittest.main()
