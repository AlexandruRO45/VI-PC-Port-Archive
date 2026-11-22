from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_src2_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Dual-Source Unsigned 32-bit Maximum:
Atomically reads two 32-bit operands from LDS memory locations A and B (computed
via base address VGPR and two 8-bit dword immediate offsets offset0 and offset1),
computes their unsigned 32-bit maximum, and writes the maximum back into LDS location A.
Location B remains unmodified unless Location A and Location B are identical.
Non-returning: VGPRs are not modified.

Syntax:
    ds_max_src2_u32 vaddr [offset0:offset8] [offset1:offset8]

Architectural Semantics:
    addr_a = (vaddr + offset0 * 4) & 0xFFFFFFFF
    addr_b = (vaddr + offset1 * 4) & 0xFFFFFFFF
    idx_a = (addr_a >> 2) & (lds_dwords - 1)
    idx_b = (addr_b >> 2) & (lds_dwords - 1)
    val_a = LDS[idx_a]
    val_b = LDS[idx_b]
    new_val_a = max(val_a, val_b)
    LDS[idx_a] = new_val_a

Unsigned 32-bit Integer Properties:
    - Range: [0, 2^32 - 1] ([0, 4294967295])
    - UINT32_MIN = 0
    - UINT32_MAX = 0xFFFFFFFF
    - Unsigned comparison: 0xFFFFFFFF > 0, 0x80000000 > 0x7FFFFFFF
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

UINT32_MIN: int = 0
UINT32_MAX: int = 0xFFFF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def ref_ds_max_src2_u32(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_max_src2_u32.

    Computes:
        max(val_a, val_b) in unsigned 32-bit integer arithmetic.
    """
    ua = u32(val_a)
    ub = u32(val_b)
    return max(ua, ub)


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = LDS_SIZE_DWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for address A and address B.

    offset0 and offset1 are 8-bit unsigned integers representing dword counts (scaled by 4).
    """
    byte_addr_a = (vaddr + ((offset0 & 0xFF) * 4)) & MASK32
    byte_addr_b = (vaddr + ((offset1 & 0xFF) * 4)) & MASK32
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective byte address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


class LDSMemory:
    """Architectural memory model representing LDS storage with dual-source atomic operations."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = (eff_addr >> 2) & (self.size_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = (eff_addr >> 2) & (self.size_dwords - 1)
        self.memory[idx] = u32(val)

    def atomic_max_src2_u32(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic dual-source unsigned maximum on LDS: LDS[A] = max(LDS[A], LDS[B])."""
        _, _, idx_a, idx_b = calc_lds_src2_addresses(vaddr, offset0, offset1, self.size_dwords)
        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]
        new_val_a = ref_ds_max_src2_u32(val_a, val_b)
        self.memory[idx_a] = new_val_a
        return new_val_a


class TestLDSMaxSrc2U32OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for unsigned 32-bit dual-source oracle and LDS model."""

    def test_oracle_unsigned_basic_ordering(self) -> None:
        self.assertEqual(ref_ds_max_src2_u32(10, 20), 20)
        self.assertEqual(ref_ds_max_src2_u32(20, 10), 20)
        self.assertEqual(ref_ds_max_src2_u32(500, 500), 500)

    def test_oracle_high_bit_unsigned_semantics(self) -> None:
        # In unsigned: 0x80000000 > 0x7FFFFFFF (unlike signed where 0x80000000 is negative)
        self.assertEqual(ref_ds_max_src2_u32(0x7FFFFFFF, 0x80000000), 0x80000000)
        self.assertEqual(ref_ds_max_src2_u32(0x80000000, 0x7FFFFFFF), 0x80000000)

    def test_oracle_uint32_extremes(self) -> None:
        self.assertEqual(ref_ds_max_src2_u32(0, UINT32_MAX), UINT32_MAX)
        self.assertEqual(ref_ds_max_src2_u32(UINT32_MAX, 0), UINT32_MAX)
        self.assertEqual(ref_ds_max_src2_u32(UINT32_MAX - 1, UINT32_MAX), UINT32_MAX)

    def test_calc_lds_src2_addresses_offsets(self) -> None:
        base = 0x100
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(base, offset0=2, offset1=4)
        self.assertEqual(byte_a, 0x100 + 2 * 4)  # 0x108
        self.assertEqual(byte_b, 0x100 + 4 * 4)  # 0x110
        self.assertEqual(idx_a, 0x108 >> 2)
        self.assertEqual(idx_b, 0x110 >> 2)

    def test_lds_memory_atomic_max_src2_u32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        base = 0x40
        # Initialize location A (base + 0*4) with 100, location B (base + 2*4) with 250
        lds.write_dword(base, 100)
        lds.write_dword(base + 8, 250)

        res = lds.atomic_max_src2_u32(base, offset0=0, offset1=2)
        self.assertEqual(res, 250)
        # Location A updated with max(100, 250) = 250
        self.assertEqual(lds.read_dword(base), 250)
        # Location B remains 250
        self.assertEqual(lds.read_dword(base + 8), 250)

    def test_lds_memory_atomic_max_src2_u32_identical_addresses(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        base = 0x60
        lds.write_dword(base, 77)

        res = lds.atomic_max_src2_u32(base, offset0=1, offset1=1)
        self.assertEqual(res, 0)  # Memory at base + 4 was 0, max(0, 0) = 0

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class DSOpcode(Enum):
    """Data Share (DS) src2 opcodes."""
    DS_MAX_SRC2_U32 = "ds_max_src2_u32"


class RegType(Enum):
    """Register file classes."""
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
            raise ValueError(f"Register index out of range for {self.reg_type.name}: {self.index}")

    def __str__(self) -> str:
        return f"{self.reg_type.value}{self.index}"


@dataclass(frozen=True)
class LDSAddressExpression:
    """Symbolic representation of dual-offset LDS addressing."""
    base_reg: RegisterOperand
    offset0: int = 0
    offset1: int = 0
    gds: bool = False

    def __post_init__(self) -> None:
        if self.base_reg.reg_type != RegType.VGPR:
            raise TypeError("LDS base address register must be a VGPR")
        if not (0 <= self.offset0 <= 255):
            raise ValueError(f"offset0 out of 8-bit range [0, 255]: {self.offset0}")
        if not (0 <= self.offset1 <= 255):
            raise ValueError(f"offset1 out of 8-bit range [0, 255]: {self.offset1}")


@dataclass(frozen=True)
class DSMaxSrc2U32Instruction:
    """Symbolic AST node for ds_max_src2_u32 instruction.

    Format:
        ds_max_src2_u32 vaddr [offset0:offset8] [offset1:offset8]
    """
    addr: LDSAddressExpression
    opcode: DSOpcode = DSOpcode.DS_MAX_SRC2_U32

    def to_assembly(self) -> str:
        """Serialize AST node to assembly string."""
        s = f"{self.opcode.value} {self.addr.base_reg}"
        if self.addr.offset0 != 0:
            s += f" offset0:{self.addr.offset0}"
        if self.addr.offset1 != 0:
            s += f" offset1:{self.addr.offset1}"
        return s

    @classmethod
    def parse(cls, asm: str) -> DSMaxSrc2U32Instruction:
        """Parse assembly string into a DSMaxSrc2U32Instruction AST node."""
        cleaned = re.sub(r"[;#].*$", "", asm).strip()
        if not cleaned:
            raise ValueError("Empty assembly string")

        tokens = cleaned.split()
        if not tokens:
            raise ValueError("Empty instruction")

        op_str = tokens[0].lower()
        if op_str != DSOpcode.DS_MAX_SRC2_U32.value:
            raise ValueError(f"Unsupported opcode '{tokens[0]}', expected '{DSOpcode.DS_MAX_SRC2_U32.value}'")

        if len(tokens) < 2:
            raise ValueError(f"Missing base address operand for {op_str}")

        # First operand: base address register
        base_str = tokens[1].rstrip(",")
        if not (base_str.startswith("v") or base_str.startswith("V")):
            raise ValueError(f"Expected base register starting with 'v', got '{base_str}'")
        try:
            reg_idx = int(base_str[1:])
        except ValueError:
            raise ValueError(f"Invalid VGPR index: '{base_str}'")

        base_reg = RegisterOperand(reg_idx, RegType.VGPR)

        offset0 = 0
        offset1 = 0

        # Remaining tokens are optional modifiers: offset0:N, offset1:M
        for mod in tokens[2:]:
            mod = mod.rstrip(",").strip()
            mod_lower = mod.lower()
            if mod_lower.startswith("offset0:"):
                raw = mod[len("offset0:"):]
                try:
                    offset0 = int(raw, 0)
                except ValueError:
                    raise ValueError(f"Invalid integer for offset0: '{raw}'")
            elif mod_lower.startswith("offset1:"):
                raw = mod[len("offset1:"):]
                try:
                    offset1 = int(raw, 0)
                except ValueError:
                    raise ValueError(f"Invalid integer for offset1: '{raw}'")
            else:
                raise ValueError(f"Unknown modifier or token: '{mod}'")

        addr_expr = LDSAddressExpression(base_reg=base_reg, offset0=offset0, offset1=offset1)
        return cls(addr=addr_expr)


class TestDSMaxSrc2U32AST(unittest.TestCase):
    """Phase 2: Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_operand_valid(self) -> None:
        v0 = RegisterOperand(0, RegType.VGPR)
        self.assertEqual(str(v0), "v0")
        v255 = RegisterOperand(255, RegType.VGPR)
        self.assertEqual(str(v255), "v255")

    def test_register_operand_bounds(self) -> None:
        with self.assertRaises(ValueError):
            RegisterOperand(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegisterOperand(256, RegType.VGPR)

    def test_instruction_formatting(self) -> None:
        vaddr = RegisterOperand(2, RegType.VGPR)
        expr1 = LDSAddressExpression(base_reg=vaddr, offset0=0, offset1=0)
        inst1 = DSMaxSrc2U32Instruction(addr=expr1)
        self.assertEqual(inst1.to_assembly(), "ds_max_src2_u32 v2")

        expr2 = LDSAddressExpression(base_reg=vaddr, offset0=10, offset1=20)
        inst2 = DSMaxSrc2U32Instruction(addr=expr2)
        self.assertEqual(inst2.to_assembly(), "ds_max_src2_u32 v2 offset0:10 offset1:20")

    def test_instruction_parsing(self) -> None:
        inst1 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v5")
        self.assertEqual(inst1.addr.base_reg.index, 5)
        self.assertEqual(inst1.addr.offset0, 0)
        self.assertEqual(inst1.addr.offset1, 0)

        inst2 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v10 offset0:4 offset1:8")
        self.assertEqual(inst2.addr.base_reg.index, 10)
        self.assertEqual(inst2.addr.offset0, 4)
        self.assertEqual(inst2.addr.offset1, 8)

    def test_instruction_roundtrip(self) -> None:
        vaddr = RegisterOperand(15, RegType.VGPR)
        expr = LDSAddressExpression(base_reg=vaddr, offset0=32, offset1=64)
        inst = DSMaxSrc2U32Instruction(addr=expr)
        parsed = DSMaxSrc2U32Instruction.parse(inst.to_assembly())
        self.assertEqual(inst, parsed)

# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarThreadState:
    """Scalar thread architectural execution state (VGPRs, SGPRs, and LDS storage)."""

    def __init__(self, lds: Optional[LDSMemory] = None) -> None:
        self.vgprs: List[int] = [0] * DEFAULT_NUM_VGPRS
        self.sgprs: List[int] = [0] * DEFAULT_NUM_SGPRS
        self.vcc: int = 0
        self.scc: int = 0
        self.lds: LDSMemory = lds if lds is not None else LDSMemory()

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)


class ScalarDSInterpreter:
    """Deterministic scalar execution interpreter for ds_max_src2_u32."""

    def execute(self, state: ScalarThreadState, inst: DSMaxSrc2U32Instruction) -> int:
        """Execute ds_max_src2_u32 on scalar thread state.

        Reads base address from inst.addr.base_reg.
        Performs dual-source atomic unsigned maximum on LDS: LDS[A] = max(LDS[A], LDS[B]).
        VGPRs are NOT written by this instruction.
        Returns the new value written to LDS[A].
        """
        vaddr = state.get_vgpr(inst.addr.base_reg.index)
        new_val_a = state.lds.atomic_max_src2_u32(vaddr, inst.addr.offset0, inst.addr.offset1)
        return new_val_a


class TestDSMaxSrc2U32ScalarInterpreter(unittest.TestCase):
    """Phase 3: Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_scalar_execution_basic(self) -> None:
        base = 0x20
        # Location A = base + 0*4 = 0x20
        # Location B = base + 1*4 = 0x24
        self.lds.write_dword(base, 100)
        self.lds.write_dword(base + 4, 300)
        self.state.set_vgpr(2, base)

        inst = DSMaxSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(2, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        res = self.interp.execute(self.state, inst)
        self.assertEqual(res, 300)
        self.assertEqual(self.lds.read_dword(base), 300)
        self.assertEqual(self.lds.read_dword(base + 4), 300)
        # Verify VGPRs remain unchanged
        self.assertEqual(self.state.get_vgpr(2), base)

    def test_scalar_execution_loc_a_already_larger(self) -> None:
        base = 0x40
        self.lds.write_dword(base, 500)
        self.lds.write_dword(base + 8, 200)
        self.state.set_vgpr(1, base)

        inst = DSMaxSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(1, RegType.VGPR),
                offset0=0,
                offset1=2,
            )
        )

        res = self.interp.execute(self.state, inst)
        self.assertEqual(res, 500)
        self.assertEqual(self.lds.read_dword(base), 500)
        self.assertEqual(self.lds.read_dword(base + 8), 200)

    def test_scalar_execution_sequential_chain(self) -> None:
        base = 0x100
        # Store values at dwords 0, 1, 2, 3
        self.lds.write_dword(base, 10)
        self.lds.write_dword(base + 4, 50)
        self.lds.write_dword(base + 8, 30)
        self.lds.write_dword(base + 12, 100)
        self.state.set_vgpr(0, base)

        # Step 1: Loc 0 = max(Loc 0, Loc 1) -> max(10, 50) = 50
        inst1 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v0 offset0:0 offset1:1")
        self.interp.execute(self.state, inst1)
        self.assertEqual(self.lds.read_dword(base), 50)

        # Step 2: Loc 0 = max(Loc 0, Loc 2) -> max(50, 30) = 50
        inst2 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v0 offset0:0 offset1:2")
        self.interp.execute(self.state, inst2)
        self.assertEqual(self.lds.read_dword(base), 50)

        # Step 3: Loc 0 = max(Loc 0, Loc 3) -> max(50, 100) = 100
        inst3 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v0 offset0:0 offset1:3")
        self.interp.execute(self.state, inst3)
        self.assertEqual(self.lds.read_dword(base), 100)

# ==============================================================================
# SIMD Wave Execution and EXEC Masking
# ==============================================================================

class SIMDWaveState:
    """SIMD wave architectural execution state supporting Wave32 and Wave64 modes."""

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


def detect_lds_src2_bank_conflicts(addr_pairs: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Detect LDS memory bank conflicts across active lanes accessing dual operands."""
    bank_to_dwords: Dict[int, Dict[int, List[int]]] = {}

    for lane, (addr_a, addr_b) in enumerate(addr_pairs):
        dw_a = (addr_a >> 2) & (LDS_SIZE_DWORDS - 1)
        dw_b = (addr_b >> 2) & (LDS_SIZE_DWORDS - 1)

        for dw in (dw_a, dw_b):
            bank = dw % LDS_BANK_COUNT
            if bank not in bank_to_dwords:
                bank_to_dwords[bank] = {}
            if dw not in bank_to_dwords[bank]:
                bank_to_dwords[bank][dw] = []
            bank_to_dwords[bank][dw].append(lane)

    conflicts: Dict[int, List[int]] = {}
    for bank, dword_map in bank_to_dwords.items():
        if len(dword_map) > 1:
            lanes: List[int] = []
            for lane_list in dword_map.values():
                lanes.extend(lane_list)
            conflicts[bank] = sorted(list(set(lanes)))
    return conflicts


class SIMDDSInterpreter:
    """SIMD wave execution interpreter handling multi-lane divergence and EXEC masking."""

    def execute(
        self,
        state: SIMDWaveState,
        inst: DSMaxSrc2U32Instruction,
    ) -> Dict[int, int]:
        """Execute ds_max_src2_u32 across all active SIMD lanes."""
        results: Dict[int, int] = {}
        for lane in range(state.wave_size):
            if not state.is_lane_active(lane):
                continue
            vaddr = state.get_lane_vgpr(lane, inst.addr.base_reg.index)
            new_val_a = state.lds.atomic_max_src2_u32(vaddr, inst.addr.offset0, inst.addr.offset1)
            results[lane] = new_val_a
        return results


class TestDSMaxSrc2U32SIMDMultiLane(unittest.TestCase):
    """Phase 4: Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=4096)
        self.interp = SIMDDSInterpreter()

    def test_wave32_independent_lanes(self) -> None:
        wave = SIMDWaveState(wave_size=32, exec_mask=0xFFFFFFFF, lds=self.lds)
        inst = DSMaxSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(1, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        for l in range(32):
            base = l * 8
            self.lds.write_dword(base, 10 + l)
            self.lds.write_dword(base + 4, 50 + l * 2)
            wave.set_lane_vgpr(l, 1, base)

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 32)

        for l in range(32):
            base = l * 8
            expected_max = 50 + l * 2
            self.assertEqual(results[l], expected_max)
            self.assertEqual(self.lds.read_dword(base), expected_max)
            self.assertEqual(self.lds.read_dword(base + 4), expected_max)

    def test_wave64_full_divergence(self) -> None:
        wave = SIMDWaveState(wave_size=64, exec_mask=0xFFFFFFFFFFFFFFFF, lds=self.lds)
        inst = DSMaxSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(2, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        for l in range(64):
            base = l * 8
            self.lds.write_dword(base, 200 + l)
            self.lds.write_dword(base + 4, 100 + l)
            wave.set_lane_vgpr(l, 2, base)

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 64)

        for l in range(64):
            base = l * 8
            expected_max = 200 + l
            self.assertEqual(results[l], expected_max)
            self.assertEqual(self.lds.read_dword(base), expected_max)

    def test_wave32_exec_masking_partial(self) -> None:
        # Lanes 0..15 active, lanes 16..31 inactive
        mask = 0x0000FFFF
        wave = SIMDWaveState(wave_size=32, exec_mask=mask, lds=self.lds)
        inst = DSMaxSrc2U32Instruction(
            addr=LDSAddressExpression(
                base_reg=RegisterOperand(1, RegType.VGPR),
                offset0=0,
                offset1=1,
            )
        )

        for l in range(32):
            base = l * 8
            self.lds.write_dword(base, 10)
            self.lds.write_dword(base + 4, 100)
            wave.set_lane_vgpr(l, 1, base)

        results = self.interp.execute(wave, inst)
        self.assertEqual(len(results), 16)

        # Active lanes updated
        for l in range(16):
            base = l * 8
            self.assertEqual(self.lds.read_dword(base), 100)

        # Inactive lanes untouched
        for l in range(16, 32):
            base = l * 8
            self.assertEqual(self.lds.read_dword(base), 10)

    def test_bank_conflict_detection(self) -> None:
        # Lane 0 accesses dword 0 and dword 1 (banks 0, 1)
        # Lane 1 accesses dword 32 and dword 33 (banks 0, 1) -> bank conflicts!
        addr_pairs_conflict = [(0 * 4, 1 * 4), (32 * 4, 33 * 4)]
        conflicts = detect_lds_src2_bank_conflicts(addr_pairs_conflict)
        self.assertIn(0, conflicts)
        self.assertIn(1, conflicts)
        self.assertEqual(conflicts[0], [0, 1])

        # No conflict: lanes access distinct banks
        addr_pairs_clean = [(0 * 4, 1 * 4), (2 * 4, 3 * 4)]
        self.assertEqual(detect_lds_src2_bank_conflicts(addr_pairs_clean), {})

class TestDSMaxSrc2U32RegisterAliasing(unittest.TestCase):
    """Phase 5: Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)
        self.state = ScalarThreadState(lds=self.lds)
        self.interp = ScalarDSInterpreter()

    def test_all_registers_preserved_non_returning(self) -> None:
        base = 0x20
        self.lds.write_dword(base, 10)
        self.lds.write_dword(base + 4, 50)

        # Put canaries across all VGPRs and SGPRs
        for r in range(DEFAULT_NUM_VGPRS):
            self.state.set_vgpr(r, 0x1000 + r)
        for s in range(DEFAULT_NUM_SGPRS):
            self.state.sgprs[s] = 0x2000 + s

        self.state.set_vgpr(5, base)  # base address in v5

        inst = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v5 offset0:0 offset1:1")
        res = self.interp.execute(self.state, inst)
        self.assertEqual(res, 50)

        # Every single VGPR must remain completely identical
        for r in range(DEFAULT_NUM_VGPRS):
            expected = base if r == 5 else (0x1000 + r)
            self.assertEqual(self.state.get_vgpr(r), expected)

        # Every SGPR untouched
        for s in range(DEFAULT_NUM_SGPRS):
            self.assertEqual(self.state.sgprs[s], 0x2000 + s)

    def test_same_offset_aliasing(self) -> None:
        # offset0 == offset1 -> Loc A == Loc B
        base = 0x40
        self.lds.write_dword(base + 8, 999)
        self.state.set_vgpr(3, base)

        inst = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v3 offset0:2 offset1:2")
        res = self.interp.execute(self.state, inst)
        self.assertEqual(res, 999)
        self.assertEqual(self.lds.read_dword(base + 8), 999)

    def test_interlocking_offset_pairs(self) -> None:
        # Loc A = base + 4, Loc B = base + 8
        base = 0x50
        self.lds.write_dword(base + 4, 15)
        self.lds.write_dword(base + 8, 30)
        self.state.set_vgpr(0, base)

        # Op 1: max(LDS[A], LDS[B]) where A=1, B=2 -> LDS[A] becomes 30
        inst1 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v0 offset0:1 offset1:2")
        self.interp.execute(self.state, inst1)
        self.assertEqual(self.lds.read_dword(base + 4), 30)
        self.assertEqual(self.lds.read_dword(base + 8), 30)

        # Op 2: reverse roles where A=2, B=1 -> LDS[B] was 30, LDS[A] becomes 30
        inst2 = DSMaxSrc2U32Instruction.parse("ds_max_src2_u32 v0 offset0:2 offset1:1")
        self.interp.execute(self.state, inst2)
        self.assertEqual(self.lds.read_dword(base + 4), 30)
        self.assertEqual(self.lds.read_dword(base + 8), 30)


if __name__ == "__main__":
    unittest.main()
