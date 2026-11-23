"""Semantic interpreter tests and translator verification for single-precision float LDS atomic minimum (ds_min_f32, ds_min_rtn_f32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_min_f32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_min_f32_hw.py
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import lds_run as L
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000

POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
ONE_F32: int = 0x3F800000           # 1.0
NEG_ONE_F32: int = 0xBF800000       # -1.0
TWO_F32: int = 0x40000000           # 2.0
NEG_TWO_F32: int = 0xC0000000       # -2.0
HALF_F32: int = 0x3F000000          # 0.5
NEG_HALF_F32: int = 0xBF000000      # -0.5
TEN_F32: int = 0x41200000           # 10.0
NEG_TEN_F32: int = 0xC1200000       # -10.0

POS_INF: int = 0x7F800000           # +infinity
NEG_INF: int = 0xFF800000           # -infinity
QNAN_VAL: int = 0x7FC00000          # Quiet NaN
SNAN_VAL: int = 0x7F800001          # Signaling NaN


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 binary32 32-bit unsigned bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 binary32 float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan(bits: int) -> bool:
    """True if 32-bit pattern is an IEEE 754 NaN."""
    b = bits & MASK32
    return ((b & EXP_MASK) == EXP_MASK) and ((b & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """True if 32-bit pattern is +0.0 or -0.0."""
    return (bits & ~SIGN_MASK) == 0


def ref_ds_min_f32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_min_f32 atomic operation.

    Follows GLSL.std.450 NMin / IEEE 754-2008 minNum:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a NaN.
      - Signed zero ordering: -0.0 is strictly less than +0.0 (min(+0.0, -0.0) -> -0.0).
      - Subnormals, infinities, and normals follow standard float ordering.
    """
    a = curr_mem & MASK32
    b = data_val & MASK32

    a_nan = is_nan(a)
    b_nan = is_nan(b)

    if a_nan and b_nan:
        return a | QNAN_BIT
    if a_nan:
        return b
    if b_nan:
        return a

    if is_zero(a) and is_zero(b):
        return NEG_ZERO if ((a == NEG_ZERO) or (b == NEG_ZERO)) else POS_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa < fb:
        return a
    elif fb < fa:
        return b
    else:
        # Same magnitude: negative sign is smaller
        return a if (a & SIGN_MASK) != 0 else b


def ref_ds_min_rtn_f32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic float minimum: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_min_f32(old, data_val)
    return old, new


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, size_bytes: int = 4096) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.storage = [0] * self.num_dwords

    def resolve_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        dword_idx = eff_addr >> 2
        return dword_idx & (self.num_dwords - 1)

    def read_dword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.resolve_dword_index(byte_addr, offset)
        return self.storage[idx]

    def write_dword(self, byte_addr: int, value: int, offset: int = 0) -> None:
        idx = self.resolve_dword_index(byte_addr, offset)
        self.storage[idx] = value & MASK32

    def atomic_min_f32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_min_f32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSMinF32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_float_bit_conversions(self) -> None:
        self.assertEqual(f32_to_bits(0.0), POS_ZERO)
        self.assertEqual(f32_to_bits(-0.0), NEG_ZERO)
        self.assertEqual(f32_to_bits(1.0), ONE_F32)
        self.assertEqual(f32_to_bits(-1.0), NEG_ONE_F32)
        self.assertEqual(bits_to_f32(ONE_F32), 1.0)
        self.assertEqual(bits_to_f32(NEG_ONE_F32), -1.0)

    def test_reference_oracle_basic_floats(self) -> None:
        # Standard positive floats: min(1.0, 2.0) = 1.0
        self.assertEqual(ref_ds_min_f32(ONE_F32, TWO_F32), ONE_F32)
        self.assertEqual(ref_ds_min_f32(TWO_F32, ONE_F32), ONE_F32)

        # Negative vs positive: min(-2.0, 1.0) = -2.0
        self.assertEqual(ref_ds_min_f32(NEG_TWO_F32, ONE_F32), NEG_TWO_F32)
        self.assertEqual(ref_ds_min_f32(ONE_F32, NEG_TWO_F32), NEG_TWO_F32)

        # Negative vs negative: min(-10.0, -2.0) = -10.0
        self.assertEqual(ref_ds_min_f32(NEG_TEN_F32, NEG_TWO_F32), NEG_TEN_F32)
        self.assertEqual(ref_ds_min_f32(NEG_TWO_F32, NEG_TEN_F32), NEG_TEN_F32)

    def test_reference_oracle_signed_zeros(self) -> None:
        # -0.0 < +0.0
        self.assertEqual(ref_ds_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_ds_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_ds_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_ds_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_reference_oracle_nan_suppression(self) -> None:
        # Single NaN: non-NaN operand is preserved (minNum / NMin semantics)
        self.assertEqual(ref_ds_min_f32(QNAN_VAL, ONE_F32), ONE_F32)
        self.assertEqual(ref_ds_min_f32(ONE_F32, QNAN_VAL), ONE_F32)
        self.assertEqual(ref_ds_min_f32(SNAN_VAL, NEG_TEN_F32), NEG_TEN_F32)

        # Both NaN: returns NaN
        res = ref_ds_min_f32(QNAN_VAL, SNAN_VAL)
        self.assertTrue(is_nan(res))

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_min_rtn_f32(TEN_F32, TWO_F32)
        self.assertEqual(old, TEN_F32)
        self.assertEqual(new, TWO_F32)

        old, new = ref_ds_min_rtn_f32(NEG_TEN_F32, ONE_F32)
        self.assertEqual(old, NEG_TEN_F32)
        self.assertEqual(new, NEG_TEN_F32)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, TEN_F32)

        # Non-returning atomic min with 2.0 -> memory becomes 2.0
        old, new = lds.atomic_min_f32(addr, TWO_F32, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, TWO_F32)
        self.assertEqual(lds.read_dword(addr), TWO_F32)

        # Returning atomic min with -1.0 -> returns 2.0, memory becomes -1.0
        old, new = lds.atomic_min_f32(addr, NEG_ONE_F32, rtn=True)
        self.assertEqual(old, TWO_F32)
        self.assertEqual(new, NEG_ONE_F32)
        self.assertEqual(lds.read_dword(addr), NEG_ONE_F32)

class Register:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index

    def __repr__(self) -> str:
        return f"{self.name}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return self.name == other.name and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.name, self.index))


class VGPR(Register):
    def __init__(self, index: int) -> None:
        if not (0 <= index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {index}")
        super().__init__("v", index)


class DSMinF32Instruction:
    """Symbolic AST node representing ds_min_f32 and ds_min_rtn_f32 instructions."""

    def __init__(
        self,
        mnemonic: str,
        vaddr: VGPR,
        vdata0: VGPR,
        vdst: Optional[VGPR] = None,
        offset: int = 0,
        gds: bool = False,
    ) -> None:
        if mnemonic not in ("ds_min_f32", "ds_min_rtn_f32"):
            raise ValueError(f"Invalid mnemonic for ds_min_f32 AST: {mnemonic}")
        if mnemonic == "ds_min_rtn_f32" and vdst is None:
            raise ValueError("ds_min_rtn_f32 requires vdst operand")
        if mnemonic == "ds_min_f32" and vdst is not None:
            raise ValueError("ds_min_f32 does not accept vdst operand")
        if not (0 <= offset <= 65535):
            raise ValueError(f"Offset out of range [0, 65535]: {offset}")

        self.mnemonic = mnemonic
        self.vaddr = vaddr
        self.vdata0 = vdata0
        self.vdst = vdst
        self.offset = offset
        self.gds = gds

    @property
    def is_return(self) -> bool:
        return self.mnemonic == "ds_min_rtn_f32"

    def __repr__(self) -> str:
        parts = [self.mnemonic]
        if self.vdst is not None:
            parts.append(str(self.vdst))
        parts.append(str(self.vaddr))
        parts.append(str(self.vdata0))
        text = f"{parts[0]} {', '.join(parts[1:])}"
        if self.offset:
            text += f" offset:{self.offset}"
        if self.gds:
            text += " gds"
        return text

    @classmethod
    def parse(cls, asm: str) -> "DSMinF32Instruction":
        line = asm.strip()
        if not line:
            raise ValueError("Empty assembly string")
        tokens = line.split()
        mnem = tokens[0]
        if mnem not in ("ds_min_f32", "ds_min_rtn_f32"):
            raise ValueError(f"Unsupported mnemonic: {mnem}")

        rest = " ".join(tokens[1:])
        mod_tokens = rest.split()
        clean_operands: list[str] = []
        offset = 0
        gds = False

        for tok in mod_tokens:
            tok_clean = tok.rstrip(",")
            if tok_clean.startswith("offset:"):
                try:
                    offset = int(tok_clean.split(":", 1)[1], 0)
                except ValueError:
                    raise ValueError(f"Invalid offset format: {tok_clean}")
            elif tok_clean == "gds":
                gds = True
            elif tok_clean:
                clean_operands.append(tok_clean)

        def parse_vgpr(reg_str: str) -> VGPR:
            m = re.fullmatch(r"v(\d+)", reg_str)
            if not m:
                raise ValueError(f"Expected VGPR (e.g. v1), got: {reg_str}")
            return VGPR(int(m.group(1)))

        if mnem == "ds_min_f32":
            if len(clean_operands) != 2:
                raise ValueError(f"ds_min_f32 requires 2 operands (vaddr, vdata0), got {len(clean_operands)}")
            vaddr = parse_vgpr(clean_operands[0])
            vdata0 = parse_vgpr(clean_operands[1])
            return cls(mnem, vaddr, vdata0, vdst=None, offset=offset, gds=gds)
        else:
            if len(clean_operands) != 3:
                raise ValueError(f"ds_min_rtn_f32 requires 3 operands (vdst, vaddr, vdata0), got {len(clean_operands)}")
            vdst = parse_vgpr(clean_operands[0])
            vaddr = parse_vgpr(clean_operands[1])
            vdata0 = parse_vgpr(clean_operands[2])
            return cls(mnem, vaddr, vdata0, vdst=vdst, offset=offset, gds=gds)


class TestDSMinF32ASTAndOpcodeDefinitions(unittest.TestCase):
    """Part 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_properties(self) -> None:
        inst_non_rtn = DSMinF32Instruction("ds_min_f32", VGPR(1), VGPR(2), offset=16)
        self.assertFalse(inst_non_rtn.is_return)
        self.assertEqual(inst_non_rtn.vaddr, VGPR(1))
        self.assertEqual(inst_non_rtn.vdata0, VGPR(2))
        self.assertIsNone(inst_non_rtn.vdst)
        self.assertEqual(inst_non_rtn.offset, 16)
        self.assertEqual(repr(inst_non_rtn), "ds_min_f32 v1, v2 offset:16")

        inst_rtn = DSMinF32Instruction("ds_min_rtn_f32", VGPR(3), VGPR(4), vdst=VGPR(5), offset=32, gds=True)
        self.assertTrue(inst_rtn.is_return)
        self.assertEqual(inst_rtn.vdst, VGPR(5))
        self.assertEqual(repr(inst_rtn), "ds_min_rtn_f32 v5, v3, v4 offset:32 gds")

    def test_ast_parsing_valid(self) -> None:
        inst1 = DSMinF32Instruction.parse("ds_min_f32 v10, v20")
        self.assertEqual(inst1.mnemonic, "ds_min_f32")
        self.assertEqual(inst1.vaddr, VGPR(10))
        self.assertEqual(inst1.vdata0, VGPR(20))

        inst2 = DSMinF32Instruction.parse("ds_min_rtn_f32 v5, v1, v2 offset:0x40")
        self.assertEqual(inst2.mnemonic, "ds_min_rtn_f32")
        self.assertEqual(inst2.vdst, VGPR(5))
        self.assertEqual(inst2.offset, 64)

    def test_ast_parsing_invalid(self) -> None:
        with self.assertRaises(ValueError):
            DSMinF32Instruction.parse("")
        with self.assertRaises(ValueError):
            DSMinF32Instruction.parse("ds_min_f32 v1")
        with self.assertRaises(ValueError):
            DSMinF32Instruction.parse("ds_min_rtn_f32 v1, v2")
        with self.assertRaises(ValueError):
            DSMinF32Instruction.parse("ds_min_f32 s1, v2")

class ScalarThreadState:
    """State of a single hardware execution lane / thread."""

    def __init__(self, lds: LDSMemory) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.lds = lds
        self.lgkm_cnt = 0

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32


class ScalarDSInterpreter:
    """Deterministic instruction interpreter for scalar LDS float min executions."""

    def __init__(self, lds: LDSMemory) -> None:
        self.lds = lds

    def execute(self, state: ScalarThreadState, inst: DSMinF32Instruction) -> Optional[int]:
        addr = state.get_vgpr(inst.vaddr)
        data = state.get_vgpr(inst.vdata0)

        old_val, new_val = self.lds.atomic_min_f32(addr, data, offset=inst.offset, rtn=inst.is_return)

        if inst.is_return and inst.vdst is not None:
            assert old_val is not None
            state.set_vgpr(inst.vdst, old_val)

        state.lgkm_cnt += 1
        return old_val


class TestDSMinF32ScalarInterpreter(unittest.TestCase):
    """Part 3: Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = ScalarDSInterpreter(self.lds)
        self.state = ScalarThreadState(self.lds)

    def test_scalar_execution_non_returning(self) -> None:
        self.lds.write_dword(0x10, TEN_F32)
        self.state.set_vgpr(VGPR(1), 0x10)
        self.state.set_vgpr(VGPR(2), TWO_F32)

        inst = DSMinF32Instruction.parse("ds_min_f32 v1, v2")
        res = self.interp.execute(self.state, inst)

        self.assertIsNone(res)
        self.assertEqual(self.lds.read_dword(0x10), TWO_F32)

    def test_scalar_execution_returning(self) -> None:
        self.lds.write_dword(0x20, ONE_F32)
        self.state.set_vgpr(VGPR(10), 0x20)
        self.state.set_vgpr(VGPR(11), NEG_TWO_F32)
        self.state.set_vgpr(VGPR(12), 0xDEADBEEF)

        inst = DSMinF32Instruction.parse("ds_min_rtn_f32 v12, v10, v11")
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, ONE_F32)
        self.assertEqual(self.state.get_vgpr(VGPR(12)), ONE_F32)
        self.assertEqual(self.lds.read_dword(0x20), NEG_TWO_F32)

    def test_scalar_execution_with_offset(self) -> None:
        self.lds.write_dword(0x34, HALF_F32)
        self.state.set_vgpr(VGPR(1), 0x30)
        self.state.set_vgpr(VGPR(2), NEG_HALF_F32)

        inst = DSMinF32Instruction.parse("ds_min_rtn_f32 v0, v1, v2 offset:4")
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, HALF_F32)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), HALF_F32)
        self.assertEqual(self.lds.read_dword(0x34), NEG_HALF_F32)

class SIMDWaveState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(self, lds: LDSMemory, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.lds = lds
        self.exec_mask = (1 << wave_size) - 1
        self.lanes = [ScalarThreadState(lds) for _ in range(wave_size)]

    def is_lane_active(self, lane_id: int) -> bool:
        return bool((self.exec_mask >> lane_id) & 1)


class SIMDDSInterpreter:
    """Multi-lane SIMD interpreter modeling divergence and atomic address clashes."""

    def __init__(self, lds: LDSMemory) -> None:
        self.lds = lds

    def execute_wave(
        self,
        wave: SIMDWaveState,
        inst: DSMinF32Instruction,
        arbitration_order: Optional[List[int]] = None,
    ) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * wave.wave_size
        active_lanes = [i for i in range(wave.wave_size) if wave.is_lane_active(i)]

        order = [i for i in arbitration_order if i in active_lanes] if arbitration_order is not None else active_lanes

        for lane_id in order:
            lane_state = wave.lanes[lane_id]
            addr = lane_state.get_vgpr(inst.vaddr)
            data = lane_state.get_vgpr(inst.vdata0)

            old_val, new_val = self.lds.atomic_min_f32(addr, data, offset=inst.offset, rtn=inst.is_return)

            if inst.is_return and inst.vdst is not None:
                assert old_val is not None
                lane_state.set_vgpr(inst.vdst, old_val)
                results[lane_id] = old_val

        return results


class TestDSMinF32SIMDDivergenceAndExecMask(unittest.TestCase):
    """Part 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = SIMDDSInterpreter(self.lds)

    def test_divergent_addresses_wave32(self) -> None:
        wave = SIMDWaveState(self.lds, wave_size=32)
        inst = DSMinF32Instruction.parse("ds_min_rtn_f32 v5, v1, v2")

        for i in range(32):
            init_val = f32_to_bits(10.0 + i)
            data_val = f32_to_bits(5.0 + i * 0.1)
            self.lds.write_dword(i * 4, init_val)
            wave.lanes[i].set_vgpr(VGPR(1), i * 4)
            wave.lanes[i].set_vgpr(VGPR(2), data_val)

        results = self.interp.execute_wave(wave, inst)

        for i in range(32):
            expected_old = f32_to_bits(10.0 + i)
            expected_new = f32_to_bits(5.0 + i * 0.1)
            self.assertEqual(results[i], expected_old)
            self.assertEqual(wave.lanes[i].get_vgpr(VGPR(5)), expected_old)
            self.assertEqual(self.lds.read_dword(i * 4), expected_new)

    def test_exec_masking_wave64(self) -> None:
        lds = LDSMemory(size_bytes=8192)
        interp = SIMDDSInterpreter(lds)
        wave = SIMDWaveState(lds, wave_size=64)
        mask = 0
        for i in range(0, 64, 2):
            mask |= (1 << i)
        wave.exec_mask = mask

        inst = DSMinF32Instruction.parse("ds_min_rtn_f32 v3, v1, v2")
        for i in range(64):
            lds.write_dword(i * 4, TEN_F32)
            wave.lanes[i].set_vgpr(VGPR(1), i * 4)
            wave.lanes[i].set_vgpr(VGPR(2), TWO_F32)
            wave.lanes[i].set_vgpr(VGPR(3), 0xCAFE)

        results = interp.execute_wave(wave, inst)

        for i in range(64):
            if i % 2 == 0:
                self.assertEqual(results[i], TEN_F32)
                self.assertEqual(wave.lanes[i].get_vgpr(VGPR(3)), TEN_F32)
                self.assertEqual(lds.read_dword(i * 4), TWO_F32)
            else:
                self.assertIsNone(results[i])
                self.assertEqual(wave.lanes[i].get_vgpr(VGPR(3)), 0xCAFE)
                self.assertEqual(lds.read_dword(i * 4), TEN_F32)

    def test_address_clash_convergence(self) -> None:
        inst = DSMinF32Instruction.parse("ds_min_f32 v1, v2")
        lane_vals = [100.0, -5.0, 3.2, -50.5, 0.0, -0.0, 15.0, -100.0] + [float(i) for i in range(24)]
        global_min_bits = f32_to_bits(-100.0)

        for order in [list(range(32)), list(reversed(range(32)))]:
            lds = LDSMemory(size_bytes=1024)
            interp = SIMDDSInterpreter(lds)
            wave = SIMDWaveState(lds, wave_size=32)

            lds.write_dword(0x40, f32_to_bits(1000.0))
            for i in range(32):
                wave.lanes[i].set_vgpr(VGPR(1), 0x40)
                wave.lanes[i].set_vgpr(VGPR(2), f32_to_bits(lane_vals[i]))

            interp.execute_wave(wave, inst, arbitration_order=order)
            self.assertEqual(lds.read_dword(0x40), global_min_bits)


if __name__ == '__main__':
    unittest.main()
