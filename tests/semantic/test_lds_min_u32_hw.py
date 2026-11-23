"""Semantic interpreter tests and reference mathematical oracle for ds_min_u32.

AMD RDNA2 / GFX10 LDS (Local Data Share) unsigned 32-bit atomic minimum.
Specification & Hardware Semantics:
  * Non-returning form: `ds_min_u32 vaddr, vdata0 [offset:offset16]`
      Computes effective byte address: eff_addr = (vaddr + offset16) & 0xFFFFFFFF
      Dword index: idx = (eff_addr >> 2) & (lds_dwords - 1)
      Atomic operation: LDS[idx] = min_u32(LDS[idx], vdata0)
      Unsigned 32-bit comparison: 0x00000000 < 0x80000000 < 0xFFFFFFFF
      Non-returning: prior value in LDS[idx] is discarded, destination register is untouched.
  * Returning form: `ds_min_rtn_u32 vdst, vaddr, vdata0 [offset:offset16]`
      Computes effective byte address and updates LDS[idx] identically.
      Returns prior 32-bit value into vdst.
  * In SPIR-V, translated via OpAtomicUMin on Workgroup memory array.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import support
    import lds_run as L
    TRANSLATOR_AVAILABLE = True
except (ImportError, TypeError):
    support = None
    L = None
    TRANSLATOR_AVAILABLE = False

MASK32 = 0xFFFFFFFF
LDS_SIZE_BYTES_DEFAULT = 4096
LDS_DWORDS_DEFAULT = 1024


def ref_min_u32(a: int, b: int) -> int:
    """Reference unsigned 32-bit minimum oracle."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua <= ub else ub


class LDSModel32:
    """Architectural memory model of RDNA2 Local Data Share (32-bit dwords)."""

    def __init__(self, size_bytes: int = LDS_SIZE_BYTES_DEFAULT):
        assert size_bytes > 0 and (size_bytes & (size_bytes - 1)) == 0, "LDS size must be power of two"
        self.size_bytes = size_bytes
        self.lds_dwords = size_bytes >> 2
        self.memory: dict[int, int] = {}

    def read_dword(self, byte_addr: int) -> int:
        idx = ((byte_addr & MASK32) >> 2) & (self.lds_dwords - 1)
        return self.memory.get(idx, 0)

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = ((byte_addr & MASK32) >> 2) & (self.lds_dwords - 1)
        self.memory[idx] = val & MASK32

    def ds_min_u32(self, vaddr: int, vdata0: int, offset: int = 0) -> int:
        eff_addr = (vaddr + offset) & MASK32
        idx = (eff_addr >> 2) & (self.lds_dwords - 1)
        old_val = self.memory.get(idx, 0)
        self.memory[idx] = ref_min_u32(old_val, vdata0)
        return old_val


class TestDsMinU32ReferenceOracle(unittest.TestCase):
    """Test suite verifying the reference mathematical oracle and architectural LDS model."""

    def test_ref_min_u32_truth_table(self):
        cases = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (5, 10, 5),
            (10, 5, 5),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),  # Unsigned: 0x7FFFFFFF < 0x80000000
            (0x80000000, 0x7FFFFFFF, 0x7FFFFFFF),
            (0, 0xFFFFFFFF, 0),
            (0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x80000000),
            (0x80000001, 0x80000000, 0x80000000),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_min_u32(a, b), expected)

    def test_lds_model_basic_atomic_min(self):
        lds = LDSModel32(4096)
        lds.write_dword(0x100, 100)
        old = lds.ds_min_u32(0x100, 42)
        self.assertEqual(old, 100)
        self.assertEqual(lds.read_dword(0x100), 42)

    def test_lds_model_offset_addition_and_alignment(self):
        lds = LDSModel32(4096)
        lds.write_dword(0x44, 500)
        # vaddr 0x40 + offset 4 = 0x44
        old = lds.ds_min_u32(0x40, 250, offset=4)
        self.assertEqual(old, 500)
        self.assertEqual(lds.read_dword(0x44), 250)
        # Unaligned byte address: low 2 bits are dropped (dword aligned)
        lds.ds_min_u32(0x45, 100, offset=1)  # eff_addr = 0x46 -> dword idx 0x44 >> 2
        self.assertEqual(lds.read_dword(0x44), 100)

    def test_lds_model_address_wrapping(self):
        lds = LDSModel32(4096)  # 1024 dwords -> 0x1000 bytes
        lds.write_dword(0x0, 999)
        # 0x1000 wraps to 0x0
        lds.ds_min_u32(0x1000, 50)
        self.assertEqual(lds.read_dword(0x0), 50)


# -----------------------------------------------------------------------------
# Part 2: AST Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int

    def __str__(self) -> str:
        return f"{self.kind.value}{self.index}"


class DsOpcode(enum.Enum):
    DS_MIN_U32 = "ds_min_u32"
    DS_MIN_RTN_U32 = "ds_min_rtn_u32"


@dataclasses.dataclass
class DsInstruction:
    opcode: DsOpcode
    vaddr: Register
    vdata0: Register
    vdst: Register | None = None
    offset: int = 0

    def format_asm(self) -> str:
        off_str = f" offset:{self.offset}" if self.offset != 0 else ""
        if self.opcode == DsOpcode.DS_MIN_RTN_U32:
            assert self.vdst is not None
            return f"{self.opcode.value} {self.vdst}, {self.vaddr}, {self.vdata0}{off_str}"
        return f"{self.opcode.value} {self.vaddr}, {self.vdata0}{off_str}"

    @classmethod
    def parse_asm(cls, line: str) -> DsInstruction:
        line = line.strip()
        parts = line.split(None, 1)
        op_str = parts[0]
        opcode = DsOpcode(op_str)
        args_str = parts[1] if len(parts) > 1 else ""

        offset = 0
        tokens = [t.strip() for t in args_str.split(",") if t.strip()]
        last_token = tokens[-1] if tokens else ""
        if "offset:" in last_token:
            m = re.search(r"offset:(\w+)", last_token)
            if m:
                raw_off = m.group(1)
                offset = int(raw_off, 16) if raw_off.startswith("0x") else int(raw_off)
            tokens[-1] = re.sub(r"\s*offset:\w+", "", last_token).strip()
            if not tokens[-1]:
                tokens.pop()

        def parse_reg(name: str) -> Register:
            name = name.strip()
            if name.startswith("v"):
                return Register(RegKind.VGPR, int(name[1:]))
            if name.startswith("s"):
                return Register(RegKind.SGPR, int(name[1:]))
            raise ValueError(f"Invalid register name: {name}")

        if opcode == DsOpcode.DS_MIN_RTN_U32:
            vdst = parse_reg(tokens[0])
            vaddr = parse_reg(tokens[1])
            vdata0 = parse_reg(tokens[2])
            if vdst.kind != RegKind.VGPR or vaddr.kind != RegKind.VGPR or vdata0.kind != RegKind.VGPR:
                raise ValueError("LDS atomic operations require VGPRs for all operands")
            return cls(opcode=opcode, vaddr=vaddr, vdata0=vdata0, vdst=vdst, offset=offset)
        else:
            vaddr = parse_reg(tokens[0])
            vdata0 = parse_reg(tokens[1])
            if vaddr.kind != RegKind.VGPR or vdata0.kind != RegKind.VGPR:
                raise ValueError("LDS atomic operations require VGPRs for all operands")
            return cls(opcode=opcode, vaddr=vaddr, vdata0=vdata0, vdst=None, offset=offset)


@dataclasses.dataclass
class DsMinU32Expr:
    addr_expr: str
    data_expr: str
    old_val_expr: str

    def evaluate(self, addr_val: int, data_val: int, mem_val: int) -> int:
        return ref_min_u32(mem_val, data_val)


class TestDsMinU32AST(unittest.TestCase):
    """Test suite verifying AST definitions and assembly roundtripping."""

    def test_instruction_formatting(self):
        inst1 = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        self.assertEqual(inst1.format_asm(), "ds_min_u32 v0, v1")

        inst2 = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1), offset=16)
        self.assertEqual(inst2.format_asm(), "ds_min_u32 v0, v1 offset:16")

        inst3 = DsInstruction(DsOpcode.DS_MIN_RTN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1),
                              vdst=Register(RegKind.VGPR, 2), offset=32)
        self.assertEqual(inst3.format_asm(), "ds_min_rtn_u32 v2, v0, v1 offset:32")

    def test_instruction_parsing(self):
        line1 = "ds_min_u32 v0, v1"
        inst1 = DsInstruction.parse_asm(line1)
        self.assertEqual(inst1.opcode, DsOpcode.DS_MIN_U32)
        self.assertEqual(inst1.vaddr, Register(RegKind.VGPR, 0))
        self.assertEqual(inst1.vdata0, Register(RegKind.VGPR, 1))
        self.assertIsNone(inst1.vdst)
        self.assertEqual(inst1.offset, 0)

        line2 = "ds_min_rtn_u32 v5, v1, v2 offset:64"
        inst2 = DsInstruction.parse_asm(line2)
        self.assertEqual(inst2.opcode, DsOpcode.DS_MIN_RTN_U32)
        self.assertEqual(inst2.vdst, Register(RegKind.VGPR, 5))
        self.assertEqual(inst2.vaddr, Register(RegKind.VGPR, 1))
        self.assertEqual(inst2.vdata0, Register(RegKind.VGPR, 2))
        self.assertEqual(inst2.offset, 64)

    def test_symbolic_expression_evaluation(self):
        expr = DsMinU32Expr("v0 + offset", "v1", "LDS[addr]")
        res = expr.evaluate(0x10, 50, 100)
        self.assertEqual(res, 50)


# -----------------------------------------------------------------------------
# Part 3: Deterministic Scalar Instruction Interpreter & Translator Integration
# -----------------------------------------------------------------------------

class ScalarRegisterFile:
    def __init__(self):
        self.vgprs: dict[int, int] = {}
        self.sgprs: dict[int, int] = {}

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0)

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarLDSInterpreter:
    def __init__(self, lds: LDSModel32 | None = None):
        self.rf = ScalarRegisterFile()
        self.lds = lds or LDSModel32()

    def execute(self, inst: DsInstruction) -> None:
        vaddr = self.rf.get_vgpr(inst.vaddr.index)
        vdata0 = self.rf.get_vgpr(inst.vdata0.index)
        old_val = self.lds.ds_min_u32(vaddr, vdata0, inst.offset)
        if inst.opcode == DsOpcode.DS_MIN_RTN_U32:
            assert inst.vdst is not None
            self.rf.set_vgpr(inst.vdst.index, old_val)


class TestDsMinU32ScalarInterpreter(unittest.TestCase):
    """Test suite for the scalar interpreter and rdna2spirv translator verification."""

    def test_scalar_interpreter_chain(self):
        interp = ScalarLDSInterpreter()
        interp.lds.write_dword(0x20, 1000)

        # Non-returning ds_min_u32
        interp.rf.set_vgpr(0, 0x20)
        interp.rf.set_vgpr(1, 500)
        inst1 = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        interp.execute(inst1)
        self.assertEqual(interp.lds.read_dword(0x20), 500)

        # Returning ds_min_rtn_u32
        interp.rf.set_vgpr(1, 200)
        inst2 = DsInstruction(DsOpcode.DS_MIN_RTN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1),
                              vdst=Register(RegKind.VGPR, 2))
        interp.execute(inst2)
        self.assertEqual(interp.rf.get_vgpr(2), 500)  # Old value
        self.assertEqual(interp.lds.read_dword(0x20), 200)  # New min

    @unittest.skipIf(not TRANSLATOR_AVAILABLE, "Translator not available in this environment")
    def test_translator_spv_emission(self):
        """Verify that tools/rdna2spirv.py emits OpAtomicUMin for ds_min_u32 and ds_min_rtn_u32."""
        spv = L.translate([
            "ds_min_u32 v0, v1",
            "ds_min_rtn_u32 v2, v0, v1 offset:8"
        ], name="cs_ds_min_u32_spv")
        self.assertIn("OpAtomicUMin", spv)
        self.assertIn("%_ptr_WG_uint", spv)
        lines = [ln.strip() for ln in spv.splitlines() if "OpAtomicUMin" in ln]
        self.assertEqual(len(lines), 2)

    @unittest.skipIf(not TRANSLATOR_AVAILABLE, "Translator not available in this environment")
    def test_translator_lds_run_chain(self):
        """Verify execution through tools/rdna2spirv.py and spirv_mini_interp via lds_run."""
        ops = [
            ("ds_min_rtn_u32", 0x40, 0, (50,)),
            ("ds_min_u32", 0x40, 0, (20,)),
            ("ds_min_rtn_u32", 0x40, 0, (70,)),  # 70 > 20, min stays 20
        ]
        rtns, fin = L.run_chain(ops, {0x40: 100})
        self.assertEqual(rtns, [100, 0xDEAD, 20])
        self.assertEqual(fin, {0x40: 20})


# -----------------------------------------------------------------------------
# Part 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDLDSWavefrontInterpreter:
    """Wavefront interpreter modeling multi-lane divergence and lane masking."""

    def __init__(self, wave_size: int = 32, lds: LDSModel32 | None = None):
        assert wave_size in (32, 64)
        self.wave_size = wave_size
        self.lds = lds or LDSModel32()
        self.exec_mask: int = (1 << wave_size) - 1
        self.lane_vgprs: list[dict[int, int]] = [{} for _ in range(wave_size)]

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.lane_vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.lane_vgprs[lane].get(reg, 0)

    def execute(self, inst: DsInstruction, arbitration_order: list[int] | None = None) -> None:
        order = arbitration_order or list(range(self.wave_size))
        for lane in order:
            if not ((self.exec_mask >> lane) & 1):
                continue
            vaddr = self.get_lane_vgpr(lane, inst.vaddr.index)
            vdata0 = self.get_lane_vgpr(lane, inst.vdata0.index)
            old_val = self.lds.ds_min_u32(vaddr, vdata0, inst.offset)
            if inst.opcode == DsOpcode.DS_MIN_RTN_U32:
                assert inst.vdst is not None
                self.set_lane_vgpr(lane, inst.vdst.index, old_val)


class TestDsMinU32SIMDMultiLane(unittest.TestCase):
    """Test suite verifying SIMD divergence, Wave32/Wave64 EXEC masking, and lane contention."""

    def test_wave32_divergence_even_lanes(self):
        simd = SIMDLDSWavefrontInterpreter(wave_size=32)
        simd.exec_mask = 0x55555555  # Even lanes active (0, 2, 4...)
        simd.lds.write_dword(0x10, 1000)

        for lane in range(32):
            simd.set_lane_vgpr(lane, 0, 0x10)
            simd.set_lane_vgpr(lane, 1, 100 + lane)

        inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.execute(inst)

        # Min among even lanes: lane 0 provided 100, which is min
        self.assertEqual(simd.lds.read_dword(0x10), 100)

    def test_wave64_exec_masking(self):
        simd = SIMDLDSWavefrontInterpreter(wave_size=64)
        simd.exec_mask = 0x00000000FFFFFFFF  # Low 32 lanes active, high 32 masked off
        simd.lds.write_dword(0x20, 1000)

        for lane in range(64):
            simd.set_lane_vgpr(lane, 0, 0x20)
            # High lanes have smaller values, but are masked off!
            simd.set_lane_vgpr(lane, 1, 500 if lane < 32 else 5)

        inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.execute(inst)

        self.assertEqual(simd.lds.read_dword(0x20), 500)

    def test_lane_contention_commutativity_and_associativity(self):
        """Atomic unsigned minimum is commutative: any arbitration order converges to the same result."""
        values = [random.randint(10, 5000) for _ in range(32)]
        expected_min = min(values)

        for trial in range(5):
            order = list(range(32))
            random.shuffle(order)

            simd = SIMDLDSWavefrontInterpreter(wave_size=32)
            simd.lds.write_dword(0x30, 0xFFFFFFFF)
            for lane in range(32):
                simd.set_lane_vgpr(lane, 0, 0x30)
                simd.set_lane_vgpr(lane, 1, values[lane])

            inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
            simd.execute(inst, arbitration_order=order)
            self.assertEqual(simd.lds.read_dword(0x30), expected_min)

    def test_multi_bank_parallel_updates(self):
        """Lanes write to 32 distinct LDS banks simultaneously."""
        simd = SIMDLDSWavefrontInterpreter(wave_size=32)
        for lane in range(32):
            addr = lane * 4
            simd.lds.write_dword(addr, 200 + lane)
            simd.set_lane_vgpr(lane, 0, addr)
            simd.set_lane_vgpr(lane, 1, 50 + lane)

        inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.execute(inst)

        for lane in range(32):
            self.assertEqual(simd.lds.read_dword(lane * 4), 50 + lane)


# -----------------------------------------------------------------------------
# Part 5: Register Aliasing & In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestDsMinU32RegisterAliasing(unittest.TestCase):
    """Test suite verifying register aliasing semantics and operand preservation."""

    def test_in_place_operand_preservation_non_returning(self):
        interp = ScalarLDSInterpreter()
        interp.lds.write_dword(0x40, 100)
        interp.rf.set_vgpr(0, 0x40)
        interp.rf.set_vgpr(1, 25)

        inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        interp.execute(inst)

        # Memory updated
        self.assertEqual(interp.lds.read_dword(0x40), 25)
        # Operands unchanged in non-returning form
        self.assertEqual(interp.rf.get_vgpr(0), 0x40)
        self.assertEqual(interp.rf.get_vgpr(1), 25)

    def test_addr_data_aliasing(self):
        interp = ScalarLDSInterpreter()
        # vaddr == vdata0: v0 is both address and data
        addr_val = 0x10
        interp.lds.write_dword(addr_val, 100)
        interp.rf.set_vgpr(0, addr_val)

        inst = DsInstruction(DsOpcode.DS_MIN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 0))
        interp.execute(inst)

        self.assertEqual(interp.lds.read_dword(addr_val), min(100, addr_val))
        self.assertEqual(interp.rf.get_vgpr(0), addr_val)

    def test_dst_addr_aliasing_returning(self):
        interp = ScalarLDSInterpreter()
        interp.lds.write_dword(0x20, 500)
        interp.rf.set_vgpr(0, 0x20)
        interp.rf.set_vgpr(1, 100)

        # vdst == vaddr: v0 gets old memory value
        inst = DsInstruction(DsOpcode.DS_MIN_RTN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1),
                              vdst=Register(RegKind.VGPR, 0))
        interp.execute(inst)

        self.assertEqual(interp.rf.get_vgpr(0), 500)  # Old memory value
        self.assertEqual(interp.lds.read_dword(0x20), 100)

    def test_dst_data_aliasing_returning(self):
        interp = ScalarLDSInterpreter()
        interp.lds.write_dword(0x30, 800)
        interp.rf.set_vgpr(0, 0x30)
        interp.rf.set_vgpr(1, 300)

        # vdst == vdata0: v1 gets old memory value
        inst = DsInstruction(DsOpcode.DS_MIN_RTN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1),
                              vdst=Register(RegKind.VGPR, 1))
        interp.execute(inst)

        self.assertEqual(interp.rf.get_vgpr(1), 800)
        self.assertEqual(interp.lds.read_dword(0x30), 300)

    def test_dst_addr_data_all_alias(self):
        interp = ScalarLDSInterpreter()
        interp.lds.write_dword(0x40, 1000)
        interp.rf.set_vgpr(0, 0x40)

        # vdst == vaddr == vdata0
        inst = DsInstruction(DsOpcode.DS_MIN_RTN_U32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 0),
                              vdst=Register(RegKind.VGPR, 0))
        interp.execute(inst)

        self.assertEqual(interp.rf.get_vgpr(0), 1000)
        self.assertEqual(interp.lds.read_dword(0x40), 0x40)
