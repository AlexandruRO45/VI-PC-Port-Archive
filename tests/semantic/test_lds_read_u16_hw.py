"""ds_read_u16: 16-bit unsigned LDS read with zero extension into VGPR.

RDNA2 ISA Data Share Instructions:
  ds_read_u16 vdst, vaddr [offset:offset]
Effective byte address ADDR = (vaddr + offset) % LDS_SIZE.
Reads a 16-bit halfword from LDS memory:
  D[15:0]  = MEM[ADDR][15:0]
  D[31:16] = 0 (zero-extended)
"""
from __future__ import annotations

import enum
import math
import os
import re
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L
import rdna2spirv
import support

LDS_DEFAULT_SIZE = 65536
M32 = 0xFFFFFFFF
M16 = 0xFFFF


def ds_read_u16_ref(lds_bytes: bytearray, vaddr: int, offset: int = 0, lds_size: int = LDS_DEFAULT_SIZE) -> int:
    eff_addr = (vaddr + offset) % lds_size
    b0 = lds_bytes[eff_addr]
    b1 = lds_bytes[(eff_addr + 1) % lds_size]
    return (b0 | (b1 << 8)) & M16


class TestReferenceOracle(unittest.TestCase):
    def setUp(self) -> None:
        self.lds = bytearray(LDS_DEFAULT_SIZE)

    def test_basic_zero_offset(self) -> None:
        self.lds[0x100] = 0x34
        self.lds[0x101] = 0x12
        got = ds_read_u16_ref(self.lds, 0x100)
        self.assertEqual(got, 0x1234)

    def test_with_positive_offset(self) -> None:
        self.lds[0x208] = 0xEF
        self.lds[0x209] = 0xBE
        got = ds_read_u16_ref(self.lds, 0x200, offset=8)
        self.assertEqual(got, 0xBEEF)

    def test_unaligned_halfword_address(self) -> None:
        self.lds[0x301] = 0x78
        self.lds[0x302] = 0x56
        got = ds_read_u16_ref(self.lds, 0x301)
        self.assertEqual(got, 0x5678)

    def test_zero_extension_guarantee(self) -> None:
        self.lds[0x400] = 0x00
        self.lds[0x401] = 0x80
        got = ds_read_u16_ref(self.lds, 0x400)
        self.assertEqual(got, 0x00008000)
        self.assertEqual(got >> 16, 0)

    def test_all_ones_16bit(self) -> None:
        self.lds[0x500] = 0xFF
        self.lds[0x501] = 0xFF
        got = ds_read_u16_ref(self.lds, 0x500)
        self.assertEqual(got, 0x0000FFFF)
        self.assertEqual(got & ~0xFFFF, 0)

    def test_address_wrapping_at_lds_boundary(self) -> None:
        self.lds[LDS_DEFAULT_SIZE - 1] = 0xAA
        self.lds[0] = 0x55
        got = ds_read_u16_ref(self.lds, LDS_DEFAULT_SIZE - 1)
        self.assertEqual(got, 0x55AA)

    def test_large_offset_wrap(self) -> None:
        self.lds[0x10] = 0x42
        self.lds[0x11] = 0x24
        got = ds_read_u16_ref(self.lds, 0x10, offset=LDS_DEFAULT_SIZE)
        self.assertEqual(got, 0x2442)


class RegType(enum.Enum):
    VGPR = "v"
    SGPR = "s"


class RegSingle:
    def __init__(self, reg_type: RegType, index: int) -> None:
        self.reg_type = reg_type
        self.index = index

    def __repr__(self) -> str:
        return f"{self.reg_type.value}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegSingle):
            return False
        return self.reg_type == other.reg_type and self.index == other.index


class DSReadU16Inst:
    def __init__(self, vdst: RegSingle, vaddr: RegSingle, offset: int = 0) -> None:
        self.vdst = vdst
        self.vaddr = vaddr
        self.offset = offset

    def emit_asm(self) -> str:
        if self.offset != 0:
            return f"ds_read_u16 {self.vdst}, {self.vaddr} offset:{self.offset}"
        return f"ds_read_u16 {self.vdst}, {self.vaddr}"


def parse_ds_read_u16(line: str) -> DSReadU16Inst:
    line = line.strip()
    match = re.match(r"^ds_read_u16\s+(v\d+),\s*(v\d+)(?:\s+offset:(\d+|0x[0-9a-fA-F]+))?$", line)
    if not match:
        raise ValueError(f"Invalid ds_read_u16 syntax: {line}")
    vdst_str, vaddr_str, off_str = match.groups()
    vdst = RegSingle(RegType.VGPR, int(vdst_str[1:]))
    vaddr = RegSingle(RegType.VGPR, int(vaddr_str[1:]))
    offset = int(off_str, 0) if off_str is not None else 0
    if offset < 0 or offset > 0xFFFF:
        raise ValueError(f"Offset out of 16-bit range: {offset}")
    return DSReadU16Inst(vdst, vaddr, offset)


class TestASTDefinitions(unittest.TestCase):
    def test_ast_construction_and_emit(self) -> None:
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 5), RegSingle(RegType.VGPR, 1), 16)
        self.assertEqual(inst.emit_asm(), "ds_read_u16 v5, v1 offset:16")

    def test_ast_construction_zero_offset(self) -> None:
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 0), 0)
        self.assertEqual(inst.emit_asm(), "ds_read_u16 v2, v0")

    def test_parse_valid_assembler(self) -> None:
        inst = parse_ds_read_u16("ds_read_u16 v10, v2 offset:0x20")
        self.assertEqual(inst.vdst, RegSingle(RegType.VGPR, 10))
        self.assertEqual(inst.vaddr, RegSingle(RegType.VGPR, 2))
        self.assertEqual(inst.offset, 32)

    def test_parse_hex_and_dec_offsets(self) -> None:
        i1 = parse_ds_read_u16("ds_read_u16 v1, v2 offset:64")
        i2 = parse_ds_read_u16("ds_read_u16 v1, v2 offset:0x40")
        self.assertEqual(i1.offset, i2.offset)

    def test_parse_invalid_registers(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_read_u16("ds_read_u16 s0, v1")
        with self.assertRaises(ValueError):
            parse_ds_read_u16("ds_read_u16 v0, s1")


class ScalarState:
    def __init__(self, lds_size: int = LDS_DEFAULT_SIZE) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.lds = bytearray(lds_size)
        self.lds_size = lds_size
        self.exec_lo = 0xFFFFFFFF
        self.exec_hi = 0xFFFFFFFF
        self.scc = 0
        self.vcc_lo = 0
        self.vcc_hi = 0

    def write_lds_u16(self, addr: int, val: int) -> None:
        eff = addr % self.lds_size
        self.lds[eff] = val & 0xFF
        self.lds[(eff + 1) % self.lds_size] = (val >> 8) & 0xFF

    def write_lds_u32(self, addr: int, val: int) -> None:
        eff = addr % self.lds_size
        for k in range(4):
            self.lds[(eff + k) % self.lds_size] = (val >> (k * 8)) & 0xFF

    def step(self, inst: DSReadU16Inst) -> None:
        if (self.exec_lo & 1) == 0:
            return
        addr = self.vgprs[inst.vaddr.index]
        res = ds_read_u16_ref(self.lds, addr, inst.offset, self.lds_size)
        self.vgprs[inst.vdst.index] = res


class TestScalarInterpreter(unittest.TestCase):
    def setUp(self) -> None:
        self.state = ScalarState()

    def test_scalar_read_execution(self) -> None:
        self.state.write_lds_u16(0x80, 0xABCD)
        self.state.vgprs[1] = 0x80
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[4], 0xABCD)

    def test_scalar_read_with_offset(self) -> None:
        self.state.write_lds_u16(0x84, 0x1122)
        self.state.vgprs[2] = 0x80
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 5), RegSingle(RegType.VGPR, 2), offset=4)
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[5], 0x1122)

    def test_scalar_inactive_exec(self) -> None:
        self.state.exec_lo = 0
        self.state.vgprs[1] = 0x80
        self.state.vgprs[4] = 0xCAFE
        self.state.write_lds_u16(0x80, 0xBEEF)
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[4], 0xCAFE)

    def test_scalar_upper_bits_clear(self) -> None:
        self.state.vgprs[3] = 0xDEADBEEF
        self.state.vgprs[0] = 0x10
        self.state.write_lds_u16(0x10, 0x7FFF)
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 3), RegSingle(RegType.VGPR, 0))
        self.state.step(inst)
        self.assertEqual(self.state.vgprs[3], 0x00007FFF)


class SimdState:
    def __init__(self, num_lanes: int = 32, lds_size: int = LDS_DEFAULT_SIZE) -> None:
        self.num_lanes = num_lanes
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.lds = bytearray(lds_size)
        self.lds_size = lds_size
        self.exec_mask = (1 << num_lanes) - 1

    def step(self, inst: DSReadU16Inst) -> None:
        for lane in range(self.num_lanes):
            if (self.exec_mask >> lane) & 1:
                vaddr = self.vgprs[inst.vaddr.index][lane]
                val = ds_read_u16_ref(self.lds, vaddr, inst.offset, self.lds_size)
                self.vgprs[inst.vdst.index][lane] = val


class TestSimdDivergence(unittest.TestCase):
    def test_simd_independent_lane_addresses(self) -> None:
        simd = SimdState(num_lanes=32)
        for lane in range(32):
            simd.vgprs[1][lane] = lane * 2
            simd.lds[lane * 2] = lane & 0xFF
            simd.lds[lane * 2 + 1] = 0x10 + (lane & 0x0F)
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 4), RegSingle(RegType.VGPR, 1))
        simd.step(inst)
        for lane in range(32):
            expected = (lane & 0xFF) | ((0x10 + (lane & 0x0F)) << 8)
            self.assertEqual(simd.vgprs[4][lane], expected, f"lane {lane}")

    def test_wave32_exec_masking(self) -> None:
        simd = SimdState(num_lanes=32)
        simd.exec_mask = 0x55555555
        for lane in range(32):
            simd.vgprs[1][lane] = 0x100
            simd.vgprs[2][lane] = 0xDEAD
        simd.lds[0x100] = 0x77
        simd.lds[0x101] = 0x88
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 1))
        simd.step(inst)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(simd.vgprs[2][lane], 0x8877)
            else:
                self.assertEqual(simd.vgprs[2][lane], 0xDEAD)

    def test_wave64_divergence(self) -> None:
        simd = SimdState(num_lanes=64)
        simd.exec_mask = 0xFFFFFFFFFFFFFFFF
        for lane in range(64):
            simd.vgprs[0][lane] = 0x200 + lane * 4
            simd.lds[0x200 + lane * 4] = lane
            simd.lds[0x200 + lane * 4 + 1] = 0x80
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 5), RegSingle(RegType.VGPR, 0))
        simd.step(inst)
        for lane in range(64):
            self.assertEqual(simd.vgprs[5][lane], 0x8000 | lane)


class TestRegisterAliasing(unittest.TestCase):
    def test_inplace_read_vdst_equals_vaddr(self) -> None:
        state = ScalarState()
        state.write_lds_u16(0x304, 0x55AA)
        state.vgprs[1] = 0x300
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 1), RegSingle(RegType.VGPR, 1), offset=4)
        state.step(inst)
        self.assertEqual(state.vgprs[1], 0x55AA)

    def test_chained_dependent_reads(self) -> None:
        state = ScalarState()
        state.write_lds_u16(0x100, 0x200)
        state.write_lds_u16(0x200, 0x4321)
        state.vgprs[0] = 0x100
        state.step(DSReadU16Inst(RegSingle(RegType.VGPR, 1), RegSingle(RegType.VGPR, 0)))
        self.assertEqual(state.vgprs[1], 0x200)
        state.step(DSReadU16Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 1)))
        self.assertEqual(state.vgprs[2], 0x4321)

    def test_neighbor_registers_unmodified(self) -> None:
        state = ScalarState()
        for i in range(10):
            state.vgprs[i] = 0x1000 + i
        state.write_lds_u16(0x40, 0x9999)
        state.vgprs[4] = 0x40
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 5), RegSingle(RegType.VGPR, 4))
        state.step(inst)
        self.assertEqual(state.vgprs[5], 0x9999)
        self.assertEqual(state.vgprs[4], 0x40)
        for i in range(10):
            if i in (4, 5):
                continue
            self.assertEqual(state.vgprs[i], 0x1000 + i)


class TestConditionCodes(unittest.TestCase):
    def test_flags_unmodified_on_zero_read(self) -> None:
        state = ScalarState()
        state.scc = 1
        state.vcc_lo = 0x12345678
        state.vcc_hi = 0x9ABCDEF0
        state.write_lds_u16(0x50, 0x0000)
        state.vgprs[1] = 0x50
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 1))
        state.step(inst)
        self.assertEqual(state.vgprs[2], 0)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc_lo, 0x12345678)
        self.assertEqual(state.vcc_hi, 0x9ABCDEF0)

    def test_flags_unmodified_on_msb_read(self) -> None:
        state = ScalarState()
        state.scc = 0
        state.vcc_lo = 0xFFFFFFFF
        state.write_lds_u16(0x60, 0x8000)
        state.vgprs[1] = 0x60
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 3), RegSingle(RegType.VGPR, 1))
        state.step(inst)
        self.assertEqual(state.vgprs[3], 0x8000)
        self.assertEqual(state.scc, 0)
        self.assertEqual(state.vcc_lo, 0xFFFFFFFF)


class TestIEEE754EdgeCases(unittest.TestCase):
    def test_half_float_bit_patterns(self) -> None:
        cases = {
            "pos_zero": 0x0000,
            "neg_zero": 0x8000,
            "min_subnormal": 0x0001,
            "max_subnormal": 0x03FF,
            "min_normal": 0x0400,
            "one": 0x3C00,
            "max_normal": 0x7BFF,
            "pos_infinity": 0x7C00,
            "neg_infinity": 0xFC00,
            "quiet_nan": 0x7E00,
            "signaling_nan": 0x7D00,
        }
        state = ScalarState()
        addr = 0x100
        for name, bits in cases.items():
            state.write_lds_u16(addr, bits)
            state.vgprs[1] = addr
            inst = DSReadU16Inst(RegSingle(RegType.VGPR, 2), RegSingle(RegType.VGPR, 1))
            state.step(inst)
            self.assertEqual(state.vgprs[2], bits, f"Failed for IEEE 754 pattern {name}")
            addr += 2

    def test_subnormals_not_flushed(self) -> None:
        state = ScalarState()
        state.write_lds_u16(0x20, 0x0042)
        state.vgprs[1] = 0x20
        inst = DSReadU16Inst(RegSingle(RegType.VGPR, 3), RegSingle(RegType.VGPR, 1))
        state.step(inst)
        self.assertEqual(state.vgprs[3], 0x0042)


class TestNegativeControls(unittest.TestCase):
    def test_reject_sgpr_dest(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_read_u16("ds_read_u16 s1, v2")

    def test_reject_sgpr_addr(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_read_u16("ds_read_u16 v1, s2")

    def test_reject_overflow_offset(self) -> None:
        with self.assertRaises(ValueError):
            parse_ds_read_u16("ds_read_u16 v1, v2 offset:65536")

    def test_reject_malformed_syntax(self) -> None:
        for bad in ("ds_read_u16", "ds_read_u16 v1", "ds_read_u16 v1, v2, v3", "ds_read_u16 v1, v2 offset:"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_ds_read_u16(bad)


class TestTranslatorExecution(unittest.TestCase):
    """Direct translation and execution via support.translate_lines and spirv_mini_interp."""

    def test_translated_ds_read_u16_compute(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 3, v0",
            "v_mov_b32_e32 v2, 0x12345678",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v3, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v4, v1 offset:2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0) + L.store("v4", 1)

        mem = L.run(body, lanes=32)
        for i in range(32):
            self.assertEqual(L.out(mem, i, 0), 0x5678, f"lane {i} low halfword")
            self.assertEqual(L.out(mem, i, 1), 0x1234, f"lane {i} high halfword")

    def test_translated_unaligned_and_offset_reads(self) -> None:
        body = [
            "v_lshlrev_b32_e32 v1, 3, v0",
            "v_mov_b32_e32 v2, 0x7F80FF01",
            "ds_write_b32 v1, v2 offset:0x20",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_u16 v5, v1 offset:0x20",
            "ds_read_u16 v6, v1 offset:0x21",
            "ds_read_u16 v7, v1 offset:0x22",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v5", 0) + L.store("v6", 1) + L.store("v7", 2)

        mem = L.run(body, lanes=8)
        for i in range(8):
            self.assertEqual(L.out(mem, i, 0), 0xFF01, f"lane {i} u16 @ +0")
            self.assertEqual(L.out(mem, i, 1), 0x80FF, f"lane {i} u16 @ +1 (unaligned)")
            self.assertEqual(L.out(mem, i, 2), 0x7F80, f"lane {i} u16 @ +2")


if __name__ == "__main__":
    unittest.main()
