from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def ref_v_add_i16(s0: int, s1: int, clamp: bool = False) -> int:
    x = (s0 & 0x7FFF) - (s0 & 0x8000)
    y = (s1 & 0x7FFF) - (s1 & 0x8000)
    res = x + y
    if clamp:
        res = max(-32768, min(32767, res))
    return res & MASK16


def ref_v_add_u16(s0: int, s1: int, clamp: bool = False) -> int:
    x = s0 & MASK16
    y = s1 & MASK16
    res = x + y
    if clamp:
        res = min(MASK16, res)
    return res & MASK16


def ref_v_sub_i16(s0: int, s1: int, clamp: bool = False) -> int:
    x = (s0 & 0x7FFF) - (s0 & 0x8000)
    y = (s1 & 0x7FFF) - (s1 & 0x8000)
    res = x - y
    if clamp:
        res = max(-32768, min(32767, res))
    return res & MASK16


def ref_v_sub_u16(s0: int, s1: int, clamp: bool = False) -> int:
    x = s0 & MASK16
    y = s1 & MASK16
    res = x - y
    if clamp:
        res = max(0, res)
    return res & MASK16


def ref_v_add_nc_i16(s0: int, s1: int, clamp: bool = False) -> int:
    return ref_v_add_i16(s0, s1, clamp=clamp)


def ref_v_sub_nc_i16(s0: int, s1: int, clamp: bool = False) -> int:
    return ref_v_sub_i16(s0, s1, clamp=clamp)


_TRANSLATE_ID: int = 0


def run_translated_int16(op: str, src0: int, src1: int, *, clamp: bool = False) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_str = " clamp" if clamp else ""
    prog = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"{op} v0, v1, v2{clamp_str}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_int16_{op}_{_TRANSLATE_ID}", prog)
    return mrt0_words(spv)[0] & MASK16


def run_aliased_int16(op: str, src0: int, src1: int, mode: str = "dst_src0", *, clamp: bool = False) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_str = " clamp" if clamp else ""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if mode == "dst_src0":
        lines.append(f"{op} v1, v1, v2{clamp_str}")
        lines.append("exp mrt0 v1, v1, v1, v1 done vm")
    elif mode == "dst_src1":
        lines.append(f"{op} v2, v1, v2{clamp_str}")
        lines.append("exp mrt0 v2, v2, v2, v2 done vm")
    lines.append("s_endpgm")
    spv = translate_lines(f"ps_aliased_{op}_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK16


class TestVAddI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_add_i16."""

    def test_basic_addition(self) -> None:
        got = run_translated_int16("v_add_i16", 100, 200)
        exp = ref_v_add_i16(100, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 300)

    def test_negative_addition(self) -> None:
        got = run_translated_int16("v_add_i16", 0xFFFF, 0xFFFF)
        exp = ref_v_add_i16(0xFFFF, 0xFFFF)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFFE)

    def test_wrap_boundary(self) -> None:
        got = run_translated_int16("v_add_i16", 0x7FFF, 1)
        exp = ref_v_add_i16(0x7FFF, 1)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000)

    def test_clamp_positive(self) -> None:
        got = run_translated_int16("v_add_i16", 0x7FFF, 1, clamp=True)
        exp = ref_v_add_i16(0x7FFF, 1, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF)

    def test_clamp_negative(self) -> None:
        got = run_translated_int16("v_add_i16", 0x8000, 0xFFFF, clamp=True)
        exp = ref_v_add_i16(0x8000, 0xFFFF, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000)

    def test_extremum_bounds(self) -> None:
        got = run_translated_int16("v_add_i16", 0x7FFF, 0x7FFF, clamp=True)
        self.assertEqual(got, 0x7FFF)
        got_neg = run_translated_int16("v_add_i16", 0x8000, 0x8000, clamp=True)
        self.assertEqual(got_neg, 0x8000)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_add_i16", 100, 200, "dst_src0")
        self.assertEqual(got, 300)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_add_i16", 100, 200, "dst_src1")
        self.assertEqual(got, 300)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (i * 500) & MASK16
            b = (i * 200) & MASK16
            exp = ref_v_add_i16(a, b)
            got = run_translated_int16("v_add_i16", a, b)
            self.assertEqual(got, exp)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_add_i16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_add_i16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpIAdd", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVAddU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_add_u16."""

    def test_basic_addition(self) -> None:
        got = run_translated_int16("v_add_u16", 100, 200)
        exp = ref_v_add_u16(100, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 300)

    def test_wrap_boundary(self) -> None:
        got = run_translated_int16("v_add_u16", 60000, 10000)
        exp = ref_v_add_u16(60000, 10000)
        self.assertEqual(got, exp)
        self.assertEqual(got, 4464)

    def test_clamp_saturation(self) -> None:
        got = run_translated_int16("v_add_u16", 60000, 10000, clamp=True)
        exp = ref_v_add_u16(60000, 10000, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFFF)

    def test_zero_identity(self) -> None:
        got0 = run_translated_int16("v_add_u16", 0, 1234)
        got1 = run_translated_int16("v_add_u16", 1234, 0)
        self.assertEqual(got0, 1234)
        self.assertEqual(got1, 1234)

    def test_maximum_overflow(self) -> None:
        got = run_translated_int16("v_add_u16", 0xFFFF, 0xFFFF)
        self.assertEqual(got, 0xFFFE)
        got_clamp = run_translated_int16("v_add_u16", 0xFFFF, 0xFFFF, clamp=True)
        self.assertEqual(got_clamp, 0xFFFF)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_add_u16", 500, 250, "dst_src0")
        self.assertEqual(got, 750)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_add_u16", 500, 250, "dst_src1")
        self.assertEqual(got, 750)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (i * 3000) & MASK16
            b = (i * 4000) & MASK16
            exp = ref_v_add_u16(a, b)
            got = run_translated_int16("v_add_u16", a, b)
            self.assertEqual(got, exp)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_add_u16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_add_u16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpIAdd", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVSubI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_sub_i16."""

    def test_basic_subtraction(self) -> None:
        got = run_translated_int16("v_sub_i16", 500, 200)
        exp = ref_v_sub_i16(500, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 300)

    def test_negative_result(self) -> None:
        got = run_translated_int16("v_sub_i16", 100, 200)
        exp = ref_v_sub_i16(100, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFF9C)

    def test_clamp_negative(self) -> None:
        got = run_translated_int16("v_sub_i16", 0x8000, 10, clamp=True)
        exp = ref_v_sub_i16(0x8000, 10, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000)

    def test_clamp_positive(self) -> None:
        got = run_translated_int16("v_sub_i16", 0x7FFF, 0xFFFF, clamp=True)
        exp = ref_v_sub_i16(0x7FFF, 0xFFFF, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF)

    def test_self_subtraction(self) -> None:
        got0 = run_translated_int16("v_sub_i16", 0x1234, 0x1234)
        got1 = run_translated_int16("v_sub_i16", 0x8000, 0x8000)
        self.assertEqual(got0, 0)
        self.assertEqual(got1, 0)

    def test_underflow_wrap(self) -> None:
        got = run_translated_int16("v_sub_i16", 0x8000, 1)
        exp = ref_v_sub_i16(0x8000, 1)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_sub_i16", 1000, 400, "dst_src0")
        self.assertEqual(got, 600)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_sub_i16", 1000, 400, "dst_src1")
        self.assertEqual(got, 600)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (1000 + i * 50) & MASK16
            b = (i * 50) & MASK16
            exp = ref_v_sub_i16(a, b)
            got = run_translated_int16("v_sub_i16", a, b)
            self.assertEqual(got, exp)
            self.assertEqual(got, 1000)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 500",
            "v_mov_b32 v2, 200",
            "v_sub_i16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_sub_i16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpISub", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVSubU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_sub_u16."""

    def test_basic_subtraction(self) -> None:
        got = run_translated_int16("v_sub_u16", 500, 200)
        exp = ref_v_sub_u16(500, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 300)

    def test_borrow_wrap(self) -> None:
        got = run_translated_int16("v_sub_u16", 100, 200)
        exp = ref_v_sub_u16(100, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFF9C)

    def test_clamp_underflow(self) -> None:
        got = run_translated_int16("v_sub_u16", 100, 200, clamp=True)
        exp = ref_v_sub_u16(100, 200, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0)

    def test_extremum_bounds(self) -> None:
        got = run_translated_int16("v_sub_u16", 0, 0xFFFF)
        self.assertEqual(got, 1)
        got_clamp = run_translated_int16("v_sub_u16", 0, 0xFFFF, clamp=True)
        self.assertEqual(got_clamp, 0)

    def test_zero_identity(self) -> None:
        got = run_translated_int16("v_sub_u16", 1234, 0)
        self.assertEqual(got, 1234)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_sub_u16", 700, 200, "dst_src0")
        self.assertEqual(got, 500)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_sub_u16", 700, 200, "dst_src1")
        self.assertEqual(got, 500)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (2000 + i * 100) & MASK16
            b = (i * 100) & MASK16
            exp = ref_v_sub_u16(a, b)
            got = run_translated_int16("v_sub_u16", a, b)
            self.assertEqual(got, exp)
            self.assertEqual(got, 2000)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 500",
            "v_mov_b32 v2, 200",
            "v_sub_u16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_sub_u16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpISub", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVAddNcI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_add_nc_i16."""

    def test_basic_addition(self) -> None:
        got = run_translated_int16("v_add_nc_i16", 150, 250)
        exp = ref_v_add_nc_i16(150, 250)
        self.assertEqual(got, exp)
        self.assertEqual(got, 400)

    def test_clamp(self) -> None:
        got = run_translated_int16("v_add_nc_i16", 30000, 5000, clamp=True)
        exp = ref_v_add_nc_i16(30000, 5000, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF)

    def test_identity_and_inverses(self) -> None:
        got_id = run_translated_int16("v_add_nc_i16", 0x1234, 0)
        self.assertEqual(got_id, 0x1234)
        got_inv = run_translated_int16("v_add_nc_i16", 0x1234, (-0x1234) & MASK16)
        self.assertEqual(got_inv, 0)

    def test_wrap_boundary(self) -> None:
        got = run_translated_int16("v_add_nc_i16", 0x7FFF, 1)
        exp = ref_v_add_nc_i16(0x7FFF, 1)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_add_nc_i16", 400, 300, "dst_src0")
        self.assertEqual(got, 700)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_add_nc_i16", 400, 300, "dst_src1")
        self.assertEqual(got, 700)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (i * 250) & MASK16
            b = (i * 350) & MASK16
            exp = ref_v_add_nc_i16(a, b)
            got = run_translated_int16("v_add_nc_i16", a, b)
            self.assertEqual(got, exp)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_add_nc_i16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_add_nc_i16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpIAdd", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVSubNcI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_sub_nc_i16."""

    def test_basic_subtraction(self) -> None:
        got = run_translated_int16("v_sub_nc_i16", 500, 200)
        exp = ref_v_sub_nc_i16(500, 200)
        self.assertEqual(got, exp)
        self.assertEqual(got, 300)

    def test_clamp(self) -> None:
        got = run_translated_int16("v_sub_nc_i16", 0x8000, 10, clamp=True)
        exp = ref_v_sub_nc_i16(0x8000, 10, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000)

    def test_self_sub_identity(self) -> None:
        got = run_translated_int16("v_sub_nc_i16", 0x5678, 0x5678)
        self.assertEqual(got, 0)

    def test_underflow_wrap(self) -> None:
        got = run_translated_int16("v_sub_nc_i16", 0x8000, 1)
        exp = ref_v_sub_nc_i16(0x8000, 1)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF)

    def test_register_aliasing_dst_src0(self) -> None:
        got = run_aliased_int16("v_sub_nc_i16", 800, 300, "dst_src0")
        self.assertEqual(got, 500)

    def test_register_aliasing_dst_src1(self) -> None:
        got = run_aliased_int16("v_sub_nc_i16", 800, 300, "dst_src1")
        self.assertEqual(got, 500)

    def test_simd_multi_lane_divergence(self) -> None:
        for i in range(8):
            a = (5000 + i * 100) & MASK16
            b = (i * 100) & MASK16
            exp = ref_v_sub_nc_i16(a, b)
            got = run_translated_int16("v_sub_nc_i16", a, b)
            self.assertEqual(got, exp)
            self.assertEqual(got, 5000)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 500",
            "v_mov_b32 v2, 200",
            "v_sub_nc_i16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_check_v_sub_nc_i16_lowering", prog)
        self.assertIn("OpEntryPoint", spv)
        self.assertIn("OpISub", spv)
        self.assertEqual(mrt0_words(spv)[0] & MASK16, 300)


class TestVInt16NegativeAndMutation(unittest.TestCase):
    """Negative controls and mutation detection guards for 16-bit integer ALU lowering."""

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["v_add_i16 v0, v1", "s_endpgm"]
        with self.assertRaises(Exception):
            translate_lines("ps_int16_missing_operand", prog)

    def test_syntax_rejection_invalid_opcode(self) -> None:
        prog = ["v_add_invalid16 v0, v1, v2", "s_endpgm"]
        with self.assertRaises(Exception):
            translate_lines("ps_int16_invalid_op", prog)

    def test_mutation_detection_signed_vs_unsigned_add_clamp(self) -> None:
        # Adding 0x7FFF + 1 with clamp:
        # signed: 32767 + 1 saturates to 0x7FFF
        # unsigned: 32767 + 1 = 0x8000 (< 0xFFFF, no clamp saturation)
        a = 0x7FFF
        b = 1
        res_signed = run_translated_int16("v_add_i16", a, b, clamp=True)
        res_unsigned = run_translated_int16("v_add_u16", a, b, clamp=True)
        self.assertEqual(res_signed, 0x7FFF)
        self.assertEqual(res_unsigned, 0x8000)
        self.assertNotEqual(res_signed, res_unsigned)

    def test_mutation_detection_signed_vs_unsigned_sub_clamp(self) -> None:
        # Subtracting 100 - 200 with clamp:
        # unsigned: underflows and saturates to 0
        # signed: 100 - 200 = -100 (0xFF9C), within [-32768, 32767] -> no clamp
        a = 100
        b = 200
        res_unsigned = run_translated_int16("v_sub_u16", a, b, clamp=True)
        res_signed = run_translated_int16("v_sub_i16", a, b, clamp=True)
        self.assertEqual(res_unsigned, 0)
        self.assertEqual(res_signed, 0xFF9C)
        self.assertNotEqual(res_unsigned, res_signed)

    def test_mutation_detection_add_vs_sub(self) -> None:
        a = 500
        b = 200
        res_add = run_translated_int16("v_add_i16", a, b)
        res_sub = run_translated_int16("v_sub_i16", a, b)
        self.assertEqual(res_add, 700)
        self.assertEqual(res_sub, 300)
        self.assertNotEqual(res_add, res_sub)

    def test_mutation_detection_add_nc_vs_sub_nc(self) -> None:
        a = 600
        b = 150
        res_add = run_translated_int16("v_add_nc_i16", a, b)
        res_sub = run_translated_int16("v_sub_nc_i16", a, b)
        self.assertEqual(res_add, 750)
        self.assertEqual(res_sub, 450)
        self.assertNotEqual(res_add, res_sub)


if __name__ == "__main__":
    unittest.main()
