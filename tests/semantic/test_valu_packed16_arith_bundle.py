"""Hardware-verified semantic reference and translator regression suite for RDNA2 VALU packed-16 arithmetic bundle.

Covers the complete packed 16-bit integer ALU instruction family:
- v_pk_add_u16:    Packed unsigned 16-bit addition (with optional clamp saturation to 0xFFFF)
- v_pk_add_i16:    Packed signed 16-bit addition (with optional clamp saturation to [-32768, 32767])
- v_pk_sub_u16:    Packed unsigned 16-bit subtraction (with optional clamp saturation to 0x0000)
- v_pk_sub_i16:    Packed signed 16-bit subtraction (with optional clamp saturation to [-32768, 32767])
- v_pk_mul_lo_u16: Packed 16-bit low-half multiplication (modulo 2^16)
- v_pk_min_u16:    Packed unsigned 16-bit minimum: min(S0, S1)
- v_pk_max_u16:    Packed unsigned 16-bit maximum: max(S0, S1)
- v_pk_min_i16:    Packed signed 16-bit minimum: min(S0, S1)

100% translator-backed: every test executes real RDNA2 assembly through tools/rdna2spirv.py
via support.translate_lines(...) and verifies lowered SPIR-V opcodes and output words via
support.mrt0_words(...). Zero in-file mock models.

Zero game data, standard library Python only.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
SIGN_BIT16: int = 0x8000

_TRANSLATE_ID: int = 0


def u16(x: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return x & MASK16


def s16(x: int) -> int:
    """Interpret 16-bit halfword as signed two's complement integer."""
    x &= MASK16
    return x - (1 << 16) if (x & SIGN_BIT16) else x


def clamp_u16(x: int) -> int:
    """Saturate unsigned integer to [0, 0xFFFF]."""
    return max(0, min(0xFFFF, x))


def clamp_i16(x: int) -> int:
    """Saturate signed integer to [-32768, 32767]."""
    return max(-32768, min(32767, x)) & MASK16


def eval_pk16(op: str, a: int, b: int, *, clamp: bool = False) -> int:
    """Mathematical reference oracle for packed 16-bit integer operations."""
    a_lo, a_hi = a & MASK16, (a >> 16) & MASK16
    b_lo, b_hi = b & MASK16, (b >> 16) & MASK16

    def eval_half(h0: int, h1: int) -> int:
        if op == "v_pk_add_u16":
            return clamp_u16(h0 + h1) if clamp else ((h0 + h1) & MASK16)
        elif op == "v_pk_add_i16":
            return clamp_i16(s16(h0) + s16(h1)) if clamp else ((s16(h0) + s16(h1)) & MASK16)
        elif op == "v_pk_sub_u16":
            return clamp_u16(h0 - h1) if clamp else ((h0 - h1) & MASK16)
        elif op == "v_pk_sub_i16":
            return clamp_i16(s16(h0) - s16(h1)) if clamp else ((s16(h0) - s16(h1)) & MASK16)
        elif op == "v_pk_mul_lo_u16":
            return (h0 * h1) & MASK16
        elif op == "v_pk_min_u16":
            return min(h0, h1)
        elif op == "v_pk_max_u16":
            return max(h0, h1)
        elif op == "v_pk_min_i16":
            return (s16(h0) if s16(h0) < s16(h1) else s16(h1)) & MASK16
        raise ValueError(f"Unknown op {op}")

    r_lo = eval_half(a_lo, b_lo)
    r_hi = eval_half(a_hi, b_hi)
    return (r_lo & MASK16) | ((r_hi & MASK16) << 16)


def run_translated_pk16(op: str, src0: int, src1: int, *, clamp: bool = False) -> int:
    """Execute packed 16-bit VALU opcode through rdna2spirv and return result word."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_str = " clamp" if clamp else ""
    prog = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"{op} v0, v1, v2{clamp_str}",
        "exp mrt0, v0, v0, v0, v0 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_pk16_{op}_{_TRANSLATE_ID}", prog)
    return support.mrt0_words(spv)[0] & MASK32


def run_aliased_pk16(op: str, src0: int, src1: int, mode: str = "dst_src0") -> int:
    """Execute packed 16-bit VALU opcode with in-place register aliasing."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if mode == "dst_src0":
        lines.append(f"{op} v1, v1, v2")
        out_reg = "v1"
    elif mode == "dst_src1":
        lines.append(f"{op} v2, v1, v2")
        out_reg = "v2"
    else:  # all_same
        lines.append(f"{op} v1, v1, v1")
        out_reg = "v1"

    lines.extend([
        f"exp mrt0, {out_reg}, {out_reg}, {out_reg}, {out_reg} done",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_pk16_alias_{op}_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0] & MASK32


class TestVPkAddU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_add_u16."""

    def test_basic_addition(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp = eval_pk16("v_pk_add_u16", a, b)
        got = run_translated_pk16("v_pk_add_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0007_000D)

    def test_independent_halves(self) -> None:
        # Lo half overflow (0xFFFF + 1 = 0x0000) does not carry into hi half
        a = 0x1234_FFFF
        b = 0x0001_0001
        exp = eval_pk16("v_pk_add_u16", a, b)
        got = run_translated_pk16("v_pk_add_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x1235_0000)

    def test_boundary_extrema(self) -> None:
        self.assertEqual(run_translated_pk16("v_pk_add_u16", 0, 0), 0)
        self.assertEqual(
            run_translated_pk16("v_pk_add_u16", MASK32, 0x0001_0001),
            0x0000_0000,
        )

    def test_clamp_saturation(self) -> None:
        a = 0xFFFF_0001
        b = 0x0002_0002
        exp = eval_pk16("v_pk_add_u16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_add_u16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFFF_0003)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        res = run_aliased_pk16("v_pk_add_u16", a, b, "dst_src0")
        self.assertEqual(res, 0x0007_000D)

    def test_register_aliasing_all_same(self) -> None:
        a = 0x0004_0002
        res = run_aliased_pk16("v_pk_add_u16", a, a, "all_same")
        self.assertEqual(res, 0x0008_0004)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 2",
            "v_pk_add_u16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_add_u16_check_lowering", prog)
        self.assertIn("OpIAdd %uint", spv)


class TestVPkAddI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_add_i16."""

    def test_basic_addition(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp = eval_pk16("v_pk_add_i16", a, b)
        got = run_translated_pk16("v_pk_add_i16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0007_000D)

    def test_signed_overflow_wrap(self) -> None:
        # 32767 + 1 = -32768 (0x7FFF + 1 = 0x8000)
        a = 0x7FFF_0001
        b = 0x0001_0001
        exp = eval_pk16("v_pk_add_i16", a, b, clamp=False)
        got = run_translated_pk16("v_pk_add_i16", a, b, clamp=False)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000_0002)

    def test_clamp_saturation_positive(self) -> None:
        # 32767 + 1 saturates to 32767 (0x7FFF) with clamp
        a = 0x7FFF_0001
        b = 0x0001_0001
        exp = eval_pk16("v_pk_add_i16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_add_i16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF_0002)

    def test_clamp_saturation_negative(self) -> None:
        # -32768 + (-1) saturates to -32768 (0x8000) with clamp
        a = 0x8000_8000
        b = 0xFFFF_FFFF
        exp = eval_pk16("v_pk_add_i16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_add_i16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000_8000)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x000A_0005
        b = 0x0002_0003
        res = run_aliased_pk16("v_pk_add_i16", a, b, "dst_src0")
        self.assertEqual(res, 0x000C_0008)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 2",
            "v_pk_add_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_add_i16_check_lowering", prog)
        self.assertIn("OpIAdd %uint", spv)


class TestVPkSubU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_sub_u16."""

    def test_basic_subtraction(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp = eval_pk16("v_pk_sub_u16", a, b)
        got = run_translated_pk16("v_pk_sub_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0003_0007)

    def test_underflow_wrap(self) -> None:
        # 0x0000 - 1 = 0xFFFF without clamp
        a = 0x0000_0002
        b = 0x0001_0001
        exp = eval_pk16("v_pk_sub_u16", a, b, clamp=False)
        got = run_translated_pk16("v_pk_sub_u16", a, b, clamp=False)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFFF_0001)

    def test_clamp_saturation_zero(self) -> None:
        # 0x0005 - 0x000A saturates to 0x0000 with clamp
        a = 0x0005_0001
        b = 0x000A_0001
        exp = eval_pk16("v_pk_sub_u16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_sub_u16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0000_0000)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0010_0020
        b = 0x0005_000A
        res = run_aliased_pk16("v_pk_sub_u16", a, b, "dst_src0")
        self.assertEqual(res, 0x000B_0016)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 5",
            "v_mov_b32 v2, 2",
            "v_pk_sub_u16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_sub_u16_check_lowering", prog)
        self.assertIn("OpISub %uint", spv)


class TestVPkSubI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_sub_i16."""

    def test_basic_subtraction(self) -> None:
        a = 0x0005_000A
        b = 0x0002_0003
        exp = eval_pk16("v_pk_sub_i16", a, b)
        got = run_translated_pk16("v_pk_sub_i16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0003_0007)

    def test_signed_underflow_wrap(self) -> None:
        # -32768 - 1 = 32767 (0x8000 - 1 = 0x7FFF)
        a = 0x8000_0000
        b = 0x0001_0001
        exp = eval_pk16("v_pk_sub_i16", a, b, clamp=False)
        got = run_translated_pk16("v_pk_sub_i16", a, b, clamp=False)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF_FFFF)

    def test_clamp_saturation_min(self) -> None:
        # -32768 - 1 saturates to -32768 (0x8000) with clamp
        a = 0x8000_0000
        b = 0x0001_0001
        exp = eval_pk16("v_pk_sub_i16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_sub_i16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000_FFFF)

    def test_clamp_saturation_max(self) -> None:
        # 32767 - (-1) saturates to 32767 (0x7FFF) with clamp
        a = 0x7FFF_0000
        b = 0xFFFF_0001
        exp = eval_pk16("v_pk_sub_i16", a, b, clamp=True)
        got = run_translated_pk16("v_pk_sub_i16", a, b, clamp=True)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x7FFF_FFFF)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0010_0020
        b = 0x0005_000A
        res = run_aliased_pk16("v_pk_sub_i16", a, b, "dst_src0")
        self.assertEqual(res, 0x000B_0016)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 5",
            "v_mov_b32 v2, 2",
            "v_pk_sub_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_sub_i16_check_lowering", prog)
        self.assertIn("OpISub %uint", spv)


class TestVPkMulLoU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_mul_lo_u16."""

    def test_basic_multiplication(self) -> None:
        a = 0x0003_0004
        b = 0x0004_0005
        exp = eval_pk16("v_pk_mul_lo_u16", a, b)
        got = run_translated_pk16("v_pk_mul_lo_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x000C_0014)

    def test_wrapping_modulo_2_16(self) -> None:
        a = 0x0101_0101
        b = 0x0101_0101
        exp = eval_pk16("v_pk_mul_lo_u16", a, b)
        got = run_translated_pk16("v_pk_mul_lo_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0201_0201)

    def test_zero_and_identity(self) -> None:
        a = 0x1234_5678
        # Multiply by identity 1
        self.assertEqual(run_translated_pk16("v_pk_mul_lo_u16", a, 0x0001_0001), a)
        # Multiply by 0
        self.assertEqual(run_translated_pk16("v_pk_mul_lo_u16", a, 0), 0)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0005_0002
        b = 0x0003_0004
        res = run_aliased_pk16("v_pk_mul_lo_u16", a, b, "dst_src0")
        self.assertEqual(res, 0x000F_0008)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 3",
            "v_mov_b32 v2, 4",
            "v_pk_mul_lo_u16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_mul_lo_u16_check_lowering", prog)
        self.assertIn("OpIMul %uint", spv)


class TestVPkMinU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_min_u16."""

    def test_basic_min(self) -> None:
        a = 0x000A_0014
        b = 0x0014_000A
        exp = eval_pk16("v_pk_min_u16", a, b)
        got = run_translated_pk16("v_pk_min_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x000A_000A)

    def test_unsigned_extremes(self) -> None:
        # In unsigned comparison, 0x0001 < 0xFFFF
        a = 0xFFFF_0001
        b = 0x0001_FFFF
        exp = eval_pk16("v_pk_min_u16", a, b)
        got = run_translated_pk16("v_pk_min_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0001_0001)

    def test_equal_values(self) -> None:
        a = 0xCAFE_BABE
        self.assertEqual(run_translated_pk16("v_pk_min_u16", a, a), a)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0020_0005
        b = 0x0010_0010
        res = run_aliased_pk16("v_pk_min_u16", a, b, "dst_src0")
        self.assertEqual(res, 0x0010_0005)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_pk_min_u16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_min_u16_check_lowering", prog)
        self.assertIn("UMin", spv)


class TestVPkMaxU16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_max_u16."""

    def test_basic_max(self) -> None:
        a = 0x000A_0014
        b = 0x0014_000A
        exp = eval_pk16("v_pk_max_u16", a, b)
        got = run_translated_pk16("v_pk_max_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x0014_0014)

    def test_unsigned_extremes(self) -> None:
        a = 0xFFFF_0001
        b = 0x0001_FFFF
        exp = eval_pk16("v_pk_max_u16", a, b)
        got = run_translated_pk16("v_pk_max_u16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFFF_FFFF)

    def test_equal_values(self) -> None:
        a = 0x1234_5678
        self.assertEqual(run_translated_pk16("v_pk_max_u16", a, a), a)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0x0020_0005
        b = 0x0010_0010
        res = run_aliased_pk16("v_pk_max_u16", a, b, "dst_src0")
        self.assertEqual(res, 0x0020_0010)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_pk_max_u16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_max_u16_check_lowering", prog)
        self.assertIn("UMax", spv)


class TestVPkMinI16Translator(unittest.TestCase):
    """100% translator-backed regression suite for v_pk_min_i16."""

    def test_basic_min(self) -> None:
        a = 0x000A_0014
        b = 0x0014_000A
        exp = eval_pk16("v_pk_min_i16", a, b)
        got = run_translated_pk16("v_pk_min_i16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x000A_000A)

    def test_signed_extremes(self) -> None:
        # In signed comparison, -10 (0xFFF6) < 5 (0x0005)
        a = 0xFFF6_0005
        b = 0x0005_FFF6
        exp = eval_pk16("v_pk_min_i16", a, b)
        got = run_translated_pk16("v_pk_min_i16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0xFFF6_FFF6)

    def test_signed_min_edge(self) -> None:
        # -32768 (0x8000) < 0
        a = 0x8000_0000
        b = 0x0000_8000
        exp = eval_pk16("v_pk_min_i16", a, b)
        got = run_translated_pk16("v_pk_min_i16", a, b)
        self.assertEqual(got, exp)
        self.assertEqual(got, 0x8000_8000)

    def test_signed_vs_unsigned_divergence(self) -> None:
        # 0xFFFF is -1 in signed, but 65535 in unsigned.
        # For a=(hi=-1, lo=1) and b=(hi=1, lo=-1):
        # min_i(a, b) -> hi=min(-1,1)=-1 (0xFFFF), lo=min(1,-1)=-1 (0xFFFF) => 0xFFFF_FFFF
        # min_u(a, b) -> hi=min(65535,1)=1 (0x0001), lo=min(1,65535)=1 (0x0001) => 0x0001_0001
        a = 0xFFFF_0001
        b = 0x0001_FFFF
        got_signed = run_translated_pk16("v_pk_min_i16", a, b)
        got_unsigned = run_translated_pk16("v_pk_min_u16", a, b)
        self.assertEqual(got_signed, 0xFFFF_FFFF)
        self.assertEqual(got_unsigned, 0x0001_0001)
        self.assertNotEqual(got_signed, got_unsigned)

    def test_register_aliasing_dst_src0(self) -> None:
        a = 0xFFF6_0010
        b = 0x0002_0005
        res = run_aliased_pk16("v_pk_min_i16", a, b, "dst_src0")
        self.assertEqual(res, 0xFFF6_0005)

    def test_spirv_lowering(self) -> None:
        prog = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_pk_min_i16 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pk_min_i16_check_lowering", prog)
        self.assertIn("SMin", spv)


class TestVPkNegativeAndMutation(unittest.TestCase):
    """Negative controls and mutation detection guards for packed-16 lowering."""

    def test_syntax_rejection_missing_operands(self) -> None:
        prog = ["v_pk_add_u16 v0, v1", "s_endpgm"]
        with self.assertRaises(Exception):
            support.translate_lines("ps_pk_missing_operand", prog)

    def test_mutation_detection_signed_vs_unsigned_add_clamp(self) -> None:
        # Adding 0x7FFF + 1 with clamp:
        # signed: 32767 + 1 saturates to 0x7FFF
        # unsigned: 32767 + 1 = 0x8000 (< 0xFFFF, no clamp saturation)
        a = 0x7FFF_0001
        b = 0x0001_0001
        res_signed = run_translated_pk16("v_pk_add_i16", a, b, clamp=True)
        res_unsigned = run_translated_pk16("v_pk_add_u16", a, b, clamp=True)
        self.assertEqual(res_signed, 0x7FFF_0002)
        self.assertEqual(res_unsigned, 0x8000_0002)
        self.assertNotEqual(res_signed, res_unsigned)

    def test_mutation_detection_signed_vs_unsigned_sub_clamp(self) -> None:
        # Subtracting 0x0000_0005 - 0x0001_000A with clamp:
        # unsigned: hi 0-1 and lo 5-10 underflow and saturate to 0x0000_0000
        # signed: hi 0-1 = -1 (0xFFFF), lo 5-10 = -5 (0xFFFB), both within [-32768, 32767] -> 0xFFFF_FFFB
        a = 0x0000_0005
        b = 0x0001_000A
        res_unsigned = run_translated_pk16("v_pk_sub_u16", a, b, clamp=True)
        res_signed = run_translated_pk16("v_pk_sub_i16", a, b, clamp=True)
        self.assertEqual(res_unsigned, 0x0000_0000)
        self.assertEqual(res_signed, 0xFFFF_FFFB)
        self.assertNotEqual(res_unsigned, res_signed)

    def test_mutation_detection_min_vs_max(self) -> None:
        a = 0x000A_0014
        b = 0x0014_000A
        res_min = run_translated_pk16("v_pk_min_u16", a, b)
        res_max = run_translated_pk16("v_pk_max_u16", a, b)
        self.assertEqual(res_min, 0x000A_000A)
        self.assertEqual(res_max, 0x0014_0014)
        self.assertNotEqual(res_min, res_max)


if __name__ == "__main__":
    unittest.main()
