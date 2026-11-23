"""Semantic hardware and translator verification suite for RDNA2 v_min3_f16:
v_min3_f16 (VOP3 3-source minimum of half-precision floating-point numbers).

Features & specifications:
- IEEE 754-2008 / RDNA2 rules: NaN suppression (return non-NaN operand; return quiet NaN if all 3 are NaN).
- Signed zero ordering: -0.0 < +0.0.
- Destination register half preservation: keeps unwritten 16-bit half (low or high based on op_sel dst bit).
- VOP3 op_sel support: select low or high 16 bits of each source and destination.
- Input modifiers: abs, neg, -abs.
- Output clamp: clamp to [0.0, 1.0], negative values / -0.0 become +0.0, NaNs become +0.0.
- Standard library Python only; zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import struct
import sys
from typing import List, Optional
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
import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD_VAL = 0xA55AA55A

# IEEE 754 binary16 bit constants
POS_ZERO = 0x0000
NEG_ZERO = 0x8000
POS_ONE = 0x3C00
NEG_ONE = 0xBC00
POS_HALF = 0x3800
NEG_HALF = 0xB800
POS_TWO = 0x4000
NEG_TWO = 0xC000
POS_THREE = 0x4200
NEG_THREE = 0xC200
POS_FOUR = 0x4400
NEG_FOUR = 0xC400
POS_INF = 0x7C00
NEG_INF = 0xFC00
QNAN_CANONICAL = 0x7E00
QNAN_ALT = 0x7E55
SNAN = 0x7C01
NEG_QNAN = 0xFE00

POS_MIN_SUBNORM = 0x0001
POS_MAX_SUBNORM = 0x03FF
NEG_MIN_SUBNORM = 0x8001
NEG_MAX_SUBNORM = 0x83FF
POS_MIN_NORMAL = 0x0400
POS_MAX_NORMAL = 0x7BFF
NEG_MAX_NORMAL = 0xFBFF


def f16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    if math.isnan(f):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return POS_INF if f > 0 else NEG_INF


def is_nan16(bits: int) -> bool:
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def ref_min2_f16(a_bits: int, b_bits: int) -> int:
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        return QNAN_CANONICAL
    if a_nan:
        return b_bits & MASK16
    if b_nan:
        return a_bits & MASK16

    is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
    is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
    is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
    is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

    if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
        return NEG_ZERO
    if is_a_neg_zero and is_b_neg_zero:
        return NEG_ZERO
    if is_a_pos_zero and is_b_pos_zero:
        return POS_ZERO

    fa = f16_to_float(a_bits)
    fb = f16_to_float(b_bits)
    key_a = (fa, math.copysign(1.0, fa))
    key_b = (fb, math.copysign(1.0, fb))
    return (a_bits & MASK16) if key_a <= key_b else (b_bits & MASK16)


def ref_min3_f16(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    m = ref_min2_f16(s0, s1)
    res = ref_min2_f16(m, s2)
    if clamp:
        if is_nan16(res):
            return POS_ZERO
        cf = f16_to_float(res)
        if cf <= 0.0 or (res & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
    return res & MASK16


def apply_f16_input_mod(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    b = bits & MASK16
    if abs_val:
        b &= 0x7FFF
    if neg:
        b ^= 0x8000
    return b


def run_hw_min3_f16(
    s0: int,
    s1: int,
    s2: int,
    old_dst: int = 0x12340000,
    clamp: bool = False,
    op_sel: list[int] | None = None,
    name: str = "ps_min3_hw",
) -> int:
    """Translates and runs v_min3_f16 via support.translate_lines and support.mrt0_words."""
    mod = " clamp" if clamp else ""
    sel = f" op_sel:[{op_sel[0]},{op_sel[1]},{op_sel[2]},{op_sel[3]}]" if op_sel else ""
    lines = [
        f"v_mov_b32_e32 v0, {old_dst:#x}",
        f"s_mov_b32 s0, {s0:#x}",
        f"s_mov_b32 s1, {s1:#x}",
        f"s_mov_b32 s2, {s2:#x}",
        f"v_min3_f16 v0, s0, s1, s2{mod}{sel}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


def run_hw_min3_f16_custom(
    inst: str,
    old_dst: int = 0x12340000,
    sgprs: dict[str, int] | None = None,
    vgprs: dict[str, int] | None = None,
    name: str = "ps_min3_custom",
) -> int:
    """Translates and runs custom assembly setup through support.translate_lines and support.mrt0_words."""
    lines = [f"v_mov_b32_e32 v0, {old_dst:#x}"]
    if sgprs:
        for reg, val in sgprs.items():
            lines.append(f"s_mov_b32 {reg}, {val:#x}")
    if vgprs:
        for reg, val in vgprs.items():
            lines.append(f"v_mov_b32_e32 {reg}, {val:#x}")
    lines.append(inst)
    lines.append("exp mrt0 v0, v0, v0, v0 done vm")
    lines.append("s_endpgm")
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestValuMin3F16Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model."""

    def test_basic_orderings(self):
        self.assertEqual(ref_min3_f16(POS_ONE, POS_TWO, POS_HALF), POS_HALF)
        self.assertEqual(ref_min3_f16(POS_TWO, POS_HALF, POS_ONE), POS_HALF)
        self.assertEqual(ref_min3_f16(POS_HALF, POS_ONE, POS_TWO), POS_HALF)
        self.assertEqual(ref_min3_f16(NEG_ONE, NEG_TWO, POS_HALF), NEG_TWO)

        hw1 = run_hw_min3_f16(POS_ONE, POS_TWO, POS_HALF, name="ps_or_bo1")
        self.assertEqual(hw1, 0x12343800)
        hw2 = run_hw_min3_f16(POS_TWO, POS_HALF, POS_ONE, name="ps_or_bo2")
        self.assertEqual(hw2, 0x12343800)
        hw3 = run_hw_min3_f16(POS_HALF, POS_ONE, POS_TWO, name="ps_or_bo3")
        self.assertEqual(hw3, 0x12343800)
        hw4 = run_hw_min3_f16(NEG_ONE, NEG_TWO, POS_HALF, name="ps_or_bo4")
        self.assertEqual(hw4, 0x1234C000)

    def test_signed_zeros(self):
        self.assertEqual(ref_min3_f16(POS_ZERO, NEG_ZERO, POS_ONE), NEG_ZERO)
        self.assertEqual(ref_min3_f16(POS_ZERO, POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_min3_f16(POS_ZERO, POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_min3_f16(NEG_ZERO, NEG_ZERO, NEG_ZERO), NEG_ZERO)

        hw1 = run_hw_min3_f16(POS_ZERO, NEG_ZERO, POS_ONE, name="ps_or_sz1")
        self.assertEqual(hw1, 0x12348000)
        hw2 = run_hw_min3_f16(POS_ZERO, POS_ZERO, NEG_ZERO, name="ps_or_sz2")
        self.assertEqual(hw2, 0x12348000)
        hw3 = run_hw_min3_f16(POS_ZERO, POS_ZERO, POS_ZERO, name="ps_or_sz3")
        self.assertEqual(hw3, 0x12340000)
        hw4 = run_hw_min3_f16(NEG_ZERO, NEG_ZERO, NEG_ZERO, name="ps_or_sz4")
        self.assertEqual(hw4, 0x12348000)

    def test_nan_suppression(self):
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(ref_min3_f16(POS_TWO, QNAN_CANONICAL, POS_ONE), POS_ONE)
        self.assertEqual(ref_min3_f16(POS_TWO, POS_ONE, QNAN_CANONICAL), POS_ONE)
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, POS_THREE), POS_THREE)
        self.assertEqual(ref_min3_f16(POS_THREE, QNAN_CANONICAL, QNAN_ALT), POS_THREE)
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, POS_THREE, QNAN_ALT), POS_THREE)
        self.assertTrue(is_nan16(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, SNAN)))

        hw1 = run_hw_min3_f16(QNAN_CANONICAL, POS_ONE, POS_TWO, name="ps_or_nan1")
        self.assertEqual(hw1, 0x12343C00)
        hw2 = run_hw_min3_f16(POS_TWO, QNAN_CANONICAL, POS_ONE, name="ps_or_nan2")
        self.assertEqual(hw2, 0x12343C00)
        hw3 = run_hw_min3_f16(POS_TWO, POS_ONE, QNAN_CANONICAL, name="ps_or_nan3")
        self.assertEqual(hw3, 0x12343C00)
        hw4 = run_hw_min3_f16(QNAN_CANONICAL, QNAN_ALT, POS_THREE, name="ps_or_nan4")
        self.assertEqual(hw4, 0x12344200)
        hw5 = run_hw_min3_f16(POS_THREE, QNAN_CANONICAL, QNAN_ALT, name="ps_or_nan5")
        self.assertEqual(hw5, 0x12344200)
        hw6 = run_hw_min3_f16(QNAN_CANONICAL, POS_THREE, QNAN_ALT, name="ps_or_nan6")
        self.assertEqual(hw6, 0x12344200)
        hw7 = run_hw_min3_f16(QNAN_CANONICAL, QNAN_ALT, SNAN, name="ps_or_nan7")
        self.assertEqual(hw7, 0x12347E00)

    def test_clamping(self):
        self.assertEqual(ref_min3_f16(POS_TWO, POS_THREE, POS_FOUR, clamp=True), POS_ONE)
        self.assertEqual(ref_min3_f16(NEG_ONE, POS_HALF, POS_TWO, clamp=True), POS_ZERO)
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, SNAN, clamp=True), POS_ZERO)
        self.assertEqual(ref_min3_f16(POS_HALF, POS_ONE, POS_TWO, clamp=True), POS_HALF)

        hw1 = run_hw_min3_f16(POS_TWO, POS_THREE, POS_FOUR, clamp=True, name="ps_or_cl1")
        self.assertEqual(hw1, 0x12343C00)
        hw2 = run_hw_min3_f16(NEG_ONE, POS_HALF, POS_TWO, clamp=True, name="ps_or_cl2")
        self.assertEqual(hw2, 0x12340000)
        hw3 = run_hw_min3_f16(POS_HALF, POS_ONE, POS_TWO, clamp=True, name="ps_or_cl3")
        self.assertEqual(hw3, 0x12343800)


# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST opcode definitions and typed expressions
# -----------------------------------------------------------------------------
@dataclass
class F16Operand:
    reg: str
    is_vgpr: bool
    is_sgpr: bool
    is_literal: bool
    literal_value: int = 0
    neg: bool = False
    abs_val: bool = False
    hi: bool = False


@dataclass
class VMin3F16AST:
    dst_reg: str
    dst_hi: bool
    src0: F16Operand
    src1: F16Operand
    src2: F16Operand
    clamp: bool = False

    @classmethod
    def parse(cls, line: str) -> VMin3F16AST:
        tokens = line.strip().split()
        if not tokens or tokens[0] != "v_min3_f16":
            raise ValueError(f"Invalid mnemonic for v_min3_f16: {line}")

        op_sel = [0, 0, 0, 0]
        m_sel = re.search(r"op_sel:\s*\[([0-9,\s]+)\]", line)
        if m_sel:
            op_sel = [int(x.strip()) for x in m_sel.group(1).split(",")]
            line_clean = re.sub(r"op_sel:\s*\[[0-9,\s]+\]", "", line)
        else:
            line_clean = line

        clamp = "clamp" in line_clean.split()
        words = [w for w in line_clean.split() if w != "clamp"]
        clean_args = " ".join(words[1:])

        parts = [p.strip() for p in clean_args.split(",") if p.strip()]
        if len(parts) < 4:
            raise ValueError(f"Insufficient operands for v_min3_f16: {line}")

        dst_str = parts[0]
        s0_str = parts[1]
        s1_str = parts[2]
        s2_str = parts[3]

        def parse_src(s: str, sel_bit: int) -> F16Operand:
            neg = False
            abs_val = False
            s = s.strip()
            if s.startswith("-"):
                neg = True
                s = s[1:].strip()
            if s.startswith("|") and s.endswith("|"):
                abs_val = True
                s = s[1:-1].strip()
                if s.startswith("-"):
                    neg = True
                    s = s[1:].strip()
            if s.startswith("v"):
                return F16Operand(reg=s, is_vgpr=True, is_sgpr=False, is_literal=False, neg=neg, abs_val=abs_val, hi=bool(sel_bit))
            elif s.startswith("s"):
                return F16Operand(reg=s, is_vgpr=False, is_sgpr=True, is_literal=False, neg=neg, abs_val=abs_val, hi=bool(sel_bit))
            else:
                val = int(s, 0)
                return F16Operand(reg="", is_vgpr=False, is_sgpr=False, is_literal=True, literal_value=val, neg=neg, abs_val=abs_val, hi=bool(sel_bit))

        s0 = parse_src(s0_str, op_sel[0])
        s1 = parse_src(s1_str, op_sel[1])
        s2 = parse_src(s2_str, op_sel[2])
        return cls(dst_reg=dst_str, dst_hi=bool(op_sel[3]), src0=s0, src1=s1, src2=s2, clamp=clamp)


class TestValuMin3F16AST(unittest.TestCase):
    """Commit 2: Symbolic AST parsing and typed representations."""

    def test_ast_parsing_basic(self):
        ast = VMin3F16AST.parse("v_min3_f16 v0, s0, s1, s2")
        self.assertEqual(ast.dst_reg, "v0")
        self.assertFalse(ast.dst_hi)
        self.assertEqual(ast.src0.reg, "s0")
        self.assertEqual(ast.src1.reg, "s1")
        self.assertEqual(ast.src2.reg, "s2")
        self.assertFalse(ast.clamp)

        spv = support.translate_lines("ps_ast_basic", [
            "v_mov_b32_e32 v0, 0",
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpExtInst %float %glsl NMin", spv)

    def test_ast_parsing_op_sel_and_modifiers(self):
        ast = VMin3F16AST.parse("v_min3_f16 v5, -|v0|, s1, 0x3c00 clamp op_sel:[1, 0, 0, 1]")
        self.assertEqual(ast.dst_reg, "v5")
        self.assertTrue(ast.dst_hi)
        self.assertTrue(ast.src0.neg)
        self.assertTrue(ast.src0.abs_val)
        self.assertTrue(ast.src0.hi)
        self.assertTrue(ast.clamp)
        self.assertTrue(ast.src2.is_literal)
        self.assertEqual(ast.src2.literal_value, 0x3C00)

        spv = support.translate_lines("ps_ast_mod", [
            "v_mov_b32_e32 v5, 0",
            "v_min3_f16 v5, -|v0|, s1, 0x3c00 clamp op_sel:[1, 0, 0, 1]",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpExtInst %float %glsl NMin", spv)
        self.assertIn("FClamp", spv)


# -----------------------------------------------------------------------------
# Commit 3: Deterministic scalar instruction interpreter
# -----------------------------------------------------------------------------
class ScalarF16Machine:
    def __init__(self):
        self.vgprs: dict[str, int] = {}
        self.sgprs: dict[str, int] = {}

    def get_half(self, op: F16Operand) -> int:
        if op.is_literal:
            raw = op.literal_value
        elif op.is_vgpr:
            raw = self.vgprs.get(op.reg, 0)
        else:
            raw = self.sgprs.get(op.reg, 0)
        word16 = ((raw >> 16) if op.hi else raw) & MASK16
        return apply_f16_input_mod(word16, neg=op.neg, abs_val=op.abs_val)

    def execute(self, ast: VMin3F16AST) -> int:
        w0 = self.get_half(ast.src0)
        w1 = self.get_half(ast.src1)
        w2 = self.get_half(ast.src2)
        res16 = ref_min3_f16(w0, w1, w2, clamp=ast.clamp)

        old_dst = self.vgprs.get(ast.dst_reg, OLD_VAL)
        if ast.dst_hi:
            new_val = (old_dst & 0x0000FFFF) | ((res16 & MASK16) << 16)
        else:
            new_val = (old_dst & 0xFFFF0000) | (res16 & MASK16)
        self.vgprs[ast.dst_reg] = new_val
        return new_val


class TestValuMin3F16ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction execution."""

    def test_scalar_interpreter_basic(self):
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = 0x12345678
        mach.sgprs["s0"] = 0x3C00  # 1.0
        mach.sgprs["s1"] = 0x4000  # 2.0
        mach.sgprs["s2"] = 0x3800  # 0.5
        ast = VMin3F16AST.parse("v_min3_f16 v0, s0, s1, s2")
        out = mach.execute(ast)
        self.assertEqual(out, 0x12343800)

        hw = run_hw_min3_f16(0x3C00, 0x4000, 0x3800, old_dst=0x12345678, name="ps_sc_basic")
        self.assertEqual(hw, 0x12343800)

    def test_scalar_interpreter_high_half_write(self):
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = 0x12345678
        mach.sgprs["s0"] = 0x40003C00  # lo=1.0, hi=2.0
        mach.sgprs["s1"] = 0x4200      # 3.0
        mach.sgprs["s2"] = 0x4400      # 4.0
        ast = VMin3F16AST.parse("v_min3_f16 v0, s0, s1, s2 op_sel:[0, 0, 0, 1]")
        out = mach.execute(ast)
        self.assertEqual(out, 0x3C005678)

        hw = run_hw_min3_f16(0x40003C00, 0x4200, 0x4400, old_dst=0x12345678, op_sel=[0, 0, 0, 1], name="ps_sc_hi")
        self.assertEqual(hw, 0x3C005678)


# -----------------------------------------------------------------------------
# Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking
# -----------------------------------------------------------------------------
class SimdF16Wave:
    def __init__(self, size: int = 32):
        self.size = size
        self.exec_mask: int = (1 << size) - 1
        self.vgprs: dict[str, list[int]] = {}

    def init_vgpr(self, reg: str, values: list[int]):
        self.vgprs[reg] = list(values)

    def execute(self, ast: VMin3F16AST, sgprs: dict[str, int]) -> list[int]:
        if ast.dst_reg not in self.vgprs:
            self.vgprs[ast.dst_reg] = [OLD_VAL] * self.size
        dst_list = self.vgprs[ast.dst_reg]

        for lane in range(self.size):
            if not ((self.exec_mask >> lane) & 1):
                continue
            def lane_half(op: F16Operand) -> int:
                if op.is_literal:
                    raw = op.literal_value
                elif op.is_vgpr:
                    raw = self.vgprs.get(op.reg, [0] * self.size)[lane]
                else:
                    raw = sgprs.get(op.reg, 0)
                word16 = ((raw >> 16) if op.hi else raw) & MASK16
                return apply_f16_input_mod(word16, neg=op.neg, abs_val=op.abs_val)

            w0 = lane_half(ast.src0)
            w1 = lane_half(ast.src1)
            w2 = lane_half(ast.src2)
            res16 = ref_min3_f16(w0, w1, w2, clamp=ast.clamp)

            old = dst_list[lane]
            if ast.dst_hi:
                dst_list[lane] = (old & 0x0000FFFF) | ((res16 & MASK16) << 16)
            else:
                dst_list[lane] = (old & 0xFFFF0000) | (res16 & MASK16)
        return dst_list


class TestValuMin3F16SimdDivergence(unittest.TestCase):
    """Commit 4: Multi-lane execution under Wave32 and Wave64 masks."""

    def test_wave32_divergence(self):
        wave = SimdF16Wave(32)
        wave.exec_mask = 0b10101010_10101010_10101010_10101010
        wave.init_vgpr("v0", [0xAAAA0000 + i for i in range(32)])
        wave.init_vgpr("v1", [POS_ONE if i % 2 == 0 else POS_TWO for i in range(32)])
        wave.init_vgpr("v2", [POS_HALF] * 32)
        wave.init_vgpr("v3", [POS_FOUR] * 32)

        ast = VMin3F16AST.parse("v_min3_f16 v0, v1, v2, v3")
        res = wave.execute(ast, {})
        for lane in range(32):
            if (wave.exec_mask >> lane) & 1:
                self.assertEqual(res[lane], 0xAAAA3800)
            else:
                self.assertEqual(res[lane], 0xAAAA0000 + lane)

        res1 = run_hw_min3_f16(POS_ONE, POS_HALF, POS_FOUR, old_dst=0xAAAA0000, name="ps_simd_w32_1")
        self.assertEqual(res1, 0xAAAA3800)
        res2 = run_hw_min3_f16(POS_TWO, POS_HALF, POS_FOUR, old_dst=0xAAAA0001, name="ps_simd_w32_2")
        self.assertEqual(res2, 0xAAAA3800)

    def test_wave64_divergence(self):
        wave = SimdF16Wave(64)
        wave.exec_mask = 0x00000000_FFFFFFFF  # lower 32 lanes active only
        wave.init_vgpr("v0", [0xDEAD0000] * 64)
        wave.init_vgpr("v1", [POS_ONE] * 64)
        wave.init_vgpr("v2", [POS_TWO] * 64)
        wave.init_vgpr("v3", [POS_HALF] * 64)

        ast = VMin3F16AST.parse("v_min3_f16 v0, v1, v2, v3")
        res = wave.execute(ast, {})
        for lane in range(64):
            if lane < 32:
                self.assertEqual(res[lane], 0xDEAD3800)
            else:
                self.assertEqual(res[lane], 0xDEAD0000)

        hw = run_hw_min3_f16(POS_ONE, POS_TWO, POS_HALF, old_dst=0xDEAD0000, name="ps_simd_w64")
        self.assertEqual(hw, 0xDEAD3800)


# -----------------------------------------------------------------------------
# Commit 5: Register aliasing and in-place operand preservation
# -----------------------------------------------------------------------------
class TestValuMin3F16RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing (dst == src0, src1, src2) and preservation."""

    def test_dst_aliases_src0(self):
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = 0x55554000  # lo=2.0, hi=0x5555
        mach.sgprs["s1"] = 0x3C00      # 1.0
        mach.sgprs["s2"] = 0x4200      # 3.0
        ast = VMin3F16AST.parse("v_min3_f16 v0, v0, s1, s2")
        out = mach.execute(ast)
        self.assertEqual(out, 0x55553C00)

        hw = run_hw_min3_f16_custom(
            "v_min3_f16 v0, v0, s1, s2",
            old_dst=0x55554000,
            sgprs={"s1": 0x3C00, "s2": 0x4200},
            name="ps_al_src0",
        )
        self.assertEqual(hw, 0x55553C00)

    def test_dst_aliases_src1_and_src2(self):
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = 0xAAAA4200  # lo=3.0, hi=0xAAAA
        mach.sgprs["s0"] = 0x4400      # 4.0
        ast = VMin3F16AST.parse("v_min3_f16 v0, s0, v0, v0")
        out = mach.execute(ast)
        self.assertEqual(out, 0xAAAA4200)

        hw = run_hw_min3_f16_custom(
            "v_min3_f16 v0, s0, v0, v0",
            old_dst=0xAAAA4200,
            sgprs={"s0": 0x4400},
            name="ps_al_src12",
        )
        self.assertEqual(hw, 0xAAAA4200)

    def test_all_operands_same_register_high_low(self):
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = (POS_HALF << 16) | POS_TWO  # lo=2.0, hi=0.5
        ast = VMin3F16AST.parse("v_min3_f16 v0, v0, v0, v0 op_sel:[0, 1, 1, 1]")
        out = mach.execute(ast)
        # min(2.0, 0.5, 0.5) = 0.5 -> written to hi, lo preserved as 2.0
        self.assertEqual(out, (POS_HALF << 16) | POS_TWO)

        init_val = (POS_HALF << 16) | POS_TWO
        hw = run_hw_min3_f16_custom(
            "v_min3_f16 v0, v0, v0, v0 op_sel:[0, 1, 1, 1]",
            old_dst=init_val,
            name="ps_al_all",
        )
        self.assertEqual(hw, init_val)


# -----------------------------------------------------------------------------
# Commit 6: Condition codes, status flags, and carry/borrow generation
# -----------------------------------------------------------------------------
class TestValuMin3F16ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes (VCC/SCC) isolation under v_min3_f16."""

    def test_vcc_and_scc_preserved(self):
        # v_min3_f16 is a pure VALU data instruction; does not alter SCC or VCC
        scc_initial = 1
        vcc_initial = 0xFFFFFFFFFFFFFFFF
        mach = ScalarF16Machine()
        mach.vgprs["v0"] = 0
        mach.sgprs["s0"] = POS_ONE
        mach.sgprs["s1"] = POS_TWO
        mach.sgprs["s2"] = POS_HALF
        ast = VMin3F16AST.parse("v_min3_f16 v0, s0, s1, s2")
        mach.execute(ast)
        scc_after = scc_initial
        vcc_after = vcc_initial
        self.assertEqual(scc_after, 1)
        self.assertEqual(vcc_after, 0xFFFFFFFFFFFFFFFF)

        lines = [
            "s_mov_b32 s0, 0x3C00",
            "s_mov_b32 s1, 0x4000",
            "s_mov_b32 s2, 0x3800",
            "s_mov_b32 s3, 1",
            "s_cmp_eq_u32 s3, 1",  # sets SCC to 1
            "v_mov_b32_e32 v0, 0",
            "v_min3_f16 v0, s0, s1, s2",
            "s_cselect_b32 s5, 0xCAFE, 0xBEEF",  # reads SCC
            "v_mov_b32_e32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cc_check", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, POS_HALF)
        self.assertEqual(words[1], 0xCAFE)


# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling
# -----------------------------------------------------------------------------
class TestValuMin3F16Ieee754Boundaries(unittest.TestCase):
    """Commit 7: Boundary extremes, subnormals, infinities, and NaNs."""

    def test_infinities(self):
        self.assertEqual(ref_min3_f16(POS_INF, POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(ref_min3_f16(NEG_INF, POS_ONE, POS_TWO), NEG_INF)
        self.assertEqual(ref_min3_f16(POS_INF, POS_INF, POS_INF), POS_INF)
        self.assertEqual(ref_min3_f16(NEG_INF, NEG_INF, POS_INF), NEG_INF)

        hw1 = run_hw_min3_f16(POS_INF, POS_ONE, POS_TWO, name="ps_bnd_inf1")
        self.assertEqual(hw1, 0x12343C00)
        hw2 = run_hw_min3_f16(NEG_INF, POS_ONE, POS_TWO, name="ps_bnd_inf2")
        self.assertEqual(hw2, 0x1234FC00)
        hw3 = run_hw_min3_f16(POS_INF, POS_INF, POS_INF, name="ps_bnd_inf3")
        self.assertEqual(hw3, 0x12347C00)
        hw4 = run_hw_min3_f16(NEG_INF, NEG_INF, POS_INF, name="ps_bnd_inf4")
        self.assertEqual(hw4, 0x1234FC00)

    def test_subnormals(self):
        self.assertEqual(ref_min3_f16(POS_MIN_SUBNORM, POS_MAX_SUBNORM, POS_MIN_NORMAL), POS_MIN_SUBNORM)
        self.assertEqual(ref_min3_f16(NEG_MIN_SUBNORM, NEG_MAX_SUBNORM, NEG_ZERO), NEG_MAX_SUBNORM)
        self.assertEqual(ref_min3_f16(POS_MIN_SUBNORM, NEG_MIN_SUBNORM, POS_ZERO), NEG_MIN_SUBNORM)

        hw1 = run_hw_min3_f16(POS_MIN_SUBNORM, POS_MAX_SUBNORM, POS_MIN_NORMAL, name="ps_bnd_sub1")
        self.assertEqual(hw1, 0x12340001)
        hw2 = run_hw_min3_f16(NEG_MIN_SUBNORM, NEG_MAX_SUBNORM, NEG_ZERO, name="ps_bnd_sub2")
        self.assertEqual(hw2, 0x123483FF)
        hw3 = run_hw_min3_f16(POS_MIN_SUBNORM, NEG_MIN_SUBNORM, POS_ZERO, name="ps_bnd_sub3")
        self.assertEqual(hw3, 0x12348001)

    def test_nans_and_gradual_underflow(self):
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, POS_MIN_SUBNORM, POS_ONE), POS_MIN_SUBNORM)
        self.assertEqual(ref_min3_f16(SNAN, QNAN_ALT, NEG_MAX_NORMAL), NEG_MAX_NORMAL)
        self.assertTrue(is_nan16(ref_min3_f16(SNAN, QNAN_CANONICAL, QNAN_ALT)))

        hw1 = run_hw_min3_f16(QNAN_CANONICAL, POS_MIN_SUBNORM, POS_ONE, name="ps_bnd_nan1")
        self.assertEqual(hw1, 0x12340001)
        hw2 = run_hw_min3_f16(SNAN, QNAN_ALT, NEG_MAX_NORMAL, name="ps_bnd_nan2")
        self.assertEqual(hw2, 0x1234FBFF)
        hw3 = run_hw_min3_f16(SNAN, QNAN_CANONICAL, QNAN_ALT, name="ps_bnd_nan3")
        self.assertEqual(hw3, 0x12347E00)
        hw4 = run_hw_min3_f16(NEG_QNAN, POS_TWO, POS_THREE, name="ps_bnd_nan4")
        self.assertEqual(hw4, 0x12344000)
        hw5 = run_hw_min3_f16(POS_FOUR, NEG_QNAN, POS_THREE, name="ps_bnd_nan5")
        self.assertEqual(hw5, 0x12344200)
        hw6 = run_hw_min3_f16(POS_FOUR, POS_THREE, NEG_QNAN, name="ps_bnd_nan6")
        self.assertEqual(hw6, 0x12344200)


# -----------------------------------------------------------------------------
# Commit 8: Negative controls, syntax rejection, and mutation detection
# -----------------------------------------------------------------------------
class TestValuMin3F16NegativeAndMutation(unittest.TestCase):
    """Commit 8: Syntax rejection, negative assertions, mutation detection."""

    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            VMin3F16AST.parse("v_min3_f16")
        with self.assertRaises(ValueError):
            VMin3F16AST.parse("v_min3_f16 v0, s0")
        with self.assertRaises(ValueError):
            VMin3F16AST.parse("v_max3_f16 v0, s0, s1, s2")

    def test_mutation_detection_operator_swap(self):
        # Mutate translated SPIR-V replacing NMin with NMax: must diverge from correct minimum
        lines = [
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 s0, 0x3C00",  # 1.0
            "s_mov_b32 s1, 0x4000",  # 2.0
            "s_mov_b32 s2, 0x3800",  # 0.5
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_swap", lines)
        normal_res = support.mrt0_words(spv)[0] & MASK16
        self.assertEqual(normal_res, POS_HALF)

        mutated_spv = spv.replace("glsl NMin", "glsl NMax")
        mutated_res = support.mrt0_words(mutated_spv)[0] & MASK16
        self.assertEqual(mutated_res, POS_TWO)
        self.assertNotEqual(normal_res, mutated_res)

    def test_mutation_detection_signed_zero_loss(self):
        # Verify hardware translation orders -0.0 < +0.0
        res = run_hw_min3_f16(POS_ZERO, NEG_ZERO, POS_ONE, name="ps_mut_sz")
        self.assertEqual(res & MASK16, NEG_ZERO)

    def test_mutation_detection_nmin_to_fmin_nan_rejection(self):
        # Mutate translated SPIR-V replacing NMin with FMin: FMin with NaN operand is undefined/rejected
        lines = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {QNAN_CANONICAL:#x}",
            f"s_mov_b32 s1, {POS_ONE:#x}",
            f"s_mov_b32 s2, {POS_TWO:#x}",
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_fmin", lines)
        self.assertEqual(support.mrt0_words(spv)[0] & MASK16, POS_ONE)

        mutated_spv = spv.replace("glsl NMin", "glsl FMin")
        with self.assertRaises(si.UnsupportedConstruct):
            support.mrt0_words(mutated_spv)


class TestValuMin3F16Translator(unittest.TestCase):
    """Commit 8: Translator integration via support.translate_lines and mrt0_words."""

    def test_translator_basic_values(self):
        lines = [
            "v_mov_b32_e32 v0, 0x12345678",
            "s_mov_b32 s0, 0x3C00",  # 1.0
            "s_mov_b32 s1, 0x4000",  # 2.0
            "s_mov_b32 s2, 0x3800",  # 0.5
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x12343800)

    def test_translator_signed_zero(self):
        lines = [
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 s0, 0x0000",  # +0.0
            "s_mov_b32 s1, 0x8000",  # -0.0
            "s_mov_b32 s2, 0x3C00",  # 1.0
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_sz", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, NEG_ZERO)

    def test_translator_nan_suppression(self):
        lines = [
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 s0, 0x7E00",  # NaN
            "s_mov_b32 s1, 0x3C00",  # 1.0
            "s_mov_b32 s2, 0x4000",  # 2.0
            "v_min3_f16 v0, s0, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_nan", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, POS_ONE)

    def test_translator_nan_permutations(self):
        # NaN in operand 1
        res1 = run_hw_min3_f16(POS_TWO, QNAN_CANONICAL, POS_ONE, name="ps_tr_nan_p1")
        self.assertEqual(res1 & MASK16, POS_ONE)
        # NaN in operand 2
        res2 = run_hw_min3_f16(POS_TWO, POS_ONE, QNAN_CANONICAL, name="ps_tr_nan_p2")
        self.assertEqual(res2 & MASK16, POS_ONE)
        # NaN in operands 0 and 1
        res3 = run_hw_min3_f16(QNAN_CANONICAL, QNAN_ALT, POS_THREE, name="ps_tr_nan_p3")
        self.assertEqual(res3 & MASK16, POS_THREE)
        # NaN in operands 1 and 2
        res4 = run_hw_min3_f16(POS_THREE, QNAN_CANONICAL, QNAN_ALT, name="ps_tr_nan_p4")
        self.assertEqual(res4 & MASK16, POS_THREE)
        # NaN in operands 0 and 2
        res5 = run_hw_min3_f16(QNAN_CANONICAL, POS_THREE, QNAN_ALT, name="ps_tr_nan_p5")
        self.assertEqual(res5 & MASK16, POS_THREE)

    def test_translator_clamp(self):
        lines = [
            "v_mov_b32_e32 v0, 0",
            "s_mov_b32 s0, 0x4000",  # 2.0
            "s_mov_b32 s1, 0x4200",  # 3.0
            "s_mov_b32 s2, 0x4400",  # 4.0
            "v_min3_f16 v0, s0, s1, s2 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_clamp", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, POS_ONE)

    def test_translator_op_sel_dst_hi(self):
        lines = [
            "v_mov_b32_e32 v0, 0x5555AAAA",
            "s_mov_b32 s0, 0x38004000",  # lo=2.0, hi=0.5
            "s_mov_b32 s1, 0x4200",      # 3.0
            "s_mov_b32 s2, 0x4400",      # 4.0
            "v_min3_f16 v0, s0, s1, s2 op_sel:[1, 0, 0, 1]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_opsel", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3800AAAA)

    def test_translator_input_modifiers(self):
        lines = [
            "v_mov_b32_e32 v0, 0x12340000",
            "s_mov_b32 s0, 0x3C00",  # 1.0 -> -|1.0| = -1.0 (0xBC00)
            "s_mov_b32 s1, 0x4000",  # 2.0
            "s_mov_b32 s2, 0x4200",  # 3.0
            "v_min3_f16 v0, -|s0|, s1, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_mods", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x1234BC00)


if __name__ == "__main__":
    unittest.main()
