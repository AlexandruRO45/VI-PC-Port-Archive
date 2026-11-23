from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(ROOT, "tools")
for p in (HERE, TOOLS):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import mrt0_words, translate_lines
import rdna2spirv

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF
POS_ZERO = 0x0000
NEG_ZERO = 0x8000
POS_INF = 0x7C00
NEG_INF = 0xFC00
QNAN = 0x7E00


def f16_to_bits(x: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(x)))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f16(h: int) -> float:
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def pack_f16x2(lo: int, hi: int) -> int:
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_f16x2(word: int) -> tuple[int, int]:
    return (word & MASK16, (word >> 16) & MASK16)


def clamp_f16(x: float) -> float:
    if math.isnan(x):
        return x
    return max(0.0, min(1.0, x))

def f16_min(x: float, y: float) -> float:
    if math.isnan(x) and math.isnan(y):
        return float('nan')
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == 0.0 and y == 0.0:
        bx = f16_to_bits(x)
        by = f16_to_bits(y)
        if (bx & NEG_ZERO) or (by & NEG_ZERO):
            return -0.0
        return 0.0
    return min(x, y)


def ref_pk_min_f16(w_a: int, w_b: int, *, clamp: bool = False,
                   op_sel: tuple[int, int] = (0, 0),
                   op_sel_hi: tuple[int, int] = (1, 1),
                   neg_lo: tuple[int, int] = (0, 0),
                   neg_hi: tuple[int, int] = (0, 0)) -> int:
    a_halves = unpack_f16x2(w_a)
    b_halves = unpack_f16x2(w_b)
    s0_lo = bits_to_f16(a_halves[op_sel[0]])
    s1_lo = bits_to_f16(b_halves[op_sel[1]])
    if neg_lo[0]: s0_lo = -s0_lo
    if neg_lo[1]: s1_lo = -s1_lo
    r_lo = f16_min(s0_lo, s1_lo)
    if clamp: r_lo = clamp_f16(r_lo)

    s0_hi = bits_to_f16(a_halves[op_sel_hi[0]])
    s1_hi = bits_to_f16(b_halves[op_sel_hi[1]])
    if neg_hi[0]: s0_hi = -s0_hi
    if neg_hi[1]: s1_hi = -s1_hi
    r_hi = f16_min(s0_hi, s1_hi)
    if clamp: r_hi = clamp_f16(r_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

def f16_max(x: float, y: float) -> float:
    if math.isnan(x) and math.isnan(y):
        return float('nan')
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == 0.0 and y == 0.0:
        bx = f16_to_bits(x)
        by = f16_to_bits(y)
        if (bx & NEG_ZERO) and (by & NEG_ZERO):
            return -0.0
        return 0.0
    return max(x, y)


def ref_pk_max_f16(w_a: int, w_b: int, *, clamp: bool = False,
                   op_sel: tuple[int, int] = (0, 0),
                   op_sel_hi: tuple[int, int] = (1, 1),
                   neg_lo: tuple[int, int] = (0, 0),
                   neg_hi: tuple[int, int] = (0, 0)) -> int:
    a_halves = unpack_f16x2(w_a)
    b_halves = unpack_f16x2(w_b)
    s0_lo = bits_to_f16(a_halves[op_sel[0]])
    s1_lo = bits_to_f16(b_halves[op_sel[1]])
    if neg_lo[0]: s0_lo = -s0_lo
    if neg_lo[1]: s1_lo = -s1_lo
    r_lo = f16_max(s0_lo, s1_lo)
    if clamp: r_lo = clamp_f16(r_lo)

    s0_hi = bits_to_f16(a_halves[op_sel_hi[0]])
    s1_hi = bits_to_f16(b_halves[op_sel_hi[1]])
    if neg_hi[0]: s0_hi = -s0_hi
    if neg_hi[1]: s1_hi = -s1_hi
    r_hi = f16_max(s0_hi, s1_hi)
    if clamp: r_hi = clamp_f16(r_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

def ref_pk_fma_f16(w_a: int, w_b: int, w_c: int, *, clamp: bool = False,
                   op_sel: tuple[int, int, int] = (0, 0, 0),
                   op_sel_hi: tuple[int, int, int] = (1, 1, 1),
                   neg_lo: tuple[int, int, int] = (0, 0, 0),
                   neg_hi: tuple[int, int, int] = (0, 0, 0)) -> int:
    a_halves = unpack_f16x2(w_a)
    b_halves = unpack_f16x2(w_b)
    c_halves = unpack_f16x2(w_c)

    s0_lo = bits_to_f16(a_halves[op_sel[0]])
    s1_lo = bits_to_f16(b_halves[op_sel[1]])
    s2_lo = bits_to_f16(c_halves[op_sel[2]])
    if neg_lo[0]: s0_lo = -s0_lo
    if neg_lo[1]: s1_lo = -s1_lo
    if neg_lo[2]: s2_lo = -s2_lo
    r_lo = s0_lo * s1_lo + s2_lo
    if clamp: r_lo = clamp_f16(r_lo)

    s0_hi = bits_to_f16(a_halves[op_sel_hi[0]])
    s1_hi = bits_to_f16(b_halves[op_sel_hi[1]])
    s2_hi = bits_to_f16(c_halves[op_sel_hi[2]])
    if neg_hi[0]: s0_hi = -s0_hi
    if neg_hi[1]: s1_hi = -s1_hi
    if neg_hi[2]: s2_hi = -s2_hi
    r_hi = s0_hi * s1_hi + s2_hi
    if clamp: r_hi = clamp_f16(r_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

def ref_pk_fmac_f16(w_acc: int, w_a: int, w_b: int) -> int:
    acc_lo, acc_hi = unpack_f16x2(w_acc)
    a_lo, a_hi = unpack_f16x2(w_a)
    b_lo, b_hi = unpack_f16x2(w_b)
    r_lo = bits_to_f16(a_lo) * bits_to_f16(b_lo) + bits_to_f16(acc_lo)
    r_hi = bits_to_f16(a_hi) * bits_to_f16(b_hi) + bits_to_f16(acc_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

def ref_pk_mul_f16(w_a: int, w_b: int, *, clamp: bool = False,
                   op_sel: tuple[int, int] = (0, 0),
                   op_sel_hi: tuple[int, int] = (1, 1),
                   neg_lo: tuple[int, int] = (0, 0),
                   neg_hi: tuple[int, int] = (0, 0)) -> int:
    a_halves = unpack_f16x2(w_a)
    b_halves = unpack_f16x2(w_b)
    s0_lo = bits_to_f16(a_halves[op_sel[0]])
    s1_lo = bits_to_f16(b_halves[op_sel[1]])
    if neg_lo[0]: s0_lo = -s0_lo
    if neg_lo[1]: s1_lo = -s1_lo
    r_lo = s0_lo * s1_lo
    if clamp: r_lo = clamp_f16(r_lo)

    s0_hi = bits_to_f16(a_halves[op_sel_hi[0]])
    s1_hi = bits_to_f16(b_halves[op_sel_hi[1]])
    if neg_hi[0]: s0_hi = -s0_hi
    if neg_hi[1]: s1_hi = -s1_hi
    r_hi = s0_hi * s1_hi
    if clamp: r_hi = clamp_f16(r_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

def ref_pk_add_f16(w_a: int, w_b: int, *, clamp: bool = False,
                   op_sel: tuple[int, int] = (0, 0),
                   op_sel_hi: tuple[int, int] = (1, 1),
                   neg_lo: tuple[int, int] = (0, 0),
                   neg_hi: tuple[int, int] = (0, 0)) -> int:
    a_halves = unpack_f16x2(w_a)
    b_halves = unpack_f16x2(w_b)
    s0_lo = bits_to_f16(a_halves[op_sel[0]])
    s1_lo = bits_to_f16(b_halves[op_sel[1]])
    if neg_lo[0]: s0_lo = -s0_lo
    if neg_lo[1]: s1_lo = -s1_lo
    r_lo = s0_lo + s1_lo
    if clamp: r_lo = clamp_f16(r_lo)

    s0_hi = bits_to_f16(a_halves[op_sel_hi[0]])
    s1_hi = bits_to_f16(b_halves[op_sel_hi[1]])
    if neg_hi[0]: s0_hi = -s0_hi
    if neg_hi[1]: s1_hi = -s1_hi
    r_hi = s0_hi + s1_hi
    if clamp: r_hi = clamp_f16(r_hi)
    return pack_f16x2(f16_to_bits(r_lo), f16_to_bits(r_hi))

class MachineState:
    def __init__(self, num_lanes: int = 64):
        self.num_lanes = num_lanes
        self.exec_mask: int = (1 << num_lanes) - 1
        self.vgpr: dict[int, list[int]] = {i: [0] * num_lanes for i in range(256)}

    def lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)
class InstructionAST:
    def __init__(self, op: str, dst: int, srcs: list[int], *, clamp: bool = False,
                 op_sel: tuple[int, ...] = (), op_sel_hi: tuple[int, ...] = (),
                 neg_lo: tuple[int, ...] = (), neg_hi: tuple[int, ...] = ()):
        self.op = op
        self.dst = dst
        self.srcs = srcs
        self.clamp = clamp
        self.op_sel = op_sel
        self.op_sel_hi = op_sel_hi
        self.neg_lo = neg_lo
        self.neg_hi = neg_hi

    def eval_scalar(self, *src_vals: int, acc: int = 0) -> int:
        if self.op == "v_pk_min_f16":
            return ref_pk_min_f16(src_vals[0], src_vals[1], clamp=self.clamp, op_sel=self.op_sel or (0, 0), op_sel_hi=self.op_sel_hi or (1, 1), neg_lo=self.neg_lo or (0, 0), neg_hi=self.neg_hi or (0, 0))
        elif self.op == "v_pk_max_f16":
            return ref_pk_max_f16(src_vals[0], src_vals[1], clamp=self.clamp, op_sel=self.op_sel or (0, 0), op_sel_hi=self.op_sel_hi or (1, 1), neg_lo=self.neg_lo or (0, 0), neg_hi=self.neg_hi or (0, 0))
        elif self.op == "v_pk_fma_f16":
            return ref_pk_fma_f16(src_vals[0], src_vals[1], src_vals[2], clamp=self.clamp, op_sel=self.op_sel or (0, 0, 0), op_sel_hi=self.op_sel_hi or (1, 1, 1), neg_lo=self.neg_lo or (0, 0, 0), neg_hi=self.neg_hi or (0, 0, 0))
        elif self.op == "v_pk_fmac_f16":
            return ref_pk_fmac_f16(acc, src_vals[0], src_vals[1])
        elif self.op == "v_pk_mul_f16":
            return ref_pk_mul_f16(src_vals[0], src_vals[1], clamp=self.clamp, op_sel=self.op_sel or (0, 0), op_sel_hi=self.op_sel_hi or (1, 1), neg_lo=self.neg_lo or (0, 0), neg_hi=self.neg_hi or (0, 0))
        elif self.op == "v_pk_add_f16":
            return ref_pk_add_f16(src_vals[0], src_vals[1], clamp=self.clamp, op_sel=self.op_sel or (0, 0), op_sel_hi=self.op_sel_hi or (1, 1), neg_lo=self.neg_lo or (0, 0), neg_hi=self.neg_hi or (0, 0))
        raise NotImplementedError(f"Unsupported op: {self.op}")

    def execute_simd(self, state: MachineState) -> None:
        for lane in range(state.num_lanes):
            if state.lane_active(lane):
                vals = [state.vgpr[s][lane] for s in self.srcs]
                acc = state.vgpr[self.dst][lane]
                state.vgpr[self.dst][lane] = self.eval_scalar(*vals, acc=acc)

class TestValuPkF16ArithBundle(unittest.TestCase):

    def test_v_pk_min_f16_reference_oracle(self):
        # 1.0 (0x3c00), 2.0 (0x4000), 3.0 (0x4200), -2.0 (0xc000)
        w1 = pack_f16x2(0x3c00, 0x4000) # lo: 1.0, hi: 2.0
        w2 = pack_f16x2(0x4000, 0x4200) # lo: 2.0, hi: 3.0
        res = ref_pk_min_f16(w1, w2)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x3c00) # min(1.0, 2.0) = 1.0
        self.assertEqual(hi, 0x4000) # min(2.0, 3.0) = 2.0

        # Clamped: min(-2.0, 2.0) -> -2.0 clamped to 0.0
        w_neg = pack_f16x2(0xc000, 0x4200)
        res_clamp = ref_pk_min_f16(w_neg, w2, clamp=True)
        c_lo, c_hi = unpack_f16x2(res_clamp)
        self.assertEqual(c_lo, 0x0000)
        self.assertEqual(c_hi, 0x3c00) # 2.0 clamped to 1.0

    def test_v_pk_min_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_min_f16", dst=0, srcs=[1, 2])
        w1 = pack_f16x2(0x3c00, 0x4000)
        w2 = pack_f16x2(0x4000, 0x4200)
        val = ast.eval_scalar(w1, w2)
        self.assertEqual(val, pack_f16x2(0x3c00, 0x4000))

    def test_v_pk_min_f16_simd_divergence_and_boundaries(self):
        state = MachineState(num_lanes=64)
        state.exec_mask = 0x00000000FFFFFFFF # first 32 active
        for l in range(64):
            state.vgpr[0][l] = 0xDEADBEEF
            state.vgpr[1][l] = pack_f16x2(0x3c00, 0x4200) # 1.0, 3.0
            state.vgpr[2][l] = pack_f16x2(0x4000, 0x4000) # 2.0, 2.0
        ast = InstructionAST("v_pk_min_f16", dst=0, srcs=[1, 2])
        ast.execute_simd(state)
        for l in range(32):
            self.assertEqual(state.vgpr[0][l], pack_f16x2(0x3c00, 0x4000))
        for l in range(32, 64):
            self.assertEqual(state.vgpr[0][l], 0xDEADBEEF)

        # Boundary checks: NaN, Inf, -0.0
        res_nan = ref_pk_min_f16(pack_f16x2(QNAN, POS_INF), pack_f16x2(0x3c00, 0x4000))
        lo, hi = unpack_f16x2(res_nan)
        self.assertEqual(lo, 0x3c00) # NMin returns non-NaN
        self.assertEqual(hi, 0x4000) # min(Inf, 2.0) = 2.0

    def test_v_pk_min_f16_translator_integration_and_negative_controls(self):
        lines = [
            "v_mov_b32 v1, 0x40003c00", # hi: 2.0, lo: 1.0
            "v_mov_b32 v2, 0x42004000", # hi: 3.0, lo: 2.0
            "v_pk_min_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_v_pk_min_f16", lines)
        out = mrt0_words(spv)[0]
        self.assertEqual(out, 0x40003c00)

        # Negative control
        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pk_min", ["v_pk_min_f16 v0", "s_endpgm"])

    def test_v_pk_max_f16_reference_oracle(self):
        w1 = pack_f16x2(0x3c00, 0x4000) # 1.0, 2.0
        w2 = pack_f16x2(0x4000, 0x4200) # 2.0, 3.0
        res = ref_pk_max_f16(w1, w2)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4000) # max(1.0, 2.0) = 2.0
        self.assertEqual(hi, 0x4200) # max(2.0, 3.0) = 3.0

        # Clamped: max(2.0, 3.0) clamped to 1.0
        res_clamp = ref_pk_max_f16(w1, w2, clamp=True)
        c_lo, c_hi = unpack_f16x2(res_clamp)
        self.assertEqual(c_lo, 0x3c00) # 1.0
        self.assertEqual(c_hi, 0x3c00) # 1.0

    def test_v_pk_max_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_max_f16", dst=0, srcs=[1, 2])
        w1 = pack_f16x2(0x3c00, 0x4000)
        w2 = pack_f16x2(0x4000, 0x4200)
        val = ast.eval_scalar(w1, w2)
        self.assertEqual(val, pack_f16x2(0x4000, 0x4200))

    def test_v_pk_max_f16_simd_divergence_and_boundaries(self):
        state = MachineState(num_lanes=64)
        state.exec_mask = 0x5555555555555555 # alternating lanes
        for l in range(64):
            state.vgpr[0][l] = 0xBAADF00D
            state.vgpr[1][l] = pack_f16x2(0x3c00, 0x4000)
            state.vgpr[2][l] = pack_f16x2(0x4000, 0x4200)
        ast = InstructionAST("v_pk_max_f16", dst=0, srcs=[1, 2])
        ast.execute_simd(state)
        for l in range(64):
            if l % 2 == 0:
                self.assertEqual(state.vgpr[0][l], pack_f16x2(0x4000, 0x4200))
            else:
                self.assertEqual(state.vgpr[0][l], 0xBAADF00D)

        # Boundary checks: NaN, -Inf, +0.0 vs -0.0
        res_nan = ref_pk_max_f16(pack_f16x2(QNAN, NEG_INF), pack_f16x2(0x3c00, 0x4000))
        lo, hi = unpack_f16x2(res_nan)
        self.assertEqual(lo, 0x3c00)
        self.assertEqual(hi, 0x4000)

    def test_v_pk_max_f16_translator_integration_and_negative_controls(self):
        lines = [
            "v_mov_b32 v1, 0x40003c00", # hi: 2.0, lo: 1.0
            "v_mov_b32 v2, 0x42004000", # hi: 3.0, lo: 2.0
            "v_pk_max_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_v_pk_max_f16", lines)
        out = mrt0_words(spv)[0]
        self.assertEqual(out, 0x42004000)

        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pk_max", ["v_pk_max_f16 v0, v1", "s_endpgm"])

    def test_v_pk_fma_f16_reference_oracle(self):
        w_a = pack_f16x2(0x4000, 0x4000) # 2.0, 2.0
        w_b = pack_f16x2(0x4200, 0x4200) # 3.0, 3.0
        w_c = pack_f16x2(0x3c00, 0x3c00) # 1.0, 1.0
        # 2.0 * 3.0 + 1.0 = 7.0 (0x4700)
        res = ref_pk_fma_f16(w_a, w_b, w_c)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4700)
        self.assertEqual(hi, 0x4700)

        # Clamped: 7.0 clamped to 1.0 (0x3c00)
        res_c = ref_pk_fma_f16(w_a, w_b, w_c, clamp=True)
        clo, chi = unpack_f16x2(res_c)
        self.assertEqual(clo, 0x3c00)
        self.assertEqual(chi, 0x3c00)

    def test_v_pk_fma_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_fma_f16", dst=0, srcs=[1, 2, 3])
        w_a = pack_f16x2(0x4000, 0x4000)
        w_b = pack_f16x2(0x4200, 0x4200)
        w_c = pack_f16x2(0x3c00, 0x3c00)
        val = ast.eval_scalar(w_a, w_b, w_c)
        self.assertEqual(val, pack_f16x2(0x4700, 0x4700))

    def test_v_pk_fma_f16_simd_divergence_and_boundaries(self):
        state = MachineState(num_lanes=64)
        state.exec_mask = 0x000000000000000F # lowest 4 lanes
        for l in range(64):
            state.vgpr[0][l] = 0xCAFEBABE
            state.vgpr[1][l] = pack_f16x2(0x4000, 0x4000)
            state.vgpr[2][l] = pack_f16x2(0x4200, 0x4200)
            state.vgpr[3][l] = pack_f16x2(0x3c00, 0x3c00)
        ast = InstructionAST("v_pk_fma_f16", dst=0, srcs=[1, 2, 3])
        ast.execute_simd(state)
        for l in range(4):
            self.assertEqual(state.vgpr[0][l], pack_f16x2(0x4700, 0x4700))
        for l in range(4, 64):
            self.assertEqual(state.vgpr[0][l], 0xCAFEBABE)

        # Boundaries: 0 * Inf + c, etc.
        res_zero_inf = ref_pk_fma_f16(pack_f16x2(POS_ZERO, POS_ZERO), pack_f16x2(POS_INF, POS_INF), pack_f16x2(0x3c00, 0x3c00))
        lo, hi = unpack_f16x2(res_zero_inf)
        self.assertTrue(math.isnan(bits_to_f16(lo)))

    def test_v_pk_fma_f16_translator_integration_and_negative_controls(self):
        lines = [
            "v_mov_b32 v1, 0x40004000",
            "v_mov_b32 v2, 0x42004200",
            "v_mov_b32 v3, 0x3c003c00",
            "v_pk_fma_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_v_pk_fma_f16", lines)
        out = mrt0_words(spv)[0]
        self.assertEqual(out, 0x47004700)

        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pk_fma", ["v_pk_fma_f16 v0, v1, v2", "s_endpgm"])

    def test_v_pk_fmac_f16_reference_oracle(self):
        w_acc = pack_f16x2(0x3c00, 0x3c00) # 1.0, 1.0
        w_a = pack_f16x2(0x4000, 0x4000)   # 2.0, 2.0
        w_b = pack_f16x2(0x4200, 0x4200)   # 3.0, 3.0
        # 1.0 + 2.0 * 3.0 = 7.0 (0x4700)
        res = ref_pk_fmac_f16(w_acc, w_a, w_b)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4700)
        self.assertEqual(hi, 0x4700)

    def test_v_pk_fmac_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_fmac_f16", dst=0, srcs=[1, 2])
        w_acc = pack_f16x2(0x3c00, 0x3c00)
        w_a = pack_f16x2(0x4000, 0x4000)
        w_b = pack_f16x2(0x4000, 0x4000)
        # 1.0 + 2.0 * 2.0 = 5.0 (0x4500)
        val = ast.eval_scalar(w_a, w_b, acc=w_acc)
        self.assertEqual(val, pack_f16x2(0x4500, 0x4500))

    def test_v_pk_fmac_f16_simd_divergence_and_boundaries(self):
        state = MachineState(num_lanes=64)
        state.exec_mask = 0xF000000000000000 # top 4 lanes
        for l in range(64):
            state.vgpr[0][l] = pack_f16x2(0x3c00, 0x3c00)
            state.vgpr[1][l] = pack_f16x2(0x4000, 0x4000)
            state.vgpr[2][l] = pack_f16x2(0x4000, 0x4000)
        ast = InstructionAST("v_pk_fmac_f16", dst=0, srcs=[1, 2])
        ast.execute_simd(state)
        for l in range(60):
            self.assertEqual(state.vgpr[0][l], pack_f16x2(0x3c00, 0x3c00))
        for l in range(60, 64):
            self.assertEqual(state.vgpr[0][l], pack_f16x2(0x4500, 0x4500))

    def test_v_pk_fmac_f16_translator_integration_and_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x40004000",
            "v_mov_b32 v2, 0x40004000",
            "v_pk_fmac_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_v_pk_fmac_f16", lines)
        out = mrt0_words(spv)[0]
        self.assertEqual(out, 0x45004500)

        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pk_fmac", ["v_pk_fmac_f16 v0, v1", "s_endpgm"])

    def test_v_pk_mul_f16_reference_oracle(self):
        w_a = pack_f16x2(0x3c00, 0x4000) # 1.0, 2.0
        w_b = pack_f16x2(0x4000, 0x4200) # 2.0, 3.0
        res = ref_pk_mul_f16(w_a, w_b)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4000) # 1.0 * 2.0 = 2.0
        self.assertEqual(hi, 0x4600) # 2.0 * 3.0 = 6.0

        # Clamped: 6.0 clamped to 1.0 (0x3c00)
        res_c = ref_pk_mul_f16(w_a, w_b, clamp=True)
        clo, chi = unpack_f16x2(res_c)
        self.assertEqual(clo, 0x3c00) # 2.0 clamped to 1.0
        self.assertEqual(chi, 0x3c00) # 6.0 clamped to 1.0

    def test_v_pk_mul_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_mul_f16", dst=0, srcs=[1, 2])
        w_a = pack_f16x2(0x3c00, 0x4000)
        w_b = pack_f16x2(0x4000, 0x4200)
        val = ast.eval_scalar(w_a, w_b)
        self.assertEqual(val, pack_f16x2(0x4000, 0x4600))

    def test_v_pk_mul_f16_simd_divergence_and_boundaries(self):
        state = MachineState(num_lanes=64)
        state.exec_mask = 0x0000FFFF0000FFFF
        for l in range(64):
            state.vgpr[0][l] = 0x12345678
            state.vgpr[1][l] = pack_f16x2(0x3c00, 0x4000)
            state.vgpr[2][l] = pack_f16x2(0x4000, 0x4200)
        ast = InstructionAST("v_pk_mul_f16", dst=0, srcs=[1, 2])
        ast.execute_simd(state)
        for l in range(64):
            if state.lane_active(l):
                self.assertEqual(state.vgpr[0][l], pack_f16x2(0x4000, 0x4600))
            else:
                self.assertEqual(state.vgpr[0][l], 0x12345678)

        # Boundaries: Inf * 0 -> NaN
        res_nan = ref_pk_mul_f16(pack_f16x2(POS_INF, 0x3c00), pack_f16x2(POS_ZERO, 0x4000))
        lo, hi = unpack_f16x2(res_nan)
        self.assertTrue(math.isnan(bits_to_f16(lo)))
        self.assertEqual(hi, 0x4000)

    def test_v_pk_mul_f16_translator_integration_and_negative_controls(self):
        lines = [
            "v_mov_b32 v1, 0x40003c00", # hi: 2.0, lo: 1.0
            "v_mov_b32 v2, 0x42004000", # hi: 3.0, lo: 2.0
            "v_pk_mul_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_v_pk_mul_f16", lines)
        out = mrt0_words(spv)[0]
        self.assertEqual(out, 0x46004000)

        with self.assertRaises((ValueError, IndexError, rdna2spirv.Unsupported)):
            translate_lines("ps_bad_pk_mul", ["v_pk_mul_f16 v0", "s_endpgm"])

    def test_v_pk_add_f16_reference_oracle(self):
        w_a = pack_f16x2(0x3c00, 0x4000) # 1.0, 2.0
        w_b = pack_f16x2(0x4000, 0x4200) # 2.0, 3.0
        res = ref_pk_add_f16(w_a, w_b)
        lo, hi = unpack_f16x2(res)
        self.assertEqual(lo, 0x4200) # 1.0 + 2.0 = 3.0
        self.assertEqual(hi, 0x4500) # 2.0 + 3.0 = 5.0

        # Clamped: 3.0 and 5.0 clamped to 1.0
        res_c = ref_pk_add_f16(w_a, w_b, clamp=True)
        clo, chi = unpack_f16x2(res_c)
        self.assertEqual(clo, 0x3c00)
        self.assertEqual(chi, 0x3c00)

    def test_v_pk_add_f16_ast_and_scalar_interpreter(self):
        ast = InstructionAST("v_pk_add_f16", dst=0, srcs=[1, 2])
        w_a = pack_f16x2(0x3c00, 0x4000)
        w_b = pack_f16x2(0x4000, 0x4200)
        val = ast.eval_scalar(w_a, w_b)
        self.assertEqual(val, pack_f16x2(0x4200, 0x4500))

if __name__ == '__main__':
    unittest.main()
