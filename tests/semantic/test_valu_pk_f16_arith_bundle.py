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

if __name__ == '__main__':
    unittest.main()
