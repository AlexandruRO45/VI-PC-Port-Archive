from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

def ref_v_add_i16(s0: int, s1: int, clamp: bool = False) -> int:
    x = (s0 & 0x7FFF) - (s0 & 0x8000)
    y = (s1 & 0x7FFF) - (s1 & 0x8000)
    res = x + y
    if clamp:
        res = max(-32768, min(32767, res))
    return res & MASK16


class TestVAddI16Reference(unittest.TestCase):
    def test_ref_add_positive(self):
        self.assertEqual(ref_v_add_i16(100, 200), 300)

    def test_ref_add_negative(self):
        self.assertEqual(ref_v_add_i16(0xFFFF, 0xFFFF), 0xFFFE)  # -1 + -1 = -2

    def test_ref_add_wrap(self):
        self.assertEqual(ref_v_add_i16(0x7FFF, 1), 0x8000)  # 32767 + 1 = -32768 (wrap)

    def test_ref_add_clamp(self):
        self.assertEqual(ref_v_add_i16(0x7FFF, 1, clamp=True), 0x7FFF)  # 32767 + 1 clamped
        self.assertEqual(ref_v_add_i16(0x8000, 0xFFFF, clamp=True), 0x8000)  # -32768 + -1 clamped

class ValuInstruction:
    def __init__(self, op: str, dst: int, s0: int, s1: int, clamp: bool = False):
        self.op = op
        self.dst = dst
        self.s0 = s0
        self.s1 = s1
        self.clamp = clamp


class ValuInterpreterState:
    HANDLERS = {}

    def __init__(self):
        self.vgpr: dict[int, int] = {i: 0 for i in range(256)}

    @classmethod
    def register(cls, op: str):
        def decorator(fn):
            cls.HANDLERS[op] = fn
            return fn
        return decorator

    def step(self, inst: ValuInstruction):
        if inst.op not in self.HANDLERS:
            raise NotImplementedError(f"Unsupported opcode: {inst.op}")
        return self.HANDLERS[inst.op](self, inst)


@ValuInterpreterState.register("v_add_i16")
def _step_add_i16(state: ValuInterpreterState, inst: ValuInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.vgpr[int(inst.s0.lstrip("v"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.vgpr[int(inst.s1.lstrip("v"))]
    res = ref_v_add_i16(s0_val, s1_val, clamp=inst.clamp)
    state.vgpr[inst.dst] = res
    return res


class TestVAddI16Interpreter(unittest.TestCase):
    def test_interpreter_step_add_i16(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_i16", dst=0, s0=100, s1=200)
        res = state.step(inst)
        self.assertEqual(res, 300)
        self.assertEqual(state.vgpr[0], 300)

    def test_interpreter_step_add_i16_clamped(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_i16", dst=1, s0=30000, s1=10000, clamp=True)
        res = state.step(inst)
        self.assertEqual(res, 0x7FFF)
        self.assertEqual(state.vgpr[1], 0x7FFF)

class TestVAddI16EdgeCases(unittest.TestCase):
    def test_extremum_bounds(self):
        self.assertEqual(ref_v_add_i16(0x7FFF, 0x7FFF, clamp=True), 0x7FFF)
        self.assertEqual(ref_v_add_i16(0x8000, 0x8000, clamp=True), 0x8000)

    def test_simd_multi_lane_divergence(self):
        lanes_a = [i * 500 for i in range(64)]
        lanes_b = [i * 200 for i in range(64)]
        res = [ref_v_add_i16(a, b) for a, b in zip(lanes_a, lanes_b)]
        expected = [(i * 700) & MASK16 for i in range(64)]
        self.assertEqual(res, expected)

class TestVAddI16Translator(unittest.TestCase):
    def test_translate_v_add_i16(self):
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_add_i16 v0, v1, v2",
            "v_mov_b32 v3, 30000",
            "v_mov_b32 v4, 10000",
            "v_add_i16 v1, v3, v4 clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_i16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [300, 0x7FFF, 300, 0x7FFF])

def ref_v_add_u16(s0: int, s1: int, clamp: bool = False) -> int:
    x = s0 & MASK16
    y = s1 & MASK16
    res = x + y
    if clamp:
        res = min(MASK16, res)
    return res & MASK16


class TestVAddU16Reference(unittest.TestCase):
    def test_ref_add_u16(self):
        self.assertEqual(ref_v_add_u16(100, 200), 300)

    def test_ref_add_u16_wrap(self):
        self.assertEqual(ref_v_add_u16(60000, 10000), 4464)  # 70000 % 65536 = 4464

    def test_ref_add_u16_clamp(self):
        self.assertEqual(ref_v_add_u16(60000, 10000, clamp=True), 0xFFFF)

@ValuInterpreterState.register("v_add_u16")
def _step_add_u16(state: ValuInterpreterState, inst: ValuInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.vgpr[int(inst.s0.lstrip("v"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.vgpr[int(inst.s1.lstrip("v"))]
    res = ref_v_add_u16(s0_val, s1_val, clamp=inst.clamp)
    state.vgpr[inst.dst] = res
    return res


class TestVAddU16Interpreter(unittest.TestCase):
    def test_interpreter_step_add_u16(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_u16", dst=0, s0=500, s1=250)
        res = state.step(inst)
        self.assertEqual(res, 750)
        self.assertEqual(state.vgpr[0], 750)

    def test_interpreter_step_add_u16_clamped(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_u16", dst=1, s0=60000, s1=10000, clamp=True)
        res = state.step(inst)
        self.assertEqual(res, 0xFFFF)
        self.assertEqual(state.vgpr[1], 0xFFFF)


if __name__ == "__main__":
    unittest.main()
