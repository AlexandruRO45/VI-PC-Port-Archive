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

class TestVAddU16EdgeCases(unittest.TestCase):
    def test_zero_identity(self):
        self.assertEqual(ref_v_add_u16(0, 1234), 1234)
        self.assertEqual(ref_v_add_u16(1234, 0), 1234)

    def test_maximum_unsigned_overflow(self):
        self.assertEqual(ref_v_add_u16(0xFFFF, 0xFFFF), 0xFFFE)
        self.assertEqual(ref_v_add_u16(0xFFFF, 0xFFFF, clamp=True), 0xFFFF)

class TestVAddU16Translator(unittest.TestCase):
    def test_translate_v_add_u16(self):
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_add_u16 v0, v1, v2",
            "v_mov_b32 v3, 60000",
            "v_mov_b32 v4, 10000",
            "v_add_u16 v1, v3, v4 clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_u16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [300, 0xFFFF, 300, 0xFFFF])

def ref_v_sub_i16(s0: int, s1: int, clamp: bool = False) -> int:
    x = (s0 & 0x7FFF) - (s0 & 0x8000)
    y = (s1 & 0x7FFF) - (s1 & 0x8000)
    res = x - y
    if clamp:
        res = max(-32768, min(32767, res))
    return res & MASK16


class TestVSubI16Reference(unittest.TestCase):
    def test_ref_sub_i16(self):
        self.assertEqual(ref_v_sub_i16(500, 200), 300)

    def test_ref_sub_i16_negative(self):
        self.assertEqual(ref_v_sub_i16(100, 200), 0xFF9C)  # -100

    def test_ref_sub_i16_clamp(self):
        self.assertEqual(ref_v_sub_i16(0x8000, 10, clamp=True), 0x8000)  # -32768 - 10 clamped
        self.assertEqual(ref_v_sub_i16(0x7FFF, 0xFFFF, clamp=True), 0x7FFF)  # 32767 - (-1) clamped

@ValuInterpreterState.register("v_sub_i16")
def _step_sub_i16(state: ValuInterpreterState, inst: ValuInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.vgpr[int(inst.s0.lstrip("v"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.vgpr[int(inst.s1.lstrip("v"))]
    res = ref_v_sub_i16(s0_val, s1_val, clamp=inst.clamp)
    state.vgpr[inst.dst] = res
    return res


class TestVSubI16Interpreter(unittest.TestCase):
    def test_interpreter_step_sub_i16(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_sub_i16", dst=0, s0=1000, s1=400)
        res = state.step(inst)
        self.assertEqual(res, 600)
        self.assertEqual(state.vgpr[0], 600)

    def test_interpreter_step_sub_i16_clamped(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_sub_i16", dst=1, s0=0x8000, s1=50, clamp=True)
        res = state.step(inst)
        self.assertEqual(res, 0x8000)
        self.assertEqual(state.vgpr[1], 0x8000)

class TestVSubI16EdgeCases(unittest.TestCase):
    def test_self_subtraction(self):
        self.assertEqual(ref_v_sub_i16(0x1234, 0x1234), 0)
        self.assertEqual(ref_v_sub_i16(0x8000, 0x8000), 0)

    def test_underflow_wrap(self):
        self.assertEqual(ref_v_sub_i16(0x8000, 1), 0x7FFF)  # -32768 - 1 = 32767 (wrap)

class TestVSubI16Translator(unittest.TestCase):
    def test_translate_v_sub_i16(self):
        lines = [
            "v_mov_b32 v1, 500",
            "v_mov_b32 v2, 200",
            "v_sub_i16 v0, v1, v2",
            "v_mov_b32 v3, 0x8000",
            "v_mov_b32 v4, 10",
            "v_sub_i16 v1, v3, v4 clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_i16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [300, 0x8000, 300, 0x8000])

def ref_v_sub_u16(s0: int, s1: int, clamp: bool = False) -> int:
    x = s0 & MASK16
    y = s1 & MASK16
    res = x - y
    if clamp:
        res = max(0, res)
    return res & MASK16


class TestVSubU16Reference(unittest.TestCase):
    def test_ref_sub_u16(self):
        self.assertEqual(ref_v_sub_u16(500, 200), 300)

    def test_ref_sub_u16_borrow(self):
        self.assertEqual(ref_v_sub_u16(100, 200), 0xFF9C)  # wrapped borrow

    def test_ref_sub_u16_clamp(self):
        self.assertEqual(ref_v_sub_u16(100, 200, clamp=True), 0)

@ValuInterpreterState.register("v_sub_u16")
def _step_sub_u16(state: ValuInterpreterState, inst: ValuInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.vgpr[int(inst.s0.lstrip("v"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.vgpr[int(inst.s1.lstrip("v"))]
    res = ref_v_sub_u16(s0_val, s1_val, clamp=inst.clamp)
    state.vgpr[inst.dst] = res
    return res


class TestVSubU16Interpreter(unittest.TestCase):
    def test_interpreter_step_sub_u16(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_sub_u16", dst=0, s0=700, s1=200)
        res = state.step(inst)
        self.assertEqual(res, 500)
        self.assertEqual(state.vgpr[0], 500)

    def test_interpreter_step_sub_u16_clamped(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_sub_u16", dst=1, s0=50, s1=100, clamp=True)
        res = state.step(inst)
        self.assertEqual(res, 0)
        self.assertEqual(state.vgpr[1], 0)

class TestVSubU16EdgeCases(unittest.TestCase):
    def test_extremum_bounds(self):
        self.assertEqual(ref_v_sub_u16(0, 0xFFFF), 1)  # (0 - 65535) & 0xFFFF = 1
        self.assertEqual(ref_v_sub_u16(0, 0xFFFF, clamp=True), 0)

    def test_simd_divergent_lanes(self):
        lanes_a = [1000 + i * 10 for i in range(64)]
        lanes_b = [i * 10 for i in range(64)]
        res = [ref_v_sub_u16(a, b) for a, b in zip(lanes_a, lanes_b)]
        self.assertTrue(all(r == 1000 for r in res))

class TestVSubU16Translator(unittest.TestCase):
    def test_translate_v_sub_u16(self):
        lines = [
            "v_mov_b32 v1, 500",
            "v_mov_b32 v2, 200",
            "v_sub_u16 v0, v1, v2",
            "v_mov_b32 v3, 100",
            "v_mov_b32 v4, 200",
            "v_sub_u16 v1, v3, v4 clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_u16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = mrt0_words(spv)
        self.assertEqual(words, [300, 0, 300, 0])

def ref_v_add_nc_i16(s0: int, s1: int, clamp: bool = False) -> int:
    return ref_v_add_i16(s0, s1, clamp=clamp)


class TestVAddNcI16Reference(unittest.TestCase):
    def test_ref_add_nc_i16_basic(self):
        self.assertEqual(ref_v_add_nc_i16(150, 250), 400)

    def test_ref_add_nc_i16_clamped(self):
        self.assertEqual(ref_v_add_nc_i16(30000, 5000, clamp=True), 0x7FFF)

@ValuInterpreterState.register("v_add_nc_i16")
def _step_add_nc_i16(state: ValuInterpreterState, inst: ValuInstruction) -> int:
    s0_val = inst.s0 if not isinstance(inst.s0, str) else state.vgpr[int(inst.s0.lstrip("v"))]
    s1_val = inst.s1 if not isinstance(inst.s1, str) else state.vgpr[int(inst.s1.lstrip("v"))]
    res = ref_v_add_nc_i16(s0_val, s1_val, clamp=inst.clamp)
    state.vgpr[inst.dst] = res
    return res


class TestVAddNcI16Interpreter(unittest.TestCase):
    def test_interpreter_step_add_nc_i16(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_nc_i16", dst=0, s0=400, s1=300)
        res = state.step(inst)
        self.assertEqual(res, 700)
        self.assertEqual(state.vgpr[0], 700)

    def test_interpreter_step_add_nc_i16_clamped(self):
        state = ValuInterpreterState()
        inst = ValuInstruction("v_add_nc_i16", dst=1, s0=0x8000, s1=0xFFFF, clamp=True)
        res = state.step(inst)
        self.assertEqual(res, 0x8000)
        self.assertEqual(state.vgpr[1], 0x8000)

class TestVAddNcI16EdgeCases(unittest.TestCase):
    def test_identity_and_inverses(self):
        self.assertEqual(ref_v_add_nc_i16(0x1234, 0), 0x1234)
        self.assertEqual(ref_v_add_nc_i16(0x1234, (-0x1234) & MASK16), 0)

    def test_wrap_boundary(self):
        self.assertEqual(ref_v_add_nc_i16(0x7FFF, 1), 0x8000)


if __name__ == "__main__":
    unittest.main()
