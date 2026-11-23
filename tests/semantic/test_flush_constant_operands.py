"""A constant operand of an op that flushes f32 denormals is flushed when the shader is translated (Gen.src_f_flushed).

v_mad_f32, v_mac_f32, v_mul_legacy_f32, v_mad_legacy_f32, v_mac_legacy_f32, the f32 operands of v_fma_mix_* and the
accumulator of v_dot2_f32_f16 read a subnormal operand as a zero of its sign (Gen.flush_f32, SHADER-118 and
SHADER-298). For an inline constant or a literal the bits are known already, so the translator writes the flushed
value and leaves out the run-time FAbs / compare / select. These tests check that the values are still the flushed
ones, with the constant in each operand slot, and that the constant operands no longer get a run-time flush.
Hand-written programs, standard library only, no game files.

    python -I tests/semantic/test_flush_constant_operands.py -v
"""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

SUB_MIN, NEG_SUB_MAX = 0x00000001, 0x807FFFFF   # the smallest positive subnormal, the largest negative one
HALF, TWO, THREE = 0x3F000000, 0x40000000, 0x40400000


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def ftz(b):
    return b & 0x80000000 if not b & 0x7F800000 else b


def mad_ref(x, y, z):
    """v_mad_f32 with every operand, the product and the sum flushed. The test values keep both roundings exact in
    binary64, so one rounding to binary32 each is what the hardware does."""
    m = ftz(bits(f32(ftz(x)) * f32(ftz(y))))
    return ftz(bits(f32(m) + f32(ftz(z))))


def run(lines):
    return support.mrt0_words(support.translate_lines("ps_flush_const", lines + ["exp mrt0 v0, v0, v0, v0 done vm",
                                                                             "s_endpgm"]))[0]


def flush_compares(spv):
    # Each run-time flush is one FAbs feeding a compare with FLT_MIN; nothing else in these programs takes FAbs.
    return sum(" FAbs " in ln for ln in spv.splitlines())


class ConstantOperandValues(unittest.TestCase):
    def test_v_mad_f32_constant_in_each_slot(self):
        regs = {1: TWO, 2: THREE, 3: HALF}
        for const, text in ((SUB_MIN, "1"), (NEG_SUB_MAX, "0x807fffff"), (HALF, "0.5"), (0x80000000, "0x80000000"),
                            (0x3E99999A, "0x3e99999a")):
            for slot in (1, 2, 3):
                ops = [f"v{r}" if r != slot else text for r in (1, 2, 3)]
                vals = [regs[r] if r != slot else const for r in (1, 2, 3)]
                with self.subTest(const=hex(const), slot=slot):
                    lines = [f"v_mov_b32 v{r}, {regs[r]:#x}" for r in (1, 2, 3)]
                    self.assertEqual(run(lines + [f"v_mad_f32 v0, {', '.join(ops)}"]), mad_ref(*vals))

    def test_v_mac_f32_constant_multiplicand(self):
        # The inline constant 1 as an f32 operand is the subnormal 2^-149 and reads as +0, so v0 keeps its 3.0.
        for text, const in (("1", SUB_MIN), ("0.5", HALF), ("0x807fffff", NEG_SUB_MAX)):
            with self.subTest(const=text):
                lines = [f"v_mov_b32 v2, {TWO:#x}", f"v_mov_b32 v0, {THREE:#x}", f"v_mac_f32 v0, {text}, v2"]
                self.assertEqual(run(lines), mad_ref(const, TWO, THREE))

    def test_v_fma_mix_f32_constant_f32_operand(self):
        # op_sel_hi 0: all three operands are f32. 64 is the subnormal 64 * 2^-149 (SHADER-298) and reads as +0.
        lines = [f"v_mov_b32 v1, {TWO:#x}", f"v_mov_b32 v2, {THREE:#x}", "v_fma_mix_f32 v0, 64, v1, v2"]
        self.assertEqual(run(lines), THREE)
        lines = [f"v_mov_b32 v1, {TWO:#x}", f"v_mov_b32 v2, {THREE:#x}", "v_fma_mix_f32 v0, 0.5, v1, v2"]
        self.assertEqual(run(lines), bits(4.0))

    def test_v_dot2_f32_f16_constant_accumulator(self):
        # v1 = v2 = (1.0h, 1.0h): the dot product is 2.0; the accumulator 1 is 2^-149 and reads as +0.
        lines = ["v_mov_b32 v1, 0x3c003c00", "v_mov_b32 v2, 0x3c003c00", "v_dot2_f32_f16 v0, v1, v2, 1"]
        self.assertEqual(run(lines), TWO)
        lines = ["v_mov_b32 v1, 0x3c003c00", "v_mov_b32 v2, 0x3c003c00", "v_dot2_f32_f16 v0, v1, v2, 0.5"]
        self.assertEqual(run(lines), bits(2.5))


    def test_legacy_multiply_constant_operands(self):
        # A subnormal constant reads as a zero of its sign, and the legacy rule makes 0 * inf +0 (SHADER-118).
        inf = 0x7F800000
        for text in ("1", "0x807fffff"):
            with self.subTest(const=text):
                self.assertEqual(run([f"v_mov_b32 v1, {inf:#x}", f"v_mul_legacy_f32 v0, {text}, v1"]), 0)
                lines = [f"v_mov_b32 v1, {inf:#x}", f"v_mov_b32 v2, {THREE:#x}", f"v_mad_legacy_f32 v0, {text}, v1, v2"]
                self.assertEqual(run(lines), THREE)
                lines = [f"v_mov_b32 v1, {TWO:#x}", f"v_mov_b32 v2, {THREE:#x}", f"v_mad_legacy_f32 v0, v1, v2, {text}"]
                self.assertEqual(run(lines), bits(6.0))
        self.assertEqual(run([f"v_mov_b32 v1, {THREE:#x}", "v_mul_legacy_f32 v0, 0.5, v1"]), bits(1.5))

class ConstantOperandsNeedNoRunTimeFlush(unittest.TestCase):
    def spv(self, op):
        return support.translate_lines("ps_flush_const", [f"v_mov_b32 v1, {TWO:#x}", f"v_mov_b32 v2, {THREE:#x}", op,
                                                       "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])

    def test_v_mad_f32(self):
        # Three VGPR operands: three operand flushes, then the product's and the sum's. A constant operand drops one.
        self.assertEqual(flush_compares(self.spv("v_mad_f32 v0, v1, v2, v1")), 5)
        for c in ("0.5", "1", "0x807fffff"):
            with self.subTest(const=c):
                self.assertEqual(flush_compares(self.spv(f"v_mad_f32 v0, v1, {c}, v2")), 4)
                self.assertEqual(flush_compares(self.spv(f"v_mad_f32 v0, {c}, {c}, v2")), 3)

    def test_a_source_modifier_keeps_the_run_time_flush(self):
        # -0.5 as a literal is a constant; neg on the constant 0.5 is a modifier, which src_f_flushed leaves to
        # flush_f32 (the modifier's own FAbs is not counted: neg emits an FNegate).
        self.assertEqual(flush_compares(self.spv("v_mad_f32 v0, v1, -0.5, v2")), 4)
        self.assertEqual(flush_compares(self.spv("v_mad_f32 v0, v1, |v2|, v2")), 6)

    def test_v_dot2_and_fma_mix(self):
        self.assertEqual(flush_compares(self.spv("v_dot2_f32_f16 v0, v1, v2, v1")), 1)
        self.assertEqual(flush_compares(self.spv("v_dot2_f32_f16 v0, v1, v2, 1")), 0)
        self.assertEqual(flush_compares(self.spv("v_fma_mix_f32 v0, v1, v2, v1")), 3)
        self.assertEqual(flush_compares(self.spv("v_fma_mix_f32 v0, 0.5, v2, 64")), 1)


if __name__ == "__main__":
    unittest.main()
