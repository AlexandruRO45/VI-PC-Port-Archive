"""cs_w32_cmp_class_f16: v_cmp_class_f16 / v_cmpx_class_f16, run here on the CPU.

Standard library only; no GPU, no LLVM, no game files.

The ten class bits of v_cmp_class_f32 (0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal,
8 +normal, 9 +inf) applied to the binary16 word of the source's low half: the high half is ignored, an inline float
constant is the f16 value, neg/abs act on bit 15, and an f16 denormal is a denormal. Measured natively on gfx1036
(docs/design.md, "Floating-point class compare"). runtime/test/shaders/cs_w32_cmp_class_f16.asm has lane 0 of a
wave32 group store one word per case (derivations in its header). This runs the fixture
(tests/semantic/regen_fixtures.py) through spirv_mini_interp, checks the eight words against a classifier written
here, and checks that check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def f16_class(word: int) -> int:
    """The class bit of the binary16 word in the low half of `word`."""
    h = word & 0xFFFF
    neg, exp, man = h >> 15, (h >> 10) & 31, h & 0x3FF
    if exp == 31:
        return (1 if man >> 9 else 0) if man else (2 if neg else 9)
    if exp == 0:
        return (5 if neg else 6) if man == 0 else (4 if neg else 7)
    return 3 if neg else 8


def word(value: int) -> int:
    """out[k] of the shader: the value's class bit (its compare with the other nine bits adds 0x400 if it is true)."""
    return 1 << f16_class(value)


EXPECTED = [
    word(0xABCD7C01),          # sNaN in the low half, high half ignored
    word(0xFE00),              # -qNaN
    word(0xFC00),              # -inf
    word(0x83FF),              # -denormal
    word(0x8000),              # -0
    word(0x0001),              # +denormal
    word(0x3C00 | 0x8000),     # -|1.0|
    word(0x3C00),              # the inline constant 1.0 as f16, e32 form; the cmpx steps leave it as it is
]


class CmpClassF16Fixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x1, 0x2, 0x4, 0x10, 0x20, 0x80, 0x8, 0x100])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_cmp_class_f16", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_cmp_class_f16"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
