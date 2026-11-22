"""cs_w32_f64inl: a float inline constant in a 64-bit integer operand, run here on the CPU.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 reads a float inline constant (0.5, 1.0, 2.0, 4.0, their negatives, 1/(2pi)) in a 64-bit integer operand as
its f64 encoding, as an f64 operand does: s_mov_b64 s[0:1], -2.0 gives 0xC0000000_00000000. Measured natively on
gfx1036 (docs/design.md, "64-bit operand literals"). runtime/test/shaders/cs_w32_f64inl.asm has lane 0 of a wave32
group store one word of each of eight ops (derivations in its header). This runs the fixture
(tests/semantic/regen_fixtures.py) through spirv_mini_interp, checks the eight words against values computed here,
and checks that check.py and scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

INV_2PI = 0x3FC45F306DC9C882      # the hardware's f64 1/(2pi); Python's 1/(2*pi) rounds one ulp higher


def f64(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def to_signed(x: int) -> int:
    return x - (1 << 64) if x >> 63 else x


EXPECTED = [
    f64(-2.0) >> 32,                                     # s_mov_b64, high word
    INV_2PI & M32,                                       # s_mov_b64 1/(2pi), low word
    ((f64(1.0) << 4) & M64) >> 32,                       # s_lshl_b64 by 4, high word
    bin(f64(-4.0)).count("1"),                           # s_bcnt1_i32_b64
    ((to_signed(f64(-0.5)) >> 4) & M64) >> 32,           # s_ashr_i64 by 4, high word
    int(0x40100000_00000000 == f64(4.0)),                # s_cmp_eq_u64 against the pair, SCC
    f64(0.5) >> 32,                                      # v_lshlrev_b64 by 0, high word
    ((0 * 0 + f64(-1.0)) & M64) >> 32,                   # v_mad_u64_u32, high word
]


class F64InlineFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0xC0000000, 0x6DC9C882, 0xFF000000, 3, 0xFBFE0000, 1, 0x3FE00000, 0xBFF00000])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_f64inl", fssm=FSSM)
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
            m = re.search(r'"cs_w32_f64inl"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
