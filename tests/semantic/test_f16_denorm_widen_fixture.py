"""cs_w32_f16_denorm_widen: f16 denormal operands of v_cvt_f32_f16, v_pk_fma_f16 and v_pk_fmac_f16, on the CPU.

Standard library only; no GPU, no LLVM, no game files.

RDNA2's f16 MODE keeps denormals, so an f16 denormal operand counts at its value, m * 2^-24. Gen.f16_in widens f16
operands exactly since 9d71037c (#3220), because NVIDIA's UnpackHalf2x16 read an SDWA-selected denormal as 0; the
three ops here still widen with UnpackHalf2x16 directly, so the GPU fixture
runtime/test/shaders/cs_w32_f16_denorm_widen.asm checks that a driver keeps those denormals (one word per case,
derivations in its header). This runs the fixture (tests/semantic/regen_fixtures.py) through spirv_mini_interp, checks
the eight words against values computed here from the IEEE half format, and checks that check.py and
scripts/run_gpu_tests.py expect the same words on the GPU.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
NAME = "cs_w32_f16_denorm_widen"

# 4 user SGPRs (the output V#), workgroup id in s4, no LDS, 32x1x1 threads (as test_misc_ext_fixture.py).
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def half(h: int) -> float:
    """Exact value of the IEEE binary16 pattern h (denormals included)."""
    sign = -1.0 if h & 0x8000 else 1.0
    e, m = (h >> 10) & 0x1F, h & 0x3FF
    return sign * (m * 2.0 ** -24 if e == 0 else (1 + m / 1024) * 2.0 ** (e - 15))


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def half_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def pk(lo: float, hi: float) -> int:
    return half_bits(lo) | half_bits(hi) << 16


EXPECTED = [
    f32_bits(half(0x0001)),                                              # v_cvt_f32_f16, VGPR
    f32_bits(half(0x83FF)),                                              # v_cvt_f32_f16, VGPR, negative
    f32_bits(half(0x0200)),                                              # v_cvt_f32_f16, literal
    f32_bits(half(0x0040)),                                              # v_cvt_f32_f16_sdwa, WORD_1
    pk(half(0x0040) * 1.0 + 0.0, half(0x8001) * 1.0 - 0.0),              # v_pk_fma_f16, denormal multiplicands
    pk(0.0 * 1.0 + half(0x0001), 0.0 * 1.0 + half(0x0200)),              # v_pk_fma_f16, denormal addends
    pk(half(0x0001) * 1.0 + half(0x0002), half(0x0001) * 1.0 + half(0x0003)),   # v_pk_fmac_f16
    f32_bits(half(0x3C00)),                                              # control: 1.0
]



def run_fixture():
    spv = support.translate_fixture(NAME, fssm=FSSM)
    mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                              "%sg_lane": i},
                           push_constants=VSHARP + [0] * 28, memory={})
    return [mem.get(4 * k) for k in range(8)]


class F16DenormWidenFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x33800000, 0xB87FC000, 0x38000000, 0x36800000, 0x80010040, 0x02000001,
                                    0x00040003, 0x3F800000])

    def test_translated_shader(self):
        for k, (got, want) in enumerate(zip(run_fixture(), EXPECTED)):
            with self.subTest(word=k):
                self.assertEqual(got, want, f"{got:#010x} != {want:#010x}")

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"' + NAME + r'"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
