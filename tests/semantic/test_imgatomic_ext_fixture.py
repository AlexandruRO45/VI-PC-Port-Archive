"""CPU checks for the six-image-atomic GPU fixture (docs/design.md, the `image_atomic_{swap,inc,dec,fmin,fmax,fcmpswap}` row).

`runtime/test/shaders/cs_imgatomic_ext.asm` runs the six atomics cs_imgatomic leaves out on texels 0..7 of
vk_store_test's r32 uint image and stores each returned old value into the same texel of its r32 sint image. This
test derives those sixteen words from the RDNA2 ISA and the fixture's own operations (an independent model, not a
copy of check.py's literals), asserts that check.py states the same words, and translates the generated fixture
(tests/semantic/fixtures/disasm, made by regen_fixtures.py) to check the six lowerings it must contain.

The mini interpreter has no image pointer model, so the words are not executed here; the GPU run of
`scripts/run_gpu_tests.py --filter cs_imgatomic_ext` or `check.py` compares them with the device's image.
"""

import os
import struct
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import STORE_WORDS  # noqa: E402  (importing check.py runs nothing)
from compute_fixture import compute_header  # noqa: E402

ASM = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_imgatomic_ext.asm")


def u32(x):
    return x & 0xFFFFFFFF


def fmin_bits(a, b):
    """GLSL.std.450 NMin on the words: the non-NaN operand, and -0 below +0 (IEEE 754 minNum)."""
    fa, fb = float_from_bits(a), float_from_bits(b)
    if fa != fa:
        return b
    if fb != fb:
        return a
    if fa == fb == 0.0:
        return a if (a >> 31) else b
    return a if fa < fb else b


def fmax_bits(a, b):
    fa, fb = float_from_bits(a), float_from_bits(b)
    if fa != fa:
        return b
    if fb != fb:
        return a
    if fa == fb == 0.0:
        return a if not (a >> 31) else b
    return a if fa > fb else b


def float_from_bits(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def model():
    """The fixture's sequence on the harness fill: (uint changes, sint changes), keyed by texel."""
    uint = {i: 100 + i for i in range(64)}
    sint = {}
    a = uint

    x = a[0]
    uint[0] = 0xDEADBEEF
    sint[0] = x  # swap
    x = a[1]
    uint[1] = 0 if x >= 200 else u32(x + 1)
    sint[1] = x  # inc, DATA 200
    x = a[2]
    uint[2] = 100 if (x == 0 or x > 100) else u32(x - 1)
    sint[2] = x  # dec, DATA 100
    uint[3] = 0x40000000  # swap seeds 2.0
    x = uint[3]
    uint[3] = fmin_bits(x, 0x3F800000)
    sint[3] = x  # fmin 1.0
    uint[4] = 0x3F800000  # swap seeds 1.0
    x = uint[4]
    uint[4] = fmax_bits(x, 0x40400000)
    sint[4] = x  # fmax 3.0
    uint[5] = 0x3F800000  # swap seeds 1.0
    x = uint[5]
    uint[5] = 0x40000000 if x == 0x3F800000 else x  # fcmpswap 2.0 when the texel is 1.0
    sint[5] = x
    x = uint[6]
    uint[6] = (
        0x40800000 if x == 0x40400000 else x
    )  # fcmpswap 4.0, compare 3.0: a miss at 0x6a
    sint[6] = x
    uint[7] = 0x7FC00000  # swap seeds a NaN
    x = uint[7]
    uint[7] = fmin_bits(x, 0x40000000)
    sint[7] = x  # fmin 2.0 takes the non-NaN operand
    return uint, sint


class ImageAtomicExtFixtureTests(unittest.TestCase):
    def test_expected_words(self):
        uint, sint = model()
        want_uint = {i: uint[i] for i in range(8)}
        want_sint = dict(sint)
        changed_uint, changed_sint, changed_volume = STORE_WORDS["cs_imgatomic_ext"]
        self.assertEqual(want_uint, dict(changed_uint))
        self.assertEqual(want_sint, dict(changed_sint))
        self.assertEqual({}, dict(changed_volume))

    def test_header_states_the_same_words(self):
        uint, sint = model()
        header = open(ASM, encoding="utf-8").read()
        for i in range(8):
            with self.subTest(texel=i):
                # the "uint word c after" column is the texel's final word, the last column the returned old word
                self.assertIn(f"0x{uint[i]:08x}", header)
                self.assertIn(f"0x{sint[i]:08x}", header)

    def test_group_size(self):
        header = compute_header(ASM)
        self.assertEqual(
            (32, 1, 1),
            tuple(int.from_bytes(header[o : o + 4], "little") for o in (0x13C, 0x140, 0x144)),
        )

    def test_translated_lowerings(self):
        spv = support.translate_fixture(
            "cs_imgatomic_ext", fssm=bytes(compute_header(ASM))
        )
        # five swaps (two seeds per float texel and the reported ones), a CAS loop behind each of the five
        # read-modify-writes, and the float lowerings the row names: NMin/NMax, and fcmpswap's ordered equality on
        # the raw words (integer compares, not OpFOrdEqual)
        self.assertEqual(5, spv.count("OpAtomicExchange %uint"))
        self.assertGreaterEqual(spv.count("OpAtomicCompareExchange %uint"), 5)
        self.assertGreaterEqual(spv.count("OpAtomicLoad %uint"), 5)
        self.assertIn("OpExtInst %float %glsl NMin", spv)
        self.assertIn("OpExtInst %float %glsl NMax", spv)
        self.assertIn("OpUGreaterThan %bool", spv)
        self.assertIn("OpImageTexelPointer %_ptr_Image_uint", spv)
        self.assertNotIn("AtomicFloat", spv)


if __name__ == "__main__":
    unittest.main()
