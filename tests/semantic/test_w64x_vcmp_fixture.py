"""cs_w64x_vcmp: the wave64 GPU test of compare masks, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w64x_vcmp.asm runs one 64-lane group and dumps 430 words: for 16 v_cmp compares (i32, u32, i64,
u64, f32 with NaN and infinity, class; into SGPR pairs and into VCC) and 7 64-bit SALU logic ops on those masks, one
flag word per lane holding that lane's bit of every mask, five v_cndmask selections made with masks, and the
population count and lowest lane of every mask. The inputs differ between the two halves, so a mask cut to its low word,
with its words swapped, or from a 64-bit compare of the low words alone is a different mask. This runs the translated
fixture through spirv_mini_interp as one 64-wide subgroup and checks every word against hand constants (the .asm
header), against a plain-Python model of the ISA pseudocode, and that check.py's table says the same.
"""
import math
import os
import struct
import sys
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import check  # noqa: E402  (runtime/test/shaders/check.py; importing it runs nothing)

M32 = 0xFFFFFFFF
ALL64 = (1 << 64) - 1
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

# Hand-derived (derivations in the .asm header): every mask as a 64-bit constant, then its population count and lowest lane
MASKS = [0x00000000FFFFFFFF, 0xFFFFFFFEFFFFFFFF, 0xA5AD2D29696B4B4A, 0x521290949484A4A5, 0x0000002000000000,
         0xFFFFFFFEFFFFFFFE, 0xFFFFFFE000000000, 0x00000000FFFFFFFF, 0x00000000FFFE0000, 0x00000002FFFFFFDF,
         0xFFFFFFFD00000020, 0x8000002000000020, 0x7FFFFFDFFFFFFFDF, 0xFFFFFEFFFFFFFFFF, 0x7FFFFEDFFFFFFFDF,
         0x7FFFFFDD00000002, 0xA5AD2D28696B4B4A, 0x00000020FFFFFFFF, 0xF7BFBDBDFDEFEFEF, 0xADED6F6A6B7B5B5A,
         0xADED6F6BFFFFFFFF, 0xFFFFFFFD00000000, 0x800000220001FFFD]
BCNT1 = [32, 63, 32, 22, 1, 62, 27, 32, 15, 32, 32, 3, 61, 63, 60, 30, 31, 33, 54, 41, 54, 31, 19]
FF1 = [0, 0, 1, 0, 37, 1, 37, 0, 17, 0, 5, 5, 0, 0, 0, 1, 1, 0, 0, 1, 0, 32, 0]
SELECT_MASKS = (0, 7, 15, 20, 5)


def lane_mask(pred):
    return sum(1 << lane for lane in range(64) if pred(lane))


def model():
    """The words the ISA pseudocode gives, one lane at a time (independent of check.py's table)."""
    def s32(x):
        return x - (1 << 32) if x >> 31 else x

    def s64(x):
        return x - (1 << 64) if x >> 63 else x

    def f32(bits):
        return struct.unpack("<f", struct.pack("<I", bits))[0]

    fbits = [struct.unpack("<I", struct.pack("<f", float(lane)))[0] for lane in range(64)]
    for lane in (5, 37):
        fbits[lane] = 0x7FC00000                      # quiet NaN
    for lane in (1, 33):
        fbits[lane] = 0xFF800000                      # -infinity
    fbits[63] = 0x7F800001                            # signaling NaN
    f = [f32(b) for b in fbits]
    v2 = [(lane - 32) & M32 for lane in range(64)]
    h = [(lane * 0x9E3779B1) & M32 for lane in range(64)]

    def klass(b):
        e, m, neg = (b >> 23) & 0xFF, b & 0x7FFFFF, b >> 31
        if e == 255:
            return (4 if neg else 512) if m == 0 else (2 if m >> 22 else 1)
        if e == 0:
            return (32 if neg else 64) if m == 0 else (16 if neg else 128)
        return 8 if neg else 256

    masks = [lane_mask(lambda l: s32(v2[l]) < 0), lane_mask(lambda l: v2[l] > 0),
             lane_mask(lambda l: s32(h[l]) < 0), lane_mask(lambda l: h[l] <= 0x55555555),
             lane_mask(lambda l: l == 37), lane_mask(lambda l: (l & 31) != 0),
             lane_mask(lambda l: s32(v2[l]) >= 5),
             lane_mask(lambda l: (((l >> 5) << 32) | (~l & M32)) < 0x180000000),
             lane_mask(lambda l: s64((((-(l >> 5)) & M32) << 32) | l) > 16),
             lane_mask(lambda l: f[l] < 32.0), lane_mask(lambda l: not f[l] < 32.0),
             lane_mask(lambda l: math.isnan(f[l])), lane_mask(lambda l: not math.isnan(f[l])),
             lane_mask(lambda l: not f[l] == 40.0), lane_mask(lambda l: f[l] < 40.0 or f[l] > 40.0),
             lane_mask(lambda l: klass(fbits[l]) & (0xC if l < 32 else 0x300) != 0)]
    masks += [masks[1] & masks[2], masks[0] | masks[4], masks[2] ^ masks[3], masks[1] & ~masks[3] & ALL64,
              (masks[0] | ~masks[3]) & ALL64, ~(masks[7] | masks[9]) & ALL64, ~(masks[8] ^ masks[15]) & ALL64]
    words = [sum(((m >> lane) & 1) << k for k, m in enumerate(masks)) for lane in range(64)]
    for k in SELECT_MASKS:
        words += [lane | 0x100 if (masks[k] >> lane) & 1 else lane for lane in range(64)]
    words += [bin(m).count("1") for m in masks]
    words += [((m & -m).bit_length() - 1) & M32 for m in masks]
    return words, masks


class VcmpFixture(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        words, masks = model()
        self.assertEqual(len(words), 430)
        self.assertEqual(masks, MASKS)
        self.assertEqual(words[384:407], BCNT1)
        self.assertEqual(words[407:430], FF1)
        self.assertEqual(check.W64X_VCMP_WORDS, words)

    def test_hand_flags(self):
        # the flag word of a lane has bit k set when the lane is in mask Mk
        words, _ = model()
        for lane in (0, 1, 5, 32, 37, 63):
            want = sum(((MASKS[k] >> lane) & 1) << k for k in range(23))
            self.assertEqual(words[lane], want, lane)
        # lane 37: the only lane of M4, a NaN (M11), not in M9 (NaN is never "less than"), in M6 (L >= 37)
        self.assertEqual([(words[37] >> k) & 1 for k in (4, 11, 9, 6)], [1, 1, 0, 1])
        self.assertEqual((words[36] >> 4) & 1, 0)

    def test_halves_are_told_apart(self):
        # a mask cut to its low word or with its words swapped is a different mask for every one of these
        for k, m in enumerate(MASKS):
            if k != 5:                                                  # (L & 31) != 0 is the same in both halves
                self.assertNotEqual(m, ((m & M32) << 32) | (m >> 32), k)
        self.assertTrue(all(m >> 32 for k, m in enumerate(MASKS) if k not in (0, 7, 8)))   # these three are low-half only
        self.assertEqual([MASKS[k] & M32 for k in (0, 7)], [M32, M32])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w64x_vcmp", fssm=FSSM)
        mem = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        want = model()[0]
        self.assertEqual(sorted(mem), [4 * k for k in range(len(want))])
        for k, w in enumerate(want):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{w:#010x}")


if __name__ == "__main__":
    unittest.main()
