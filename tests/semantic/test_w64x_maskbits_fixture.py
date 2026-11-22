"""cs_w64x_maskbits: the wave64 GPU test of scalar bit scans on 64-bit masks, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w64x_maskbits.asm runs one 64-lane group and dumps 64 words: s_bcnt0_i32_b64 (and the SCC it
writes) and s_ff0_i32_b64 of thirteen 64-bit values (v_cmp lane masks, pairs filled by s_mov_b32, most with little or
nothing in the low half) and of EXEC when only the upper lanes are enabled, and s_flbit_i32_b64 / s_flbit_i32_i64 of the
data pairs (a lane mask stays Unsupported for those, docs/design.md, #210). This runs the
translated fixture through spirv_mini_interp as one 64-wide subgroup and checks every word against hand constants (the
.asm header), against a plain-Python model of the ISA pseudocode, and that check.py's table says the same.
"""
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
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

# Hand-derived (the .asm header): the thirteen values, then per value bcnt0, ff0, the bcnt0 SCC; flbit_b64, flbit_i64 of the data pairs
VALUES = [0xFFFFFFFF00000000, 0x0000010000000000, 0xC000000000000000, 0x00000000FFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0,
          0x0000000100000000, 0x0080000000000000, 0xFFFFFFFF00000000, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
          0x000000FFFFFFFFFF, 0x0000010000000000]
HAND = [(32, 0, 1), (63, 0, 1), (62, 0, 1), (32, 32, 1), (0, -1, 0), (64, 0, 1), (63, 0, 1), (63, 0, 1), (32, 0, 1),
        (1, 63, 1), (63, 0, 1), (24, 40, 1), (63, 0, 1)]
FLBIT = {7: (8, 8), 8: (0, 32), 9: (1, 1), 10: (0, 1), 12: (23, 23)}
EXEC_CASES = [(32, 0, 1), (62, 0, 1), (63, 0, 1), (24, 40, 1), (32, 32, 1)]    # EXEC = the lanes of masks 0, 2, 6, 11, 3


def model(x):
    """bcnt0, ff0, SCC, flbit_b64 and flbit_i64 of one 64-bit value, bit by bit from the ISA pseudocode."""
    bit = [(x >> i) & 1 for i in range(64)]
    zeros = bit.count(0)
    ff0 = -1
    for i in range(64):
        if bit[i] == 0:
            ff0 = i
            break
    fb = -1
    for i in range(64):
        if bit[63 - i] == 1:
            fb = i
            break
    fi = -1
    for i in range(1, 64):
        if bit[63 - i] != bit[63]:
            fi = i
            break
    return (zeros, ff0, int(zeros != 0), fb, fi)


def words_of():
    w = [v & M32 for c in HAND for v in c]
    for k in sorted(FLBIT):
        w += [v & M32 for v in FLBIT[k]]
    for c in EXEC_CASES:
        w += [v & M32 for v in c]
    return w


class MaskBitsFixture(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        self.assertEqual([model(x)[:3] for x in VALUES], HAND)
        self.assertEqual({k: model(VALUES[k])[3:] for k in FLBIT}, FLBIT)
        self.assertEqual([model(VALUES[k])[:3] for k in (0, 2, 6, 11, 3)], EXEC_CASES)
        want = words_of()
        self.assertEqual(len(want), 64)
        self.assertEqual(check.W64X_MASKBITS_WORDS, want)

    def test_halves_are_told_apart(self):
        # a scan of the low word alone, or of the words swapped, gives another answer for these values
        for k in (0, 1, 2, 6, 7, 8, 10, 12):
            swapped = ((VALUES[k] & M32) << 32) | (VALUES[k] >> 32)
            self.assertNotEqual(model(swapped), model(VALUES[k]), k)
        self.assertEqual(HAND[3][1], 32)                                 # ff0 in the upper half when the low is all ones
        self.assertEqual(HAND[11][1], 40)

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w64x_maskbits", fssm=FSSM)
        mem = si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        want = words_of()
        self.assertEqual(sorted(mem), [4 * k for k in range(len(want))])
        for k, w in enumerate(want):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{w:#010x}")


if __name__ == "__main__":
    unittest.main()
