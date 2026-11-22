"""cs_w64x_carry: the wave64 GPU test of the carry-chain ops, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w64x_carry.asm runs one 64-lane group and dumps 922 words: per-lane results of v_add_co_u32,
v_add_co_ci_u32, a three-stage carry ripple, v_sub_co_u32, v_subrev_co_u32, v_sub_co_ci_u32, v_subrev_co_ci_u32, a
VCC-only v_sub_co_ci_u32 and a three-stage VCC chain (13 ops), one flag word per lane holding that lane's bit of every
carry/borrow mask, and the population count and lowest lane of each mask. The carry-in is the 64-bit mask
0x0ff0a55b_5a5a5a5a, so it differs between the two halves. This runs the translated fixture through spirv_mini_interp
as one 64-wide subgroup and checks every word against hand constants (the .asm header), against a plain-Python model of
the ISA pseudocode, and that check.py's table says the same, so the GPU runners expect what the CPU gives.
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
CIN = 0x0FF0A55B5A5A5A5A

# Hand-derived (derivations in the .asm header): the mask of each op as a 64-bit constant
MASKS = [0xFFFFFFFF00000000, 0xFFFFFFFFFFC00000, 0x55555555AAAAAAAA, 0x55555555AAAAAAAA, 0, 0xFFFE000000000000,
         0x000000000000FFFF, 0x00000001FFFFFFFF, 0x0000FFFFFFFFFFFF, 0x000001FFFFFFFFFF, 0xFFFFFFFFFFE00000,
         0xFFFFFFFFFFE00000, 0]
BCNT1 = [32, 42, 32, 32, 0, 15, 16, 33, 48, 41, 43, 43, 0]
FF1 = [32, 22, 1, 1, M32, 49, 0, 0, 0, 0, 21, 21, M32]
FLAGS = {0: 0x3C0, 22: 0xF82, 32: 0xF8F, 33: 0xF03, 49: 0xC23}                       # lane -> flag word
# (op, lane) -> D
WORDS = {(0, 0): 0xFFFFFFE0, (0, 31): 0xFFFFFFFF, (0, 32): 0, (0, 63): 0x1F,
         (1, 0): 0xFFFFFFE9, (1, 1): 0xFFFFFFEB, (1, 22): 0, (1, 63): 0x28,
         (2, 0): M32, (2, 1): 0, (2, 32): 0, (2, 33): M32,
         (4, 0): 0x7FFFFFFF, (4, 1): 0x80000000, (4, 32): 0x80000000,
         (5, 0): 0x30, (5, 48): 0, (5, 49): M32, (5, 63): 0xFFFFFFF1,
         (6, 0): 0xFFFFFFF0, (6, 15): M32, (6, 16): 0, (6, 63): 0x2F,
         (7, 31): M32, (7, 32): M32, (7, 33): 0, (7, 63): 0x1F,
         (8, 46): M32, (8, 47): M32, (8, 48): 1, (8, 63): 0x10,
         (12, 0): 0xFFFFFFFE, (12, 22): M32}


def model():
    """The words the ISA pseudocode gives, one lane at a time, op by op."""
    lanes = range(64)
    cin = [(CIN >> lane) & 1 for lane in lanes]
    words, masks = [], []

    def emit(sums, borrow=False):
        masks.append(sum(int(s < 0 if borrow else s > M32) << lane for lane, s in enumerate(sums)))
        words.extend(s & M32 for s in sums)

    emit([0xFFFFFFE0 + lane for lane in lanes])
    emit([0xFFFFFFE9 + lane + cin[lane] for lane in lanes])
    b0 = [((lane >> 5) ^ lane) & 1 for lane in lanes]            # the lane's low bit in the low half, its complement above
    carry = b0
    for base in (M32, M32, 0x7FFFFFFF):                          # -1 + B0, -1 + 0 + c, 0x7fffffff + 0 + c
        sums = [base + c for c in carry]
        emit(sums)
        carry = [int(s > M32) for s in sums]
    emit([48 - lane for lane in lanes], borrow=True)
    emit([lane - 16 for lane in lanes], borrow=True)
    emit([lane - 32 - cin[lane] for lane in lanes], borrow=True)
    emit([lane - 47 - cin[lane] for lane in lanes], borrow=True)
    emit([lane - 40 - cin[lane] for lane in lanes], borrow=True)
    sums = [0xFFFFFFEB + lane for lane in lanes]
    emit(sums)
    carry = [int(s > M32) for s in sums]
    sums = [M32 + c for c in carry]
    emit(sums)
    carry = [int(s > M32) for s in sums]
    emit([0x7FFFFFFF + 0x7FFFFFFF + c for c in carry])
    words += [sum(((m >> lane) & 1) << k for k, m in enumerate(masks)) for lane in lanes]
    words += [bin(m).count("1") for m in masks]
    words += [((m & -m).bit_length() - 1) & M32 for m in masks]
    return words, masks


class CarryFixture(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        words, masks = model()
        self.assertEqual(len(words), 922)
        self.assertEqual(masks, MASKS)
        self.assertEqual(words[896:909], BCNT1)
        self.assertEqual(words[909:922], FF1)
        for lane, flags in FLAGS.items():
            self.assertEqual(words[832 + lane], flags, lane)
        for (op, lane), want in WORDS.items():
            self.assertEqual(words[64 * op + lane], want, (op, lane))
        self.assertEqual(check.W64X_CARRY_WORDS, words)

    def test_halves_differ(self):
        # a result cut to the low half, or with the halves swapped, would change these
        for k, m in enumerate(MASKS):
            if m and k != 3:   # op 3 only ripples op 2's carry; its halves differ as op 2's do
                self.assertNotEqual(m & M32, m >> 32, k)

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w64x_carry", fssm=FSSM)
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
