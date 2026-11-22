"""cs_w32_dot4_carry: the GPU test of v_dot4_i32_i8, v_dot4c_i32_i8 and the carry forms, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_dot4_carry.asm runs one wave32 group (32 lanes) and dumps 456 words: eight dot-product
words and 32 per-lane v_dot4c words, then per-lane results of v_add_co_u32, v_add_co_ci_u32 (carry-in from the SGPR mask
0x5a5a5a5a, so it differs per lane), a three-stage carry ripple, v_sub_co_u32, v_subrev_co_u32, v_sub_co_ci_u32 and the
four clamp forms, and one flag word per lane holding that lane's carry/borrow bits. This runs the translated fixture through spirv_mini_interp and checks
every word against hand constants (dot products, masks), against a plain-Python model of the ISA pseudocode (the
per-lane arrays), and that check.py's table says the same, so the GPU runners expect what the CPU gives.
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
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
CIN = 0x5A5A5A5A

# Hand-derived (derivations in the .asm header)
DOT4 = [0x00010000, 0xFFFF0200, 0xFFFFFF04, 0x8000FC03, 0xFFFFC083, 0x8000FC03, 0x7FFFFE03, 0x000001F0]
FLAGS = {0: 0x818, 1: 0x878, 15: 0x068, 16: 0x101, 17: 0x767, 31: 0x767}    # lane -> flag word


def s8(x: int) -> int:
    return x - 256 if x & 0x80 else x


def dot4(a: int, b: int, c: int) -> int:
    return (sum(s8((a >> s) & 0xFF) * s8((b >> s) & 0xFF) for s in (0, 8, 16, 24)) + c) & M32


def model():
    """The words the ISA pseudocode gives, one lane at a time."""
    words = [dot4(0x80808080, 0x80808080, 0), dot4(0x7F7F7F7F, 0x80808080, 0), dot4(M32, 0x7F7F7F7F, 0x100),
             dot4(0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF), dot4(0x7F80FF01, 0x80FF7F02, 0)]
    acc = dot4(0x7F7F7F7F, 0x7F7F7F7F, 0x7FFFFFFF)
    words.append(acc)
    words.append(dot4(0x80808080, 0x7F7F7F7F, acc))
    words.append(dot4(0x80808080, M32, (-16) & M32))
    words += [dot4(lane * 0x01010101, 0x80808080, 0x100) for lane in range(32)]
    lanes = range(32)
    cin = [(CIN >> lane) & 1 for lane in lanes]
    masks = []

    def emit(sums, borrow=False, clamp=False):
        vals, bits = [], 0
        for lane, s in enumerate(sums):
            hit = s < 0 if borrow else s >= 1 << 32
            bits |= int(hit) << lane
            vals.append((0 if borrow else M32) if clamp and hit else s & M32)
        words.extend(vals)
        masks.append(bits)

    emit([0xFFFFFFF0 + lane for lane in lanes])                                   # v_add_co_u32
    emit([0xFFFFFFEE + lane + cin[lane] for lane in lanes])                       # v_add_co_ci_u32
    # ripple through vcc_lo: word 0 adds (lane & 1), the later words add the carry of the one before
    s0 = [M32 + (lane & 1) for lane in lanes]
    s1 = [M32 + (s > M32) for s in s0]
    s2 = [0x7FFFFFFF + (s > M32) for s in s1]
    for sums in (s0, s1, s2):
        emit(sums)
    emit([16 - lane for lane in lanes], borrow=True)                              # v_sub_co_u32
    emit([lane - 16 for lane in lanes], borrow=True)                              # v_subrev_co_u32
    emit([lane - 14 - cin[lane] for lane in lanes], borrow=True)                  # v_sub_co_ci_u32
    emit([0xFFFFFFF0 + lane for lane in lanes], clamp=True)
    emit([0xFFFFFFEE + lane + cin[lane] for lane in lanes], clamp=True)
    emit([16 - lane for lane in lanes], borrow=True, clamp=True)
    emit([lane - 14 - cin[lane] for lane in lanes], borrow=True, clamp=True)
    k1, k2, k3a, k3b, k3c, k4, k5, k6, c1, c2, c3, c4 = masks
    order = [k1, k2, k4, k5, k6, k3a, k3b, k3c, c1, c2, c3, c4]
    return words + [sum(((m >> lane) & 1) << k for k, m in enumerate(order)) for lane in lanes]


class Dot4CarryFixture(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        want = model()
        self.assertEqual(len(want), 456)
        self.assertEqual(want[:8], DOT4)
        for lane, flags in FLAGS.items():
            self.assertEqual(want[424 + lane], flags, lane)
        self.assertEqual(want[8:12], [0x100, 0xFFFFFF00, 0xFFFFFD00, 0xFFFFFB00])        # 256 - 512 L
        self.assertEqual(want[40:44], [0xFFFFFFF0, 0xFFFFFFF1, 0xFFFFFFF2, 0xFFFFFFF3])    # v_add_co_u32
        self.assertEqual(want[72:76], [0xFFFFFFEE, 0xFFFFFFF0, 0xFFFFFFF0, 0xFFFFFFF2])    # lane 1 has a carry-in
        self.assertEqual(want[104 + 16:104 + 20], [0xFFFFFFFF, 0, 0xFFFFFFFF, 0])          # ripple, word 0
        self.assertEqual(want[168 + 16:168 + 20], [0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000])
        self.assertEqual(want[392 + 14:392 + 18], [0, 1, 2, 2])                              # clamp, L - 14 - CIN
        self.assertEqual(check.DOT4_CARRY_WORDS, want)

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_dot4_carry", fssm=FSSM)
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                    "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        want = model()
        self.assertEqual(sorted(mem), [4 * k for k in range(len(want))])
        for k, w in enumerate(want):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{w:#010x}")


if __name__ == "__main__":
    unittest.main()
