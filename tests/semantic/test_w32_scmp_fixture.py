"""cs_w32_scmp: the GPU test of the scalar integer compares and s_bitcmp, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_scmp.asm turns the SCC of 246 scalar compares (s_cmp_{eq,lg,gt,ge,lt,le}_{i32,u32} with
two SGPRs, with the inline constant -1 and with a literal, s_bitcmp0/1 in the _b32 and _b64 forms) into one bit each and
stores eight words. This runs the fixture through spirv_mini_interp, checks the words against the hand-derived constants
(the .asm header) and a plain-Python reference written from the ISA definitions, and checks that check.py and
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
M32 = 0xFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]

HAND = [0x8ecb2a69, 0x2c8e3b23, 0xa69c8e3b, 0x8ec8ec8e, 0x9cb2cb2c, 0xa69cb2a6, 0xa23b238e, 0x239c6e]

CMP = {"eq": lambda a, b: a == b, "lg": lambda a, b: a != b, "gt": lambda a, b: a > b,
       "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b, "le": lambda a, b: a <= b}


def signed(x):
    return x - (1 << 32) if x >> 31 else x


def reference():
    """The 246 results in the order of the .asm header, packed 32 to a word."""
    bits = []

    def block(pairs):
        for a, b in pairs:                              # eq, lg, gt, ge, lt, le of i32, then of u32
            bits.extend(int(CMP[c](signed(a), signed(b))) for c in CMP)
            bits.extend(int(CMP[c](a, b)) for c in CMP)

    block([(5, 5), (3, 7), (7, 3), (M32, 1), (1, M32), (0x80000000, 0x7FFFFFFF), (0x7FFFFFFF, 0x80000000),
           (0x80000000, 0x80000000)])
    block([(a, M32) for a in (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, M32)])
    block([(a, 0x12345678) for a in (0x12345677, 0x12345678, 0x12345679, 0xF2345678)])
    x32, x64 = 0x80000005, 0x0000000180000004
    for want in (0, 1):                                 # s_bitcmp0_b32, then s_bitcmp1_b32
        bits.extend(int(((x32 >> (i & 31)) & 1) == want) for i in (0, 1, 2, 31, 32, 33, 34))
    for want in (0, 1):                                 # s_bitcmp0_b64, then s_bitcmp1_b64
        bits.extend(int(((x64 >> (i & 63)) & 1) == want) for i in (0, 2, 31, 32, 33, 63, 64, 66))
    assert len(bits) == 246
    return [sum(bits[32 * w + b] << b for b in range(32) if 32 * w + b < len(bits)) for w in range(8)]


class ScmpFixture(unittest.TestCase):
    def test_reference_matches_hand_constants(self):
        self.assertEqual(reference(), HAND)

    def test_signed_and_unsigned_differ(self):
        # 0xffffffff vs 1 (A3, bits 36..47): less than as i32, greater than as u32, so the i32 and u32 halves differ
        a3 = (HAND[1] >> 4) & 0xFFF
        self.assertEqual(a3 & 0x3F, 0b110010)       # i32: lg, lt, le  (eq lg gt ge lt le, lowest bit first)
        self.assertEqual(a3 >> 6, 0b001110)         # u32: lg, gt, ge

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_scmp", fssm=FSSM)
        mem = si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                  "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(HAND):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{want:#010x}")

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_scmp"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], HAND)


if __name__ == "__main__":
    unittest.main()
