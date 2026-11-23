"""cs_w32_global_fcmpswap: the GPU test of global_atomic_fcmpswap, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_global_fcmpswap.asm (derivations in its header) runs three single-lane
global_atomic_fcmpswap cases (-0 against +0, NaN against its own bits, one ulp off without glc), then has all 32 lanes
of the wave swap at one +0 dword with CMP -0 and DATA 1.0 + lane bits, and stores how many lanes won and lost. This
runs its fixture as one wave32 group through spirv_mini_interp with vk_compute_test's buffer (every byte 0xEE, a V#
with num_records 0xFFFFFFFF), checks the words against a float-compare reference computed here, and checks that
check.py and scripts/run_gpu_tests.py expect the same words.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
import spirv_mini_interp as si  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)      # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]                      # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE
PZ, NZ, ONE, TWO, THREE, NAN = 0, 0x80000000, 0x3F800000, 0x40000000, 0x40400000, 0x7FC00000
LANES = 32


def f32(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def fcmpswap(m, data, cmp):
    """(old, new) of a float compare-and-swap: -0 == +0, NaN matches nothing."""
    return m, (data if f32(m) == f32(cmp) else m)


def contend(order):
    """(old value per lane, final word) when the lanes swap at a +0 word in the given order, CMP -0."""
    m, old = PZ, {}
    for lane in order:
        old[lane], m = fcmpswap(m, ONE + lane, NZ)
    return old, m


def counts(order):
    old, final = contend(order)
    return [sum(old[i] == 0 for i in range(LANES)), sum(old[i] == final for i in range(LANES)),
            sum(old[i] == 0 and ONE + i == final for i in range(LANES))]


EXPECTED = [
    fcmpswap(NZ, TWO, PZ)[0], fcmpswap(NZ, TWO, PZ)[1],     # -0 matches +0: old, then memory
    fcmpswap(NAN, ONE, NAN)[0], fcmpswap(NAN, ONE, NAN)[1],  # NaN against its own bits: old, unchanged
    fcmpswap(ONE, THREE, 0x3F800001)[1],                    # one ulp off, no glc: no match
    *counts(range(LANES)),
]


class GlobalFcmpswapFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x80000000, 0x40000000, 0x7FC00000, 0x7FC00000, 0x3F800000, 1, 31, 1])

    def test_counts_do_not_depend_on_which_lane_wins(self):
        for order in (range(LANES), range(LANES - 1, -1, -1), [*range(7, LANES), *range(7)]):
            with self.subTest(first=list(order)[0]):
                self.assertEqual(counts(order), EXPECTED[5:])

    def test_a_bit_compare_gives_other_words(self):
        # OpAtomicCompareExchange alone compares bits: -0 would not match +0 and NaN would match its own bits.
        self.assertNotEqual(NZ, PZ)
        self.assertEqual(EXPECTED[1], TWO)
        self.assertEqual(EXPECTED[3], NAN)

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_global_fcmpswap", fssm=FSSM)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(spv, LANES, LANES, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                         push_constants=VSHARP + [0] * 28, memory=mem)
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_global_fcmpswap"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
