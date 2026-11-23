"""cs_w32_buffer_atomic_fminmax: the GPU fixture for buffer_atomic_{fmin,fmax}, atomic natural alignment, and 64-bit bounds, run on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_buffer_atomic_fminmax.asm (derivations in its header) runs:
- single-lane buffer_atomic_fmin / buffer_atomic_fmax cases (-0.0 vs +0.0 ordering, qNaN returning the non-NaN operand,
  glc and non-glc);
- 32-lane buffer_atomic_fmax contention on one dword (-1.0 vs 1.0 + lane * 2^-7 for lanes 0..31);
- 32-bit natural alignment guard ((addr & 3) != 0 at 0x52 and 0x51 dropped: memory at 0x50 untouched, active lane 0
  glc returns 0, inactive lane 1 preserves vdata);
- 64-bit buffer_atomic_swap_x2 8-byte natural alignment guard ((0x64 & 7) != 0 dropped even when 4 + 8 <= num_records)
  and all-or-nothing 8-byte partial bounds (offset 8 with num_records 12 dropped, returning (0, 0) and leaving low
  dword 0x68 untouched).

This test runs the fixture as one wave32 workgroup through spirv_mini_interp, checks the words against a Python
reference model, verifies a mutation negative control with Gen.atomic_aligned bypassed, and checks that check.py and
scripts/run_gpu_tests.py expect the same words.
"""
import math
import os
import re
import struct
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)  # 4 user SGPRs, wg id in s4, 512 bytes of LDS, 32x1x1
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # vk_compute_test's V#: base 0, num_records max, raw
FILL = 0xEEEEEEEE
PZ, NZ, ONE, NEG1, NEG2, QNAN = 0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0xC0000000, 0x7FC00000
LANES = 32


def f32(w):
    return struct.unpack("<f", struct.pack("<I", w))[0]


def is_nan32(w):
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


def nmin_u32(a, b):
    """GLSL.std.450 NMin on IEEE-754 binary32 words: NaN returns the non-NaN operand; -0.0 < +0.0."""
    if is_nan32(a):
        return b
    if is_nan32(b):
        return a
    if (a | b) == NZ:
        return NZ
    return a if f32(a) < f32(b) else b


def nmax_u32(a, b):
    """GLSL.std.450 NMax on IEEE-754 binary32 words: NaN returns the non-NaN operand; +0.0 > -0.0."""
    if is_nan32(a):
        return b
    if is_nan32(b):
        return a
    if (a | b) == NZ:
        return PZ
    return a if f32(a) > f32(b) else b


def contend_fmax(order):
    m = NEG1
    for lane in order:
        m = nmax_u32(m, ONE + (lane << 16))
    return m


def reference_words():
    # out[0]: 0x40 seeded +0.0 -> fmin(-0.0) -> fmin(qNaN) -> fmin(1.0) glc returns old
    m40 = PZ
    m40 = nmin_u32(m40, NZ)
    m40 = nmin_u32(m40, QNAN)
    out0 = m40

    # out[1], out[2]: 0x44 seeded qNaN -> fmax(-2.0) glc -> fmax(-0.0) glc (returns -2.0) -> fmax(+0.0)
    m44 = QNAN
    m44 = nmax_u32(m44, NEG2)
    out1 = m44
    m44 = nmax_u32(m44, NZ)
    m44 = nmax_u32(m44, PZ)
    out2 = m44

    # out[3]: 32-lane fmax contention on 0x48 (seeded -1.0)
    out3 = contend_fmax(range(LANES))

    # out[4], out[5]: 32-bit natural alignment guard at 0x52 and 0x51; active lane 0 gets 0, inactive lane 1 keeps v10
    m50 = ONE
    active_fmax_glc = 0
    active_add_glc = 0
    out4 = m50 | active_fmax_glc | active_add_glc
    out5 = 0x4000CAFE

    # out[6], out[7]: 64-bit swap_x2 on V# base 0x60, num_records 12:
    # offset 4 fails 8-byte alignment -> dropped, returns (0, 0), +4 stays 0x33334444 for aligned +0 swap_x2;
    # offset 8 fails 8-byte all-or-nothing bounds (8 + 8 > 12) -> dropped, returns (0, 0), +8 stays 0xAAAABBBB.
    out6 = 0x33334444 | 0 | 0
    out7 = 0xAAAABBBB | 0 | 0
    return [out0, out1, out2, out3, out4, out5, out6, out7]


EXPECTED = reference_words()


class BufferAtomicFminmaxFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(
            EXPECTED,
            [0x80000000, 0xC0000000, 0x00000000, 0x3F9F0000, 0x3F800000, 0x4000CAFE, 0x33334444, 0xAAAABBBB],
        )
        self.assertTrue(math.isnan(f32(QNAN)))

    def test_contention_order_invariant(self):
        for order in (range(LANES), range(LANES - 1, -1, -1), [*range(13, LANES), *range(13)]):
            with self.subTest(first=list(order)[0]):
                self.assertEqual(contend_fmax(order), EXPECTED[3])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_buffer_atomic_fminmax", fssm=FSSM)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(
            spv,
            LANES,
            LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        for k, want in enumerate(EXPECTED):
            with self.subTest(word=k):
                self.assertEqual(mem[4 * k], want)

    def test_bypassing_alignment_guard_corrupts_words(self):
        # Negative control 1: bypassing 4-byte alignment lets unaligned 0x52 reach PhysicalStorageBuffer OpAtomicLoad.
        orig_aligned = rdna2spirv.Gen.atomic_aligned
        with mock.patch.object(rdna2spirv.Gen, "atomic_aligned", lambda self, addr, align: "%true"):
            spv_unaligned4 = support.translate_fixture("cs_w32_buffer_atomic_fminmax", fssm=FSSM)
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_workgroup(
                spv_unaligned4,
                LANES,
                LANES,
                inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                push_constants=VSHARP + [0] * 28,
                memory={4 * k: FILL for k in range(64)},
            )
        # Negative control 2: bypassing 8-byte alignment lets the 4-byte-aligned swap_x2 at 0x64 overwrite 0x64..0x6B.
        with mock.patch.object(
            rdna2spirv.Gen,
            "atomic_aligned",
            lambda self, addr, align: "%true" if align == 8 else orig_aligned(self, addr, align),
        ):
            spv_unaligned8 = support.translate_fixture("cs_w32_buffer_atomic_fminmax", fssm=FSSM)
        mem = {4 * k: FILL for k in range(64)}
        si.run_workgroup(
            spv_unaligned8,
            LANES,
            LANES,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory=mem,
        )
        self.assertNotEqual(mem[4 * 6], EXPECTED[6])
        self.assertNotEqual(mem[4 * 7], EXPECTED[7])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_buffer_atomic_fminmax"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
