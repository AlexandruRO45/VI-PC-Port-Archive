"""cs_w32_pk_int_const: packed 16-bit integer ops with a constant operand, run here on the CPU.

Standard library only; no GPU, no LLVM, no game files.

A constant operand of v_pk_add_u16, v_pk_mad_u16, v_pk_sub_i16, v_pk_min_i16, v_pk_mul_lo_u16 (and the other packed
16-bit integer ops) is its 32-bit value split into halves like a register: an integer inline constant sign-extended,
a float inline constant as its f32 bits, a literal as written. op_sel picks the half each source gives the result's
low half, op_sel_hi (default 1) the half it gives the high half. Measured natively on gfx1036 (docs/design.md,
"Packed 16-bit integer constants"). runtime/test/shaders/cs_w32_pk_int_const.asm has lane 0 of a wave32 group store
one word of each (derivations in its header). This runs the fixture (tests/semantic/regen_fixtures.py) through
spirv_mini_interp, checks the eight words against values computed here, and checks that check.py and
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

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def half(v: int, hi: int) -> int:
    return (v >> (16 * hi)) & 0xFFFF


def s16(x: int) -> int:
    return x - 0x10000 if x >> 15 else x


def pk(fn, srcs, op_sel=None, op_sel_hi=None):
    """One VOP3P op on 32-bit source words: result half 0 from the op_sel halves, half 1 from the op_sel_hi halves."""
    op_sel = op_sel or [0] * len(srcs)
    op_sel_hi = op_sel_hi or [1] * len(srcs)
    lo = fn(*(half(s, b) for s, b in zip(srcs, op_sel))) & 0xFFFF
    hi = fn(*(half(s, b) for s, b in zip(srcs, op_sel_hi))) & 0xFFFF
    return lo | hi << 16


V1, V2 = 0x00050003, 0
EXPECTED = [
    pk(lambda a, b: a + b, [V1, 1]),                                          # v_pk_add_u16 v1, 1
    pk(lambda a, b: a + b, [V1, -1 & M32]),                                   # v_pk_add_u16 v1, -1
    pk(lambda a, b: a + b, [V2, f32(1.0)]),                                   # v_pk_add_u16 v2, 1.0
    pk(lambda a, b: a + b, [V2, 0x12345678], [0, 1], [1, 0]),                 # v_pk_add_u16 v2, literal, swapped
    pk(lambda a, b, c: a * b + c, [V1, 0x00200010, f32(1.0)]),                # v_pk_mad_u16 v1, literal, 1.0
    pk(lambda a, b: a - b, [0x00050007, V1]),                                 # v_pk_sub_i16 literal, v1
    pk(lambda a, b: min(s16(a), s16(b)), [V2, -16 & M32]),                    # v_pk_min_i16 v2, -16
    pk(lambda a, b: a * b, [V1, f32(-2.0)]),                                  # v_pk_mul_lo_u16 v1, -2.0
]


class PkIntConstFixture(unittest.TestCase):
    def test_reference(self):
        self.assertEqual(EXPECTED, [0x00050004, 0x00040002, 0x3F800000, 0x56781234, 0x40200030, 4, 0xFFFFFFF0,
                                    0xC0000000])

    def test_translated_shader(self):
        spv = support.translate_fixture("cs_w32_pk_int_const", fssm=FSSM)
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
            m = re.search(r'"cs_w32_pk_int_const"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], EXPECTED)


if __name__ == "__main__":
    unittest.main()
