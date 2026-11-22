"""runtime/test/shaders/cs_w32_ds_minmax.asm: ds_max_u32, ds_min_i32, ds_max_i32, ds_min_u32 and their _rtn_ forms.

The GPU words (check.py CS_DUMPS, run_gpu_tests.py) come from 32 lanes of one wave32 group that run each op on one
address with lane L's data 0x7FFFFFF0 + L. spirv_mini_interp keeps LDS per invocation and rejects OpControlBarrier, so
the CPU run is one invocation (lane 0) on the fixture without its s_barrier lines: lane 0 alone contributes
0x7FFFFFF0 to each contended word, and the _rtn_ words (lane 0 on both) are the same as on the GPU. The plain-Python
reference models the ISA pseudocode (unsigned or signed compare) for the 32 lanes and for lane 0.
"""
import os
import re
import sys
import unittest

import spirv_mini_interp as si
import support

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")
NAME = "cs_w32_ds_minmax"
GPU = [0x8000000F, 0x80000000, 0x7FFFFFFF, 0x7FFFFFF0, 0xF0000000, 0xFFFFFFFE, 0xEFFFFFFF, 0xFFFFFF01]


def s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


OPS = {   # name: (does DATA replace the old value?, signed compare)
    "max_u32": lambda old, d: d if d > old else old,
    "min_i32": lambda old, d: d if s32(d) < s32(old) else old,
    "max_i32": lambda old, d: d if s32(d) > s32(old) else old,
    "min_u32": lambda old, d: d if d < old else old,
}
CONTENDED = (("max_u32", 0), ("min_i32", 0x7FFFFFFF), ("max_i32", 0x80000000), ("min_u32", 0xFFFFFFFF))
RTN = (("max_u32", 0x70000000, 0x80000000), ("min_i32", 1, 0xFFFFFFFF),
       ("min_u32", 0x90000000, 0x7FFFFFFF), ("max_i32", 0xFFFFFF00, 1))


def reference(lanes):
    out = []
    for op, seed in CONTENDED:
        m = seed
        for lane in range(lanes):   # min/max do not depend on the order, so any order gives the same word
            m = OPS[op](m, (0x7FFFFFF0 + lane) & 0xFFFFFFFF)
        out.append(m)
    for op, seed, data in RTN:
        out.append(seed ^ OPS[op](seed, data))   # returned old value XOR stored word
    return out


class LdsMinMax(unittest.TestCase):
    def test_reference_gpu_words(self):
        self.assertEqual(reference(32), GPU)

    def test_signedness_matters(self):
        # every contended word and every _rtn_ word differs from the one the other compare would give
        for op, seed in CONTENDED:
            other = {"max_u32": "max_i32", "max_i32": "max_u32", "min_i32": "min_u32", "min_u32": "min_i32"}[op]
            a = b = seed
            for lane in range(32):
                a = OPS[op](a, 0x7FFFFFF0 + lane)
                b = OPS[other](b, 0x7FFFFFF0 + lane)
            self.assertNotEqual(a, b, op)
        for op, seed, data in RTN:
            other = {"max_u32": "max_i32", "max_i32": "max_u32", "min_i32": "min_u32", "min_u32": "min_i32"}[op]
            self.assertNotEqual(OPS[op](seed, data), OPS[other](seed, data), op)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % NAME, text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], GPU)

    def test_translated_shader_lane0(self):
        sys.path.insert(0, SHADERS)
        from compute_fixture import compute_header
        header = compute_header(os.path.join(SHADERS, NAME + ".asm"))
        with open(os.path.join(support.ROOT, "tests", "semantic", "fixtures", "disasm", NAME + ".s"),
                  encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines()[1:] if ln.strip() and ln.strip() != "s_barrier"]
        spv = support.translate_lines(NAME, lines, fssm=bytes(header))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        want = reference(1)
        self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])
        self.assertEqual(want[:4], [0x7FFFFFF0] * 4)
        self.assertEqual(want[4:], GPU[4:])
        self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")


if __name__ == "__main__":
    unittest.main()
