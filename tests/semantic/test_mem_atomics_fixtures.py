"""runtime/test/shaders/cs_w32_buf_atomics_{ops,rtn}.asm, cs_w32_glob_atomics_{ops,rtn}.asm and cs_w32_atomics_x2_{ops,rtn}.asm:
the 32-bit buffer_atomic_* and global_atomic_* ops and the 64-bit global_atomic_{and,or,xor,swap}_x2 / buffer_atomic_xor_x2.

The GPU words (check.py CS_DUMPS, run_gpu_tests.py) come from 32 lanes of one wave32 group that run each op on one dword
or element. spirv_mini_interp rejects OpControlBarrier, so the CPU run is one invocation (lane 0) on the fixture without
its s_barrier lines: lane 0 alone contributes its own data to each contended word, and the plain-Python model below
(the ISA pseudocode: tmp = M; M = op(tmp, DATA); glc returns tmp) gives the words for either lane count.
"""
import os
import re
import sys
import unittest

import spirv_mini_interp as si
import support

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")
M = 0xFFFFFFFF

GPU = {
    "cs_w32_buf_atomics_ops": [0x0, 0x15, 0x80000000, 0x7FFFFFF0, 0x7FFFFFFF, 0x8000000F, 0xF0000000, 0xFFFFFFFF],
    "cs_w32_buf_atomics_rtn": [0xF0F0F0F0, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x55555554, 0xFFFFFFFF],
    "cs_w32_glob_atomics_ops": [0x0, 0x15, 0x80000000, 0x7FFFFFF0, 0x7FFFFFFF, 0x8000000F, 0xF0000000, 0xFFFFFFFF],
    "cs_w32_glob_atomics_rtn": [0xF0F0F0F0, 0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x55555554, 0xFFFFFFFF],
    "cs_w32_atomics_x2_ops": [0xFFFFFFFF, 0x55555555, 0xF0000000, 0xAAAAAAAA, 0xF0F0F0F0, 0xA5A5A5A5, 0xF0F0F0F0, 0xA5A5A5A5],
    "cs_w32_atomics_x2_rtn": [0x11111111, 0x22222222, 0x33333333, 0x44444444, 0xF000F000, 0x00000FF0, 0x88888888, 0x77777777],
}


def s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


OPS = {
    "add": lambda m, d: (m + d) & M, "sub": lambda m, d: (m - d) & M,
    "smin": lambda m, d: m if s32(m) <= s32(d) else d, "umin": min,
    "smax": lambda m, d: m if s32(m) >= s32(d) else d, "umax": max,
    "and": lambda m, d: m & d, "or": lambda m, d: m | d, "xor": lambda m, d: m ^ d,
}


def lane_data(lane):
    b = 1 << lane
    return {"d": (0x7FFFFFF0 + lane) & M, "b": b, "n": (~b | 0xF0000000) & M, "h": b & 0x55555555,
            "nh": ~(b & 0x55555555) & M}


def run(op, seed, key, lanes):
    m = seed
    for lane in range(lanes):
        m = OPS[op](m, lane_data(lane)[key])
    return m


def model_ops(lanes):
    spec = (("add", 0x10, "d"), ("sub", 0x5, "d"), ("smin", 0x7FFFFFFF, "d"), ("umin", M, "d"),
            ("smax", 0x80000000, "d"), ("umax", 0, "d"), ("and", M, "n"), ("or", 0, "b"))
    return [run(op, seed, key, lanes) for op, seed, key in spec]


def model_rtn(lanes):
    cmp_hit = 0x44444444 if True else None          # DATA when the compare matches (seed 0x33333333 == CMP)
    return [run("xor", 0x0F0F0F0F, "b", lanes), 0x11111111, 0x22222222, 0x33333333, cmp_hit, 0x55555555,
            0x55555555 ^ 1, 0 ^ 0x80000000 ^ OPS["sub"](0x80000000, 1)]


def model_x2_ops(lanes):
    return [run("or", 0, "b", lanes), run("or", 0, "h", lanes), run("and", M, "n", lanes), run("and", M, "nh", lanes),
            run("xor", 0x0F0F0F0F, "b", lanes), run("xor", 0xF0F0F0F0, "h", lanes),
            run("xor", 0x0F0F0F0F, "b", lanes), run("xor", 0xF0F0F0F0, "h", lanes)]


def model_x2_rtn():
    sw = (0x33333333, 0x44444444)
    a_lo, a_hi = 0xFF00FF00 & 0x0F0F0F0F, 0x0FF00FF0 & 0xFFFF0000
    x_lo, x_hi = 0x12345678 ^ 0xFFFF0000, 0x9ABCDEF0 ^ 0x0000FFFF
    return [0x11111111, 0x22222222, sw[0], sw[1], 0xFF00FF00 ^ a_lo, 0x0FF00FF0 ^ a_hi,
            0x12345678 ^ 0x9ABCDEF0, x_lo ^ x_hi]


MODEL = {
    "cs_w32_buf_atomics_ops": model_ops, "cs_w32_glob_atomics_ops": model_ops,
    "cs_w32_buf_atomics_rtn": model_rtn, "cs_w32_glob_atomics_rtn": model_rtn,
    "cs_w32_atomics_x2_ops": model_x2_ops, "cs_w32_atomics_x2_rtn": lambda lanes: model_x2_rtn(),
}


class MemAtomicsFixtures(unittest.TestCase):
    def test_models_give_the_gpu_words(self):
        for name, want in GPU.items():
            with self.subTest(shader=name):
                self.assertEqual(MODEL[name](32), want)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            for name, want in GPU.items():
                m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % name, text)
                with self.subTest(file="/".join(rel), shader=name):
                    self.assertIsNotNone(m)
                    self.assertEqual([int(v, 0) for v in m.group(1).split(",")], want)

    def test_translated_shaders_lane0(self):
        sys.path.insert(0, SHADERS)
        from compute_fixture import compute_header
        for name in GPU:
            header = compute_header(os.path.join(SHADERS, name + ".asm"))
            with open(os.path.join(support.ROOT, "tests", "semantic", "fixtures", "disasm", name + ".s"),
                      encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines()[1:] if ln.strip() and ln.strip() != "s_barrier"]
            with self.subTest(shader=name):
                spv = support.translate_lines(name, lines, fssm=bytes(header))
                memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
                si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                                 push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
                want = MODEL[name](1)
                self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])


if __name__ == "__main__":
    unittest.main()
