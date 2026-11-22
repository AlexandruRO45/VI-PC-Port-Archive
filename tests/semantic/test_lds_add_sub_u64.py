"""Value tests for RDNA2 ds_add_u64, ds_sub_u64 and their _rtn_ forms, run through spirv_mini_interp.

RDNA2 ISA, "Data Share Instructions": M = M + DATA (or M - DATA) on the 64-bit element at ADDR = vaddr + offset (the
dwords at ADDR and ADDR + 4, low first), wrapping at 2^64; the _rtn_ forms return the old 64 bits in vdst[0:1]. The
translator does two 32-bit atomics: the low dword takes DATA.lo, and the carry or borrow it made goes into the high
dword's DATA.hi. Each case is a compute shader run as one invocation: it seeds both dwords with ds_write_b64, runs the
op, reads the element back with ds_read_b64 and stores (vdst lo, vdst hi, M lo, M hi) to a buffer. Standard library
only; no GPU, no LLVM, no game files.
"""
import os
import random
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_bitwise_b64 import run_b64, split  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402
from test_lds_atomics_values import run_listing  # noqa: E402

M64 = (1 << 64) - 1
OPS = {"add": lambda m, d: (m + d) & M64, "sub": lambda m, d: (m - d) & M64}
VALUES = [0, 1, M64, 0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000, 0x12345678_9ABCDEF0, 0x80000000_00000000,
          0x7FFFFFFF_FFFFFFFF, 0x00000001_00000000]


class LdsAddSubU64(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in OPS.items():
            for m in VALUES:
                for d in VALUES:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        got = run_b64(f"ds_{kind}_rtn_u64 v[6:7], v1, v[4:5]", m, d)
                        self.assertEqual(got, split(m) + split(fn(m, d)))

    def test_non_returning_forms_leave_vdst_alone(self):
        for kind, fn in OPS.items():
            for m, d in ((0x12345678_FFFFFFF0, 0x00000001_00000020), (0, M64)):
                with self.subTest(op=kind, m=hex(m), d=hex(d)):
                    self.assertEqual(run_b64(f"ds_{kind}_u64 v1, v[4:5]", m, d), (0xDEAD, 0xBEEF) + split(fn(m, d)))

    def test_carry_and_borrow(self):
        self.assertEqual(run_b64("ds_add_rtn_u64 v[6:7], v1, v[4:5]", 0x00000000_FFFFFFFF, 1)[2:], (0, 1))
        self.assertEqual(run_b64("ds_sub_rtn_u64 v[6:7], v1, v[4:5]", 0x00000001_00000000, 1)[2:], (0xFFFFFFFF, 0))
        # DATA.hi = ~0 with a carry: the high dword adds 0, which is DATA mod 2^64
        self.assertEqual(run_b64("ds_add_rtn_u64 v[6:7], v1, v[4:5]", 0x00000005_00000001, 0xFFFFFFFF_FFFFFFFF)[2:],
                         (0, 5))

    def test_offset(self):
        m, d = 0x11111111_FFFFFFFE, 0x00000001_00000003
        self.assertEqual(run_b64("ds_add_rtn_u64 v[6:7], v1, v[4:5]", m, d, addr=0x20, offset=0x18),
                         split(m) + split(OPS["add"](m, d)))

    def test_vdst_aliasing_data(self):
        m, d = 0xF0F0F0F0_FFFFFFFF, 0x00000002_00000001
        got = run_b64("ds_add_rtn_u64 v[4:5], v1, v[4:5]", m, d)
        self.assertEqual(got[2:], split(OPS["add"](m, d)))

    def test_seeded(self):
        rng = random.Random(0xADD64)
        for _ in range(30):
            kind, m, d = rng.choice(list(OPS)), rng.getrandbits(64), rng.getrandbits(64)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(run_b64(f"ds_{kind}_rtn_u64 v[6:7], v1, v[4:5]", m, d),
                                 split(m) + split(OPS[kind](m, d)))

    def test_bad_operands_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_listing(["v_mov_b32_e32 v1, 0", "ds_add_u64 v1, v4"])


NAME = "cs_w32_ds_add_sub_u64"
GPU = [0xF00000F0, 0x205, 0xFFFFFE15, 0xD, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x1]


def model(lanes):
    """The fixture's ISA model: `lanes` lanes add and subtract on elements 0x00 and 0x08, then lane 0 runs the _rtn_ forms."""
    add, sub = (1 << 32) | 0xFFFFFF00, (0x10 << 32) | 5
    for lane in range(lanes):
        add = OPS["add"](add, (lane << 32) | ((0x90000000 + lane * 0x01000001) & 0xFFFFFFFF))
        sub = OPS["sub"](sub, 0x10000000 + lane)
    a_old, s_old = M64, 1 << 32
    a_new, s_new = OPS["add"](a_old, 2), OPS["sub"](s_old, 1)
    fold = (a_new & 0xFFFFFFFF) ^ (a_new >> 32) ^ (s_new & 0xFFFFFFFF) ^ (s_new >> 32)
    return [add & 0xFFFFFFFF, add >> 32, sub & 0xFFFFFFFF, sub >> 32, a_old & 0xFFFFFFFF, a_old >> 32, fold,
            (s_old & 0xFFFFFFFF) ^ (s_old >> 32)]


class LdsAddSubU64Fixture(unittest.TestCase):
    def test_model_gives_the_gpu_words(self):
        self.assertEqual(model(32), GPU)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % NAME, text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], GPU)

    def test_translated_shader_lane0(self):
        shaders = os.path.join(support.ROOT, "runtime", "test", "shaders")
        sys.path.insert(0, shaders)
        from compute_fixture import compute_header
        header = compute_header(os.path.join(shaders, NAME + ".asm"))
        with open(os.path.join(support.ROOT, "tests", "semantic", "fixtures", "disasm", NAME + ".s"),
                  encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines()[1:] if ln.strip() and ln.strip() != "s_barrier"]
        spv = support.translate_lines(NAME, lines, fssm=bytes(header))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in model(1)])
        self.assertEqual(model(1)[4:], GPU[4:])
        self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")


if __name__ == "__main__":
    unittest.main()
