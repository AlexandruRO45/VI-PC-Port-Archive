"""global_atomic_inc/dec and buffer_atomic_inc/dec, run through spirv_mini_interp on the CPU.

RDNA2 ISA (unsigned): inc M = (M >= D) ? 0 : M + 1; dec M = (M == 0 || M > D) ? D : M - 1; with glc the old M comes
back. The translator has no SPIR-V atomic for these and emits an OpAtomicCompareExchange loop (Gen.atomic_cas_loop),
so the tests also run 32 lanes on one dword: the lanes run in lockstep in the interpreter, their exchanges collide and
retry, and the result must be the same as 32 inc / dec applied one after another. Standard library only.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_global_atomics import OUT, TARGET, run, si, translate  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF


def inc(m, d):
    return 0 if m >= d else (m + 1) & M32


def dec(m, d):
    return d if m == 0 or m > d else m - 1


MODEL = {"inc": inc, "dec": dec}
VALUES = [0, 1, 2, 5, 6, 7, 0x7FFFFFFF, 0x80000000, M32 - 1, M32]


def global_op(kind, m, d, glc=True):
    body = [f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {d:#x}", "v_mov_b32_e32 v5, 0xdead"]
    body.append(f"global_atomic_{kind} v5, v1, v2, s[0:1] glc" if glc else f"global_atomic_{kind} v1, v2, s[0:1]")
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v5, s[0:1]"]
    mem = run(body, {TARGET: m})
    return mem.get(OUT), mem.get(TARGET)


BUF = 0x100


def buffer_op(kind, m, d, offset=0x40, records=0x80):
    body = ["s_mov_b32 s8, 0x100", "s_mov_b32 s9, 0", f"s_mov_b32 s10, {records:#x}", "s_mov_b32 s11, 0x31016fac",
            f"v_mov_b32_e32 v1, {offset:#x}", f"v_mov_b32_e32 v2, {d:#x}",
            f"buffer_atomic_{kind} v2, v1, s[8:11], 0 offen glc", "s_waitcnt vmcnt(0)",
            f"v_mov_b32_e32 v6, {OUT:#x}", "global_store_dword v6, v2, s[0:1]"]
    mem = run(body, {BUF + offset: m})
    return mem.get(OUT), mem.get(BUF + offset)


class AtomicIncDec(unittest.TestCase):
    def test_reference(self):
        self.assertEqual([inc(m, 5) for m in (0, 4, 5, 6)], [1, 5, 0, 0])
        self.assertEqual([dec(m, 5) for m in (0, 1, 5, 6)], [5, 0, 4, 5])
        self.assertEqual(inc(M32, M32), 0)

    def test_global_values(self):
        for kind, fn in MODEL.items():
            for m in VALUES:
                for d in VALUES[::3]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        self.assertEqual(global_op(kind, m, d), (m, fn(m, d)))

    def test_global_without_glc_keeps_vdst(self):
        for kind, fn in MODEL.items():
            with self.subTest(op=kind):
                self.assertEqual(global_op(kind, 3, 5, glc=False), (0xDEAD, fn(3, 5)))

    def test_buffer_values_and_bounds(self):
        for kind, fn in MODEL.items():
            for m, d in ((0, 5), (5, 5), (6, 5), (0x80000000, M32)):
                with self.subTest(op=kind, m=hex(m), d=hex(d)):
                    self.assertEqual(buffer_op(kind, m, d), (m, fn(m, d)))
            with self.subTest(op=kind, oob=True):       # offset + 4 > num_records: memory untouched, glc gives 0
                self.assertEqual(buffer_op(kind, 3, 5, offset=0x80), (0, 3))

    def test_32_lanes_on_one_dword(self):
        for kind, fn in MODEL.items():
            for m, d in ((0, 100), (90, 100), (3, 7), (0, 0)):
                with self.subTest(op=kind, m=m, d=d):
                    spv = translate([f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {d:#x}",
                                     f"global_atomic_{kind} v1, v2, s[0:1]"])
                    mem = {TARGET: m}
                    si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0),
                                                                         "%local_id": (i, 0, 0), "%sg_lane": i},
                                     push_constants=[0] * 36, memory=mem)
                    want = m
                    for _ in range(32):
                        want = fn(want, d)
                    self.assertEqual(mem[TARGET], want)

    def test_seeded(self):
        rng = random.Random(0x1DEC)
        for _ in range(20):
            kind = rng.choice(list(MODEL))
            d = rng.choice((rng.getrandbits(4), rng.getrandbits(32)))
            m = rng.choice((rng.randrange(0, d + 2) & M32, rng.getrandbits(32)))
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(global_op(kind, m, d), (m, MODEL[kind](m, d)))

    def test_forms_still_rejected(self):
        for line in ("global_atomic_inc_x2 v[4:5], v1, v[2:3], s[0:1] glc", "global_atomic_inc v[4:5], v1, v2, s[0:1] glc",
                     "global_atomic_dec v5, v1, v2, s[0:1] glc slc"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["v_mov_b32_e32 v1, 0", line])


if __name__ == "__main__":
    unittest.main()
