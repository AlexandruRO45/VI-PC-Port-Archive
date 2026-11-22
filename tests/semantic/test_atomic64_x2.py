"""64-bit buffer and global atomics (smin/umin/smax/umax/inc/dec/cmpswap _x2) as real Int64 atomics, against the words
measured natively on RDNA2 (cs_atomic64_x2.asm) and a plain-Python reference of the RDNA2 ISA rules.

The translation declares Int64Atomics (shaderBufferInt64Atomics); rdna2spirv.py --no-int64-atomics keeps the old
refusal for a device without it. Standard library only; no GPU, no LLVM, no game files.
"""
import os
import sys
import unittest

import support
import spirv_mini_interp as si
import test_global_atomics as G       # the one-invocation compute harness (translate / run)

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import ATOMIC64_X2_WORDS
from compute_fixture import compute_header

M64 = (1 << 64) - 1
TARGET, OUT = 0x2000, 0x3000


def s64(x):
    return x - (1 << 64) if x >> 63 else x


def ref(op, m, d, c=None):
    """The new 64-bit memory value (RDNA2 ISA)."""
    return {"smin": lambda: m if s64(m) <= s64(d) else d, "smax": lambda: m if s64(m) >= s64(d) else d,
            "umin": lambda: min(m, d), "umax": lambda: max(m, d),
            "inc": lambda: 0 if m >= d else m + 1, "dec": lambda: d if m == 0 or m > d else m - 1,
            "cmpswap": lambda: d if m == c else m}[op]()


def fixture_reference():
    """The words cs_atomic64_x2 leaves, recomputed from the reference (the .asm header lists the operations)."""
    words = [0xEEEEEEEE] * 64
    mem = {0x80: 0, 0x88: 0x1_FFFFFFF0, 0x90: 0, 0x98: 0x12345678_9ABCDEF0, 0xA0: 0x1_00000000, 0xA8: 0,
           0xB0: 0x1_00000000, 0xB8: 0x80000000_00000000}
    for lane in range(32):
        mem[0x80] = ref("umax", mem[0x80], lane << 32 | 3 * lane)
        mem[0x88] = ref("inc", mem[0x88], M64)
        mem[0x90] = ref("smin", mem[0x90], -lane & M64)
    olds = {}
    for addr, op, d, c in ((0x98, "cmpswap", 0xAAAABBBB_CCCCDDDD, 0x12345678_9ABCDEF0),
                           (0xA0, "cmpswap", 0x5_00000005, 0), (0xA8, "dec", 7, None),
                           (0xB0, "umin", 0xFFFFFFFF, None), (0xB8, "smax", 1, None)):
        olds[addr] = mem[addr]
        mem[addr] = ref(op, mem[addr], d, c)
    for addr, v in list(mem.items()) + [(0xC0 + 8 * i, olds[a]) for i, a in enumerate(sorted(olds))]:
        words[addr // 4], words[addr // 4 + 1] = v & 0xFFFFFFFF, v >> 32
    return words


def single(op, form, m, d, c=None, glc=True):
    """One invocation: (memory after, returned 64 bits) for a global or buffer _x2 atomic on TARGET."""
    body = [f"v_mov_b32_e32 v1, {TARGET:#x}", f"v_mov_b32_e32 v2, {d & 0xFFFFFFFF:#x}",
            f"v_mov_b32_e32 v3, {d >> 32:#x}", "v_mov_b32_e32 v8, 0x11111111", "v_mov_b32_e32 v9, 0x22222222"]
    data = "v[2:3]"
    if c is not None:
        body += [f"v_mov_b32_e32 v4, {c & 0xFFFFFFFF:#x}", f"v_mov_b32_e32 v5, {c >> 32:#x}"]
        data = "v[2:5]"
    g = " glc" if glc else ""
    if form == "global":
        body.append(f"global_atomic_{op}_x2 v[8:9], v1, {data}, s[0:1]{g}" if glc else
                    f"global_atomic_{op}_x2 v1, {data}, s[0:1]")
        ret = "v[8:9]"
    else:
        body += ["s_mov_b32 s8, 0", "s_mov_b32 s9, 0", "s_mov_b32 s10, 0x10000", "s_mov_b32 s11, 0x31016fac",
                 f"buffer_atomic_{op}_x2 {data}, v1, s[8:11], 0 offen{g}"]
        ret = "v[2:3]"
    body += ["s_waitcnt vmcnt(0)", f"v_mov_b32_e32 v6, {OUT:#x}", f"global_store_dwordx2 v6, {ret}, s[0:1]"]
    mem = G.run(body, {TARGET: m & 0xFFFFFFFF, TARGET + 4: m >> 32})
    return mem[TARGET] | mem[TARGET + 4] << 32, mem[OUT] | mem[OUT + 4] << 32


VALUES = [0, 1, 0xFFFFFFFF, 0x1_00000000, 0x7FFFFFFF_FFFFFFFF, 0x80000000_00000000, M64, 0x12345678_9ABCDEF0]


class Atomic64X2(unittest.TestCase):
    def test_reference_gives_the_hardware_words(self):
        self.assertEqual([f"{w:#010x}" for w in fixture_reference()], [f"{w:#010x}" for w in ATOMIC64_X2_WORDS])

    def test_translated_fixture(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_atomic64_x2.asm")
        spv = support.translate_fixture("cs_atomic64_x2", fssm=bytes(compute_header(asm)))
        self.assertIn("OpCapability Int64Atomics", spv)
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 32, 32,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 32, memory=memory)
        got = [memory[4 * i] for i in range(64)]
        self.assertEqual([f"{w:#010x}" for w in got], [f"{w:#010x}" for w in ATOMIC64_X2_WORDS])

    def test_single_lane_rules(self):
        for op in ("smin", "umin", "smax", "umax", "inc", "dec"):
            for form in ("global", "buffer"):
                for m in VALUES[::2]:
                    for d in VALUES[1::2]:
                        with self.subTest(op=op, form=form, m=hex(m), d=hex(d)):
                            self.assertEqual(single(op, form, m, d), (ref(op, m, d), m))

    def test_cmpswap_compares_all_64_bits(self):
        m, d = 0x12345678_9ABCDEF0, 0xAAAAAAAA_55555555
        for form in ("global", "buffer"):
            for c in (m, m ^ 1, m ^ (1 << 32)):
                with self.subTest(form=form, cmp=hex(c)):
                    self.assertEqual(single("cmpswap", form, m, d, c), (d if c == m else m, m))

    def test_without_glc_the_return_registers_are_kept(self):
        # global: vdst untouched; buffer: vdata keeps DATA (measured on gfx1036 for both)
        self.assertEqual(single("umax", "global", 5, 9, glc=False), (9, 0x22222222_11111111))
        self.assertEqual(single("umax", "buffer", 5, 9, glc=False), (9, 9))

    def test_no_int64_atomics_keeps_the_refusal(self):
        saved = G.rdna2spirv.INT64_ATOMICS
        G.rdna2spirv.INT64_ATOMICS = False
        try:
            for line in ("global_atomic_umax_x2 v1, v[2:3], s[0:1]", "buffer_atomic_inc_x2 v[2:3], v1, s[8:11], 0 offen"):
                with self.subTest(line=line), self.assertRaises(G.rdna2spirv.Unsupported):
                    G.translate([line])
        finally:
            G.rdna2spirv.INT64_ATOMICS = saved


if __name__ == "__main__":
    unittest.main()
