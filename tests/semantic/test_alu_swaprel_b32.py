"""Value tests for v_swaprel_b32 (RDNA2 ISA, VOP1 opcode 104):

  tmp  = VGPR[D.addr + M0[25:16]]
  VGPR[D.addr + M0[25:16]] = VGPR[S0.addr + M0[9:0]]
  VGPR[S0.addr + M0[9:0]]  = tmp

Both operands are destinations. Both registers are read before either is written, so a swap with itself
leaves the register unchanged. An out-of-range destination suppresses the entire swap (the translator's
%vgpr array has 256 entries; the validity bit guards the writes).

Each case is a synthetic hand-written pixel-shader listing: v_mov_b32 loads a distinct sentinel into
v1..v30, s_mov_b32 sets M0, one v_swaprel_b32 runs, and four registers go out through exp mrt0. The
exported registers include the written one, the one read and a neighbour, so the tests show which
registers were read and written.

The expected words are hand-computed constants; a plain-Python model of the register file (no translator)
is checked against the same constants, so a typo in either is caught. Standard library only. No GPU,
no LLVM, no game files.
"""
import gc
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
NUM_VGPRS = 256         # size of the translator's %vgpr array
OP = "v_swaprel_b32"


def sentinel(i):
    """The value v_mov_b32 loads into v<i> before the swap."""
    return 0xC0DE0000 | (i * 0x111)


S = {i: sentinel(i) for i in range(1, 31)}
V0_VALUE = 0x5EED0000           # loaded into v0 by the case that reads it
JUNK = 0xFC00FC00               # M0 bits 31:26 and 15:10, which v_swaprel_b32 ignores


def m0_of(dst_off, src_off, extra=0):
    return ((dst_off << 16) | src_off | extra) & MASK32


def model(dst, src, m0, init):
    """Plain-Python register file: apply `v_swaprel_b32 dst, src` with M0 = m0 to a copy of `init`."""
    vgpr = [0] * NUM_VGPRS
    for r, v in init.items():
        vgpr[r] = v & MASK32
    d_addr = dst + ((m0 >> 16) & 0x3FF)
    s_addr = src + (m0 & 0x3FF)
    if d_addr < NUM_VGPRS and s_addr < NUM_VGPRS:
        vgpr[d_addr], vgpr[s_addr] = vgpr[s_addr], vgpr[d_addr]
    return vgpr


# (dst, src, m0, load v0?, exported registers, expected words) -- expected words worked out by hand.
# After the swap, VGPR[d_addr] holds the old VGPR[s_addr] and vice versa.
CASES = [
    (1, 2, m0_of(0, 0), False, (1, 2, 3, 4), (S[2], S[1], S[3], S[4])),             # M0 = 0: plain v1 <-> v2
    (1, 1, m0_of(5, 2), False, (1, 3, 6, 7), (S[1], S[6], S[3], S[7])),             # v6 <-> v3: offsets differ
    (2, 7, m0_of(0, 1), False, (2, 7, 8, 3), (S[8], S[7], S[2], S[3])),             # source offset only: v2 <-> v8
    (2, 4, m0_of(6, 0), False, (2, 4, 8, 6), (S[2], S[8], S[4], S[6])),             # destination offset only: v8 <-> v4
    (3, 0, m0_of(4, 0), True, (3, 7, 8, 0), (S[3], V0_VALUE, S[8], S[7])),          # v7 <-> v0
    (1, 1, m0_of(3, 4, JUNK), False, (1, 4, 5, 6), (S[1], S[5], S[4], S[6])),       # ignored M0 bits set: v4 <-> v5
    (5, 7, m0_of(20, 10), False, (5, 17, 25, 26), (S[5], S[25], S[17], S[26])),     # v25 <-> v17
    (4, 4, m0_of(0, 0), False, (4, 5, 6, 7), (S[4], S[5], S[6], S[7])),             # swap with itself: unchanged
]


_TMP = None


def setUpModule():
    global _TMP
    _TMP = tempfile.TemporaryDirectory()
    os.mkdir(os.path.join(_TMP.name, "disasm"))
    with open(os.path.join(_TMP.name, "ps_swaprel.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP
    if _TMP is not None:
        gc.collect()
        _TMP.cleanup()
        _TMP = None


def translate_listing(body):
    """Write `body` (instruction lines) as ps_swaprel.s, translate it and return the SPIR-V text."""
    lines = ["; test_swaprel"] + [f"\t{line}" for line in body] + ["\ts_endpgm", ""]
    path = os.path.join(_TMP.name, "disasm", "ps_swaprel.s")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(path)
        gc.collect()
    return spv


def initial_regs(load_v0):
    regs = dict(S)
    if load_v0:
        regs[0] = V0_VALUE
    return regs


def run_case(dst, src, m0, load_v0, exported, op=OP):
    body = [f"v_mov_b32 v{r}, {v:#x}" for r, v in sorted(initial_regs(load_v0).items())]
    body.append(f"s_mov_b32 m0, {m0:#x}")
    body.append(f"{op} v{dst}, v{src}")
    body.append("exp mrt0 " + ", ".join(f"v{r}" for r in exported) + " done vm")
    return tuple(mrt0_words(translate_listing(body)))


def case_id(case):
    dst, src, m0 = case[:3]
    return f"{OP} v{dst}, v{src} (m0={m0:#x})"


class ModelMatchesHandConstants(unittest.TestCase):
    def test_model(self):
        for case in CASES:
            dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                vgpr = model(dst, src, m0, initial_regs(load_v0))
                self.assertEqual(tuple(vgpr[r] for r in exported), expected)

    def test_sentinels_distinct(self):
        values = list(S.values()) + [V0_VALUE]
        self.assertEqual(len(set(values)), len(values))


class TranslatedValues(unittest.TestCase):
    def test_v_swaprel_b32(self):
        for case in CASES:
            dst, src, m0, load_v0, exported, expected = case
            with self.subTest(case=case_id(case)):
                got = run_case(dst, src, m0, load_v0, exported)
                self.assertEqual(got, expected, f"{case_id(case)}: got {[hex(w) for w in got]}, "
                                                f"expected {[hex(w) for w in expected]}")

    def test_e32_suffix(self):
        """LLVM prints the VOP1 form as v_swaprel_b32_e32; it translates the same."""
        dst, src, m0, load_v0, exported, expected = CASES[1]
        self.assertEqual(run_case(dst, src, m0, load_v0, exported, op=OP + "_e32"), expected)


if __name__ == "__main__":
    unittest.main()
