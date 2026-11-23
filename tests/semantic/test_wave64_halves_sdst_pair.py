"""SHADER-10: the lane-mask destination of a wave64 VALU instruction that the listing prints as one SGPR is the SGPR
pair for tools/wave64_halves.py too.

The shaders are disassembled for wave32, so the 64-bit mask destination of a wave64 `v_cmp_*_e64` or carry-out prints
as one register (`v_cmp_lt_u32_e64 s14, ...`), and the wave writes `s[14:15]` (docs/design.md, "Wave64 VALU mask
destination printed as one SGPR": the translator and tools/lane_masks.py). The tier analysis took it as a write of
s14 alone. Then s15 kept the taint and the reaching definitions of the value it held before, and a later
`s_and_saveexec_b64 ..., s[14:15]` was tier C (`tainted-exec`) in a shader that is exact.

Every program is a hand-written vertex shader (no llvm-mc, GPU or game data), written twice: with the one-register
destination, as the corpus prints it, and with the pair. It is translated and run on the strict wave interpreter as one
wave of 64 lanes and as two waves of 32 lanes, with vertex index v for global lane v:

  * the accepted shapes are tier A in the two spellings, export the same words both ways, and export the words each
    test derives;
  * the near misses stay tier C in the two spellings, and where the test says so the two runs differ.

    python -I tests/semantic/test_wave64_halves_sdst_pair.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import wave64_halves  # noqa: E402
from test_wave64_halves import meta_of, out_param, run_cut, vs  # noqa: E402

PUSH = [0, 0, 0, 0, 9, 0, 0, 0, 20]       # s4 = 9, s8 = 20
SPELLINGS = (("one", "s14"), ("pair", "s[14:15]"))
# the compare is true for vertices 21..63; the lanes of that mask add 1 to v6 (param0.x)
CMP = "v_cmp_lt_u32_e64 {d}, s8, v0"
USE = ["s_and_saveexec_b64 s[16:17], s[14:15]", "v_add_nc_u32_e32 v6, 1, v6", "s_or_b64 exec, exec, s[16:17]"]
NARROW = ["v_cmp_gt_u32_e32 vcc, 48, v0", "s_and_saveexec_b64 s[12:13], vcc"]      # vertices 0..47
RESTORE = ["s_or_b64 exec, exec, s[12:13]"]


def run_both(name, body):
    (tier, reasons, idioms), asm = meta_of(name, vs(body))
    whole, halves = out_param(run_cut(asm, 64, PUSH)), out_param(run_cut(asm, 32, PUSH))
    return tier, reasons, [w[0] for w in whole], [w[0] for w in halves]


class OneRegisterMaskDestination(unittest.TestCase):
    """Tier A, and one wave of 64 lanes and two waves of 32 lanes export the same words, in the two spellings."""

    def invariant(self, name, body, want):
        for spelling, d in SPELLINGS:
            tier, reasons, whole, halves = run_both(f"vs_sdst_{name}_{spelling}", [x.format(d=d) for x in body])
            self.assertEqual((tier, reasons), ("A", []), (name, spelling))
            self.assertEqual(whole, halves, (name, spelling))
            self.assertEqual(whole, [want(k) for k in range(64)], (name, spelling))

    def test_a_high_register_held_a_lane_count(self):
        self.invariant("old_count", ["s_bcnt1_i32_b64 s15, exec", CMP] + USE, lambda k: int(k > 20))

    def test_a_high_register_held_the_first_lane_of_a_varying_value(self):
        self.invariant("old_rfl", ["v_readfirstlane_b32 s15, v0", CMP] + USE, lambda k: int(k > 20))

    def test_a_high_register_was_set_in_a_skip_region(self):
        # s15 has two reaching definitions before the compare, one of them behind an EXEC vote; the compare writes
        # the pair, so none of them reaches the saveexec
        self.invariant("old_stale", ["s_mov_b32 s15, 0", "v_cmp_gt_u32_e32 vcc, 8, v0",
                                     "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 1", "s_mov_b32 s15, 7",
                                     "s_or_b64 exec, exec, s[12:13]", CMP] + USE, lambda k: int(k > 20))

    def test_a_carry_out(self):
        # v0 + 0xfffffff0 carries for vertices 16..63
        self.invariant("carry", ["s_bcnt1_i32_b64 s15, exec", "v_add_co_u32_e64 v9, {d}, v0, 0xfffffff0"] + USE,
                       lambda k: int(k > 15))

    def test_a_census_shapes_under_a_narrowed_exec(self):
        # the largest rows of the --census --why run on the corpus (#4686): EXEC narrowed (vertices 0..47), the high
        # register last written by a scalar load, the compare into the low register, then a 64-bit mask instruction
        # and a vote. Vertices 21..47 add 1.
        load = ["s_buffer_load_dword s15, s[0:3], 0"]
        skip = ["s_cbranch_vccz 3", "s_and_saveexec_b64 s[16:17], vcc", "v_add_nc_u32_e32 v6, 1, v6",
                "s_or_b64 exec, exec, s[16:17]"]
        self.invariant("and_exec", load + NARROW + [CMP, "s_and_b64 vcc, exec, s[14:15]"] + skip + RESTORE,
                       lambda k: int(20 < k < 48))
        # two compares ANDed: vertices 21..39
        self.invariant("and_two", load + NARROW + [CMP, "v_cmp_gt_u32_e32 vcc, 40, v0",
                                                  "s_and_b64 vcc, vcc, s[14:15]"] + skip + RESTORE,
                       lambda k: int(20 < k < 40))
        self.invariant("saveexec", load + NARROW + [CMP] + USE[:1] + ["s_cbranch_execz 1"] + USE[1:] + RESTORE,
                       lambda k: int(20 < k < 48))


class NearMiss(unittest.TestCase):
    """Tier C in the two spellings: the pair is a real reason there."""

    def test_c_compare_of_the_lane_id(self):
        # lane id > 20: vertices 21..63 in the wave, 21..31 and 53..63 in the halves
        for spelling, d in SPELLINGS:
            tier, reasons, whole, halves = run_both(f"vs_sdst_lane_id_{spelling}", [
                "v_mbcnt_lo_u32_b32 v10, -1, 0", "v_mbcnt_hi_u32_b32 v10, -1, v10",
                f"v_cmp_lt_u32_e64 {d}, s8, v10"] + USE)
            self.assertEqual(tier, "C", spelling)
            self.assertIn("tainted-exec", reasons, spelling)
            self.assertEqual(whole, [int(k > 20) for k in range(64)], spelling)
            self.assertEqual(halves, [int(k % 32 > 20) for k in range(64)], spelling)

    def test_c_compare_behind_a_vote(self):
        # the pair is all vertices; vertices 0..7 enter a skip region and write the pair again there (to no vertex:
        # 21..63 are off). The wave has the new, empty pair. The second half skipped the region and keeps the old one.
        for spelling, d in SPELLINGS:
            tier, reasons, whole, halves = run_both(f"vs_sdst_behind_vote_{spelling}", [
                f"v_cmp_gt_u32_e64 {d}, 64, v0", "v_cmp_gt_u32_e32 vcc, 8, v0", "s_and_saveexec_b64 s[12:13], vcc",
                "s_cbranch_execz 2", f"v_cmp_lt_u32_e64 {d}, s8, v0", "s_or_b64 exec, exec, s[12:13]"] + USE)
            self.assertEqual(tier, "C", spelling)
            self.assertIn("tainted-exec", reasons, spelling)
            self.assertEqual(whole, [0] * 64, spelling)
            self.assertEqual(halves, [0] * 32 + [1] * 32, spelling)

    def test_c_high_register_read_as_data(self):
        # s15 is the mask of lanes 32..63 after the compare, in the two spellings: a mask read as data
        for spelling, d in SPELLINGS:
            (tier, reasons, _), _ = meta_of(f"vs_sdst_hi_data_{spelling}",
                                            vs([f"v_cmp_lt_u32_e64 {d}, s8, v0", "v_mov_b32_e32 v6, s15"]))
            self.assertEqual((tier, reasons), ("C", ["cross-lane"]), spelling)


class Operands(unittest.TestCase):
    def regs(self, op, *args):
        dests, _, defs, _ = wave64_halves.operands({"op": op, "args": list(args), "mods": []})
        return [r for d in dests for r in wave64_halves._regs(d)], defs

    def test_which_destinations_are_pairs(self):
        s14, pair = {"k": "sgpr", "r": 14, "n": 1}, {"k": "sgpr", "r": 14, "n": 2}
        v0, v9, s8 = {"k": "vgpr", "r": 0, "n": 1}, {"k": "vgpr", "r": 9, "n": 1}, {"k": "sgpr", "r": 8, "n": 1}
        both = [("s", 14), ("s", 15)]
        self.assertEqual(self.regs("v_cmp_lt_u32_e64", s14, s8, v0), (both, []))
        self.assertEqual(self.regs("v_cmp_lt_u32_e64", pair, s8, v0), (both, []))
        self.assertEqual(self.regs("v_cmp_class_f32_e64", s14, v0, s8), (both, []))
        self.assertEqual(self.regs("v_add_co_u32_e64", v9, s14, v0, s8), ([("v", 9)] + both, []))
        self.assertEqual(self.regs("v_mad_u64_u32", {"k": "vgpr", "r": 9, "n": 2}, s14, v0, v0, v9),
                         ([("v", 9), ("v", 10)] + both, []))
        # not a VALU mask destination: one register stays one register
        self.assertEqual(self.regs("v_cmp_lt_u32_e32", s8, v0), ([], ["vcc"]))
        self.assertEqual(self.regs("v_readfirstlane_b32", s14, v0), ([("s", 14)], []))
        self.assertEqual(self.regs("s_mov_b32", s14, s8), ([("s", 14)], []))
        self.assertEqual(self.regs("s_buffer_load_dword", s14, {"k": "sgpr", "r": 0, "n": 4}, {"k": "imm", "v": 0}),
                         ([("s", 14)], []))


if __name__ == "__main__":
    unittest.main()
