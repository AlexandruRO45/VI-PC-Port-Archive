"""SHADER-10: a buffer_load with idxen from an address that is the same in every lane gives a wave-uniform VGPR.

tools/wave64_halves.py (Analysis.uniform_load) puts the destination of such a load into the uniform set, as it does
for a VGPR written from scalars and constants, so a v_readfirstlane_b32 / v_readlane_b32 of it is tier B
(readfirstlane-uniform, readlane-uniform) and not tier C. The rule holds when EXEC is the launch mask, the index and
offset VGPRs are uniform, the descriptor and soffset are untainted scalars, and the shader has no store and no atomic.

Every program is a hand-written vertex shader (no llvm-mc, GPU or game data). It is translated, and then run on the
strict wave interpreter (spirv_wave_interp.py, with its `memory`) as one wave of 64 lanes and as two waves of
32 lanes, with vertex index v for global lane v:

  * the accepted shapes are tier B, export the same words both ways, and export the words each test derives;
  * the near misses stay tier C, and the two runs differ (the detector is right to refuse them). Two near misses are
    classification only: a shader with a store (what a store of another wave does is not a property of one run),
    and a load without idxen,
    for which the test shows on one 64-lane wave that a V# with ADD_TID_ENABLE reads a different word in each lane.

The memory: a structured buffer of 16 records of one dword at address 0x1000, record i = 1000 + i, and the dwords
after it (the soffset near miss reads past the records).

    python -I tests/semantic/test_wave64_uniform_load.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import spirv_wave_interp as wi  # noqa: E402
from test_wave64_halves import meta_of, out_param, vs  # noqa: E402

BASE = 0x1000
MEMORY = {BASE + 4 * i: 1000 + i for i in range(64)}
# V#: word 0 the base address, word 1 the stride in bits 16..29, word 2 num_records, word 3 the flags (OOB_SELECT 0:
# the index must be below num_records and the offset + 4 must fit the stride; no swizzle, no ADD_TID_ENABLE).
RECORDS = [BASE, 4 << 16, 16, 0]              # 16 records of 4 bytes
PAIRS = [BASE, 8 << 16, 8, 0]                 # 8 records of 8 bytes
ADD_TID = [BASE, 4 << 16, 16, 1 << 23]        # ADD_TID_ENABLE: without idxen the index is the lane number
LOAD = "buffer_load_dword v4, v2, s[0:3], 0 idxen"
READ = ["v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v6, s20, v0"]      # param0.x = the scalar + the vertex number


def run_cut(asm, wave, push, memory=MEMORY):
    """param0 of every lane of a 64-lane program run as 64/wave waves of `wave` lanes (vertex index = global lane)."""
    words = []
    for half in range(64 // wave):
        res = wi.run_wave(asm, wave=wave, push_constants=push, memory=dict(memory),    # each wave its own copy
                          builtins={"VertexIndex": lambda lane, base=half * wave: base + lane,
                                    "InstanceIndex": lambda lane: 0, "BaseVertex": lambda lane: 0,
                                    "BaseInstance": lambda lane: 0})
        words += [o and tuple(sorted(o.items())) for o in res["outputs"]]
    return out_param(words)


class UniformLoad(unittest.TestCase):
    """Tier B, and one wave of 64 lanes and two waves of 32 lanes export the same words."""

    def invariant(self, name, body, idioms, want, push=RECORDS, memory=MEMORY):
        (tier, reasons, got), asm = meta_of(name, vs(body))
        self.assertEqual((tier, reasons, got), ("B", [], idioms), name)
        whole, halves = run_cut(asm, 64, push, memory), run_cut(asm, 32, push, memory)
        self.assertEqual(whole, halves, name)
        self.assertEqual([w[:2] for w in whole], [want(k) for k in range(64)], name)

    def test_b_readfirstlane_of_a_load_at_a_constant_index(self):
        self.invariant("vs_ul_rfl", ["v_mov_b32_e32 v2, 3", LOAD] + READ, ["readfirstlane-uniform"],
                       lambda k: (1003 + k, 0))

    def test_b_readlane_at_any_lane_of_a_load_at_a_scalar_index(self):
        # s4 = 5 comes from the push constants; lane 40 does not exist in a 32-lane wave, so the translator's
        # rewrite (a readlane of a uniform VGPR reads the lane's own copy) is what makes the halves run
        self.invariant("vs_ul_rl", ["v_mov_b32_e32 v2, s4", LOAD, "v_readlane_b32 s20, v4, 40",
                                    "v_readlane_b32 s21, v4, 7", "v_add_nc_u32_e32 v6, s20, v0",
                                    "v_mov_b32_e32 v7, s21"], ["readlane-uniform"], lambda k: (1005 + k, 1005),
                       push=RECORDS + [5])

    def test_b_index_and_offset(self):
        # idxen and offen: v2 is the index, v3 the offset; record 3 of 8 bytes, byte 4: dword 7
        self.invariant("vs_ul_idx_off", ["v_mov_b32_e32 v2, 3", "v_mov_b32_e32 v3, 4",
                                         "buffer_load_dword v4, v[2:3], s[0:3], 0 idxen offen"] + READ,
                       ["readfirstlane-uniform"], lambda k: (1007 + k, 0), push=PAIRS)

    def test_b_loaded_value_is_the_index_of_a_second_load(self):
        # record 2 holds 1002; 1002 & 15 = 10 is the index of the second load
        self.invariant("vs_ul_chain", ["v_mov_b32_e32 v2, 2", LOAD, "v_and_b32_e32 v2, 15, v4",
                                       "buffer_load_dword v8, v2, s[0:3], 0 idxen", "v_readfirstlane_b32 s20, v8",
                                       "v_add_nc_u32_e32 v6, s20, v0"], ["readfirstlane-uniform"],
                       lambda k: (1010 + k, 0))

    def test_b_scalar_branch_on_the_loaded_value(self):
        # the scalar decides a scalar branch with code on the two sides; each side runs with a memory of its own
        body = ["v_mov_b32_e32 v2, 3", LOAD, "v_readfirstlane_b32 s20, v4", "s_cmp_eq_u32 s20, 0x3eb",
                "s_cbranch_scc0 2", "s_mov_b32 s21, 7", "s_branch 1", "s_mov_b32 s21, 9",
                "v_add_nc_u32_e32 v6, s21, v0"]
        self.invariant("vs_ul_branch", body, ["readfirstlane-uniform"], lambda k: (7 + k, 0))
        other = dict(MEMORY)
        other[BASE + 12] = 55
        self.invariant("vs_ul_branch", body, ["readfirstlane-uniform"], lambda k: (9 + k, 0), memory=other)


class NearMiss(unittest.TestCase):
    """Tier C, and one wave of 64 lanes and two waves of 32 lanes export different words."""

    def differs(self, name, body, reason, push=RECORDS):
        (tier, reasons, idioms), asm = meta_of(name, vs(body))
        self.assertEqual(tier, "C", f"{name}: tier {tier}, idioms {idioms}")
        self.assertIn(reason, reasons, name)
        whole, halves = run_cut(asm, 64, push), run_cut(asm, 32, push)
        self.assertNotEqual(whole, halves, name)
        return [w[0] for w in whole], [w[0] for w in halves]

    def test_c_index_differs_in_each_lane(self):
        # index = vertex >> 3: the first lane is vertex 0 (record 0) in the wave and vertex 32 (record 4) in the
        # second half
        whole, halves = self.differs("vs_ul_varying", ["v_lshrrev_b32_e32 v2, 3, v0", LOAD] + READ, "cross-lane")
        self.assertEqual(whole, [1000 + k for k in range(64)])
        self.assertEqual(halves, [1000 + k for k in range(32)] + [1004 + k for k in range(32, 64)])

    def test_c_load_under_a_narrowed_exec(self):
        # only lanes 0..31 load: the second half has no lane that did, and its first lane still holds 0
        whole, halves = self.differs("vs_ul_narrowed", [
            "v_mov_b32_e32 v2, 3", "v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_saveexec_b64 s[12:13], vcc", LOAD,
            "s_or_b64 exec, exec, s[12:13]"] + READ, "cross-lane")
        self.assertEqual(whole, [1003 + k for k in range(64)])
        self.assertEqual(halves, [1003 + k for k in range(32)] + list(range(32, 64)))

    def test_c_index_written_under_a_narrowed_exec(self):
        # lanes 0..31 get index 5, lanes 32..63 keep index 3
        whole, halves = self.differs("vs_ul_index_narrowed", [
            "v_mov_b32_e32 v2, 3", "v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_saveexec_b64 s[12:13], vcc",
            "v_mov_b32_e32 v2, 5", "s_or_b64 exec, exec, s[12:13]", LOAD] + READ, "cross-lane")
        self.assertEqual(whole, [1005 + k for k in range(64)])
        self.assertEqual(halves, [1005 + k for k in range(32)] + [1003 + k for k in range(32, 64)])

    def test_c_index_written_in_part(self):
        # the SDWA write replaces byte 0 of the index only: index = vertex << 8 | 3, in bounds for vertex 0 alone
        # (an out-of-bounds load gives 0). The wave reads record 3 through its first lane, the second half reads 0.
        whole, halves = self.differs("vs_ul_index_partial", [
            "v_lshlrev_b32_e32 v2, 8, v0",
            "v_mov_b32_sdwa v2, s4 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD", LOAD] + READ,
            "cross-lane", push=RECORDS + [3])
        self.assertEqual(whole, [1003 + k for k in range(64)])
        self.assertEqual(halves, [1003 + k for k in range(32)] + list(range(32, 64)))

    def test_c_soffset_is_a_lane_count(self):
        # soffset = the number of lanes: 64 in the wave (dword 16 + 3), 32 in a half (dword 8 + 3)
        whole, halves = self.differs("vs_ul_soffset", [
            "v_mov_b32_e32 v2, 3", "s_bcnt1_i32_b64 s5, exec", "buffer_load_dword v4, v2, s[0:3], s5 idxen"] + READ,
            "lane-count")
        self.assertEqual(whole, [1019 + k for k in range(64)])
        self.assertEqual(halves, [1011 + k for k in range(64)])

    def tier(self, name, body):
        (tier, reasons, idioms), asm = meta_of(name, vs(body))
        return tier, reasons, idioms, asm

    def test_c_shader_with_a_store(self):
        # the two halves read the word one after the other, and the store could change it between the two reads
        tier, reasons, idioms, _ = self.tier("vs_ul_store", [
            "v_mov_b32_e32 v2, 3", LOAD, "v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v6, s20, v0",
            "v_mov_b32_e32 v9, 9", "buffer_store_dword v0, v9, s[0:3], 0 idxen"])
        self.assertEqual((tier, idioms), ("C", []), reasons)
        self.assertIn("cross-lane", reasons)
        tier, reasons, idioms, _ = self.tier("vs_ul_atomic", [
            "v_mov_b32_e32 v2, 3", LOAD, "v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v6, s20, v0",
            "v_mov_b32_e32 v9, 9", "v_mov_b32_e32 v11, 1", "buffer_atomic_add v11, v9, s[0:3], 0 idxen"])
        self.assertEqual(tier, "C", reasons)
        self.assertNotIn("readfirstlane-uniform", idioms)

    def test_c_load_without_idxen(self):
        # without idxen the V# decides if the lane number is added to the index (ADD_TID_ENABLE), and the V# is a
        # run-time value: the load is not taken as uniform
        tier, reasons, idioms, _ = self.tier("vs_ul_offen", [
            "v_mov_b32_e32 v2, 12", "buffer_load_dword v4, v2, s[0:3], 0 offen"] + READ)
        self.assertEqual((tier, idioms), ("C", []), reasons)
        self.assertIn("cross-lane", reasons)
        # why: with ADD_TID_ENABLE in the V#, one 64-lane wave reads record `lane` in lane `lane` (0 past record 15)
        _, _, _, asm = self.tier("vs_ul_add_tid", ["buffer_load_dword v4, off, s[0:3], 0", "v_mov_b32_e32 v6, v4"])
        self.assertEqual([w[0] for w in run_cut(asm, 64, ADD_TID)], [1000 + k for k in range(16)] + [0] * 48)
        # with idxen the same V# gives every lane record 3: idxen ignores ADD_TID_ENABLE
        (tier, reasons, idioms), asm = meta_of("vs_ul_add_tid_idxen", vs(["v_mov_b32_e32 v2, 3", LOAD] + READ))
        self.assertEqual((tier, reasons, idioms), ("B", [], ["readfirstlane-uniform"]))
        whole, halves = run_cut(asm, 64, ADD_TID), run_cut(asm, 32, ADD_TID)
        self.assertEqual(whole, halves)
        self.assertEqual([w[0] for w in whole], [1003 + k for k in range(64)])


if __name__ == "__main__":
    unittest.main()
