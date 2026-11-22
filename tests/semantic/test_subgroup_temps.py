"""--subgroup-temps (tools/rdna2spirv.py, SUBGROUP_TEMPS): a dead OpCopyObject after every subgroup operation.

SPIRV-Cross (MoltenVK) forwards a value read once into the expression that reads it, and writes OpLogicalAnd /
OpLogicalOr / OpSelect as the short-circuiting && / || / ?:. A ballot moved to the right of && then runs only in the
invocations whose left side is true, and the lanes that skipped it are missing from it: the GPU test cs_dpp_ops read
100 instead of 10 in lane 1 on MoltenVK 1.4.2, where the DPP source check `in_row && ballot_bit(src)` became one
expression. The copy is a second read, so SPIRV-Cross binds the subgroup result to a temporary where it is computed.

Checked here: without the flag the text has no copy; with it, the text is the same plus exactly one copy right after
each OpGroupNonUniform* result (same type, same block), and the program still gives cs_dpp_ops's words. The input is
the GPU test runtime/test/shaders/cs_dpp_ops.asm (fixture from regen_fixtures.py), run by
spirv_mini_interp.run_workgroup as one 32-thread workgroup on one 32-wide subgroup.
"""
import re
import struct
import unittest

import support
import rdna2spirv
import spirv_mini_interp as si

# 4 user SGPRs (the output V#), workgroup id in s4, 32x1x1 threads.
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]
# cs_dpp_ops.asm: lane 0 of each row has no row_shr:1 source and keeps 100; lane l > 0 computes (l - 1) + 10.
EXPECTED = [100 if lane % 16 == 0 else lane - 1 + 10 for lane in range(32)]
COPY = re.compile(r"\n\s*%\w+_sgt = OpCopyObject [^\n]*")


def translate(temps):
    old = rdna2spirv.SUBGROUP_TEMPS
    rdna2spirv.SUBGROUP_TEMPS = temps
    try:
        return support.translate_fixture("cs_dpp_ops", fssm=FSSM)
    finally:
        rdna2spirv.SUBGROUP_TEMPS = old


def run(spv):
    memory = si.run_workgroup(
        spv, 32, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28, memory={})
    return [memory.get(4 * lane) for lane in range(32)]


class SubgroupTemps(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(rdna2spirv.SUBGROUP_TEMPS)
        spv = translate(False)
        self.assertIn("OpGroupNonUniformBallot", spv)
        self.assertNotIn("_sgt", spv)

    def test_only_copies_are_added(self):
        off, on = translate(False), translate(True)
        self.assertNotEqual(off, on)
        self.assertEqual(COPY.sub("", on), off)

    def test_each_subgroup_result_is_copied_where_it_is_computed(self):
        lines = translate(True).splitlines()
        results = 0
        for i, line in enumerate(lines):
            m = re.match(r"^(\s*)(%\w+) = OpGroupNonUniform\w+ (%\w+) ", line)
            if not m:
                continue
            results += 1
            self.assertEqual(lines[i + 1], f"{m.group(1)}{m.group(2)}_sgt = OpCopyObject {m.group(3)} {m.group(2)}")
        self.assertGreaterEqual(results, 3)     # the ballot, its bit extract and the shuffle of the DPP source
        self.assertEqual(sum("_sgt = OpCopyObject" in ln for ln in lines), results)

    def test_same_words_both_ways(self):
        self.assertEqual(run(translate(False)), EXPECTED)
        self.assertEqual(run(translate(True)), EXPECTED)


if __name__ == "__main__":
    unittest.main()
