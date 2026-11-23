"""Native subgroup writeback gates must read EXEC through a ballot after the shuffle.

NVIDIA's Linux 580.65.06 driver (RTX 5080) returns the shuffle from select(active, shuffle, old) where active is false
when active was computed before the shuffle (docs/design.md, "Write gate of a subgroup result"). This checks the
emitted SPIR-V order that avoids it, for one v_permlane16_b32 and one ds_swizzle_b32 write."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from test_valu_permlane16 import translate_cs  # noqa: E402


class SubgroupWritebackGateTests(unittest.TestCase):
    def assert_post_shuffle_gate(self, spv, swizzle):
        rows = [line.split() for line in spv.splitlines()]
        defs = {row[0]: (number, row[2:]) for number, row in enumerate(rows)
                if len(row) >= 3 and row[1] == "="}

        def find(value, opcode):
            self.assertIn(value, defs)
            order, args = defs[value]
            self.assertEqual(args[0], opcode, f"{value} is defined by {' '.join(args)}")
            return order, args[1:]

        shuffles = [(name, number) for name, (number, args) in defs.items()
                    if args[0] == "OpGroupNonUniformShuffle"]
        self.assertEqual(len(shuffles), 1)
        shuffle, shuffle_order = shuffles[0]

        # Follow the stored value of v2, ignoring swizzle's separate pre-shuffle source-presence ballot.
        stores = [(number, row[2]) for number, row in enumerate(rows)
                  if number > shuffle_order and len(row) == 3 and row[0] == "OpStore"
                  and defs.get(row[1], (None, []))[1] ==
                  ["OpAccessChain", "%_ptr_Function_uint", "%vgpr", "%c2"]]
        self.assertEqual(len(stores), 1)
        store_order, stored = stores[0]
        if swizzle:
            _, (_, write, _) = find(stored, "OpBitwiseOr")
            _, (_, selected_value, mask) = find(write, "OpBitwiseAnd")
            _, (_, fetched, _) = find(selected_value, "OpBitwiseAnd")
            select_order, (_, gate, _, _) = find(mask, "OpSelect")
        else:
            select_order, (_, gate, fetched, _) = find(stored, "OpSelect")
        self.assertEqual(fetched, shuffle)

        extract_order, (_, _, ballot, lane) = find(gate, "OpGroupNonUniformBallotBitExtract")
        _, lane_source = find(lane, "OpLoad")
        self.assertEqual(lane_source, ["%uint", "%sg_lane"])
        ballot_order, (_, _, active) = find(ballot, "OpGroupNonUniformBallot")
        _, active_source = find(active, "OpLoad")
        self.assertEqual(active_source, ["%bool", "%active"])
        self.assertTrue(shuffle_order < ballot_order < extract_order < select_order < store_order,
                        "v2 must use an EXEC ballot taken after its shuffle")

    def test_native_writeback_uses_a_post_shuffle_exec_ballot(self):
        cases = (
            ("permlane", "v_permlane16_b32 v2, v1, 0, 0"),
            ("swizzle", "ds_swizzle_b32 v2, v1 offset:swizzle(QUAD_PERM, 1, 0, 3, 2)"),
        )
        for name, instruction in cases:
            with self.subTest(op=name):
                spv, _ = translate_cs(f"cs_writeback_gate_{name}", [
                    "v_mov_b32_e32 v1, v0",
                    "v_mov_b32_e32 v2, 99",
                    "s_mov_b32 exec_lo, 0x0000ffff",
                    instruction,
                ], 32, emulate=False)
                self.assert_post_shuffle_gate(spv, swizzle=name == "swizzle")


if __name__ == "__main__":
    unittest.main()
