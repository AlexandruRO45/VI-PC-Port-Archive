"""Native subgroup writeback gates must read EXEC after the shuffle, once per write, each with its own value.

NVIDIA's Linux 580.65.06 driver (RTX 5080) returns the shuffle from select(active, shuffle, old) where active is false
when active was computed before the shuffle, and merges two identical post-shuffle EXEC reads so that the second one
is computed before it again (docs/design.md, "Write gate of a subgroup result"). This checks the emitted SPIR-V order
and that no two gates of a module read the same value, for one v_permlane16_b32 / ds_swizzle_b32 write and for two
permlanes with a v_writelane_b32 between them."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from test_valu_permlane16 import translate_cs  # noqa: E402


class SubgroupWritebackGateTests(unittest.TestCase):
    def assert_post_shuffle_gate(self, spv, swizzle, source_activity=False):
        """source_activity: a v_permlane16 with FI=0, whose value is select(source lane active, shuffle, old) and whose
        source lane's EXEC bit comes through a shuffle of its own, between the data shuffle and the gate's."""
        rows = [line.split() for line in spv.splitlines()]
        defs = {row[0]: (number, row[2:]) for number, row in enumerate(rows)
                if len(row) >= 3 and row[1] == "="}

        eager = {row[1]: row[2] for row in rows if len(row) == 3 and row[0] == "OpStore" and row[1].startswith("%eager")}

        def unwrap(value):
            """The value an eager() copy (a Function variable stored and loaded back, SHADER-529) holds."""
            while value in defs and defs[value][1][0] == "OpLoad" and defs[value][1][2] in eager:
                value = eager[defs[value][1][2]]
            return value

        def find(value, opcode):
            self.assertIn(value, defs)
            order, args = defs[value]
            self.assertEqual(args[0], opcode, f"{value} is defined by {' '.join(args)}")
            return order, args[1:]

        shuffles = [(name, number) for name, (number, args) in defs.items()
                    if args[0] == "OpGroupNonUniformShuffle"]
        # In program order: the data shuffle, the FI=0 source-activity shuffle, the gate's.
        self.assertEqual(len(shuffles), 3 if source_activity else 2)
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
        if source_activity:
            _, (_, _, fetched, _) = find(unwrap(fetched), "OpSelect")
        self.assertEqual(unwrap(fetched), shuffle)

        compare_order, (_, got, salt) = find(gate, "OpIEqual")
        gate_shuffle_order, (_, _, flag, lane) = find(got, "OpGroupNonUniformShuffle")
        self.assertEqual(gate_shuffle_order, shuffles[-1][1], "the gate's shuffle comes after every other")
        _, lane_source = find(lane, "OpLoad")
        self.assertEqual(lane_source, ["%uint", "%sg_lane"])
        flag_order, (_, active_value, flag_salt, _) = find(flag, "OpSelect")
        self.assertEqual(flag_salt, salt)
        _, active_source = find(active_value, "OpLoad")
        self.assertEqual(active_source, ["%bool", "%active"])
        self.assertTrue(shuffle_order < flag_order < compare_order < select_order < store_order,
                        "v2 must use an EXEC read taken after its shuffle")

    def gate_salts(self, spv):
        """The salt of every gate: the constant an EXEC-on lane shuffles from itself."""
        defs = {}
        for line in spv.splitlines():
            row = line.split()
            if len(row) >= 3 and row[1] == "=":
                defs[row[0]] = row[2:]
        gates = [args for args in defs.values() if args[0] == "OpIEqual"
                 and defs.get(args[2], [""])[0] == "OpGroupNonUniformShuffle"]
        return [args[3] for args in gates]

    def test_native_writeback_reads_exec_after_its_shuffle(self):
        cases = (
            ("permlane", "v_permlane16_b32 v2, v1, 0, 0"),   # FI=0: the source lane's EXEC bit is shuffled too
            ("permlane_fi", "v_permlane16_b32 v2, v1, 0, 0 op_sel:[1,0]"),
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
                self.assert_post_shuffle_gate(spv, swizzle=name == "swizzle", source_activity=name == "permlane")

    def test_two_gates_in_a_module_read_different_values(self):
        # Two identical EXEC reads are merged by the NVIDIA compiler, which moves the second before the
        # v_writelane_b32 select and the shuffle (crosslane_fuzz seed 70102, cs_xl32_7010410418).
        spv, _ = translate_cs("cs_writeback_gate_two", [
            "v_mov_b32_e32 v1, v0",
            "v_mov_b32_e32 v4, 7",
            "s_mov_b32 s8, 99",
            "s_mov_b32 exec_lo, 0",
            "v_permlanex16_b32 v5, v1, 0, 0",
            "v_writelane_b32 v4, s8, 25",
            "v_permlanex16_b32 v4, v1, 0, 0",
        ], 32, emulate=False)
        salts = self.gate_salts(spv)
        self.assertEqual(len(salts), 3)   # both permlanes and the v_writelane_b32's lane test
        self.assertEqual(len(set(salts)), 3, "two gates read their lane flag through the same value")

    def assert_write_reads_lane_after_shuffle(self, spv):
        """The last write of v2 is select(c, value, old) where c (through any OpLogicalAnd) is a gate: an own-lane
        shuffle of a salted lane flag, compared with the salt, emitted after the data shuffle."""
        rows = [line.split() for line in spv.splitlines()]
        defs = {row[0]: (number, row[2:]) for number, row in enumerate(rows) if len(row) >= 3 and row[1] == "="}
        shuffles = sorted(number for number, args in defs.values() if args[0] == "OpGroupNonUniformShuffle")
        stores = [row[2] for row in rows if len(row) == 3 and row[0] == "OpStore"
                  and defs.get(row[1], (None, []))[1] == ["OpAccessChain", "%_ptr_Function_uint", "%vgpr", "%c2"]]
        self.assertTrue(stores)
        order, args = defs[stores[-1]]
        self.assertEqual(args[0], "OpSelect")
        conds = [args[2]]
        gates = []
        while conds:
            _, (op, *rest) = defs[conds.pop()]
            if op == "OpLogicalAnd":
                conds += rest[1:]
            elif op == "OpIEqual":
                gates.append(rest[1:])
        self.assertEqual(len(gates), 1, "the write is gated by one post-shuffle lane read")
        got, salt = gates[0]
        got_order, (op, _, _, flag, lane) = defs[got]
        self.assertEqual(op, "OpGroupNonUniformShuffle")
        self.assertEqual(defs[lane][1], ["OpLoad", "%uint", "%sg_lane"])
        self.assertEqual(defs[flag][1][0], "OpSelect")
        self.assertEqual(defs[flag][1][3], salt)
        self.assertLess(shuffles[0], got_order, "the gate reads the lane after the data shuffle")

    def test_writelane_of_a_readlane_reads_its_lane_test_after_the_shuffle(self):
        # NVIDIA 580.65.06 gives the shuffle to every lane of select(lane == 0, shuffle(v, 7), old):
        # crosslane_fuzz cs_xl32_203106417, a v_writelane_b32 of a v_readlane_b32 result.
        spv, _ = translate_cs("cs_writeback_gate_writelane", [
            "v_mov_b32_e32 v1, v0",
            "v_mov_b32_e32 v2, 99",
            "s_mov_b32 s28, 7",
            "v_readlane_b32 s4, v1, s28",
            "v_writelane_b32 v2, s4, 0",
        ], 32, emulate=False)
        self.assert_write_reads_lane_after_shuffle(spv)


if __name__ == "__main__":
    unittest.main()
