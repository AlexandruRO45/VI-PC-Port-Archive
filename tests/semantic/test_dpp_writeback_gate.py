"""A DPP instruction's write is a subgroup write, so its EXEC gate must be read after the shuffle too.

DPP reads another lane's value (row_shl / row_share / quad_perm / dpp8), so a VALU write fed by one carries a
subgroup value exactly like `v_permlane16_b32` or `ds_swizzle_b32`. The Blackwell select-after-shuffle miscompile
in docs/design.md ("Write gate of a subgroup result") therefore applies to it for the same reason: the driver
returns the shuffle from `select(active, shuffle, old)` when `active` was computed before the shuffle and `old`
is itself a select, so a DPP result lands in lanes where EXEC is 0. SHADER-1035: two `crosslane_fuzz` programs
(seed 91001, `--dpp --no-maskdata`, reduced to 3-4 DPP ops) on an NVIDIA Blackwell driver.

Before this change the DPP path reached `dst()` with `after_subgroup=False`, so its write took `exec_value()`
(EXEC read before the shuffle) and the gate was a no-op there. It now goes through `gate_after_subgroup` like the
other subgroup writes. `dpp_guard` is set only while `dpp_insn` re-enters `_insn`, so ordinary VALU writes are
untouched - which is what the last test pins.

Standard library only; no GPU, no LLVM, no SPIR-V tools, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import test_valu_permlane16 as h  # noqa: E402

DPP = ["v_mbcnt_lo_u32_b32 v0, -1, 0",
       "v_mov_b32_dpp v1, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"]
PLAIN = ["v_mbcnt_lo_u32_b32 v0, -1, 0",
         "v_mov_b32_e32 v1, v0"]


def parse(spv):
    """([rows], {id: (order, [opcode, type, operands...])}) for the SSA lines of a module."""
    rows = [line.split() for line in spv.splitlines()]
    return rows, {row[0]: (number, row[2:]) for number, row in enumerate(rows)
                  if len(row) >= 3 and row[1] == "="}


class Gate:
    def __init__(self, test, spv):
        self.t = test
        self.rows, self.defs = parse(spv)

    def args(self, value):
        self.t.assertIn(value, self.defs, f"{value} is not defined")
        return self.defs[value][1]

    def order(self, value):
        self.t.assertIn(value, self.defs, f"{value} is not defined")
        return self.defs[value][0]

    def op(self, value, opcode):
        a = self.args(value)
        self.t.assertEqual(a[0], opcode, f"{value} is defined by {' '.join(a)}")
        return a

    def gated_shuffles(self):
        """Shuffles an OpIEqual compares: a gate. The source-presence ballot is compared with 0 instead."""
        return [name for name, (_o, a) in self.defs.items()
                if a[0] == "OpGroupNonUniformShuffle"
                and any(x[0] == "OpIEqual" and name in x[2:] for _oo, x in self.defs.values())]

    def reaches(self, value, target, limit=60):
        """True when `target` (e.g. %sg_lane) appears in the definition chain of `value`."""
        seen, todo = set(), [value]
        while todo and len(seen) < limit:
            v = todo.pop()
            if v in seen or v not in self.defs:
                continue
            seen.add(v)
            args = self.args(v)
            if target in args[2:]:
                return True
            todo.extend(a for a in args[2:] if a.startswith("%"))
        return False

    def stores(self, reg):
        return [(number, row[2]) for number, row in enumerate(self.rows)
                if len(row) == 3 and row[0] == "OpStore"
                and self.defs.get(row[1], (None, []))[1] ==
                ["OpAccessChain", "%_ptr_Function_uint", "%vgpr", reg]]


class DppWritebackGateTests(unittest.TestCase):
    def setUp(self):
        spv, _ = h.translate_cs("cs_dpp_writeback_gate", DPP, 32, emulate=False)
        self.g = Gate(self, spv)

    def test_dpp_write_is_gated_by_a_salted_self_shuffle(self):
        gates = self.g.gated_shuffles()
        self.assertEqual(len(gates), 1, "one DPP write, one gate")
        gate = gates[0]
        a = self.g.op(gate, "OpGroupNonUniformShuffle")
        _scope, _source, lane = a[2], a[3], a[4]
        self.assertTrue(self.g.reaches(lane, "%sg_lane"),
                        "the gate shuffles from the lane's own index (SubgroupLocalInvocationId)")

        equal = [n for n, (_o, x) in self.g.defs.items() if x[0] == "OpIEqual" and gate in x[2:]][0]
        eq_args = self.g.op(equal, "OpIEqual")
        salt = eq_args[3]
        flags = [n for n, (_o, x) in self.g.defs.items()
                 if x[0] == "OpSelect" and len(x) == 5 and x[3] == salt and x[4] == "%c0"]
        self.assertEqual(len(flags), 1, "the flag is `salt` where the lane is active, 0 where it is not")
        # The flag's condition is the lane's EXEC bit, possibly through a mask expression (a DPP write's
        # guard is the value that would otherwise be the write mask), so follow the chain instead of assuming
        # a bare OpLoad of %active.
        self.assertTrue(self.g.reaches(self.g.op(flags[0], "OpSelect")[2], "%active"),
                        "the flag must be built from the lane's EXEC bit (%active)")

    def test_the_gate_is_read_after_the_dpp_shuffle_and_before_the_write(self):
        gate = self.g.gated_shuffles()[0]
        data = [n for n, (_o, a) in self.g.defs.items()
                if a[0] == "OpGroupNonUniformShuffle" and n != gate]
        self.assertTrue(data, "the DPP's own source read must still be there")
        self.assertLess(self.g.order(data[0]), self.g.order(gate))

        equal = [n for n, (_o, x) in self.g.defs.items() if x[0] == "OpIEqual" and gate in x[2:]][0]
        store = self.g.stores("%c1")
        self.assertEqual(len(store), 1, "one store to v1")
        store_order, stored = store[0]
        select_order = self.g.order(stored)
        self.g.op(stored, "OpSelect")
        condition = self.g.args(stored)[2]
        reachable = []
        if self.g.defs[condition][1][0] == "OpLogicalAnd":
            reachable = self.g.defs[condition][1][2:]
        self.assertTrue(equal in reachable or condition == equal,
                        "the gate's comparison must reach the write's condition (ANDed with the DPP mask)")
        self.assertLess(self.g.order(gate), self.g.order(equal))
        self.assertLess(self.g.order(equal), select_order)
        self.assertLess(select_order, store_order)

    def test_an_ordinary_valu_write_gets_no_gate(self):
        """`dpp_guard` is set only during dpp_insn's re-entry, so nothing else may acquire a gate."""
        spv, _ = h.translate_cs("cs_plain_writeback_gate", PLAIN, 32, emulate=False)
        g = Gate(self, spv)
        self.assertEqual(g.gated_shuffles(), [], "a lane-local VALU write is not a subgroup write")


if __name__ == "__main__":
    unittest.main()
