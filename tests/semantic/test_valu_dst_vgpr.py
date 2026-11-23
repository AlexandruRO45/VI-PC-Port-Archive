"""A VALU instruction writes a VGPR; a scalar destination is refused instead of being translated as a VGPR write.

The destination field of VOP1, VOP2 and VOP3 is a VGPR number, so `v_add_f32 s8, v1, v2` cannot be encoded. The
translator used to accept it for 212 of the 251 VALU opcodes in tools/rdna2spirv's ISA tables and wrote the result to
a register the listing never named. `Gen._insn` now raises Unsupported for any `v_*` instruction whose first operand is
not a VGPR, except the ops whose scalar destination is the point: v_readlane_b32 and v_readfirstlane_b32 (an SGPR) and
the v_cmp/v_cmpx compares (an SGPR pair, VCC or EXEC).
Standard library only, no GPU, no game files.
"""
import unittest

import support  # puts tools/ on the path
import rdna2spirv  # noqa: E402
import rdna2_isa_tables as isa  # noqa: E402
import shader_ir  # noqa: E402

SCALAR_DST_OK = ("v_cmp", "v_readlane", "v_readfirstlane")
TAIL = ["exp mrt0 v1, v1, v1, v1 done vm", "s_endpgm"]


def valu_ops():
    names = set()
    for table in ("VOP1", "VOP2", "VOP3", "VOP3P"):
        for v in getattr(isa, table).values():
            names.add(v if isinstance(v, str) else v[0])
    return sorted(n for n in names if n.startswith("v_") and not n.startswith(SCALAR_DST_OK)
                  and n not in rdna2spirv.NOOPS)


def translate(line):
    return support.translate_lines("ps_valu_dst", ["v_mov_b32 v1, 1", line] + TAIL)


class ScalarDestinationRefused(unittest.TestCase):
    def test_every_valu_op_refuses_an_sgpr_destination(self):
        # One three-source line per op, given to a fresh Gen: the destination is looked at before the operand count,
        # so every op is refused for its destination whatever number of sources it takes.
        wrong = {}
        for op in valu_ops():
            try:
                rdna2spirv.Gen("ps")._insn(shader_ir.parse_line(f"	{op} s8, v1, v2, v3"))
                wrong[op] = "accepted"
            except rdna2spirv.Unsupported as e:
                if "VGPR" not in str(e):
                    wrong[op] = str(e)[:60]
            except Exception as e:
                wrong[op] = type(e).__name__
        self.assertEqual(wrong, {}, "VALU ops that were not refused for an SGPR destination")

    def test_special_registers_are_refused_too(self):
        for d in ("vcc_lo", "exec_lo", "null", "m0"):
            with self.subTest(dst=d):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(f"v_add_f32 {d}, v1, v2")


class LaneReadsWriteAScalar(unittest.TestCase):
    """The mirror rule: v_readlane_b32 and v_readfirstlane_b32 put a lane's value into an SGPR, never a VGPR."""

    def test_a_vgpr_destination_is_refused(self):
        for line in ("v_readfirstlane_b32 v8, v1", "v_readlane_b32 v8, v1, 0", "v_readlane_b32 v[8:9], v1, s2"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    rdna2spirv.Gen("ps")._insn(shader_ir.parse_line("	" + line))
                self.assertIn("SGPR", str(cm.exception))


class ScalarDestinationStillAllowedWhereDefined(unittest.TestCase):
    def test_readlane_readfirstlane_and_compares_translate(self):
        for line in ("v_readfirstlane_b32 s8, v1", "v_readlane_b32 s8, v1, 0",
                     "v_cmp_lt_f32 vcc_lo, v1, v1", "v_cmp_lt_f32_e64 s8, v1, v1", "v_cmpx_lt_u32 vcc_lo, v1, v1"):
            with self.subTest(line=line):
                translate(line)

    def test_vgpr_pairs_and_carry_forms_still_work(self):
        for line in ("v_add_co_u32 v2, vcc_lo, v1, v1", "v_lshlrev_b64 v[2:3], 1, v[0:1]",
                     "v_mad_u64_u32 v[2:3], s8, v1, v1, v[0:1]", "v_add_f32 v2, v1, v1"):
            with self.subTest(line=line):
                translate(line)


if __name__ == "__main__":
    unittest.main()
