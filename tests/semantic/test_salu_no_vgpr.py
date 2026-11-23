"""A scalar instruction cannot name a VGPR, and the translator refuses one instead of treating it as a vector access.

The scalar ALU and the scalar memory unit read and write SGPRs, constants and special registers only; no SOP1, SOP2,
SOPK, SOPC, SOPP or SMEM operand is a VGPR, so `s_add_u32 v8, s1, s2` or `s_mov_b32 s8, v1` cannot be encoded.
`Gen._insn` used to accept a VGPR destination on 62 of the 143 SOP1/SOP2/SOPK ops and wrote the result into the
vector register file. It now raises Unsupported for any `s_*` instruction with a VGPR operand.
Standard library only, no GPU, no game files.
"""
import unittest

import support  # puts tools/ on the path
import rdna2spirv  # noqa: E402
import rdna2_isa_tables as isa  # noqa: E402
import shader_ir  # noqa: E402


def salu_ops():
    names = set()
    for table in ("SOP1", "SOP2", "SOPK", "SOPC"):
        for v in getattr(isa, table).values():
            names.add(v if isinstance(v, str) else v[0])
    return sorted(n for n in names if n.startswith("s_") and n not in rdna2spirv.NOOPS)


def insn(line):
    rdna2spirv.Gen("ps")._insn(shader_ir.parse_line("\t" + line))


class VgprOperandRefused(unittest.TestCase):
    def refused(self, line):
        try:
            insn(line)
        except rdna2spirv.Unsupported as e:
            return "VGPR" in str(e)
        except Exception:
            return False
        return False

    def test_every_salu_op_refuses_a_vgpr_destination(self):
        wrong = [op for op in salu_ops() if not self.refused(f"{op} v8, s1, s2")]
        self.assertEqual(wrong, [], "SALU ops not refused for a VGPR destination")

    def test_every_salu_op_refuses_a_vgpr_source(self):
        wrong = [op for op in salu_ops() if not self.refused(f"{op} s8, v1, s2")
                 and not self.refused(f"{op} s8, s1, v2")]
        self.assertEqual(wrong, [], "SALU ops not refused for a VGPR source")

    def test_vgpr_pairs_and_the_memory_unit_too(self):
        for line in ("s_mov_b64 v[8:9], s[2:3]", "s_load_dwordx2 v[8:9], s[4:5], 0x0",
                     "s_load_dword s8, v[4:5], 0x0", "s_cmp_eq_u32 v1, s2", "s_cbranch_scc1 v1"):
            with self.subTest(line=line):
                self.assertTrue(self.refused(line), line)


class ScalarOperandsStillAccepted(unittest.TestCase):
    def test_sgpr_forms_translate(self):
        for line in ("s_mov_b32 s8, s1", "s_add_u32 s8, s1, s2", "s_mov_b64 s[8:9], s[2:3]", "s_not_b32 s8, 5",
                     "s_and_b32 s8, s1, 0x1234"):
            with self.subTest(line=line):
                insn(line)


if __name__ == "__main__":
    unittest.main()
