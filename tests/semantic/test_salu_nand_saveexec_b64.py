"""s_nand_saveexec_b64 through the translator, on every lane of a wave64.

RDNA2 ISA, SOP1:
  s_nand_saveexec_b64: D = EXEC; EXEC = ~(S0 & EXEC); SCC = (EXEC != 0).

Each case translates a hand-written compute program with tools/rdna2spirv.py and runs 64 lanes in
tests/semantic/spirv_mini_interp.py (harness: saveexec_run.py), with S0 from VCC, an SGPR pair written by v_cmp, an
SGPR pair loaded with literals, and EXEC itself, against the ISA formula in plain Python. Until this file, the test of
this name ran a self-built interpreter and never called the translator.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import saveexec_run  # noqa: E402  (support puts tools/ on sys.path)
import support  # noqa: E402

ALL64 = (1 << 64) - 1


class Saveexec(unittest.TestCase):
    def test_nand_by_hand(self):
        # S0 = lanes 0..31, EXEC = lanes 16..47: EXEC = ~(S0 & EXEC) written out by hand.
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        s0, e = lanes(0, 32), lanes(16, 48)
        want = (lanes(0, 16) | lanes(32, 64)) & ALL64
        self.assertEqual(saveexec_run.model("nand", s0, e), (want, e, 1))

    def test_nand_zero_and_full_exec_transitions(self):
        # When old_exec == 0, ~(S0 & 0) == ~0 == ALL64 (inactive lanes reactivate)
        new_exec, d_out, scc = saveexec_run.model("nand", 0x123456789ABCDEF0, 0)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(d_out, 0)
        self.assertEqual(scc, 1)

        # When S0 == ALL64 and old_exec == ALL64, ~(ALL64 & ALL64) == 0
        new_exec_zero, d_out_full, scc_zero = saveexec_run.model("nand", ALL64, ALL64)
        self.assertEqual(new_exec_zero, 0)
        self.assertEqual(d_out_full, ALL64)
        self.assertEqual(scc_zero, 0)

    def test_nand_self_exec_inversion(self):
        exec_val = 0xAAAAAAAAAAAAAAAA
        new_exec, d_out, scc = saveexec_run.model("nand", exec_val, exec_val)
        self.assertEqual(new_exec, (~exec_val) & ALL64)
        self.assertEqual(d_out, exec_val)
        self.assertEqual(scc, 1)

    def test_nand_spirv_emission(self):
        lines = [
            "s_mov_b32 s0, 0x0000ffff",
            "s_mov_b32 s1, 0x00000000",
            "s_nand_saveexec_b64 s[4:5], s[0:1]",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_test_nand_emit", lines, fssm=saveexec_run.fssm(threads=64))
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

    def test_nand_every_lane(self):
        saveexec_run.check(self, "nand")


if __name__ == "__main__":
    unittest.main()
