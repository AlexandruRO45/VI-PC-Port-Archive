"""s_movreld_b32, s_movreld_b64 and s_movrelsd_2_b32: relative SGPR writes through M0.

RDNA2 ISA SOP1: S_MOVRELD_B32 SGPR[D + M0] = S0; S_MOVRELD_B64 writes the pair at D + M0 (both halves read first);
S_MOVRELSD_2_B32 (GFX10) SGPR[D + M0[25:16]] = SGPR[S0 + M0[9:0]]. SCC and M0 are not written. The shader is
runtime/test/shaders/ps_movreld.asm (also run on the GPU by check.py and run_gpu_tests.py); its header derives the
four self-check words by hand. The same facts were measured on an RDNA2 GPU with tools/int_hw_probe.py
(--group salu_movreld). The reference here runs the ISA text in plain Python on the same register fill and counts the
checks that must pass in each word.
"""
import unittest

import support
import rdna2spirv

EXPECTED = [0x403FFFFF, 0x400FFFFF, 0x4FFFFFFF, 0x40000007]       # derivation in ps_movreld.asm
F = {r: 0x40000000 | (r << 20) | (r * 0x111) for r in range(10, 20)}
S4, S5, LIT = 0x4AFE0004, 0x4AFE0005, 0x4A123456


def fresh():
    regs = dict(F)
    regs[4], regs[5] = S4, S5
    return regs


def movreld_b32(regs, m0, d, value):
    regs[d + m0] = value


def movreld_b64(regs, m0, d, s):
    lo, hi = regs[s], regs[s + 1]
    regs[d + m0], regs[d + m0 + 1] = lo, hi


def movrelsd_2(regs, m0, d, s):
    regs[d + ((m0 >> 16) & 0x3FF)] = regs[s + (m0 & 0x3FF)]


def reference_words():
    """The checks of each word of the header, evaluated on the ISA semantics: (register, expected) pairs per case;
    a check passes when the model's register has the expected value, which is what the shader's s_cmp_eq_u32 tests."""
    words = [0x40000000] * 4
    counts = [0] * 4

    def check(w, regs, reg, expected):
        if regs[reg] == expected:
            words[w] |= 1 << counts[w]
        counts[w] += 1

    def same(w, regs, names):
        for r in names:
            check(w, regs, r, F[r])

    for _ in range(2):                                  # SCC 0 and 1: SCC is kept (one check, always true here)
        r = fresh(); movreld_b32(r, 2, 10, r[4])
        check(0, {"scc": 1, **r}, "scc", 1); check(0, r, 12, S4); same(0, r, (10, 11, 13, 14))
    r = fresh(); movreld_b32(r, 0, 10, r[4]); check(0, r, 10, S4); same(0, r, (11, 12))
    r = fresh(); movreld_b32(r, 3, 10, LIT); check(0, r, 13, LIT); same(0, r, (10, 12, 14))
    r = fresh(); movreld_b32(r, 1, 11, r[12]); check(0, r, 12, F[12]); same(0, r, (11, 13))

    for _ in range(2):
        r = fresh(); movreld_b64(r, 2, 10, 4)
        check(1, {"scc": 1, **r}, "scc", 1); check(1, r, 12, S4); check(1, r, 13, S5); same(1, r, (10, 11, 14))
    r = fresh(); movreld_b64(r, 1, 10, 4); check(1, r, 11, S4); check(1, r, 12, S5); same(1, r, (10, 13))
    r = fresh(); movreld_b64(r, 1, 10, 10); check(1, r, 11, F[10]); check(1, r, 12, F[11]); same(1, r, (10, 13))

    for _ in range(2):
        r = fresh(); movrelsd_2(r, 0x30002, 10, 14)
        check(2, r, 13, F[16]); check(2, {"scc": 1, **r}, "scc", 1); same(2, r, (10, 11, 12, 14, 16))
    r = fresh(); movrelsd_2(r, 0xFC030802, 10, 14); check(2, r, 13, F[16]); same(2, r, (10, 11, 12, 14, 16))
    r = fresh(); movrelsd_2(r, 0x10003, 10, 14); check(2, r, 11, F[17]); same(2, r, (10, 12, 13, 14))
    r = fresh(); movrelsd_2(r, 0x10000, 10, 10); check(2, r, 11, F[10]); same(2, r, (10, 12))

    r = fresh(); movreld_b32(r, 2, 10, r[4]); check(3, r, 12, S4); check(3, {"m0": 2, **r}, "m0", 2)
    check(3, {"m0": 0x20001, **r}, "m0", 0x20001)
    return words, counts


def hexes(words):
    return [f"{w:#010x}" for w in words]


def translate(lines):
    text = "; ps_movreld_case\n" + "".join("\t" + ln + "\n" for ln in lines)
    return support.translate_listing("ps_movreld_case", text.encode("utf-8"))


EXPORT = ["v_mov_b32 v10, s20", "v_mov_b32 v11, s21", "v_mov_b32 v12, s22", "v_mov_b32 v13, s23",
          "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]


class MovrelD(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        words, counts = reference_words()
        self.assertEqual(counts, [22, 20, 28, 3])
        self.assertEqual(hexes(words), hexes(EXPECTED))

    def test_ps_movreld(self):
        spv = support.translate_fixture("ps_movreld")
        self.assertEqual(hexes(support.mrt0_words(spv)), hexes(EXPECTED))

    def test_values_after_each_op(self):
        """Each op on its own: export the SGPRs it may have reached."""
        fill = [f"s_mov_b32 s{n}, {F[n]:#x}" for n in range(10, 20)] + [f"s_mov_b32 s4, {S4:#x}", f"s_mov_b32 s5, {S5:#x}"]
        cases = [
            (["s_mov_b32 m0, 2", "s_movreld_b32 s10, s4"], {12: S4}),
            (["s_mov_b32 m0, 130", "s_movreld_b32 s10, s4"], {12: S4}),      # the index wraps to the 128 slots
            (["s_mov_b32 m0, 1", "s_movreld_b64 s[10:11], s[4:5]"], {11: S4, 12: S5}),
            (["s_mov_b32 m0, 0x30002", "s_movrelsd_2_b32 s10, s14"], {13: F[16]}),
        ]
        for ops, changed in cases:
            with self.subTest(ops=ops):
                regs = fresh(); regs.update(changed)
                lines = fill + ops + ["v_mov_b32 v10, s10", "v_mov_b32 v11, s11", "v_mov_b32 v12, s12",
                                      "v_mov_b32 v13, s13", "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]
                got = support.mrt0_words(translate(lines))
                want = [regs[10], regs[11], regs[12], regs[13]]
                self.assertEqual(hexes(got), hexes(want))

    def test_literal_knowledge_is_dropped(self):
        """A literal an s_mov_b32 left in a register is not its value once a relative write may have changed it: the
        shift count of v_lshlrev_b32 comes from s10, which s_movreld_b32 overwrote (8, not 4)."""
        lines = ["s_mov_b32 s10, 4", "s_mov_b32 s4, 8", "s_mov_b32 m0, 0", "s_movreld_b32 s10, s4",
                 "v_mov_b32 v1, 0x100", "v_lshlrev_b32 v10, s10, v1", "v_mov_b32 v11, s10", "v_mov_b32 v12, 0",
                 "v_mov_b32 v13, 0", "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]
        got = support.mrt0_words(translate(lines))
        self.assertEqual(hexes(got), hexes([0x10000, 8, 0, 0]))

    def test_operand_forms_that_stay_unsupported(self):
        for ops in (["s_movreld_b32 v0, s4"], ["s_movreld_b64 s[10:11], s4"], ["s_movrelsd_2_b32 s10, 5"]):
            with self.subTest(ops=ops):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate(["s_mov_b32 m0, 1"] + ops + EXPORT)


if __name__ == "__main__":
    unittest.main()
