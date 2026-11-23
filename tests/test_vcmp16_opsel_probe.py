"""tools/vcmp16_opsel_probe.py without a GPU: the raw compare and control words are llvm-mc's with OPSEL where llvm-mc
puts it, the rows tell the two hypotheses apart, and the kernel builds.

    python -I tests/test_vcmp16_opsel_probe.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hw_oracle"))

import vcmp16_opsel_probe as p  # noqa: E402


def le_bytes(w0, w1):
    return list(w0.to_bytes(4, "little") + w1.to_bytes(4, "little"))


class EncodingTest(unittest.TestCase):
    def test_compares_without_opsel_are_llvm_mc_words(self):
        # llvm-mc 21.1.8 -mcpu=gfx1036 -show-encoding for `v_cmp_<op>_e64 s10, v1, v2`
        self.assertEqual(le_bytes(*p.words("lt", "f16", (0, 0))), [0x0A, 0x00, 0xC9, 0xD4, 0x01, 0x05, 0x02, 0x00])
        self.assertEqual(le_bytes(*p.words("gt", "u16", (0, 0))), [0x0A, 0x00, 0xAC, 0xD4, 0x01, 0x05, 0x02, 0x00])
        self.assertEqual(le_bytes(*p.words("lt", "i16", (0, 0))), [0x0A, 0x00, 0x89, 0xD4, 0x01, 0x05, 0x02, 0x00])
        # `v_cmp_lt_f16_e64 s10, -v1, v2`
        self.assertEqual(le_bytes(*p.words("lt", "f16", (0, 0), True))[7], 0x20)

    def test_opsel_sits_where_llvm_mc_puts_it_on_v_add_nc_u16(self):
        # llvm-mc for `v_add_nc_u16 v0, v1, v2 op_sel:[1,0,0]` / [0,1,0] / [1,1,0]: byte 1 = 0x08 / 0x10 / 0x18
        self.assertEqual(p.OPSEL_SRC0 >> 8, 0x08)
        self.assertEqual(p.OPSEL_SRC1 >> 8, 0x10)
        self.assertEqual(le_bytes(*p.words("lt", "f16", (1, 1)))[1], 0x18)   # the finding's d4c9180a

    def test_control_answers(self):
        self.assertEqual(p.control_answer((0, 0), True), 0x000C)
        self.assertEqual(p.control_answer((1, 1), True), 0x0340)
        self.assertEqual({p.control_answer(s, False) for _n, s in p.control_rows()}, {0x000C})


class RowsTest(unittest.TestCase):
    def test_names_are_unique(self):
        rows = p.rows()
        self.assertEqual(len({r[0] for r in rows}), len(rows))

    def test_every_kind_and_opsel_bit_has_a_row_that_tells_the_hypotheses_apart(self):
        telling = [r for r in p.rows() if p.answer(r, True) != p.answer(r, False)]
        for kind in ("f16", "u16", "i16"):
            for opsel in ((1, 0), (0, 1), (1, 1)):
                with self.subTest(kind=kind, opsel=opsel):
                    self.assertTrue(any(r[2] == kind and r[4] == opsel for r in telling))

    def test_i16_reads_the_high_half_signed(self):
        row = next(r for r in p.rows() if r[0] == "i16_lt_opsel01")
        self.assertEqual(p.half_value(0x8000, "i16"), -32768)
        self.assertEqual(p.answer(row, True), 0)       # 5 < 3: false
        row = next(r for r in p.rows() if r[0] == "i16_lt_opsel10")
        self.assertEqual(p.answer(row, True), 1)       # -32768 < 7

    def test_kernel_builds(self):
        rows = p.rows()
        lines = p.kernel(rows)
        self.assertEqual(sum(ln.startswith(".long 0xd4") for ln in lines), len(rows))
        self.assertEqual(sum(ln.startswith(".long 0xd703") for ln in lines), len(p.control_rows()))
        self.assertEqual("\n".join(lines).count("global_store_b32"), len(rows) + len(p.control_rows()))


class _FakeOracle:
    """A gfx1030 oracle that answers every kernel with `words` (the measured rule: compares ignore OPSEL)."""
    arch, device = "gfx1030", "fake gfx1030"

    def __init__(self, words):
        self.words = words

    def run(self, kernels, words=4):
        return [list(self.words) for _ in kernels]

    def close(self):
        pass


class HardwareResultTest(unittest.TestCase):
    @staticmethod
    def clean_words():
        return ([p.answer(r, False) for r in p.rows()] +
                [p.control_answer(opsel, True) for _n, opsel in p.control_rows()])

    def run_main(self, words):
        import contextlib
        import io
        from unittest import mock
        import runner
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["vcmp16_opsel_probe.py", "--oracle", "hw"]), \
                mock.patch.object(runner, "open_oracle", lambda **kw: _FakeOracle(words)), \
                contextlib.redirect_stdout(out):
            rc = p.main()
        return rc, out.getvalue()

    def test_a_clean_result_passes(self):
        # Control for the next test: the same words with nothing left unwritten are a pass.
        rc, out = self.run_main(self.clean_words())
        self.assertEqual(rc, 0, out[-600:])

    def test_a_word_the_kernel_never_stored_is_a_harness_error_not_a_pass(self):
        import runner
        for index in (0, len(p.rows())):            # a compare row, and the first control word
            with self.subTest(word=index):
                words = self.clean_words()
                words[index] = runner.SENTINEL
                rc, out = self.run_main(words)
                self.assertEqual(rc, 2, out[-600:])
                self.assertIn(f"left word {index} unwritten", out)


if __name__ == "__main__":
    unittest.main()
