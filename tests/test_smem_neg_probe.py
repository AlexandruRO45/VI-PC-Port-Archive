"""tools/smem_neg_probe.py without a GPU: encodings, translator expectations, kernel builder, and oracle lifecycle.

    python -I tests/test_smem_neg_probe.py
"""
import contextlib
import io
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "smem_neg_probe.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "hw_oracle"))

import runner  # noqa: E402
import smem_neg_probe as p  # noqa: E402


def words(row):
    """.long 0x<word> lines, as ints (the hand-encoded rows llvm-mc would not print)."""
    return [int(line.split()[-1], 16) for line in row]


class SmemNegProbeTest(unittest.TestCase):
    def test_seed_bounds(self):
        self.assertEqual(p.seed(-0x40), (p.SEED - 0x40) & 0xFFFFFFFF)
        self.assertEqual(p.seed(0), p.SEED)
        self.assertEqual(p.seed(0xBC), p.SEED + 0xBC)
        self.assertIsNone(p.seed(-0x44))
        self.assertIsNone(p.seed(0xC0))
        # the seeded window is -0x40 .. 0xBF of BASE (0x140), so 0xBF is the last seeded byte
        self.assertEqual(p.seed(0xBF), (p.SEED + 0xBF) & 0xFFFFFFFF)
        self.assertIsNone(p.seed(-0x41))

    def test_sbuf_raw_encoding(self):
        lines = p.sbuf_raw(48, 12, 0x7D, -4)
        self.assertEqual(lines, [".long 0xf4200c06", ".long 0xfa1ffffc"])
        w0, w1 = words(lines)
        self.assertEqual(w0, 0xF4200000 | (48 << 6) | (12 >> 1))
        self.assertEqual(w1, ((0x7D & 0x7F) << 25) | (-4 & 0x1FFFFF))
        # an even sbase is encoded as its own half (s[12:15] -> sbase>>1), an odd one truncates down
        self.assertEqual(words(p.sbuf_raw(8, 0x40, 0, 0))[0], 0xF4200000 | (8 << 6) | (0x40 >> 1))
        self.assertEqual(words(p.sbuf_raw(8, 0x41, 0, 0))[0], 0xF4200000 | (8 << 6) | (0x41 >> 1))

    def test_translator_sbuf_bounds(self):
        self.assertEqual(p.translator_sbuf(0, 0), [0])
        self.assertEqual(p.translator_sbuf(0xFC, 0), [0xFC])
        self.assertEqual(p.translator_sbuf(0x100, 0), [None])
        self.assertEqual(p.translator_sbuf(0xFFFFFFFC, 0), [None])
        # (sOFF & ~3) + (IMM & ~3), the IMM kept as the unsigned 32-bit value, the sum formed in 64 bits:
        # it does not wrap back into range. tools/rdna2spirv.py scalar_load says the same ("no 32-bit wrap",
        # the rule the translator got in #1686), so s20 = -4 with offset:8 is out of range, not byte 4.
        self.assertEqual(p.translator_sbuf(0xFFFFFFFC, 8), [None])
        self.assertEqual(p.translator_sbuf(None, -4), [None])       # 0x1ffffc, past the table
        self.assertEqual(p.translator_sbuf(4, -8), [None])          # 4 + 0xfffffff8 stays past it
        self.assertEqual(p.translator_sbuf(0x10, 8), [0x18])
        self.assertEqual(p.translator_sbuf(0x80000000, 0x80000000), [None])   # 2**32, kept wide

    def test_expected_and_kernel_lines(self):
        want = p.expected()
        self.assertEqual(list(want.keys()), [r[0] for r in p.ROWS])
        self.assertEqual(want["sload_imm_m4"], [(p.SEED - 4) & 0xFFFFFFFF])
        self.assertEqual(want["sload_x2_soff4_m10"], [(p.SEED - 0xC) & 0xFFFFFFFF, (p.SEED - 8) & 0xFFFFFFFF])
        # s20 = -4 with offset:8: the sum stays wide (0x100000004), so the bound zeroes the dword
        self.assertEqual(want["sbuf_soffm4_imm8"], [None])
        self.assertEqual(want["sbuf_soffm4"], [None])

        lines, order = p.kernel_lines()
        self.assertEqual(len(order), sum(len(r[2]) for r in p.ROWS))
        self.assertEqual(lines[-1], "s_endpgm")
        self.assertEqual(order, [r for _n, _l, regs, _s, _x in p.ROWS for r in regs])
        self.assertEqual(len([l for l in lines if l.startswith("buffer_store_dwordx4")]), 16)
        for name, ls, _regs, _soff, _xs in p.ROWS:
            self.assertIn(ls[0], lines, name)

    def test_not_modelled_rows_are_the_bit20_immediates(self):
        # The two hand-encoded rows set bit 20 in the IMM field; hardware then reads data that is neither 0
        # nor the buffer's own words, which the translator does not model (SHADER-336). The probe must
        # report those rows without expecting a match, and model every other sbuf row.
        self.assertEqual(p.NOT_MODELLED, {"sbuf_imm_m4", "sbuf_soff8_imm_m4"})
        seen = 0
        for name, ls, _regs, _soff, _xs in p.ROWS:
            if not ls[0].startswith(".long"):
                continue                                            # only sbuf_raw rows are hand-encoded
            imm = int(ls[1].split()[-1], 16) & 0x1FFFFF
            self.assertEqual(name in p.NOT_MODELLED, bool(imm & 0x100000), name)
            seen += 1
        self.assertEqual(seen, 2)
        want = p.expected()
        for name in p.NOT_MODELLED:
            self.assertIsNone(want[name][0], name)                  # the model itself predicts 0

    def test_expected_is_defined_for_every_row(self):
        want = p.expected()
        for name, _lines, regs, _soff, _xs in p.ROWS:
            self.assertIn(name, want)
            self.assertEqual(len(want[name]), len(regs), name)

    def test_translator_mode_prints_every_row(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = p.main(["--oracle", "translator"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        for name, *_ in p.ROWS:
            self.assertIn(name, out)

    def test_translator_mode_prints_one_row_per_case(self):
        # exactly one line per row, in ROWS order, with the modelled values on it
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = p.main(["--oracle", "translator"])
        self.assertEqual(rc, 0)
        want = p.expected()
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), len(p.ROWS))
        for line, (name, *_rest) in zip(lines, p.ROWS):
            got, rule, vals = line.split(None, 2)
            self.assertEqual(got, name)
            self.assertEqual(rule, "translator")
            self.assertEqual(vals.split(), ["0" if v is None else hex(v) for v in want[name]])


class OracleLifecycleTest(unittest.TestCase):
    def _expected_flat(self):
        want = p.expected()
        flat = []
        for name, *_ in p.ROWS:
            flat.extend(0 if v is None else v for v in want[name])
        return flat

    def _run_hw(self, fake_oracle):
        buf = io.StringIO()
        with (
            mock.patch.object(runner, "open_oracle", return_value=fake_oracle) as open_mock,
            mock.patch.object(runner, "Oracle", side_effect=AssertionError("Oracle() called directly")),
            contextlib.redirect_stdout(buf),
        ):
            rc = p.main(["--oracle", "hw"])
        open_mock.assert_called_once_with()
        return rc, buf.getvalue()

    def test_hw_uses_open_oracle_and_closes_on_match(self):
        flat = self._expected_flat()

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return [list(flat)]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 0)
        self.assertTrue(fake.closed)
        self.assertIn("device: fake-gpu (gfx1036)", out)
        self.assertNotIn("DIFFERS", out)

    def test_hw_closes_on_mismatch(self):
        flat = self._expected_flat()

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return [[0] * len(flat)]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 1)
        self.assertTrue(fake.closed)
        self.assertIn("DIFFERS", out)

    def test_hw_closes_on_kernel_rejection(self):
        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def run(self, kernels):
                return ["assembler rejected"]

            def close(self):
                self.closed = True

        fake = FakeOracle()
        rc, out = self._run_hw(fake)
        self.assertEqual(rc, 2)
        self.assertTrue(fake.closed)
        self.assertIn("kernel rejected", out)

    def test_hw_reports_the_not_modelled_rows_without_matching_them(self):
        # The bit-20 IMM rows are the ones hardware answers with data the translator does not model. The probe
        # has to show what the GPU returned and still call the run a match: silently dropping the row, or
        # scoring it against the model, would both hide the result the row exists to collect.
        pos, off = {}, 0
        for name, _ls, regs, _soff, _xs in p.ROWS:
            pos[name] = off
            off += len(regs)
        flat = self._expected_flat()
        for name in p.NOT_MODELLED:
            self.assertLess(pos[name], len(flat), name)

        class FakeOracle:
            device = "fake-gpu"
            arch = "gfx1036"
            closed = False

            def __init__(self, values):
                self.values = values

            def run(self, kernels):
                return [list(self.values)]

            def close(self):
                self.closed = True

        odd = list(flat)
        for name in p.NOT_MODELLED:
            odd[pos[name]] = 0xDEADBEEF          # what a bit-20 IMM row really reads back
        rc, out = self._run_hw(FakeOracle(odd))
        self.assertEqual(rc, 0, out)
        self.assertNotIn("DIFFERS", out)
        for name in p.NOT_MODELLED:
            line = [l for l in out.splitlines() if l.startswith(name + " ")]
            self.assertEqual(len(line), 1, out)
            self.assertIn("not modelled", line[0])
            self.assertIn(hex(0xDEADBEEF), line[0])

    def test_hw_unavailable_returns_77(self):
        buf = io.StringIO()
        with (
            mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no device")),
            mock.patch.object(runner, "Oracle", side_effect=AssertionError("Oracle() called directly")),
            contextlib.redirect_stdout(buf),
        ):
            rc = p.main(["--oracle", "hw"])
        self.assertEqual(rc, 77)
        self.assertIn("no device", buf.getvalue())

    def test_hw_mode_is_skipped_77_without_an_amd_oracle(self):
        # the same contract from a real process: main() takes argv, so a runner passing its own
        # arguments cannot leak into the tool, and --oracle hw on a machine with no AMD device exits 77
        proc = subprocess.run([sys.executable, "-I", TOOL, "--oracle", "hw"],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 77, proc.stdout + proc.stderr)
        self.assertIn("hardware oracle unavailable", proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
