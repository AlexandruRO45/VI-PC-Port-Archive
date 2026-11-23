"""Unit tests for tools/fspx_prologue_probe.py (TEST-849). Standard library only; no GPU or game files needed."""
import io
import sys
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import fspx_prologue_probe as probe  # noqa: E402
from hw_oracle import asm, runner as hw_runner  # noqa: E402


class FspxPrologueProbeTests(unittest.TestCase):
    def test_prologue_layout_and_invariants(self):
        self.assertGreaterEqual(len(probe.PROLOGUES), 5)
        for name, words in probe.PROLOGUES.items():
            self.assertEqual(len(words), 9, name)
            self.assertEqual(words[0] & probe.WORD0_FIXED_MASK, probe.WORD0_FIXED_VAL, name)
            self.assertEqual(words[1] & probe.WORD1_FIXED_MASK, probe.WORD1_FIXED_VAL, name)
            self.assertEqual(words[3] & probe.WORD0_FIXED_MASK, probe.WORD0_FIXED_VAL, name)
            self.assertEqual(words[4] & probe.WORD1_FIXED_MASK, probe.WORD1_FIXED_VAL, name)
            self.assertEqual(tuple(words[6:]), probe.TAIL_WORDS, name)
            asm_lines = probe.prologue_asm_lines(words)
            self.assertEqual(len(asm_lines), 9)
            self.assertTrue(all(line.startswith(".long 0x") and len(line) == 16 for line in asm_lines))

        with self.assertRaises(ValueError):
            probe.prologue_asm_lines((0,) * 8)

    def test_probe_plan_and_wave_assembly(self):
        positives, negatives = probe.build_probe_plan()
        self.assertEqual(len(positives), len(probe.PROLOGUES) * 5 * 2)
        self.assertEqual(len(negatives), 8)
        tags = [tag for tag, _, _ in positives] + [tag for tag, _, _, _ in negatives]
        self.assertEqual(len(tags), len(set(tags)))

        for wave in (32, 64):
            kernels = [k for _, k, _ in positives if k.wave == wave]
            text, names, _ = asm.module_text(kernels, "gfx1030", wave=wave)
            self.assertEqual(len(names), len(positives) // 2)
            self.assertIn(".long 0xd5800000", text)
            self.assertIn(".long 0x00000ac0", text)
            self.assertIn(f".amdhsa_wavefront_size32 {1 if wave == 32 else 0}", text)

    def test_self_test_exits_zero(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = probe.main(["--self-test"])
        self.assertEqual(rc, 0)
        self.assertIn("self-test:", buf.getvalue())

    def test_hw_unavailable_returns_77(self):
        err = io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", side_effect=hw_runner.Unavailable("no GPU")):
            with redirect_stderr(err):
                rc = probe.main(["--oracle", "hw"])
        self.assertEqual(rc, 77)
        self.assertIn("no GPU", err.getvalue())

    def _mock_oracle(self, outputs):
        mock_oracle = unittest.mock.MagicMock()
        mock_oracle.device = "device=mock arch=gfx1031"
        mock_oracle.run.return_value = outputs
        return mock_oracle

    def _passing_outputs(self):
        positives, negatives = probe.build_probe_plan()
        out = [list(exp) for _, _, exp in positives]
        for tag, _, exp, _ in negatives:
            bad = list(exp)
            if tag.startswith("neg_vgpr"):
                bad[0] ^= 1
            elif tag.startswith("neg_sgpr_scc"):
                bad[4] ^= 1
                bad[12] ^= 1
            elif tag.startswith("neg_ctrl"):
                bad[0] = 0x11111111
                bad[1] ^= 1
                bad[3] ^= 2
            elif tag.startswith("neg_multilane"):
                for i in range(0, 64, 2):
                    bad[i] ^= 1
            out.append(bad)
        return out

    def test_mocked_hw_pass_and_failure_detection(self):
        good = self._passing_outputs()
        out_buf = io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=self._mock_oracle(good)):
            with redirect_stdout(out_buf):
                rc = probe.main(["--oracle", "hw"])
        self.assertEqual(rc, 0)
        self.assertIn("PASS: 50/50", out_buf.getvalue())

        # Positive clobber must fail
        clobbered = [list(row) for row in good]
        clobbered[0][0] ^= 1
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=self._mock_oracle(clobbered)):
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = probe.main(["--oracle", "hw"])
        self.assertEqual(rc, 1)
        self.assertIn("FAIL", err_buf.getvalue())

        # Missed negative control must also fail
        missed_neg = [list(row) for row in good]
        _, negatives = probe.build_probe_plan()
        missed_neg[-len(negatives)] = list(negatives[0][2])
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=self._mock_oracle(missed_neg)):
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = probe.main(["--oracle", "hw"])
        self.assertEqual(rc, 1)
        self.assertIn("negative control failed", err_buf.getvalue())

    def _run_malformed(self, outputs):
        oracle = self._mock_oracle(outputs)
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", return_value=oracle):
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                rc = probe.main(["--oracle", "hw"])
        oracle.close.assert_called_once_with()
        return rc, out_buf.getvalue(), err_buf.getvalue()

    def assert_harness_error(self, outputs, detail):
        rc, out, err = self._run_malformed(outputs)
        self.assertEqual(rc, 2)
        self.assertNotIn("PASS", out)
        self.assertEqual(err.count("\n"), 1, err)
        self.assertTrue(err.startswith("harness error: "), err)
        self.assertIn(detail, err)
        self.assertNotIn("Traceback", err)

    def test_truncated_results_are_a_harness_error(self):
        good = self._passing_outputs()
        # Dropping the negative controls used to print "PASS: 50/50 ... 8/8 negative controls" and exit 0
        _, negatives = probe.build_probe_plan()
        self.assert_harness_error(good[:-len(negatives)], f"returned 50 results for {len(good)} kernels")
        self.assert_harness_error(good[:10], f"returned 10 results for {len(good)} kernels")

    def test_empty_results_are_a_harness_error(self):
        self.assert_harness_error([], f"returned 0 results for {len(self._passing_outputs())} kernels")

    def test_extra_results_are_a_harness_error(self):
        good = self._passing_outputs()
        self.assert_harness_error(good + [list(good[0])], f"returned {len(good) + 1} results for {len(good)} kernels")

    def test_none_result_list_is_a_harness_error(self):
        self.assert_harness_error(None, "returned NoneType results")

    def test_none_entry_is_a_harness_error(self):
        # runner.run() pre-fills its answer with None; a kernel that never ran comes back as None
        for i in (0, -1):
            outputs = self._passing_outputs()
            outputs[i] = None
            self.assert_harness_error(outputs, "returned None, not")

    def test_int_entry_is_a_harness_error(self):
        outputs = self._passing_outputs()
        outputs[3] = 0
        self.assert_harness_error(outputs, "kernel 3: the hardware runner returned 0, not")

    def test_short_word_list_is_a_harness_error(self):
        outputs = self._passing_outputs()
        positives, _ = probe.build_probe_plan()
        want = positives[0][1].words
        outputs[0] = outputs[0][:want - 1]   # a matching prefix used to raise StopIteration
        self.assert_harness_error(outputs, f"kernel 0: the hardware runner returned {want - 1} words, not {want}")
        outputs = self._passing_outputs()
        outputs[-1] = []                     # neg_multilane: an empty list used to count as a missed control
        self.assert_harness_error(outputs, "returned 0 words")

    def test_non_list_result_is_a_harness_error(self):
        self.assert_harness_error(5, "returned int results")
        self.assert_harness_error(iter(self._passing_outputs()), "returned list_iterator results")
        self.assert_harness_error(tuple(self._passing_outputs()), "returned tuple results")

    def test_non_int_word_is_a_harness_error(self):
        for bad in (None, 1.0, "0"):
            outputs = self._passing_outputs()
            outputs[2] = [bad] + outputs[2][1:]
            self.assert_harness_error(outputs, f"kernel 2: the hardware runner returned the word {bad!r}, not an int")

    def test_rejection_string_is_still_a_failure_not_a_harness_error(self):
        outputs = self._passing_outputs()
        outputs[0] = "assembler rejected the kernel"
        rc, _, err = self._run_malformed(outputs)
        self.assertEqual(rc, 1)
        self.assertIn("assembler rejected the kernel", err)
        self.assertNotIn("harness error", err)


if __name__ == "__main__":
    unittest.main()
