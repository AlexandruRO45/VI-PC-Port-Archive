"""runtime/test/shaders/check.py has a row for every compute fixture scripts/run_gpu_tests.py judges.

    python -I tests/test_shader_check_compute_rows.py

check.py is the second runner of the GPU tests: CONTRIBUTING.md and docs/testing.md describe building the fixtures
with build.py and checking them with it. cs_buf_load_sub and cs_buf_fmt8 (TEST-510) and cs_f64_add had a check in
run_gpu_tests.py only, so that run never ran them. This collects check.py's rows without a GPU (its helpers are
replaced by stand-ins), compares the words it expects with each fixture's .asm header, and judges right and wrong
dumps through compute_test. No build, no GPU and no game data.
"""
import contextlib
import io
import os
import re
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_gpu_tests  # noqa: E402  (puts runtime/test/shaders on the path)
import check  # noqa: E402

ADDED = ("cs_buf_load_sub", "cs_buf_fmt8", "cs_f64_add")


def check_rows():
    """The names of the rows check.py's main() prints, every helper and test program replaced by a stand-in."""
    helpers = {name: (lambda *args, **kwargs: "") for name, value in vars(check).items()
               if isinstance(value, types.FunctionType) and name.endswith("_test")}
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
        for name, value in helpers.items():
            stack.enter_context(mock.patch.object(check, name, value))
        stack.enter_context(mock.patch.object(check, "run", return_value=(0, "0 bad\n0 bad\n0 bad\n")))
        stack.enter_context(mock.patch.object(check, "pick_build_dir", side_effect=lambda build, programs: build))
        stack.enter_context(mock.patch.object(sys, "argv", ["check.py", tmp, tmp]))
        stack.enter_context(contextlib.redirect_stdout(out))
        check.main()
    return [line.split()[0] for line in out.getvalue().splitlines()[:-1] if line.strip()]


def expected_words(name):
    """What check.py expects of a one-group dump fixture: its BOUNDS entry, or F64_ADD_WORDS."""
    if name == "cs_f64_add":
        return check.F64_ADD_WORDS
    return dict(check.BOUNDS)[name]


def header_words(name):
    """The output words the fixture's .asm header names: the hex words of its "out[0..7] = ..." line, or the
    "low/high words" of cs_f64_add."""
    with open(os.path.join(run_gpu_tests.SHADER_DIR, name + ".asm"), encoding="utf-8") as f:
        header = " ".join(line.lstrip("; ").rstrip() for line in f if line.startswith(";"))
    m = re.search(r"out\[0\.\.7\] = ((?:0x[0-9a-f]{8} ?)+)", header) or re.search(r"low/high words: (0x[0-9a-f]{8}), "
                                                                                    r"(0x[0-9a-f]{8})", header)
    if not m:
        raise AssertionError(f"{name}.asm names no output words")
    return [int(w, 16) for w in re.findall(r"0x[0-9a-f]{8}", m.group(0))]


def dump_line(words):
    return "out[0..7] = " + " ".join(str(w) for w in (list(words) + [0] * 8)[:8]) + "\n"


class ComputeRows(unittest.TestCase):
    def test_every_compute_fixture_run_gpu_tests_judges_has_a_check_py_row(self):
        fixtures = sorted(f[:-4] for f in os.listdir(run_gpu_tests.SHADER_DIR)
                          if f.startswith("cs_") and f.endswith(".asm"))
        judged = [n for n in fixtures if n not in run_gpu_tests.OTHER_HARNESS and not n.startswith("cs_bench_")]
        self.assertGreater(len(judged), 100, judged)
        rows = check_rows()
        self.assertEqual([n for n in judged if n not in rows], [],
                         "run_gpu_tests.py judges these compute fixtures and check.py has no row for them")
        self.assertEqual(len(rows), len(set(rows)), "a check.py row appears twice")

    def test_the_words_are_the_headers(self):
        for name in ADDED:
            with self.subTest(name=name):
                self.assertEqual(expected_words(name), header_words(name))


class JudgedDumps(unittest.TestCase):
    def judge(self, name, code, output):
        with mock.patch.object(check, "run", return_value=(code, output)), \
                mock.patch.object(check, "subgroup_size", return_value=None):
            return check.compute_test("tools", "spirv", name, "1", "dump", expected_words=expected_words(name))

    def test_the_header_words_pass_and_a_wrong_word_fails(self):
        for name in ADDED:
            want = expected_words(name)
            with self.subTest(name=name):
                self.assertEqual(self.judge(name, 0, "device: X (subgroup 32..32)\n" + dump_line(want)), "")
                wrong = list(want)
                wrong[-1] ^= 1
                problem = self.judge(name, 0, "device: X (subgroup 32..32)\n" + dump_line(wrong))
                self.assertTrue(problem)
                self.assertNotIsInstance(problem, check.Skip)

    def test_cs_f64_add_skips_without_float64(self):
        problem = self.judge("cs_f64_add", 77, "unsupported: cs_f64_add.spv uses Float64 and the device has no "
                                               "shaderFloat64\n")
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("shaderFloat64", problem)
        self.assertIn("exit 77", self.judge("cs_f64_add", 77, "something else\n"))


if __name__ == "__main__":
    unittest.main()
