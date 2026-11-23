"""tools/report_output.py: the output-path check, the input/output collision check and the JSON report writer.

    python -I tests/test_report_output.py

Standard library only; everything runs in a temporary folder on hand-made files, no game data.
"""
import json
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import report_output as ro  # noqa: E402


def touch(path, data=b"x"):
    with open(path, "wb") as f:
        f.write(data)


class TempDir(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = os.path.realpath(tmp.name)

    def p(self, *parts):
        return os.path.join(self.dir, *parts)


class CheckOutputPathTests(TempDir):
    def test_a_new_file_and_missing_parents_are_fine(self):
        self.assertIsNone(ro.check_output_path(self.p("report.json")))
        self.assertIsNone(ro.check_output_path(self.p("a", "b", "report.json")))
        touch(self.p("old.json"))
        self.assertIsNone(ro.check_output_path(self.p("old.json")), "replacing an earlier report is allowed")

    def test_refusals_name_the_flag(self):
        touch(self.p("blocker"))
        cases = (
            ("", "must not be empty"),
            ("   ", "must not be empty"),
            (self.p("a\x00b.json"), "NUL byte"),
            (self.dir, "is a directory"),
            (self.p("out") + os.sep, "is a directory"),
            (self.p("blocker", "r.json"), "parent is not a directory"),
            (self.p("blocker", "deeper", "r.json"), "parent is not a directory"),
        )
        for path, expected in cases:
            with self.subTest(path=path):
                error = ro.check_output_path(path, "--csv")
                self.assertIsNotNone(error)
                self.assertIn(expected, error)
                self.assertTrue(error.startswith("--csv "), error)

    def test_default_flag(self):
        self.assertTrue(ro.check_output_path("").startswith("--out "))

    def test_nothing_is_created(self):
        ro.check_output_path(self.p("a", "b", "report.json"))
        self.assertFalse(os.path.exists(self.p("a")))


class OutputCollisionTests(TempDir):
    def setUp(self):
        super().setUp()
        self.src = self.p("input.bin")
        touch(self.src, b"keep me")

    def test_distinct_paths_pass(self):
        self.assertIsNone(ro.output_collision([("--json", self.p("r.json")), ("--csv", self.p("r.csv"))], [self.src]))
        self.assertIsNone(ro.output_collision([], [self.src]))
        self.assertIsNone(ro.output_collision([("--json", self.p("r.json"))]))

    def test_unset_outputs_and_inputs_are_skipped(self):
        self.assertIsNone(ro.output_collision([("--json", None), ("--csv", ""), ("--out", self.p("o"))], [None, ""]))

    def test_an_output_that_is_an_input(self):
        error = ro.output_collision([("--json", self.src)], [self.p("other.bin"), self.src])
        self.assertEqual(error, f"--json {self.src} would overwrite the input {self.src}")

    def test_another_spelling_of_an_input(self):
        dotted = os.path.join(self.dir, "sub", "..", "input.bin")
        self.assertIn("would overwrite the input", ro.output_collision([("--out", dotted)], [self.src]))

    @unittest.skipUnless(os.name == "nt", "the volume folds letter case on Windows")
    def test_a_case_only_spelling_of_an_input(self):
        upper = self.p("INPUT.BIN")
        self.assertIn("would overwrite the input", ro.output_collision([("--out", upper)], [self.src]))

    def test_a_hard_link_to_an_input(self):
        link = self.p("link.json")
        try:
            os.link(self.src, link)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"no hard links here: {e}")
        self.assertIn("would overwrite the input", ro.output_collision([("--json", link)], [self.src]))

    def test_a_symlink_to_an_input(self):
        link = self.p("sym.json")
        try:
            os.symlink(self.src, link)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"no symlinks here: {e}")
        self.assertIn("would overwrite the input", ro.output_collision([("--json", link)], [self.src]))

    def test_two_outputs_that_name_one_file(self):
        path = self.p("r.json")
        error = ro.output_collision([("--json", path), ("--csv", self.p("x", "..", "r.json"))], [self.src])
        self.assertIn("name the same file", error)
        self.assertIn("--csv", error)
        self.assertIn("--json", error)

    def test_an_input_folder_is_not_an_output_file(self):
        self.assertIsNone(ro.output_collision([("--out", self.p("report.json"))], [self.dir]))

    def test_nothing_is_written(self):
        ro.output_collision([("--json", self.src)], [self.src])
        with open(self.src, "rb") as f:
            self.assertEqual(f.read(), b"keep me")


class WriteJsonReportTests(TempDir):
    def test_writes_lf_json_with_a_final_newline(self):
        path = self.p("r.json")
        self.assertIsNone(ro.write_json_report(path, {"b": [1, 2], "a": "é"}))
        with open(path, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"}\n"))
        self.assertEqual(json.loads(raw.decode("utf-8")), {"b": [1, 2], "a": "é"})
        self.assertIn(b'\n "b"', raw, "indent 1 by default")

    def test_indent(self):
        path = self.p("r.json")
        ro.write_json_report(path, {"a": 1}, indent=2)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b'{\n  "a": 1\n}\n')

    def test_creates_missing_parents(self):
        path = self.p("a", "b", "r.json")
        self.assertIsNone(ro.write_json_report(path, []))
        self.assertTrue(os.path.isfile(path))

    def test_a_failed_write_is_one_message(self):
        touch(self.p("blocker"))
        error = ro.write_json_report(self.p("blocker", "r.json"), {}, flag="--report")
        self.assertIsNotNone(error)
        self.assertTrue(error.startswith("cannot "), error)
        self.assertIn("--report", error)
        error = ro.write_json_report(self.dir, {})
        self.assertIsNotNone(error)
        self.assertIn("cannot write --json", error)

    def test_unserialisable_data_raises_and_leaves_an_old_report(self):
        path = self.p("r.json")
        touch(path, b"old")
        with self.assertRaises(TypeError):
            ro.write_json_report(path, {"x": object()})
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"old", "the file is opened only after the text is built")


class SharedCompareTests(unittest.TestCase):
    def test_uses_report_paths(self):
        import report_paths
        self.assertIs(ro.same_report_path, report_paths.same_report_path)


if __name__ == "__main__":
    unittest.main()
