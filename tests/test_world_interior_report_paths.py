"""Directory report aliases must not replace synthetic interior resources.

    python -I tests/test_world_interior_report_paths.py
"""
from contextlib import redirect_stderr, redirect_stdout
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("interior_fixtures", ROOT / "tests/test_world_interior_names.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
T = F.T


class DirectoryReportPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "carved"
        (self.folder / "nested").mkdir(parents=True)
        self.inputs = {self.folder / "interiors.bin": F.interiors(),
                       self.folder / "nested/placements.bin": F.placements()}
        for path, data in self.inputs.items():
            path.write_bytes(data)
        self.names = self.root / "names.json"
        self.names.write_text(json.dumps(F.NAMES), encoding="utf-8", newline="\n")
        self.names_before = self.names.read_bytes()

    def args(self, report):
        return [str(self.folder), "--names", str(self.names), "--out", str(report)]

    def cli(self, report):
        return subprocess.run([sys.executable, "-I", str(ROOT / "tools/world_interior_names.py"),
                               *self.args(report)], capture_output=True, text=True, timeout=30)

    def assert_inputs_unchanged(self):
        for path, data in self.inputs.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertEqual(self.names.read_bytes(), self.names_before)

    def test_cli_refuses_external_hard_links_to_flat_and_nested_inputs(self):
        for index, (source, data) in enumerate(self.inputs.items()):
            with self.subTest(source=source.relative_to(self.folder)):
                report = self.root / f"outside-{index}.json"
                os.link(source, report)
                result = self.cli(report)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("overwrite the input", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(report.read_bytes(), data)
                self.assert_inputs_unchanged()

    def test_alias_is_refused_before_scanning(self):
        source = next(iter(self.inputs))
        report = self.root / "outside.json"
        os.link(source, report)
        err = io.StringIO()
        with mock.patch.object(T, "scan", side_effect=AssertionError("scan ran before the refusal")), \
                redirect_stderr(err), self.assertRaises(SystemExit) as raised:
            T.main(self.args(report))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("overwrite the input", err.getvalue())
        self.assert_inputs_unchanged()

    def test_existing_independent_report_with_identical_bytes_is_allowed(self):
        report = self.root / "outside.json"
        report.write_bytes(next(iter(self.inputs.values())))
        result = self.cli(report)
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual((doc["summary"]["interior_names"], doc["summary"]["placements"]), (2, 3))
        self.assert_inputs_unchanged()

    def test_new_report_in_a_new_parent_is_allowed(self):
        report = self.root / "reports/new/interiors.json"
        result = self.cli(report)
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual((doc["summary"]["interior_names"], doc["summary"]["placements"]), (2, 3))
        self.assert_inputs_unchanged()

    def test_failed_directory_check_preserves_prior_report(self):
        report = self.root / "outside.json"
        previous = b"previous report\n"
        report.write_bytes(previous)
        err = io.StringIO()
        with mock.patch.object(T.WI, "_directory_files", side_effect=OSError(errno.EACCES, "denied", str(self.folder))), \
                mock.patch.object(T, "scan", side_effect=AssertionError("scan ran before the refusal")), \
                redirect_stderr(err), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            T.main(self.args(report))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("cannot check --out", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())
        self.assertEqual(report.read_bytes(), previous)
        self.assert_inputs_unchanged()


if __name__ == "__main__":
    unittest.main()
