"""Exercise archive selection and report preservation through the tex_headers CLI."""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
TOOL = TESTS.parent / "tools" / "tex_headers.py"
sys.path.insert(0, str(TESTS))
from test_tex_headers_report_paths import make_carve


class ArchiveSelectionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="texture headers ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.carve, self.cpu, _ = make_carve(self.root)
        self.common = self.carve / "common"

    def run_cli(self, output, selection=None):
        args = [sys.executable, "-I", str(TOOL), str(self.carve), "--out", str(output)]
        if selection is not None:
            args.extend(["--archives", selection])
        return subprocess.run(args, cwd=self.root, capture_output=True, timeout=30)

    def check_rejected(self, selection):
        for existing in (False, True):
            with self.subTest(selection=selection, existing=existing):
                report = self.root / ("existing.json" if existing else "absent.json")
                sentinel = b'{"previous":true}\r\n'
                if existing:
                    report.write_bytes(sentinel)
                result = self.run_cli(report, selection)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, b"")
                diagnostic = result.stderr.splitlines()[-1]
                self.assertIn(b"--archives", diagnostic)
                for name in selection.split(","):
                    self.assertIn(name.encode("ascii"), diagnostic)
                if existing:
                    self.assertEqual(report.read_bytes(), sentinel)
                else:
                    self.assertFalse(report.exists())

    def test_exact_duplicate(self):
        self.check_rejected("common,common")

    def test_duplicate_is_rejected_before_reading_the_index(self):
        (self.common / "index.json").write_bytes(b"invalid JSON")
        self.check_rejected("common,./common")

    def test_case_alias(self):
        upper = self.carve / "COMMON"
        if not upper.exists() or not upper.samefile(self.common):
            self.skipTest("COMMON does not resolve to common in this directory")
        self.check_rejected("common,COMMON")
        self.check_rows("COMMON", [("COMMON", 512)])

    def check_rows(self, selection, expected):
        report = self.root / "valid.json"
        result = self.run_cli(report, selection)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(report.read_bytes())
        self.assertEqual([(row["archive"], row["w"]) for row in rows], expected)

    def test_distinct_directories_keep_selection_order_and_header_data(self):
        first = self.common.rename(self.carve / "strasse")
        # These names casefold alike but refer to distinct directories on NTFS.
        second = self.carve / "stra\u00dfe"
        if second.exists():
            second = self.carve / "second"
        shutil.copytree(first, second)
        self.assertFalse(first.samefile(second))
        # Give the second synthetic BC1 descriptor its own width.
        cpu = second / self.cpu.name
        data = cpu.read_bytes()
        descriptor = struct.pack("<HHHBB", 512, 256, 1, 0x42, 71)
        self.assertEqual(data.count(descriptor), 1)
        cpu.write_bytes(data.replace(descriptor, struct.pack("<HHHBB", 256, 256, 1, 0x42, 71)))
        ordered = [(first.name, 512), (second.name, 256)]
        self.check_rows(first.name, ordered[:1])
        self.check_rows(f"{second.name},{first.name}", ordered[::-1])
        self.check_rows(None, sorted(ordered))

    def test_missing_index_keeps_its_diagnostic(self):
        report = self.root / "missing.json"
        result = self.run_cli(report, "common,missing")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(b"no index.json for missing", result.stderr)
        self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
