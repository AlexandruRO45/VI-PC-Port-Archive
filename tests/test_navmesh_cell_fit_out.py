"""tools/navmesh_cell_fit.py: an unusable --out is one usage error, not a traceback. No game files."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import navmesh_cell_fit as cf  # noqa: E402


class OutGuards(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.archive = os.path.join(self.root, "empty")
        os.mkdir(self.archive)

    def call(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                cf.main([self.archive, *args])
        return caught.exception.code, out.getvalue(), err.getvalue()

    def test_out_directory_is_refused_before_scanning(self):
        code, out, err = self.call("--out", self.root)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("--out is a directory", err)
        self.assertNotIn("Traceback", err)

    def test_out_missing_parent_is_refused_before_scanning(self):
        missing = os.path.join(self.root, "no-such-dir", "report.json")
        code, out, err = self.call("--out", missing)
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("--out parent directory does not exist", err)
        self.assertNotIn("Traceback", err)

    def test_late_write_failure_is_one_error(self):
        target = os.path.join(self.root, "report.json")
        real_open = open

        def refusing_open(path, *args, **kwargs):
            if os.fspath(path) == target:
                raise OSError("permission denied")
            return real_open(path, *args, **kwargs)

        out, err = io.StringIO(), io.StringIO()
        with mock.patch("builtins.open", side_effect=refusing_open):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as caught:
                    cf.main([self.archive, "--out", target])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("cannot write --out", err.getvalue())
        self.assertIn("permission denied", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_writable_out_still_stores_the_summary(self):
        target = os.path.join(self.root, "report.json")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cf.main([self.archive, "--out", target])
        self.assertEqual(code, 0)
        self.assertEqual(err.getvalue(), "")
        self.assertIn('"meshes": 0', out.getvalue())
        with open(target, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["local_space_meshes"]["meshes"], 0)


if __name__ == "__main__":
    unittest.main()
