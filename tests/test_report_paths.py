"""tools/report_paths.py: the one `same_report_path` that the tools use before writing a report.

    python -I tests/test_report_paths.py

Runs in a temporary folder on hand-built files; no game data. It checks the helper itself (same file, hard link,
symlink, a path that does not exist yet, letter case on a volume that ignores it) and that the tools that used to carry
their own copy now use this one, so a fix here reaches all of them.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import report_paths  # noqa: E402

# Tools whose local copy of the comparison moved here (BUILD-589).
USERS = ("awc_manifest", "gamedata_index", "gamedata_xml", "navmesh_index", "rpf_probe", "tile_probe", "world_names")


def load(name):
    spec = importlib.util.spec_from_file_location(f"rp_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SameReportPathTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        probe = self.dir / "Probe"
        probe.touch()
        self.insensitive = (self.dir / "probe").exists()
        probe.unlink()

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_file_under_two_spellings(self):
        f = self.dir / "a.json"
        f.write_text("{}", encoding="utf-8")
        self.assertTrue(report_paths.same_report_path(str(f), str(f)))
        self.assertTrue(report_paths.same_report_path(str(f), str(self.dir / "." / "a.json")))
        self.assertTrue(report_paths.same_report_path(str(f), str(self.dir / "sub" / ".." / "a.json")))

    def test_different_files(self):
        a, b = self.dir / "a.json", self.dir / "b.json"
        a.write_text("{}", encoding="utf-8")
        b.write_text("{}", encoding="utf-8")
        self.assertFalse(report_paths.same_report_path(str(a), str(b)))

    def test_a_report_that_does_not_exist_yet(self):
        a = self.dir / "a.json"
        a.write_text("{}", encoding="utf-8")
        new = self.dir / "new" / "deeper" / "report.json"
        self.assertFalse(report_paths.same_report_path(str(new), str(a)))
        self.assertTrue(report_paths.same_report_path(str(new), str(new)))

    def test_hard_link_is_the_same_report(self):
        a, link = self.dir / "a.json", self.dir / "link.json"
        a.write_text("{}", encoding="utf-8")
        try:
            os.link(a, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("hard links not available here")
        self.assertTrue(report_paths.same_report_path(str(a), str(link)))
        self.assertTrue(report_paths.same_report_path(str(link), str(a)))

    def test_symlink_is_the_same_report(self):
        a, link = self.dir / "a.json", self.dir / "sym.json"
        a.write_text("{}", encoding="utf-8")
        try:
            os.symlink(a, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("symlinks not available here")
        self.assertTrue(report_paths.same_report_path(str(a), str(link)))

    def test_letter_case_follows_the_volume(self):
        upper, lower = self.dir / "new" / "B.json", self.dir / "new" / "b.json"
        self.assertEqual(report_paths.same_report_path(str(upper), str(lower)), self.insensitive)
        self.assertFalse(report_paths.same_report_path(str(upper), str(self.dir / "new" / "c.json")))

    def test_case_query_for_a_missing_folder_asks_its_first_existing_parent(self):
        missing = self.dir / "no" / "such" / "folder" / "r.json"
        self.assertEqual(report_paths.case_insensitive_volume(str(missing)),
                         sys.platform == "darwin" and self.insensitive)


class ToolsShareTheHelperTest(unittest.TestCase):
    def test_each_user_binds_the_shared_function(self):
        for name in USERS:
            with self.subTest(tool=name):
                try:
                    module = load(name)
                except ImportError as error:      # tile_probe needs numpy; the source check below still covers it
                    self.assertEqual(name, "tile_probe", error)
                    continue
                self.assertIs(module.same_report_path, report_paths.same_report_path)

    def test_no_user_keeps_its_own_copy(self):
        for name in USERS:
            with self.subTest(tool=name):
                source = (TOOLS / f"{name}.py").read_text(encoding="utf-8")
                self.assertNotIn("def same_report_path", source)
                self.assertNotIn("def _same_report_path", source)
                self.assertNotIn("def _path_key", source)
                self.assertNotIn("def _case_insensitive_volume", source)


if __name__ == "__main__":
    unittest.main()
