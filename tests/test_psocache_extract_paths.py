"""Regression tests for output path guards and tempdir containment in tools/psocache_extract.py.

Synthetic RPF9 archives only; no game data.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TESTS_DIR)
TOOLS_DIR = os.path.join(ROOT, "tools")
for _dir in (TESTS_DIR, TOOLS_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

import psocache_extract as pe  # noqa: E402
from test_psocache_extract import build_archive, build_cache  # noqa: E402

TOOL = os.path.join(TOOLS_DIR, "psocache_extract.py")


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-I", TOOL, *args], capture_output=True, text=True
    )


class PsocacheExtractOutputGuardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = build_cache()
        self.archive_bytes = build_archive(self.cache)
        self.folder = os.path.join(self.tmp.name, "pack")
        os.mkdir(self.folder)
        self.first = os.path.join(self.folder, "01.rpf")
        self.second = os.path.join(self.folder, "02.rpf")
        self.sentinel = b"SENTINEL_ARCHIVE_02"
        with open(self.first, "wb") as f:
            f.write(self.archive_bytes)
        with open(self.second, "wb") as f:
            f.write(self.sentinel)
        self.out = os.path.join(self.tmp.name, "cache.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def test_refuses_output_aliasing_other_archive_in_folder(self):
        proc = run_cli(self.folder, self.second)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("overwrite an input archive", proc.stderr)
        self.assertNotIn("02.rpf", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        with open(self.second, "rb") as f:
            self.assertEqual(f.read(), self.sentinel)

    def test_refuses_dotdot_alias_to_input_archive(self):
        aliased = os.path.join(self.folder, "..", "pack", "02.rpf")
        proc = run_cli(self.folder, aliased)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("overwrite an input archive", proc.stderr)
        self.assertNotIn("02.rpf", proc.stderr)
        with open(self.second, "rb") as f:
            self.assertEqual(f.read(), self.sentinel)

    def test_refuses_output_hardlink_to_input_archive(self):
        link = os.path.join(self.tmp.name, "hardlink.bin")
        try:
            os.link(self.second, link)
        except OSError as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        proc = run_cli(self.folder, link)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("overwrite an input archive", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        with open(self.second, "rb") as f:
            self.assertEqual(f.read(), self.sentinel)

    def test_refuses_single_archive_same_path_before_scanning(self):
        # Even with a tiny budget that would miss the cache, aliasing the input must be refused upfront.
        proc = run_cli(self.first, self.first, "--budget", "4")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("overwrite an input archive", proc.stderr)
        self.assertNotIn("01.rpf", proc.stderr)
        with open(self.first, "rb") as f:
            self.assertEqual(f.read(), self.archive_bytes)

    def test_refuses_case_variant_of_input_archive_on_case_folding_os(self):
        if os.path.normcase("A") == os.path.normcase("a"):
            variant = self.first.upper()
            proc = run_cli(self.first, variant)
            self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
            self.assertIn("overwrite an input archive", proc.stderr)
            self.assertNotIn("01.RPF", proc.stderr)
            with open(self.first, "rb") as f:
                self.assertEqual(f.read(), self.archive_bytes)
        else:
            self.skipTest("platform paths are case-sensitive")

    def test_refuses_existing_directory_output_before_scanning(self):
        outdir = os.path.join(self.tmp.name, "existing_dir")
        os.mkdir(outdir)
        proc = run_cli(self.first, outdir, "--budget", "4")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("directory", proc.stderr)
        self.assertNotIn("existing_dir", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_refuses_missing_parent_directory_before_scanning(self):
        missing_parent_out = os.path.join(self.tmp.name, "no_such_dir", "cache.bin")
        proc = run_cli(self.first, missing_parent_out, "--budget", "4")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("parent directory", proc.stderr)
        self.assertNotIn("no_such_dir", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_refuses_trailing_separator_output_before_scanning(self):
        trailing = os.path.join(self.tmp.name, "not_created_yet") + os.sep
        proc = run_cli(self.first, trailing, "--budget", "4")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("directory", proc.stderr)
        self.assertNotIn("not_created_yet", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_refuses_symlink_output_to_input_archive(self):
        link = os.path.join(self.tmp.name, "symlink_to_archive.bin")
        try:
            os.symlink(self.second, link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        proc = run_cli(self.folder, link)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("overwrite an input archive", proc.stderr)
        with open(self.second, "rb") as f:
            self.assertEqual(f.read(), self.sentinel)

    def test_inside_tempdir_handles_filesystem_root(self):
        root = os.path.splitdrive(os.path.abspath(self.tmp.name))[0] + os.sep
        candidate = os.path.join(root, "psocache_probe.bin")
        with mock.patch("tempfile.gettempdir", return_value=root):
            self.assertTrue(pe.inside_tempdir(candidate))
            self.assertFalse(pe.inside_tempdir(root))

    def test_inside_tempdir_rejects_other_drive_on_windows(self):
        if os.name != "nt":
            self.skipTest("Windows drive-letter check")
        drive = os.path.splitdrive(os.path.abspath(self.tmp.name))[0].upper()
        other = "Z:\\" if drive != "Z:" else "Y:\\"
        self.assertFalse(pe.inside_tempdir(os.path.join(other, "cache.bin")))

    def test_inside_tempdir_rejects_sibling_prefix_and_empty(self):
        base = os.path.join(self.tmp.name, "tmpbase")
        sibling = os.path.join(self.tmp.name, "tmpbase2", "cache.bin")
        os.mkdir(base)
        with mock.patch("tempfile.gettempdir", return_value=base):
            self.assertFalse(pe.inside_tempdir(sibling))
            self.assertFalse(pe.inside_tempdir(base))
            self.assertFalse(pe.inside_tempdir(""))

    def test_normal_extraction_and_reextraction_still_succeed(self):
        proc = run_cli(self.folder, self.out)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), self.cache)
        # Re-extracting to the same non-input output file continues to work.
        proc2 = run_cli(self.folder, self.out)
        self.assertEqual(proc2.returncode, 0, proc2.stdout + proc2.stderr)
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), self.cache)


if __name__ == "__main__":
    unittest.main()
