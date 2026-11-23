"""Report publication through links, using the existing synthetic DDS fixture."""
from contextlib import redirect_stderr, redirect_stdout
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
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import mask_channels as tool
from test_mask_channels import H, W, image, rgba8_dds


class ReportLinkTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "synthetic.dds"
        self.source.write_bytes(rgba8_dds(W, H, image()))
        self.links = self.root / "links"
        self.targets = self.root / "reports"
        self.links.mkdir()
        self.targets.mkdir()
        self.target = self.targets / "mask.json"
        self.output = self.links / "mask.json"
        self.previous = b'{"previous": true}\n'

    def symlink(self, path, target):
        try:
            path.symlink_to(os.path.relpath(target, path.parent))
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"symlinks unavailable: {error}")

    def run_cli(self):
        return subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "mask_channels.py"),
             str(self.source), "--json", str(self.output)],
            capture_output=True, text=True, encoding="utf-8")

    def assert_report(self):
        report = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual((report["width"], report["height"]), (W, H))
        self.assertEqual(report["channels"]["R"]["thresholds"]["128"]["cover"], 0.25)
        self.assertEqual(sorted(p.name for p in self.targets.iterdir()), [self.target.name])

    def test_existing_symlink_updates_target_and_keeps_link(self):
        self.target.write_bytes(self.previous)
        self.symlink(self.output, self.target)
        link_text = os.readlink(self.output)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(os.readlink(self.output), link_text)
        self.assert_report()

    def test_dangling_symlink_creates_target(self):
        self.symlink(self.output, self.target)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_symlink())
        self.assert_report()

    def test_symlink_chain_updates_final_target(self):
        self.target.write_bytes(self.previous)
        intermediate = self.links / "intermediate.json"
        self.symlink(intermediate, self.target)
        self.symlink(self.output, intermediate)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.output.is_symlink())
        self.assertTrue(intermediate.is_symlink())
        self.assert_report()

    def test_failed_publication_keeps_link_target_and_cleans_staging(self):
        self.target.write_bytes(self.previous)
        self.symlink(self.output, self.target)
        error = io.StringIO()
        with mock.patch.object(tool.os, "replace", side_effect=OSError("synthetic publish failure")), \
                redirect_stdout(io.StringIO()), redirect_stderr(error):
            self.assertEqual(tool.main([str(self.source), "--json", str(self.output)]), 2)
        self.assertEqual(len(error.getvalue().splitlines()), 1)
        self.assertTrue(self.output.is_symlink())
        self.assertEqual(self.target.read_bytes(), self.previous)
        self.assertEqual(list(self.targets.iterdir()), [self.target])
        self.assertEqual(list(self.links.iterdir()), [self.output])

    def test_hard_link_keeps_atomic_destination_name_semantics(self):
        self.target.write_bytes(self.previous)
        try:
            os.link(self.target, self.output)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"hard links unavailable: {error}")
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.target.read_bytes(), self.previous)
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8"))["width"], W)
        self.assertFalse(os.path.samefile(self.target, self.output))

    def test_destination_is_captured_before_decoding(self):
        self.target.write_bytes(self.previous)
        self.symlink(self.output, self.target)
        original = self.source.read_bytes()
        decode = tool.dds_png.decode_dds

        def retarget_then_decode(data):
            self.output.unlink()
            self.output.symlink_to(self.source)
            return decode(data)

        with mock.patch.object(tool.dds_png, "decode_dds", side_effect=retarget_then_decode), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(tool.main([str(self.source), "--json", str(self.output)]), 0)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertTrue(self.output.is_symlink())
        self.assertTrue(os.path.samefile(self.output, self.source))
        self.assert_report()


if __name__ == "__main__":
    unittest.main()
