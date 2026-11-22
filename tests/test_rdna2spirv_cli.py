"""rdna2spirv.py command-line validation and SPIR-V output tests.

    python -I tests/test_rdna2spirv_cli.py

Standard library only: the output tests substitute the SPIR-V tools and use synthetic input.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
TOOL = os.path.join(ROOT, "tools", "rdna2spirv.py")

import rdna2spirv  # noqa: E402


class ArgumentTypes(unittest.TestCase):
    def test_limit(self):
        self.assertEqual(rdna2spirv._limit_arg("200"), 200)
        self.assertEqual(rdna2spirv._limit_arg("1"), 1)
        for text in ("0", "-1", "-200", "x", ""):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._limit_arg(text)

    def test_shard(self):
        self.assertEqual(rdna2spirv._shard_arg("0/8"), (0, 8))
        self.assertEqual(rdna2spirv._shard_arg("7/8"), (7, 8))
        self.assertEqual(rdna2spirv._shard_arg("0/1"), (0, 1))
        for text in ("0/0", "8/8", "9/8", "-1/8", "1/-8", "a/b", "0", "0/8/1", "", "/8"):
            with self.subTest(text=text), self.assertRaises(argparse.ArgumentTypeError):
                rdna2spirv._shard_arg(text)


class SourceDiscovery(unittest.TestCase):
    """The src argument is a .s file or a folder of .s files; other names are not inputs."""

    def run_main(self, src, out):
        def spirv_tool(cmd, **kwargs):
            tool = cmd[0]
            if tool == "spirv-as":
                Path(cmd[3]).write_bytes(b"new module")
            elif tool == "spirv-val":
                self.assertEqual(Path(cmd[-1]).read_bytes(), b"new module")
            else:
                self.fail(f"unexpected tool: {tool}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        output = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(src), "--out", str(out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), \
                redirect_stdout(output):
            rdna2spirv.main()
        return output.getvalue()

    def test_missing_src_is_refused_with_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "spirv")
            # The .s suffix matters: without isfile() a missing shader path would be trusted as a file.
            p = subprocess.run([sys.executable, "-I", TOOL, os.path.join(tmp, "absent.s"), "--out", out],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("not a .s file or a folder of .s files", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertFalse(os.path.exists(out))

    def test_directory_ending_in_s_as_src_is_a_directory(self):
        # A .s suffix alone is not a shader: an existing directory is still scanned as a folder.
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "shader.s")
            os.mkdir(src)
            p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", os.path.join(tmp, "spirv")],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertIn("0 shaders:", p.stdout)

    def test_non_file_non_dir_entry_is_skipped(self):
        # Covers entries that are neither files nor directories (broken symlinks, FIFOs, devices).
        # is_file() rejects them; `not is_dir()` would feed one to translate() and crash.
        class OddEntry:
            name = "foo.s"
            path = os.path.join("not-a-file", "foo.s")

            def is_file(self):
                return False

            def is_dir(self):
                return False

        class FakeScandir:
            def __enter__(self):
                return iter([OddEntry()])

            def __exit__(self, *exc):
                return False

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            output = io.StringIO()
            with mock.patch.object(sys, "argv", [TOOL, src, "--out", os.path.join(tmp, "spirv")]), \
                    mock.patch.object(rdna2spirv.os, "scandir", lambda path: FakeScandir()), \
                    redirect_stdout(output):
                rdna2spirv.main()
            self.assertIn("0 shaders:", output.getvalue())

    def test_directory_ending_in_s_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "disasm"
            src.mkdir()
            (src / "ps_ok.s").write_text("\ts_endpgm\n", encoding="utf-8")
            (root / "ps_ok.dxbc").write_bytes(b"DXBC" + bytes(28))
            (src / "nested.s").mkdir()  # a directory that looks like a shader
            out = root / "spirv"
            out.mkdir()
            stdout = self.run_main(src, out)
            self.assertIn("1 pass spirv-val", stdout)
            self.assertTrue((out / "ps_ok.spv").is_file())
            self.assertFalse((out / "nested.spv").exists())

    def test_non_s_file_as_src_is_refused_with_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "notes.txt")
            Path(src).write_text("not a shader", encoding="utf-8")
            p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", os.path.join(tmp, "spirv")],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)


class CommandLine(unittest.TestCase):
    def test_bad_limit_or_shard_is_refused_before_out_is_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "disasm")
            os.mkdir(src)
            with open(os.path.join(src, "ps_x.s"), "w", encoding="utf-8") as f:
                f.write("\ts_endpgm\n")
            for args, message in ((["--limit", "0"], "must be at least 1"),
                                  (["--limit", "-5"], "must be at least 1"),
                                  (["--shard", "0/0"], "need n >= 1"),
                                  (["--shard", "9/8"], "need n >= 1 and 0 <= i < n"),
                                  (["--shard", "a/b"], "expected i/n")):
                out = os.path.join(tmp, "spirv")
                with self.subTest(args=args):
                    p = subprocess.run([sys.executable, "-I", TOOL, src, "--out", out, *args],
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 2)
                    self.assertIn(message, p.stderr)
                    self.assertNotIn("Traceback", p.stderr)
                    self.assertFalse(os.path.exists(out))     # 9/8 used to write an empty index.part9.json


class MergeShards(unittest.TestCase):
    """--merge publishes index.json only from a complete --shard set (BUILD-501)."""

    def setUp(self):
        self.out = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def part(self, file_index, i, n, entries):
        (self.out / f"index.part{file_index}.json").write_text(
            json.dumps({"shard": [i, n], "index": entries}), encoding="utf-8")

    def merge(self):
        return subprocess.run([sys.executable, "-I", TOOL, str(self.out), "--out", str(self.out), "--merge"],
                              capture_output=True, text=True)

    def refused(self, message):
        p = self.merge()
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn(message, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertFalse((self.out / "index.json").exists())
        self.assertFalse((self.out / "stubs").exists())     # nothing is published before the check

    def test_a_complete_set_merges(self):
        self.part(0, 0, 2, {"cs_a": {"stage": "cs"}})
        self.part(1, 1, 2, {"cs_b": {"stage": "cs"}})
        p = self.merge()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("merged 2 entries", p.stdout)
        self.assertEqual(json.loads((self.out / "index.json").read_text(encoding="utf-8")),
                         {"cs_a": {"stage": "cs"}, "cs_b": {"stage": "cs"}})

    def test_a_missing_shard_is_refused(self):
        self.part(0, 0, 2, {"cs_a": {"stage": "cs"}})      # the finding's case: 1 of 2 parts merged as finished
        self.refused("1 of 2 shards missing (1)")

    def test_many_missing_shards_are_listed_briefly(self):
        self.part(0, 0, 100, {})
        self.refused("99 of 100 shards missing (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...)")

    def test_parts_from_another_shard_count_are_refused(self):
        self.part(0, 0, 2, {})
        self.part(1, 1, 2, {})
        self.part(2, 2, 4, {})                               # left over from an earlier 4-way run
        self.refused("parts from different --shard counts (2 of 2; 1 of 4)")

    def test_a_part_without_a_shard_record_is_refused(self):
        (self.out / "index.part0.json").write_text(json.dumps({"cs_a": {"stage": "cs"}}), encoding="utf-8")
        self.refused("index.part0.json has no shard record")

    def test_a_renamed_part_is_refused(self):
        self.part(0, 0, 2, {})
        self.part(1, 0, 2, {})                               # a copy of part 0 named as part 1
        self.refused("index.part1.json holds shard 0/2")

    def test_an_unreadable_part_is_refused(self):
        self.part(0, 0, 2, {})
        (self.out / "index.part1.json").write_text("{", encoding="utf-8")
        self.refused("index.part1.json: unreadable")

    def test_no_parts_is_refused(self):
        self.refused("no index.part*.json in")

    def test_shard_runs_record_their_place_and_merge(self):
        src = self.out / "disasm"
        src.mkdir()
        for name in ("ps_a", "ps_b", "ps_c"):
            (src / f"{name}.s").write_text("\ts_endpgm\n", encoding="utf-8")
            (self.out / f"{name}.dxbc").write_bytes(b"DXBC" + bytes(28))   # an empty signature, as above
        spirv = self.out / "spirv"

        def spirv_tool(cmd, **kwargs):
            if cmd[0] == "spirv-as":
                Path(cmd[3]).write_bytes(b"module")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        for shard in ("1/2", "0/2"):
            with mock.patch.object(sys, "argv", [TOOL, str(src), "--out", str(spirv), "--shard", shard]), \
                    mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), \
                    redirect_stdout(io.StringIO()):
                rdna2spirv.main()
        part1 = json.loads((spirv / "index.part1.json").read_text(encoding="utf-8"))
        self.assertEqual(part1["shard"], [1, 2])
        self.assertEqual(sorted(part1["index"]), ["ps_b"])
        self.assertEqual(sorted(rdna2spirv.merge_parts(str(spirv))), ["ps_a", "ps_b", "ps_c"])


class ValidatedOutput(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.src = root / "disasm" / "ps_fixture.s"
        self.src.parent.mkdir()
        self.src.write_text("\ts_endpgm\n", encoding="utf-8")
        (root / "ps_fixture.dxbc").write_bytes(b"DXBC" + bytes(28))
        self.out = root / "spirv"
        self.out.mkdir()
        self.module = self.out / "ps_fixture.spv"
        self.index = self.out / "index.json"

    def translate(self, *, reject_at=None):
        def spirv_tool(cmd, **kwargs):
            tool = cmd[0]
            if tool == "spirv-as":
                Path(cmd[3]).write_bytes(b"new module")
            elif tool == "spirv-val":
                self.assertEqual(Path(cmd[-1]).read_bytes(), b"new module")
                if reject_at is None:
                    self.assertFalse(self.module.exists(), "the binary was published before validation")
            else:
                self.fail(f"unexpected tool: {tool}")
            rejected = tool == reject_at
            return subprocess.CompletedProcess(cmd, int(rejected), "", "synthetic rejection" if rejected else "")

        output = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(self.src), "--out", str(self.out)]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=spirv_tool), redirect_stdout(output):
            rdna2spirv.main()
        return output.getvalue()

    def test_existing_single_s_file_still_translates(self):
        self.assertIn("1 pass spirv-val", self.translate())

    def test_validator_rejection_does_not_publish_binary(self):
        self.assertIn("INVALID: synthetic rejection", self.translate(reject_at="spirv-val"))
        self.assertFalse(self.module.exists())
        self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), {})
        self.assertTrue((self.out / "ps_fixture.spvasm").is_file())
        self.assertEqual(list(self.out.glob("*.tmp")), [])

    def test_rejection_clears_retried_shader_but_preserves_unrelated_output(self):
        other = self.out / "cs_other.spv"
        other.write_bytes(b"unrelated module")
        previous = {"cs_other": {"stage": "cs"}}
        for tool in ("spirv-as", "spirv-val"):
            with self.subTest(tool=tool):
                self.module.write_bytes(b"old module")
                self.index.write_text(json.dumps({**previous, "ps_fixture": {"stage": "ps"}}), encoding="utf-8")
                self.assertIn("INVALID: synthetic rejection", self.translate(reject_at=tool))
                self.assertFalse(self.module.exists())
                self.assertEqual(json.loads(self.index.read_text(encoding="utf-8")), previous)
                self.assertEqual(other.read_bytes(), b"unrelated module")

    def test_validation_success_publishes_binary_and_metadata(self):
        self.assertIn("1 pass spirv-val", self.translate())
        self.assertEqual(self.module.read_bytes(), b"new module")
        self.assertIn("ps_fixture", json.loads(self.index.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
