"""rdna2spirv.py --merge shard completeness and consistency tests.

    python -I tests/test_rdna2spirv_merge.py

Standard library only: synthetic inputs, no GPU, no Vulkan SDK, no llvm-mc, no game data.
"""
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
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
TOOL = os.path.join(ROOT, "tools", "rdna2spirv.py")

import make_game  # noqa: E402
import rdna2spirv  # noqa: E402


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _run_merge(out, *extra, src=None):
    return subprocess.run(
        [sys.executable, "-I", TOOL, str(src or out), "--out", str(out), "--merge", *extra],
        capture_output=True, text=True,
    )


class MergeCompletenessTest(unittest.TestCase):
    def test_complete_shard_set_merges_and_cleans_up_parts_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            for i, name in ((0, "ps_a"), (1, "ps_b")):
                _write_json(out / f"index.part{i}.json", {name: {"stage": "ps", "inputs": [0]}})
                _write_json(out / f"index.part{i}.meta.json", {"shard": i, "shards": 2})

            r = _run_merge(out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("merged 2 entries", r.stdout)

            merged = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(merged.keys()), ["ps_a", "ps_b"])
            self.assertTrue((out / "stubs" / "vs_0.spv").is_file())
            self.assertEqual([p.name for p in out.glob("index.part*")], [])

    def test_missing_shard_from_metadata_refuses_and_preserves_existing_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            existing = {"old_shader": {"stage": "cs"}}
            _write_json(out / "index.json", existing)
            _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})
            _write_json(out / "index.part0.meta.json", {"shard": 0, "shards": 2})

            r = _run_merge(out)
            self.assertNotEqual(r.returncode, 0)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("missing 1", r.stderr)
            # Existing index.json and finished part0 files stay intact for retry.
            self.assertEqual(json.loads((out / "index.json").read_text(encoding="utf-8")), existing)
            self.assertTrue((out / "index.part0.json").is_file())
            self.assertTrue((out / "index.part0.meta.json").is_file())

    def test_gap_in_parts_refused_with_and_without_metadata(self):
        for with_meta in (True, False):
            with self.subTest(with_meta=with_meta), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "spirv"
                out.mkdir()
                for i, name in ((0, "ps_a"), (2, "ps_c")):
                    _write_json(out / f"index.part{i}.json", {name: {"stage": "ps"}})
                    if with_meta:
                        _write_json(out / f"index.part{i}.meta.json", {"shard": i, "shards": 3})

                r = _run_merge(out)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("missing 1", r.stderr)
                self.assertFalse((out / "index.json").exists())

    def test_legacy_part1_without_part0_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            _write_json(out / "index.part1.json", {"ps_b": {"stage": "ps"}})

            r = _run_merge(out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing 0", r.stderr)
            self.assertFalse((out / "index.json").exists())

    def test_conflicting_shard_totals_across_parts_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})
            _write_json(out / "index.part0.meta.json", {"shard": 0, "shards": 2})
            _write_json(out / "index.part1.json", {"ps_b": {"stage": "ps"}})
            _write_json(out / "index.part1.meta.json", {"shard": 1, "shards": 4})

            r = _run_merge(out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("conflicting shard totals", r.stderr)
            self.assertFalse((out / "index.json").exists())

    def test_explicit_shards_flag_checks_completeness_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})

            r = _run_merge(out, "--shards", "2")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing 1", r.stderr)
            self.assertFalse((out / "index.json").exists())

            _write_json(out / "index.part1.json", {"ps_b": {"stage": "ps"}})
            r = _run_merge(out, "--shards", "2")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(json.loads((out / "index.json").read_text(encoding="utf-8"))), ["ps_a", "ps_b"])
            self.assertEqual([p.name for p in out.glob("index.part*")], [])

    def test_explicit_shards_conflicts_with_metadata_or_extra_parts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            for i in range(3):
                _write_json(out / f"index.part{i}.json", {f"ps_{i}": {"stage": "ps"}})

            # No metadata, but --shards 2 catches the stray part2.
            r = _run_merge(out, "--shards", "2")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("unexpected 2", r.stderr)
            self.assertFalse((out / "index.json").exists())

            # Metadata says 3, --shards says 2: conflicting counts.
            _write_json(out / "index.part0.meta.json", {"shard": 0, "shards": 3})
            r = _run_merge(out, "--shards", "2")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("conflicting shard counts", r.stderr)
            self.assertFalse((out / "index.json").exists())

    def test_shards_json_plan_enforces_job_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            _write_json(out / make_game.SHARD_PLAN, {"jobs": 3, "wave64_emulate": False})
            _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})
            _write_json(out / "index.part1.json", {"ps_b": {"stage": "ps"}})

            r = _run_merge(out)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing 2", r.stderr)
            self.assertFalse((out / "index.json").exists())

            _write_json(out / "index.part2.json", {"ps_c": {"stage": "ps"}})
            r = _run_merge(out)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(json.loads((out / "index.json").read_text(encoding="utf-8"))),
                             ["ps_a", "ps_b", "ps_c"])
            self.assertTrue((out / make_game.SHARD_PLAN).is_file())
            self.assertEqual([p.name for p in out.glob("index.part*")], [])

    def test_shards_json_conflict_with_explicit_shards_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            _write_json(out / make_game.SHARD_PLAN, {"jobs": 2})
            _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})
            _write_json(out / "index.part1.json", {"ps_b": {"stage": "ps"}})

            r = _run_merge(out, "--shards", "4")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("conflicting shard counts", r.stderr)
            self.assertFalse((out / "index.json").exists())

    def test_empty_or_missing_out_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            r = _run_merge(empty)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no index.part<N>.json files", r.stderr)
            self.assertFalse((empty / "index.json").exists())

            missing = Path(tmp) / "missing"
            r = _run_merge(missing)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("does not exist", r.stderr)
            self.assertFalse(missing.exists())

    def test_corrupt_plan_meta_or_part_is_refused_without_traceback(self):
        cases = [
            ("bad_plan", lambda out: (out / "shards.json").write_text("{", encoding="utf-8")),
            ("bad_plan_jobs", lambda out: _write_json(out / "shards.json", {"jobs": 0})),
            ("bad_meta_json", lambda out: (out / "index.part0.meta.json").write_text("{", encoding="utf-8")),
            ("bad_meta_shard_id", lambda out: _write_json(out / "index.part0.meta.json", {"shard": 1, "shards": 2})),
            ("bad_part_type", lambda out: _write_json(out / "index.part0.json", ["not", "a", "dict"])),
        ]
        for label, corrupt in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "spirv"
                out.mkdir()
                _write_json(out / "index.part0.json", {"ps_a": {"stage": "ps"}})
                corrupt(out)

                r = _run_merge(out)
                self.assertNotEqual(r.returncode, 0)
                self.assertNotIn("Traceback", r.stderr)
                self.assertFalse((out / "index.json").exists())

    def test_invalid_cli_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "spirv"
            out.mkdir()
            for args, expected in (
                (["--shards", "2"], "--shards requires --merge"),
                (["--merge", "--shard", "0/2"], "--merge cannot be combined with --shard"),
                (["--merge", "--shards", "0"], "must be at least 1"),
            ):
                with self.subTest(args=args):
                    r = subprocess.run([sys.executable, "-I", TOOL, str(out), "--out", str(out), *args],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 2)
                    self.assertIn(expected, r.stderr)
                    self.assertNotIn("Traceback", r.stderr)


class ShardAndMergeEndToEndTest(unittest.TestCase):
    """Run real rdna2spirv.main() for --shard i/n (with mocked spirv-as/spirv-val) and then --merge."""

    def _run_shard(self, src, out, shard_spec):
        def fake_spirv_tool(cmd, **kwargs):
            if cmd[0] == "spirv-as":
                Path(cmd[3]).write_bytes(b"fake spv")
            elif cmd[0] == "spirv-val":
                pass
            else:
                self.fail(f"unexpected subprocess: {cmd}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        buf = io.StringIO()
        with mock.patch.object(sys, "argv", [TOOL, str(src), "--out", str(out), "--shard", shard_spec]), \
                mock.patch.object(rdna2spirv.subprocess, "run", side_effect=fake_spirv_tool), \
                redirect_stdout(buf):
            rc = rdna2spirv.main()
        self.assertIn(rc, (None, 0))
        return buf.getvalue()

    def test_real_shard_writes_meta_and_merge_refuses_until_all_shards_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "disasm"
            src.mkdir()
            for name in ("ps_0", "ps_1"):
                (src / f"{name}.s").write_text("\ts_endpgm\n", encoding="utf-8")
                (root / f"{name}.dxbc").write_bytes(b"DXBC" + bytes(28))
            out = root / "spirv"

            self._run_shard(src, out, "0/2")
            self.assertTrue((out / "index.part0.json").is_file())
            self.assertEqual(json.loads((out / "index.part0.meta.json").read_text(encoding="utf-8")),
                             {"shard": 0, "shards": 2})

            # Partial merge after only 0/2 must fail and leave no index.json.
            r = _run_merge(out, src=src)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("missing 1", r.stderr)
            self.assertFalse((out / "index.json").exists())

            # Complete the second shard; merge now succeeds and removes part/meta files.
            self._run_shard(src, out, "1/2")
            r = _run_merge(out, src=src)
            self.assertEqual(r.returncode, 0, r.stderr)
            merged = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(merged.keys()), ["ps_0", "ps_1"])
            self.assertEqual([p.name for p in out.glob("index.part*")], [])

    def test_make_game_replan_cleans_up_meta_files_and_real_merge_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            # Pre-populate a stale 4-shard run with companion .meta.json files.
            _write_json(spirv / make_game.SHARD_PLAN, {"jobs": 4, "wave64_emulate": False,
                                                       "fallbacks": [], "translator": "old"})
            for i in range(4):
                _write_json(spirv / f"index.part{i}.json", {f"old_{i}": {"stage": "ps"}})
                _write_json(spirv / f"index.part{i}.meta.json", {"shard": i, "shards": 4})

            class FakeShardProc:
                def __init__(self, args):
                    spec = args[args.index("--shard") + 1]
                    self.i, self.n = (int(x) for x in spec.split("/"))

                def wait(self):
                    _write_json(spirv / f"index.part{self.i}.meta.json", {"shard": self.i, "shards": self.n})
                    _write_json(spirv / f"index.part{self.i}.json", {f"new_{self.i}": {"stage": "ps"}})
                    return 0

            def real_merge(script, *args):
                self.assertEqual(script, "rdna2spirv.py")
                buf = io.StringIO()
                with mock.patch.object(sys, "argv", [TOOL, *args]), redirect_stdout(buf):
                    rc = rdna2spirv.main()
                self.assertEqual(rc, 0, buf.getvalue())

            with mock.patch.object(make_game.subprocess, "Popen", side_effect=FakeShardProc), \
                    mock.patch.object(make_game, "py", side_effect=real_merge), \
                    mock.patch.object(make_game, "translator_digest", return_value="new"), \
                    mock.patch.object(make_game, "log", lambda *a: None):
                make_game.translate("disasm", str(spirv), jobs=2)

            merged = json.loads((spirv / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(merged.keys()), ["new_0", "new_1"])
            self.assertEqual([p.name for p in spirv.glob("index.part*")], [])


if __name__ == "__main__":
    unittest.main()
