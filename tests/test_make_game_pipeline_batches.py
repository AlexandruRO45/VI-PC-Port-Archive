"""Pipeline batching and memory-crash recovery through the real make_game.pipelines orchestration.

The child is a small scripted checker speaking the production CLI/file protocol. Each invocation is a real
process, with synthetic index entries and controlled exits; no GPU, compiler, game data or packages are needed.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_game  # noqa: E402


CHECKER = r'''
import json
import os
from pathlib import Path
import signal
import sys

spirv, cache, *args = sys.argv[1:]
spirv, cache = Path(spirv), Path(cache)
config = json.loads((spirv / "control.json").read_text(encoding="utf-8"))
names = sorted(json.loads((spirv / "index.json").read_text(encoding="utf-8")), key=lambda n: n.encode("utf-8"))
def option(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
start = int(option("--start", 0))
count = int(option("--count", len(names)))
selected = names[start:start + count]
skip_path = option("--skip-file")
skipped = Path(skip_path).read_text(encoding="utf-8").splitlines() if skip_path else []
active = [name for name in selected if name not in skipped]
inflight = option("--inflight-file")
def record(event):
    with (spirv / "calls.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(event=event, pid=os.getpid(), selected=selected, active=active, args=args)) + "\n")
record("started")
if config.get("interrupt_while_saving"):
    Path(f"{cache}.tmp.{os.getpid()}.0").write_text("partial", encoding="utf-8")
if config.get("setup_error") or config.get("probe_setup_error") and count == 1 and option("--jobs") == "1":
    print("no Vulkan physical devices found", file=sys.stderr)
    sys.exit(config.get("setup_exit", 2))
if config.get("interrupt") or config.get("interrupt_probe") and count == 1 and option("--jobs") == "1":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.ExitProcess(0xC000013A)
    else:
        os.kill(os.getpid(), signal.SIGTERM)
if config.get("crash_before_work"):
    sys.exit(70)
if config.get("probe_crash_before_work") and count == 1 and option("--jobs") == "1":
    report = config.get("probe_report", "missing")
    if inflight and report != "missing":
        Path(inflight).write_text("unrelated\tpipeline\n" if report == "other" else "",
                                  encoding="utf-8", newline="\n")
    sys.exit(70)
crasher = config.get("crasher")
if crasher in active or len(active) > config.get("max_batch", 1000000):
    if inflight:
        suspects = [crasher] if crasher in active else active[:config.get("suspects", 2)]
        Path(inflight).write_text("".join(n + "\tpipeline\n" for n in suspects), encoding="utf-8", newline="\n")
    sys.exit(70)
# This stand-in's cache contains completed names so the test can detect lost checkpoints and repeated work.
completed = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
for i, arg in enumerate(args):
    if arg == "--merge-cache":
        completed += json.loads(Path(args[i + 1]).read_text(encoding="utf-8"))
cache.write_text(json.dumps(completed + active), encoding="utf-8", newline="\n")
record("completed")
if inflight:
    Path(inflight).write_text("", encoding="utf-8", newline="\n")
sys.exit(1 if skipped and set(selected).intersection(skipped) or config.get("invalid") in active else 0)
'''


class PipelineBatchesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pipeline batches ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "work"
        self.spirv = self.work / "shaders" / "spirv"
        self.spirv.mkdir(parents=True)
        self.out = self.root / "out"
        self.names = [f"m_{i:02d}" for i in range(7)]
        self.write_index(self.names)
        self.child = self.root / "checker.py"
        self.child.write_text(CHECKER, encoding="utf-8", newline="\n")
        self.commands = []

    def write_index(self, names):
        (self.spirv / "index.json").write_text(json.dumps({n: {} for n in reversed(names)}),
                                                encoding="utf-8", newline="\n")

    def invoke(self, control=None, range_support=True, merge_support=True):
        (self.spirv / "control.json").write_text(json.dumps(control or {}), encoding="utf-8", newline="\n")
        def run(args, hint="", check=True, **kwargs):
            args = [str(a) for a in args]
            self.commands.append(args)
            done = subprocess.run([sys.executable, "-I", str(self.child), *args[1:]],
                                  capture_output=True, text=True, timeout=15)
            if check and done.returncode:
                raise SystemExit(1)
            return done.returncode
        def supports(exe, *options):
            return (range_support or "--start" not in options) and (merge_support or "--merge-cache" not in options)
        logs, err = [], io.StringIO()
        # create=True keeps the regression runnable on the old implementation, which lacks this constant.
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="synthetic-checker"), \
                mock.patch.object(make_game, "checker_supports", side_effect=supports), \
                mock.patch.object(make_game, "PIPELINE_BATCH_SIZE", 3, create=True), \
                mock.patch.object(make_game, "run", side_effect=run), \
                mock.patch.object(make_game, "log", side_effect=logs.append), contextlib.redirect_stderr(err):
            try:
                make_game.pipelines(str(self.work), str(self.out), jobs=2)
            except SystemExit as exc:
                return exc.code, logs, err.getvalue()
        return 0, logs, err.getvalue()

    def calls(self, event=None):
        path = self.spirv / "calls.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return [row for row in rows if event is None or row["event"] == event]

    def test_bounded_processes_cover_the_sorted_index_and_tail(self):
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        completed = [r for r in self.calls("completed") if "--merge-cache" not in r["args"]]
        self.assertEqual([r["selected"] for r in completed], [self.names[:3], self.names[3:6], self.names[6:]])
        self.assertEqual(len({r["pid"] for r in completed}), 3)
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), self.names)
        self.assertTrue(all(r["args"][r["args"].index("--jobs") + 1] == "2" for r in completed))

    def test_collective_memory_crash_splits_without_skipping_healthy_modules(self):
        code, logs, err = self.invoke({"max_batch": 2})
        self.assertEqual(code, 0, err)
        # Diagnostic one-module probes are also successful; the final bounded work must still cover every name.
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names))
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertTrue(any("smaller" in line for line in logs), logs)

    def test_a_real_crasher_stays_failed_when_later_batches_pass(self):
        code, _, err = self.invoke({"crasher": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual((self.out / "pipeline_crashes.txt").read_text(encoding="utf-8"), "m_01\n")
        self.assertIn("m_01", err)
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names) - {"m_01"})

    def test_one_inflight_module_is_not_blacklisted_when_it_survives_alone(self):
        self.write_index(self.names[:2])   # small corpora need the same protection as multi-batch runs
        code, _, err = self.invoke({"max_batch": 1, "suspects": 1})
        self.assertEqual(code, 0, err)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertEqual({n for row in self.calls("completed") for n in row["active"]}, set(self.names[:2]))

    def test_interrupting_a_diagnostic_probe_stops_before_more_batches(self):
        code, _, _ = self.invoke({"max_batch": 2, "interrupt_probe": True})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # the failed batch and the interrupted diagnostic probe
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_a_validation_failure_does_not_hide_the_remaining_batches(self):
        code, _, _ = self.invoke({"invalid": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual([n for row in self.calls("completed") for n in row["active"]], self.names)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_setup_errors_and_interrupts_stop_immediately(self):
        for control in ({"setup_error": True}, {"interrupt": True}):
            with self.subTest(control=control):
                trace = self.spirv / "calls.jsonl"
                if trace.exists():
                    trace.unlink()
                self.commands.clear()
                code, _, _ = self.invoke(control)
                self.assertNotEqual(code, 0)
                self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_stops_instead_of_checking_every_range(self):
        code, _, _ = self.invoke({"setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_in_a_probe_stops_before_splitting(self):
        code, _, _ = self.invoke({"max_batch": 2, "probe_setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def assert_inconclusive_probe_stops(self, report):
        code, logs, err = self.invoke({"max_batch": 2, "probe_crash_before_work": True,
                                      "probe_report": report})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # failed batch, then the first inconclusive probe
        self.assertEqual(self.calls()[-1]["selected"], [self.names[0]])
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertFalse(any("took vk_pipeline_check down" in line for line in logs), logs)
        self.assertIn("inconclusive", err)

    def test_abnormal_probe_without_a_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("missing")

    def test_abnormal_probe_with_an_empty_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("empty")

    def test_abnormal_probe_reporting_another_module_does_not_blacklist_the_selected_one(self):
        self.assert_inconclusive_probe_stops("other")

    def test_a_crash_before_work_cannot_retry_forever_or_claim_success(self):
        code, _, err = self.invoke({"crash_before_work": True})
        self.assertNotEqual(code, 0)
        self.assertLessEqual(len(self.calls()), 4)
        self.assertTrue(err)

    def test_an_older_checker_gets_no_unsupported_range_arguments(self):
        code, _, err = self.invoke(range_support=False)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn("--start", self.commands[0])
        self.assertNotIn("--count", self.commands[0])

    def test_small_indices_keep_the_single_run(self):
        self.write_index(self.names[:2])
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.commands[0][2], str(self.out / "pipeline_cache.bin"))
        self.assertFalse((self.out / "pipeline_cache.parts").exists())

    def batch_and_merge_commands(self):
        merges = [c for c in self.commands if "--merge-cache" in c]
        return [c for c in self.commands if c not in merges], merges

    def test_each_batch_keeps_its_own_cache_part_and_the_parts_are_merged_once(self):
        code, logs, err = self.invoke()
        self.assertEqual(code, 0, err)
        batches, merges = self.batch_and_merge_commands()
        parts = [c[2] for c in batches]
        self.assertEqual(len(set(parts)), 3)
        self.assertTrue(all(Path(p).parent == self.out / "pipeline_cache.parts" for p in parts), parts)
        self.assertEqual(len(merges), 1)
        merge = merges[0]
        self.assertEqual([merge[i + 1] for i, a in enumerate(merge) if a == "--merge-cache"], parts)
        self.assertEqual(merge[merge.index("--start") + 1], str(len(self.names)))   # checks no shader itself
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), self.names)
        self.assertTrue(all(Path(p).exists() for p in parts))   # they seed the same ranges on the next run
        self.assertFalse((self.out / "pipeline_cache.bin.merging").exists())
        self.assertTrue(any("merging 3 batch cache(s)" in line for line in logs), logs)

    def test_a_rerun_seeds_each_batch_from_its_own_part(self):
        self.assertEqual(self.invoke()[0], 0)
        first = [c[2] for c in self.batch_and_merge_commands()[0]]
        self.commands.clear()
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual([c[2] for c in self.batch_and_merge_commands()[0]], first)

    def test_parts_of_an_earlier_batch_layout_are_removed_after_the_merge(self):
        parts = self.out / "pipeline_cache.parts"
        parts.mkdir(parents=True)
        stale = parts / "000000-000500.bin"
        stale.write_text(json.dumps(["old"]), encoding="utf-8", newline="\n")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertFalse(stale.exists())
        self.assertNotIn("old", json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")))

    def test_an_interrupted_run_keeps_the_previous_cache(self):
        self.out.mkdir()
        (self.out / "pipeline_cache.bin").write_text(json.dumps(["previous"]), encoding="utf-8", newline="\n")
        code, _, _ = self.invoke({"interrupt": True})
        self.assertNotEqual(code, 0)
        self.assertEqual(self.batch_and_merge_commands()[1], [])
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), ["previous"])

    def test_staging_files_a_stopped_checker_left_beside_the_parts_are_removed(self):
        parts = self.out / "pipeline_cache.parts"
        parts.mkdir(parents=True)
        staged = [parts / "000000-000003.bin.tmp.1234.0", self.out / "pipeline_cache.bin.merging.tmp.1234.0"]
        for p in staged:
            p.write_text("partial", encoding="utf-8")
        unrelated = parts / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual([p for p in staged if p.exists()], [])
        self.assertTrue(unrelated.exists())
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), self.names)

    def test_the_part_staging_file_of_an_interrupted_batch_is_removed(self):
        code, _, _ = self.invoke({"interrupt": True, "interrupt_while_saving": True})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls("started")), 1)
        self.assertEqual(list((self.out / "pipeline_cache.parts").iterdir()), [])

    def test_a_checker_without_merge_cache_seeds_every_batch_with_the_whole_cache(self):
        code, _, err = self.invoke(merge_support=False)
        self.assertEqual(code, 0, err)
        self.assertEqual({c[2] for c in self.commands}, {str(self.out / "pipeline_cache.bin")})
        self.assertEqual(len(self.commands), 3)


if __name__ == "__main__":
    unittest.main()
