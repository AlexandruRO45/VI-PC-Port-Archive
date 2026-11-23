"""CTest completeness and observed runtime results from synthetic dashboard inputs."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import progress_dashboard as dashboard  # noqa: E402


START = "Start testing: Nov 23 12:00 UTC\n"
END = "End testing: Nov 23 12:01 UTC\n"
FIRST = "1/2 Testing: first\n1/2 Test: first\nTest time = 0.02 sec\nTest Passed.\n"
SECOND = "2/2 Testing: second\n2/2 Test: second\nOutput:\n" + "-" * 58 + "\n"
VERDICT = "<end of output>\nTest time = 0.03 sec\nTest Failed.\n"
NESTED = START + "1/1 Test: inner\nTest Passed.\n" + END


class InterruptedCTest(unittest.TestCase):
    def check_report(self, log, expected_source, verdicts):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            tools = root / "tools"
            tools.mkdir(parents=True)
            for name in ("progress_dashboard.py", "progress_dashboard.html", "file_mode.py"):
                shutil.copyfile(Path(dashboard.__file__).with_name(name), tools / name)
            saved = root / "runtime/build/Testing/Temporary/LastTest.log"
            saved.parent.mkdir(parents=True)
            saved.write_text(log, encoding="utf-8", newline="\n")
            junit = root / "independent.xml"
            junit.write_text('<testsuite><testcase name="junit_control"/></testsuite>', encoding="utf-8", newline="\n")
            out = Path(tmp) / "out"
            result = subprocess.run([sys.executable, "-I", "-B", str(tools / "progress_dashboard.py"),
                                     "--out", str(out), "--junit", str(junit)], cwd=tmp, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            snapshot_path, = (out / "snapshots").glob("*.json")
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(dashboard.parse_lasttest(log), verdicts)
        sources = {s["id"]: s for s in snapshot["sources"]}
        source = sources["ctest_lasttest"]
        self.assertEqual((source["status"], source["records"]), (expected_source, len(verdicts)))
        if expected_source == "incomplete":
            self.assertTrue(source.get("note"))
        self.assertEqual((sources["junit"]["status"], sources["junit"]["records"]), ("ok", 1))
        expected = {"junit_control": "verified"}
        expected.update({name: {"pass": "verified", "fail": "failed"}[result] for name, result, _ in verdicts})
        observed = [(c["name"], c["status"]) for c in snapshot["components"] if c["panel"] == "runtime"]
        self.assertCountEqual(observed, expected.items())
        metric = next(m for m in snapshot["metrics"] if m["id"] == "runtime.current")
        self.assertEqual((metric["numerator"], metric["denominator"]),
                         (sum(value == "verified" for value in expected.values()), len(expected)))

    def test_interrupted_outer_test_preserves_observed_results(self):
        for label, tail in (("testing header only", "2/2 Testing: second\n"),
                            ("captured output", SECOND + "partial output\n"),
                            ("nested footer in captured output", SECOND + NESTED)):
            with self.subTest(cut=label):
                self.check_report(START + FIRST + tail, "incomplete", [("first", "pass", 0.02)])

    def test_completed_logs_keep_their_results(self):
        complete = [("first", "pass", 0.02), ("second", "fail", 0.03)]
        for label, log, verdicts in (("complete with nested output", START + FIRST + SECOND + NESTED + VERDICT + END, complete),
                                     ("completed test without footer", FIRST, [("first", "pass", 0.02)]),
                                     ("empty complete run", START + END, [])):
            with self.subTest(log=label):
                self.check_report(log, "ok", verdicts)


if __name__ == "__main__":
    unittest.main()
