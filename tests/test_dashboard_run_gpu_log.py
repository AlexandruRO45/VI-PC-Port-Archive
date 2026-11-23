"""Exercise saved GPU runner logs through the dashboard CLI without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DASHBOARD = Path(__file__).resolve().parents[1] / "tools" / "progress_dashboard.py"
HEADER = "=== GPU TEST RESULTS ===\n"


class DashboardGpuLogs(unittest.TestCase):
    def snapshot(self, logs):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = [sys.executable, "-I", "-B", DASHBOARD, "--out", "output", "--gpu-log"]
            for label, text in logs.items():
                path = root / (label + ".log")
                path.write_text(text, encoding="utf-8", newline="\n")
                args.append(f"{path.name}={label}")
            proc = subprocess.run(args, cwd=root, capture_output=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            snapshot, = (root / "output" / "snapshots").glob("*.json")
            return json.loads(snapshot.read_bytes())

    def test_runner_and_legacy_results_share_a_snapshot(self):
        logs = {
            "runner": "=== rdna2spirv.py ===\n  PASS  translation_only\n" + HEADER
                      + "  PASS  cs_ok   exact\n  FAIL  cs_bad   mismatch\n"
                        "  SKIP  cs_skip   subgroup unavailable\n1/3 passed, 1 skipped\n",
            "no-skips": HEADER + "  PASS  cs_second   exact\n1/1 passed\n",
            "legacy": f"{'cs_original':20} ok\n1 of 1 pass, 0 skipped\n",
        }
        snap = self.snapshot(logs)
        sources = {Path(s["path"]).stem: (s["status"], s["records"])
                   for s in snap["sources"] if s["id"] == "gpu_log"}
        self.assertEqual(sources, {"runner": ("ok", 3), "no-skips": ("ok", 1), "legacy": ("ok", 1)})
        outcomes = {(c["group"], c["name"]): c for c in snap["components"] if c["panel"] == "gpu"}
        self.assertEqual(set(outcomes), {("runner", "cs_ok"), ("runner", "cs_bad"), ("runner", "cs_skip"),
                                         ("no-skips", "cs_second"), ("legacy", "cs_original")})
        for key in (("runner", "cs_ok"), ("no-skips", "cs_second"), ("legacy", "cs_original")):
            self.assertIn(outcomes[key]["status"], ("verified", "partial"))
        self.assertEqual(outcomes["runner", "cs_bad"]["status"], "failed")
        self.assertIn("mismatch", outcomes["runner", "cs_bad"]["detail"])
        self.assertEqual(outcomes["runner", "cs_skip"]["status"], "unknown")
        metric = next(m for m in snap["metrics"] if m["id"] == "gpu.current")
        self.assertEqual(metric["denominator"], 4)

    def test_incomplete_or_inconsistent_logs_supply_no_results(self):
        snap = self.snapshot({
            "summary": "1/1 passed\n",
            "interrupted": HEADER + "  PASS  cs_only\n",
            "inconsistent": HEADER + "  FAIL  cs_only   mismatch\n1/1 passed\n",
        })
        sources = [s for s in snap["sources"] if s["id"] == "gpu_log"]
        self.assertEqual(len(sources), 3)
        self.assertTrue(all(s["status"] == "invalid" and s["records"] == 0 for s in sources), sources)
        self.assertFalse(any(c["panel"] == "gpu" for c in snap["components"]))


if __name__ == "__main__":
    unittest.main()
