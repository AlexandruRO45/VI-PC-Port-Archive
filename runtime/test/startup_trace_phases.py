"""Exercise flushed lifecycle records through gta6rt with the synthetic command-line fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

RUNTIME, FIXTURE = sys.argv[1:]


class StartupPhases(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.trace = Path(temp.name) / "trace.jsonl"

    def command(self, *args):
        return [RUNTIME, "--run", FIXTURE, "--trace", str(self.trace), "--", *args]

    def events(self):
        return [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]

    def test_return(self):
        subprocess.run(self.command(), check=True, timeout=20)
        events = self.events()
        self.assertTrue(all(type(e["tid"]) is int and e["tid"] > 0 for e in events))
        for begin, end in [("tls_callback", "tls_callback_return"), ("dllmain_enter", "dllmain")]:
            entries = [e for e in events if e["kind"] == begin]
            self.assertTrue(entries)
            for entry in entries:
                matching = [e for e in events if e["kind"] == end and e["phase_id"] == entry["phase_id"]]
                self.assertEqual(len(matching), 1)
                self.assertEqual(matching[0]["tid"], entry["tid"])
                self.assertGreater(matching[0]["seq"], entry["seq"])
        game = next(e for e in events if e["kind"] == "game_entry")
        self.assertNotEqual(game["tid"], events[0]["tid"])
        self.assertIn(Path(FIXTURE).name, game["at"])
        self.assertEqual(events[-1]["code"], 0)

    def check_incomplete_attach(self, events):
        entry = [e for e in events if e["kind"] == "dllmain_enter"][-1]
        self.assertEqual(entry["module"], "tiny_startup_cmdline_dll.dll")
        self.assertEqual(entry["reason"], 1)
        self.assertFalse(any(e["kind"] == "dllmain" and e["phase_id"] == entry["phase_id"] for e in events))
        return entry

    def test_fault(self):
        result = subprocess.run(self.command("trace-fault"), timeout=20)
        self.assertEqual(result.returncode & 0xffffffff, 0xe0006378)
        events = self.events()
        entry = self.check_incomplete_attach(events)
        fault = next(e for e in events if e.get("code") == "0xe0006378")
        self.assertEqual(entry["tid"], fault["tid"])
        self.assertGreater(fault["seq"], entry["seq"])

    def test_wait(self):
        marker = self.trace.with_suffix(".stderr")
        with marker.open("wb") as err, subprocess.Popen(self.command("trace-wait"), stderr=err) as process:
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and process.poll() is None:
                    if b"inside synthetic DllMain" in marker.read_bytes():
                        break
                    time.sleep(0.05)
                else:
                    self.fail("callback did not reach its wait")
            finally:
                process.kill()
        self.check_incomplete_attach(self.events())


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])
