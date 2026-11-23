"""Exercise guest-scene forwarding and persisted results without a native GPU run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class GuestSceneTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.cwd = Path(scratch.name)
        self.cli = SCRIPTS / "dev.py"
        self.producer = SCRIPTS / "run_guest_scene.py"

    def invoke(self, *argv):
        return subprocess.run(
            [sys.executable, "-I", "-B", *argv],
            cwd=self.cwd, capture_output=True, timeout=15,
        )

    def scene_args(self):
        # Python stands in for Wine and interprets our plain-text loader on all hosts.
        (self.cwd / "loader.txt").write_text(
            "import sys\nprint('synthetic failure', file=sys.stderr)\nsys.exit(23)\n",
            encoding="utf-8", newline="\n",
        )
        (self.cwd / "guest.txt").touch()
        (self.cwd / "shaders" / "spirv").mkdir(parents=True)
        return ["--wine", sys.executable, "--loader", "loader.txt", "--guest", "guest.txt",
                "--shaders", "shaders"]

    def test_help_without_output_directory(self):
        direct = self.invoke(self.producer, "--help")
        result = self.invoke(self.cli, "guest-scene", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"--negative-controls", result.stdout)
        self.assertEqual(result.stdout, direct.stdout)
        self.assertEqual(result.stderr, direct.stderr)

    def test_forwarding_and_required_output(self):
        scripts = self.cwd / "checkout with spaces" / "scripts"
        scripts.mkdir(parents=True)
        cli = scripts / "dev.py"
        shutil.copyfile(self.cli, cli)
        (scripts / "run_guest_scene.py").write_text(
            "import json, os, sys\n"
            "print(json.dumps([sys.argv[1:], os.getcwd(), sys.flags.isolated]))\n"
            "print('runner diagnostic', file=sys.stderr)\nsys.exit(19)\n",
            encoding="utf-8", newline="\n",
        )
        missing = self.invoke(cli, "guest-scene", "--loader", "loader with spaces")
        self.assertEqual(missing.returncode, 2)
        self.assertEqual(missing.stdout, b"")
        self.assertIn(b"--out", missing.stderr)
        argv = ["--loader", "loader with spaces", "--out", "r\u00e9sultat", "--negative-controls"]
        forwarded = self.invoke(cli, "guest-scene", *argv)
        self.assertEqual(forwarded.returncode, 19)
        args, cwd, isolated = json.loads(forwarded.stdout)
        self.assertEqual(args, argv)
        self.assertEqual(Path(cwd).resolve(), self.cwd.resolve())
        self.assertEqual(isolated, 1)
        self.assertEqual(forwarded.stderr.strip(), b"runner diagnostic")

    def test_failed_run_and_occupied_retry(self):
        args = self.scene_args()
        out = self.cwd / "r\u00e9sultat de test"
        out.mkdir()
        cmd = [self.cli, "guest-scene", *args, f"--out={out.name}"]
        failed = self.invoke(*cmd)
        self.assertEqual(failed.returncode, 1, failed.stderr)
        self.assertIn(b"FAIL:", failed.stdout)
        report = out / "report.json"
        contents = report.read_bytes()
        result = json.loads(contents)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["runs"]["full"]["returncode"], 23)
        self.assertEqual((out / "logs/full.stderr.log").read_bytes().strip(), b"synthetic failure")
        retry = self.invoke(*cmd)
        self.assertEqual(retry.returncode, 2)
        self.assertEqual(retry.stdout, b"")
        self.assertIn(b"empty", retry.stderr)
        self.assertEqual(hashlib.sha256(report.read_bytes()).digest(), hashlib.sha256(contents).digest())

    def test_output_creation_failure(self):
        args = self.scene_args()
        (self.cwd / "occupied").touch()
        output = Path("occupied") / "run"
        result = self.invoke(self.cli, "guest-scene", *args, "--out", str(output))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b"")
        self.assertTrue(result.stderr)
        self.assertFalse((self.cwd / output / "report.json").exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX process groups and flock")
    def test_wine_timeout_stops_children(self):
        import fcntl

        args = self.scene_args()
        (self.cwd / "loader.txt").write_text(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import fcntl, time; lock = open(\"child.lock\", \"w\"); "
            "fcntl.flock(lock, fcntl.LOCK_EX); print(\"child ready\", flush=True); time.sleep(5)'], "
            "stdout=subprocess.PIPE)\n"
            "print(child.stdout.readline().decode().strip(), flush=True)\n"
            "time.sleep(30)\n",
            encoding="utf-8", newline="\n",
        )
        runner = (
            "import runpy, sys; script = sys.argv.pop(1); "
            "main = runpy.run_path(script)['main']; main.__globals__['TIMEOUT'] = 2; sys.exit(main())"
        )
        result = self.invoke("-c", runner, self.producer, *args, "--out", "out")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn(b"timed out after 2s", result.stdout)
        out = self.cwd / "out"
        record = json.loads((out / "logs" / "full.json").read_bytes())
        self.assertEqual(record["outcome"], "timed_out")
        self.assertEqual((out / "logs" / "full.stdout.log").read_bytes().strip(), b"child ready")
        # The lock is released when the child exits, even if its PID is still a zombie.
        with (out / "full" / "child.lock").open("r+") as lock:
            deadline = time.monotonic() + 1
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        self.fail("Wine launcher child survived timeout")
                    time.sleep(0.02)


if __name__ == "__main__":
    unittest.main()
