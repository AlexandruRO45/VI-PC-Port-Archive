"""Exercise the real CLI stage selection and locking in separate synthetic processes."""

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = """
import sys
from pathlib import Path
repo, ready, hold, *cli = sys.argv[1:]
sys.path.insert(0, repo)
import make_game

def fake_stage(*args, **kwargs):
    Path(ready).touch()
    if hold == "yes":
        sys.stdin.buffer.read(1)

make_game.check = lambda *args: {}
make_game.preflight = lambda *args, **kwargs: None
make_game.below_normal = lambda: None
make_game.ROOT = str(Path(ready).parent / "synthetic checkout")
make_game.assets = make_game.runtime = make_game.pipelines = fake_stage
sys.argv = ["make_game.py", *cli]
sys.exit(make_game.main())
"""


class BuildLockTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.game = self.root / "synthetic input"
        self.game.mkdir()

    def command(self, stages, out, work, ready, hold=False, prebuilt=True):
        args = [sys.executable, "-I", "-c", DRIVER, str(ROOT), str(ready), "yes" if hold else "no",
                "--game", str(self.game), "--out", str(out), "--work", str(work), "--stages", stages]
        if prebuilt:
            args.extend(("--runtime-build", str(self.root / "synthetic prebuilt")))
        return args

    @staticmethod
    def stop(proc):
        if proc.poll() is None:
            proc.kill()
        proc.communicate(timeout=10)

    def start_build(self, stages, out, work, *, prebuilt=True):
        ready = self.root / "ready"
        proc = subprocess.Popen(self.command(stages, out, work, ready, hold=True, prebuilt=prebuilt),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop, proc)
        until = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < until and proc.poll() is None:
            time.sleep(0.02)
        self.assertTrue(ready.exists(), "the first build did not enter its synthetic stage")
        return proc

    def attempt(self, stages, out, work, *, prebuilt=True):
        ready = self.root / "second ready"
        ready.unlink(missing_ok=True)
        result = subprocess.run(self.command(stages, out, work, ready, prebuilt=prebuilt),
                                capture_output=True, text=True, timeout=10)
        return result, ready.exists()

    def assert_rejected(self, result, entered, path):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertFalse(entered)
        self.assertIn("Another Leonida build is using", result.stderr)
        self.assertIn(str(path).casefold(), result.stderr.casefold())

    def test_shared_work_is_exclusive_and_death_releases_it(self):
        work = self.root / "Shared Work"
        first = self.start_build("assets", self.root / "out1", work)
        alias = work.parent / "SHARED WORK" if sys.platform == "win32" else self.root / "work alias"
        if sys.platform != "win32":
            alias.symlink_to(work, target_is_directory=True)
        result, entered = self.attempt("assets", self.root / "out2", alias)
        self.assert_rejected(result, entered, work)

        result, entered = self.attempt("assets", self.root / "out2", self.root / "work2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(entered)
        self.stop(first)
        result, entered = self.attempt("assets", self.root / "out2", alias)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(entered)

    def test_shared_output_is_exclusive_across_writing_stages(self):
        out = self.root / "shared output"
        first = self.start_build("runtime", out, self.root / "work1")
        work = self.root / "a separate work"
        result, entered = self.attempt("pipelines", out, work)
        self.assert_rejected(result, entered, out)

        # Contention on --out must release a --work lock acquired earlier in the same attempt.
        result, entered = self.attempt("assets", self.root / "other out", work)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(entered)
        stdout, stderr = first.communicate(input="x", timeout=10)
        self.assertEqual(first.returncode, 0, stdout + stderr)

    def test_a_folder_shared_by_work_and_output_is_locked_once(self):
        folder = self.root / "combined"
        result, entered = self.attempt("pipelines", folder, folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(entered)

    def test_runtime_builds_in_one_checkout_share_the_builder_lock(self):
        self.start_build("runtime", self.root / "out1", self.root / "work1", prebuilt=False)
        result, entered = self.attempt("runtime", self.root / "out2", self.root / "work2", prebuilt=False)
        self.assert_rejected(result, entered, self.root / "synthetic checkout" / "runtime" / "build")
        result, entered = self.attempt("runtime", self.root / "out2", self.root / "work2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(entered)


if __name__ == "__main__":
    unittest.main()
