"""Exercise the GAME: CMake script with a synthetic loader and guest file."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


CHECK = Path(__file__).resolve().parents[1] / "runtime/test/check_game_dir.cmake"
CMAKE = shutil.which("cmake")
LOADER = """\
import os
from pathlib import Path
import sys
import time

args = sys.argv[1:]
dot = args[:2] == ["--game-dir", "."]
assert args[2 if dot else 0:] == ["--run", "tiny_game.exe", "--", "hello", "42", "game-file"]
cwd = Path.cwd()
assert (cwd / "tiny_game.exe").read_bytes() == b"synthetic guest"
if os.environ["LEONIDA_TEST_GAME_DIR_PAUSE"] == "1" and not dot:
    gate = Path(os.environ["LEONIDA_TEST_GAME_DIR_GATE"])
    (gate / "ready").touch()
    deadline = time.monotonic() + 10
    while not (gate / "release").exists():
        if time.monotonic() >= deadline:
            raise SystemExit("the synthetic controller did not release the loader")
        time.sleep(0.01)
assert (cwd / "tiny_game_probe.txt").read_bytes() == b"game", "missing synthetic GAME probe"
with open(os.environ["LEONIDA_TEST_GAME_DIR_LOG"], "a", encoding="utf-8", newline="\\n") as log:
    print("dot" if dot else "bare", file=log)
    print(cwd, file=log)
print(f"console paths: GAME: -> {cwd}, writes -> {cwd / 'overlay'}")
print("tiny_game: PASS GAME: probe\\ntiny_game: OK\\ngame exited with code 0")
"""


@unittest.skipUnless(CMAKE, "CMake is required to exercise the GAME: check")
class GameDirCheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="leonida game dir ")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.guest_dir = self.root / "guest fixture"
        self.guest_dir.mkdir()
        self.guest = self.guest_dir / "tiny_game.exe"
        self.guest.write_bytes(b"synthetic guest")
        loader = self.root / "loader.py"
        loader.write_text(LOADER, encoding="utf-8", newline="\n")
        self.command = [CMAKE, f"-DRUNNER={sys.executable}", f"-DLOADER={loader}", f"-DGUEST={self.guest}",
                        "-P", str(CHECK)]

    def env(self, name, pause=False):
        return dict(os.environ, LEONIDA_TEST_GAME_DIR_GATE=str(self.root),
                    LEONIDA_TEST_GAME_DIR_PAUSE="1" if pause else "0",
                    LEONIDA_TEST_GAME_DIR_LOG=str(self.root / name))

    def run_check(self, name):
        return subprocess.run(self.command, cwd=self.root, env=self.env(name), capture_output=True,
                              text=True, timeout=15)

    def check_variants(self, name):
        bare, first_dir, dot, second_dir = (self.root / name).read_text(encoding="utf-8").splitlines()
        self.assertEqual((bare, dot), ("bare", "dot"))
        self.assertEqual(first_dir, second_dir)
        self.assertFalse(Path(first_dir).exists(), "the staging directory was not removed")
        return first_dir

    def test_existing_probe_is_untouched(self):
        original = self.guest_dir / "tiny_game_probe.txt"
        original.write_bytes(b"existing synthetic file")
        result = self.run_check("single")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(original.is_file(), "the original probe was removed")
        self.assertEqual(original.read_bytes(), b"existing synthetic file")
        self.assertEqual(self.guest.read_bytes(), b"synthetic guest")
        self.check_variants("single")

    def test_concurrent_checks_use_distinct_probe_directories(self):
        with subprocess.Popen(self.command, cwd=self.root, env=self.env("first", pause=True),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as first:
            try:
                deadline = time.monotonic() + 5
                while not (self.root / "ready").exists():
                    if first.poll() is not None or time.monotonic() >= deadline:
                        self.fail("the first synthetic loader did not reach its synchronization point")
                    time.sleep(0.01)
                second = self.run_check("second")
                (self.root / "release").touch()
                _, stderr = first.communicate(timeout=15)
                self.assertEqual(second.returncode, 0, second.stderr)
                self.assertEqual(first.returncode, 0, stderr)
            finally:
                if first.poll() is None:
                    (self.root / "release").touch()
                    first.communicate(timeout=15)
        self.assertNotEqual(self.check_variants("first"), self.check_variants("second"))


if __name__ == "__main__":
    unittest.main()
