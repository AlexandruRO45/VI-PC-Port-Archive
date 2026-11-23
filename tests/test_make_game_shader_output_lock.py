"""Two shader builds may share the machine, but only one may write a given cache."""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402

# Exit 37 when the build has reached the toolchain, without starting any tools.
PROBE = """
import sys
from unittest import mock
sys.path.insert(0, sys.argv[1])
import make_game
with mock.patch.object(make_game, "llvm_mc_signature", side_effect=SystemExit(37)):
    make_game.shaders("unused", sys.argv[2], 1)
"""


class ShaderOutputLockTest(unittest.TestCase):
    def probe(self, work, *, env=None):
        return subprocess.run([sys.executable, "-I", "-c", PROBE, ROOT, work],
                              capture_output=True, text=True, timeout=10, env=env)

    def test_overlapping_builds_are_refused_only_for_the_same_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            os.mkdir(work)
            other_temp = os.path.join(tmp, "temp")
            os.mkdir(other_temp)
            linked_work = os.path.join(tmp, "linked")
            try:
                os.symlink(work, linked_work, target_is_directory=True)
            except (OSError, NotImplementedError):
                linked_work = None

            def during_build():
                same = self.probe(work)
                self.assertEqual(same.returncode, 1, same.stderr)
                self.assertIn(os.path.join(work, "shaders"), same.stderr)
                # These builds still write exactly the same cache.
                with self.subTest(spelling="different temporary directory"):
                    env = dict(os.environ, TMPDIR=other_temp, TEMP=other_temp, TMP=other_temp)
                    self.assertEqual(self.probe(work, env=env).returncode, 1)
                alias = os.path.join(tmp, "WORK")
                if os.path.exists(alias) and os.path.samefile(work, alias):
                    with self.subTest(spelling="case-insensitive filesystem"):
                        self.assertEqual(self.probe(alias).returncode, 1)
                if linked_work is not None:
                    with self.subTest(spelling="linked work directory"):
                        self.assertEqual(self.probe(linked_work).returncode, 1)
                other = self.probe(os.path.join(tmp, "other"))
                self.assertEqual(other.returncode, 37, other.stderr)
                raise RuntimeError("synthetic stage failure")

            with mock.patch.object(make_game, "llvm_mc_signature", side_effect=during_build):
                with self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
                    make_game.shaders("unused", work, 1)
            after = self.probe(work)
            self.assertEqual(after.returncode, 37, after.stderr)
            self.assertEqual(os.path.getsize(os.path.join(work, ".shaders.lock")), 0)

    def test_redirected_lock_leaves_its_target_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            os.mkdir(work)
            target = os.path.join(tmp, "empty")
            open(target, "wb").close()
            lock = os.path.join(work, ".shaders.lock")
            for link in (os.symlink, os.link):
                with self.subTest(link=link.__name__):
                    try:
                        link(target, lock)
                    except (OSError, NotImplementedError):
                        self.skipTest(f"cannot create a link with {link.__name__}")
                    result = self.probe(work)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(".shaders.lock", result.stderr)
                    self.assertEqual(os.path.getsize(target), 0)
                    os.unlink(lock)


if __name__ == "__main__":
    unittest.main()
