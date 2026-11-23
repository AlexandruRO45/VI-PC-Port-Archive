"""Exercise submodule initialization and pin updates with local Git repositories and a synthetic file."""
from __future__ import annotations

from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_game  # noqa: E402


@unittest.skipUnless(shutil.which("git"), "requires Git")
class EnsureSubmoduleTest(unittest.TestCase):
    submodule = Path("third_party", "rpf9-explorer")

    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="leonida submodule ")
        self.addCleanup(td.cleanup)
        self.root = Path(td.name)
        # Restoring a patched os.environ can drop inherited empty values on Windows.
        self.git_env = os.environ.copy()
        self.git_env.pop("GIT_CONFIG_COUNT", None)
        self.git_env.pop("GIT_CONFIG_PARAMETERS", None)
        self.git_env.update({
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
        })
        # Git hooks export repository settings that would redirect commands for our fixtures.
        for name in self.git(self.root, "rev-parse", "--local-env-vars").splitlines():
            self.git_env.pop(name, None)

        dependency = self.root / "dependency"
        self.dependency = dependency
        self.git(self.root, "init", "-q", dependency)
        (dependency / "Core").mkdir()
        (dependency / "Core" / "fixture.txt").write_text("synthetic\n", encoding="utf-8")
        self.git(dependency, "add", "Core")
        self.git(dependency, "commit", "-qm", "Add fixture")

        self.source = self.root / "source"
        self.git(self.root, "init", "-q", self.source)
        self.git(self.source, "submodule", "add", "-q", dependency.as_posix(), self.submodule.as_posix())
        self.git(self.source, "commit", "-qm", "Add submodule")

    def git(self, cwd, *args):
        result = subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                                 *args], cwd=cwd, env=self.git_env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def assert_initializes_submodule(self, checkout):
        fixture = checkout / self.submodule / "Core" / "fixture.txt"
        self.assertFalse(fixture.exists())
        run_git = partial(subprocess.run, env=self.git_env)
        with (mock.patch.object(make_game, "ROOT", str(checkout)),
              mock.patch.object(make_game.subprocess, "run", new=run_git)):
            make_game.ensure_submodule()
        self.assertEqual(fixture.read_text(encoding="utf-8"), "synthetic\n")

    def run_ensure(self, checkout):
        commands = []
        real_run = subprocess.run

        def run_git(*args, **kwargs):
            commands.extend(args)
            return real_run(*args, env=self.git_env, **kwargs)

        with (mock.patch.object(make_game, "ROOT", str(checkout)),
              mock.patch.object(make_game.subprocess, "run", new=run_git)):
            make_game.ensure_submodule()
        return commands

    def advance_decoder(self):
        (self.dependency / "Core" / "fixture.txt").write_text("updated\n", encoding="utf-8")
        self.git(self.dependency, "add", "Core")
        self.git(self.dependency, "commit", "-qm", "Update fixture")
        pin = self.git(self.dependency, "rev-parse", "HEAD").strip()
        checkout = self.source / self.submodule
        self.git(checkout, "fetch", "-q", self.dependency.as_posix(), "HEAD")
        self.git(checkout, "checkout", "-q", pin)
        self.git(self.source, "add", self.submodule.as_posix())
        self.git(self.source, "commit", "-qm", "Update submodule pin")
        return pin

    def test_initializes_submodule_in_clone(self):
        checkout = self.root / "clone"
        self.git(self.root, "clone", "-q", "--no-recurse-submodules", self.source, checkout)
        self.assertTrue((checkout / ".git").is_dir())
        self.assert_initializes_submodule(checkout)

    def test_initializes_submodule_in_linked_worktree(self):
        checkout = self.root / "worktree"
        self.git(self.source, "worktree", "add", "-q", "--detach", checkout)
        self.assertTrue((checkout / ".git").is_file())
        self.assert_initializes_submodule(checkout)

    def test_current_submodule_is_left_alone(self):
        checkout = self.root / "current"
        self.git(self.root, "clone", "-q", "--recurse-submodules", self.source, checkout)
        commands = self.run_ensure(checkout)
        self.assertFalse(any("submodule" in command for command in commands))
        fixture = checkout / self.submodule / "Core" / "fixture.txt"
        self.assertEqual(fixture.read_text(encoding="utf-8"), "synthetic\n")

    def test_updates_stale_submodule_to_parent_pin(self):
        old_pin = self.git(self.source, "rev-parse", "HEAD:third_party/rpf9-explorer").strip()
        pin = self.advance_decoder()
        checkout = self.root / "stale"
        self.git(self.root, "clone", "-q", "--recurse-submodules", self.source, checkout)
        self.git(checkout / self.submodule, "checkout", "-q", old_pin)
        self.run_ensure(checkout)
        fixture = checkout / self.submodule / "Core" / "fixture.txt"
        self.assertEqual(fixture.read_text(encoding="utf-8"), "updated\n")
        actual = self.git(checkout / self.submodule, "rev-parse", "HEAD").strip()
        self.assertEqual(actual, pin)

    def test_initializes_submodule_with_inherited_repository_settings(self):
        git_dir = self.source / ".git"
        hook_env = {
            **self.git_env,
            "GIT_DIR": str(git_dir),
            "GIT_WORK_TREE": str(self.source),
            "GIT_INDEX_FILE": str(git_dir / "index"),
        }
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()),
             "EnsureSubmoduleTest.test_initializes_submodule_in_linked_worktree"],
            # errors="replace": git prints paths in UTF-8, which the Windows ANSI code page can't always decode.
            cwd=self.root, env=hook_env, capture_output=True, text=True, errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
