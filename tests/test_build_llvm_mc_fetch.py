"""Exercise LLVM sparse/partial fetch against a local Git repo (no network or LLVM build)."""
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"
UPSTREAM_REPO = "https://github.com/llvm/llvm-project"


@unittest.skipUnless(os.name == "nt" and shutil.which("git"), "Windows and Git required")
class FetchSourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.origin = self.root / "origin"
        self.origin.mkdir()
        for rel, body in (
            ("llvm/CMakeLists.txt", "# release 1\n"),
            ("cmake/Modules/Helper.cmake", "# cmake\n"),
            ("third-party/README.txt", "# third-party\n"),
            ("libc/shared/str.h", "# libc\n"),
            ("clang/CMakeLists.txt", "# excluded by sparse-checkout\n"),
        ):
            p = self.origin / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8", newline="\n")

        self.real_git = shutil.which("git")
        self.git(self.origin, "init", "-q")
        self.git(self.origin, "config", "uploadpack.allowFilter", "true")
        self.git(self.origin, "config", "uploadpack.allowAnySHA1InWant", "true")
        self.git(self.origin, "add", ".")
        self.git(self.origin, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "release 21.1.8")
        self.commit_annotated = self.git(self.origin, "rev-parse", "HEAD")
        self.git(self.origin, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "tag", "-am", "annotated release", "llvmorg-21.1.8")

        (self.origin / "llvm" / "CMakeLists.txt").write_text("# release 2\n", encoding="utf-8", newline="\n")
        self.git(self.origin, "add", ".")
        self.git(self.origin, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "release 23.1.5")
        self.commit_lightweight = self.git(self.origin, "rev-parse", "HEAD")
        self.git(self.origin, "tag", "llvmorg-23.1.5")

        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$src = Join-Path $Temp "src"')
        block = text[start:text.index("# DIA SDK off:", start)]
        self.assertEqual(block.count(UPSTREAM_REPO), 1)
        block = block.replace(UPSTREAM_REPO, self.origin.as_uri())

        real_git_ps = self.real_git.replace("'", "''")
        self.driver = self.root / "fetch.ps1"
        self.driver.write_text(
            'param([string]$Temp, [string]$Version)\n'
            '$ErrorActionPreference = "Stop"\n'
            f'$script:realGit = \'{real_git_ps}\'\n'
            'function git {\n'
            '    if ($args.Count -gt 0 -and $args[0] -eq "clone") {\n'
            '        [Console]::Error.WriteLine("warning: refs/tags/llvmorg-21.1.8 0000000000000000000000000000000000000000 is not a commit!")\n'
            '        $global:LASTEXITCODE = 128\n'
            '        return\n'
            '    }\n'
            '    & $script:realGit @args\n'
            '}\n'
            + block + '\nWrite-Output "SOURCE_ACCEPTED"\n',
            encoding="utf-8", newline="\n",
        )
        self.work = self.root / "work"

    def git(self, cwd, *args):
        return subprocess.run(
            [self.real_git, "-C", str(cwd), *args],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()

    def run_fetch(self, version):
        return subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(self.driver), "-Temp", str(self.work), "-Version", version],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
        )

    def test_annotated_tag_fetches_peeled_commit_without_shallow_branch_clone(self):
        res = self.run_fetch("21.1.8")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("SOURCE_ACCEPTED", res.stdout)
        src = self.work / "src"
        self.assertEqual(self.git(src, "rev-parse", "HEAD"), self.commit_annotated)
        self.assertEqual(
            self.git(src, "rev-parse", "--verify", "--quiet", "refs/tags/llvmorg-21.1.8^{commit}"),
            self.commit_annotated,
        )
        self.assertEqual(self.git(src, "config", "--get", "core.longpaths"), "true")
        self.assertEqual(self.git(src, "config", "--get", "remote.origin.promisor"), "true")
        self.assertEqual(self.git(src, "config", "--get", "remote.origin.partialclonefilter"), "blob:none")
        self.assertEqual((src / "llvm" / "CMakeLists.txt").read_text(encoding="utf-8"), "# release 1\n")
        self.assertTrue((src / "cmake" / "Modules" / "Helper.cmake").is_file())
        self.assertTrue((src / "third-party" / "README.txt").is_file())
        self.assertTrue((src / "libc" / "shared" / "str.h").is_file())
        self.assertFalse((src / "clang").exists())

    def test_lightweight_tag_falls_back_and_fetches_commit(self):
        res = self.run_fetch("23.1.5")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("SOURCE_ACCEPTED", res.stdout)
        src = self.work / "src"
        self.assertEqual(self.git(src, "rev-parse", "HEAD"), self.commit_lightweight)
        self.assertEqual((src / "llvm" / "CMakeLists.txt").read_text(encoding="utf-8"), "# release 2\n")

    def test_missing_release_tag_fails_without_creating_src(self):
        res = self.run_fetch("99.99.99")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("Release tag llvmorg-99.99.99 was not found", res.stderr)
        self.assertFalse((self.work / "src").exists())

    def test_stale_partial_src_without_cmakelists_is_cleaned_and_refetched(self):
        src = self.work / "src"
        stale_dir = src / "stale"
        stale_dir.mkdir(parents=True)
        stale_file = stale_dir / "leftover.txt"
        stale_file.write_text("stale\n", encoding="utf-8", newline="\n")
        os.chmod(stale_file, stat.S_IREAD)
        self.addCleanup(lambda: os.chmod(stale_file, stat.S_IWRITE) if stale_file.exists() else None)

        res = self.run_fetch("21.1.8")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertFalse(stale_file.exists())
        self.assertEqual(self.git(src, "rev-parse", "HEAD"), self.commit_annotated)


if __name__ == "__main__":
    unittest.main()
