"""Runtime artifacts carry their actual source commit and only public build fields.

    python -I tests/test_runtime_build_info.py

Synthetic cache and CI metadata only; no build, network or game files.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from write_runtime_build_info import write_build_info

CACHE = ("CMAKE_BUILD_TYPE:STRING=RelWithDebInfo\r\n"
         "CMAKE_GENERATOR:INTERNAL=Ninja\r\n"
         "LEONIDA_D3D12_SHIM_TESTS:BOOL=ON\r\n"
         "LEONIDA_REQUIRE_VULKAN:BOOL=ON\r\n"
         "CMAKE_HOME_DIRECTORY:INTERNAL=C:/private/checkout/runtime\r\n"
         "UNRELATED_SECRET:STRING=cache-secret-sentinel\r\n")
CI = {"GITHUB_SERVER_URL": "https://github.com", "GITHUB_REPOSITORY": "example/runtime",
      "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "2", "GITHUB_SHA": "0" * 40,
      "GITHUB_TOKEN": "env-secret-sentinel", "RUNNER_TEMP": "/private/runner/temp"}


@unittest.skipUnless(shutil.which("git"), "needs git")
class RuntimeBuildInfo(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "source checkout"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-q", "--allow-empty", "-m", "synthetic source")
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()
        self.build = Path(self.tmp.name) / "runtime build"
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_bytes(CACHE.encode("utf-8"))

    def git(self, *args):
        return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                              cwd=self.repo, check=True, capture_output=True, text=True)

    def test_actual_checkout_and_whitelist(self):
        output = write_build_info(self.build, repo_root=self.repo, environ=CI)
        self.assertEqual(output.name, "build-info.json")
        self.assertEqual(json.loads(output.read_bytes()), {
            "schema_version": 1, "source_commit": self.commit,
            "workflow_run_url": "https://github.com/example/runtime/actions/runs/12345/attempts/2",
            "build_configuration": {"CMAKE_BUILD_TYPE": "RelWithDebInfo", "CMAKE_GENERATOR": "Ninja",
                                    "LEONIDA_D3D12_SHIM_TESTS": "ON", "LEONIDA_REQUIRE_VULKAN": "ON"}})
        data = output.read_bytes()
        self.assertTrue(data.endswith(b"\n"))
        self.assertNotIn(b"\r", data)
        for private in (b"secret-sentinel", b"private", b"GITHUB_SHA", b"GITHUB_TOKEN"):
            self.assertNotIn(private, data)
        write_build_info(self.build, repo_root=self.repo, environ=CI)
        self.assertEqual(output.read_bytes(), data)

    def test_missing_configuration_preserves_previous_manifest(self):
        output = self.build / "build-info.json"
        output.write_bytes(b"previous manifest\n")
        (self.build / "CMakeCache.txt").write_bytes(b"CMAKE_GENERATOR:INTERNAL=Ninja\n")
        with self.assertRaisesRegex(ValueError, "missing CMake cache fields: CMAKE_BUILD_TYPE"):
            write_build_info(self.build, repo_root=self.repo, environ=CI)
        self.assertEqual(output.read_bytes(), b"previous manifest\n")

    def test_missing_ci_metadata_does_not_write_a_manifest(self):
        metadata = dict(CI)
        del metadata["GITHUB_RUN_ID"]
        with self.assertRaisesRegex(ValueError, "missing CI metadata: GITHUB_RUN_ID"):
            write_build_info(self.build, repo_root=self.repo, environ=metadata)
        self.assertFalse((self.build / "build-info.json").exists())

    def test_output_must_not_alias_cache(self):
        cache = self.build / "CMakeCache.txt"
        os.link(cache, self.build / "build-info.json")
        with self.assertRaisesRegex(ValueError, "must not overwrite CMakeCache.txt"):
            write_build_info(self.build, repo_root=self.repo, environ=CI)
        self.assertEqual(cache.read_bytes(), CACHE.encode("utf-8"))

    def test_cli_uses_its_checkout_from_another_working_directory(self):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/write_runtime_build_info.py"),
                                 str(self.build)], cwd=self.repo, env={**os.environ, **CI},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  check=True, capture_output=True, text=True).stdout.strip()
        info = json.loads((self.build / "build-info.json").read_bytes())
        self.assertEqual(info["source_commit"], expected)
        self.assertNotEqual(info["source_commit"], self.commit)


if __name__ == "__main__":
    unittest.main()
