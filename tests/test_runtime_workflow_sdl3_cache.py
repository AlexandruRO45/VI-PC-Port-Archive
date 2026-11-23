"""The Runtime ``windows`` job caches the SDL3 fetch and build, keyed on the pinned tarball and the toolchain.

    python -I tests/test_runtime_workflow_sdl3_cache.py

The GameInput shim builds SDL3 from the pinned release (``-DLEONIDA_SDL3_FETCH=ON``), which is C: sccache does
not help it, so the job paid that build on every push. ``.github/workflows/runtime.yml`` now caches the fetched
sources and the build tree. The cache is only safe when a run cannot pick up a tree another pin or another
toolchain wrote, so its key has to carry the tarball SHA-256, the salt that is bumped when the build options
change, and the cl/CMake/Ninja versions. This test reads the workflow rather than a copy of it and keeps those
inputs, the cached paths and the version read in step with ``runtime/CMakeLists.txt``.

The version step reads ``cl``'s ``FileVersion``: ``cl`` with no arguments writes its usage and its banner on
different streams, so ``(& cl 2>&1 | Select-Object -First 1)`` sometimes returned the usage line and produced a
key without the compiler version -- a compiler update that did not change the key, reusing objects another
compiler wrote. Only the tracked workflow and ``runtime/CMakeLists.txt`` are read; no YAML package, no build
tools, no game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW = os.path.join(".github", "workflows", "runtime.yml")
CMAKE = os.path.join("runtime", "CMakeLists.txt")
CACHE_STEP = "- name: Cache the SDL3 fetch and build"
TOOLSET_STEP = "- name: Read the tool versions for the SDL3 cache key"
SDL3_PATHS = (
    "runtime/build/_deps/sdl3-src",
    "runtime/build/_deps/sdl3-subbuild",
    "runtime/build/_deps/sdl3-build",
)


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def windows_job(text):
    """The `windows` job only: the next two-space-indented job key ends it."""
    match = re.search(r"\n  windows:\n(.*?)(?=\n  [A-Za-z0-9_-]+:\n)", text, re.S)
    if match is None:
        raise AssertionError("runtime.yml has no windows job")
    return match.group(1)


def step(job, header):
    """The text of one step, from its `- name:` line to the next one."""
    start = job.find("      " + header)
    if start < 0:
        raise AssertionError("runtime.yml has no step %r" % header)
    end = job.find("\n      - ", start)
    return job[start:] if end < 0 else job[start:end]


class Sdl3CacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.job = windows_job(read(WORKFLOW))
        cls.cache = step(cls.job, CACHE_STEP)
        cls.toolset = step(cls.job, TOOLSET_STEP)

    def test_the_job_env_pins_the_tarball_that_cmake_fetches(self):
        pinned = re.search(r"FetchContent_Declare\(SDL3.*?URL_HASH SHA256=([0-9a-f]{64})",
                           read(CMAKE), re.S)
        self.assertIsNotNone(pinned, "runtime/CMakeLists.txt no longer pins the SDL3 tarball")
        self.assertIn("SDL3_TARBALL_SHA256: %s" % pinned.group(1), self.job)
        self.assertIn('SDL3_CACHE_SALT: "1"', self.job)

    def test_the_cache_covers_the_fetched_sources_and_the_build_tree(self):
        self.assertIn("actions/cache@", self.cache)
        block = self.cache.split("path: |", 1)
        self.assertEqual(len(block), 2, "the SDL3 cache step has no path list")
        for path in SDL3_PATHS:
            self.assertIn(path, block[1], "the SDL3 cache does not cover %s" % path)

    def test_the_key_carries_the_tarball_the_salt_and_the_toolchain(self):
        key = re.search(r"key: (sdl3-.+)", self.cache)
        self.assertIsNotNone(key, "the SDL3 cache step has no key")
        self.assertIn("sdl3-3.4.20-", key.group(1))
        self.assertIn("${{ env.SDL3_TARBALL_SHA256 }}", key.group(1))
        self.assertIn("${{ env.SDL3_CACHE_SALT }}", key.group(1))
        self.assertIn("${{ steps.sdl3-toolset.outputs.version }}", key.group(1))
        self.assertIn("version=$version", self.toolset)

    def test_the_toolset_step_reads_cls_file_version(self):
        self.assertIn("shell: pwsh", self.toolset)
        self.assertIn("Get-Command cl -CommandType Application", self.toolset)
        self.assertIn(".VersionInfo.FileVersion", self.toolset)
        # The merged-stream form returns the usage line on some runs; it must not come back.
        self.assertNotIn("(& cl", self.toolset)
        self.assertIn("cmake --version", self.toolset)
        self.assertIn("ninja --version", self.toolset)

    def test_the_cache_is_restored_before_the_configure(self):
        configure = self.job.find("- name: Configure")
        self.assertGreater(configure, 0, "the windows job has no configure step")
        self.assertLess(self.job.find(CACHE_STEP), configure)
        self.assertLess(self.job.find(TOOLSET_STEP), self.job.find(CACHE_STEP))


if __name__ == "__main__":
    unittest.main()
