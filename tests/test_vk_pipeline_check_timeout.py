"""vk_pipeline_check --module-timeout: a compile that never returns ends the run with exit status 75, the in-flight file
naming only that module, so make_game.py's pipelines stage can skip it (a hung Metal compile on Apple GPUs).

No game data: five tiny compute shaders are assembled here with spirv-as. The hang is simulated with the tool's test
hook VK_PIPELINE_CHECK_SIMULATE_CRASH=<name>:HANG (it sleeps just before the pipeline of the named module), so no
hanging driver is needed. Skips (not fails) when vk_pipeline_check isn't built, spirv-as is missing or there is no
usable Vulkan device. Pin a GPU with VK_LOADER_DRIVERS_SELECT / LEONIDA_VK_DEVICE as for the other Vulkan tools. Set
LEONIDA_VK_PIPELINE_CHECK to use a binary elsewhere.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["m_alpha", "m_bravo", "m_charlie", "m_delta", "m_echo"]
HANGER = "m_charlie"

ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {size} 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


def find_binary():
    env = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
    if env:
        return Path(env) if Path(env).is_file() else None
    build = ROOT / "runtime" / "build"
    for p in (build / "linux" / "vk_pipeline_check", build / "vk_pipeline_check"):
        if p.is_file():
            return p
    return None


class ModuleTimeoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built (cmake -S runtime -B runtime/build/linux)")
        spirv_as = shutil.which("spirv-as")
        if not spirv_as:
            raise unittest.SkipTest("spirv-as not found")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.dir = Path(cls.tmp.name) / "spirv"
        cls.dir.mkdir()
        index = {}
        for i, name in enumerate(NAMES):
            asm = Path(cls.tmp.name) / (name + ".spvasm")
            asm.write_text(ASM.format(size=1 << i))
            subprocess.run([spirv_as, "--target-env", "vulkan1.1", str(asm), "-o", str(cls.dir / (name + ".spv"))],
                           check=True)
            index[name] = {"stage": "cs", "subgroup_size": None, "features": []}
        (cls.dir / "index.json").write_text(json.dumps(index))
        probe = cls.run_tool([])
        if probe.returncode != 0 or "cs pipelines: 5 ok, 0 failed" not in probe.stdout:
            raise unittest.SkipTest("no usable Vulkan device (exit %d): %s" % (probe.returncode, probe.stderr.strip()[-300:]))

    @classmethod
    def run_tool(cls, args, hang=None):
        env = dict(os.environ)
        env.pop("VK_PIPELINE_CHECK_SIMULATE_CRASH", None)
        if hang:
            env["VK_PIPELINE_CHECK_SIMULATE_CRASH"] = hang + ":HANG"
        cmd = [str(cls.exe), str(cls.dir), "--jobs", "2"] + args
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)

    def path(self, name):
        return Path(self.tmp.name) / name

    def test_a_hung_compile_ends_the_run_and_names_only_that_module(self):
        inflight = self.path("inflight.txt")
        r = self.run_tool(["--module-timeout", "3", "--inflight-file", str(inflight)], hang=HANGER)
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertIn("MODULE TIMEOUT %s (pipeline still in the driver after" % HANGER, r.stdout)
        self.assertEqual(r.stdout.count("MODULE TIMEOUT"), 1, r.stdout)
        self.assertEqual(inflight.read_text(), HANGER + "\tpipeline\n")

    def test_a_cache_save_that_never_finishes_does_not_stop_the_exit(self):
        # A driver can hold the cache's lock for the whole of a hung compile (MoltenVK does), so the checkpoint before
        # the exit may never return; the exit must not wait for it, and the old cache file must stay as it was.
        cache = self.path("held.cache")
        cache.write_bytes(b"previous checkpoint")
        inflight = self.path("held_inflight.txt")
        env_args = ["--module-timeout", "2", "--inflight-file", str(inflight)]
        env = dict(os.environ, VK_PIPELINE_CHECK_SIMULATE_CRASH=HANGER + ":HANG",
                   VK_PIPELINE_CHECK_SIMULATE_SAVE_HANG="1")
        r = subprocess.run([str(self.exe), str(self.dir), str(cache), "--jobs", "2"] + env_args, env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertIn("MODULE TIMEOUT %s" % HANGER, r.stdout)
        self.assertIn("the last saved checkpoint is kept", r.stderr)
        self.assertEqual(cache.read_bytes(), b"previous checkpoint")
        self.assertEqual(inflight.read_text(), HANGER + "\tpipeline\n")

    def test_without_a_timeout_nothing_changes(self):
        r = self.run_tool(["--module-timeout", "60"])
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("cs pipelines: 5 ok, 0 failed, 0 skipped", r.stdout)
        self.assertNotIn("MODULE TIMEOUT", r.stdout)

    def test_the_skip_file_reason_is_printed(self):
        skip = self.path("skip.txt")
        skip.write_text(HANGER + "\tdid not finish within 3 s in an earlier run\n")
        r = self.run_tool(["--module-timeout", "3", "--skip-file", str(skip)], hang=HANGER)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)   # the skipped module counts as failed
        self.assertIn("MODULE CRASH %s (cs, did not finish within 3 s in an earlier run; not retried)" % HANGER,
                      r.stdout)
        self.assertIn("cs pipelines: 4 ok, 1 failed, 0 skipped", r.stdout)

    def test_a_plain_skip_line_keeps_the_crash_reason(self):
        skip = self.path("skip_plain.txt")
        skip.write_text(HANGER + "\n")
        r = self.run_tool(["--skip-file", str(skip)])
        self.assertIn("MODULE CRASH %s (cs, killed the driver in an earlier run; not retried)" % HANGER, r.stdout)

    def test_bad_values_and_isolate_are_refused(self):
        for args in (["--module-timeout", "-1"], ["--module-timeout", "nan"], ["--module-timeout", "x"]):
            with self.subTest(args=args):
                self.assertEqual(self.run_tool(args).returncode, 2)
        if os.name == "posix":
            r = self.run_tool(["--isolate", "--module-timeout", "5"])
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("--isolate cannot be combined with --module-timeout", r.stderr)


def main():
    """unittest.main(), except that a file whose tests all skipped succeeds: unittest exits 5 for an empty run, and
    CI runs each file as `python -I tests/test_*.py || exit 1`, so a runner without vk_pipeline_check or on Windows
    would fail the job here. A file that has no tests at all (not even skips) is still a failure."""
    result = unittest.main(exit=False).result
    if result.failures or result.errors or result.unexpectedSuccesses:
        return 1
    if result.testsRun == 0 and not result.skipped:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
