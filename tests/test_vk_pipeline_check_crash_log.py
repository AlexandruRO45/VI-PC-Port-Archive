"""vk_pipeline_check: the result lines printed before a driver crash are in a redirected log.

The pipelines stage and testers run the checker with stdout redirected to a file. A driver that crashes the process
doesn't flush stdio, so a line still in a buffer is lost: on Windows the CRT has no line buffering (_IOLBF is full
buffering there), and a crashed run used to keep only its first 64 KiB of results.

No game data: three tiny compute shaders are assembled here with spirv-as. The crash is simulated with the tool's test
hook VK_PIPELINE_CHECK_SIMULATE_CRASH, which doesn't flush either. Skips (not fails) when vk_pipeline_check isn't
built, spirv-as is missing or there is no usable Vulkan device. Pin a GPU with VK_LOADER_DRIVERS_SELECT /
LEONIDA_VK_DEVICE as for the other Vulkan tools. Set LEONIDA_VK_PIPELINE_CHECK to use a binary elsewhere.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["m_alpha", "m_bravo", "m_charlie"]   # checked in this order with --jobs 1
CRASHER = "m_charlie"

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
    exe = "vk_pipeline_check" + (".exe" if os.name == "nt" else "")
    for p in (build / "linux" / exe, build / exe, build / "RelWithDebInfo" / exe, build / "Release" / exe):
        if p.is_file():
            return p
    return None


class CrashLogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = find_binary()
        if cls.exe is None:
            raise unittest.SkipTest("vk_pipeline_check isn't built")
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
        code, log, err = cls.run_to_log("probe.log")
        # Every exit without a Vulkan device comes before the checker names the device it picked; only that skips.
        # Once a device is picked, a probe that doesn't check all three pipelines cleanly is a failure, not a skip.
        if "device: " not in err:
            raise unittest.SkipTest("no usable Vulkan device (exit %d): %s" % (code, err.strip()[-300:]))
        if code != 0 or "cs pipelines: 3 ok, 0 failed" not in log:
            raise AssertionError("the run without a crash failed (exit %d):\n%s\n%s" % (code, log, err))

    @classmethod
    def run_to_log(cls, log_name, crash=None):
        """Runs the checker with stdout redirected to a file, as a log is written: (exit code, log, stderr)."""
        env = dict(os.environ)
        env.pop("VK_PIPELINE_CHECK_SIMULATE_CRASH", None)
        if crash:
            env["VK_PIPELINE_CHECK_SIMULATE_CRASH"] = crash
        log = Path(cls.tmp.name) / log_name
        # --slow 0 prints a SLOW line for every pipeline that finishes, so the log names each module checked.
        cmd = [str(cls.exe), str(cls.dir), "--jobs", "1", "--slow", "0"]
        with open(log, "wb") as out:
            r = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.PIPE, timeout=300)
        return r.returncode, log.read_text(encoding="utf-8", errors="replace"), r.stderr.decode("utf-8", "replace")

    def test_results_before_a_crash_reach_the_log(self):
        code, log, _ = self.run_to_log("crash.log", crash=CRASHER)
        self.assertNotIn(code, (0, 1, 2), "the simulated crash must end the process: " + log)
        if os.name == "nt":   # the hook's default SIGFPE ends a Windows process with EXCEPTION_FLT_DIVIDE_BY_ZERO
            self.assertEqual(code & 0xFFFFFFFF, 0xC000008E, "exit code %#x" % (code & 0xFFFFFFFF))
        self.assertNotIn("cs pipelines:", log, "a crashed run prints no summary")
        for name in NAMES:
            if name != CRASHER:
                self.assertIn(" %s " % name, log, "the result line of %s was lost in the crash" % name)


def main():
    """unittest.main(), except that a file whose tests all skipped succeeds: unittest exits 5 for an empty run, and
    CI runs each file as `python -I tests/test_*.py || exit 1`, so a runner without vk_pipeline_check would fail the
    job here. A file that has no tests at all (not even skips) is still a failure."""
    result = unittest.main(exit=False).result
    if result.failures or result.errors or result.unexpectedSuccesses:
        return 1
    if result.testsRun == 0 and not result.skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
