"""The shared Vulkan fuzzer must stop a hung child, without a GPU or llvm-mc."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import vk_fuzz_run as runner
import hw_oracle.runner


class GpuTimeout(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / ("vk_compute_test.exe" if os.name == "nt" else "vk_compute_test")).touch()
        self.gpu_calls = 0
        self.delay = 0
        self.real_run = subprocess.run
        self.patch("mc_command.mc_command", return_value=["synthetic-mc"])
        self.patch("mc_command.run", return_value=subprocess.CompletedProcess([], 0, "", ""))
        self.patch("compute_fixture.compute_header", return_value=b"")
        self.patch("vk_fuzz_run.subprocess.run", side_effect=self.run_child)
        env = patch.dict(os.environ, {"VK_FUZZ_KEEP": "", "LEONIDA_GPU_TEST_TIMEOUT": "0.1"})
        env.start()
        self.addCleanup(env.stop)

    def patch(self, target, **kwargs):
        patched = patch(target, **kwargs)
        value = patched.start()
        self.addCleanup(patched.stop)
        return value

    def run_child(self, args, **kwargs):
        if "--out" in args:
            spirv = Path(args[args.index("--out") + 1])
            spirv.mkdir()
            (spirv / "cs_vkfuzz_0.spv").write_bytes(b"")
            (spirv / "index.json").write_text(json.dumps({"cs_vkfuzz_0": {}}), encoding="utf-8", newline="\n")
            return subprocess.CompletedProcess(args, 0, "", "")
        self.gpu_calls += 1
        # This really sleeps in a subprocess; only the unavailable compiler/GPU are replaced.
        child = ("import os, struct, time; from pathlib import Path; "
                 f"time.sleep({self.delay!r}); "
                 "n = int(os.environ['VKC_OUT_WORDS']); "
                 "Path(os.environ['VKC_DUMP']).write_bytes(struct.pack('<' + str(n) + 'I', *([7] * n)))")
        return self.real_run([sys.executable, "-c", child], **kwargs)

    def run_program(self):
        return runner.run_programs([["s_nop 0"]], build=str(self.root), work_parent=str(self.root))

    def test_hung_child_fails_without_retry_and_cleans_work(self):
        self.delay = 1.0
        with self.assertRaisesRegex(RuntimeError, "cs_vkfuzz_0.*timed out after 0.1 s"):
            self.run_program()
        self.assertEqual(self.gpu_calls, 1)
        self.assertEqual(list(self.root.glob("vk_fuzz_*")), [])

    def test_success_and_disabled_timeout_keep_results(self):
        for limit in ("2", "0", ""):
            with self.subTest(limit=limit), patch.dict(os.environ, {"LEONIDA_GPU_TEST_TIMEOUT": limit}):
                self.delay = 0.2 if limit == "0" else 0
                self.assertEqual(self.run_program(), [[7, 7, 7, 7]])

    def test_all_fuzzer_clis_report_timeout_as_harness_error(self):
        self.delay = 1.0
        for name in ("pk16_diff", "vint_diff", "sdwa_diff", "wint_diff", "fmisc_diff"):
            tool = importlib.import_module(name)
            oracle = Mock(device="synthetic", arch="gfx1030")
            out = io.StringIO()
            with self.subTest(tool=name), \
                    patch.object(hw_oracle.runner, "open_oracle", return_value=oracle), \
                    patch.object(tool, "run_suite", side_effect=lambda *a, **kw: self.run_program()), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(tool.main(["--oracle", "hw,vk", "--programs", "1"]), 2)
            self.assertIn("harness error:", out.getvalue())
            self.assertIn("timed out after 0.1 s", out.getvalue())
            oracle.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
