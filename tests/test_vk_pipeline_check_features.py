"""Unsupported metadata must be skipped before shader module creation.

Uses only a synthetic compute module. Set LEONIDA_VK_PIPELINE_CHECK to a built
checker. A Vulkan device is required.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from spirv_asm import assemble

ASM = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


class FeatureGateTest(unittest.TestCase):
    def setUp(self):
        configured = os.environ.get("LEONIDA_VK_PIPELINE_CHECK")
        build = ROOT / "runtime" / "build"
        candidates = [Path(configured)] if configured else [
            build / "linux" / "vk_pipeline_check", build / "vk_pipeline_check",
            build / "vk_pipeline_check.exe", build / "RelWithDebInfo" / "vk_pipeline_check.exe",
            build / "Release" / "vk_pipeline_check.exe",
        ]
        binary = next((path for path in candidates if path.is_file()), None)
        if binary is None:
            self.skipTest("set LEONIDA_VK_PIPELINE_CHECK to a built checker")
        self.binary = binary

    def test_missing_or_invalid_size_is_failed_before_feature_skip(self):
        for payload in (None, b"", b"short", bytes(21)):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if payload is not None:
                    (root / "cs_absent.spv").write_bytes(payload)
                with (root / "index.json").open("w", encoding="utf-8", newline="\n") as output:
                    json.dump({"cs_absent": {"stage": "cs", "push_constant_bytes": 512}}, output)
                result = subprocess.run(
                    [str(self.binary), str(root), str(root / "cache.bin"), "--jobs", "1"],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("MODULE FAIL cs_absent", result.stdout)
                self.assertIn("cs: 0 modules ok, 1 failed", result.stdout)
                self.assertNotIn("SKIPPED cs_absent", result.stdout)

    def test_unsupported_entry_never_creates_a_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = assemble(ASM)
            for name in ("supported", "unsupported"):
                (root / (name + ".spv")).write_bytes(code)
            with (root / "index.json").open("w", encoding="utf-8", newline="\n") as output:
                json.dump({"supported": {"stage": "cs"},
                           "unsupported": {"stage": "cs", "float_controls": ["syntheticUnavailable"]}}, output)
            result = subprocess.run([str(self.binary), str(root), str(root / "cache.bin")],
                                    capture_output=True, text=True, timeout=60)
        if "no usable Vulkan" in result.stderr or "no Vulkan" in result.stderr:
            self.skipTest(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cs: 1 modules ok, 0 failed", result.stdout)
        self.assertIn("cs pipelines: 1 ok, 0 failed, 1 skipped", result.stdout)
        self.assertIn("SKIPPED unsupported (unknown float control syntheticUnavailable)", result.stdout)


if __name__ == "__main__":
    unittest.main()
