"""runtime/test/shaders/build.py chooses --f64-soft as scripts/run_gpu_tests.py and make_game.py do.

LEONIDA_F64_SOFT=1 turns it on and 0 off; unset, it is on where tools/vk_subgroup_info.py --harness reports float64
false for the device the test programs pick (MoltenVK, KosmicKrisp, Intel Arc), and off where the device has
shaderFloat64 or the probe can't tell. Before this, a standalone build.py on a Mac translated ps_f64_* with native
doubles and check.py skipped them, while run_gpu_tests.py ran them with --f64-soft.

Standard library only; the probe and the translator are replaced by stand-ins. No llvm-mc, Vulkan, GPU or game data.

    python -I tests/test_shader_build_f64_soft.py
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "runtime", "test", "shaders", "build.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vk_subgroup_info  # noqa: E402


def load_build():
    spec = importlib.util.spec_from_file_location("shaders_build_f64_under_test", BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildF64SoftTest(unittest.TestCase):
    def setUp(self):
        self.build = load_build()

    def wanted(self, env_value, probe):
        """build.f64_soft() with LEONIDA_F64_SOFT set to env_value (None: unset) and the probe answering probe."""
        env = {k: v for k, v in os.environ.items() if k != "LEONIDA_F64_SOFT"}
        if env_value is not None:
            env["LEONIDA_F64_SOFT"] = env_value
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(vk_subgroup_info, "harness_info", return_value=probe) as probe_call:
            return self.build.f64_soft(), probe_call.call_count

    def test_one_turns_it_on_without_asking_the_device(self):
        for v in ("1", "true", "on", " YES "):
            with self.subTest(value=v):
                self.assertEqual(self.wanted(v, {"float64": True}), (True, 0))

    def test_zero_turns_it_off_without_asking_the_device(self):
        for v in ("0", "false", "off", "No"):
            with self.subTest(value=v):
                self.assertEqual(self.wanted(v, {"float64": False}), (False, 0))

    def test_unset_follows_the_harness_device(self):
        self.assertEqual(self.wanted(None, {"float64": False}), (True, 1))     # MoltenVK, KosmicKrisp, Intel Arc
        self.assertEqual(self.wanted(None, {"float64": True}), (False, 1))
        self.assertEqual(self.wanted(None, {}), (False, 1))                    # no answer: native doubles
        self.assertEqual(self.wanted("", {"float64": False}), (True, 1))       # empty is unset

    def test_translator_command(self):
        """translate_tests passes --f64-soft to rdna2spirv.py exactly when f64_soft() says so."""
        for on in (True, False):
            with self.subTest(f64_soft=on), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(self.build, "f64_soft", return_value=on), \
                    mock.patch.object(self.build.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
                self.assertEqual(self.build.translate_tests(Path(tmp), []), 0)
                cmd = run.call_args[0][0]
                self.assertEqual(os.path.basename(cmd[2]), "rdna2spirv.py")
                self.assertEqual("--f64-soft" in cmd, on)


if __name__ == "__main__":
    unittest.main()
