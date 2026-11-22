"""runtime/test/shaders/fallbacks.py: the GPU tests turn an exact fallback on where the device lacks the native
feature, and LEONIDA_* = 1 or 0 forces it (no GPU needed: the device's answer is faked)."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import fallbacks

VARS = {name: spec[0] for name, spec in fallbacks.FALLBACKS.items()}
# tools/vk_subgroup_info.py on the three Mac drivers (Apple M3 Max): MoltenVK 1.4.2, KosmicKrisp and llvmpipe
MOLTENVK = {"subgroup_size": 32, "min_subgroup_size": 4, "max_subgroup_size": 32, "subgroup_size_control": True,
            "compute_in_required_stages": False, "compute_in_supported_stages": True, "per_vertex_inputs": False,
            "mesh_shader": False, "geometry_shader": False}
LLVMPIPE = {"subgroup_size": 4, "min_subgroup_size": 4, "max_subgroup_size": 4, "subgroup_size_control": True,
            "compute_in_required_stages": True, "compute_in_supported_stages": True, "per_vertex_inputs": False,
            "mesh_shader": True, "geometry_shader": True, "driver_id": fallbacks.VK_DRIVER_ID_MESA_LLVMPIPE}
RDNA = {"subgroup_size": 64, "min_subgroup_size": 32, "max_subgroup_size": 64, "subgroup_size_control": True,
        "compute_in_required_stages": True, "compute_in_supported_stages": True, "per_vertex_inputs": True,
        "mesh_shader": True, "geometry_shader": True}


def choices(device, env=None):
    clean = {k: v for k, v in os.environ.items() if k not in VARS.values()}
    with patch.dict(os.environ, {**clean, **(env or {})}, clear=True), \
            patch.object(fallbacks, "device", return_value=device):
        return {name: fallbacks.enabled(name) for name in fallbacks.FALLBACKS}


class DeviceChoice(unittest.TestCase):
    def test_moltenvk_needs_every_fallback(self):
        self.assertEqual(choices(MOLTENVK), dict.fromkeys(fallbacks.FALLBACKS, True))

    def test_llvmpipe_emulates_wave64_and_keeps_its_geometry_stage(self):
        # its emulated mesh shaders come out wrong (no primitives), so under --wave64-emulate they run as compute
        self.assertEqual(choices(LLVMPIPE), {"w64_emulate": True, "mesh_compute": True, "gs_compute": False,
                                             "f16_interp": True})
        with patch.dict(os.environ, {"LEONIDA_W64_EMULATE": "1"}), \
                patch.object(fallbacks, "device", return_value=LLVMPIPE):
            self.assertIn("llvmpipe", fallbacks.describe(("mesh_compute",)))

    def test_llvmpipe_without_emulation_keeps_its_mesh_stage(self):
        self.assertEqual(choices(LLVMPIPE, {"LEONIDA_W64_EMULATE": "0"}),
                         {"w64_emulate": False, "mesh_compute": False, "gs_compute": False, "f16_interp": True})

    def test_a_32_wide_gpu_with_mesh_shaders_keeps_them_under_emulation(self):
        nvidia = {**RDNA, "subgroup_size": 32, "max_subgroup_size": 32, "min_subgroup_size": 32, "driver_id": 4}
        self.assertEqual(choices(nvidia), {"w64_emulate": True, "mesh_compute": False, "gs_compute": False,
                                           "f16_interp": False})

    def test_a_native_wave64_gpu_needs_none(self):
        self.assertEqual(choices(RDNA), dict.fromkeys(fallbacks.FALLBACKS, False))

    def test_an_unknown_device_turns_nothing_on(self):
        self.assertEqual(choices({}), dict.fromkeys(fallbacks.FALLBACKS, False))


class Variables(unittest.TestCase):
    def test_zero_forces_off_and_one_forces_on(self):
        off = {var: "0" for var in VARS.values()}
        self.assertEqual(choices(MOLTENVK, off), dict.fromkeys(fallbacks.FALLBACKS, False))
        on = {var: "1" for var in VARS.values()}
        self.assertEqual(choices(RDNA, on), dict.fromkeys(fallbacks.FALLBACKS, True))

    def test_yes_no_words_and_cmd_exe_trailing_space(self):
        for value, want in (("true", True), ("ON", True), ("1 ", True), ("no", False), ("off", False)):
            with self.subTest(value=value):
                self.assertIs(choices(MOLTENVK if not want else RDNA,
                                      {"LEONIDA_W64_EMULATE": value})["w64_emulate"], want)

    def test_another_word_leaves_the_choice_to_the_device(self):
        self.assertTrue(choices(MOLTENVK, {"LEONIDA_GS_COMPUTE": "maybe"})["gs_compute"])

    def test_forced_never_asks_the_device(self):
        with patch.dict(os.environ, {"LEONIDA_MESH_COMPUTE": "1"}), \
                patch.object(fallbacks, "device", side_effect=AssertionError("asked")):
            self.assertTrue(fallbacks.enabled("mesh_compute"))


class ComputeNative64(unittest.TestCase):
    def test_required_or_default_64(self):
        self.assertTrue(fallbacks.compute_native64(RDNA))
        self.assertTrue(fallbacks.compute_native64({"subgroup_size": 64, "compute_in_supported_stages": True}))
        self.assertFalse(fallbacks.compute_native64(MOLTENVK))
        self.assertFalse(fallbacks.compute_native64({**RDNA, "compute_in_required_stages": False,
                                                     "subgroup_size": 32}))
        self.assertIsNone(fallbacks.compute_native64({}))


class Describe(unittest.TestCase):
    def test_names_what_is_on_and_why(self):
        with patch.dict(os.environ, {"LEONIDA_MESH_COMPUTE": "1", "LEONIDA_W64_EMULATE": "0",
                                     "LEONIDA_GS_COMPUTE": "", "LEONIDA_F16_INTERP_FALLBACK": ""}), \
                patch.object(fallbacks, "device", return_value=MOLTENVK):
            line = fallbacks.describe()
        self.assertEqual(line, "fallbacks: --mesh-compute (LEONIDA_MESH_COMPUTE=1), gs compute (no geometryShader; "
                               "LEONIDA_GS_COMPUTE=0 turns it off), --f16-interp-fallback (no PerVertexKHR inputs; "
                               "LEONIDA_F16_INTERP_FALLBACK=0 turns it off)")

    def test_nothing_on_prints_nothing(self):
        clean = {k: v for k, v in os.environ.items() if k not in VARS.values()}
        with patch.dict(os.environ, clean, clear=True), patch.object(fallbacks, "device", return_value=RDNA):
            self.assertEqual(fallbacks.describe(), "")


class DeviceQuery(unittest.TestCase):
    def test_an_error_answer_is_no_answer(self):
        import subprocess
        done = subprocess.CompletedProcess([], 1, '{"error": "RuntimeError: no Vulkan device"}\n', "")
        with patch.object(fallbacks, "_DEVICE", []), patch.object(fallbacks.subprocess, "run", return_value=done):
            self.assertEqual(fallbacks.device(), {})

    def test_asked_once(self):
        import subprocess
        done = subprocess.CompletedProcess([], 0, '{"mesh_shader": false}\n', "")
        with patch.object(fallbacks, "_DEVICE", []), \
                patch.object(fallbacks.subprocess, "run", return_value=done) as run:
            self.assertEqual(fallbacks.device(), {"mesh_shader": False})
            self.assertEqual(fallbacks.device(), {"mesh_shader": False})
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
