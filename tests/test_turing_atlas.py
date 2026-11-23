"""Exercise the GPU atlas with synthetic vulkaninfo reports."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import turing_atlas

MATRIX_KEYS = ("vulkaninfo", "device", "subgroup", "subgroup_size_control", "mesh",
               "barycentric", "maintenance8", "descriptor_heap", "storageInputOutput16")


def report():
    return {"capabilities": {"device": {
        "extensions": {"VK_EXT_mesh_shader": 1},
        "features": {},
        "properties": {
            "VkPhysicalDeviceProperties": {"deviceName": "Example GPU", "driverVersion": 7, "apiVersion": 42},
        },
    }}}


class VulkanMatrixTests(unittest.TestCase):
    def probe(self, document, returncode=0):
        def run(cmd, cwd):
            Path(cwd, "VP_VULKANINFO_Example.json").write_text(json.dumps(document), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode)

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                return turing_atlas.vulkan_matrix()

    def assert_unknown(self, matrix):
        self.assertEqual(matrix, dict.fromkeys(MATRIX_KEYS))

    def test_recognized_profile(self):
        doc = report()
        device = doc["capabilities"]["device"]
        device["properties"]["VkPhysicalDeviceVulkan11Properties"] = {"subgroupSize": 32}
        device["properties"]["VkPhysicalDeviceVulkan13Properties"] = {"minSubgroupSize": 16, "maxSubgroupSize": 64}
        device["features"]["VkPhysicalDeviceVulkan11Features"] = {"storageInputOutput16": False}
        device["features"]["VkPhysicalDeviceVulkan13Features"] = {"subgroupSizeControl": True}
        matrix = self.probe(doc)
        self.assertEqual(tuple(matrix), MATRIX_KEYS)
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"], {"name": "Example GPU", "driver": 7, "api": 42})
        self.assertEqual(matrix["subgroup"]["size"], 32)
        self.assertEqual(matrix["subgroup"]["min"], 16)
        self.assertEqual(matrix["subgroup"]["max"], 64)
        self.assertIs(matrix["subgroup_size_control"]["subgroupSizeControl"], True)
        self.assertIs(matrix["mesh"], True)
        self.assertIs(matrix["barycentric"], False)
        self.assertIs(matrix["storageInputOutput16"], False)

    def test_older_device_without_versioned_structs(self):
        matrix = self.probe(report())
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertIsNone(matrix["subgroup"]["size"])
        self.assertIsNone(matrix["subgroup_size_control"]["subgroupSizeControl"])
        self.assertIsNone(matrix["storageInputOutput16"])

    def test_failed_subprocess_discards_even_a_valid_report(self):
        self.assert_unknown(self.probe(report(), returncode=1))

    def test_unusable_profiles_leave_the_matrix_unknown(self):
        no_name = report()
        del no_name["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"]["deviceName"]
        missing_extensions = report()
        del missing_extensions["capabilities"]["device"]["extensions"]
        invalid_extensions = report()
        invalid_extensions["capabilities"]["device"]["extensions"] = []
        invalid_feature_block = report()
        invalid_feature_block["capabilities"]["device"]["features"]["VkPhysicalDeviceVulkan11Features"] = []
        for doc in ([], {}, {"capabilities": []}, no_name, missing_extensions, invalid_extensions, invalid_feature_block):
            with self.subTest(document=doc):
                self.assert_unknown(self.probe(doc))

    def test_multiple_profiles_pick_native_discrete_over_igpu_dozen_and_corrupt(self):
        dozen = report()
        dozen["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Microsoft Direct3D12 (Discrete)",
            "vendorID": 0x1414,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })
        igpu = report()
        igpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Integrated Graphics",
            "vendorID": 0x1002,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        })
        dgpu = report()
        dgpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Native Discrete GPU",
            "vendorID": 0x1002,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })

        def run(cmd, cwd):
            Path(cwd, "VP_VULKANINFO_0_corrupt.json").write_text("{not valid json", encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_1_dozen.json").write_text(json.dumps(dozen), encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_2_igpu.json").write_text(json.dumps(igpu), encoding="utf-8")
            Path(cwd, "VP_VULKANINFO_3_dgpu.json").write_text(json.dumps(dgpu), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0)

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                matrix = turing_atlas.vulkan_matrix()
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"]["name"], "Native Discrete GPU")

    def test_enumerates_gpus_from_summary_when_plain_j_only_writes_gpu0(self):
        llvmpipe = report()
        llvmpipe["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "llvmpipe (LLVM 19.1.0, 256 bits)",
            "vendorID": 0x10005,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_CPU",
        })
        dgpu = report()
        dgpu["capabilities"]["device"]["properties"]["VkPhysicalDeviceProperties"].update({
            "deviceName": "Discrete GPU 1",
            "vendorID": 0x10DE,
            "deviceType": "VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
        })
        calls = []

        def run(cmd, cwd):
            calls.append(tuple(cmd))
            if cmd == ["vulkaninfo", "-j"]:
                Path(cwd, "VP_VULKANINFO_llvmpipe.json").write_text(json.dumps(llvmpipe), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd == ["vulkaninfo", "--summary"]:
                out = "Devices:\n========\nGPU0:\n\tdeviceName = llvmpipe\nGPU1:\n\tdeviceName = Discrete GPU 1\n"
                return subprocess.CompletedProcess(cmd, 0, out, "")
            if cmd == ["vulkaninfo", "-j=1"]:
                Path(cwd, "VP_VULKANINFO_dgpu1.json").write_text(json.dumps(dgpu), encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 1, "", "")

        with patch.object(turing_atlas.shutil, "which", return_value="vulkaninfo"):
            with patch.object(turing_atlas, "run", run):
                matrix = turing_atlas.vulkan_matrix()
        self.assertEqual(matrix["vulkaninfo"], "ok")
        self.assertEqual(matrix["device"]["name"], "Discrete GPU 1")
        self.assertEqual(calls, [("vulkaninfo", "-j"), ("vulkaninfo", "--summary"), ("vulkaninfo", "-j=1")])

    def test_missing_vulkaninfo_falls_back_to_vk_subgroup_info(self):
        payload = {
            "device": "NVIDIA GeForce GTX 1070",
            "subgroup_size": 32,
            "min_subgroup_size": 32,
            "max_subgroup_size": 32,
            "subgroup_size_control": True,
            "per_vertex_inputs": False,
            "driver_id": 4,
            "supported_subgroup_stages": 63,
            "required_subgroup_size_stages": 32,
        }
        calls = []
        real_which = turing_atlas.shutil.which
        real_run = turing_atlas.run

        def which_without_vulkaninfo(name, *args, **kw):
            return None if name == "vulkaninfo" else real_which(name, *args, **kw)

        def fake_run(cmd, **kw):
            if len(cmd) >= 3 and cmd[:2] == [sys.executable, "-I"] and cmd[2].endswith("vk_subgroup_info.py"):
                calls.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, json.dumps(payload) + "\n", "")
            return real_run(cmd, **kw)

        with patch.object(turing_atlas.shutil, "which", side_effect=which_without_vulkaninfo), \
                patch.object(turing_atlas, "run", side_effect=fake_run):
            matrix = turing_atlas.vulkan_matrix()

        self.assertEqual(tuple(matrix), MATRIX_KEYS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], [sys.executable, "-I"])
        self.assertTrue(calls[0][2].endswith("vk_subgroup_info.py"))
        self.assertEqual(calls[0][3:], ["--stage-caps"])
        self.assertEqual(matrix["vulkaninfo"], "fallback:vk_subgroup_info")
        self.assertEqual(matrix["device"], {"name": "NVIDIA GeForce GTX 1070", "driver": None, "api": None})
        self.assertEqual(matrix["subgroup"], {"size": 32, "stages": 63, "min": 32, "max": 32})
        self.assertEqual(matrix["subgroup_size_control"], {
            "subgroupSizeControl": True,
            "computeFullSubgroups": None,
            "requiredSubgroupSizeStages": 32,
        })
        self.assertIs(matrix["barycentric"], False)
        self.assertIsNone(matrix["mesh"])
        self.assertIsNone(matrix["maintenance8"])
        self.assertIsNone(matrix["descriptor_heap"])
        self.assertIsNone(matrix["storageInputOutput16"])

        has_spirv_tools = bool(real_which("spirv-as") and real_which("spirv-val"))
        out = io.StringIO()
        with patch.object(sys, "argv", ["turing_atlas.py"]), \
                patch.object(turing_atlas.shutil, "which", side_effect=which_without_vulkaninfo), \
                patch.object(turing_atlas, "run", side_effect=fake_run), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = turing_atlas.main()
        self.assertEqual(code, 0)
        atlas = json.loads(out.getvalue())
        self.assertEqual(atlas["gpu"], "NVIDIA GeForce GTX 1070")
        self.assertEqual(atlas["features"]["vulkaninfo"], "fallback:vk_subgroup_info")
        if has_spirv_tools:
            self.assertNotIn("skipped", atlas["cases"])
            self.assertEqual(set(atlas["cases"]), set(turing_atlas.CASES))
            self.assertTrue(atlas["cases"]["compute32"]["valid"])
            self.assertTrue(atlas["cases"]["compute32"]["control_ok"])
            self.assertTrue(atlas["cases"]["broken_negative"]["control_ok"])
        else:
            self.assertIn("skipped", atlas["cases"])

    def test_fallback_guards_moltenvk_barycentric_with_real_schema(self):
        # Matches the exact fields emitted by tools/vk_subgroup_info.py --stage-caps on main.
        moltenvk = {
            "device": "Apple M2",
            "subgroup_size": 32,
            "min_subgroup_size": 32,
            "max_subgroup_size": 32,
            "subgroup_size_control": False,
            "compute_in_required_stages": False,
            "mesh_in_required_stages": False,
            "compute_in_supported_stages": True,
            "mesh_in_supported_stages": False,
            "native64_compute": False,
            "native64": False,
            "per_vertex_inputs": False,
            "barycentric_fallback": False,
            "max_push_constants_size": 4096,
            "vendor_id": 0x106B,
            "device_id": 0,
            "driver_id": 14,
            "rdna1_amd_windows": False,
            "rdna2_amd_windows": False,
            "supported_subgroup_stages": 63,
            "required_subgroup_size_stages": 0,
            "max_compute_workgroup_subgroups": 32,
        }
        with patch.object(turing_atlas.shutil, "which", return_value=None), \
                patch.object(turing_atlas, "run",
                             return_value=subprocess.CompletedProcess(["python"], 0, json.dumps(moltenvk) + "\n", "")):
            matrix = turing_atlas.vulkan_matrix()
        self.assertEqual(matrix["device"]["name"], "Apple M2")
        self.assertIsNone(matrix["barycentric"])
        self.assertIsNone(matrix["mesh"])
        self.assertIsNone(matrix["maintenance8"])
        self.assertIsNone(matrix["descriptor_heap"])
        self.assertIsNone(matrix["storageInputOutput16"])

    def test_fallback_forward_compat_optional_feature_booleans(self):
        # Forward-compat: if a newer vk_subgroup_info.py emits optional boolean feature probes, surface them.
        extended = {
            "device": "Future GPU",
            "subgroup_size": 32,
            "min_subgroup_size": 32,
            "max_subgroup_size": 64,
            "subgroup_size_control": True,
            "per_vertex_inputs": True,
            "driver_id": 4,
            "mesh_shader": False,
            "maintenance8": True,
        }
        with patch.object(turing_atlas.shutil, "which", return_value=None), \
                patch.object(turing_atlas, "run",
                             return_value=subprocess.CompletedProcess(["python"], 0, json.dumps(extended) + "\n", "")):
            matrix = turing_atlas.vulkan_matrix()
        self.assertIs(matrix["mesh"], False)
        self.assertIs(matrix["maintenance8"], True)
        self.assertIs(matrix["barycentric"], True)

    def test_fallback_failures_leave_the_matrix_unknown(self):
        cases = [
            subprocess.CompletedProcess(["python"], 1, json.dumps({"error": "no Vulkan loader"}), ""),
            subprocess.CompletedProcess(["python"], 0, json.dumps({"error": "failed"}), ""),
            subprocess.CompletedProcess(["python"], 0, "not-json\n", ""),
            subprocess.CompletedProcess(["python"], 0, "", ""),
            None,
        ]
        for result in cases:
            with self.subTest(result=result), \
                    patch.object(turing_atlas.shutil, "which", return_value=None), \
                    patch.object(turing_atlas, "run", return_value=result):
                self.assert_unknown(turing_atlas.vulkan_matrix())

        with patch.object(turing_atlas.shutil, "which", return_value=None), \
                patch.object(turing_atlas, "run", side_effect=subprocess.TimeoutExpired(["python"], 30)):
            self.assert_unknown(turing_atlas.vulkan_matrix())




class ValidationGateTests(unittest.TestCase):
    """The malformed control must fail, or the atlas results prove nothing. Tools are mocked."""

    def fake_run(self, accept_control=False, fail_assembly=False, fail_validation=False):
        def run(cmd, **kw):
            broken = any("broken_negative" in str(arg) for arg in cmd)
            if cmd[0] == "spirv-as":
                ok = 1 if fail_assembly and broken else 0
                return subprocess.CompletedProcess(cmd, ok, "", "syntax error" if ok else "")
            ok = 1 if fail_validation or (broken and not accept_control) else 0
            return subprocess.CompletedProcess(cmd, ok, "", "" if ok else "invalid module")
        return run

    def probe(self, accept_control=False, fail_assembly=False):
        with patch.object(turing_atlas.shutil, "which", return_value="tool"), \
                patch.object(turing_atlas, "run", self.fake_run(accept_control, fail_assembly)):
            return turing_atlas.validate_cases()

    def main_with(self, accept_control=False, fail_assembly=False, tools=True, argv=None, fail_validation=False):
        err = io.StringIO()
        with patch.object(sys, "argv", argv or ["turing_atlas.py"]), \
                patch.object(turing_atlas.shutil, "which", return_value="tool" if tools else None), \
                patch.object(turing_atlas, "run", self.fake_run(accept_control, fail_assembly, fail_validation)), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = turing_atlas.main()
        return code, err.getvalue()

    def test_valid_cases_are_reported_normally_with_a_rejected_control(self):
        cases = self.probe()
        self.assertEqual(len(cases), len(turing_atlas.CASES))
        self.assertTrue(all(cases[n]["assemble"] and cases[n]["valid"] for n in cases
                            if n != "broken_negative"))
        self.assertTrue(cases["compute32"]["control_ok"])
        control = cases["broken_negative"]
        self.assertTrue(control["assemble"])
        self.assertFalse(control["valid"])
        self.assertTrue(control["control_ok"])
        self.assertNotIn("error", control)

    def test_assembly_rejection_of_the_control_also_counts(self):
        control = self.probe(fail_assembly=True)["broken_negative"]
        self.assertFalse(control["assemble"])
        self.assertTrue(control["control_ok"])
        self.assertNotIn("error", control)

    def test_an_accepted_control_is_flagged(self):
        control = self.probe(accept_control=True)["broken_negative"]
        self.assertFalse(control["control_ok"])
        self.assertIn("not rejecting malformed modules", control["error"])

    def test_main_exits_nonzero_when_the_control_was_accepted(self):
        code, err = self.main_with(accept_control=True)
        self.assertEqual(code, 2)
        self.assertIn("turing_atlas: error:", err)
        self.assertIn("broken_negative", err)

    def test_main_fails_when_the_validator_rejects_the_known_good_control(self) -> None:
        code, err = self.main_with(fail_validation=True)
        self.assertEqual(code, 2)
        self.assertIn("compute32", err)

    def test_main_succeeds_when_the_control_was_rejected(self):
        for kwargs in ({}, {"fail_assembly": True}):
            with self.subTest(**kwargs):
                code, err = self.main_with(**kwargs)
                self.assertEqual(code, 0)
                self.assertEqual(err, "")

    def test_out_in_a_missing_folder_creates_it(self):
        # #934's class: a --out whose parent does not exist used to lose the whole run to a traceback.
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "dir", "atlas.json")
            code, err = self.main_with(argv=["turing_atlas.py", "--out", out])
            self.assertEqual((code, err), (0, ""))
            self.assertTrue(os.path.isfile(out))

    def test_reports_atlas_output_write_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "existing-file")
            with open(parent, "w", encoding="utf-8") as f:
                f.write("keep me")
            out = os.path.join(parent, "atlas.json")
            code, err = self.main_with(argv=["turing_atlas.py", "--out", out])

            self.assertEqual(code, 2)
            self.assertIn(f"cannot write {out}", err)
            with open(parent, encoding="utf-8") as f:
                self.assertEqual(f.read(), "keep me")

    def test_out_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, err = self.main_with(argv=["turing_atlas.py", "--out", tmp])
            self.assertEqual(code, 2)
            self.assertIn("--out is a folder", err)

            trailing = os.path.join(tmp, "new_folder") + os.sep
            code, err = self.main_with(argv=["turing_atlas.py", "--out", trailing])
            self.assertEqual(code, 2)
            self.assertIn("--out is a folder", err)

            for empty_out in ("", "   "):
                with self.subTest(empty_out=repr(empty_out)):
                    code, err = self.main_with(argv=["turing_atlas.py", "--out", empty_out])
                    self.assertEqual(code, 2)
                    self.assertIn("turing_atlas: error:", err)


    def test_missing_tools_skip_without_treating_it_as_a_control_failure(self):
        with patch.object(turing_atlas.shutil, "which", return_value=None):
            self.assertEqual(turing_atlas.validate_cases(),
                             {"skipped": "spirv-as/spirv-val not on PATH (Vulkan SDK)"})
        code, err = self.main_with(tools=False)
        self.assertEqual(code, 0)
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
