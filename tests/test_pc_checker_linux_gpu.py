"""Linux GPU identification from synthetic PCI and Vulkan reports; no GPU or game files needed."""
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import pc_checker as pc  # noqa: E402


AMD_IGPU = ('65:00.0 "VGA compatible controller [0300]" '
            '"Advanced Micro Devices, Inc. [AMD/ATI] [1002]" "HawkPoint1 [1900]"')
AMD_DGPU = ('03:00.0 "Display controller [0380]" '
            '"Advanced Micro Devices, Inc. [AMD/ATI] [1002]" "Navi 32 [747e]"')
NVIDIA = ('01:00.0 "3D controller [0302]" "NVIDIA Corporation [10de]" '
          '"AD104 [GeForce RTX 4070] [2786]"')
VULKAN_IGPU = {"device": "AMD Radeon 780M Graphics (RADV PHOENIX)",
               "vendor_id": 0x1002, "device_id": 0x1900}


def pci_output(cmd, report):
    if "-nn" in cmd:
        return report
    report = re.sub(r"\bDevice \[([0-9a-fA-F]{4})\]", r"Device \1", report)
    return re.sub(r" \[[0-9a-fA-F]{4}\]", "", report)


class LinuxGpuIdentification(unittest.TestCase):
    def detect(self, pci=AMD_IGPU, vulkan=VULKAN_IGPU, smi=None):
        def output(cmd, **kwargs):
            if cmd[0] == "lspci":
                return pci_output(cmd, pci)
            if cmd[0] == "nvidia-smi":
                return smi or ""
            return vulkan if isinstance(vulkan, str) else json.dumps(vulkan)

        with mock.patch.object(pc.sys, "platform", "linux"), \
                mock.patch.object(pc.shutil, "which", return_value="nvidia-smi" if smi else None), \
                mock.patch.object(pc, "_run", side_effect=output) as run:
            result = pc.detect_gpus()
        return result, run

    def test_resolves_matching_pci_codename(self):
        result, _ = self.detect()
        self.assertEqual(result, [(VULKAN_IGPU["device"], None)])

    def test_keeps_recognized_pci_product_without_probing(self):
        result, run = self.detect(pci=NVIDIA)
        self.assertEqual(result, [("NVIDIA GeForce RTX 4070", None)])
        self.assertEqual(len(run.call_args_list), 1)

    def test_matches_device_id_not_just_vendor(self):
        result, _ = self.detect(pci=AMD_DGPU + "\n" + AMD_IGPU)
        self.assertEqual(result, [("AMD Navi 32", None), (VULKAN_IGPU["device"], None)])

    def test_unmatched_device_does_not_rename_another_adapter(self):
        result, _ = self.detect(pci=AMD_DGPU)
        self.assertEqual(result, [("AMD Navi 32", None)])
        result, _ = self.detect(vulkan={**VULKAN_IGPU, "vendor_id": 0x8086})
        self.assertEqual(result, [("AMD HawkPoint1", None)])

    def test_ambiguous_duplicate_pci_ids_keep_the_pci_names(self):
        result, _ = self.detect(pci=AMD_IGPU + "\n" + AMD_IGPU.replace("65:00.0", "66:00.0"))
        self.assertEqual(result, [("AMD HawkPoint1", None)] * 2)

    def test_failed_or_malformed_vulkan_report_keeps_pci_fallback(self):
        for report in ("", "not JSON", "[]", "null", "{}",
                       {**VULKAN_IGPU, "device": ""},
                       {**VULKAN_IGPU, "device": None},
                       {**VULKAN_IGPU, "vendor_id": "1002"},
                       {**VULKAN_IGPU, "vendor_id": True},
                       {**VULKAN_IGPU, "device_id": -1}):
            with self.subTest(report=report):
                result, _ = self.detect(vulkan=report)
                self.assertEqual(result, [("AMD HawkPoint1", None)])

    def test_no_numeric_pci_identity_keeps_fallback(self):
        result, _ = self.detect(pci='65:00.0 "VGA compatible controller" "AMD" "HawkPoint1"')
        self.assertEqual(result, [("AMD HawkPoint1", None)])

    def test_unknown_pci_product_retains_its_numeric_identity(self):
        result, _ = self.detect(pci=AMD_IGPU.replace("HawkPoint1", "Device"), vulkan="")
        self.assertEqual(result, [("AMD Device 1900", None)])

    def test_nvidia_smi_name_and_vram_are_preserved(self):
        result, _ = self.detect(pci=NVIDIA + "\n" + AMD_IGPU,
                                smi="NVIDIA GeForce RTX 4070, 12282")
        self.assertEqual(result, [("NVIDIA GeForce RTX 4070", 12), (VULKAN_IGPU["device"], None)])

    def test_vulkan_query_timeout_keeps_pci_fallback(self):
        def run(cmd, **kwargs):
            if cmd[0] == "lspci":
                return subprocess.CompletedProcess(cmd, 0, stdout=pci_output(cmd, AMD_IGPU), stderr="")
            self.assertGreater(kwargs["timeout"], 0)
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

        with mock.patch.object(pc.sys, "platform", "linux"), \
                mock.patch.object(pc.shutil, "which", return_value=None), \
                mock.patch.object(pc.subprocess, "run", side_effect=run):
            self.assertEqual(pc.detect_gpus(), [("AMD HawkPoint1", None)])

    def test_resolved_name_does_not_invent_a_performance_score(self):
        result, _ = self.detect()
        with mock.patch.object(pc, "detect_gpus", return_value=result), \
                mock.patch.object(pc, "detect_cpu", return_value="AMD Ryzen 5 5600"), \
                mock.patch.object(pc, "detect_ram_gb", return_value=32):
            report = pc.SpecChecker().check(cpu="AMD Ryzen 5 5600", ram=32)
        gpu = next(check for check in report.checks if check.component == "GPU")
        self.assertEqual(gpu.detected, VULKAN_IGPU["device"])
        self.assertEqual(gpu.status, pc.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
