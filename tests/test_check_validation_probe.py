"""The verdict of runtime/test/check_validation_probe.py on synthetic runs: no GPU, no Wine."""
import contextlib
import importlib.util
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("check_validation_probe",
                                              ROOT / "runtime" / "test" / "check_validation_probe.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

UNLISTED = ("LEONIDA_VK_VALIDATION: the Vulkan loader does not list VK_LAYER_KHRONOS_validation; relying on a layer "
            "enabled outside this program (VK_INSTANCE_LAYERS), checked by a probe")
NOT_ACTIVE = "LEONIDA_VK_VALIDATION is set but validation is not active: a deliberate invalid call was not reported"
ACTIVE = "validation probe: reported, validation is active"


def verdict(*runs):
    """main()'s exit code when the test's runs give these (exit code, output) pairs in turn."""
    results = [subprocess.CompletedProcess([], code, stdout=output, stderr="") for code, output in runs]
    with mock.patch.object(sys, "argv", ["check_validation_probe.py", "d3d12_com_test.exe"]), \
            mock.patch.object(check.subprocess, "run", side_effect=results), contextlib.redirect_stdout(io.StringIO()):
        return check.main()


class VerdictTest(unittest.TestCase):
    def test_reported_with_the_layer_and_not_active_without_it_passes(self):
        self.assertEqual(verdict((0, ACTIVE), (1, NOT_ACTIVE)), 0)

    def test_a_layer_the_loader_does_not_list_and_nothing_reports_is_not_installed(self):
        self.assertEqual(verdict((1, f"{UNLISTED}\n{NOT_ACTIVE}")), check.SKIP)

    def test_a_loader_that_refuses_the_layer_means_not_installed(self):
        self.assertEqual(verdict((1, "vkCreateInstance failed (VkResult -6)")), check.SKIP)

    def test_another_instance_failure_with_the_layer_fails(self):
        self.assertEqual(verdict((1, "vkCreateInstance failed (VkResult -3)")), 1)

    def test_a_listed_layer_that_reports_nothing_fails(self):
        self.assertEqual(verdict((1, NOT_ACTIVE)), 1)

    def test_another_instance_failure_without_the_layer_fails(self):
        self.assertEqual(verdict((0, ACTIVE), (1, "vkCreateInstance failed (VkResult -3)")), 1)


if __name__ == "__main__":
    unittest.main()
