"""scripts/run_gpu_tests.py store_result: vk_store_test's exit 77 is a SKIP with the helper's unsupported line, as
the other helpers' 77 is reported, not a FAIL that hides the reason (found on Intel UHD by @richardsenra, #2463)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt  # noqa: E402


class StoreResult(unittest.TestCase):
    def test_pass_and_fail_keep_the_last_line(self):
        self.assertEqual(rgt.store_result(0, "device: X\nok 4096 texels\n"), (True, "ok 4096 texels"))
        self.assertEqual(rgt.store_result(1, "device: X\nFAILED: texel 3\n"), (False, "FAILED: texel 3"))
        self.assertEqual(rgt.store_result(1, ""), (False, ""))

    def test_unsupported_is_a_skip_with_its_reason(self):
        msg = "device: X\nunsupported: shaderStorageImageWriteWithoutFormat=0\n"
        self.assertEqual(rgt.store_result(77, msg), (None, "unsupported: shaderStorageImageWriteWithoutFormat=0"))
        msg = "device: X\nunsupported: shaderInt64=0 and cs_x.spv declares Int64\nexit\n"
        self.assertEqual(rgt.store_result(77, msg), (None, "unsupported: shaderInt64=0 and cs_x.spv declares Int64"))

    def test_unsupported_without_a_reason_line(self):
        self.assertEqual(rgt.store_result(77, "device: X\nbye\n"), (None, "bye"))
        self.assertEqual(rgt.store_result(77, ""), (None, "unsupported"))


if __name__ == "__main__":
    unittest.main()
