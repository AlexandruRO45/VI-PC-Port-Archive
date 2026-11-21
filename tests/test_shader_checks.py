"""Exercise the synthetic Vulkan test runner without a Vulkan device."""
import contextlib
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime" / "test" / "shaders"))
import check


# The other wave64-only dump tests (#73's cs_wm_*, the cs_w64x_* cross-half tests): name -> expected out[] words.
WAVE64_DUMPS = {n: w for n, w in check.CS_DUMPS + check.W64X_DUMPS + check.W64X_PARTIAL
                if n not in (*check.WAVE32_DUMPS, "cs_wave64_ballot")}
# A group smaller than a wave can't ask for full subgroups: it asks for subgroup size 64 and reads the device's range.
PARTIAL = {n for n, _ in check.W64X_PARTIAL}


# every test check.py runs: compute, cs_wavemask, the dumps (#73, #152 cross-half, partial), the two store
# tests, pixel tests, and ms_w64x_ballot (skipped here: build.py does not build mesh shaders)
TOTAL = (len(check.COMPUTE) + 1 + len(check.CS_DUMPS) + len(check.W64X_DUMPS) + len(check.W64X_PARTIAL) + 2
         + len(check.PIXEL_TESTS) + 1)


class Wave64Checks(unittest.TestCase):
    def run_checks(self, wavemask, ballot, *, missing_ballot=False, emulated=False):
        def harness(_tool, shader, *args):
            name = Path(shader).stem
            if name in PARTIAL:
                self.assertNotIn("--require-wave64", args)
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                narrow = wavemask[0] == 77
                return 0, f"device: gpu (subgroup 32..{32 if narrow else 64})\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if emulated:
                self.assertNotIn("--require-wave64", args)
                if name in WAVE64_DUMPS:
                    words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                    return 0, "out[0..7] = " + " ".join(map(str, words)) + "\n"
                if name == "cs_wavemask":
                    return 0, "0 mismatches\n"
                if name == "cs_wave64_ballot":
                    return 0, "out[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"
            if name in WAVE64_DUMPS:
                self.assertIn("--require-wave64", args)
                if wavemask[0] == 77:          # same device as cs_wavemask: no 64-wide subgroups
                    return wavemask
                words = (WAVE64_DUMPS[name] + [0] * 8)[:8]
                return 0, "compute subgroup size: 64\nout[0..7] = " + " ".join(map(str, words)) + "\n"
            if name in ("cs_wavemask", "cs_wave64_ballot"):
                self.assertIn("--require-wave64", args)
                if name == "cs_wavemask":
                    return wavemask
                self.assertFalse(missing_ballot, "missing shader was sent to the harness")
                return ballot
            self.assertNotIn("--require-wave64", args)
            if name == "cs_store":
                return 0, "out[0..7] = 1073774847 3758097407 1006649344 4008636142 1073757184 3154130944 0 0\n"
            if name == "cs_smem_soffset_imm":
                return 0, "out[0..7] = " + " ".join(map(str, check.CS_SMEM)) + "\n"
            if name == "cs_workgroup_yz":     # #186: vk_compute_test's 2x3x4 Y/Z histogram line
                return 0, "2x3x4 workgroup Y/Z: 0 mismatches\n"
            if name in ("cs_imgstore", "cs_imgload"):
                return 0, "0 bad\n0 bad\n0 bad\n"
            return 0, "0 mismatches\n"

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / "cs_wavemask.spv").touch()
            for name in WAVE64_DUMPS:
                (spirv / f"{name}.spv").touch()
            if not missing_ballot:
                (spirv / "cs_wave64_ballot.spv").touch()
            output = io.StringIO()
            with (patch.object(sys, "argv", ["check.py", tmp]), patch.object(check, "run", side_effect=harness),
                  patch.object(check, "pixel_test", return_value=""), patch.object(check, "W64_EMULATED", emulated),
                  contextlib.redirect_stdout(output)):
                status = check.main()
        return status, output.getvalue()

    def test_wave32_only_device_skips_full_wave_checks(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wavemask +SKIP")
        self.assertRegex(out, r"cs_wave64_ballot +SKIP")
        self.assertRegex(out, r"cs_exec64 +ok")
        self.assertIn(f"{TOTAL - 3 - len(WAVE64_DUMPS)} of {TOTAL} pass, {3 + len(WAVE64_DUMPS)} skipped", out)

    def test_full_wave_checks_pass_with_expected_results(self):
        status, out = self.run_checks(
            (0, "compute subgroup size: 64\n0 mismatches\n"),
            (0, "compute subgroup size: 64\nout[0..7] = 32 16 24 40 8 48 4008636142 4008636142\n"),
        )
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_wave64_ballot +ok")
        self.assertIn(f"{TOTAL - 1} of {TOTAL} pass, 1 skipped", out)

    def test_emulated_wave64_runs_without_a_64_wide_subgroup(self):
        # LEONIDA_W64_EMULATE=1: the shaders were translated with --wave64-emulate, so no wave64 test asks the
        # harness for a 64-wide subgroup, and a 32-wide device runs them all.
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, emulated=True)
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"cs_w64x_wg128_branches +ok")
        self.assertIn(f"{TOTAL - 1} of {TOTAL} pass, 1 skipped", out)

    def test_ballot_mismatches_and_harness_errors_fail(self):
        wavemask = 0, "compute subgroup size: 64\n0 mismatches\n"
        for ballot, detail in (
            ((0, "compute subgroup size: 64\nout[0..7] = 16 16 0 64 0 64 0 0\n"), "exit 0"),
            ((0, "compute subgroup size: 64\nout[0..7] = corrupt\n"), "exit 0"),
            ((0, "out[0..7] = 32 16 24 40 8 48 0 0\n"), "64-wide subgroup was not confirmed"),
            ((1, "Vulkan pipeline failed\n"), "exit 1"),
        ):
            with self.subTest(ballot=ballot):
                status, out = self.run_checks(wavemask, ballot)
                self.assertEqual(status, 1, out)
                self.assertRegex(out, rf"cs_wave64_ballot +{detail}")
                self.assertIn(f"{TOTAL - 2} of {TOTAL} pass, 1 skipped", out)

    def test_missing_wave64_fixture_fails_even_on_wave32(self):
        unsupported = 77, "full 64-wide compute subgroups unavailable\n"
        status, out = self.run_checks(unsupported, unsupported, missing_ballot=True)
        self.assertEqual(status, 1, out)
        self.assertIn("missing cs_wave64_ballot.spv", out)


class PixelHarnessChecks(unittest.TestCase):
    """The pixel path of check.py against a fake vk_harness: the UNSUPPORTED exit (#203) skips, and
    ps_bitmask_perm is only judged where the device has the 32 lanes the check reads."""

    @staticmethod
    def device_line(width: int) -> str:
        return f"device: fake (subgroup {width}..{width})\n"

    @staticmethod
    def lane_ids_raw() -> bytes:
        # 64x64 pixels carrying lane ids cycling 0-31 without the permuted bitmasks: judged, they report
        # mismatches, so a skip and a verdict are tellingly different
        return b"".join(struct.pack("<4I", i % 32, i % 32, 0, 0) for i in range(64 * 64))

    def run_pixel(self, name: str, code: int, out: str, raw: bytes = b""):
        def harness(_tool, _vs, _fs, raw_path, env=None):
            Path(raw_path).write_bytes(raw)
            return code, out

        with tempfile.TemporaryDirectory() as tmp:
            spirv = Path(tmp) / "spirv"
            spirv.mkdir()
            (spirv / f"{name}.spv").touch()
            (spirv / "vs_ids.spv").touch()
            with patch.object(check, "run", side_effect=harness):
                return check.pixel_test(tmp, spirv, name)

    def test_unsupported_harness_exit_is_a_skip(self):
        out = self.device_line(8) + ("unsupported: ps_bary.spv reads BaryCoordKHR and the device has no "
                                     "VK_KHR_fragment_shader_barycentric\n")
        problem = self.run_pixel("ps_bary", 77, out)
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("SKIP (unsupported:", problem)

    def test_other_harness_errors_stay_failures(self):
        problem = self.run_pixel("ps_bary", 1, self.device_line(8) + "VkResult -9\n")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertTrue(problem, "a failing harness must not read as ok")

    def test_bitmask_perm_skips_on_narrow_subgroups(self):
        problem = self.run_pixel("ps_bitmask_perm", 0, self.device_line(8), self.lane_ids_raw())
        self.assertIsInstance(problem, check.Skip)
        self.assertIn("subgroup 8 < 32", problem)

    def test_bitmask_perm_is_judged_on_32_lane_subgroups(self):
        problem = self.run_pixel("ps_bitmask_perm", 0, self.device_line(32), self.lane_ids_raw())
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("wrong", problem)   # the fake lanes do not carry the wanted bitmasks


if __name__ == "__main__":
    unittest.main()
