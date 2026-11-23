"""#5273: check.py's mesh_w64x_test asks for a 64-wide subgroup only of ms_w64x_ballot.

ms_wave_info_s3's wave id is LocalInvocationIndex >> 6, the shift tools/rdna2spirv.py emits for the s3 prologue
(see test_translate.py's test_mesh_s3_wave_id_emits_local_index_shift_and_bfe), so its picture does not depend on
the real subgroup width: the main runner draws 4096/4096 correct pixels on a 32-wide device and gates only
ms_w64x_ballot on the width. The standalone checker skipped ms_wave_info_s3 too, before it read a single pixel.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
import check


def gradient(blue=0.5):
    """The 64x64 picture mesh_w64x_test wants: ((x+.5)/64, (y+.5)/64, blue, 1.0)."""
    return b"".join(struct.pack("<4f", (i % 64 + .5) / 64, (i // 64 + .5) / 64, blue, 1.0)
                    for i in range(64 * 64))


class MeshWidthRequirement(unittest.TestCase):
    """mesh_w64x_test with a fake vk_harness that writes that picture and reports one subgroup width."""

    def run_mesh(self, name, stderr, pixels=None, emulated=False):
        def harness(cmd, **_kw):
            Path(cmd[3]).write_bytes(gradient() if pixels is None else pixels)
            return subprocess.CompletedProcess(cmd, 0, "", stderr)

        # check.py reads LEONIDA_W64_EMULATE once, at import (check.py:294), so pin the flag rather than inherit
        # whatever the shell running this test happens to have set.
        with tempfile.TemporaryDirectory() as tmp:
            for spv in (f"{name}.spv", "fs_passthru.spv"):
                (Path(tmp) / spv).touch()
            with (patch.object(check.subprocess, "run", side_effect=harness),
                  patch.object(check, "W64_EMULATED", emulated)):
                return check.mesh_w64x_test(tmp, tmp, name)

    def test_wave_info_s3_is_checked_on_a_32_wide_subgroup(self):
        """The bug: this returned None (skip) instead of checking the picture."""
        self.assertEqual(self.run_mesh("ms_wave_info_s3", "device: fake (subgroup 32..32)\n"), "")

    def test_wave_info_s3_still_reports_a_wrong_picture(self):
        """Checked at 32 wide, not waved through: blue 0 instead of 0.5 fails."""
        problem = self.run_mesh("ms_wave_info_s3", "device: fake (subgroup 32..32)\n", gradient(blue=0.0))
        self.assertIn("4096 of 4096 pixels differ", problem)

    def test_ballot_still_skips_without_a_64_wide_subgroup(self):
        """The native ballot fixture keeps its width requirement."""
        self.assertIsNone(self.run_mesh("ms_w64x_ballot", "device: fake (subgroup 32..32)\n"))

    def test_ballot_is_checked_when_the_subgroup_is_64_wide(self):
        self.assertEqual(self.run_mesh("ms_w64x_ballot", "device: fake (subgroup 64..64)\n"), "")

    def test_wave_info_s3_is_checked_on_a_64_wide_subgroup(self):
        self.assertEqual(self.run_mesh("ms_wave_info_s3", "device: fake (subgroup 64..64)\n"), "")

    def test_emulation_drops_the_width_requirement(self):
        """LEONIDA_W64_EMULATE=1: neither fixture needs a native 64-wide subgroup."""
        self.assertEqual(self.run_mesh("ms_w64x_ballot", "device: fake (subgroup 32..32)\n", emulated=True), "")
        self.assertEqual(self.run_mesh("ms_wave_info_s3", "device: fake (subgroup 32..32)\n", emulated=True), "")


if __name__ == "__main__":
    unittest.main()
