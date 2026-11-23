import os
import shutil
import subprocess
import tempfile
import unittest
import support


class ImageSampleLLodTests(unittest.TestCase):
    def test_image_sample_l_runtime_descriptor_uses_lod_operand(self):
        lines = [
            "v_mov_b32_e32 v0, 0",
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 0x3f800000",
            "image_sample_l v[0:3], v[1:2], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_sample_l_lod", lines)
        self.assertIn("OpImageSampleExplicitLod", spv)
        self.assertNotIn("Lod %f0", spv, "image_sample_l must use the instruction's LOD operand, not LOD 0")
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            with tempfile.TemporaryDirectory() as tmp:
                sa = os.path.join(tmp, "s.spvasm")
                sv = os.path.join(tmp, "s.spv")
                with open(sa, "w", encoding="utf-8") as f:
                    f.write(spv)
                r1 = subprocess.run(["spirv-as", sa, "-o", sv, "--target-env", "vulkan1.2"], capture_output=True, text=True)
                self.assertEqual(r1.returncode, 0, r1.stderr)
                r2 = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", sv], capture_output=True, text=True)
                self.assertEqual(r2.returncode, 0, r2.stderr)


if __name__ == "__main__":
    unittest.main()
