"""The translator's sampled 1D array type, address order and set 0 binding."""
import re
import shutil
import unittest

import support
from test_image_sample_bias_nonps import DESC, samples, source_vgpr, spirv_validate, translate


class TestTexture1DArraySampling(unittest.TestCase):
    def test_array_sample_coordinates_and_binding(self):
        for opcode, addresses, spirv_op in (
            ("image_sample", "v[4:5]", "OpImageSampleImplicitLod"),
            ("image_sample_lz", "v[4:5]", "OpImageSampleExplicitLod"),
            ("image_sample_l", "v[4:6]", "OpImageSampleExplicitLod"),
        ):
            with self.subTest(opcode=opcode):
                assembly = translate(
                    "ps_array1d", f"{opcode} v[0:3], {addresses}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY"
                )
                self.assertIn("OpCapability Sampled1D", assembly)
                self.assertIn("%img1da = OpTypeImage %float 1D 0 1 0 1 Unknown", assembly)
                self.assertIn("OpDecorate %images_1d_arr DescriptorSet 0", assembly)
                self.assertIn("OpDecorate %images_1d_arr Binding 26", assembly)
                meta = support.rdna2spirv.shader_meta(support.rdna2spirv.translate.last_gen, assembly)
                self.assertEqual(meta["bindings"], [0, 1, 2, 26])

                sample, = samples(assembly)
                _, _, operation, result_type, _, coord, *lod = sample.split()
                self.assertEqual((operation, result_type), (spirv_op, "%v4float"))
                match = re.search(rf"^\s*{re.escape(coord)} = OpCompositeConstruct %v2float (%\w+) (%\w+)",
                                  assembly, re.M)
                self.assertIsNotNone(match)
                self.assertEqual(tuple(source_vgpr(assembly, reg) for reg in match.groups()), (4, 5))
                if opcode == "image_sample_l":
                    self.assertEqual(lod[0], "Lod")
                    self.assertEqual(source_vgpr(assembly, lod[1]), 6)

    def test_single_layer_keeps_binding_five(self):
        assembly = translate("ps_1d", f"image_sample_l v[0:3], v[4:5], {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D")
        self.assertIn("%img1d = OpTypeImage %float 1D 0 0 0 1 Unknown", assembly)
        self.assertIn("OpDecorate %images_1d Binding 5", assembly)
        self.assertNotIn("Binding 26", assembly)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs SPIR-V tools")
    def test_array_sample_validates(self):
        assembly = translate("ps_array1d", f"image_sample_l v[0:3], v[4:6], {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY")
        spirv_validate(assembly)
