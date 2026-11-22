"""Structural translator tests for derivative and depth-compare image_sample* instructions:

  - image_sample_cd, image_sample_cd_o, image_sample_cd_cl, image_sample_cd_cl_o
  - image_sample_c_d, image_sample_c_d_o, image_sample_c_d_cl, image_sample_c_d_cl_o
  - image_sample_c_cd, image_sample_c_cd_o, image_sample_c_cd_cl, image_sample_c_cd_cl_o

Verifies OpImageSampleExplicitLod / OpImageSampleDrefExplicitLod emission,
Grad / Grad|Offset / Grad|MinLod / Grad|Offset|MinLod operand order, 1D/2D/2D_ARRAY,
a16 (keeping offset, z-compare, and gradients 32-bit while unpacking coords), d16,
Unsupported on 3D gradients / insufficient coords, and clean spirv_asm assembly.

Standard library only; no game data.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, translate_lines  # noqa: E402  (support puts tools/ on sys.path)
import spirv_asm  # noqa: E402

DESC = "s[8:15], s[16:19]"


def translate_and_assemble(lines, name="ps_sample_cd"):
    prog = [f"v_mov_b32 v{r}, {0x3f800000 + r * 0x00100000:#x}" for r in range(10, 21)] + list(lines) + [
        "exp mrt0 v4, v5, v6, v7 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, prog)
    binary = spirv_asm.assemble(spv)
    if len(binary) < 20 or binary[:4] != b"\x03\x02\x23\x07":
        raise AssertionError(f"spirv_asm produced invalid SPIR-V header for {name}")
    return spv


class ImageSampleCdAndCompareGradTests(unittest.TestCase):
    def test_all_twelve_ops_2d_and_operand_order(self):
        cases = [
            # (op, addr, is_dref, expected_mask, n_operand_ids)
            ("image_sample_cd", "v[10:15]", False, "Grad", 2),
            ("image_sample_cd_o", "v[10:16]", False, "Grad|Offset", 3),
            ("image_sample_cd_cl", "v[10:16]", False, "Grad|MinLod", 3),
            ("image_sample_cd_cl_o", "v[10:17]", False, "Grad|Offset|MinLod", 4),
            ("image_sample_c_d", "v[10:16]", True, "Grad", 2),
            ("image_sample_c_d_o", "v[10:17]", True, "Grad|Offset", 3),
            ("image_sample_c_d_cl", "v[10:17]", True, "Grad|MinLod", 3),
            ("image_sample_c_d_cl_o", "v[10:18]", True, "Grad|Offset|MinLod", 4),
            ("image_sample_c_cd", "v[10:16]", True, "Grad", 2),
            ("image_sample_c_cd_o", "v[10:17]", True, "Grad|Offset", 3),
            ("image_sample_c_cd_cl", "v[10:17]", True, "Grad|MinLod", 3),
            ("image_sample_c_cd_cl_o", "v[10:18]", True, "Grad|Offset|MinLod", 4),
        ]
        for op, addr, is_dref, mask, n_args in cases:
            with self.subTest(op=op):
                dmask = "0x1" if is_dref else "0xf"
                vdata = "v4" if is_dref else "v[4:7]"
                spv = translate_and_assemble([f"{op} {vdata}, {addr}, {DESC} dmask:{dmask} dim:SQ_RSRC_IMG_2D"])
                if is_dref:
                    pat = rf"%\w+ = OpImageSampleDrefExplicitLod %float %\w+ (%\w+) (%\w+) {re.escape(mask)} ((?:%\w+ ?){{{n_args}}})$"
                else:
                    pat = rf"%\w+ = OpImageSampleExplicitLod %v4float %\w+ (%\w+) {re.escape(mask)} ((?:%\w+ ?){{{n_args}}})$"
                m = re.search(pat, spv, re.M)
                self.assertIsNotNone(m, f"missing expected {mask} sample for {op}")

    def test_1d_and_2d_array_dimensions(self):
        # 1D: scalar gradients (ds/dx, ds/dy) -> 1D coord u
        spv_1d_cd = translate_and_assemble([f"image_sample_cd v[4:7], v[10:12], {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D"])
        self.assertRegex(spv_1d_cd, r"(?m)OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad %\w+ %\w+$")

        spv_1d_cd_o = translate_and_assemble([f"image_sample_c_d_cl_o v4, v[10:15], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_1D"])
        self.assertRegex(
            spv_1d_cd_o,
            r"(?m)OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Grad\|Offset\|MinLod %\w+ %\w+ %\w+ %\w+$",
        )

        # 2D_ARRAY: %v2float gradients + %v3float coords (u, v, layer)
        spv_arr = translate_and_assemble([
            f"image_sample_c_cd_cl_o v4, v[10:19], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
        ])
        self.assertRegex(
            spv_arr,
            r"(?m)OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Grad\|Offset\|MinLod %\w+ %\w+ %\w+ %\w+$",
        )
        self.assertRegex(spv_arr, r"OpCompositeConstruct %v3float")

    def test_a16_keeps_extra_dwords_32bit_and_unpacks_coords(self):
        # image_sample_c_cd_cl_o ... a16:
        #   v10=offset, v11=z-compare, v12..v15=gradients (all 6 stay 32-bit), v16=(u,v) f16 pair, v17=minlod f16 pair
        spv = translate_and_assemble([
            f"image_sample_c_cd_cl_o v4, v[10:17], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
        ])
        unpacks = re.findall(r"%\w+ = OpExtInst %v2float %glsl UnpackHalf2x16 %\w+", spv)
        self.assertEqual(len(unpacks), 2, spv)
        self.assertRegex(
            spv,
            r"(?m)OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Grad\|Offset\|MinLod %\w+ %\w+ %\w+ %\w+$",
        )

        # image_sample_c_d ... a16: v10=ref, v11..v14=grads, v15=(u,v) f16 pair -> 1 UnpackHalf2x16
        spv_cd = translate_and_assemble([
            f"image_sample_c_d v4, v[10:15], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
        ])
        unpacks_cd = re.findall(r"%\w+ = OpExtInst %v2float %glsl UnpackHalf2x16 %\w+", spv_cd)
        self.assertEqual(len(unpacks_cd), 1, spv_cd)

        # image_sample_cd ... a16 (coarse derivatives, NO z-compare):
        #   v10..v13=grads (4 words), v14=(u,v) f16 pair -> 5 address VGPRs, 1 UnpackHalf2x16
        spv_coarse = translate_and_assemble([
            f"image_sample_cd v[4:7], v[10:14], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D a16",
        ])
        self.assertEqual(len(re.findall(r"%\w+ = OpExtInst %v2float %glsl UnpackHalf2x16 %\w+", spv_coarse)), 1, spv_coarse)
        self.assertRegex(spv_coarse, r"(?m)OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad %\w+ %\w+$")

        # image_sample_cd_cl ... a16:
        #   v10..v13=grads (4 words), v14=(u,v) f16 pair, v15=minlod f16 pair -> 6 address VGPRs, 2 UnpackHalf2x16
        spv_coarse_cl = translate_and_assemble([
            f"image_sample_cd_cl v[4:7], v[10:15], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D a16",
        ])
        self.assertEqual(len(re.findall(r"%\w+ = OpExtInst %v2float %glsl UnpackHalf2x16 %\w+", spv_coarse_cl)), 2, spv_coarse_cl)
        self.assertRegex(spv_coarse_cl, r"(?m)OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad\|MinLod %\w+ %\w+ %\w+$")

    def test_d16_packing(self):
        # Depth-compare d16 packs 1 f16 result into low half (uses f16_bits bit math, not raw OpBitcast %uint)
        spv_c = translate_and_assemble([f"image_sample_c_d v4, v[10:16], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D d16"])
        self.assertRegex(spv_c, r"OpImageSampleDrefExplicitLod %float")

        spv_cd = translate_and_assemble([f"image_sample_cd v[4:5], v[10:15], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D d16"])
        self.assertRegex(spv_cd, r"OpImageSampleExplicitLod %v4float")

    def test_3d_gradients_and_short_coords(self):
        good_3d = [
            (f"image_sample_cd v[4:7], v[10:19], {DESC} dmask:0xf dim:SQ_RSRC_IMG_3D", "OpImageSampleExplicitLod %v4float"),
            (f"image_sample_c_d v4, v[10:19], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D", "OpImageSampleDrefExplicitLod %float"),
            (f"image_sample_c_cd v4, v[10:19], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D", "OpImageSampleDrefExplicitLod %float"),
        ]
        for ins, op_pat in good_3d:
            with self.subTest(ins=ins):
                spv = translate_and_assemble([ins])
                self.assertIn(op_pat, spv)
                self.assertIn("OpCompositeConstruct %v3float", spv)

        bad_cases = [
            f"image_sample_cd v[4:7], v[10:14], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D",     # missing v coord
            f"image_sample_c_d v4, v[10:15], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D",       # missing v coord
            f"image_sample_c_cd_cl v4, v[10:16], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D",   # missing minlod
        ]
        for ins in bad_cases:
            with self.subTest(ins=ins):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_and_assemble([ins])


if __name__ == "__main__":
    unittest.main()
