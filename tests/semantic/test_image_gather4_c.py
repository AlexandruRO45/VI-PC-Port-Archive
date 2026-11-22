"""Structural translator tests for image_gather4, image_gather4_o, and image_gather4_c* in tools/rdna2spirv.py.

Covers:
  - image_gather4, image_gather4_o -> OpImageGather (2D, 2D_ARRAY, CUBE, d16)
  - image_gather4_c, image_gather4_c_o, image_gather4_c_lz, image_gather4_c_lz_o
    -> OpImageDrefGather (operand order: uv from coords, ref from z-compare,
       optional signed 6-bit Offset, 2D/2D_ARRAY/CUBE, d16, a16 keeping z-compare 32-bit)
  - Unsupported rejections: 3D, multi-bit/zero dmask, insufficient coords, _o + a16.

Standard library only; no game data.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import rdna2spirv, translate_lines  # noqa: E402

DESC = "s[8:15], s[16:19]"
# Minimal compute header (0x138 + 16 bytes): rsrc2=0, group size=(1, 1, 1) at offset 0x138
STUB_FSSM = bytes(0x138) + struct.pack("<4I", 0, 1, 1, 1)


def translate_ps(lines, name="ps_gather4_c"):
    prog = [
        "v_mov_b32 v10, 0x3f800000",
        "v_mov_b32 v11, 0x40000000",
        "v_mov_b32 v12, 0x40400000",
        "v_mov_b32 v13, 0x40800000",
        "v_mov_b32 v14, 0x40a00000",
    ] + list(lines) + [
        "exp mrt0 v4, v5, v6, v7 done vm",
        "s_endpgm",
    ]
    return translate_lines(name, prog)


def translate_cs(lines, name="cs_gather4_c"):
    prog = [
        "v_mov_b32 v10, 0x3f800000",
        "v_mov_b32 v11, 0x40000000",
        "v_mov_b32 v12, 0x40400000",
        "v_mov_b32 v13, 0x40800000",
        "s_mov_b32 s0, 0",
        "s_mov_b32 s1, 0",
    ] + list(lines) + [
        "s_endpgm",
    ]
    return translate_lines(name, prog, fssm=STUB_FSSM)


class Gather4PlainTests(unittest.TestCase):
    def test_image_gather4_and_o_emit_op_image_gather(self):
        spv = translate_ps([f"image_gather4 v[4:7], v[10:11], {DESC} dmask:0x2 dim:SQ_RSRC_IMG_2D"])
        m = re.search(r"%\w+ = OpImageGather %v4float %\w+ (%\w+) (%\w+)$", spv, re.M)
        self.assertIsNotNone(m, spv)
        self.assertNotIn("Offset", m.group(0))

        spv_o = translate_ps([f"image_gather4_o v[4:7], v[10:12], {DESC} dmask:0x4 dim:SQ_RSRC_IMG_2D"])
        m_o = re.search(r"%\w+ = OpImageGather %v4float %\w+ (%\w+) (%\w+) Offset (%\w+)$", spv_o, re.M)
        self.assertIsNotNone(m_o, spv_o)
        self.assertIn("OpBitFieldSExtract", spv_o)

    def test_image_gather4_array_cube_and_d16(self):
        spv_arr = translate_ps([f"image_gather4 v[4:7], v[10:12], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY"])
        self.assertRegex(spv_arr, r"OpImageGather %v4float %\w+ %\w+ %\w+")
        self.assertRegex(spv_arr, r"OpCompositeConstruct %v3float")

        spv_cube = translate_ps([f"image_gather4 v[4:7], v[10:12], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_CUBE"])
        self.assertRegex(spv_cube, r"OpImageGather %v4float %\w+ %\w+ %\w+")
        self.assertRegex(spv_cube, r"OpCompositeConstruct %v3float")

        # d16 packs 4 taps into 2 VGPRs (v4, v5); v6/v7 are not written by the gather
        spv_d16 = translate_ps([f"image_gather4 v[4:5], v[10:11], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D d16"])
        self.assertRegex(spv_d16, r"OpImageGather %v4float")
        self.assertNotIn("OpStore %v6", spv_d16)


class Gather4CompareTests(unittest.TestCase):
    def test_dref_gather_forms_and_offset(self):
        for op, addr, has_offset in (
            ("image_gather4_c", "v[10:12]", False),
            ("image_gather4_c_lz", "v[10:12]", False),
            ("image_gather4_c_o", "v[10:13]", True),
            ("image_gather4_c_lz_o", "v[10:13]", True),
        ):
            with self.subTest(op=op):
                spv = translate_ps([f"{op} v[4:7], {addr}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D"])
                if has_offset:
                    m = re.search(r"%\w+ = OpImageDrefGather %v4float (%\w+) (%\w+) (%\w+) Offset (%\w+)$", spv, re.M)
                else:
                    m = re.search(r"%\w+ = OpImageDrefGather %v4float (%\w+) (%\w+) (%\w+)$", spv, re.M)
                    self.assertNotIn("Offset", spv.split("OpImageDrefGather")[1].splitlines()[0])
                self.assertIsNotNone(m, f"missing OpImageDrefGather for {op}")

                si_ssa, uv_ssa, ref_ssa = m.group(1), m.group(2), m.group(3)
                self.assertRegex(spv, rf"{re.escape(si_ssa)} = OpSampledImage %\w+ %\w+ %\w+")
                # Verify ref comes from the expected VGPR (v10 without _o, v11 with _o)
                ref_u = re.search(rf"{re.escape(ref_ssa)} = OpBitcast %float (%\w+)", spv)
                self.assertIsNotNone(ref_u)
                ref_load = re.search(rf"{re.escape(ref_u.group(1))} = OpLoad %uint (%\w+)", spv)
                self.assertIsNotNone(ref_load)
                expected_reg = "%cb" if has_offset else "%ca"
                self.assertRegex(spv, rf"{re.escape(ref_load.group(1))} = OpAccessChain %_ptr_Function_uint %vgpr {expected_reg}")
                # Verify uv is constructed from the two coordinates after ref
                uv_m = re.search(rf"{re.escape(uv_ssa)} = OpCompositeConstruct %v2float (%\w+) (%\w+)", spv)
                self.assertIsNotNone(uv_m)

    def test_dref_gather_2d_array_cube_d16_and_compute(self):
        spv_arr = translate_ps([f"image_gather4_c v[4:7], v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY"])
        self.assertRegex(spv_arr, r"OpImageDrefGather %v4float %\w+ %\w+ %\w+")
        self.assertRegex(spv_arr, r"OpCompositeConstruct %v3float")

        spv_cube = translate_ps([f"image_gather4_c_lz v[4:7], v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_CUBE"])
        self.assertRegex(spv_cube, r"OpImageDrefGather %v4float %\w+ %\w+ %\w+")
        self.assertRegex(spv_cube, r"OpCompositeConstruct %v3float")

        spv_d16 = translate_ps([f"image_gather4_c v[4:5], v[10:12], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D d16"])
        self.assertRegex(spv_d16, r"OpImageDrefGather %v4float")

        spv_cs = translate_cs([f"image_gather4_c_lz_o v[4:7], v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D"])
        self.assertRegex(spv_cs, r"OpImageDrefGather %v4float %\w+ %\w+ %\w+ Offset %\w+")

    def test_a16_keeps_z_compare_32bit_and_unpacks_coords(self):
        # For image_gather4_c ... a16: v10 is 32-bit z-compare, v11 is packed (u, v) f16 pair.
        spv = translate_ps([f"image_gather4_c v[4:7], v[10:11], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16"])
        m = re.search(r"%\w+ = OpImageDrefGather %v4float %\w+ (%\w+) (%\w+)$", spv, re.M)
        self.assertIsNotNone(m, spv)
        uv_ssa, ref_ssa = m.group(1), m.group(2)

        # Exactly one UnpackHalf2x16 (for v11), NOT two (v10 stays 32-bit)
        unpacks = re.findall(r"(%\w+) = OpExtInst %v2float %glsl UnpackHalf2x16 (%\w+)", spv)
        self.assertEqual(len(unpacks), 1, spv)
        v2_ssa = unpacks[0][0]
        # ref must not come from UnpackHalf2x16
        self.assertNotIn(v2_ssa, ref_ssa)
        uv_m = re.search(rf"{re.escape(uv_ssa)} = OpCompositeConstruct %v2float (%\w+) (%\w+)", spv)
        self.assertIsNotNone(uv_m)
        for comp_idx, f_ssa in enumerate((uv_m.group(1), uv_m.group(2))):
            u_m = re.search(rf"{re.escape(f_ssa)} = OpBitcast %float (%\w+)", spv)
            if u_m is not None:
                ext_m = re.search(rf"{re.escape(u_m.group(1))} = OpBitcast %uint (%\w+)", spv)
                self.assertIsNotNone(ext_m)
                f_ssa = ext_m.group(1)
            self.assertIn(f"{f_ssa} = OpCompositeExtract %float {v2_ssa} {comp_idx}", spv)


class Gather4UnsupportedTests(unittest.TestCase):
    def test_rejects_invalid_forms(self):
        bad_cases = [
            f"image_gather4 v[4:7], v[10:12], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D",
            f"image_gather4_c v[4:7], v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D",
            f"image_gather4 v[4:7], v[10:11], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D",
            f"image_gather4_c v[4:7], v[10:12], {DESC} dmask:0x3 dim:SQ_RSRC_IMG_2D",
            f"image_gather4_c v[4:7], v[10:12], {DESC} dmask:0x0 dim:SQ_RSRC_IMG_2D",
            f"image_gather4_c v[4:7], v[10:11], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D",      # missing coord
            f"image_gather4_o v[4:7], v10, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
            f"image_gather4_c_o v[4:7], v[10:11], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
            f"image_gather4_c_lz_o v[4:7], v[10:11], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
        ]
        for ins in bad_cases:
            with self.subTest(ins=ins):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_ps([ins])


if __name__ == "__main__":
    unittest.main()
