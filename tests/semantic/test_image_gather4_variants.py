"""Semantic, structural, and spirv-val tests for the 14 RDNA2 image_gather4_* variants.

Covers:
  - Explicit-LOD depth-compare (compute and pixel shaders):
      image_gather4_c_l, image_gather4_c_l_o
  - Clamped / biased non-compare and depth-compare (pixel shaders via OpImageQueryLod):
      image_gather4_cl, image_gather4_cl_o,
      image_gather4_c_cl, image_gather4_c_cl_o,
      image_gather4_b, image_gather4_b_o,
      image_gather4_b_cl, image_gather4_b_cl_o,
      image_gather4_c_b, image_gather4_c_b_o,
      image_gather4_c_b_cl, image_gather4_c_b_cl_o
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")
if not SPIRV_AS:
    _sdk = r"C:\VulkanSDK\1.4.363.0\Bin\spirv-as.exe"
    if os.path.isfile(_sdk):
        SPIRV_AS = _sdk
if not SPIRV_VAL:
    _sdk = r"C:\VulkanSDK\1.4.363.0\Bin\spirv-val.exe"
    if os.path.isfile(_sdk):
        SPIRV_VAL = _sdk


ALL_14_OPS = (
    # (op, vaddr_2d, vaddr_2d_array, vaddr_2d_a16, vaddr_2d_array_a16)
    ("image_gather4_c_l", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:2]"),
    ("image_gather4_c_l_o", "v[0:4]", "v[0:5]", "v[0:3]", "v[0:3]"),
    ("image_gather4_cl", "v[0:2]", "v[0:3]", "v[0:1]", "v[0:1]"),
    ("image_gather4_cl_o", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:2]"),
    ("image_gather4_c_cl", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:2]"),
    ("image_gather4_c_cl_o", "v[0:4]", "v[0:5]", "v[0:3]", "v[0:3]"),
    ("image_gather4_b", "v[0:2]", "v[0:3]", "v[0:1]", "v[0:2]"),
    ("image_gather4_b_o", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:3]"),
    ("image_gather4_b_cl", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:2]"),
    ("image_gather4_b_cl_o", "v[0:4]", "v[0:5]", "v[0:3]", "v[0:3]"),
    ("image_gather4_c_b", "v[0:3]", "v[0:4]", "v[0:2]", "v[0:3]"),
    ("image_gather4_c_b_o", "v[0:4]", "v[0:5]", "v[0:3]", "v[0:4]"),
    ("image_gather4_c_b_cl", "v[0:4]", "v[0:5]", "v[0:3]", "v[0:3]"),
    ("image_gather4_c_b_cl_o", "v[0:5]", "v[0:6]", "v[0:4]", "v[0:4]"),
)

PR_3201_OPS = (
    "image_gather4",
    "image_gather4_o",
    "image_gather4_c",
    "image_gather4_c_o",
    "image_gather4_c_lz",
    "image_gather4_c_lz_o",
)


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _ps_shader(inst: str, s0_lit: int | None = 0x3092, runtime_s: bool = False) -> list[str]:
    lines = [
        "s_load_dwordx8 s[8:15], s[0:1], 0x0",
        "s_waitcnt lgkmcnt(0)",
    ]
    if runtime_s:
        lines += [
            "s_load_dwordx4 s[16:19], s[0:1], 0x20",
            "s_waitcnt lgkmcnt(0)",
        ]
    else:
        lines += [
            f"s_mov_b32 s16, {hex(s0_lit)}",
            "s_mov_b32 s17, 0xc00000",
            "s_mov_b32 s18, 0",
            "s_mov_b32 s19, 0",
        ]
    lines += [
        "v_mov_b32_e32 v0, 0x0",
        "v_mov_b32_e32 v1, 0.5",
        "v_mov_b32_e32 v2, 0.5",
        "v_mov_b32_e32 v3, 0.5",
        "v_mov_b32_e32 v4, 0.5",
        "v_mov_b32_e32 v5, 0.0",
        "v_mov_b32_e32 v6, 0.0",
        inst,
        "s_waitcnt vmcnt(0)",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]
    return lines


def _cs_fssm() -> bytes:
    blob = bytearray(0x148)
    struct.pack_into("<4I", blob, 0x138, 0x8, 32, 1, 1)
    return bytes(blob)


def _cs_shader(inst: str, s0_lit: int = 0x3092) -> list[str]:
    return [
        "s_load_dwordx8 s[8:15], s[0:1], 0x0",
        "s_waitcnt lgkmcnt(0)",
        f"s_mov_b32 s16, {hex(s0_lit)}",
        "s_mov_b32 s17, 0xc00000",
        "s_mov_b32 s18, 0",
        "s_mov_b32 s19, 0",
        "v_mov_b32_e32 v0, 0x0",
        "v_mov_b32_e32 v1, 0.5",
        "v_mov_b32_e32 v2, 0.5",
        "v_mov_b32_e32 v3, 0.5",
        "v_mov_b32_e32 v4, 0.5",
        "v_mov_b32_e32 v5, 0.0",
        inst,
        "s_waitcnt vmcnt(0)",
        "s_endpgm",
    ]


class TestImageGather4Variants(unittest.TestCase):
    def test_no_overlap_with_pr_3201(self) -> None:
        for op in PR_3201_OPS:
            self.assertNotIn(op, rdna2spirv.Gen.GATHER4_LOD_OPS)
        self.assertEqual(len(rdna2spirv.Gen.GATHER4_LOD_OPS), 14)

    def test_all_14_ops_2d_validate(self) -> None:
        for op, vaddr_2d, _, _, _ in ALL_14_OPS:
            with self.subTest(op=op, dim="2D"):
                inst = f"{op} v[8:11], {vaddr_2d}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                spv = support.translate_lines(f"ps_{op}_2d", _ps_shader(inst))
                _validate(spv)
                self.assertIn("OpImageFetch", spv)
                self.assertIn("OpCapability ImageQuery", spv)
                if "_l" not in op:
                    self.assertIn("OpImageQueryLod %v2float", spv)
                else:
                    self.assertNotIn("OpImageQueryLod", spv)

    def test_all_14_ops_2d_array_validate(self) -> None:
        for op, _, vaddr_arr, _, _ in ALL_14_OPS:
            with self.subTest(op=op, dim="2D_ARRAY"):
                inst = f"{op} v[8:11], {vaddr_arr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY"
                spv = support.translate_lines(f"ps_{op}_arr", _ps_shader(inst))
                _validate(spv)
                self.assertIn("OpImageQuerySizeLod %v3int", spv)
                self.assertIn("OpCompositeConstruct %v3int", spv)

    def test_all_14_ops_d16_validate(self) -> None:
        for op, vaddr_2d, _, _, _ in ALL_14_OPS:
            with self.subTest(op=op, d16=True):
                inst = f"{op} v[8:9], {vaddr_2d}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16"
                spv = support.translate_lines(f"ps_{op}_d16", _ps_shader(inst))
                _validate(spv)

    def test_all_14_ops_a16_validate(self) -> None:
        for op, _, _, vaddr_2d_a16, vaddr_arr_a16 in ALL_14_OPS:
            with self.subTest(op=op, dim="2D", a16=True):
                inst = f"{op} v[8:11], {vaddr_2d_a16}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16"
                spv = support.translate_lines(f"ps_{op}_a16", _ps_shader(inst))
                _validate(spv)
                self.assertIn("UnpackHalf2x16", spv)
            with self.subTest(op=op, dim="2D_ARRAY", a16=True):
                inst = f"{op} v[8:11], {vaddr_arr_a16}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY a16"
                spv = support.translate_lines(f"ps_{op}_arr_a16", _ps_shader(inst))
                _validate(spv)

    def test_compute_stage_explicit_and_clamped(self) -> None:
        cs_ops = (
            ("image_gather4_c_l", "v[0:3]"),
            ("image_gather4_c_l_o", "v[0:4]"),
            ("image_gather4_cl", "v[0:2]"),
            ("image_gather4_cl_o", "v[0:3]"),
            ("image_gather4_c_cl", "v[0:3]"),
            ("image_gather4_c_cl_o", "v[0:4]"),
        )
        for op, vaddr in cs_ops:
            with self.subTest(op=op):
                inst = f"{op} v[8:11], {vaddr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                spv = support.translate_lines(f"cs_{op}", _cs_shader(inst), fssm=_cs_fssm())
                _validate(spv)
                self.assertNotIn("OpImageQueryLod", spv)
                self.assertIn("OpImageFetch", spv)

    def test_compute_stage_rejects_biased(self) -> None:
        for op, vaddr in (("image_gather4_b", "v[0:2]"), ("image_gather4_c_b_cl_o", "v[0:5]")):
            with self.subTest(op=op):
                inst = f"{op} v[8:11], {vaddr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines(f"cs_{op}", _cs_shader(inst), fssm=_cs_fssm())

    def test_all_8_depth_compare_funcs_literal_and_runtime(self) -> None:
        expected_ops = {
            0: None,
            1: "OpFOrdLessThan %bool",
            2: "OpFOrdEqual %bool",
            3: "OpFOrdLessThanEqual %bool",
            4: "OpFOrdGreaterThan %bool",
            5: "OpFUnordNotEqual %bool",
            6: "OpFOrdGreaterThanEqual %bool",
            7: None,
        }
        inst = "image_gather4_c_l v[8:11], v[0:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
        for func, spv_op in expected_ops.items():
            with self.subTest(depth_compare_func=func):
                s0 = 0x92 | (func << 12)
                spv = support.translate_lines(f"ps_cmp_{func}", _ps_shader(inst, s0_lit=s0))
                _validate(spv)
                if spv_op is None:
                    self.assertNotIn("OpFOrdLessThan", spv)
                    self.assertNotIn("OpFOrdEqual", spv)
                    self.assertNotIn("OpFOrdGreaterThan", spv)
                    self.assertNotIn("OpFUnordNotEqual", spv)
                else:
                    self.assertEqual(spv.count(spv_op), 4)
        # Runtime S# decodes DEPTH_COMPARE_FUNC [14:12] dynamically
        spv_rt = support.translate_lines("ps_cmp_rt", _ps_shader(inst, runtime_s=True))
        _validate(spv_rt)
        self.assertIn("OpBitFieldUExtract %uint", spv_rt)
        self.assertEqual(spv_rt.count("OpFOrdLessThan %bool"), 4)
        self.assertEqual(spv_rt.count("OpFOrdEqual %bool"), 4)
        self.assertEqual(spv_rt.count("OpFOrdGreaterThan %bool"), 4)

    def test_unorm_modifier(self) -> None:
        for op, vaddr in (("image_gather4_c_l", "v[0:3]"), ("image_gather4_cl", "v[0:2]")):
            with self.subTest(op=op):
                inst = f"{op} v[8:11], {vaddr}, s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"
                spv = support.translate_lines(f"ps_{op}_unorm", _ps_shader(inst))
                _validate(spv)
                self.assertNotIn("OpFMul %v2float", spv)

    def test_dmask_channels_and_offset_extraction(self) -> None:
        for ch, dmask in enumerate(("0x1", "0x2", "0x4", "0x8")):
            with self.subTest(dmask=dmask):
                inst = f"image_gather4_cl_o v[8:11], v[0:3], s[8:15], s[16:19] dmask:{dmask} dim:SQ_RSRC_IMG_2D"
                spv = support.translate_lines(f"ps_cl_o_ch{ch}", _ps_shader(inst))
                _validate(spv)
                self.assertIn("OpBitFieldSExtract %uint", spv)
                self.assertIn(f"OpCompositeExtract %float", spv)
                self.assertIn("OpExtInst %float %glsl FMax", spv)

    def test_border_modes_on_depth_compare_gather4(self) -> None:
        # Mode 6 (clamp border) with opaque white (word3 [31:30] = 2)
        inst = "image_gather4_c_l_o v[8:11], v[0:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D"
        lines = _ps_shader(inst, s0_lit=0x3036)
        lines = [ln.replace("s_mov_b32 s19, 0", "s_mov_b32 s19, 0x80000100") for ln in lines]
        spv = support.translate_lines("ps_c_l_o_border", lines)
        _validate(spv)
        self.assertIn("OpSelect %float", spv)


if __name__ == "__main__":
    unittest.main()
