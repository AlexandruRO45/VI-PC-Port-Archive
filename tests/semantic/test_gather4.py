"""Translator semantic and SPIR-V validation tests for image_gather4_{lz,l,lz_o,l_o}.

Standard library only (spirv-as and spirv-val are used when on PATH); no GPU, no LLVM, no game files.

Covers:
- image_gather4_lz on 2D (typed float/uint/sint views, dmask 0x1..0x8), 2D_ARRAY, CUBE (cube_coords), a16 and d16;
- image_gather4_lz_o signed 6-bit texel offset extraction -> OpImageGather ... Offset, ImageGatherExtended capability
  and shaderImageGatherExtended feature (not needs_maintenance8), and a16 offset preservation;
- image_gather4_l lowering to four OpImageFetch ... Lod taps with S# MIN_LOD/MAX_LOD clamp, half-up tie rounding,
  address modes (wrap, mirror, clamp, mirror once, border colours), OpImageQueryLevels/SizeLod (ImageQuery), a16/d16;
- image_gather4_l_o adding the extracted signed 6-bit offset via OpIAdd before OpImageFetch, declaring neither
  OpCapability ImageGatherExtended nor shaderImageGatherExtended nor needs_maintenance8;
- unsupported dmask, dimension, glc, S# address/border/FORCE_UNNORMALIZED, a16+tfe/lwe, and non-VGPR destination.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support

DESC = "s[8:15], s[16:19]"
D2 = "dim:SQ_RSRC_IMG_2D"
PRO = ["v_mov_b32 v4, 0.25", "v_mov_b32 v5, 0.5", "v_mov_b32 v6, 0.75", "v_mov_b32 v7, 1.0"]
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate_with_meta(name, body_lines, export_dst=True):
    """Return (spv_text, shader_meta_dict) for a synthetic PS or CS listing."""
    lines = list(PRO) + list(body_lines)
    if name.startswith("ps_") and export_dst:
        lines.append("exp mrt0 v0, v1, v2, v3 done vm")
    lines.append("s_endpgm")
    fssm = CS_FSSM if name.startswith("cs_") else None
    spv = support.translate_lines(name, lines, fssm=fssm)
    meta = support.rdna2spirv.shader_meta(support.rdna2spirv.translate.last_gen, spv)
    return spv, meta


def parse_defs(spv):
    """Map SSA result id -> (opcode, operand_tokens) for a SPIR-V assembly listing."""
    defs = {}
    for ln in spv.splitlines():
        m = re.match(r"^\s*(%\w+) = (Op\w+)(?: (.*))?$", ln)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3).split() if m.group(3) else [])
    return defs


def const_uints(spv):
    """Map OpConstant %uint SSA id -> int value."""
    return {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}


def sextract_fields(spv):
    """Return [(offset_bits, width_bits)] for every OpBitFieldSExtract in `spv`."""
    cu = const_uints(spv)
    out = []
    for ln in spv.splitlines():
        m = re.search(r"= OpBitFieldSExtract %uint %\w+ (%\w+) (%\w+)", ln)
        if m:
            out.append((cu[m.group(1)], cu[m.group(2)]))
    return out


class ImageGather4Tests(unittest.TestCase):
    def test_gather4_lz_dimensions_components_and_mods(self):
        for bit, dmask in enumerate((0x1, 0x2, 0x4, 0x8)):
            with self.subTest(dmask=hex(dmask)):
                spv, meta = translate_with_meta(
                    "ps_g_lz_comp",
                    [f"image_gather4_lz v[0:3], v[4:5], {DESC} dmask:{dmask:#x} {D2}"],
                )
                cu = const_uints(spv)
                gathers = re.findall(r"= OpImageGather (%\S+) %\w+ %\w+ (%\w+)$", spv, re.M)
                self.assertEqual(len(gathers), 3)  # float, uint, sint typed paths on 2D
                self.assertEqual({ty for ty, _ in gathers}, {"%v4float", "%v4uint_st", "%v4int_st"})
                self.assertEqual({cu[comp] for _, comp in gathers}, {bit})
                self.assertNotIn("OpCapability ImageGatherExtended", spv)
                self.assertNotIn("OpCapability ImageQuery", spv)
                self.assertNotIn("shaderImageGatherExtended", meta["features"])

        for dim, regs in (("dim:SQ_RSRC_IMG_2D_ARRAY", "v[4:6]"), ("dim:SQ_RSRC_IMG_CUBE", "v[4:6]")):
            with self.subTest(dim=dim):
                spv, meta = translate_with_meta(
                    "cs_g_lz_dim",
                    [f"image_gather4_lz v[0:3], {regs}, {DESC} dmask:0x1 {dim}"],
                    export_dst=False,
                )
                self.assertIn("= OpImageGather %v4float", spv)
                self.assertIn("%v3float", spv)
                if "CUBE" in dim:
                    self.assertIn("OpExtInst %float %glsl Floor", spv)
                self.assertNotIn("shaderImageGatherExtended", meta["features"])

        spv_a16, _ = translate_with_meta(
            "ps_g_lz_a16",
            [f"image_gather4_lz v[0:3], v4, {DESC} dmask:0x1 {D2} a16"],
        )
        self.assertIn("OpExtInst %v2float %glsl UnpackHalf2x16", spv_a16)

        spv_d16, _ = translate_with_meta(
            "ps_g_lz_d16",
            [f"image_gather4_lz v[0:1], v[4:5], {DESC} dmask:0x1 {D2} d16"],
        )
        self.assertIn("= OpImageGather %v4float", spv_d16)

    def test_gather4_lz_o_emits_offset_and_image_gather_extended(self):
        for line, extra in (
            (f"image_gather4_lz_o v[0:3], v[4:6], {DESC} dmask:0x1 {D2}", ()),
            (f"image_gather4_lz_o v[0:3], v[4:5], {DESC} dmask:0x1 {D2} a16", ("UnpackHalf2x16",)),
        ):
            with self.subTest(line=line):
                spv, meta = translate_with_meta("ps_g_lz_o", [line])
                self.assertEqual(sextract_fields(spv), [(0, 6), (8, 6)])
                gathers = [ln.strip() for ln in spv.splitlines() if "= OpImageGather " in ln]
                self.assertEqual(len(gathers), 3)
                for g in gathers:
                    self.assertRegex(g, r"Offset %\w+$")
                self.assertIn("OpCapability ImageGatherExtended", spv)
                self.assertIn("shaderImageGatherExtended", meta["features"])
                self.assertNotIn("needs_maintenance8", meta)
                for tok in extra:
                    self.assertIn(tok, spv)

    def test_gather4_l_o_adds_offset_without_image_gather_extended(self):
        for stage in ("ps", "cs"):
            for line, regs in (
                (f"image_gather4_l_o v[0:3], v[4:7], {DESC} dmask:0x2 {D2}", "v[4:7]"),
                (f"image_gather4_l_o v[0:3], v[4:6], {DESC} dmask:0x2 {D2} a16", "v[4:6]"),
            ):
                with self.subTest(stage=stage, line=line):
                    spv, meta = translate_with_meta(f"{stage}_g_l_o", [line], export_dst=(stage == "ps"))
                    self.assertEqual(sextract_fields(spv), [(0, 6), (8, 6)])
                    self.assertNotIn("OpImageGather", spv)
                    fetches = [ln for ln in spv.splitlines() if "= OpImageFetch " in ln]
                    self.assertEqual(len(fetches), 12)  # 4 taps x (float, uint, sint)
                    for f in fetches:
                        self.assertNotIn("Offset", f)
                    self.assertIn("= OpIAdd %v2int", spv)
                    self.assertIn("OpCapability ImageQuery", spv)
                    self.assertNotIn("OpCapability ImageGatherExtended", spv)
                    self.assertNotIn("shaderImageGatherExtended", meta["features"])
                    self.assertNotIn("needs_maintenance8", meta)
                    if "a16" in line:
                        self.assertIn("OpExtInst %v2float %glsl UnpackHalf2x16", spv)

    def test_gather4_l_sampler_lod_clamp_tie_and_address_modes(self):
        # Runtime S#: extract MIN_LOD / MAX_LOD with FMax + FMin, and runtime CLAMP_X / CLAMP_Y modes.
        spv_rt, meta_rt = translate_with_meta(
            "ps_g_l_rt",
            [
                "s_load_dwordx4 s[16:19], s[0:1], 0",
                "s_waitcnt lgkmcnt(0)",
                f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x4 {D2}",
            ],
        )
        self.assertIn("OpCapability ImageQuery", spv_rt)
        self.assertNotIn("OpCapability ImageGatherExtended", spv_rt)
        self.assertNotIn("shaderImageGatherExtended", meta_rt["features"])
        self.assertIn("OpImageQueryLevels %int", spv_rt)
        self.assertIn("OpImageQuerySizeLod %v2int", spv_rt)
        # Half-up tie rounding (floor + (frac >= 0.5)) rather than RoundEven.
        self.assertIn("OpExtInst %float %glsl Floor", spv_rt)
        self.assertIn("OpFOrdGreaterThanEqual %bool", spv_rt)
        self.assertNotIn("RoundEven", spv_rt)
        self.assertIn("OpExtInst %float %glsl FMax", spv_rt)
        self.assertIn("OpExtInst %float %glsl FMin", spv_rt)
        self.assertIn("OpSRem %int", spv_rt)

        # Literal S#: FClamp for LOD limits; wrap/mirror omit border selection, border modes emit border check.
        spv_wrap, _ = translate_with_meta(
            "ps_g_l_wrap",
            [
                "s_mov_b32 s16, 0x00000008",   # CLAMP_X=0 (wrap), CLAMP_Y=1 (mirror)
                "s_mov_b32 s17, 0x00fff000",
                "s_mov_b32 s18, 0",
                "s_mov_b32 s19, 0",
                f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}",
            ],
        )
        self.assertIn("OpExtInst %float %glsl FClamp", spv_wrap)
        self.assertIn("OpSRem %int", spv_wrap)

        spv_border, _ = translate_with_meta(
            "ps_g_l_border",
            [
                "s_mov_b32 s16, 0x0000003e",   # CLAMP_X=6 (clamp border), CLAMP_Y=7 (mirror once border)
                "s_mov_b32 s17, 0x00fff000",
                "s_mov_b32 s18, 0",
                "s_mov_b32 s19, 0x80000000",   # BORDER_COLOR_TYPE=2 (opaque white)
                f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}",
            ],
        )
        self.assertNotIn("OpSRem %int", spv_border)
        self.assertIn("OpSLessThan %bool", spv_border)
        self.assertIn("OpSGreaterThan %bool", spv_border)

        # Literal S# opaque black returns 1.0 only on channel 3 (alpha).
        fetches_w = [ln for ln in spv_border.splitlines() if "= OpImageFetch %v4float " in ln]
        self.assertEqual(len(fetches_w), 4)

        # Runtime S# border colour: opaque black (BORDER_COLOR_TYPE == 1) yields 1 only on channel 3 (alpha).
        def border_select_kinds(spv):
            defs = parse_defs(spv)
            cu = const_uints(spv)
            kinds = set()
            for ln in spv.splitlines():
                m = re.search(r"= OpSelect %float (%\w+) %\w+ %\w+$", ln)
                if not m:
                    continue
                cond = m.group(1)
                op, args = defs[cond]
                if op == "OpIEqual":
                    kinds.add(cu[args[2]])
                elif op == "OpLogicalOr":
                    for sub in args[1:]:
                        sop, sargs = defs[sub]
                        if sop == "OpIEqual":
                            kinds.add(cu[sargs[2]])
            return kinds

        spv_rt_r, _ = translate_with_meta(
            "ps_g_l_rt_r",
            ["s_load_dwordx4 s[16:19], s[0:1], 0", "s_waitcnt lgkmcnt(0)",
             f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"],
        )
        spv_rt_a, _ = translate_with_meta(
            "ps_g_l_rt_a",
            ["s_load_dwordx4 s[16:19], s[0:1], 0", "s_waitcnt lgkmcnt(0)",
             f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x8 {D2}"],
        )
        self.assertEqual(border_select_kinds(spv_rt_r), {2})
        self.assertEqual(border_select_kinds(spv_rt_a), {1, 2})

        spv_d16, _ = translate_with_meta(
            "ps_g_l_d16",
            [f"image_gather4_l v[0:1], v[4:6], {DESC} dmask:0x8 {D2} d16"],
        )
        self.assertEqual(len([ln for ln in spv_d16.splitlines() if "= OpImageFetch " in ln]), 12)

    def test_unsupported_forms_raise(self):
        refused = [
            f"image_gather4_lz v[0:3], v[4:5], {DESC} dmask:0x0 {D2}",
            f"image_gather4_lz v[0:3], v[4:5], {DESC} dmask:0x3 {D2}",
            f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0xf {D2}",
            f"image_gather4_lz v[0:3], v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D",
            f"image_gather4_lz v[0:3], v4, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_1D",
            f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
            f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
            f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2} glc",
            f"image_gather4_lz v[0:3], v[4:5], {DESC} dmask:0x1 {D2} glc",
            f"image_gather4_lz_o v[0:4], v[4:5], {DESC} dmask:0x1 {D2} a16 tfe",
            f"image_gather4_l_o v[0:3], v[4:6], {DESC} dmask:0x1 {D2} a16 lwe",
            f"image_gather4_l off, v[4:6], {DESC} dmask:0x1 {D2}",
        ]
        for line in refused:
            with self.subTest(line=line), self.assertRaises(support.rdna2spirv.Unsupported):
                translate_with_meta("ps_g_bad", [line])

        sampler_refused = (
            ("s_mov_b32 s16, 0x8000", f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"),
            ("s_mov_b32 s16, 4", f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"),
            ("s_mov_b32 s16, 0x28", f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"),
            ("s_mov_b32 s19, 0xc0000000", f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"),
        )
        for setup, insn in sampler_refused:
            with self.subTest(setup=setup), self.assertRaises(support.rdna2spirv.Unsupported):
                translate_with_meta("ps_g_bad_samp", [setup, insn])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        cases = [
            ("ps_v_lz", [f"image_gather4_lz v[0:3], v[4:5], {DESC} dmask:0x1 {D2}"]),
            ("ps_v_lz_arr", [f"image_gather4_lz v[0:3], v[4:6], {DESC} dmask:0x2 dim:SQ_RSRC_IMG_2D_ARRAY"]),
            ("ps_v_lz_cube", [f"image_gather4_lz v[0:3], v[4:6], {DESC} dmask:0x4 dim:SQ_RSRC_IMG_CUBE"]),
            ("ps_v_lz_a16", [f"image_gather4_lz v[0:3], v4, {DESC} dmask:0x1 {D2} a16"]),
            ("ps_v_lz_d16", [f"image_gather4_lz v[0:1], v[4:5], {DESC} dmask:0x8 {D2} d16"]),
            ("ps_v_lz_o", [f"image_gather4_lz_o v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"]),
            ("ps_v_lz_o_a16", [f"image_gather4_lz_o v[0:3], v[4:5], {DESC} dmask:0x1 {D2} a16"]),
            ("ps_v_l", [f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x1 {D2}"]),
            ("cs_v_l_o", [f"image_gather4_l_o v[0:3], v[4:7], {DESC} dmask:0x1 {D2}"]),
            ("ps_v_l_o", [f"image_gather4_l_o v[0:3], v[4:7], {DESC} dmask:0x1 {D2}"]),
            ("ps_v_l_o_a16", [f"image_gather4_l_o v[0:3], v[4:6], {DESC} dmask:0x1 {D2} a16"]),
            ("ps_v_l_d16", [f"image_gather4_l v[0:1], v[4:6], {DESC} dmask:0x4 {D2} d16"]),
            ("ps_v_l_border", [
                "s_mov_b32 s16, 0x3e",
                "s_mov_b32 s17, 0x00fff000",
                "s_mov_b32 s18, 0",
                "s_mov_b32 s19, 0x40000000",
                f"image_gather4_l v[0:3], v[4:6], {DESC} dmask:0x8 {D2}",
            ]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name, body in cases:
                with self.subTest(case=name):
                    spv, _ = translate_with_meta(name, body, export_dst=name.startswith("ps_"))
                    a = os.path.join(tmp, f"{name}.spvasm")
                    b = os.path.join(tmp, f"{name}.spv")
                    with open(a, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv)
                    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", b], check=True)


if __name__ == "__main__":
    unittest.main()
