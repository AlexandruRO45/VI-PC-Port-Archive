"""--min-lod-fallback: image_sample_d_cl without shaderResourceMinLod (Intel Xe-LP has no such feature).

The translator is fed hand-written listings, so this needs neither llvm-mc, spirv-tools, a GPU nor game data.
Without the flag the sample carries the MinLod image operand and the module declares the MinLod capability. With
it, neither appears: the gradients are replaced by an isotropic footprint of 2^minlod texels wherever their own
footprint (in level-0 texels) is finer than that, and the sample is a plain Grad sample. Run as

    python -I tests/test_min_lod_fallback.py

The sampled values are checked on a GPU by runtime/test/shaders/ps_texture_grad_cl_a16.asm (LEONIDA_MINLOD_FALLBACK=1).
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import rdna2spirv  # noqa: E402
from listing import translate_listing  # noqa: E402

DESC = "s[0:7], s[8:11]"
PROLOGUE = "\n".join(f"v_mov_b32 v{r}, 0" for r in (3, *range(10, 22)))
# (dim, instruction, gradient width, SPIR-V type of the size query, texel size shuffled to the gradient width)
FORMS = [
    ("1d", f"image_sample_d_cl v3, v[10:14], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_1D", 1, "%int", False),
    ("2d", f"image_sample_d_cl v3, v[10:17], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D", 2, "%v2int", False),
    ("2d_array", f"image_sample_d_cl v3, v[10:18], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY", 2, "%v3int", True),
    ("cube", f"image_sample_d_cl v3, v[10:18], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_CUBE", 2, "%v3int", True),
    ("3d", f"image_sample_d_cl v3, v[10:19], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_3D", 3, "%v3int", False),
    ("2d_a16", f"image_sample_d_cl v3, v[10:17], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D a16", 2, "%v2int", False),
]


def translate(asm, fallback):
    old = rdna2spirv.MIN_LOD_FALLBACK if hasattr(rdna2spirv, "MIN_LOD_FALLBACK") else None
    rdna2spirv.MIN_LOD_FALLBACK = fallback
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return translate_listing(Path(tmp), PROLOGUE + "\n" + asm)
    finally:
        rdna2spirv.MIN_LOD_FALLBACK = bool(old)


def instrs(spv, op):
    return [ln.strip() for ln in spv.splitlines() if re.search(rf"= {op}\b", ln)]


class MinLodFallbackTest(unittest.TestCase):
    def test_without_the_flag_the_sample_uses_minlod(self):
        for name, asm, *_ in FORMS:
            with self.subTest(form=name):
                spv = translate(asm, False)
                self.assertIn("OpCapability MinLod", spv)
                self.assertEqual(len([s for s in instrs(spv, "OpImageSampleExplicitLod") if "Grad|MinLod" in s]), 1)

    def test_with_the_flag_no_minlod_is_needed(self):
        for name, asm, *_ in FORMS:
            with self.subTest(form=name):
                spv = translate(asm, True)
                self.assertNotIn("MinLod", spv)
                self.assertEqual(len([s for s in instrs(spv, "OpImageSampleExplicitLod") if " Grad " in s]), 1)
                self.assertIn("OpCapability ImageQuery", spv)

    def test_sample_takes_the_selected_gradients(self):
        for name, asm, width, qtype, shuffled in FORMS:
            with self.subTest(form=name):
                spv = translate(asm, True)
                vec = "%float" if width == 1 else f"%v{width}float"
                sample = instrs(spv, "OpImageSampleExplicitLod")[0].split()
                gx, gy = sample[-2], sample[-1]
                selects = {m[1]: m for m in re.finditer(rf"(%\w+) = OpSelect {vec} (%\w+) (%\w+) (%\w+)", spv)}
                self.assertIn(gx, selects)
                self.assertIn(gy, selects)
                # one condition for both: rho^2 (max of the two squared texel-space lengths) < (2^minlod)^2
                self.assertEqual(selects[gx][2], selects[gy][2])
                cond = re.search(rf"{re.escape(selects[gx][2])} = OpFOrdLessThan %bool (%\w+) (%\w+)", spv)
                self.assertIsNotNone(cond)
                self.assertRegex(spv, rf"{re.escape(cond[1])} = OpExtInst %float %glsl FMax %\w+ %\w+")
                unit2 = re.search(rf"{re.escape(cond[2])} = OpFMul %float (%\w+) (%\w+)", spv)
                self.assertEqual(unit2[1], unit2[2])
                self.assertRegex(spv, rf"{re.escape(unit2[1])} = OpExtInst %float %glsl Exp2 %\w+")
                # the size comes from level 0 of the image
                self.assertRegex(spv, rf"= OpImageQuerySizeLod {re.escape(qtype)} %\w+ %\w+")
                self.assertEqual("OpVectorShuffle %v2int" in spv, shuffled)

    def test_unclamped_gradients_are_kept_as_they_are(self):
        # The false arm of each select is the shader's own gradient vector, so a footprint at or above the clamp
        # samples exactly as image_sample_d would.
        asm = FORMS[1][1]
        spv = translate(asm, True)
        plain = translate(asm.replace("image_sample_d_cl", "image_sample_d"), True)
        own = instrs(plain, "OpImageSampleExplicitLod")[0].split()[-2:]
        sample = instrs(spv, "OpImageSampleExplicitLod")[0].split()[-2:]
        for sel, mine in zip(sample, own):
            kept = re.search(rf"{re.escape(sel)} = OpSelect %v2float %\w+ %\w+ (%\w+)", spv)[1]
            self.assertEqual(re.search(rf"{re.escape(kept)} = OpCompositeConstruct .*", spv)[0].split(" = ")[1],
                             re.search(rf"{re.escape(mine)} = OpCompositeConstruct .*", plain)[0].split(" = ")[1])

    def test_clamped_footprint_is_two_to_the_minlod_texels_on_its_own_axis(self):
        spv = translate(FORMS[1][1], True)
        axes = []
        for sel in instrs(spv, "OpImageSampleExplicitLod")[0].split()[-2:]:
            iso = re.search(rf"{re.escape(sel)} = OpSelect %v2float %\w+ (%\w+)", spv)[1]
            parts = re.search(rf"{re.escape(iso)} = OpCompositeConstruct %v2float (%\w+) (%\w+)", spv)
            axes.append([p != "%f0" for p in parts.groups()])
        self.assertEqual(axes, [[True, False], [False, True]])

    def test_other_samples_are_unchanged(self):
        asm = f"image_sample_d v3, v[10:16], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D"
        self.assertEqual(translate(asm, True), translate(asm, False))

    def test_flag_is_parsed(self):
        # tools/rdna2spirv.py's own argument parser takes the flag and sets the global the translator reads
        src = Path(HERE, "..", "tools", "rdna2spirv.py").read_text(encoding="utf-8")
        self.assertIn('"--min-lod-fallback"', src)
        self.assertIn("MIN_LOD_FALLBACK = a.min_lod_fallback", src)


if __name__ == "__main__":
    unittest.main()
