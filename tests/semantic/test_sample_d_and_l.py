"""Translator tests for explicit-gradient and explicit-LOD image_sample_* forms:
  image_sample_d, image_sample_d_cl,
  image_sample_l, image_sample_l_o,
  image_sample_lz, image_sample_lz_o,
  image_sample_c_l, image_sample_c_lz_o.

Standard library only; no GPU, no LLVM, no game files.
"""
import os
import re
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

DESC = "s[8:15], s[16:19]"
D2 = "dim:SQ_RSRC_IMG_2D"
PRO = [f"v_mov_b32 v{r}, 0.5" for r in range(4, 12)]
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(name, line):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = PRO + [line] + (["exp mrt0 v0, v1, v2, v3 done vm"] if name.startswith("ps_") else []) + ["s_endpgm"]
    return support.translate_lines(name, body, fssm=fssm)


def samples(spv, ty=None):
    lines = [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]
    return [ln for ln in lines if f" {ty} " in ln] if ty else lines


def source_vgpr(spv, value):
    """The VGPR index `value` was loaded from (through bitcasts), or None."""
    consts = {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}
    for _ in range(8):
        m = re.search(rf"^\s*{re.escape(value)} = (\w+) (?:%\w+ )?(%\w+)", spv, re.M)
        if not m:
            return None
        op, src = m.groups()
        if op == "OpLoad":
            m2 = re.search(rf"^\s*{re.escape(src)} = OpAccessChain %\S+ %vgpr (%\w+)", spv, re.M)
            return consts.get(m2.group(1)) if m2 else None
        value = src
    return None


class SampleDAndL(unittest.TestCase):
    def one(self, name, line, ty="%v4float"):
        (s,) = samples(translate(name, line), ty=ty)
        return s

    def test_d_and_d_cl_emit_grad_operands_in_ps_and_cs(self):
        for prefix in ("ps", "cs"):
            with self.subTest(stage=prefix):
                s_d = self.one(f"{prefix}_d", f"image_sample_d v[0:3], v[4:9], {DESC} dmask:0xf {D2}")
                self.assertRegex(s_d, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad %\w+ %\w+$")

                s_dcl = self.one(f"{prefix}_dcl", f"image_sample_d_cl v[0:3], v[4:10], {DESC} dmask:0xf {D2}")
                self.assertRegex(s_dcl, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad\|MinLod %\w+ %\w+ %\w+$")

    def test_l_and_l_o_emit_explicit_lod(self):
        spv_l = translate("ps_l", f"image_sample_l v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
        (s_l,) = samples(spv_l, ty="%v4float")
        self.assertRegex(s_l, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod %\w+$")
        self.assertEqual(source_vgpr(spv_l, s_l.split()[-1]), 6)

        s_lo = self.one("ps_lo", f"image_sample_l_o v[0:3], v[4:7], {DESC} dmask:0xf {D2}")
        self.assertRegex(s_lo, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %\w+ %\w+$")

    def test_lz_and_lz_o_emit_lod_zero(self):
        for prefix in ("ps", "cs"):
            with self.subTest(stage=prefix):
                s_lz = self.one(f"{prefix}_lz", f"image_sample_lz v[0:3], v[4:5], {DESC} dmask:0xf {D2}")
                self.assertRegex(s_lz, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod %f0$")

                s_lzo = self.one(f"{prefix}_lzo", f"image_sample_lz_o v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
                self.assertRegex(s_lzo, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod\|Offset %f0 %\w+$")

    def test_c_l_and_c_lz_o_emit_dref_explicit_lod(self):
        spv_cl = translate("ps_cl", f"image_sample_c_l v0, v[4:7], {DESC} dmask:0x1 {D2}")
        (s_cl,) = samples(spv_cl, ty="%float")
        self.assertRegex(s_cl, r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod %\w+$")
        # GFX10 address order for _c_l: z-compare (v4), u (v5), v (v6), lod (v7).
        self.assertEqual(source_vgpr(spv_cl, s_cl.split()[6]), 4)
        self.assertEqual(source_vgpr(spv_cl, s_cl.split()[-1]), 7)

        s_clzo = self.one("ps_clzo", f"image_sample_c_lz_o v0, v[4:7], {DESC} dmask:0x1 {D2}", ty="%float")
        self.assertRegex(s_clzo, r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %f0 %\w+$")


if __name__ == "__main__":
    unittest.main()
