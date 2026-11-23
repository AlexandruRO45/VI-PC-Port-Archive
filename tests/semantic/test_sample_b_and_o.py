"""Translator tests for implicit-LOD bias and offset image_sample_* forms:
  image_sample_b, image_sample_b_o, image_sample_b_cl_o,
  image_sample_c_cl_o, image_sample_c_b_cl_o.

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


class SampleBAndO(unittest.TestCase):
    def one(self, name, line, ty="%v4float"):
        (s,) = samples(translate(name, line), ty=ty)
        return s

    def test_b_emits_bias_in_ps_and_is_refused_outside_ps(self):
        spv = translate("ps_b", f"image_sample_b v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
        (s,) = samples(spv, ty="%v4float")
        self.assertRegex(s, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ Bias %\w+$")
        self.assertEqual(source_vgpr(spv, s.split()[-1]), 4)

        for name in ("vs_b", "cs_b"):
            with self.subTest(stage=name):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    translate(name, f"image_sample_b v[0:3], v[4:6], {DESC} dmask:0xf {D2}")

    def test_b_o_and_b_cl_o_emit_bias_and_offset_operands(self):
        s_bo = self.one("ps_bo", f"image_sample_b_o v[0:3], v[4:7], {DESC} dmask:0xf {D2}")
        self.assertRegex(s_bo, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ Bias\|Offset %\w+ %\w+$")

        s_bclo = self.one("ps_bclo", f"image_sample_b_cl_o v[0:3], v[4:8], {DESC} dmask:0xf {D2}")
        self.assertRegex(s_bclo, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ Bias\|Offset\|MinLod %\w+ %\w+ %\w+$")

    def test_dref_cl_o_and_c_b_cl_o_emit_dref_operands(self):
        s_cclo = self.one("ps_cclo", f"image_sample_c_cl_o v0, v[4:8], {DESC} dmask:0x1 {D2}", ty="%float")
        self.assertRegex(s_cclo, r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+ Offset\|MinLod %\w+ %\w+$")

        s_cbclo = self.one("ps_cbclo", f"image_sample_c_b_cl_o v0, v[4:9], {DESC} dmask:0x1 {D2}", ty="%float")
        self.assertRegex(
            s_cbclo,
            r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+ Bias\|Offset\|MinLod %\w+ %\w+ %\w+$",
        )

    def test_d16_packs_two_f16_halves(self):
        spv = translate("ps_bd16", f"image_sample_b v0, v[4:6], {DESC} dmask:0x3 {D2} d16")
        idx = spv.index("OpImageSampleImplicitLod %v4float")
        tail = spv[idx:]
        self.assertIn("OpShiftLeftLogical %uint", tail)
        self.assertIn("OpBitwiseOr %uint", tail)
        self.assertRegex(tail, r"OpSelect %uint %\w+ %\w+ %\w+\n\s*OpStore %reg_v0 ")


if __name__ == "__main__":
    unittest.main()
