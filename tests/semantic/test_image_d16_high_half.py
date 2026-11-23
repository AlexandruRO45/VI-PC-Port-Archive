"""SHADER-1065: a d16 image read that fills an odd number of halves writes only the low half of its last VGPR, and
RDNA2 keeps that VGPR's high half. tools/d16_image_probe.py measures it for dmask 0x1 image_load, image_sample_lz and
bilinear image_sample on 2D, and image_load and image_sample_lz on 2D_ARRAY and CUBE (`d16_high_half_kept`, every lane
on gfx1031 and gfx1036); a one-off oracle run on gfx1036 (wave32 and wave64) found the same for image_sample_b,
image_sample_l, image_sample_d, the five image_sample_c* kinds, image_load_mip, 1D and 3D, and dmask 0x7 (the second
VGPR); image_sample_c_d is not translated at all, so it is not listed here. The translator paired the odd last component with 0.0 on the generic sample path and wrote the whole d16
image_sample_c* result, so the high half came back 0.

spirv_mini_interp has no image instructions, so this checks the emitted code: the last store to that VGPR is
`(old & 0xFFFF0000) | low half` (Gen.write_lo16), and a read that fills whole VGPRs (an even count, or no d16) still
writes them whole."""
import os
import re
import shutil
import tempfile
import unittest

import support
import rdna2spirv

PRE = [f"v_mov_b32 v{r}, {r}" for r in range(4, 12)]
EPI = ["s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
SAMPLER = "s[0:7], s[8:11]"
# (instruction, the VGPR whose low half only is written)
ODD = [
    (f"image_sample_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY d16", 0),
    (f"image_sample_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_CUBE d16", 0),
    (f"image_sample_lz v[0:1], v[4:6], {SAMPLER} dmask:0x7 dim:SQ_RSRC_IMG_2D_ARRAY d16", 1),
    (f"image_sample_lz v[0:1], v[4:6], {SAMPLER} dmask:0x7 dim:SQ_RSRC_IMG_CUBE d16", 1),
    (f"image_sample v0, v[4:5], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_b v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_l v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_d v0, v[4:9], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_lz v[0:1], v[4:5], {SAMPLER} dmask:0x7 dim:SQ_RSRC_IMG_2D d16", 1),
    (f"image_sample_lz v0, v4, {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_1D d16", 0),
    (f"image_sample_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_3D d16", 0),
    (f"image_sample_lz v[0:1], v[4:6], {SAMPLER} dmask:0x7 dim:SQ_RSRC_IMG_3D d16", 1),
    (f"image_sample_c v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_c_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_c_l v0, v[4:7], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_c_b v0, v[4:7], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 0),
    # image_read_typed kept the high half already; covered so the two paths stay alike
    ("image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16", 0),
    ("image_load v[0:1], v[4:5], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16", 1),
]
# Every half of the last VGPR is a result: it is written whole.
WHOLE = [
    (f"image_sample_lz v0, v[4:6], {SAMPLER} dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY d16", 0),
    (f"image_sample_lz v[0:1], v[4:6], {SAMPLER} dmask:0xf dim:SQ_RSRC_IMG_CUBE d16", 1),
    (f"image_sample_b v0, v[4:6], {SAMPLER} dmask:0x3 dim:SQ_RSRC_IMG_2D d16", 0),
    (f"image_sample_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY", 0),
    (f"image_sample_c_lz v0, v[4:6], {SAMPLER} dmask:0x1 dim:SQ_RSRC_IMG_2D", 0),
]
HIGH = "%cffff0000"


def translate(line):
    return support.translate_lines("ps_d16_image_high_half", PRE + [line] + EPI)


def defs(text):
    """{result id: (opcode, operands)} of the module."""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+) (.*)", ln)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3).split())
    return out


def last_store(text, reg):
    stores = re.findall(rf"^\s*OpStore %reg_v{reg} (%\w+)\s*$", text, re.M)
    if not stores:
        raise AssertionError(f"no store to v{reg}")
    return stores[-1]


def kept_high_half(text, reg):
    """True when the last value stored to v<reg> is (v<reg> & 0xFFFF0000) | x, through the exec select."""
    d = defs(text)
    v = last_store(text, reg)
    op, args = d[v]
    if op == "OpSelect":                     # active lanes take the new value, the others keep theirs
        v = args[2]
        op, args = d[v]
    if op != "OpBitwiseOr":
        return False
    for a in args[1:]:
        if a in d and d[a][0] == "OpBitwiseAnd" and d[a][1][2] == HIGH:
            src = d[a][1][1]
            while d.get(src, ("",))[0] == "OpCopyObject":
                src = d[src][1][1]
            if d.get(src) == ("OpLoad", ["%uint", f"%reg_v{reg}"]):
                return True
    return False


class HighHalf(unittest.TestCase):
    def test_odd_last_half_keeps_the_vgpr_high_half(self):
        for line, reg in ODD:
            with self.subTest(line=line):
                self.assertTrue(kept_high_half(translate(line), reg))

    def test_full_vgprs_are_written_whole(self):
        for line, reg in WHOLE:
            with self.subTest(line=line):
                self.assertFalse(kept_high_half(translate(line), reg))

    def test_the_vgprs_before_the_last_are_written_whole(self):
        for line, reg in ODD:
            if reg:
                with self.subTest(line=line):
                    self.assertFalse(kept_high_half(translate(line), reg - 1))


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class Validation(unittest.TestCase):
    def test_spirv_val_accepts_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, (line, _) in enumerate(ODD + WHOLE):
                with self.subTest(line=line):
                    sa, sv = os.path.join(tmp, f"{i}.spvasm"), os.path.join(tmp, f"{i}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(translate(line))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
