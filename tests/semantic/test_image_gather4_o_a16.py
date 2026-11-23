"""Semantic and SPIR-V validation tests for a16 packed f16 coordinates on image_gather4_lz_o and image_gather4_l_o.

Under GFX10 a16, the first address VGPR of an *_o gather keeps the packed signed 6-bit texel offsets (x in [5:0],
y in [13:8]) as a full 32-bit dword, while the following VGPRs hold packed f16 pairs for coordinates (and LOD).

    python -I tests/semantic/test_image_gather4_o_a16.py -v
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402
import test_image_load_typed as typed  # noqa: E402


def _validate_spv(spvasm):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, "out.spv")
        r = subprocess.run([spirv_as, "--target-env", "vulkan1.2", "-o", spv, "-"],
                           input=spvasm, text=True, capture_output=True)
        if r.returncode:
            raise AssertionError(f"spirv-as failed:\n{r.stderr}")
        r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"spirv-val failed:\n{r.stderr}")


def _translate_ps(name, body):
    lines = list(body) + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines(name, lines)
    _validate_spv(spvasm)
    return spvasm


def _loaded_vgpr(m, ssa_id):
    op, args = m.defs[ssa_id]
    assert op == "OpLoad", (ssa_id, op, args)
    pop, pargs = m.defs[args[1]]
    assert pop == "OpAccessChain" and pargs[1] == "%vgpr", (pop, pargs)
    return m.value(pargs[2], 0, 0)


def _offset_sextracts(spvasm):
    """Return [(src_vgpr_index, offset_bit, width)] for every 6-bit OpBitFieldSExtract feeding a v2int offset."""
    m = typed.Module(spvasm)
    out = []
    for _, op, args in m.order:
        if op == "OpBitFieldSExtract" and len(args) == 4:
            off_bit = m.value(args[2], 0, 0)
            width = m.value(args[3], 0, 0)
            if width == 6 and off_bit in (0, 8):
                out.append((_loaded_vgpr(m, args[1]), off_bit, width))
    return out


def _unpacked_vgprs(spvasm):
    """Return the VGPR indices passed to UnpackHalf2x16 in order."""
    m = typed.Module(spvasm)
    vgprs = []
    for _, op, args in m.order:
        if op == "OpExtInst" and len(args) >= 4 and args[2] == "UnpackHalf2x16":
            vgprs.append(_loaded_vgpr(m, args[3]))
    return vgprs


class ImageGather4OffsetA16Tests(unittest.TestCase):
    def test_gather4_lz_o_a16_2d_keeps_offset_whole_and_unpacks_uv(self):
        for d16_mod, dst in (("", "v[0:3]"), (" d16", "v[0:1]")):
            with self.subTest(d16=bool(d16_mod)):
                spv = _translate_ps("ps_g4_lz_o_2d_a16", [
                    f"image_gather4_lz_o {dst}, v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D a16{d16_mod}",
                ])
                self.assertIn("OpCapability ImageGatherExtended", spv)
                self.assertEqual(_offset_sextracts(spv), [(4, 0, 6), (4, 8, 6)])
                # v4 is the raw integer offset dword (never unpacked as f16); v5 holds (u, v).
                self.assertEqual(_unpacked_vgprs(spv), [5])
                gathers = [ln for ln in spv.splitlines() if "= OpImageGather " in ln]
                self.assertEqual(len(gathers), 3)  # float / uint / sint typed aliases on 2D
                self.assertTrue(all(" Offset " in ln for ln in gathers))

    def test_gather4_lz_o_a16_2d_array_and_cube(self):
        for dim in ("SQ_RSRC_IMG_2D_ARRAY", "SQ_RSRC_IMG_CUBE"):
            with self.subTest(dim=dim):
                spv = _translate_ps(f"ps_g4_lz_o_{dim.lower()}_a16", [
                    f"image_gather4_lz_o v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x4 dim:{dim} a16",
                ])
                self.assertIn("OpCapability ImageGatherExtended", spv)
                self.assertEqual(_offset_sextracts(spv), [(4, 0, 6), (4, 8, 6)])
                # v4 = offset, v5 = (u, v), v6 = (layer/face, pad)
                self.assertEqual(_unpacked_vgprs(spv), [5, 6])
                gathers = [ln for ln in spv.splitlines() if "= OpImageGather " in ln]
                self.assertEqual(len(gathers), 1)
                self.assertIn(" Offset ", gathers[0])

    def test_gather4_l_o_a16_2d_keeps_offset_whole_and_unpacks_uv_and_lod(self):
        for d16_mod, dst in (("", "v[0:3]"), (" d16", "v[0:1]")):
            with self.subTest(d16=bool(d16_mod)):
                spv = _translate_ps("ps_g4_l_o_2d_a16", [
                    f"image_gather4_l_o {dst}, v[4:6], s[0:7], s[8:11] dmask:0x2 dim:SQ_RSRC_IMG_2D a16{d16_mod}",
                ])
                # image_gather4_l_o folds the integer offset via OpIAdd + OpImageFetch Lod (no ImageGatherExtended).
                self.assertNotIn("OpCapability ImageGatherExtended", spv)
                self.assertEqual(_offset_sextracts(spv), [(4, 0, 6), (4, 8, 6)])
                # v4 = offset, v5 = (u, v), v6 = (lod, pad)
                self.assertEqual(_unpacked_vgprs(spv), [5, 6])
                self.assertIn("OpImageFetch", spv)
                self.assertTrue(re.search(r"OpIAdd %v2int %\w+ %\w+", spv))

    def test_short_a16_coord_lists_are_refused(self):
        for bad in (
            "image_gather4_lz_o v[0:3], v4, s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
            "image_gather4_lz_o v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY a16",
            "image_gather4_l_o v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_g4_o_bad", [bad, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
