"""Semantic and SPIR-V validation tests for image_load_mip across all sampled dimensions (1D, 2D, 2D_ARRAY, CUBE, 3D).

Covers 32-bit and a16 packed unsigned 16-bit coordinates+mip, normal and d16 packed output, typed float/uint/sint
fetch aliases on bindings 0/3/5/6, absence of storage-image capabilities, spirv-val validation when on PATH, and
refusal of glc, tfe/lwe status returns, short coordinate lists, and unsupported dimensions.

    python -I tests/semantic/test_image_load_mip_dims.py -v
"""
import os
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


class ImageLoadMipDimsTests(unittest.TestCase):
    def test_all_five_dims_32bit_coords_bindings_types_and_lod(self):
        # (dim, addr, binding, spv_dim, coord_type, expected_vgprs)
        cases = [
            ("SQ_RSRC_IMG_1D",       "v[4:5]", 5, "1D", None,     [4, 5]),
            ("SQ_RSRC_IMG_2D",       "v[4:6]", 0, "2D", "%v2int", [4, 5, 6]),
            ("SQ_RSRC_IMG_2D_ARRAY", "v[4:7]", 3, "2D", "%v3int", [4, 5, 6, 7]),
            ("SQ_RSRC_IMG_CUBE",     "v[4:7]", 3, "2D", "%v3int", [4, 5, 6, 7]),
            ("SQ_RSRC_IMG_3D",       "v[4:7]", 6, "3D", "%v3int", [4, 5, 6, 7]),
        ]
        for dim, addr, binding, spv_dim, vec_ty, want_regs in cases:
            for d16_mod, dst in (("", "v[0:3]"), (" d16", "v[0:1]")):
                with self.subTest(dim=dim, d16=bool(d16_mod)):
                    spv = _translate_ps(
                        f"ps_mip_{dim.lower()}",
                        [f"image_load_mip {dst}, {addr}, s[0:7] dmask:0xf dim:{dim}{d16_mod}"],
                    )
                    m = typed.Module(spv)
                    fetches = m.fetches()
                    self.assertEqual({image[0] for _, _, image in fetches}, {"%float", "%uint", "%int"})
                    self.assertFalse(m.caps & typed.STORAGE_IMAGE_CAPS)
                    self.assertNotIn("OpCapability ImageQuery", spv)
                    for rid, var, image in fetches:
                        self.assertEqual((m.binding[var], image[1]), (binding, spv_dim))
                        _, _, coord, mode, lod = m.defs[rid][1]
                        self.assertEqual(mode, "Lod")
                        if vec_ty is None:
                            self.assertEqual([m.integer_coord_vgpr(coord), m.integer_coord_vgpr(lod)], want_regs)
                        else:
                            op, args = m.defs[coord]
                            self.assertEqual((op, args[0]), ("OpCompositeConstruct", vec_ty))
                            self.assertEqual([m.integer_coord_vgpr(v) for v in [*args[1:], lod]], want_regs)

    def test_all_five_dims_a16_unpacks_coords_and_mip(self):
        regs = {4: 0x00120005, 5: 0x00070003}  # (lo=5, hi=18), (lo=3, hi=7)
        cases = [
            ("SQ_RSRC_IMG_1D",       "v4",     5, None,     (5, 18)),
            ("SQ_RSRC_IMG_2D",       "v[4:5]", 0, "%v2int", (5, 18, 3)),
            ("SQ_RSRC_IMG_2D_ARRAY", "v[4:5]", 3, "%v3int", (5, 18, 3, 7)),
            ("SQ_RSRC_IMG_CUBE",     "v[4:5]", 3, "%v3int", (5, 18, 3, 7)),
            ("SQ_RSRC_IMG_3D",       "v[4:5]", 6, "%v3int", (5, 18, 3, 7)),
        ]
        for dim, addr, binding, vec_ty, expected in cases:
            for d16_mod, dst in (("", "v[0:1]"), (" d16", "v0")):
                with self.subTest(dim=dim, d16=bool(d16_mod)):
                    spv = _translate_ps(
                        f"ps_mip_a16_{dim.lower()}",
                        [f"image_load_mip {dst}, {addr}, s[0:7] dmask:0x9 dim:{dim} a16{d16_mod}"],
                    )
                    m = typed.Module(spv, vgprs=regs)
                    fetches = m.fetches()
                    self.assertEqual({image[0] for _, _, image in fetches}, {"%float", "%uint", "%int"})
                    for rid, var, _ in fetches:
                        self.assertEqual(m.binding[var], binding)
                        _, _, coord, mode, lod = m.defs[rid][1]
                        self.assertEqual(mode, "Lod")
                        if vec_ty is None:
                            got = (m.value(coord, 0, 0), m.value(lod, 0, 0))
                        else:
                            op, args = m.defs[coord]
                            self.assertEqual((op, args[0]), ("OpCompositeConstruct", vec_ty))
                            got = tuple(m.value(v, 0, 0) for v in (*args[1:], lod))
                        self.assertEqual(got, expected)

    def test_status_modifier_compatibility_matches_main(self):
        # Preserve main's corpus compatibility for non-a16 status forms (except 3D d16 status).
        compat_cases = [
            "image_load_mip v[0:4], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D tfe",
            "image_load_mip v[0:4], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D lwe",
            "image_load_mip v[0:2], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D d16 tfe",
            "image_load_mip v[0:4], v[4:7], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D tfe",
        ]
        for line in compat_cases:
            with self.subTest(line=line):
                _translate_ps("ps_mip_status_compat", [line])

    def test_refused_forms_stay_unsupported(self):
        bad_cases = [
            "image_load_mip v[0:3], v[4:7], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY glc",
            "image_load_mip v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D a16 tfe",
            "image_load_mip v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D a16 lwe",
            "image_load_mip v[0:2], v[4:7], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D d16 tfe",
            "image_load_mip v[0:2], v[4:7], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D d16 lwe",
            "image_load_mip v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY",   # needs 4 coords
            "image_load_mip v[0:3], v4, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_3D a16",          # needs 2 regs
            "image_load_mip v[0:3], v4, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D",              # needs 2 regs
            "image_load_mip v[0:3], v[4:6], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D_ARRAY",    # no 1D_ARRAY binding
        ]
        for line in bad_cases:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines("ps_mip_bad", [line, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
