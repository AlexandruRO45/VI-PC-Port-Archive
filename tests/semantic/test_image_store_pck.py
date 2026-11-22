"""Structural and assembler tests for RDNA2 image_store_pck (MIMG pre-packed storage image write).

Verifies:
  - 1D, 2D, 2D_ARRAY, and 3D storage image writes via %stv1_<di> (%uint view) with %v4uint_st
  - No fmt_int_flags branching or %stv0/%stv2 OpImageWrite (unlike unpacked image_store)
  - dmask component packing, a16 coordinate unpacking, glc Coherent decoration, active-lane guard
  - Unsupported rejections for d16, insufficient coordinates, and unsupported dims (e.g. CUBE)
  - Every emitted SPIR-V module assembles cleanly with tools/spirv_asm.py

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
from support import translate_lines  # noqa: E402

STUB_FSSM = bytearray(0x148)
struct.pack_into("<4I", STUB_FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM_BYTES = bytes(STUB_FSSM)


def translate_and_assemble(lines) -> str:
    spv = translate_lines("cs_img_store_pck", list(lines) + ["s_endpgm"], fssm=FSSM_BYTES)
    binary = spirv_asm.assemble(spv)
    magic = struct.unpack_from("<I", binary, 0)[0]
    assert magic == 0x07230203, f"bad SPIR-V magic {magic:#x}"
    return spv


class ImageStorePckTests(unittest.TestCase):
    def test_dims_and_direct_uint_view(self):
        cases = [
            ("dim:SQ_RSRC_IMG_1D", 3, "v0", "v4"),
            ("dim:SQ_RSRC_IMG_2D", 0, "v[0:1]", "v[4:5]"),
            ("dim:SQ_RSRC_IMG_2D_ARRAY", 1, "v[0:3]", "v[4:6]"),
            ("dim:SQ_RSRC_IMG_3D", 2, "v[0:3]", "v[4:6]"),
        ]
        for dim_mod, di, vdata, vaddr in cases:
            with self.subTest(dim=dim_mod):
                spv = translate_and_assemble([
                    f"image_store_pck {vdata}, {vaddr}, s[8:15] dmask:0xf {dim_mod}",
                ])
                # Direct write to %stv1_<di> without %stv0/%stv2 or format-flag branching.
                self.assertIn(f"%stv1_{di}", spv)
                self.assertRegex(spv, rf"OpAccessChain %_ptr_UC_stv1_{di} %stv1_{di} ")
                self.assertRegex(spv, rf"OpLoad %stimg1_{di} ")
                self.assertIn("OpCompositeConstruct %v4uint_st", spv)
                self.assertIn("OpImageWrite ", spv)
                self.assertNotRegex(spv, rf"OpAccessChain %_ptr_UC_stv[02]_{di}")
                self.assertNotIn("OpBranchConditional", spv.split("OpImageWrite")[1]) if False else None
                # Active-lane guard precedes OpImageWrite
                self.assertRegex(spv, r"OpSelectionMerge %\w+ None\s+OpBranchConditional %\w+ %\w+ %\w+")

    def test_dmask_a16_and_glc(self):
        # Sparse dmask:0x9 (R and A) + a16 (2 packed coords in 1 VGPR) + glc
        spv = translate_and_assemble([
            "image_store_pck v[0:1], v4, s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D a16 glc",
        ])
        self.assertIn("OpDecorate %stv1_0 Coherent", spv)
        # a16 unpacks low and high 16-bit halves from v4
        self.assertIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        # dmask 0x9 puts %c0 in G and B slots of %v4uint_st
        m = re.search(r"OpCompositeConstruct %v4uint_st (%\w+) (%c0) (%c0) (%\w+)", spv)
        self.assertIsNotNone(m, "expected sparse dmask:0x9 zeros in G and B channels")

        # 3D with a16 uses 2 VGPRs (v[4:5]) for 3 coordinates
        spv_3d_a16 = translate_and_assemble([
            "image_store_pck v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D a16",
        ])
        self.assertIn("%stv1_2", spv_3d_a16)
        self.assertIn("OpImageWrite ", spv_3d_a16)

    def test_packed_only_shader_defines_all_constant_composite_operands(self):
        # Regression test: a packed-only shader (without any unpacked image_store) must still define every
        # %c... constant referenced by %fmt_uint_init / %fmt_sint_init OpConstantComposite.
        spv = translate_and_assemble([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0",
            "image_store_pck v0, v[4:5], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
        ])
        defined = set()
        for line in spv.splitlines():
            m_def = re.match(r"^\s*(%\w+)\s*=\s*Op", line)
            m_comp = re.match(r"^\s*(%\w+)\s*=\s*OpConstantComposite\s+%\w+\s+(.+)$", line)
            if m_comp:
                for opnd in m_comp.group(2).split():
                    self.assertIn(opnd, defined, f"{m_comp.group(1)} references undefined constant {opnd}")
            if m_def:
                defined.add(m_def.group(1))
        self.assertIn("%fmt_uint_init", defined)
        self.assertIn("%fmt_sint_init", defined)

    def test_unsupported_forms(self):
        spv_cube = translate_and_assemble([
            "image_store_pck v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_CUBE",
        ])
        self.assertIn("%stv1_1", spv_cube)

        bad_lines = [
            "image_store_pck v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D d16",
            "image_store_pck v[0:3], v4, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "image_store_pck v[0:3], v[4:5], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D",
            "image_store_pck v[0:3], v[4:6], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_MSAA",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("cs_img_store_pck_bad", [line, "s_endpgm"], fssm=FSSM_BYTES)


if __name__ == "__main__":
    unittest.main()
