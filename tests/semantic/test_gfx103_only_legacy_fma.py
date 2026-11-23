"""v_fmac_legacy_f32 / v_fma_legacy_f32 are refused on purpose, with a message that says why.

They are the gfx103x spellings of VOP2 0x06 / VOP3 0x106 / 0x140 (fused legacy multiply-add). The console's gfx1013
runs those slots as the unfused v_mac_legacy_f32 / v_mad_legacy_f32, which translate (docs/design.md, "GFX10.1
(console gfx1013) / GFX10.3 (gfx1030, gfx1036 oracle) opcode differences"). A listing that prints the fused names was
disassembled for the wrong target (llvm-mc --mcpu gfx1030, as tests/test_rdna2_decode.py does for 0x140), so it must
fail loudly rather than be translated as a fused multiply-add; the message points to --mcpu gfx1013 and the row."""
import unittest

import support
import rdna2spirv

PRE = ["v_mov_b32 v1, 1.0", "v_mov_b32 v2, 2.0", "v_mov_b32 v3, 3.0"]
EPI = ["exp mrt0 v1, v1, v1, v1 done vm", "s_endpgm"]
REFUSED = [
    "v_fmac_legacy_f32 v1, v2, v3",
    "v_fmac_legacy_f32_e32 v1, v2, v3",
    "v_fmac_legacy_f32_e64 v1, v2, v3",
    "v_fma_legacy_f32 v1, v2, v3, v1",
]
GFX1013 = ["v_mac_legacy_f32 v1, v2, v3", "v_mad_legacy_f32 v1, v2, v3, v1"]


def translate(line):
    return support.translate_lines("ps_gfx103_only_legacy_fma", PRE + [line] + EPI)


class Gfx103OnlyLegacyFma(unittest.TestCase):
    def test_refused_with_the_reason(self):
        for line in REFUSED:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    translate(line)
                msg = str(cm.exception)
                self.assertIn("gfx103x-only", msg)
                self.assertIn("--mcpu gfx1013", msg)
                self.assertIn("v_mad_legacy_f32", msg)

    def test_the_gfx1013_spellings_still_translate(self):
        for line in GFX1013:
            with self.subTest(line=line):
                self.assertIn("OpFMul", translate(line))


if __name__ == "__main__":
    unittest.main()
