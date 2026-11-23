"""tools/pso_state_decode.py on hand-built state blobs (no game data)."""
import json
import os
import runpy
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import pso_state_decode as psd  # noqa: E402

LIB_TESTS = runpy.run_path(os.path.join(HERE, "test_shader_library.py"))


def state(layout, size, fields):
    """A blob of `size` bytes with word 1 = layout and the given {VS-layout offset: value} words, shifted for MS."""
    w = [0] * (size // 4)
    w[1] = layout
    s = psd.SHIFT[layout]
    for i in range(8):   # blending off on every target unless a field sets it: CB_BLEND 0x80000000, SX_MRT disabled
        w[(psd.BLEND + s) // 4 + i] = 0x80000000
        w[(psd.MRT_BLEND_OPT + s) // 4 + i] = psd.BLEND_OPT_DISABLED
    for off, v in fields.items():
        w[(off + s) // 4] = v
    return struct.pack(f"<{len(w)}I", *w)


ALPHA_BLEND = 0xE5010504   # color SRC_ALPHA ADD INV_SRC_ALPHA, separate alpha ONE ADD INV_SRC_ALPHA
FIELDS = {
    psd.COL_FORMAT: 0x44,           # two FP16_ABGR exports
    psd.SHADER_MASK: 0xFF,
    psd.TARGET_MASK: 0x7F,          # RT0 RGBA, RT1 RGB
    psd.BLEND: ALPHA_BLEND,
    psd.MRT_BLEND_OPT: 0x01510154,
    psd.DOWNCONVERT: 0x35,          # RT0 8_8_8_8, RT1 10_11_11
    psd.DEPTH_CONTROL: 0x2002E7,    # stencil on, depth on + write, GREATER_EQUAL; stencil EQUAL
    psd.STENCIL_CONTROL: 0x30030,   # pass REPLACE_TEST front and back
    psd.STENCIL_REF_MASK: 0x01FF0F05,
    psd.STENCIL_REF_MASK_BF: 0x01FF0F00,
}


class DecodeTests(unittest.TestCase):
    def check(self, layout, size):
        d = psd.decode(state(layout, size, FIELDS))
        self.assertEqual(d["layout"], "ms" if layout == psd.MS_LAYOUT else "vs")
        self.assertEqual(d["render_targets"], 2)
        self.assertEqual(d["export_format"], ["FP16_ABGR", "FP16_ABGR"])
        self.assertEqual(d["downconvert"], ["8_8_8_8", "10_11_11"])
        self.assertEqual(d["write_mask"], [15, 7])
        self.assertEqual(d["blend"][0], {"src": "SRC_ALPHA", "op": "ADD", "dest": "INV_SRC_ALPHA",
                                         "src_alpha": "ONE", "op_alpha": "ADD", "dest_alpha": "INV_SRC_ALPHA"})
        self.assertIsNone(d["blend"][1])
        self.assertTrue(d["blend_opt_consistent"])
        self.assertEqual(d["depth"], {"enable": True, "write": True, "func": "GREATER_EQUAL", "bounds": False})
        st = d["stencil"]
        self.assertTrue(st["enable"])
        self.assertEqual((st["func"], st["front"]["pass"], st["back"]["pass"]), ("EQUAL", "REPLACE_TEST", "REPLACE_TEST"))
        self.assertEqual((st["ref"], st["read_mask"], st["write_mask"]), (5, 0x0F, 0xFF))

    def test_vs_layout(self):
        self.check(psd.VS_LAYOUT, 388)

    def test_ms_layout_is_shifted(self):
        self.check(psd.MS_LAYOUT, 464)

    def test_inconsistent_blend_opt_is_reported(self):
        fields = {**FIELDS, psd.MRT_BLEND_OPT: psd.BLEND_OPT_DISABLED}   # blend on, optimizer says off
        self.assertFalse(psd.decode(state(psd.VS_LAYOUT, 388, fields))["blend_opt_consistent"])

    def test_unknown_layouts_are_refused(self):
        for blob in (state(psd.VS_LAYOUT, 388, FIELDS)[:100], b"\0" * 4 + struct.pack("<I", 5) + bytes(380)):
            with self.subTest(size=len(blob)), self.assertRaises(ValueError):
                psd.decode(blob)

    def test_cli_on_a_library(self):
        blobs = [state(psd.VS_LAYOUT, 388, FIELDS), b"POSITION\0", LIB_TESTS["layout_blob"](2, LIB_TESTS["D_PS"]),
                 LIB_TESTS["layout_blob"](2, LIB_TESTS["D_VS"], 2)]
        lib = LIB_TESTS["build"]([(0, 1, {0: (0, 2), 1: (1, 3)})], [], blobs,
                                 [LIB_TESTS["prog"](1), LIB_TESTS["prog"](2, 5)])
        with tempfile.TemporaryDirectory() as tmp:
            src, out = os.path.join(tmp, "lib.bin"), os.path.join(tmp, "state.json")
            with open(src, "wb") as f:
                f.write(lib)
            self.assertEqual(psd.main([src, "--out", out]), 0)
            with open(out, encoding="utf-8") as f:
                records = json.load(f)
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0]["record"], records[0]["state"], records[0]["render_targets"]), (0, 0, 2))


if __name__ == "__main__":
    unittest.main()
