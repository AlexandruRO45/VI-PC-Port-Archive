"""image_load_pck, image_load_pck_sgn, image_load_mip_pck and image_load_mip_pck_sgn: raw texel dwords, no format
conversion. Measured on an AMD Radeon RDNA2 iGPU (gfx1036, driver 26.10.1) with raw kernels through tools/hw_oracle:

* the result is the texel's memory bytes as dwords (little endian, zero padded) whatever the FORMAT's numeric type is
  (UNORM, UINT, SINT, FLOAT give the same bytes) and whatever the T# DST_SEL says;
* dmask bit k selects dword k of the texel, the selected dwords go to consecutive VGPRs in ascending bit order, and a
  dword past the texel's size is 0 (dmask 0x2 on a 4 byte texel gives 0);
* _sgn sign extends dword 0 of a texel under 32 bits (1 byte from bit 7, 2 bytes from bit 15) and leaves bigger ones;
* the mip forms with mip 0 are the same; an out of range coordinate gives 0.

The translator rebuilds the bytes from the typed fetch (the integer, UNORM, FLOAT16 and FLOAT32 formats of 8, 16 and 32
bit components; any other FORMAT gives 0). The words below are what the hardware returned; the model repeats the
translator's table-driven packing in Python. The emitted SPIR-V runs on the GPUs in cs_imgpck_rgba8 / _rgba16 / _r8.
Standard library only; the validation tests need spirv-as and spirv-val.
"""
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, os.path.join(HERE, "semantic"))

import rdna2spirv  # noqa: E402
from test_image_load_typed import translate  # noqa: E402

GFX10 = {"8_UNORM": 1, "8_UINT": 5, "8_SINT": 6, "8_8_UNORM": 14, "8_8_UINT": 18, "8_8_8_8_UNORM": 56, "8_8_8_8_UINT": 60,
         "8_8_8_8_SINT": 61, "16_UNORM": 7, "16_UINT": 11, "16_FLOAT": 13, "16_16_16_16_UNORM": 65,
         "16_16_16_16_UINT": 69, "16_16_16_16_FLOAT": 71, "32_UINT": 20, "32_SINT": 21, "32_FLOAT": 22,
         "32_32_UINT": 62, "32_32_32_UINT": 72, "32_32_32_32_UINT": 75, "32_32_32_32_FLOAT": 77}
SIZES = {1: 1, 2: 2, 3: 4, 4: 8, 5: 12, 6: 16}
TABLE = rdna2spirv.Gen.pck_format_table()


def model(fmt, comps, dmask, sgn):
    """Dwords image_load_pck gives for a texel of FORMAT `fmt` whose components (what the typed fetch returns: raw ints,
    or floats for UNORM / FLOAT) are `comps`, as the translator computes them from its table."""
    ent = TABLE[GFX10[fmt]]
    size, kind = ent & 7, ent >> 3 & 7
    widths = [ent >> (6 + 6 * i) & 63 for i in range(4)]
    raw, off, words = [], 0, [0, 0, 0, 0]
    for c, w in zip(comps + [0] * 4, widths):
        if kind == 1:
            r = int(c) & ((1 << w) - 1)
        elif kind == 2:
            r = min(max(int(c * ((1 << w) - 1) + 0.5), 0), (1 << w) - 1)
        elif kind == 3:
            import struct
            r = struct.unpack("<H", struct.pack("<e", c))[0]
        else:
            r = c
        words[off >> 5] |= (r << (off & 31)) & 0xFFFFFFFF
        off += w
    if sgn and size in (1, 2):
        bits = 8 * size
        if words[0] >> (bits - 1) & 1:
            words[0] |= 0xFFFFFFFF << bits & 0xFFFFFFFF
    return [words[k] for k in range(4) if dmask >> k & 1]


class Table(unittest.TestCase):
    def test_sizes_kinds_and_widths(self):
        self.assertEqual(len(TABLE), 512)
        for fmt, code in GFX10.items():
            ent = TABLE[code]
            ws = [int(t) for t in fmt.split("_")[:-1]]
            ty = fmt.rsplit("_", 1)[1]
            with self.subTest(fmt=fmt):
                self.assertEqual(SIZES[ent & 7], sum(ws) // 8)
                self.assertEqual([ent >> (6 + 6 * i) & 63 for i in range(4)], ws + [0] * (4 - len(ws)))
                self.assertEqual(ent >> 3 & 7, {"UINT": 1, "SINT": 1, "UNORM": 2, "FLOAT": 3 if ws[0] == 16 else 4}[ty])

    def test_formats_that_cannot_be_rebuilt_are_zero(self):
        # packed bit fields, SNORM, sRGB, scaled, block compressed, depth: no raw bits to rebuild from the fetch
        for code, name in rdna2spirv.GFX10_FORMATS.items():
            if any(t in name for t in ("5_6_5", "10_10_10_2", "2_10_10_10", "11_11_10", "SNORM", "SRGB", "SCALED", "BC", "8_24")):
                with self.subTest(name=name):
                    self.assertEqual(TABLE[code], 0)


class Hardware(unittest.TestCase):
    """The words tools/hw_oracle measured on gfx1036 for a texel of bytes 81 8c 97 a2 ad b8 c3 ce d9 e4 ef fa 05 10 1b 26."""
    B = bytes.fromhex("818c97a2adb8c3ced9e4effa05101b26")

    def comps(self, fmt):
        ws = [int(t) for t in fmt.split("_")[:-1]]
        out, off = [], 0
        for w in ws:
            v = int.from_bytes(self.B, "little") >> off & ((1 << w) - 1)
            ty = fmt.rsplit("_", 1)[1]
            out.append(v / ((1 << w) - 1) if ty == "UNORM" else v)
            off += w
        return out

    def test_measured_words(self):
        for fmt, dmask, sgn, want in (
                ("8_8_8_8_UINT", 0xF, False, [0xA2978C81, 0, 0, 0]),
                ("8_8_8_8_UNORM", 0x1, False, [0xA2978C81]),
                ("16_16_16_16_UINT", 0xF, False, [0xA2978C81, 0xCEC3B8AD, 0, 0]),
                ("16_16_16_16_UINT", 0x2, False, [0xCEC3B8AD]),
                ("16_16_16_16_UINT", 0xA, False, [0xCEC3B8AD, 0]),
                ("16_16_16_16_UINT", 0x5, False, [0xA2978C81, 0]),
                ("32_32_32_UINT", 0x7, False, [0xA2978C81, 0xCEC3B8AD, 0xFAEFE4D9]),
                ("32_32_32_32_UINT", 0xF, False, [0xA2978C81, 0xCEC3B8AD, 0xFAEFE4D9, 0x261B1005]),
                ("8_UINT", 0x1, False, [0x81]), ("8_UINT", 0x1, True, [0xFFFFFF81]),
                ("8_UNORM", 0x1, True, [0xFFFFFF81]),
                ("8_8_UINT", 0x1, False, [0x8C81]), ("8_8_UINT", 0x1, True, [0xFFFF8C81]),
                ("16_UINT", 0x1, True, [0xFFFF8C81]),
                ("8_8_8_8_UINT", 0x1, True, [0xA2978C81]),          # 4 bytes: no extension
                ("16_16_16_16_UINT", 0x1, True, [0xA2978C81])):
            with self.subTest(fmt=fmt, dmask=dmask, sgn=sgn):
                self.assertEqual(model(fmt, self.comps(fmt), dmask, sgn), want)


class Translation(unittest.TestCase):
    def shader(self, op, dmask=0xF, dim="2D", extra=""):
        mip = "_mip" in op
        addr = {"2D": "v[4:5]", "3D": "v[4:6]", "2D_ARRAY": "v[4:6]", "1D": "v4"}[dim]
        if mip:
            addr = "v[4:6]"
        return translate(["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0",
                          f"{op} v[0:3], {addr}, s[0:7] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_{dim} unorm{extra}",
                          "s_waitcnt vmcnt(0)"])

    def test_every_form_translates_for_the_typed_dimensions(self):
        for op in ("image_load_pck", "image_load_pck_sgn"):
            for dim in ("1D", "2D", "2D_ARRAY", "3D"):
                for dmask in (0x1, 0x3, 0xF):
                    with self.subTest(op=op, dim=dim, dmask=dmask):
                        text = self.shader(op, dmask, dim)
                        self.assertIn("%fmt_pck", text)
                        self.assertIn("OpImageFetch", text)
        for op in ("image_load_mip_pck", "image_load_mip_pck_sgn"):
            self.assertIn("%fmt_pck", self.shader(op, 0x3))

    def test_the_sign_extension_is_only_in_the_sgn_forms(self):
        self.assertNotIn("OpBitFieldSExtract", self.shader("image_load_pck"))
        self.assertIn("OpBitFieldSExtract", self.shader("image_load_pck_sgn"))

    def test_refused_forms(self):
        for op, dim, extra in (("image_load_pck", "2D", " d16"), ("image_load_pck", "2D", " a16"),
                               ("image_load_pck", "2D", " glc"), ("image_load_pck", "2D", " tfe"),
                               ("image_load_mip_pck", "3D", ""),
                               ("image_load_mip_pck_sgn", "2D_ARRAY", "")):
            with self.subTest(op=op, dim=dim, extra=extra), self.assertRaises(rdna2spirv.Unsupported):
                self.shader(op, 0x1, dim, extra)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
    def test_spirv_val_accepts_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            for op in ("image_load_pck", "image_load_pck_sgn", "image_load_mip_pck", "image_load_mip_pck_sgn"):
                for dmask in (0x1, 0x6, 0xF):
                    with self.subTest(op=op, dmask=dmask):
                        sa, sv = os.path.join(tmp, f"{op}{dmask}.spvasm"), os.path.join(tmp, f"{op}{dmask}.spv")
                        with open(sa, "w", encoding="utf-8") as f:
                            f.write(self.shader(op, dmask))
                        result = rdna2spirv.assemble_spirv(sa, sv)
                        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
