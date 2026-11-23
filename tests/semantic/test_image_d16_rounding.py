"""SHADER-308: a d16 image result is narrowed by the typed-buffer d16 rule (docs/design.md, "d16 and f16 format
narrowing"): a UNORM/SNORM format takes the exact quotient rounded once, a 32-bit float channel is narrowed toward
zero, and every other float format rounds to nearest even.

The image path reuses `Gen.d16_norm_f16`, whose values are checked on the CPU through the buffer path
(test_buffer_load_format.py `D16NormalizedRounding`, test_buffer_f16_narrowing_hw.py). What the image path adds is
the format class read from the T# at run time (`Gen.fmt_float_flags` over `Gen.float_format_masks`) and the wiring of
every d16 component through that helper; those are checked here, for the reads that go through
`Gen.image_read_typed` (2D) and for the 2D_ARRAY and CUBE `image_sample_lz` and `image_gather4_lz`, which read the
float view directly and narrow through `Gen.d16_fmt_pair`. spirv_mini_interp has no image instructions, so the image
values themselves are the GPU measurement of tools/d16_image_probe.py (`--dims 2d_array,cube` for the second group)."""
import os
import re
import shutil
import tempfile
import unittest

import support
import rdna2spirv

PRE = ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0"]
EPI = ["s_waitcnt vmcnt(0)", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# (instruction, number of f16 halves it narrows)
D16_READS = [
    ("image_load v[0:1], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16", 4),
    ("image_load v[0:1], v[4:5], s[0:7] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16", 3),
    ("image_load v0, v[4:5], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm d16", 1),
    ("image_sample_lz v[0:1], v[4:5], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D d16", 4),
    ("image_gather4_lz v[0:1], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D d16", 4),
]
# The 2D_ARRAY and CUBE forms the review on #4318 found rounding to nearest even (d16_fmt_pair).
D16_ARRAY_READS = [
    ("image_sample_lz v[0:1], v[4:6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY d16", 4),
    ("image_sample_lz v0, v[4:6], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY d16", 2),
    ("image_sample_lz v[0:1], v[4:6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_CUBE d16", 4),
    ("image_gather4_lz v[0:1], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY d16", 4),
    ("image_gather4_lz v[0:1], v[4:6], s[0:7], s[8:11] dmask:0x8 dim:SQ_RSRC_IMG_CUBE d16", 4),
]
F32_READ = "image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"
# Not d16, or another op on these dimensions: no format tables.
NO_TABLES = [
    F32_READ,
    "image_sample_lz v[0:3], v[4:6], s[0:7], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY",
    "image_gather4_lz v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE",
]


def translate(line):
    return support.translate_lines("ps_d16_image_rounding", PRE + [line] + EPI)


def code(name):
    """GFX10 T# FORMAT code of GFX10_FORMAT_<name>."""
    return next(k for k, n in rdna2spirv.GFX10_FORMATS.items() if n == "GFX10_FORMAT_" + name)


def classes(name):
    """(in the UNORM/SNORM table, in the 32-bit-float table) for a format name."""
    norm, f32 = rdna2spirv.Gen.float_format_masks()
    k = code(name)
    return (norm[k >> 5] >> (k & 31)) & 1, (f32[k >> 5] >> (k & 31)) & 1


class FormatTables(unittest.TestCase):
    def test_tables_cover_the_9_bit_format_field(self):
        norm, f32 = rdna2spirv.Gen.float_format_masks()
        self.assertEqual((len(norm), len(f32)), (16, 16))   # 16 x 32 bits = the 512 codes of FORMAT [28:20]
        self.assertFalse(any(a & b for a, b in zip(norm, f32)), "a format is in both tables")

    def test_normalized_formats_take_the_single_rounding(self):
        for name in ("8_UNORM", "8_SNORM", "16_UNORM", "16_SNORM", "16_16_UNORM", "8_8_8_8_UNORM", "10_10_10_2_UNORM"):
            with self.subTest(name=name):
                self.assertEqual(classes(name), (1, 0))

    def test_32_bit_float_channels_are_narrowed_toward_zero(self):
        for name in ("32_FLOAT", "32_32_FLOAT", "32_32_32_FLOAT", "32_32_32_32_FLOAT"):
            with self.subTest(name=name):
                self.assertEqual(classes(name), (0, 1))

    def test_other_formats_keep_nearest_even(self):
        for name in ("16_FLOAT", "16_16_16_16_FLOAT", "10_11_11_FLOAT", "8_UINT", "16_SINT", "32_UINT"):
            with self.subTest(name=name):
                self.assertEqual(classes(name), (0, 0))


class Emitted(unittest.TestCase):
    def check_wired(self, line, halves):
        lo, hi = (f"%c{v:x}" for v in sorted(rdna2spirv.Gen.D16_NORM_HALFWAY))
        text = translate(line)
        for table in ("%fmt_norm_mask", "%fmt_f32_mask"):
            self.assertRegex(text, rf"{table} = OpVariable ")
            self.assertRegex(text, rf"OpAccessChain %_ptr_Private_uint {table} ")
        # d16_norm_f16 compares each half's f32 with both halfway quotients once
        for c in (lo, hi):
            self.assertEqual(len(re.findall(rf"= OpIEqual %bool %\w+ {c}\b", text)), halves, c)
        self.assertEqual(text.count("%_ptr_Private_uint = OpTypePointer"), 1)

    def test_every_d16_component_goes_through_the_format_class(self):
        for line, halves in D16_READS:
            with self.subTest(line=line):
                self.check_wired(line, halves)

    def test_array_and_cube_sample_and_gather_go_through_the_format_class(self):
        for line, halves in D16_ARRAY_READS:
            with self.subTest(line=line):
                self.check_wired(line, halves)

    def test_reads_without_d16_need_no_float_tables(self):
        for line in NO_TABLES:
            with self.subTest(line=line):
                text = translate(line)
                self.assertNotIn("%fmt_norm_mask", text)
                self.assertNotIn("%fmt_f32_mask", text)


@unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val")
class Validation(unittest.TestCase):
    def test_spirv_val_accepts_the_d16_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, (line, _) in enumerate(D16_READS + D16_ARRAY_READS):
                with self.subTest(line=line):
                    sa, sv = os.path.join(tmp, f"{i}.spvasm"), os.path.join(tmp, f"{i}.spv")
                    with open(sa, "w", encoding="utf-8") as f:
                        f.write(translate(line))
                    result = rdna2spirv.assemble_spirv(sa, sv)
                    self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
