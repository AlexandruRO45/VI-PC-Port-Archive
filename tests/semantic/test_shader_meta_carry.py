"""Carry-dependent SGPR words must not become constant build-time metadata."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import shader_ir  # noqa: E402
import shader_meta  # noqa: E402


def translate_meta(lines):
    asm = support.translate_lines("ps_meta_carry", lines)
    meta = support.rdna2spirv.shader_meta(support.rdna2spirv.translate.last_gen, asm)
    return asm, meta


def analyse(blocks, succ):
    return shader_meta.analyse_instructions(
        [[shader_ir.parse_line(line) for line in block] for block in blocks], succ)


class CarryMetadataTests(unittest.TestCase):
    def test_pointer_high_word_is_not_assumed_zero(self):
        for op, carry_word in (("s_addc_u32", 1), ("s_subb_u32", 0xFFFFFFFF)):
            for carry in (0, 1):
                for src in ("0", "s3"):
                    with self.subTest(op=op, carry=carry, src=src):
                        asm, meta = translate_meta([
                            "s_mov_b32 s3, 0",
                            f"s_cmp_eq_u32 0, {0 if carry else 1}",
                            f"{op} s3, {src}, 0",
                            "s_load_dwordx4 s[8:11], s[2:3], 0",
                            "v_mov_b32 v0, s3",
                            "exp mrt0 v0, v0, v0, v0 done vm",
                            "s_endpgm",
                        ])
                        self.assertEqual(support.mrt0_words(asm)[0], carry_word if carry else 0)
                        self.assertEqual(meta["user_sgpr_reads"], [{"sgpr": 2, "kind": "ptr64_lo"}])
                        self.assertEqual(meta["descriptor_access"], {
                            "table_loads": True, "truncates_table_ptr_high": False,
                        })

    def test_carry_word_is_not_published_as_literal_sampler(self):
        for op in ("s_addc_u32", "s_subb_u32"):
            with self.subTest(op=op):
                _, meta = translate_meta([
                    "s_mov_b32 s12, 0x92",
                    "s_mov_b32 s13, 0x00c00000",
                    "s_mov_b32 s14, 0x14500000",
                    "s_mov_b32 s15, 0",
                    "s_cmp_eq_u32 0, 0",
                    f"{op} s15, s15, 0",
                    "v_mov_b32 v0, 0.5",
                    "v_mov_b32 v1, 0.5",
                    "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
                    "exp mrt0 v4, v5, v6, v7 done vm",
                    "s_endpgm",
                ])
                self.assertNotIn("literal_samplers", meta)

    def test_carry_on_one_path_prevents_zero_at_join(self):
        for op in ("s_addc_u32", "s_subb_u32"):
            with self.subTest(op=op):
                meta = analyse([
                    ["s_mov_b32 s3, 0"],
                    ["s_cmp_eq_u32 0, 0", f"{op} s3, s3, 0"],
                    ["s_mov_b32 s3, 0"],
                    ["s_load_dwordx4 s[8:11], s[2:3], 0", "s_endpgm"],
                ], [[1, 2], [3], [3], []])
                self.assertFalse(meta["truncates_high"])
                self.assertEqual(meta["user_sgpr_reads"], [{"sgpr": 2, "kind": "ptr64_lo"}])

    def test_explicit_zero_still_marks_truncated_pointer(self):
        meta = analyse([[
            "s_mov_b32 s3, 0",
            "s_load_dwordx4 s[8:11], s[2:3], 0",
            "s_endpgm",
        ]], [[]])
        self.assertTrue(meta["truncates_high"])
        self.assertEqual(meta["user_sgpr_reads"], [{"sgpr": 2, "kind": "ptr32_lo"}])

    def test_explicit_constant_after_carry_restores_literal_sampler(self):
        meta = analyse([[
            "s_mov_b32 s12, 0x92",
            "s_mov_b32 s13, 0x00c00000",
            "s_mov_b32 s14, 0x14500000",
            "s_cmp_eq_u32 0, 0",
            "s_addc_u32 s15, 0, 0",
            "s_mov_b32 s15, 0",
            "image_sample v[4:7], v[0:1], s[16:23], s[12:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_endpgm",
        ]], [[]])
        self.assertEqual(meta["literal_samplers"], [[0x92, 0x00C00000, 0x14500000, 0]])

    def test_user_high_word_keeps_its_origin(self):
        for op in ("s_addc_u32", "s_subb_u32"):
            with self.subTest(op=op):
                meta = analyse([[
                    "s_mov_b32 s4, s0",
                    f"{op} s5, s1, 0",
                    "s_load_dwordx4 s[8:11], s[4:5], 0",
                    "s_endpgm",
                ]], [[]])
                self.assertEqual(meta["user_sgpr_reads"], [
                    {"sgpr": 0, "kind": "ptr64_lo", "pair": [0, 1]},
                    {"sgpr": 1, "kind": "ptr64_hi", "pair": [0, 1]},
                ])
                self.assertFalse(meta["truncates_high"])


if __name__ == "__main__":
    unittest.main()
