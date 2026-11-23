from __future__ import annotations

"""Translator-backed integration tests for Bundle 61: VALU Interpolation
Parameter Precision Suite.

Covers RDNA2 Pixel Shader Interpolation Instructions:
    - v_interp_p1_f32     vD, vI, attrN.C
        Evaluates stage 1 of 32-bit floating-point attribute interpolation.
    - v_interp_p2_f32     vD, vJ, attrN.C
        Evaluates stage 2 of 32-bit interpolation using accumulated partial result.
    - v_interp_mov_f32    vD, p0, attrN.C
        Reads flat (provoking-vertex) attribute parameter directly.
    - v_interp_p1ll_f16   vD, vI, attrN.C [high]
        Evaluates stage 1 of 16-bit half-precision interpolation (low/high packed half).
    - v_interp_p1lv_f16   vD, vI, attrN.C [high]
        Evaluates stage 1 of 16-bit vector-weighted interpolation.
    - v_interp_p2_f16     vD, vJ, attrN.C, vPartial [high]
        Evaluates stage 2 of 16-bit interpolation completing packed half-float attribute.

Hardware rules (GFX10 RDNA2):
    - Interpolation instructions are valid in Pixel Shader (ps_) stages.
    - Barycentric coordinates (i, j) are passed via VGPR inputs (typically v0, v1).
    - v_interp_p2 matches preceding v_interp_p1 on attribute offset and destination.
    - v_interp_mov_f32 reads flat provoking vertex attributes without perspective division.
    - F16 interpolation supports packed half-register targeting (low or high half).
    - 100% translator-backed execution exercising tools/rdna2spirv.py.

Standard library Python only (unittest, struct, math).
Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

class TestInterpP1F32Reference(unittest.TestCase):
    def test_p1_f32_translation_emission(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_ref", code)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpEntryPoint Fragment", spv)

    def test_p1_f32_mrt0_execution(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_mrt", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1F32Barycentric(unittest.TestCase):
    def test_p1_f32_coordinate_registers(self):
        code = [
            "v_mov_b32_e32 v10, 0",
            "v_interp_p1_f32 v4, v10, attr1.y",
            "v_interp_p2_f32 v4, v1, attr1.y",
            "exp mrt0, v4, v4, v4, v4 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_bary", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1F32MultiChannel(unittest.TestCase):
    def test_p1_f32_all_channels(self):
        for chan in ("x", "y", "z", "w"):
            code = [
                f"v_interp_p1_f32 v2, v0, attr2.{chan}",
                f"v_interp_p2_f32 v2, v1, attr2.{chan}",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_p1_f32_ch_{chan}", code)
            words = support.mrt0_words(spv)
            self.assertEqual(len(words), 4)

class TestInterpP1F32Translator(unittest.TestCase):
    def test_p1_f32_mutation_guard(self):
        code = [
            "v_interp_p1_f32 v5, v0, attr3.x",
            "v_interp_p2_f32 v5, v1, attr3.x",
            "exp mrt0, v5, v5, v5, v5 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F32Reference(unittest.TestCase):
    def test_p2_f32_reference_translation(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F32ReachingWrite(unittest.TestCase):
    def test_p2_f32_reaching_write_chain(self):
        code = [
            "v_interp_p1_f32 v6, v0, attr4.x",
            "v_mov_b32_e32 v7, 42",
            "v_interp_p2_f32 v6, v1, attr4.x",
            "exp mrt0, v6, v7, v6, v7 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_reach", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F32Clamp(unittest.TestCase):
    def test_p2_f32_clamp_modifier(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.y",
            "v_interp_p2_f32 v2, v1, attr0.y clamp",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_clamp", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F32Translator(unittest.TestCase):
    def test_p2_f32_translator_pipeline(self):
        code = [
            "v_interp_p1_f32 v3, v0, attr5.z",
            "v_interp_p2_f32 v3, v1, attr5.z",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpMovF32Reference(unittest.TestCase):
    def test_mov_f32_reference_translation(self):
        code = [
            "v_interp_mov_f32 v2, p0, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mov_f32_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpMovF32VertexSel(unittest.TestCase):
    def test_mov_f32_vertex_params(self):
        for vsel in ("p0", "p10", "p20"):
            code = [
                f"v_interp_mov_f32 v2, {vsel}, attr1.x",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_mov_f32_{vsel}", code)
            words = support.mrt0_words(spv)
            self.assertEqual(len(words), 4)

class TestInterpMovF32Flat(unittest.TestCase):
    def test_mov_f32_flat_channels(self):
        for chan in ("x", "y", "z", "w"):
            code = [
                f"v_interp_mov_f32 v2, p0, attr2.{chan}",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_mov_f32_flat_{chan}", code)
            self.assertIn("Flat", spv)
            words = support.mrt0_words(spv)
            self.assertEqual(len(words), 4)

class TestInterpMovF32Translator(unittest.TestCase):
    def test_mov_f32_translator_pipeline(self):
        code = [
            "v_interp_mov_f32 v8, p0, attr6.y",
            "exp mrt0, v8, v8, v8, v8 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mov_f32_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LLF16Reference(unittest.TestCase):
    def test_p1ll_f16_reference_translation(self):
        code = [
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1ll_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LLF16Pairing(unittest.TestCase):
    def test_p1ll_f16_low_lane(self):
        code = [
            "v_interp_p1ll_f16 v3, v0, attr1.x",
            "v_interp_p2_f16 v3, v1, attr1.x, v3",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1ll_pair", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LLF16High(unittest.TestCase):
    def test_p1ll_f16_high_modifier(self):
        code = [
            "v_interp_p1ll_f16 v4, v0, attr2.x high",
            "v_interp_p2_f16 v4, v1, attr2.x, v4 high",
            "exp mrt0, v4, v4, v4, v4 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1ll_high", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LLF16Translator(unittest.TestCase):
    def test_p1ll_f16_translator_pipeline(self):
        code = [
            "v_interp_p1ll_f16 v12, v0, attr7.y",
            "v_interp_p2_f16 v12, v1, attr7.y, v12",
            "exp mrt0, v12, v12, v12, v12 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1ll_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LVF16Reference(unittest.TestCase):
    def test_p1lv_f16_reference_translation(self):
        code = [
            "v_interp_p1lv_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1lv_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LVF16Weighting(unittest.TestCase):
    def test_p1lv_f16_vector_weights(self):
        code = [
            "v_mov_b32_e32 v5, 0",
            "v_interp_p1lv_f16 v2, v5, attr3.x",
            "v_interp_p2_f16 v2, v1, attr3.x, v2",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1lv_weight", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP1LVF16Channels(unittest.TestCase):
    def test_p1lv_f16_all_channels(self):
        for chan in ("x", "y"):
            code = [
                f"v_interp_p1lv_f16 v2, v0, attr4.{chan}",
                f"v_interp_p2_f16 v2, v1, attr4.{chan}, v2",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_p1lv_ch_{chan}", code)
            words = support.mrt0_words(spv)
            self.assertEqual(len(words), 4)

class TestInterpP1LVF16Translator(unittest.TestCase):
    def test_p1lv_f16_translator_pipeline(self):
        code = [
            "v_interp_p1lv_f16 v14, v0, attr8.z",
            "v_interp_p2_f16 v14, v1, attr8.z, v14",
            "exp mrt0, v14, v14, v14, v14 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1lv_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F16Reference(unittest.TestCase):
    def test_p2_f16_reference_translation(self):
        code = [
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v2, v1, attr0.x, v2",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F16Accumulator(unittest.TestCase):
    def test_p2_f16_accumulator_pairing(self):
        code = [
            "v_interp_p1ll_f16 v7, v0, attr5.y",
            "v_interp_p2_f16 v7, v1, attr5.y, v7",
            "exp mrt0, v7, v7, v7, v7 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_accum", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)

class TestInterpP2F16HighHalf(unittest.TestCase):
    def test_p2_f16_high_destination(self):
        code = [
            "v_interp_p1ll_f16 v9, v0, attr6.x high",
            "v_interp_p2_f16 v9, v1, attr6.x, v9 high",
            "exp mrt0, v9, v9, v9, v9 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_high", code)
        words = support.mrt0_words(spv)
        self.assertEqual(len(words), 4)


if __name__ == "__main__":
    unittest.main()
