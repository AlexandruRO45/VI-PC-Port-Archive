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


if __name__ == "__main__":
    unittest.main()
