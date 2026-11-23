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
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv
import support

MASK32 = 0xFFFFFFFF


def f2u(val: float) -> int:
    """Pack float to 32-bit unsigned integer bit pattern."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u2f(bits: int) -> float:
    """Unpack 32-bit unsigned integer bit pattern to float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


class TestInterpP1F32Reference(unittest.TestCase):
    def test_p1_f32_translation_emission(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_ref", code)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv, inputs={"%in_attr0": (1.5, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(1.5))

    def test_p1_f32_mrt0_execution(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_mrt", code)
        words = support.mrt0_words(spv, inputs={"%in_attr0": (3.25, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(3.25))
        self.assertEqual(words[1], f2u(3.25))


class TestInterpP1F32Barycentric(unittest.TestCase):
    def test_p1_f32_coordinate_registers(self):
        code = [
            "v_mov_b32_e32 v10, v0",
            "v_interp_p1_f32 v4, v10, attr1.y",
            "v_interp_p2_f32 v4, v1, attr1.y",
            "exp mrt0, v4, v4, v4, v4 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1_f32_bary", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv, inputs={"%in_attr1": (0.0, 2.75, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(2.75))


class TestInterpP1F32MultiChannel(unittest.TestCase):
    def test_p1_f32_all_channels(self):
        expected_values = {"x": 1.125, "y": 2.25, "z": 3.375, "w": 4.5}
        for chan, expected in expected_values.items():
            code = [
                f"v_interp_p1_f32 v2, v0, attr2.{chan}",
                f"v_interp_p2_f32 v2, v1, attr2.{chan}",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_p1_f32_ch_{chan}", code)
            words = support.mrt0_words(spv, inputs={"%in_attr2": (1.125, 2.25, 3.375, 4.5)})
            self.assertEqual(words[0], f2u(expected))


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
        words = support.mrt0_words(spv, inputs={"%in_attr3": (7.0, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(7.0))

    def test_p1_f32_wrong_i_coordinate_rejected(self):
        # Passing J coordinate (v1) instead of I coordinate (v0) must be rejected
        code = [
            "v_interp_p1_f32 v5, v1, attr3.x",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_p1_wrong_i_coord", code)


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
        words = support.mrt0_words(spv, inputs={"%in_attr0": (-4.5, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(-4.5))


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
        words = support.mrt0_words(spv, inputs={"%in_attr4": (8.0, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(8.0))
        self.assertEqual(words[1], 42)


class TestInterpP2F32Clamp(unittest.TestCase):
    def test_p2_f32_clamp_modifier_high(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.y",
            "v_interp_p2_f32 v2, v1, attr0.y clamp",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_clamp_hi", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv, inputs={"%in_attr0": (0.0, 2.5, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(1.0))

    def test_p2_f32_clamp_modifier_low(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.y",
            "v_interp_p2_f32 v2, v1, attr0.y clamp",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f32_clamp_lo", code)
        words = support.mrt0_words(spv, inputs={"%in_attr0": (0.0, -1.5, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(0.0))


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
        words = support.mrt0_words(spv, inputs={"%in_attr5": (0.0, 0.0, 16.5, 0.0)})
        self.assertEqual(words[0], f2u(16.5))

    def test_p2_f32_wrong_j_coordinate_rejected(self):
        # Passing I coordinate (v0) instead of J coordinate (v1) must be rejected
        code = [
            "v_interp_p1_f32 v3, v0, attr5.z",
            "v_interp_p2_f32 v3, v0, attr5.z",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_p2_wrong_j_coord", code)


class TestInterpMovF32Reference(unittest.TestCase):
    def test_mov_f32_reference_translation(self):
        code = [
            "v_interp_mov_f32 v2, p0, attr0.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mov_f32_ref", code)
        self.assertIn("OpCapability Shader", spv)
        words = support.mrt0_words(spv, inputs={"%in_attr0": (42.0, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(42.0))


class TestInterpMovF32VertexSel(unittest.TestCase):
    def test_mov_f32_vertex_params(self):
        code = [
            "v_interp_mov_f32 v2, p0, attr1.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mov_f32_p0", code)
        words = support.mrt0_words(spv, inputs={"%in_attr1": (99.0, 0.0, 0.0, 0.0)})
        self.assertEqual(words[0], f2u(99.0))

    def test_mov_f32_invalid_vsel_rejected(self):
        code = [
            "v_interp_mov_f32 v2, p3, attr1.x",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mov_bad_vsel", code)


class TestInterpMovF32Flat(unittest.TestCase):
    def test_mov_f32_flat_channels(self):
        test_inputs = {"x": 11.0, "y": 22.0, "z": 33.0, "w": 44.0}
        for chan, expected in test_inputs.items():
            code = [
                f"v_interp_mov_f32 v2, p0, attr2.{chan}",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_mov_f32_flat_{chan}", code)
            self.assertIn("Flat", spv)
            words = support.mrt0_words(spv, inputs={"%in_attr2": (11.0, 22.0, 33.0, 44.0)})
            self.assertEqual(words[0], f2u(expected))


class TestInterpMovF32Translator(unittest.TestCase):
    def test_mov_f32_translator_pipeline(self):
        code = [
            "v_interp_mov_f32 v8, p0, attr6.y",
            "exp mrt0, v8, v8, v8, v8 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mov_f32_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        words = support.mrt0_words(spv, inputs={"%in_attr6": (0.0, 123.456, 0.0, 0.0)})
        self.assertAlmostEqual(u2f(words[0]), 123.456, places=3)


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
        v0 = [0x3C00, 0, 0, 0]
        inputs = support.varyings(spv, {0: [v0, v0, v0]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x3C00)


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
        v1 = [0x4000, 0, 0, 0]
        inputs = support.varyings(spv, {1: [v1, v1, v1]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4000)


class TestInterpP1LLF16High(unittest.TestCase):
    def test_p1ll_f16_high_modifier(self):
        code = [
            "v_interp_p1ll_f16 v4, v0, attr2.x high",
            "v_interp_p2_f16 v4, v1, attr2.x, v4 high",
            "exp mrt0, v4, v4, v4, v4 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p1ll_high", code)
        v2 = [(0x4200 << 16) | 0x3C00, 0, 0, 0]
        inputs = support.varyings(spv, {2: [v2, v2, v2]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4200)


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
        v7 = [0, 0x4500, 0, 0]
        inputs = support.varyings(spv, {7: [v7, v7, v7]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4500)


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
        v0 = [0x3800, 0, 0, 0]
        inputs = support.varyings(spv, {0: [v0, v0, v0]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x3800)


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
        v3 = [0x3C00, 0, 0, 0]
        inputs = support.varyings(spv, {3: [v3, v3, v3]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x3C00)


class TestInterpP1LVF16Channels(unittest.TestCase):
    def test_p1lv_f16_all_channels(self):
        expected_f16 = {"x": 0x3C00, "y": 0x4000}
        for chan, expected in expected_f16.items():
            code = [
                f"v_interp_p1lv_f16 v2, v0, attr4.{chan}",
                f"v_interp_p2_f16 v2, v1, attr4.{chan}, v2",
                "exp mrt0, v2, v2, v2, v2 done",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_p1lv_ch_{chan}", code)
            v4 = [0x3C00, 0x4000, 0, 0]
            inputs = support.varyings(spv, {4: [v4, v4, v4]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            words = support.mrt0_words(spv, inputs=inputs)
            self.assertEqual(words[0] & 0xFFFF, expected)


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
        v8 = [0, 0, 0x4400, 0]
        inputs = support.varyings(spv, {8: [v8, v8, v8]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4400)


class TestInterpP2F16Reference(unittest.TestCase):
    def test_p2_f16_reference_translation(self):
        code = [
            "v_mov_b32_e32 v3, 0xdead0000",
            "v_interp_p1ll_f16 v2, v0, attr0.x",
            "v_interp_p2_f16 v3, v1, attr0.x, v2",
            "exp mrt0, v3, v3, v3, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_ref", code)
        self.assertIn("OpCapability Shader", spv)
        v0 = [0x3C00, 0, 0, 0]
        inputs = support.varyings(spv, {0: [v0, v0, v0]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF0000, 0xDEAD0000)
        self.assertEqual(words[0] & 0xFFFF, 0x3C00)


class TestInterpP2F16Accumulator(unittest.TestCase):
    def test_p2_f16_accumulator_pairing(self):
        code = [
            "v_interp_p1ll_f16 v7, v0, attr5.y",
            "v_interp_p2_f16 v7, v1, attr5.y, v7",
            "exp mrt0, v7, v7, v7, v7 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_accum", code)
        v5 = [0, 0x4800, 0, 0]
        inputs = support.varyings(spv, {5: [v5, v5, v5]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4800)


class TestInterpP2F16HighHalf(unittest.TestCase):
    def test_p2_f16_clamp_modifier(self):
        code = [
            "v_interp_p1ll_f16 v9, v0, attr6.x",
            "v_interp_p2_f16 v9, v1, attr6.x, v9 clamp",
            "exp mrt0, v9, v9, v9, v9 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_clamp", code)
        v6 = [0x4200, 0, 0, 0]
        inputs = support.varyings(spv, {6: [v6, v6, v6]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x3C00)


class TestInterpP2F16Translator(unittest.TestCase):
    def test_p2_f16_translator_pipeline(self):
        code = [
            "v_interp_p1ll_f16 v16, v0, attr9.w",
            "v_interp_p2_f16 v16, v1, attr9.w, v16",
            "exp mrt0, v16, v16, v16, v16 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_p2_f16_trans", code)
        self.assertIn("OpEntryPoint Fragment", spv)
        v9 = [0, 0, 0, 0x4900]
        inputs = support.varyings(spv, {9: [v9, v9, v9]}, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
        words = support.mrt0_words(spv, inputs=inputs)
        self.assertEqual(words[0] & 0xFFFF, 0x4900)

    def test_interp_rejected_in_non_ps_stage(self):
        code = [
            "v_interp_p1_f32 v2, v0, attr0.x",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("vs_bad_interp_stage", code)


if __name__ == "__main__":
    unittest.main()
