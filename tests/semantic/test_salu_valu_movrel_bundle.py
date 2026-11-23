from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
from support import rdna2spirv

class TestSMovrelsB32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        # Seed s20..s22 with sentinels and test m0 = 0, 1, 2
        cases = [
            (0, 0x11112222),
            (1, 0x33334444),
            (2, 0x55556666),
        ]
        for m0, expected in cases:
            with self.subTest(m0=m0):
                lines = [
                    "s_mov_b32 s20, 0x11112222",
                    "s_mov_b32 s21, 0x33334444",
                    "s_mov_b32 s22, 0x55556666",
                    f"s_mov_b32 m0, {m0}",
                    "s_movrels_b32 s10, s20",
                    "v_mov_b32 v0, s10",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_smovrels_b32_oracle_{m0}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], expected)

    def test_oracle_zero_offset(self):
        lines = [
            "s_mov_b32 s15, 0x77778888",
            "s_mov_b32 m0, 0",
            "s_movrels_b32 s12, s15",
            "v_mov_b32 v0, s12",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b32_zero", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x77778888)

    def test_oracle_boundary_offset(self):
        # Index wraps modulo 128 (120 + 13 = 133 % 128 = 5)
        lines = [
            "s_mov_b32 s5, 0x9999AAAA",
            "s_mov_b32 m0, 13",
            "s_movrels_b32 s10, s120",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b32_boundary", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x9999AAAA)


class TestSMovrelsB32AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        lines = [
            "s_mov_b32 s10, 0x12345678",
            "s_mov_b32 m0, 0",
            "s_movrels_b32 s11, s10",
            "v_mov_b32 v0, s11",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b32_ast", lines)
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr", spv)
        self.assertIn("OpLoad %uint %m0", spv)

    def test_ast_execution_semantics(self):
        lines = [
            "s_mov_b32 s30, 0xABCDEF01",
            "s_mov_b32 m0, 2",
            "s_mov_b32 s32, 0xCAFEBABE",
            "s_movrels_b32 s10, s30",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b32_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xCAFEBABE)


class TestSMovrelsB32Indexing(unittest.TestCase):
    def test_dynamic_indexing_sweep(self):
        for offset in range(4):
            with self.subTest(offset=offset):
                val = 0x10101010 * (offset + 1)
                lines = [
                    f"s_mov_b32 s{40 + offset}, {val}",
                    f"s_mov_b32 m0, {offset}",
                    "s_movrels_b32 s50, s40",
                    "v_mov_b32 v0, s50",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_smovrels_sweep_{offset}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], val)

    def test_destination_overlap_source(self):
        # Read happens before destination write even when destination is base register
        lines = [
            "s_mov_b32 s20, 0x11111111",
            "s_mov_b32 s22, 0x33333333",
            "s_mov_b32 m0, 2",
            "s_movrels_b32 s20, s20",
            "v_mov_b32 v0, s20",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_overlap", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x33333333)

    def test_scc_preserved(self):
        # s_movrels_b32 must not overwrite SCC
        lines = [
            "s_cmp_eq_u32 5, 5",  # SCC = 1
            "s_mov_b32 s10, 0x44444444",
            "s_mov_b32 m0, 0",
            "s_movrels_b32 s12, s10",
            "s_cselect_b32 s13, 0x1111, 0x2222",  # Should pick 0x1111 because SCC == 1
            "v_mov_b32 v0, s13",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_scc", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x1111)


class TestSMovrelsB32Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        lines = [
            "s_mov_b32 s16, 0x55555555",
            "s_mov_b32 m0, 0",
            "s_movrels_b32 s18, s16",
            "v_mov_b32 v0, s18",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_full", lines)
        self.assertIn("OpCapability Shader", spv)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x55555555)

    def test_negative_rejects_vgpr_operands(self):
        lines = [
            "s_movrels_b32 v0, s10",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_smovrels_vgpr_dst", lines)

    def test_negative_rejects_literal_operands(self):
        lines = [
            "s_movrels_b32 s10, 42",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_smovrels_lit_src", lines)


class TestSMovrelsB64Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        cases = [
            (0, (0x11223344, 0x55667788)),
            (2, (0x99AABBCC, 0xDDEEFF00)),
        ]
        for m0, (exp0, exp1) in cases:
            with self.subTest(m0=m0):
                lines = [
                    "s_mov_b32 s20, 0x11223344",
                    "s_mov_b32 s21, 0x55667788",
                    "s_mov_b32 s22, 0x99AABBCC",
                    "s_mov_b32 s23, 0xDDEEFF00",
                    f"s_mov_b32 m0, {m0}",
                    "s_movrels_b64 s[10:11], s[20:21]",
                    "v_mov_b32 v0, s10",
                    "v_mov_b32 v1, s11",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_smovrels_b64_oracle_{m0}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], exp0)
                self.assertEqual(out[1], exp1)

    def test_oracle_consecutive_words(self):
        lines = [
            "s_mov_b32 s30, 0xDEADBEEF",
            "s_mov_b32 s31, 0xFEEDFACE",
            "s_mov_b32 m0, 0",
            "s_movrels_b64 s[12:13], s[30:31]",
            "v_mov_b32 v0, s12",
            "v_mov_b32 v1, s13",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_consec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xDEADBEEF)
        self.assertEqual(out[1], 0xFEEDFACE)


class TestSMovrelsB64AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        lines = [
            "s_mov_b32 s20, 0x01020304",
            "s_mov_b32 s21, 0x05060708",
            "s_mov_b32 m0, 0",
            "s_movrels_b64 s[10:11], s[20:21]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s11",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_ast", lines)
        self.assertIn("OpAccessChain %_ptr_Function_uint %sgpr", spv)

    def test_ast_execution_semantics(self):
        lines = [
            "s_mov_b32 s22, 0x77889900",
            "s_mov_b32 s23, 0xAABBCCDD",
            "s_mov_b32 m0, 2",
            "s_movrels_b64 s[14:15], s[20:21]",
            "v_mov_b32 v0, s14",
            "v_mov_b32 v1, s15",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x77889900)
        self.assertEqual(out[1], 0xAABBCCDD)


class TestSMovrelsB64Indexing(unittest.TestCase):
    def test_dynamic_pair_indexing(self):
        lines = [
            "s_mov_b32 s24, 0x12344321",
            "s_mov_b32 s25, 0x56788765",
            "s_mov_b32 m0, 4",
            "s_movrels_b64 s[16:17], s[20:21]",
            "v_mov_b32 v0, s16",
            "v_mov_b32 v1, s17",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_idx", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x12344321)
        self.assertEqual(out[1], 0x56788765)

    def test_scc_preserved_b64(self):
        lines = [
            "s_cmp_eq_u32 7, 7",  # SCC = 1
            "s_mov_b32 s20, 0x11111111",
            "s_mov_b32 s21, 0x22222222",
            "s_mov_b32 m0, 0",
            "s_movrels_b64 s[10:11], s[20:21]",
            "s_cselect_b32 s12, 0x9999, 0x8888",
            "v_mov_b32 v0, s12",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_scc", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x9999)


class TestSMovrelsB64Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        lines = [
            "s_mov_b32 s20, 0xAAAA5555",
            "s_mov_b32 s21, 0x5555AAAA",
            "s_mov_b32 m0, 0",
            "s_movrels_b64 s[10:11], s[20:21]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s11",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_smovrels_b64_full", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xAAAA5555)
        self.assertEqual(out[1], 0x5555AAAA)

    def test_negative_rejects_single_sgpr_operands(self):
        lines = [
            "s_movrels_b64 s10, s[20:21]",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_smovrels_b64_single_dst", lines)

    def test_negative_rejects_single_source(self):
        lines = [
            "s_movrels_b64 s[10:11], s20",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_smovrels_b64_single_src", lines)


class TestVMovreldB32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        for m0 in (0, 3):
            with self.subTest(m0=m0):
                lines = [
                    "v_mov_b32 v5, 0x33445566",
                    f"s_mov_b32 m0, {m0}",
                    "v_movreld_b32 v2, v5",
                    f"v_mov_b32 v0, v{2 + m0}",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovreld_oracle_{m0}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], 0x33445566)

    def test_oracle_zero_offset(self):
        lines = [
            "v_mov_b32 v4, 0x778899AA",
            "s_mov_b32 m0, 0",
            "v_movreld_b32 v1, v4",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovreld_zero", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x778899AA)


class TestVMovreldB32AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        lines = [
            "v_mov_b32 v2, 0x11111111",
            "s_mov_b32 m0, 1",
            "v_movreld_b32 v3, v2",
            "v_mov_b32 v0, v4",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovreld_ast", lines)
        self.assertIn("OpAccessChain %_ptr_Function_uint %vgpr", spv)
        self.assertIn("OpStore", spv)

    def test_ast_execution_semantics(self):
        lines = [
            "v_mov_b32 v8, 0xCAFE1234",
            "s_mov_b32 m0, 5",
            "v_movreld_b32 v1, v8",
            "v_mov_b32 v0, v6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovreld_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xCAFE1234)


class TestVMovreldB32Indexing(unittest.TestCase):
    def test_unwritten_registers_preserved(self):
        # When writing v[2 + 1] = v3, v2 must retain its original sentinel
        lines = [
            "v_mov_b32 v2, 0xAAAAAAAA",
            "v_mov_b32 v3, 0xBBBBBBBB",
            "s_mov_b32 m0, 1",
            "v_movreld_b32 v2, v3",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovreld_unwritten", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xAAAAAAAA)

    def test_dynamic_destination_sweep(self):
        for offset in (1, 2, 4):
            with self.subTest(offset=offset):
                val = 0x20202020 * offset
                lines = [
                    f"v_mov_b32 v1, {val}",
                    f"s_mov_b32 m0, {offset}",
                    "v_movreld_b32 v10, v1",
                    f"v_mov_b32 v0, v{10 + offset}",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovreld_sweep_{offset}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], val)


class TestVMovreldB32Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        lines = [
            "v_mov_b32 v5, 0x55667788",
            "s_mov_b32 m0, 2",
            "v_movreld_b32 v2, v5",
            "v_mov_b32 v0, v4",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovreld_full", lines)
        self.assertIn("OpCapability Shader", spv)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x55667788)

    def test_negative_rejects_sgpr_destination(self):
        lines = [
            "v_movreld_b32 s0, v1",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_vmovreld_sgpr_dst", lines)

    def test_negative_rejects_sgpr_source(self):
        lines = [
            "v_movreld_b32 v0, s1",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_vmovreld_sgpr_src", lines)


class TestVMovrelsB32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        cases = [
            (0, 0x44556677),
            (2, 0x8899AABB),
        ]
        for m0, expected in cases:
            with self.subTest(m0=m0):
                lines = [
                    "v_mov_b32 v4, 0x44556677",
                    "v_mov_b32 v6, 0x8899AABB",
                    f"s_mov_b32 m0, {m0}",
                    "v_movrels_b32 v1, v4",
                    "v_mov_b32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovrels_oracle_{m0}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], expected)

    def test_oracle_base_read(self):
        lines = [
            "v_mov_b32 v7, 0x12344321",
            "s_mov_b32 m0, 0",
            "v_movrels_b32 v1, v7",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrels_base", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x12344321)


class TestVMovrelsB32AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        lines = [
            "v_mov_b32 v3, 0x99887766",
            "s_mov_b32 m0, 1",
            "v_movrels_b32 v1, v2",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrels_ast", lines)
        self.assertIn("OpAccessChain %_ptr_Function_uint %vgpr", spv)

    def test_ast_execution_semantics(self):
        lines = [
            "v_mov_b32 v5, 0xFACEB00C",
            "s_mov_b32 m0, 3",
            "v_movrels_b32 v1, v2",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrels_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xFACEB00C)


class TestVMovrelsB32Indexing(unittest.TestCase):
    def test_dynamic_source_sweep(self):
        for offset in (0, 1, 3):
            with self.subTest(offset=offset):
                val = 0x30303030 * (offset + 1)
                lines = [
                    f"v_mov_b32 v{10 + offset}, {val}",
                    f"s_mov_b32 m0, {offset}",
                    "v_movrels_b32 v1, v10",
                    "v_mov_b32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovrels_sweep_{offset}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], val)

    def test_aliasing_same_register(self):
        # Reading from v[1 + 2] = v3 and overwriting v1
        lines = [
            "v_mov_b32 v1, 0x11111111",
            "v_mov_b32 v3, 0x33333333",
            "s_mov_b32 m0, 2",
            "v_movrels_b32 v1, v1",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrels_alias", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x33333333)


class TestVMovrelsB32Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        lines = [
            "v_mov_b32 v6, 0x66778899",
            "s_mov_b32 m0, 1",
            "v_movrels_b32 v1, v5",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrels_full", lines)
        self.assertIn("OpCapability Shader", spv)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x66778899)

    def test_negative_rejects_sgpr_operands(self):
        lines = [
            "v_movrels_b32 v1, s2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_vmovrels_sgpr", lines)


class TestVMovrelsdB32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        cases = [
            (0, 0x22222222),
            (2, 0x44444444),
        ]
        for m0, expected in cases:
            with self.subTest(m0=m0):
                lines = [
                    "v_mov_b32 v2, 0x22222222",
                    "v_mov_b32 v4, 0x44444444",
                    f"s_mov_b32 m0, {m0}",
                    "v_movrelsd_b32 v5, v2",
                    f"v_mov_b32 v0, v{5 + m0}",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovrelsd_oracle_{m0}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], expected)

    def test_oracle_identity_offset(self):
        lines = [
            "v_mov_b32 v3, 0x33333333",
            "s_mov_b32 m0, 0",
            "v_movrelsd_b32 v6, v3",
            "v_mov_b32 v0, v6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd_identity", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x33333333)


class TestVMovrelsdB32AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        lines = [
            "v_mov_b32 v3, 0x77777777",
            "s_mov_b32 m0, 1",
            "v_movrelsd_b32 v5, v2",
            "v_mov_b32 v0, v6",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd_ast", lines)
        self.assertIn("OpAccessChain %_ptr_Function_uint %vgpr", spv)
        self.assertIn("OpStore", spv)

    def test_ast_execution_semantics(self):
        lines = [
            "v_mov_b32 v6, 0x88888888",
            "s_mov_b32 m0, 3",
            "v_movrelsd_b32 v7, v3",
            "v_mov_b32 v0, v10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x88888888)


class TestVMovrelsdB32Indexing(unittest.TestCase):
    def test_symmetric_offset_sweep(self):
        for offset in (1, 2):
            with self.subTest(offset=offset):
                val = 0x40404040 * offset
                lines = [
                    f"v_mov_b32 v{4 + offset}, {val}",
                    f"s_mov_b32 m0, {offset}",
                    "v_movrelsd_b32 v8, v4",
                    f"v_mov_b32 v0, v{8 + offset}",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_vmovrelsd_sweep_{offset}", lines)
                out = support.mrt0_words(spv)
                self.assertEqual(out[0], val)

    def test_unaffected_registers_preserved(self):
        lines = [
            "v_mov_b32 v5, 0x11112222",
            "v_mov_b32 v6, 0x33334444",
            "v_mov_b32 v7, 0x55556666",
            "s_mov_b32 m0, 1",
            "v_movrelsd_b32 v8, v5",
            "v_mov_b32 v0, v7",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd_preserve", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x55556666)


class TestVMovrelsdB32Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        lines = [
            "v_mov_b32 v4, 0x99999999",
            "s_mov_b32 m0, 1",
            "v_movrelsd_b32 v6, v3",
            "v_mov_b32 v0, v7",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd_full", lines)
        self.assertIn("OpCapability Shader", spv)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x99999999)

    def test_negative_rejects_sgpr_operands(self):
        lines = [
            "v_movrelsd_b32 s0, v1",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_vmovrelsd_sgpr", lines)


class TestVMovrelsd2B32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        # M0: dst_off in [25:16], src_off in [9:0]
        # m0 = (3 << 16) | 1 -> dst_off = 3, src_off = 1
        m0 = (3 << 16) | 1
        lines = [
            "v_mov_b32 v3, 0x12345678",
            f"s_mov_b32 m0, {m0}",
            "v_movrelsd_2_b32 v5, v2",
            "v_mov_b32 v0, v8",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_oracle", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x12345678)

    def test_oracle_zero_offsets(self):
        lines = [
            "v_mov_b32 v2, 0x99887766",
            "s_mov_b32 m0, 0",
            "v_movrelsd_2_b32 v4, v2",
            "v_mov_b32 v0, v4",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_zero", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x99887766)


class TestVMovrelsd2B32AST(unittest.TestCase):
    def test_ast_structure_and_emission(self):
        m0 = (2 << 16) | 4
        lines = [
            "v_mov_b32 v6, 0xCAFED00D",
            f"s_mov_b32 m0, {m0}",
            "v_movrelsd_2_b32 v5, v2",
            "v_mov_b32 v0, v7",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_ast", lines)
        self.assertIn("OpShiftRightLogical", spv)
        self.assertIn("OpBitwiseAnd", spv)

    def test_ast_execution_semantics(self):
        m0 = (4 << 16) | 2
        lines = [
            "v_mov_b32 v3, 0x11223344",
            f"s_mov_b32 m0, {m0}",
            "v_movrelsd_2_b32 v6, v1",
            "v_mov_b32 v0, v10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_exec", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x11223344)


class TestVMovrelsd2B32Bitfields(unittest.TestCase):
    def test_ignored_bits_invariance(self):
        # Bits 15:10 and 31:26 are ignored (0xFC00FC00)
        junk = 0xFC00FC00
        clean_m0 = (3 << 16) | 2
        noisy_m0 = clean_m0 | junk
        lines = [
            "v_mov_b32 v3, 0xFEED5555",
            f"s_mov_b32 m0, {noisy_m0}",
            "v_movrelsd_2_b32 v4, v1",
            "v_mov_b32 v0, v7",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_junk", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xFEED5555)

    def test_destination_selective_overwrite(self):
        # Check that neighbouring registers to the target remain untouched
        m0 = (2 << 16) | 1
        lines = [
            "v_mov_b32 v2, 0x11111111",
            "v_mov_b32 v5, 0xAAAA5555",
            "v_mov_b32 v6, 0x33333333",
            f"s_mov_b32 m0, {m0}",
            "v_movrelsd_2_b32 v4, v1",
            "v_mov_b32 v0, v5",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_neighbour", lines)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0xAAAA5555)


class TestVMovrelsd2B32Integration(unittest.TestCase):
    def test_full_pipeline_compilation(self):
        m0 = (1 << 16) | 2
        lines = [
            "v_mov_b32 v4, 0x7777AAAA",
            f"s_mov_b32 m0, {m0}",
            "v_movrelsd_2_b32 v2, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vmovrelsd2_full", lines)
        self.assertIn("OpCapability Shader", spv)
        out = support.mrt0_words(spv)
        self.assertEqual(out[0], 0x7777AAAA)

    def test_negative_rejects_sgpr_operands(self):
        lines = [
            "v_movrelsd_2_b32 s0, v1",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_vmovrelsd2_sgpr", lines)


if __name__ == "__main__":
    unittest.main()
