from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
from support import rdna2spirv

MASK32 = 0xFFFFFFFF
NUM_VGPRS = 256
NUM_SGPRS = 128


def oracle_s_movrels_b32(dst: int, src: int, m0: int, sgprs: list[int]) -> list[int]:
    out = list(sgprs)
    idx = (src + m0) & (NUM_SGPRS - 1)
    out[dst] = out[idx] & MASK32
    return out


def oracle_s_movrels_b64(dst: int, src: int, m0: int, sgprs: list[int]) -> list[int]:
    out = list(sgprs)
    idx0 = (src + m0) & (NUM_SGPRS - 1)
    idx1 = (src + m0 + 1) & (NUM_SGPRS - 1)
    out[dst] = out[idx0] & MASK32
    out[dst + 1] = out[idx1] & MASK32
    return out


def oracle_v_movreld_b32(dst: int, src: int, m0: int, vgprs: list[int]) -> list[int]:
    out = list(vgprs)
    idx = (dst + m0) & (NUM_VGPRS - 1)
    out[idx] = out[src] & MASK32
    return out


def oracle_v_movrels_b32(dst: int, src: int, m0: int, vgprs: list[int]) -> list[int]:
    out = list(vgprs)
    idx = (src + m0) & (NUM_VGPRS - 1)
    out[dst] = out[idx] & MASK32
    return out


def oracle_v_movrelsd_b32(dst: int, src: int, m0: int, vgprs: list[int]) -> list[int]:
    out = list(vgprs)
    src_idx = (src + m0) & (NUM_VGPRS - 1)
    dst_idx = (dst + m0) & (NUM_VGPRS - 1)
    out[dst_idx] = out[src_idx] & MASK32
    return out


def oracle_v_movrelsd_2_b32(dst: int, src: int, m0: int, vgprs: list[int]) -> list[int]:
    out = list(vgprs)
    src_off = m0 & 0x3FF
    dst_off = (m0 >> 16) & 0x3FF
    src_idx = (src + src_off) & (NUM_VGPRS - 1)
    dst_idx = (dst + dst_off) & (NUM_VGPRS - 1)
    out[dst_idx] = out[src_idx] & MASK32
    return out


class TestSMovrelsB32Oracle(unittest.TestCase):
    def test_oracle_model_against_translator(self):
        # Seed s20..s23 with sentinels and test m0 = 0, 1, 2
        for m0 in (0, 1, 2):
            with self.subTest(m0=m0):
                sgprs = [0] * NUM_SGPRS
                sgprs[20] = 0x11112222
                sgprs[21] = 0x33334444
                sgprs[22] = 0x55556666
                expected_sgprs = oracle_s_movrels_b32(10, 20, m0, sgprs)
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
                self.assertEqual(out[0], expected_sgprs[10])

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
        # Index wraps modulo 128
        sgprs = [0] * NUM_SGPRS
        sgprs[5] = 0x9999AAAA
        expected = oracle_s_movrels_b32(10, 120, 13, sgprs)
        self.assertEqual(expected[10], 0x9999AAAA)


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
        for m0 in (0, 2):
            with self.subTest(m0=m0):
                sgprs = [0] * NUM_SGPRS
                sgprs[20] = 0x11223344
                sgprs[21] = 0x55667788
                sgprs[22] = 0x99AABBCC
                sgprs[23] = 0xDDEEFF00
                expected = oracle_s_movrels_b64(10, 20, m0, sgprs)
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
                self.assertEqual(out[0], expected[10])
                self.assertEqual(out[1], expected[11])

    def test_oracle_consecutive_words(self):
        sgprs = [0] * NUM_SGPRS
        sgprs[30] = 0xDEADBEEF
        sgprs[31] = 0xFEEDFACE
        expected = oracle_s_movrels_b64(12, 30, 0, sgprs)
        self.assertEqual(expected[12], 0xDEADBEEF)
        self.assertEqual(expected[13], 0xFEEDFACE)


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
                vgprs = [0] * NUM_VGPRS
                vgprs[5] = 0x33445566
                expected = oracle_v_movreld_b32(2, 5, m0, vgprs)
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
                self.assertEqual(out[0], expected[2 + m0])

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
        for m0 in (0, 2):
            with self.subTest(m0=m0):
                vgprs = [0] * NUM_VGPRS
                vgprs[4] = 0x44556677
                vgprs[6] = 0x8899AABB
                expected = oracle_v_movrels_b32(1, 4, m0, vgprs)
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
                self.assertEqual(out[0], expected[1])

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
