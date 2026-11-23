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
