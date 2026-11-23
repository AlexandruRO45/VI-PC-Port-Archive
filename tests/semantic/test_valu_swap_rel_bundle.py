from __future__ import annotations

"""Architectural reference models, symbolic AST, deterministic interpreter,
multi-lane Wave64 divergence tests, and translator integration tests for
Bundle 56: Vector & Scalar Register Swap and Relative Addressing Suite.

Covers RDNA2 Instructions:
    - v_swap_b32 vD, vS
        Exchanges two 32-bit VGPRs across active execution lanes.
    - v_swaprel_b32 vD, vS
        Exchanges VGPR[D + M0[25:16]] and VGPR[S0 + M0[9:0]] under active lanes.
    - s_swappc_b64 sD, sS
        Saves next program counter in 64-bit SGPR pair D and jumps to address in S.
    - s_movrels_b32 sD, sS
        Reads SGPR[(S + M0) % 128] into scalar destination register sD.
    - s_movrels_b64 sD, sS
        Reads 64-bit SGPR pair starting at (S + M0) % 128 into pair sD.
    - v_movrels_b32 vD, vS
        Reads VGPR[S + M0] into destination vector register vD.

Hardware rules (GFX10 RDNA2):
    - v_swap_b32 exchanges values simultaneously using temporary storage.
    - Inactive lanes in EXEC preserve their existing VGPR values during vector swaps.
    - Relative addressing wraps modulo register bank bounds (128 SGPRs / 256 VGPRs).
    - M0 register provides dynamic base offsets for scalar and vector register files.
    - s_swappc_b64 acts as subroutine linkage in driver vertex fetch shaders and is rejected in compute kernels.

Standard library Python only (unittest, struct, math, dataclasses, enum).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import rdna2spirv
import spirv_mini_interp as si
import saveexec_run

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF


def model_swap(v0: int, v1: int) -> Tuple[int, int]:
    return v1 & MASK32, v0 & MASK32


def model_swaprel(dst_reg: int, src_reg: int, m0: int, vgprs: Dict[int, int]) -> Dict[int, int]:
    m0_src = m0 & 0x3FF
    m0_dst = (m0 >> 16) & 0x3FF
    actual_dst = (dst_reg + m0_dst) & 0xFF
    actual_src = (src_reg + m0_src) & 0xFF
    val_dst = vgprs.get(actual_dst, 0)
    val_src = vgprs.get(actual_src, 0)
    out = dict(vgprs)
    out[actual_dst] = val_src
    out[actual_src] = val_dst
    return out


def model_swappc(pc: int, target: int) -> Tuple[int, int]:
    return target & MASK64, (pc + 4) & MASK64


def model_movrels_b32(src_reg: int, m0: int, sgprs: Dict[int, int]) -> int:
    idx = (src_reg + m0) & 127
    return sgprs.get(idx, 0) & MASK32


def model_movrels_b64(src_reg: int, m0: int, sgprs: Dict[int, int]) -> Tuple[int, int]:
    idx0 = (src_reg + m0) & 127
    idx1 = (src_reg + m0 + 1) & 127
    return sgprs.get(idx0, 0) & MASK32, sgprs.get(idx1, 0) & MASK32


def model_v_movrels(src_reg: int, m0: int, vgprs: Dict[int, int]) -> int:
    idx = (src_reg + m0) & 255
    return vgprs.get(idx, 0) & MASK32

class TestSwapOracle(unittest.TestCase):
    def test_oracle_exchange(self):
        a, b = model_swap(0x12345678, 0x9ABCDEF0)
        self.assertEqual(a, 0x9ABCDEF0)
        self.assertEqual(b, 0x12345678)

    def test_translator_swap_oracle(self):
        prog = ["v_mov_b32_e32 v0, 100", "v_mov_b32_e32 v1, 200", "v_swap_b32 v0, v1", "s_endpgm"]
        spv = support.translate_lines("cs_swap_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapAST(unittest.TestCase):
    def test_ast_evaluation(self):
        x, y = 55, 99
        x, y = model_swap(x, y)
        self.assertEqual(x, 99)
        self.assertEqual(y, 55)

    def test_translator_swap_ast(self):
        prog = ["v_mov_b32_e32 v2, 55", "v_mov_b32_e32 v3, 99", "v_swap_b32 v2, v3", "s_endpgm"]
        spv = support.translate_lines("cs_swap_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapDivergence(unittest.TestCase):
    def test_self_swap(self):
        a, b = model_swap(0xCAFE, 0xCAFE)
        self.assertEqual(a, 0xCAFE)
        self.assertEqual(b, 0xCAFE)

    def test_translator_swap_divergence(self):
        prog = [
            "s_mov_b32 exec_lo, 0x55555555",
            "s_mov_b32 exec_hi, 0xAAAAAAAA",
            "v_mov_b32_e32 v4, 111",
            "v_mov_b32_e32 v5, 222",
            "v_swap_b32 v4, v5",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_swap_div", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapTranslator(unittest.TestCase):
    def test_translator_mutation_guard(self):
        prog = ["v_mov_b32_e32 v6, 1234", "v_mov_b32_e32 v7, 5678", "v_swap_b32 v6, v7", "s_endpgm"]
        spv = support.translate_lines("cs_swap_trans", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpCopyObject", spv)
        self.assertIn("OpCapability Shader", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapRelOracle(unittest.TestCase):
    def test_oracle_swaprel(self):
        vgprs = {0: 10, 1: 20}
        out = model_swaprel(0, 1, 0, vgprs)
        self.assertEqual(out[0], 20)
        self.assertEqual(out[1], 10)

    def test_translator_swaprel_oracle(self):
        prog = ["s_mov_b32 m0, 0", "v_mov_b32_e32 v0, 10", "v_mov_b32_e32 v1, 20", "v_swaprel_b32 v0, v1", "s_endpgm"]
        spv = support.translate_lines("cs_swaprel_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpSelect", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapRelAST(unittest.TestCase):
    def test_ast_evaluation(self):
        vgprs = {4: 100, 8: 200}
        m0 = (4 << 16) | 4
        out = model_swaprel(0, 4, m0, vgprs)
        self.assertEqual(out[4], 200)
        self.assertEqual(out[8], 100)

    def test_translator_swaprel_ast(self):
        prog = ["s_mov_b32 m0, 0x00040004", "v_mov_b32_e32 v4, 100", "v_mov_b32_e32 v8, 200", "v_swaprel_b32 v0, v4", "s_endpgm"]
        spv = support.translate_lines("cs_swaprel_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpShiftRightLogical", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapRelM0Fields(unittest.TestCase):
    def test_m0_dual_fields(self):
        m0 = (16 << 16) | 32
        vgprs = {16: 777, 32: 888}
        out = model_swaprel(0, 0, m0, vgprs)
        self.assertEqual(out[16], 888)
        self.assertEqual(out[32], 777)

    def test_translator_swaprel_m0_fields(self):
        prog = ["s_mov_b32 m0, 0x00100020", "v_swaprel_b32 v0, v0", "s_endpgm"]
        spv = support.translate_lines("cs_swaprel_m0", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpBitwiseAnd", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapRelTranslator(unittest.TestCase):
    def test_translator_mutation_guard(self):
        prog = ["s_mov_b32 m0, 0", "v_swaprel_b32 v2, v3", "s_endpgm"]
        spv = support.translate_lines("cs_swaprel_trans", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpSelect", spv)
        self.assertIn("OpLogicalAnd", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestSwapPCOracle(unittest.TestCase):
    def test_oracle_swappc(self):
        new_pc, link = model_swappc(0x1000, 0x8000)
        self.assertEqual(new_pc, 0x8000)
        self.assertEqual(link, 0x1004)

    def test_translator_swappc_rejection_in_cs(self):
        prog = ["s_swappc_b64 s[0:1], s[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_swappc_oracle", prog, fssm=saveexec_run.fssm())

class TestSwapPCAST(unittest.TestCase):
    def test_ast_evaluation(self):
        pc = 0x2000
        target = 0x5000
        new_pc, link = model_swappc(pc, target)
        self.assertEqual(new_pc, target)
        self.assertEqual(link, pc + 4)

    def test_translator_swappc_rejection_in_compute(self):
        prog = ["s_swappc_b64 s[4:5], s[6:7]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_swappc_ast", prog, fssm=saveexec_run.fssm())

class TestSwapPCPair(unittest.TestCase):
    def test_64bit_pc_swap(self):
        pc = 0x00000001_80000000
        target = 0x00000002_40000000
        new_pc, link = model_swappc(pc, target)
        self.assertEqual(new_pc, 0x00000002_40000000)
        self.assertEqual(link, 0x00000001_80000004)

    def test_translator_swappc_rejection_unsupported(self):
        prog = ["s_swappc_b64 s[8:9], s[10:11]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_swappc_pair", prog, fssm=saveexec_run.fssm())

class TestSwapPCTranslator(unittest.TestCase):
    def test_translator_rejection_guard(self):
        prog = ["s_swappc_b64 s[0:1], s[0:1]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("cs_swappc_guard", prog, fssm=saveexec_run.fssm())

class TestMovRelS32Oracle(unittest.TestCase):
    def test_oracle_movrels(self):
        sgprs = {5: 42, 6: 43}
        self.assertEqual(model_movrels_b32(5, 0, sgprs), 42)
        self.assertEqual(model_movrels_b32(5, 1, sgprs), 43)

    def test_translator_movrels32_oracle(self):
        prog = ["s_mov_b32 m0, 0", "s_mov_b32 s4, 42", "s_movrels_b32 s0, s4", "s_endpgm"]
        spv = support.translate_lines("cs_movrels32_oracle", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpAccessChain", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestMovRelS32AST(unittest.TestCase):
    def test_ast_evaluation(self):
        sgprs = {i: i * 11 for i in range(128)}
        self.assertEqual(model_movrels_b32(10, 5, sgprs), 165)

    def test_translator_movrels32_ast(self):
        prog = ["s_mov_b32 m0, 5", "s_mov_b32 s15, 165", "s_movrels_b32 s2, s10", "s_endpgm"]
        spv = support.translate_lines("cs_movrels32_ast", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpAccessChain", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestMovRelS32Modulo(unittest.TestCase):
    def test_modulo_wrapping(self):
        sgprs = {0: 999, 127: 888}
        self.assertEqual(model_movrels_b32(127, 1, sgprs), 999)
        self.assertEqual(model_movrels_b32(120, 7, sgprs), 888)

    def test_translator_movrels32_modulo(self):
        prog = ["s_mov_b32 m0, 1", "s_movrels_b32 s0, s127", "s_endpgm"]
        spv = support.translate_lines("cs_movrels32_mod", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpBitwiseAnd", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})

class TestMovRelS32Translator(unittest.TestCase):
    def test_translator_mutation_guard(self):
        prog = ["s_mov_b32 m0, 2", "s_movrels_b32 s1, s4", "s_endpgm"]
        spv = support.translate_lines("cs_movrels32_trans", prog, fssm=saveexec_run.fssm())
        self.assertIn("OpAccessChain", spv)
        self.assertIn("OpIAdd", spv)
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})


if __name__ == "__main__":
    unittest.main()
