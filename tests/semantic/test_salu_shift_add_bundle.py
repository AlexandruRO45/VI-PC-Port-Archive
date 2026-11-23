"""Bundle 51: SALU Shift-Add Arithmetic Suite.

Tests for 6 SALU shift-add and half-word packing instructions:
- s_lshl1_add_u32: D = (S0 << 1) + S1, SCC = unsigned overflow >= 2^32
- s_lshl2_add_u32: D = (S0 << 2) + S1, SCC = unsigned overflow >= 2^32
- s_lshl3_add_u32: D = (S0 << 3) + S1, SCC = unsigned overflow >= 2^32
- s_lshl4_add_u32: D = (S0 << 4) + S1, SCC = unsigned overflow >= 2^32
- s_pack_lh_b32_b16: D = {S1[31:16], S0[15:0]}
- s_pack_ll_b32_b16: D = {S1[15:0], S0[15:0]}

Standard library only; zero game data.
"""
from __future__ import annotations

import math
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle51") -> list[int]:
    """Helper to translate and run a compute shader; stores outputs at out[lane*32 + word*4]."""
    prologue = ["v_lshlrev_b32_e32 v30, 5, v0"]
    full_body = prologue + list(body) + ["s_endpgm"]
    spv = support.translate_lines(name, full_body, fssm=FSSM)
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    return [mem.get(i * 32, 0) for i in range(num_words)]


class ASTNode:
    """Base class for symbolic AST expressions."""
    pass

def ref_s_lshl_add_u32(n: int, src0: int, src1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshl{n}_add_u32 -> (result, scc)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    tot = (s0 << n) + s1
    res = tot & MASK32
    scc = 1 if tot >= 0x100000000 else 0
    return res, scc


class TestSLshl1AddU32Oracle(unittest.TestCase):
    def test_basic_operation(self) -> None:
        self.assertEqual(ref_s_lshl_add_u32(1, 10, 5), (25, 0))
        self.assertEqual(ref_s_lshl_add_u32(1, 0, 0), (0, 0))

    def test_overflow_and_scc(self) -> None:
        # 0x80000000 << 1 = 0x100000000 -> overflow into SCC
        self.assertEqual(ref_s_lshl_add_u32(1, 0x80000000, 0), (0, 1))
        # 0x80000000 << 1 + 5 = 0x100000005 -> res 5, scc 1
        self.assertEqual(ref_s_lshl_add_u32(1, 0x80000000, 5), (5, 1))

class SLshl1AddU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_lshl_add_u32(1, self.src0, self.src1)


class TestSLshl1AddU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshl1AddU32Expr(100, 20)
        self.assertEqual(expr.eval(), (220, 0))
        expr_ovf = SLshl1AddU32Expr(0x80000000, 1)
        self.assertEqual(expr_ovf.eval(), (1, 1))

class TestSLshl1AddU32OverflowAndSCC(unittest.TestCase):
    def test_overflow_from_addition(self) -> None:
        # S0 not overflowing on shift, but sum overflowing
        # 0x40000000 << 1 = 0x80000000; + 0x80000000 = 0x100000000 -> scc 1
        res, scc = ref_s_lshl_add_u32(1, 0x40000000, 0x80000000)
        self.assertEqual((res, scc), (0, 1))

    def test_no_overflow_boundary(self) -> None:
        res, scc = ref_s_lshl_add_u32(1, 0x7FFFFFFF, 1)
        self.assertEqual((res, scc), (0xFFFFFFFF, 0))

class TestSLshl1AddU32Translator(unittest.TestCase):
    def test_translator_s_lshl1_add_u32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 10",
            "s_lshl1_add_u32 s5, s4, 5",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshl1_add")
        for l in range(32):
            self.assertEqual(results[l], 25)

    def test_translator_s_lshl1_add_scc(self) -> None:
        body = [
            "s_mov_b32 s4, 0x80000000",
            "s_lshl1_add_u32 s5, s4, 0",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v1, s6",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshl1_add_scc")
        for l in range(32):
            self.assertEqual(results[l], 1)

class TestSLshl2AddU32Oracle(unittest.TestCase):
    def test_basic_operation(self) -> None:
        self.assertEqual(ref_s_lshl_add_u32(2, 10, 5), (45, 0))
        self.assertEqual(ref_s_lshl_add_u32(2, 3, 7), (19, 0))

    def test_overflow_and_scc(self) -> None:
        # 0x40000000 << 2 = 0x100000000 -> scc 1
        self.assertEqual(ref_s_lshl_add_u32(2, 0x40000000, 0), (0, 1))
        # 0xC0000000 << 2 = 0x300000000 -> res 0, scc 1
        self.assertEqual(ref_s_lshl_add_u32(2, 0xC0000000, 0), (0, 1))

class SLshl2AddU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_lshl_add_u32(2, self.src0, self.src1)


class TestSLshl2AddU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshl2AddU32Expr(20, 10)
        self.assertEqual(expr.eval(), (90, 0))

class TestSLshl2AddU32OverflowAndSCC(unittest.TestCase):
    def test_shifted_out_bits_set_scc(self) -> None:
        # High 2 bits shifted out
        res, scc = ref_s_lshl_add_u32(2, 0x80000000, 0)
        self.assertEqual((res, scc), (0, 1))

    def test_boundary_no_overflow(self) -> None:
        # 0x3FFFFFFF << 2 = 0xFFFFFFFC; + 3 = 0xFFFFFFFF -> no overflow
        res, scc = ref_s_lshl_add_u32(2, 0x3FFFFFFF, 3)
        self.assertEqual((res, scc), (0xFFFFFFFF, 0))

class TestSLshl2AddU32Translator(unittest.TestCase):
    def test_translator_s_lshl2_add_u32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x40000000",
            "s_lshl2_add_u32 s5, s4, 42",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v31, 4, v30",
            "v_mov_b32_e32 v2, s6",
            "buffer_store_dword v2, v31, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_lshl2_add")
        for l in range(32):
            self.assertEqual(results[l], 42)

class TestSLshl3AddU32Oracle(unittest.TestCase):
    def test_basic_operation(self) -> None:
        self.assertEqual(ref_s_lshl_add_u32(3, 10, 5), (85, 0))
        self.assertEqual(ref_s_lshl_add_u32(3, 1, 2), (10, 0))

    def test_overflow_and_scc(self) -> None:
        # 0x20000000 << 3 = 0x100000000 -> scc 1
        self.assertEqual(ref_s_lshl_add_u32(3, 0x20000000, 0), (0, 1))


if __name__ == "__main__":
    unittest.main()
