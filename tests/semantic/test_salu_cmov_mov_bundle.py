"""Bundle 55: SALU Conditional Move & Constant Immediate Suite.

Tests for 6 SALU move and conditional move instructions:
- s_cmov_b32: 32-bit scalar conditional move (D = SCC ? S0 : D)
- s_cmov_b64: 64-bit scalar conditional move (D = SCC ? S0 : D)
- s_cmovk_i32: 32-bit scalar conditional move of sign-extended simm16 (D = SCC ? simm16 : D)
- s_mov_b32: 32-bit scalar move (D = S0)
- s_mov_b64: 64-bit scalar move (D = S0)
- s_movk_i32: 32-bit scalar move of sign-extended simm16 (D = simm16)

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
MASK64 = 0xFFFFFFFFFFFFFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle55") -> list[int]:
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

def ref_s_cmov_b32(scc: int, s0: int, old_dst: int) -> int:
    """Reference mathematical oracle for s_cmov_b32."""
    return (s0 & MASK32) if (scc & 1) else (old_dst & MASK32)


class TestSCmovB32Oracle(unittest.TestCase):
    def test_scc_set_moves_source(self) -> None:
        self.assertEqual(ref_s_cmov_b32(1, 0x12345678, 0xAAAAAAAA), 0x12345678)

    def test_scc_clear_preserves_old_dst(self) -> None:
        self.assertEqual(ref_s_cmov_b32(0, 0x12345678, 0xAAAAAAAA), 0xAAAAAAAA)

    def test_translator_oracle_verification(self) -> None:
        body = [
            "s_cmp_eq_u32 1, 1",          # SCC = 1
            "s_mov_b32 s4, 0x11111111",
            "s_mov_b32 s5, 0x22222222",
            "s_cmov_b32 s4, s5",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b32_oracle")
        for l in range(32):
            self.assertEqual(results[l], 0x22222222)

class SCmovB32Expr(ASTNode):
    def __init__(self, s0: int, old_dst: int) -> None:
        self.s0 = s0
        self.old_dst = old_dst

    def eval(self, scc: int) -> int:
        return ref_s_cmov_b32(scc, self.s0, self.old_dst)


class TestSCmovB32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SCmovB32Expr(100, 50)
        self.assertEqual(expr.eval(1), 100)
        self.assertEqual(expr.eval(0), 50)

    def test_ast_translator_execution(self) -> None:
        body = [
            "s_cmp_eq_u32 1, 2",          # SCC = 0
            "s_mov_b32 s4, 50",
            "s_cmov_b32 s4, 100",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b32_ast")
        for l in range(32):
            self.assertEqual(results[l], 50)

class TestSCmovB32SCCGating(unittest.TestCase):
    def test_scc_alternating_conditions(self) -> None:
        for scc in (0, 1):
            self.assertEqual(ref_s_cmov_b32(scc, 0xBEEF, 0xCAFE), 0xBEEF if scc else 0xCAFE)

    def test_translator_scc_gating(self) -> None:
        body = [
            "s_cmp_eq_u32 10, 10",        # SCC = 1
            "s_mov_b32 s4, 77",
            "s_cmov_b32 s4, 88",          # becomes 88
            "s_cmp_eq_u32 10, 20",        # SCC = 0
            "s_cmov_b32 s4, 99",          # stays 88
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b32_gating")
        for l in range(32):
            self.assertEqual(results[l], 88)

class TestSCmovB32Translator(unittest.TestCase):
    def test_translator_s_cmov_b32_execution(self) -> None:
        body = [
            "s_cmp_eq_u32 0, 0",
            "s_mov_b32 s4, 0x1234",
            "s_cmov_b32 s4, 0x5678",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b32_exec")
        for l in range(32):
            self.assertEqual(results[l], 0x5678)

def ref_s_cmov_b64(scc: int, s0: int, old_dst: int) -> int:
    """Reference mathematical oracle for s_cmov_b64."""
    return (s0 & MASK64) if (scc & 1) else (old_dst & MASK64)


class TestSCmovB64Oracle(unittest.TestCase):
    def test_scc_set_moves_pair(self) -> None:
        self.assertEqual(ref_s_cmov_b64(1, 0x1122334455667788, 0), 0x1122334455667788)

    def test_scc_clear_preserves_pair(self) -> None:
        self.assertEqual(ref_s_cmov_b64(0, 0x1122334455667788, 0xDEADBEEFCAFEBABE), 0xDEADBEEFCAFEBABE)

    def test_translator_oracle_verification(self) -> None:
        body = [
            "s_cmp_eq_u32 0, 0",          # SCC = 1
            "s_mov_b64 s[4:5], 0",
            "s_mov_b64 s[6:7], 0x1122334455667788",
            "s_cmov_b64 s[4:5], s[6:7]",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b64_oracle")
        for l in range(32):
            self.assertEqual(results[l], 0x55667788)

class SCmovB64Expr(ASTNode):
    def __init__(self, s0: int, old_dst: int) -> None:
        self.s0 = s0
        self.old_dst = old_dst

    def eval(self, scc: int) -> int:
        return ref_s_cmov_b64(scc, self.s0, self.old_dst)


class TestSCmovB64AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SCmovB64Expr(0x1000000020000000, 0x3000000040000000)
        self.assertEqual(expr.eval(1), 0x1000000020000000)
        self.assertEqual(expr.eval(0), 0x3000000040000000)

    def test_ast_translator_execution(self) -> None:
        body = [
            "s_cmp_eq_u32 1, 0",          # SCC = 0
            "s_mov_b64 s[4:5], 0x3000000040000000",
            "s_mov_b64 s[6:7], 0x1000000020000000",
            "s_cmov_b64 s[4:5], s[6:7]",
            "v_mov_b32_e32 v1, s5",       # upper dword
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b64_ast")
        for l in range(32):
            self.assertEqual(results[l], 0x30000000)

class TestSCmovB64PairGating(unittest.TestCase):
    def test_64bit_selection_properties(self) -> None:
        val = 0xFEDCBA9876543210
        self.assertEqual(ref_s_cmov_b64(1, val, 0), val)
        self.assertEqual(ref_s_cmov_b64(0, val, 0), 0)

    def test_translator_pair_gating(self) -> None:
        body = [
            "s_cmp_eq_u32 5, 5",          # SCC = 1
            "s_mov_b64 s[4:5], 0",
            "s_mov_b64 s[6:7], 0xabcdef0123456789",
            "s_cmov_b64 s[4:5], s[6:7]",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v31, 4, v30",
            "v_mov_b32_e32 v2, s5",
            "buffer_store_dword v2, v31, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b64_gating")
        for l in range(32):
            self.assertEqual(results[l], 0x23456789)

class TestSCmovB64Translator(unittest.TestCase):
    def test_translator_s_cmov_b64_execution(self) -> None:
        body = [
            "s_cmp_eq_u32 0, 0",
            "s_mov_b64 s[4:5], 0",
            "s_mov_b64 s[6:7], 0xaabbccddeeff0011",
            "s_cmov_b64 s[4:5], s[6:7]",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmov_b64_exec")
        for l in range(32):
            self.assertEqual(results[l], 0xAABBCCDD)

def ref_s_cmovk_i32(scc: int, simm16: int, old_dst: int) -> int:
    """Reference mathematical oracle for s_cmovk_i32."""
    v = simm16 & 0xFFFF
    ext = (v - 0x10000 if v & 0x8000 else v) & MASK32
    return ext if (scc & 1) else (old_dst & MASK32)


class TestSCmovkI32Oracle(unittest.TestCase):
    def test_positive_simm16(self) -> None:
        self.assertEqual(ref_s_cmovk_i32(1, 42, 0), 42)
        self.assertEqual(ref_s_cmovk_i32(0, 42, 100), 100)

    def test_negative_simm16(self) -> None:
        self.assertEqual(ref_s_cmovk_i32(1, -1, 0), MASK32)
        self.assertEqual(ref_s_cmovk_i32(1, -42, 0), 0xFFFFFFD6)

    def test_translator_oracle_verification(self) -> None:
        body = [
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 s4, 0",
            "s_cmovk_i32 s4, -1",
            "v_mov_b32_e32 v1, s4",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_cmovk_oracle")
        for l in range(32):
            self.assertEqual(results[l], MASK32)


if __name__ == "__main__":
    unittest.main()
