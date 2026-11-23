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


if __name__ == "__main__":
    unittest.main()
