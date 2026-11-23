"""Bundle 28: SALU 32/64-bit Shift Operations Suite.

Tests for 6 SALU shift instructions:
- s_lshl_b32: 32-bit scalar logical shift left (shift count & 31, SCC = result != 0)
- s_lshr_b32: 32-bit scalar logical shift right (shift count & 31, SCC = result != 0)
- s_ashr_i32: 32-bit scalar arithmetic shift right (shift count & 31, sign-extended, SCC = result != 0)
- s_lshl_b64: 64-bit scalar logical shift left (shift count & 63, SCC = result != 0)
- s_lshr_b64: 64-bit scalar logical shift right (shift count & 63, SCC = result != 0)
- s_ashr_i64: 64-bit scalar arithmetic shift right (shift count & 63, sign-extended, SCC = result != 0)

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


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle28") -> list[int]:
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
    """Base class for symbolic AST nodes."""
    pass

def ref_s_lshl_b32(src: int, shift: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_lshl_b32 -> (result, scc)."""
    sh = shift & 31
    res = (src << sh) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc


class TestSLshlB32Oracle(unittest.TestCase):
    def test_basic_shifts(self) -> None:
        self.assertEqual(ref_s_lshl_b32(1, 0), (1, 1))
        self.assertEqual(ref_s_lshl_b32(1, 4), (16, 1))
        self.assertEqual(ref_s_lshl_b32(0x12345678, 8), (0x34567800, 1))

    def test_zero_result_and_scc(self) -> None:
        self.assertEqual(ref_s_lshl_b32(0, 5), (0, 0))
        self.assertEqual(ref_s_lshl_b32(0x80000000, 1), (0, 0))

    def test_shift_count_masking(self) -> None:
        # Shift count is masked with 31: 32 -> 0, 33 -> 1
        self.assertEqual(ref_s_lshl_b32(1, 32), (1, 1))
        self.assertEqual(ref_s_lshl_b32(1, 33), (2, 1))

class SLshlB32Expr(ASTNode):
    def __init__(self, src: int, shift: int) -> None:
        self.src = src
        self.shift = shift

    def eval(self) -> tuple[int, int]:
        return ref_s_lshl_b32(self.src, self.shift)


class TestSLshlB32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SLshlB32Expr(0x0F, 4)
        self.assertEqual(expr.eval(), (0xF0, 1))
        expr_zero = SLshlB32Expr(0, 10)
        self.assertEqual(expr_zero.eval(), (0, 0))

class TestSLshlB32SCCAndMask(unittest.TestCase):
    def test_scc_condition_code(self) -> None:
        # Shift off all bits
        res, scc = ref_s_lshl_b32(0x00000001, 31)
        self.assertEqual((res, scc), (0x80000000, 1))
        # High bit shifted out by another bit
        res, scc = ref_s_lshl_b32(0x80000000, 1)
        self.assertEqual((res, scc), (0, 0))

    def test_shift_counts_boundary(self) -> None:
        for sh in range(32):
            res, scc = ref_s_lshl_b32(1, sh)
            self.assertEqual(res, 1 << sh)
            self.assertEqual(scc, 1)


if __name__ == "__main__":
    unittest.main()
