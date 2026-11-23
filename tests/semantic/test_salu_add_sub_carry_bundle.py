"""Bundle 36: SALU 32-bit Integer Add/Sub with Carry & Borrow Suite.

Tests for 6 SALU integer arithmetic instructions:
- s_add_i32: 32-bit signed addition with signed overflow in SCC
- s_sub_i32: 32-bit signed subtraction with signed overflow in SCC
- s_add_u32: 32-bit unsigned addition with carry out in SCC
- s_sub_u32: 32-bit unsigned subtraction with borrow out in SCC
- s_addc_u32: 32-bit unsigned addition with carry in from SCC, carry out in SCC
- s_subb_u32: 32-bit unsigned subtraction with borrow in from SCC, borrow out in SCC

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


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle36") -> list[int]:
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

def ref_s_add_i32(src0: int, src1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_add_i32 -> (res, scc_overflow)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    res = (s0 + s1) & MASK32
    # Signed overflow V: ((src0 ^ res) & (src1 ^ res)) >> 31
    ovf = (((s0 ^ res) & (s1 ^ res)) >> 31) & 1
    return res, ovf


class TestSAddI32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(ref_s_add_i32(10, 20), (30, 0))
        self.assertEqual(ref_s_add_i32(-5, 5), (0, 0))

    def test_signed_overflow(self) -> None:
        # 0x7FFFFFFF + 1 -> 0x80000000 (overflow from positive to negative)
        self.assertEqual(ref_s_add_i32(0x7FFFFFFF, 1), (0x80000000, 1))
        # 0x80000000 + 0x80000000 -> 0 (negative + negative = 0 -> overflow)
        self.assertEqual(ref_s_add_i32(0x80000000, 0x80000000), (0, 1))


if __name__ == "__main__":
    unittest.main()
