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

class SAddI32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_add_i32(self.src0, self.src1)


class TestSAddI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SAddI32Expr(100, -50)
        self.assertEqual(expr.eval(), (50, 0))
        expr_ovf = SAddI32Expr(0x7FFFFFFF, 10)
        self.assertEqual(expr_ovf.eval(), ((0x7FFFFFFF + 10) & MASK32, 1))

class TestSAddI32OverflowAndSCC(unittest.TestCase):
    def test_no_overflow_opposite_signs(self) -> None:
        # Opposite signs can never overflow
        self.assertEqual(ref_s_add_i32(0x7FFFFFFF, 0x80000000), (0xFFFFFFFF, 0))

    def test_boundary_values(self) -> None:
        # Max positive int32 + 0 -> no overflow
        self.assertEqual(ref_s_add_i32(0x7FFFFFFF, 0), (0x7FFFFFFF, 0))
        # Min negative int32 + -1 -> overflow
        self.assertEqual(ref_s_add_i32(0x80000000, 0xFFFFFFFF), (0x7FFFFFFF, 1))

class TestSAddI32Translator(unittest.TestCase):
    def test_translator_s_add_i32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x7fffffff",
            "s_add_i32 s5, s4, 1",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_add_i32")
        for l in range(32):
            self.assertEqual(results[l], 0x80000000)

def ref_s_sub_i32(src0: int, src1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_sub_i32 -> (res, scc_overflow)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    res = (s0 - s1) & MASK32
    # Signed overflow V: ((src0 ^ src1) & (src0 ^ res)) >> 31
    ovf = (((s0 ^ s1) & (s0 ^ res)) >> 31) & 1
    return res, ovf


class TestSSubI32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        self.assertEqual(ref_s_sub_i32(50, 20), (30, 0))
        self.assertEqual(ref_s_sub_i32(10, 10), (0, 0))

    def test_signed_overflow(self) -> None:
        # 0x80000000 - 1 -> 0x7FFFFFFF (negative - positive = positive -> overflow)
        self.assertEqual(ref_s_sub_i32(0x80000000, 1), (0x7FFFFFFF, 1))
        # 0x7FFFFFFF - (-1) -> 0x80000000 (overflow)
        self.assertEqual(ref_s_sub_i32(0x7FFFFFFF, 0xFFFFFFFF), (0x80000000, 1))

class SSubI32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_sub_i32(self.src0, self.src1)


class TestSSubI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SSubI32Expr(100, 40)
        self.assertEqual(expr.eval(), (60, 0))
        expr_ovf = SSubI32Expr(0x80000000, 1)
        self.assertEqual(expr_ovf.eval(), (0x7FFFFFFF, 1))

class TestSSubI32OverflowAndSCC(unittest.TestCase):
    def test_same_signs_no_overflow(self) -> None:
        # Subtracting same signs never overflows
        self.assertEqual(ref_s_sub_i32(0x7FFFFFFF, 0x7FFFFFFF), (0, 0))
        self.assertEqual(ref_s_sub_i32(0x80000000, 0x80000000), (0, 0))

    def test_sub_zero_no_overflow(self) -> None:
        self.assertEqual(ref_s_sub_i32(0x80000000, 0), (0x80000000, 0))

class TestSSubI32Translator(unittest.TestCase):
    def test_translator_s_sub_i32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0x80000000",
            "s_sub_i32 s5, s4, 1",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_sub_i32")
        for l in range(32):
            self.assertEqual(results[l], 0x7FFFFFFF)

def ref_s_add_u32(src0: int, src1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_add_u32 -> (res, scc_carry)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    total = s0 + s1
    res = total & MASK32
    carry = 1 if total >= 0x100000000 else 0
    return res, carry


class TestSAddU32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(ref_s_add_u32(100, 200), (300, 0))

    def test_carry_out(self) -> None:
        self.assertEqual(ref_s_add_u32(0xFFFFFFFF, 1), (0, 1))
        self.assertEqual(ref_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFE, 1))

class SAddU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_add_u32(self.src0, self.src1)


class TestSAddU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SAddU32Expr(0xFFFFFFFF, 2)
        self.assertEqual(expr.eval(), (1, 1))

class TestSAddU32CarryAndSCC(unittest.TestCase):
    def test_no_carry_just_below_limit(self) -> None:
        self.assertEqual(ref_s_add_u32(0xFFFFFFFE, 1), (0xFFFFFFFF, 0))

    def test_carry_with_zero(self) -> None:
        self.assertEqual(ref_s_add_u32(0, 0), (0, 0))
        self.assertEqual(ref_s_add_u32(0xFFFFFFFF, 0), (0xFFFFFFFF, 0))

class TestSAddU32Translator(unittest.TestCase):
    def test_translator_s_add_u32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0xffffffff",
            "s_add_u32 s5, s4, 1",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_add_u32")
        for l in range(32):
            self.assertEqual(results[l], 0)

def ref_s_sub_u32(src0: int, src1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_sub_u32 -> (res, scc_borrow)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    res = (s0 - s1) & MASK32
    borrow = 1 if s0 < s1 else 0
    return res, borrow


class TestSSubU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        self.assertEqual(ref_s_sub_u32(500, 200), (300, 0))
        self.assertEqual(ref_s_sub_u32(100, 100), (0, 0))

    def test_borrow_out(self) -> None:
        self.assertEqual(ref_s_sub_u32(0, 1), (0xFFFFFFFF, 1))
        self.assertEqual(ref_s_sub_u32(10, 20), (0xFFFFFFFF - 9, 1))

class SSubU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int) -> None:
        self.src0 = src0
        self.src1 = src1

    def eval(self) -> tuple[int, int]:
        return ref_s_sub_u32(self.src0, self.src1)


class TestSSubU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SSubU32Expr(1, 2)
        self.assertEqual(expr.eval(), (0xFFFFFFFF, 1))

class TestSSubU32BorrowAndSCC(unittest.TestCase):
    def test_borrow_boundary(self) -> None:
        self.assertEqual(ref_s_sub_u32(0xFFFFFFFF, 0xFFFFFFFF), (0, 0))
        self.assertEqual(ref_s_sub_u32(0xFFFFFFFE, 0xFFFFFFFF), (0xFFFFFFFF, 1))

class TestSSubU32Translator(unittest.TestCase):
    def test_translator_s_sub_u32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0",
            "s_sub_u32 s5, s4, 1",
            "v_mov_b32_e32 v1, s5",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_sub_u32")
        for l in range(32):
            self.assertEqual(results[l], 0xFFFFFFFF)

def ref_s_addc_u32(src0: int, src1: int, cin: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_addc_u32 -> (res, cout)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    c = cin & 1
    total = s0 + s1 + c
    res = total & MASK32
    cout = 1 if total >= 0x100000000 else 0
    return res, cout


class TestSAddcU32Oracle(unittest.TestCase):
    def test_with_carry_in_zero(self) -> None:
        self.assertEqual(ref_s_addc_u32(10, 20, 0), (30, 0))
        self.assertEqual(ref_s_addc_u32(0xFFFFFFFF, 0, 0), (0xFFFFFFFF, 0))

    def test_with_carry_in_one(self) -> None:
        self.assertEqual(ref_s_addc_u32(10, 20, 1), (31, 0))
        self.assertEqual(ref_s_addc_u32(0xFFFFFFFF, 0, 1), (0, 1))
        self.assertEqual(ref_s_addc_u32(0xFFFFFFFF, 1, 1), (1, 1))

class SAddcU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int, cin: int) -> None:
        self.src0 = src0
        self.src1 = src1
        self.cin = cin

    def eval(self) -> tuple[int, int]:
        return ref_s_addc_u32(self.src0, self.src1, self.cin)


class TestSAddcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SAddcU32Expr(0x70000000, 0x10000000, 1)
        self.assertEqual(expr.eval(), (0x80000001, 0))

class TestSAddcU32ChainingAndPropagation(unittest.TestCase):
    def test_64bit_add_chain(self) -> None:
        # 64-bit addition: A + B
        a_lo, a_hi = 0xFFFFFFFF, 0x00000001
        b_lo, b_hi = 0x00000001, 0x00000002
        r_lo, cout = ref_s_add_u32(a_lo, b_lo)
        r_hi, cout2 = ref_s_addc_u32(a_hi, b_hi, cout)
        self.assertEqual((r_lo, r_hi), (0, 4))
        self.assertEqual(cout2, 0)

class TestSAddcU32Translator(unittest.TestCase):
    def test_translator_s_addc_u32_execution(self) -> None:
        body = [
            "s_mov_b32 s4, 0xffffffff",
            "s_add_u32 s5, s4, 1",        # produces carry = 1 in SCC
            "s_mov_b32 s6, 100",
            "s_addc_u32 s7, s6, 200",     # 100 + 200 + 1 = 301
            "v_mov_b32_e32 v1, s7",
            "buffer_store_dword v1, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_s_addc_u32")
        for l in range(32):
            self.assertEqual(results[l], 301)

def ref_s_subb_u32(src0: int, src1: int, bin_borrow: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_subb_u32 -> (res, bout)."""
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    b = bin_borrow & 1
    diff = s0 - s1 - b
    res = diff & MASK32
    bout = 1 if s0 < (s1 + b) else 0
    return res, bout


class TestSSubbU32Oracle(unittest.TestCase):
    def test_with_borrow_in_zero(self) -> None:
        self.assertEqual(ref_s_subb_u32(50, 20, 0), (30, 0))
        self.assertEqual(ref_s_subb_u32(10, 20, 0), (0xFFFFFFFF - 9, 1))

    def test_with_borrow_in_one(self) -> None:
        self.assertEqual(ref_s_subb_u32(50, 20, 1), (29, 0))
        self.assertEqual(ref_s_subb_u32(0, 0, 1), (0xFFFFFFFF, 1))

class SSubbU32Expr(ASTNode):
    def __init__(self, src0: int, src1: int, bin_borrow: int) -> None:
        self.src0 = src0
        self.src1 = src1
        self.bin_borrow = bin_borrow

    def eval(self) -> tuple[int, int]:
        return ref_s_subb_u32(self.src0, self.src1, self.bin_borrow)


class TestSSubbU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = SSubbU32Expr(10, 5, 1)
        self.assertEqual(expr.eval(), (4, 0))

class TestSSubbU32ChainingAndPropagation(unittest.TestCase):
    def test_64bit_sub_chain(self) -> None:
        # 64-bit subtraction: A - B
        a_lo, a_hi = 0x00000000, 0x00000005
        b_lo, b_hi = 0x00000001, 0x00000002
        r_lo, bout = ref_s_sub_u32(a_lo, b_lo)
        r_hi, bout2 = ref_s_subb_u32(a_hi, b_hi, bout)
        self.assertEqual((r_lo, r_hi), (0xFFFFFFFF, 2))
        self.assertEqual(bout2, 0)


if __name__ == "__main__":
    unittest.main()
