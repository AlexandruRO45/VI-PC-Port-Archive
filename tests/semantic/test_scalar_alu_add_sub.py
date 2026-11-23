"""Semantic interpreter tests for RDNA2 64-bit scalar addition and subtraction with carry.

Covers:
  * s_add_u32  (D = S0 + S1, SCC = unsigned carry-out)
  * s_addc_u32 (D = S0 + S1 + SCC, SCC = unsigned carry-out)
  * s_sub_u32  (D = S0 - S1, SCC = unsigned borrow-out)
  * s_subb_u32 (D = S0 - S1 - SCC, SCC = unsigned borrow-out)
  * Chained 64-bit scalar addition:
      s_add_u32  d_lo, s0_lo, s1_lo
      s_addc_u32 d_hi, s0_hi, s1_hi
  * Chained 64-bit scalar subtraction:
      s_sub_u32  d_lo, s0_lo, s1_lo
      s_subb_u32 d_hi, s0_hi, s1_hi
  * SPIR-V translation lowering equivalence (OpIAddCarry / OpISubBorrow lowering in tools/rdna2spirv.py).

Run directly:
    python -I tests/semantic/test_scalar_alu_add_sub.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN32 = 0x80000000
PUSH_VSHARP = [0x1000, 0, 0x1000, 0x31016FAC] + [0] * 28


def s32(v: int) -> int:
    """Interpret a 32-bit unsigned integer as signed two's complement."""
    u = v & MASK32
    return u - 0x100000000 if (u & SIGN32) else u


def ref_s_add_i32(s0: int, s1: int) -> tuple[int, int]:
    """Reference implementation of s_add_i32: D = (S0 + S1) mod 2^32, SCC = signed overflow."""
    a, b = s32(s0), s32(s1)
    full = a + b
    res = full & MASK32
    scc = 1 if (full < -0x80000000 or full > 0x7FFFFFFF) else 0
    return res, scc


def ref_s_sub_i32(s0: int, s1: int) -> tuple[int, int]:
    """Reference implementation of s_sub_i32: D = (S0 - S1) mod 2^32, SCC = signed overflow."""
    a, b = s32(s0), s32(s1)
    full = a - b
    res = full & MASK32
    scc = 1 if (full < -0x80000000 or full > 0x7FFFFFFF) else 0
    return res, scc


class ScalarALUInterpreter:
    """Deterministic semantic interpreter for RDNA2 scalar ALU add/sub operations.

    Tracks 32-bit SGPRs and the 1-bit SCC (Scalar Condition Code) register.
    """

    def __init__(self, num_sgprs=106):
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self):
        self.sgpr = [0] * len(self.sgpr)
        self.scc = 0

    @staticmethod
    def eval_s_add_u32(s0, s1):
        """Execute s_add_u32: D = (S0 + S1) mod 2^32; SCC = (S0 + S1 >= 2^32)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        full = s0 + s1
        d = full & MASK32
        scc = 1 if full >= (1 << 32) else 0
        return d, scc

    @staticmethod
    def eval_s_addc_u32(s0, s1, scc_in):
        """Execute s_addc_u32: D = (S0 + S1 + SCC) mod 2^32; SCC = (S0 + S1 + SCC >= 2^32)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        cin = 1 if scc_in else 0
        full = s0 + s1 + cin
        d = full & MASK32
        scc = 1 if full >= (1 << 32) else 0
        return d, scc

    @staticmethod
    def eval_s_sub_u32(s0, s1):
        """Execute s_sub_u32: D = (S0 - S1) mod 2^32; SCC = (S0 < S1)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        d = (s0 - s1) & MASK32
        scc = 1 if s0 < s1 else 0
        return d, scc

    @staticmethod
    def eval_s_subb_u32(s0, s1, scc_in):
        """Execute s_subb_u32: D = (S0 - S1 - SCC) mod 2^32; SCC = (S0 < S1 + SCC)."""
        s0 = s0 & MASK32
        s1 = s1 & MASK32
        bin_ = 1 if scc_in else 0
        d = (s0 - s1 - bin_) & MASK32
        scc = 1 if s0 < (s1 + bin_) else 0
        return d, scc

    def step(self, insn):
        """Execute an instruction dictionary with 'op', 'dst', 's0', 's1'."""
        op = insn["op"]
        s0 = self._resolve_val(insn["s0"])
        s1 = self._resolve_val(insn["s1"])
        dst_idx = self._resolve_dst(insn["dst"])

        if op == "s_add_u32":
            res, scc = self.eval_s_add_u32(s0, s1)
        elif op == "s_addc_u32":
            res, scc = self.eval_s_addc_u32(s0, s1, self.scc)
        elif op == "s_sub_u32":
            res, scc = self.eval_s_sub_u32(s0, s1)
        elif op == "s_subb_u32":
            res, scc = self.eval_s_subb_u32(s0, s1, self.scc)
        elif op == "s_mov_b32":
            res, scc = s0 & MASK32, self.scc
        else:
            raise NotImplementedError(f"Unsupported op: {op}")

        if dst_idx is not None:
            self.sgpr[dst_idx] = res
        if op in ("s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32"):
            self.scc = scc
        return res, self.scc

    def run_sequence(self, instructions):
        """Execute a list of instruction dictionaries sequentially."""
        last_res = 0
        for insn in instructions:
            last_res, _ = self.step(insn)
        return last_res

    def _resolve_val(self, operand):
        if isinstance(operand, int):
            return operand & MASK32
        if isinstance(operand, dict):
            if operand.get("k") == "imm":
                return operand["v"] & MASK32
            if operand.get("k") == "sgpr":
                return self.sgpr[operand["r"]] & MASK32
        if isinstance(operand, str) and operand.startswith("s"):
            return self.sgpr[int(operand[1:])] & MASK32
        raise ValueError(f"Unknown operand format: {operand}")

    def _resolve_dst(self, operand):
        if isinstance(operand, dict) and operand.get("k") == "sgpr":
            return operand["r"]
        if isinstance(operand, str) and operand.startswith("s"):
            return int(operand[1:])
        if isinstance(operand, int):
            return operand
        return None


def reference_add_64(a_lo, a_hi, b_lo, b_hi):
    """Reference implementation of 64-bit integer addition."""
    a64 = ((a_hi & MASK32) << 32) | (a_lo & MASK32)
    b64 = ((b_hi & MASK32) << 32) | (b_lo & MASK32)
    sum64 = a64 + b64
    carry_out = 1 if sum64 >= (1 << 64) else 0
    res = sum64 & MASK64
    return res & MASK32, (res >> 32) & MASK32, carry_out


def reference_sub_64(a_lo, a_hi, b_lo, b_hi):
    """Reference implementation of 64-bit integer subtraction."""
    a64 = ((a_hi & MASK32) << 32) | (a_lo & MASK32)
    b64 = ((b_hi & MASK32) << 32) | (b_lo & MASK32)
    borrow_out = 1 if a64 < b64 else 0
    diff64 = (a64 - b64) & MASK64
    return diff64 & MASK32, (diff64 >> 32) & MASK32, borrow_out


def spirv_lowering_model_add_sub(op, x, y, scc_in):
    """Mathematical emulation of the SPIR-V lowering emitted by tools/rdna2spirv.py.

    Emulates OpIAddCarry / OpISubBorrow with OpCompositeExtract and OpBitwiseOr.
    """
    x = x & MASK32
    y = y & MASK32
    sub = op.startswith("s_sub")

    if not sub:
        full1 = x + y
        r1 = full1 & MASK32
        c1 = 1 if full1 >= (1 << 32) else 0
    else:
        full1 = x - y
        r1 = full1 & MASK32
        c1 = 1 if x < y else 0

    r, carry = r1, c1

    if op in ("s_addc_u32", "s_subb_u32"):
        cin = 1 if scc_in else 0
        if not sub:
            full2 = r1 + cin
            r2 = full2 & MASK32
            c2 = 1 if full2 >= (1 << 32) else 0
        else:
            full2 = r1 - cin
            r2 = full2 & MASK32
            c2 = 1 if r1 < cin else 0
        cc = c1 | c2
        r, carry = r2, cc

    return r, carry


class TestScalarAddU32(unittest.TestCase):
    """Unit tests for s_add_u32 semantics and condition code updates."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_zero_addition(self):
        res, scc = self.interp.eval_s_add_u32(0, 0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_identity_operand(self):
        for val in (1, 42, 0x12345678, 0xFFFFFFFF):
            with self.subTest(val=val):
                res1, scc1 = self.interp.eval_s_add_u32(val, 0)
                res2, scc2 = self.interp.eval_s_add_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)
                self.assertEqual(res2, val)
                self.assertEqual(scc2, 0)

    def test_basic_addition_no_carry(self):
        res, scc = self.interp.eval_s_add_u32(100, 200)
        self.assertEqual(res, 300)
        self.assertEqual(scc, 0)

    def test_carry_boundary_single_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(scc, 1)

    def test_carry_boundary_just_below_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 0)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

    def test_carry_boundary_high_bit_addition(self):
        res, scc = self.interp.eval_s_add_u32(0x80000000, 0x80000000)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(scc, 1)

    def test_max_inputs_carry_overflow(self):
        res, scc = self.interp.eval_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

    def test_commutativity(self):
        pairs = [
            (0, 0),
            (12345, 67890),
            (0xFFFFFFFF, 2),
            (0x7FFFFFFF, 0x80000001),
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                r1, s1 = self.interp.eval_s_add_u32(a, b)
                r2, s2 = self.interp.eval_s_add_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_interpreter_step_and_state(self):
        self.interp.step({"op": "s_mov_b32", "dst": "s0", "s0": 10, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s1", "s0": 25, "s1": 0})
        res, scc = self.interp.step({"op": "s_add_u32", "dst": "s2", "s0": "s0", "s1": "s1"})
        self.assertEqual(res, 35)
        self.assertEqual(scc, 0)
        self.assertEqual(self.interp.sgpr[2], 35)
        self.assertEqual(self.interp.scc, 0)


class TestScalarAddcU32(unittest.TestCase):
    """Unit tests for s_addc_u32 semantics and carry-in propagation."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_carry_in_zero_matches_add(self):
        cases = [(0, 0), (10, 20), (0xFFFFFFFF, 1), (0xFFFFFFFF, 0xFFFFFFFF)]
        for s0, s1 in cases:
            with self.subTest(s0=s0, s1=s1):
                r_add, s_add = self.interp.eval_s_add_u32(s0, s1)
                r_addc, s_addc = self.interp.eval_s_addc_u32(s0, s1, 0)
                self.assertEqual(r_add, r_addc)
                self.assertEqual(s_add, s_addc)

    def test_carry_in_one_no_overflow(self):
        res, scc = self.interp.eval_s_addc_u32(100, 200, 1)
        self.assertEqual(res, 301)
        self.assertEqual(scc, 0)

    def test_carry_in_one_first_sum_overflows(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 5, 1)
        self.assertEqual(res, 5)
        self.assertEqual(scc, 1)

    def test_carry_in_one_second_sum_overflows(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 0, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

    def test_carry_in_one_split_overflow(self):
        res, scc = self.interp.eval_s_addc_u32(0x7FFFFFFF, 0x80000000, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 1)

    def test_carry_in_all_ones(self):
        res, scc = self.interp.eval_s_addc_u32(0xFFFFFFFF, 0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_carry_in_only_zero_operands(self):
        res, scc = self.interp.eval_s_addc_u32(0, 0, 1)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 0)


class TestScalarSubU32(unittest.TestCase):
    """Unit tests for s_sub_u32 semantics and condition code updates."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_zero_subtraction(self):
        res, scc = self.interp.eval_s_sub_u32(0, 0)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_identity_zero_subtrahend(self):
        for val in (1, 42, 0x12345678, 0xFFFFFFFF):
            with self.subTest(val=val):
                res, scc = self.interp.eval_s_sub_u32(val, 0)
                self.assertEqual(res, val)
                self.assertEqual(scc, 0)

    def test_identical_operands(self):
        for val in (0, 1, 0x7FFFFFFF, 0xFFFFFFFF):
            with self.subTest(val=val):
                res, scc = self.interp.eval_s_sub_u32(val, val)
                self.assertEqual(res, 0)
                self.assertEqual(scc, 0)

    def test_basic_subtraction_no_borrow(self):
        res, scc = self.interp.eval_s_sub_u32(300, 100)
        self.assertEqual(res, 200)
        self.assertEqual(scc, 0)

    def test_borrow_boundary_zero_minus_one(self):
        res, scc = self.interp.eval_s_sub_u32(0, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_underflow(self):
        res, scc = self.interp.eval_s_sub_u32(10, 20)
        self.assertEqual(res, 0xFFFFFFF6)
        self.assertEqual(scc, 1)

    def test_max_underflow(self):
        res, scc = self.interp.eval_s_sub_u32(0, 0xFFFFFFFF)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

    def test_max_min_difference(self):
        res, scc = self.interp.eval_s_sub_u32(0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFE)
        self.assertEqual(scc, 0)


class TestScalarSubbU32(unittest.TestCase):
    """Unit tests for s_subb_u32 semantics and borrow-in propagation."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_borrow_in_zero_matches_sub(self):
        cases = [(0, 0), (30, 10), (0, 1), (10, 20), (0, 0xFFFFFFFF)]
        for s0, s1 in cases:
            with self.subTest(s0=s0, s1=s1):
                r_sub, s_sub = self.interp.eval_s_sub_u32(s0, s1)
                r_subb, s_subb = self.interp.eval_s_subb_u32(s0, s1, 0)
                self.assertEqual(r_sub, r_subb)
                self.assertEqual(s_sub, s_subb)

    def test_borrow_in_one_no_underflow(self):
        res, scc = self.interp.eval_s_subb_u32(50, 20, 1)
        self.assertEqual(res, 29)
        self.assertEqual(scc, 0)

    def test_borrow_in_one_exact_zero_result(self):
        res, scc = self.interp.eval_s_subb_u32(10, 9, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_borrow_in_one_first_sub_borrows(self):
        res, scc = self.interp.eval_s_subb_u32(0, 5, 1)
        self.assertEqual(res, 0xFFFFFFFA)
        self.assertEqual(scc, 1)

    def test_borrow_in_one_second_sub_borrows(self):
        res, scc = self.interp.eval_s_subb_u32(5, 5, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_in_one_zero_minus_zero(self):
        res, scc = self.interp.eval_s_subb_u32(0, 0, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_borrow_in_all_ones(self):
        res, scc = self.interp.eval_s_subb_u32(0xFFFFFFFF, 0xFFFFFFFF, 1)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)


class Test64BitScalarArithmetic(unittest.TestCase):
    """Semantic tests for chained 64-bit scalar addition and subtraction with carry/borrow."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def run_add64(self, a_lo, a_hi, b_lo, b_hi):
        seq = [
            {"op": "s_mov_b32", "dst": "s0", "s0": a_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s1", "s0": a_hi, "s1": 0},
            {"op": "s_mov_b32", "dst": "s2", "s0": b_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s3", "s0": b_hi, "s1": 0},
            {"op": "s_add_u32", "dst": "s4", "s0": "s0", "s1": "s2"},
            {"op": "s_addc_u32", "dst": "s5", "s0": "s1", "s1": "s3"},
        ]
        self.interp.run_sequence(seq)
        return self.interp.sgpr[4], self.interp.sgpr[5], self.interp.scc

    def run_sub64(self, a_lo, a_hi, b_lo, b_hi):
        seq = [
            {"op": "s_mov_b32", "dst": "s0", "s0": a_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s1", "s0": a_hi, "s1": 0},
            {"op": "s_mov_b32", "dst": "s2", "s0": b_lo, "s1": 0},
            {"op": "s_mov_b32", "dst": "s3", "s0": b_hi, "s1": 0},
            {"op": "s_sub_u32", "dst": "s4", "s0": "s0", "s1": "s2"},
            {"op": "s_subb_u32", "dst": "s5", "s0": "s1", "s1": "s3"},
        ]
        self.interp.run_sequence(seq)
        return self.interp.sgpr[4], self.interp.sgpr[5], self.interp.scc

    def test_add64_without_carry(self):
        r_lo, r_hi, scc = self.run_add64(0x00000002, 0x00000001, 0x00000004, 0x00000003)
        self.assertEqual(r_lo, 0x00000006)
        self.assertEqual(r_hi, 0x00000004)
        self.assertEqual(scc, 0)

    def test_add64_low_to_high_carry_propagation(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0x00000001, 0x00000001, 0x00000002)
        self.assertEqual(r_lo, 0x00000000)
        self.assertEqual(r_hi, 0x00000004)
        self.assertEqual(scc, 0)

    def test_add64_full_chain_overflow(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0x00000000)
        self.assertEqual(r_lo, 0x00000000)
        self.assertEqual(r_hi, 0x00000000)
        self.assertEqual(scc, 1)

    def test_add64_max_unsigned(self):
        r_lo, r_hi, scc = self.run_add64(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(r_lo, 0xFFFFFFFE)
        self.assertEqual(r_hi, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_sub64_without_borrow(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000005, 0x00000008, 0x00000002, 0x00000003)
        self.assertEqual(r_lo, 0x00000003)
        self.assertEqual(r_hi, 0x00000005)
        self.assertEqual(scc, 0)

    def test_sub64_low_to_high_borrow_propagation(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000000, 0x00000005, 0x00000001, 0x00000002)
        self.assertEqual(r_lo, 0xFFFFFFFF)
        self.assertEqual(r_hi, 0x00000002)
        self.assertEqual(scc, 0)

    def test_sub64_full_chain_underflow(self):
        r_lo, r_hi, scc = self.run_sub64(0x00000000, 0x00000000, 0x00000001, 0x00000000)
        self.assertEqual(r_lo, 0xFFFFFFFF)
        self.assertEqual(r_hi, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_add64_sub64_roundtrip(self):
        test_vectors = [
            (0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321),
            (0xFFFFFFFF, 0x00000000, 0x00000001, 0x00000000),
            (0x00000000, 0x80000000, 0x00000000, 0x40000000),
            (0xCAFEBABE, 0xDEADBEEF, 0x11111111, 0x22222222),
        ]
        for a_lo, a_hi, b_lo, b_hi in test_vectors:
            with self.subTest(a_lo=a_lo, a_hi=a_hi, b_lo=b_lo, b_hi=b_hi):
                sum_lo, sum_hi, _ = self.run_add64(a_lo, a_hi, b_lo, b_hi)
                res_lo, res_hi, _ = self.run_sub64(sum_lo, sum_hi, b_lo, b_hi)
                self.assertEqual(res_lo, a_lo)
                self.assertEqual(res_hi, a_hi)

    def test_deterministic_pseudorandom_64bit_sweep(self):
        rng = random.Random(42)
        for i in range(250):
            a_lo = rng.getrandbits(32)
            a_hi = rng.getrandbits(32)
            b_lo = rng.getrandbits(32)
            b_hi = rng.getrandbits(32)

            exp_add_lo, exp_add_hi, exp_add_c = reference_add_64(a_lo, a_hi, b_lo, b_hi)
            got_add_lo, got_add_hi, got_add_c = self.run_add64(a_lo, a_hi, b_lo, b_hi)
            self.assertEqual(got_add_lo, exp_add_lo, f"Add iter {i} lo mismatch")
            self.assertEqual(got_add_hi, exp_add_hi, f"Add iter {i} hi mismatch")
            self.assertEqual(got_add_c, exp_add_c, f"Add iter {i} carry mismatch")

            exp_sub_lo, exp_sub_hi, exp_sub_b = reference_sub_64(a_lo, a_hi, b_lo, b_hi)
            got_sub_lo, got_sub_hi, got_sub_b = self.run_sub64(a_lo, a_hi, b_lo, b_hi)
            self.assertEqual(got_sub_lo, exp_sub_lo, f"Sub iter {i} lo mismatch")
            self.assertEqual(got_sub_hi, exp_sub_hi, f"Sub iter {i} hi mismatch")
            self.assertEqual(got_sub_b, exp_sub_b, f"Sub iter {i} borrow mismatch")


class TestRdna2SpirvLoweringSemantics(unittest.TestCase):
    """Tests cross-checking RDNA2 scalar ALU semantics against the SPIR-V lowering emitted by rdna2spirv.py."""

    def test_gen_spirv_emission_add_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_add_u32",
            "args": [{"k": "sgpr", "r": 4}, {"k": "imm", "v": 0xFFFFFFFF}, {"k": "imm", "v": 1}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertIn("OpIAddCarry %umul", body)
        self.assertIn("OpCompositeExtract %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_gen_spirv_emission_addc_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_addc_u32",
            "args": [{"k": "sgpr", "r": 5}, {"k": "imm", "v": 10}, {"k": "imm", "v": 20}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertEqual(body.count("OpIAddCarry %umul"), 2)
        self.assertIn("OpLoad %uint %scc", body)
        self.assertIn("OpBitwiseOr %uint", body)

    def test_gen_spirv_emission_sub_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_sub_u32",
            "args": [{"k": "sgpr", "r": 6}, {"k": "imm", "v": 50}, {"k": "imm", "v": 30}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertIn("OpISubBorrow %umul", body)
        self.assertIn("OpCompositeExtract %uint", body)
        self.assertIn("OpStore %scc", body)

    def test_gen_spirv_emission_subb_u32(self):
        gen = rdna2spirv.Gen("ps")
        insn = {
            "op": "s_subb_u32",
            "args": [{"k": "sgpr", "r": 7}, {"k": "imm", "v": 50}, {"k": "imm", "v": 30}],
            "mods": [],
        }
        gen._insn(insn)
        body = "\n".join(gen.body)
        self.assertEqual(body.count("OpISubBorrow %umul"), 2)
        self.assertIn("OpLoad %uint %scc", body)
        self.assertIn("OpBitwiseOr %uint", body)

    def test_gen_spirv_emission_add_sub_addk_i32(self):
        for op, spv_op in (("s_add_i32", "OpIAdd %uint"), ("s_sub_i32", "OpISub %uint")):
            gen = rdna2spirv.Gen("ps")
            gen._insn({
                "op": op,
                "args": [{"k": "sgpr", "r": 4}, {"k": "sgpr", "r": 5}, {"k": "sgpr", "r": 6}],
                "mods": [],
            })
            body = "\n".join(gen.body)
            self.assertIn(spv_op, body)
            self.assertIn("OpBitwiseXor %uint", body)
            self.assertIn("OpBitwiseAnd %uint", body)
            self.assertIn("OpShiftRightLogical %uint", body)
            self.assertIn("OpStore %scc", body)

        gen_k = rdna2spirv.Gen("ps")
        gen_k._insn({
            "op": "s_addk_i32",
            "args": [{"k": "sgpr", "r": 4}, {"k": "imm", "v": 0xFFFE}],
            "mods": [],
        })
        body_k = "\n".join(gen_k.body)
        self.assertIn("OpIAdd %uint", body_k)
        self.assertIn("OpBitwiseXor %uint", body_k)
        self.assertIn("OpStore %scc", body_k)

    def test_spirv_model_equivalence_sweep(self):
        rng = random.Random(1337)
        for _ in range(200):
            x = rng.getrandbits(32)
            y = rng.getrandbits(32)
            cin = rng.choice([0, 1])

            # s_add_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_add_u32(x, y)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_add_u32", x, y, 0)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_addc_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_addc_u32(x, y, cin)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_addc_u32", x, y, cin)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_sub_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_sub_u32(x, y)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_sub_u32", x, y, 0)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)

            # s_subb_u32
            isa_res, isa_scc = ScalarALUInterpreter.eval_s_subb_u32(x, y, cin)
            spv_res, spv_scc = spirv_lowering_model_add_sub("s_subb_u32", x, y, cin)
            self.assertEqual(isa_res, spv_res)
            self.assertEqual(isa_scc, spv_scc)


class TestAddressArithmeticUseCases(unittest.TestCase):
    """Simulates real-world 64-bit scalar address arithmetic patterns seen in shaders."""

    def setUp(self):
        self.interp = ScalarALUInterpreter()

    def test_s_getpc_b64_forward_displacement(self):
        pc_base = 0x00007FF8_10000000
        offset = 0x00000000_00002A40

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_add_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_addc_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base + offset)

    def test_s_getpc_b64_forward_displacement_spanning_4gb_boundary(self):
        pc_base = 0x00007FF8_FFFF8000
        offset = 0x00000000_00010000

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_add_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_addc_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base + offset)

    def test_backward_loop_edge_displacement(self):
        pc_base = 0x00007FF8_00001000
        offset = 0x00000000_00000500

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_sub_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_subb_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base - offset)

    def test_backward_loop_edge_spanning_4gb_boundary(self):
        pc_base = 0x00007FF8_00002000
        offset = 0x00000000_00008000

        pc_lo = pc_base & MASK32
        pc_hi = (pc_base >> 32) & MASK32
        off_lo = offset & MASK32
        off_hi = (offset >> 32) & MASK32

        self.interp.step({"op": "s_mov_b32", "dst": "s4", "s0": pc_lo, "s1": 0})
        self.interp.step({"op": "s_mov_b32", "dst": "s5", "s0": pc_hi, "s1": 0})
        self.interp.step({"op": "s_sub_u32", "dst": "s4", "s0": "s4", "s1": off_lo})
        self.interp.step({"op": "s_subb_u32", "dst": "s5", "s0": "s5", "s1": off_hi})

        final_addr = ((self.interp.sgpr[5] & MASK32) << 32) | (self.interp.sgpr[4] & MASK32)
        self.assertEqual(final_addr, pc_base - offset)


def translate_ps_add_sub(lines: list[str]) -> str:
    """Translate a hand-written pixel shader listing exporting v0..v3 to mrt0."""
    return translate_lines(
        "ps_salu_add_sub",
        [*lines, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


class EndToEndScalarAddSubTranslatorTests(unittest.TestCase):
    """End-to-end rdna2spirv.translate + spirv_mini_interp execution for scalar add/sub ops."""

    def test_runtime_s_add_u32_and_s_sub_u32_carry_borrow_and_scc_transitions(self) -> None:
        # Both operands (s4, s5) are loaded from buffer memory at runtime so neither is in sgpr_imm,
        # exercising 0->1, 0->0, 1->0, and 1->1 SCC unsigned carry/borrow transitions.
        spv = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_add_u32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when carry = 0)
            "s_add_u32 s8, s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_sub_u32 s10, s4, s5",
            "s_cselect_b32 s11, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when borrow = 0)
            "s_sub_u32 s12, s4, s5",
            "s_cselect_b32 s13, 1, 0",
            # Pack add SCC transitions in v1 (bit0: from 0, bit1: from 1) and sub SCC transitions in v3
            "s_lshl_b32 s9, s9, 1",
            "s_or_b32 s7, s7, s9",
            "s_lshl_b32 s13, s13, 1",
            "s_or_b32 s11, s11, s13",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])

        pairs = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (100, 200),
            (300, 100),
            (0x7FFFFFFF, 0),
            (0x7FFFFFFF, 1),
            (0x7FFFFFFF, 0x7FFFFFFF),
            (0x7FFFFFFF, 0x80000000),
            (0x7FFFFFFF, 0x80000001),
            (0x80000000, 0),
            (0x80000000, 1),
            (0x80000000, 0x7FFFFFFF),
            (0x80000000, 0x80000000),
            (0xFFFFFFFE, 1),
            (0xFFFFFFFE, 2),
            (0xFFFFFFFE, 0xFFFFFFFF),
            (0xFFFFFFFF, 0),
            (0xFFFFFFFF, 1),
            (0, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xDEADBEEF, 0xCAFEBABE),
            (0xCAFEBABE, 0xDEADBEEF),
            (0x12345678, 0x87654321),
        ]
        for s0, s1 in pairs:
            mem = {0x1000: s0 & MASK32, 0x1004: s1 & MASK32}
            exp_add, exp_add_c = ScalarALUInterpreter.eval_s_add_u32(s0, s1)
            exp_sub, exp_sub_b = ScalarALUInterpreter.eval_s_sub_u32(s0, s1)
            exp_add_mask = (exp_add_c << 1) | exp_add_c
            exp_sub_mask = (exp_sub_b << 1) | exp_sub_b
            self.assertEqual(
                mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem),
                [exp_add, exp_add_mask, exp_sub, exp_sub_mask],
                f"s0={s0:#010x}, s1={s1:#010x}",
            )

    def test_runtime_s_addc_u32_and_s_subb_u32_carry_and_borrow_in_propagation(self) -> None:
        # Both operands (s4, s5) are loaded from buffer memory at runtime; executes s_addc_u32 and
        # s_subb_u32 under both SCC_in = 0 and SCC_in = 1 and exports (res, scc_out).
        spv_addc = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC_in = 0
            "s_addc_u32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC_in = 1
            "s_addc_u32 s8, s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s9",
        ])
        spv_subb = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC_in = 0
            "s_subb_u32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC_in = 1
            "s_subb_u32 s8, s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s9",
        ])

        cases = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (5, 5),
            (10, 9),
            (100, 200),
            (50, 20),
            (0, 5),
            (0x7FFFFFFF, 0x80000000),     # addc: c1 = 0, c2 = 1 when SCC_in = 1 -> carry_out = 1
            (0x80000000, 0x7FFFFFFF),
            (0x80000000, 0x80000000),     # addc: c1 = 1, c2 = 0 -> carry_out = 1
            (0xFFFFFFFF, 0),              # addc: c1 = 0, c2 = 1 when SCC_in = 1 -> carry_out = 1
            (0, 0xFFFFFFFF),              # subb: c1 = 1, r1 = 1, c2 = 0 when SCC_in = 1 -> res = 0, borrow = 1
            (0xFFFFFFFF, 1),
            (0xFFFFFFFE, 1),              # addc: c1 = 0, c2 = 1 when SCC_in = 1 -> res = 0, carry_out = 1
            (0xFFFFFFFF, 5),
            (0xFFFFFFFF, 0xFFFFFFFF),     # addc: c1 = 1; subb: c1 = 0, c2 = 1 when SCC_in = 1
            (0xDEADBEEF, 0xCAFEBABE),
        ]
        for s0, s1 in cases:
            mem = {0x1000: s0 & MASK32, 0x1004: s1 & MASK32}
            add0, add0_c = ScalarALUInterpreter.eval_s_addc_u32(s0, s1, 0)
            add1, add1_c = ScalarALUInterpreter.eval_s_addc_u32(s0, s1, 1)
            self.assertEqual(
                mrt0_words(spv_addc, push_constants=PUSH_VSHARP, memory=mem),
                [add0, add0_c, add1, add1_c],
                f"s_addc_u32 s0={s0:#010x}, s1={s1:#010x}",
            )
            sub0, sub0_b = ScalarALUInterpreter.eval_s_subb_u32(s0, s1, 0)
            sub1, sub1_b = ScalarALUInterpreter.eval_s_subb_u32(s0, s1, 1)
            self.assertEqual(
                mrt0_words(spv_subb, push_constants=PUSH_VSHARP, memory=mem),
                [sub0, sub0_b, sub1, sub1_b],
                f"s_subb_u32 s0={s0:#010x}, s1={s1:#010x}",
            )

    def test_runtime_chained_64bit_add_and_sub_with_carry_borrow(self) -> None:
        # Chained 64-bit scalar addition (s_add_u32 + s_addc_u32) and subtraction (s_sub_u32 + s_subb_u32)
        # on runtime-loaded 64-bit pairs s[4:5] and s[6:7].
        spv_add64 = translate_ps_add_sub([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_add_u32 s8, s4, s6",
            "s_cselect_b32 s10, 1, 0",    # low carry-out
            "s_addc_u32 s9, s5, s7",
            "s_cselect_b32 s11, 1, 0",    # 64-bit carry-out
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])
        spv_sub64 = translate_ps_add_sub([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_sub_u32 s8, s4, s6",
            "s_cselect_b32 s10, 1, 0",    # low borrow-out
            "s_subb_u32 s9, s5, s7",
            "s_cselect_b32 s11, 1, 0",    # 64-bit borrow-out
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])

        vectors = [
            (0x00000002, 0x00000001, 0x00000004, 0x00000003),
            (0xFFFFFFFF, 0x00000001, 0x00000001, 0x00000002),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x00000005, 0x00000008, 0x00000002, 0x00000003),
            (0x00000000, 0x00000005, 0x00000001, 0x00000002),
            (0x00000000, 0x00000000, 0x00000001, 0x00000000),
            (0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFF8000, 0x00007FF8, 0x00010000, 0x00000000),
            (0x00002000, 0x00007FF8, 0x00008000, 0x00000000),
            (0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321),
        ]
        rng = random.Random(20261123)
        for _ in range(30):
            vectors.append((
                rng.getrandbits(32),
                rng.getrandbits(32),
                rng.getrandbits(32),
                rng.getrandbits(32),
            ))

        for a_lo, a_hi, b_lo, b_hi in vectors:
            mem = {0x1000: a_lo, 0x1004: a_hi, 0x1008: b_lo, 0x100C: b_hi}
            exp_add_lo, exp_add_hi, exp_add_c = reference_add_64(a_lo, a_hi, b_lo, b_hi)
            _, exp_lo_c = ScalarALUInterpreter.eval_s_add_u32(a_lo, b_lo)
            got_add = mrt0_words(spv_add64, push_constants=PUSH_VSHARP, memory=mem)
            self.assertEqual(got_add, [exp_add_lo, exp_add_hi, exp_lo_c, exp_add_c])

            exp_sub_lo, exp_sub_hi, exp_sub_b = reference_sub_64(a_lo, a_hi, b_lo, b_hi)
            _, exp_lo_b = ScalarALUInterpreter.eval_s_sub_u32(a_lo, b_lo)
            got_sub = mrt0_words(spv_sub64, push_constants=PUSH_VSHARP, memory=mem)
            self.assertEqual(got_sub, [exp_sub_lo, exp_sub_hi, exp_lo_b, exp_sub_b])

            # Round-trip (a + b) - b == a through the translated 64-bit sub shader
            mem_rt = {0x1000: got_add[0], 0x1004: got_add[1], 0x1008: b_lo, 0x100C: b_hi}
            got_rt = mrt0_words(spv_sub64, push_constants=PUSH_VSHARP, memory=mem_rt)
            self.assertEqual((got_rt[0], got_rt[1]), (a_lo, a_hi))

    def test_runtime_s_add_i32_s_sub_i32_and_s_addk_i32_signed_overflow(self) -> None:
        # Runtime-loaded s_add_i32 and s_sub_i32 set SCC on signed two's-complement overflow
        # (not unsigned carry/borrow), and s_addk_i32 sign-extends its 16-bit immediate.
        spv = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_add_i32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when overflow = 0)
            "s_add_i32 s8, s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_sub_i32 s10, s4, s5",
            "s_cselect_b32 s11, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when overflow = 0)
            "s_sub_i32 s12, s4, s5",
            "s_cselect_b32 s13, 1, 0",
            "s_lshl_b32 s9, s9, 1",
            "s_or_b32 s7, s7, s9",
            "s_lshl_b32 s13, s13, 1",
            "s_or_b32 s11, s11, s13",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])

        signed_cases = [
            (0, 0),
            (100, 200),
            (0xFFFFFFFF, 1),              # -1 + 1 = 0 (unsigned carry = 1, signed overflow = 0!)
            (0xFFFFFFFF, 0xFFFFFFFF),     # -1 + -1 = -2 (unsigned carry = 1, signed overflow = 0!)
            (0x7FFFFFFF, 0),              # INT32_MAX + 0 = INT32_MAX (no overflow)
            (0x7FFFFFFF, 1),              # INT32_MAX + 1 = 0x80000000 (signed overflow = 1!)
            (0x7FFFFFFF, 0x7FFFFFFF),     # positive + positive -> negative (signed overflow = 1)
            (0x7FFFFFFF, 0xFFFFFFFF),     # INT32_MAX + (-1) -> no add overflow; INT32_MAX - (-1) -> sub overflow = 1!
            (0x80000000, 0),
            (0x80000000, 1),              # INT32_MIN + 1 -> no add overflow; INT32_MIN - 1 -> sub overflow = 1!
            (0x80000000, 0xFFFFFFFF),     # INT32_MIN + (-1) -> add overflow = 1; INT32_MIN - (-1) -> no sub overflow
            (0x80000000, 0x80000000),     # INT32_MIN + INT32_MIN -> add overflow = 1; sub = 0, no sub overflow
            (0xC0000000, 0xC0000000),     # -2^30 + -2^30 = -2^31 (0x80000000, still in range -> overflow = 0)
            (0xC0000000, 0xBFFFFFFF),     # -2^30 + (-2^30 - 1) -> add overflow = 1
            (0x40000000, 0x40000000),     # 2^30 + 2^30 = 2^31 -> add overflow = 1
            (0x40000000, 0x3FFFFFFF),     # 2^30 + (2^30 - 1) = INT32_MAX -> overflow = 0
        ]
        for s0, s1 in signed_cases:
            mem = {0x1000: s0 & MASK32, 0x1004: s1 & MASK32}
            exp_add, exp_add_ov = ref_s_add_i32(s0, s1)
            exp_sub, exp_sub_ov = ref_s_sub_i32(s0, s1)
            self.assertEqual(
                mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem),
                [exp_add, (exp_add_ov << 1) | exp_add_ov, exp_sub, (exp_sub_ov << 1) | exp_sub_ov],
                f"s0={s0:#010x}, s1={s1:#010x}",
            )

        # SOPK s_addk_i32 with positive (+1, +32767) and sign-extended negative (-1 = 0xFFFF, -32768 = 0x8000) simm16
        spv_addk = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 s6, s4",
            "s_addk_i32 s6, 0x0001",      # +1
            "s_cselect_b32 s7, 1, 0",
            "s_mov_b32 s8, s5",
            "s_addk_i32 s8, 0xffff",      # -1 (sign-extended from 16-bit immediate)
            "s_cselect_b32 s9, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s9",
        ])
        for s4, s5 in ((0x7FFFFFFF, 0x80000000), (0x7FFFFFFE, 0x80000001), (0xFFFFFFFF, 0)):
            mem_k = {0x1000: s4, 0x1004: s5}
            e_add1, e_ov1 = ref_s_add_i32(s4, 1)
            e_sub1, e_ov2 = ref_s_add_i32(s5, 0xFFFFFFFF)
            self.assertEqual(
                mrt0_words(spv_addk, push_constants=PUSH_VSHARP, memory=mem_k),
                [e_add1, e_ov1, e_sub1, e_ov2],
            )

    def test_runtime_in_place_aliasing_and_null_destination(self) -> None:
        # Verifies D == S0, D == S1, D == S0 == S1 in-place register aliasing and `null` destinations
        # that update SCC while leaving SGPRs untouched.
        spv = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            # 1. In-place s_add_u32 (D == S0) + s_addc_u32 (D == S1)
            "s_mov_b32 s6, s4",
            "s_mov_b32 s7, s5",
            "s_add_u32 s6, s6, s5",
            "s_addc_u32 s7, s4, s7",
            # 2. In-place s_sub_u32 (D == S0 == S1) -> 0, SCC = 0
            "s_mov_b32 s8, s4",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1
            "s_sub_u32 s8, s8, s8",
            "s_cselect_b32 s9, 1, 0",
            # 3. `null` destination on s_add_u32, s_sub_u32, s_add_i32, s_sub_i32 updates SCC only
            "s_cmp_eq_u32 0, 1",
            "s_add_u32 null, s4, s5",
            "s_cselect_b32 s10, 1, 0",
            "s_cmp_eq_u32 0, 1",
            "s_sub_u32 null, s5, s4",
            "s_cselect_b32 s11, 1, 0",
            "s_cmp_eq_u32 0, 1",
            "s_add_i32 null, s4, s5",
            "s_cselect_b32 s12, 1, 0",
            "s_lshl_b32 s11, s11, 1",
            "s_lshl_b32 s12, s12, 2",
            "s_or_b32 s10, s10, s11",
            "s_or_b32 s10, s10, s12",
            "s_lshl_b32 s9, s9, 3",
            "s_or_b32 s10, s10, s9",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s8",
            "v_mov_b32 v3, s10",
        ])
        # s4 = 0x7FFFFFFF, s5 = 0x80000002:
        # s6 = 0x7FFFFFFF + 0x80000002 = 0x00000001 (carry = 1)
        # s7 = 0x7FFFFFFF + 0x80000002 + 1 = 0x00000002
        # s8 = s4 - s4 = 0, s9 (SCC) = 0
        # null s_add_u32(s4, s5): carry = 1 (bit 0)
        # null s_sub_u32(s5, s4): 0x80000002 >= 0x7FFFFFFF -> borrow = 0 (bit 1)
        # null s_add_i32(s4, s5): +2147483647 + (-2147483646) = +1 -> signed overflow = 0 (bit 2)
        mem = {0x1000: 0x7FFFFFFF, 0x1004: 0x80000002}
        self.assertEqual(
            mrt0_words(spv, push_constants=PUSH_VSHARP, memory=mem),
            [0x00000001, 0x00000002, 0x00000000, 0b0001],
        )


class TranslatedScalarAddSubMutationTests(unittest.TestCase):
    """Negative controls verifying that sabotaging emitted scalar add/sub SPIR-V fails execution."""

    def test_translated_add_sub_spirv_mutations_caught(self) -> None:
        # 1. s_addc_u32: swapping OpIAddCarry -> OpISubBorrow, replacing OpBitwiseOr -> OpBitwiseAnd
        #    (which drops carry when only one of c1, c2 is 1), or redirecting OpStore %scc must fail.
        spv_addc = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 1, 1",          # seed SCC_in = 1
            "s_addc_u32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        # 0xFFFFFFFE + 1 + 1 -> r1 = 0xFFFFFFFF (c1 = 0), r2 = 0 (c2 = 1), carry = c1 | c2 = 1
        mem_addc = {0x1000: 0xFFFFFFFE, 0x1004: 1}
        self.assertEqual(mrt0_words(spv_addc, push_constants=PUSH_VSHARP, memory=mem_addc)[:2], [0, 1])

        bad_op = spv_addc.replace("OpIAddCarry %umul", "OpISubBorrow %umul", 1)
        self.assertNotEqual(bad_op, spv_addc)
        self.assertNotEqual(mrt0_words(bad_op, push_constants=PUSH_VSHARP, memory=mem_addc)[:2], [0, 1])

        bad_or = spv_addc.replace("OpBitwiseOr %uint", "OpBitwiseAnd %uint", 1)
        self.assertNotEqual(bad_or, spv_addc)
        self.assertNotEqual(mrt0_words(bad_or, push_constants=PUSH_VSHARP, memory=mem_addc)[:2], [0, 1])

        bad_scc = spv_addc.replace("OpStore %scc", "OpStore %vcc_hi")
        self.assertNotEqual(bad_scc, spv_addc)
        self.assertNotEqual(mrt0_words(bad_scc, push_constants=PUSH_VSHARP, memory=mem_addc)[:2], [0, 1])

        # 2. s_add_i32: sabotaging the sign-bit shift (OpShiftRightLogical %uint ... 31) in scc_add_overflow
        spv_addi = translate_ps_add_sub([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",
            "s_add_i32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        mem_addi = {0x1000: 0x7FFFFFFF, 0x1004: 1}
        self.assertEqual(mrt0_words(spv_addi, push_constants=PUSH_VSHARP, memory=mem_addi)[:2], [0x80000000, 1])

        bad_ov_xor = spv_addi.replace("OpBitwiseXor %uint", "OpBitwiseAnd %uint", 1)
        self.assertNotEqual(bad_ov_xor, spv_addi)
        self.assertNotEqual(
            mrt0_words(bad_ov_xor, push_constants=PUSH_VSHARP, memory=mem_addi)[:2],
            [0x80000000, 1],
        )


if __name__ == "__main__":
    unittest.main()
