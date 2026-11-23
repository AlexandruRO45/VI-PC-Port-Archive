"""Semantic interpreter tests for RDNA2 scalar logical and arithmetic shifts.

Instructions tested:
  - s_lshl_b64 D, S0, S1: Logical Shift Left 64-bit
  - s_lshr_b64 D, S0, S1: Logical Shift Right 64-bit (zero extended)
  - s_ashr_i64 D, S0, S1: Arithmetic Shift Right 64-bit (sign extended)

RDNA2 ISA rules (Scalar ALU Operations / SOP2):
  - D is an even-aligned 64-bit SGPR pair s[d:d+1].
  - S0 is a 64-bit scalar operand: SGPR pair s[s:s+1] or 64-bit immediate constant.
  - S1 is a 32-bit shift count operand: SGPR or immediate constant. Only bits [5:0]
    (S1 & 63) are used by hardware; upper bits [31:6] are ignored.
  - Logical shifts (s_lshl_b64, s_lshr_b64) treat S0 as an unsigned 64-bit integer,
    shifting in zeroes from the right or left.
  - Arithmetic shift (s_ashr_i64) treats S0 as a signed two's complement 64-bit integer,
    replicating bit 63 into the vacated upper bit positions.
  - Condition code SCC: SCC = 1 if D != 0 else 0 (set to 1 if the 64-bit result is
    non-zero; cleared to 0 if the 64-bit result is all zeroes).

Translator integration (rdna2spirv):
  - s_lshl_b64, s_lshr_b64 and s_ashr_i64 pack SGPR pairs into %ulong, mask the shift count with 63,
    perform OpShiftLeftLogical / OpShiftRightLogical / OpShiftRightArithmetic, unpack to low and high dwords,
    and update %scc via OpSelect.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN64 = 1 << 63
PUSH_VSHARP = [0x1000, 0, 0x1000, 0x31016FAC] + [0] * 28


def u64(v):
    """Normalize integer to 64-bit unsigned."""
    return v & MASK64


def s64(v):
    """Interpret 64-bit unsigned integer as signed two's complement."""
    v &= MASK64
    return v - (1 << 64) if v & SIGN64 else v


def split_dwords(v64):
    """Split 64-bit uint into (low_u32, high_u32)."""
    v = v64 & MASK64
    return v & MASK32, (v >> 32) & MASK32


def join_dwords(lo, hi):
    """Join (low_u32, high_u32) into a 64-bit uint."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def ref_s_lshl_b64(src0, src1):
    """Reference implementation of s_lshl_b64: D = src0 << (src1 & 63), SCC = D != 0."""
    sh = src1 & 63
    res = (u64(src0) << sh) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def ref_s_lshr_b64(src0, src1):
    """Reference implementation of s_lshr_b64: D = (unsigned)src0 >> (src1 & 63), SCC = D != 0."""
    sh = src1 & 63
    res = (u64(src0) >> sh) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


def ref_s_ashr_i64(src0, src1):
    """Reference implementation of s_ashr_i64: D = (signed)src0 >> (src1 & 63), SCC = D != 0."""
    sh = src1 & 63
    signed_src = s64(src0)
    res = (signed_src >> sh) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc


class ScalarAluInterpreter:
    """Minimal semantic interpreter for RDNA2 scalar ALU 64-bit shift operations."""

    def __init__(self, num_sgprs=106):
        self.num_sgprs = num_sgprs
        self.sgpr = [0] * num_sgprs
        self.scc = 0

    def reset(self):
        self.sgpr = [0] * self.num_sgprs
        self.scc = 0

    def get_sgpr_u32(self, r):
        return self.sgpr[r] & MASK32

    def set_sgpr_u32(self, r, val):
        self.sgpr[r] = val & MASK32

    def get_sgpr_u64(self, r):
        lo = self.get_sgpr_u32(r)
        hi = self.get_sgpr_u32(r + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, r, val64):
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(r, lo)
        self.set_sgpr_u32(r + 1, hi)

    def resolve_src64(self, opnd):
        """Resolve a 64-bit source operand (SGPR pair or immediate constant)."""
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "sgpr":
                r = opnd["r"]
                return self.get_sgpr_u64(r)
            if k == "imm":
                return opnd["v"] & MASK64
            raise ValueError(f"unknown 64-bit operand kind: {k}")
        if isinstance(opnd, int):
            return opnd & MASK64
        raise TypeError(f"unsupported operand type: {type(opnd)}")

    def resolve_src32(self, opnd):
        """Resolve a 32-bit shift count operand (SGPR or immediate constant)."""
        if isinstance(opnd, dict):
            k = opnd.get("k")
            if k == "sgpr":
                r = opnd["r"]
                return self.get_sgpr_u32(r)
            if k == "imm":
                return opnd["v"] & MASK32
            raise ValueError(f"unknown 32-bit operand kind: {k}")
        if isinstance(opnd, int):
            return opnd & MASK32
        raise TypeError(f"unsupported operand type: {type(opnd)}")

    def step(self, op, dst, src0, src1):
        """Execute a single scalar shift instruction.

        Arguments:
          op: 's_lshl_b64', 's_lshr_b64', or 's_ashr_i64'
          dst: dict {'k': 'sgpr', 'r': int, 'n': 2} or int register index
          src0: 64-bit source (dict or int)
          src1: 32-bit shift count (dict or int)
        """
        val64 = self.resolve_src64(src0)
        sh32 = self.resolve_src32(src1)

        if op == "s_lshl_b64":
            res64, scc = ref_s_lshl_b64(val64, sh32)
        elif op == "s_lshr_b64":
            res64, scc = ref_s_lshr_b64(val64, sh32)
        elif op == "s_ashr_i64":
            res64, scc = ref_s_ashr_i64(val64, sh32)
        else:
            raise KeyError(f"unsupported scalar shift opcode: {op}")

        dst_r = dst["r"] if isinstance(dst, dict) else dst
        self.set_sgpr_u64(dst_r, res64)
        self.scc = scc
        return res64, scc

    def execute_ir(self, ins):
        """Execute an IR instruction dictionary matching tools/rdna2spirv.py parser format."""
        op = ins["op"]
        args = ins["args"]
        return self.step(op, args[0], args[1], args[2])


class TestScalarShiftDirectSemantics(unittest.TestCase):
    """Direct mathematical unit tests for 64-bit scalar logical and arithmetic shifts."""

    def test_lshl_b64_basic_powers_of_two(self):
        for shift in range(64):
            with self.subTest(shift=shift):
                res, scc = ref_s_lshl_b64(1, shift)
                expected = (1 << shift) & MASK64
                self.assertEqual(res, expected)
                self.assertEqual(scc, 1)

    def test_lshl_b64_zero_yields_zero_and_clears_scc(self):
        for shift in (0, 1, 31, 32, 63):
            with self.subTest(shift=shift):
                res, scc = ref_s_lshl_b64(0, shift)
                self.assertEqual(res, 0)
                self.assertEqual(scc, 0)

    def test_lshl_b64_overflow_shifts_out_to_zero(self):
        # 0x8000000000000000 shifted left by 1 drops off the 64-bit boundary.
        res, scc = ref_s_lshl_b64(0x8000000000000000, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_lshr_b64_basic_powers_of_two(self):
        msb = 0x8000000000000000
        for shift in range(64):
            with self.subTest(shift=shift):
                res, scc = ref_s_lshr_b64(msb, shift)
                expected = msb >> shift
                self.assertEqual(res, expected)
                self.assertEqual(scc, 1)

    def test_lshr_b64_underflow_shifts_out_to_zero(self):
        res, scc = ref_s_lshr_b64(1, 1)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        res, scc = ref_s_lshr_b64(0x00000000FFFFFFFF, 32)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_ashr_i64_positive_matches_lshr(self):
        positive_samples = [
            0x0000000000000001,
            0x000000007FFFFFFF,
            0x0000000100000000,
            0x7FFFFFFFFFFFFFFF,
            0x123456789ABCDEF0,
            0x0000000000000000,
        ]
        for val in positive_samples:
            for shift in (0, 1, 15, 31, 32, 48, 63):
                with self.subTest(val=hex(val), shift=shift):
                    ashr_res, ashr_scc = ref_s_ashr_i64(val, shift)
                    lshr_res, lshr_scc = ref_s_lshr_b64(val, shift)
                    self.assertEqual(ashr_res, lshr_res)
                    self.assertEqual(ashr_scc, lshr_scc)

    def test_ashr_i64_negative_sign_extension(self):
        # All ones (-1) shifted arithmetically right by any count remains -1 (0xFFFFFFFFFFFFFFFF).
        for shift in range(64):
            with self.subTest(shift=shift):
                res, scc = ref_s_ashr_i64(MASK64, shift)
                self.assertEqual(res, MASK64)
                self.assertEqual(scc, 1)

        # 0x8000000000000000 shifted right arithmetically replicates 1s.
        res, scc = ref_s_ashr_i64(0x8000000000000000, 1)
        self.assertEqual(res, 0xC000000000000000)
        self.assertEqual(scc, 1)

        res, scc = ref_s_ashr_i64(0x8000000000000000, 32)
        self.assertEqual(res, 0xFFFFFFFF80000000)
        self.assertEqual(scc, 1)

        res, scc = ref_s_ashr_i64(0x8000000000000000, 63)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_contrast_lshr_vs_ashr_on_negative_operands(self):
        negative_val = 0x8000000000000000
        lshr_res, _ = ref_s_lshr_b64(negative_val, 1)
        ashr_res, _ = ref_s_ashr_i64(negative_val, 1)
        self.assertEqual(lshr_res, 0x4000000000000000)
        self.assertEqual(ashr_res, 0xC000000000000000)
        self.assertNotEqual(lshr_res, ashr_res)

        lshr_full, _ = ref_s_lshr_b64(negative_val, 63)
        ashr_full, _ = ref_s_ashr_i64(negative_val, 63)
        self.assertEqual(lshr_full, 1)
        self.assertEqual(ashr_full, MASK64)

    def test_shift_count_modulo_64_masking(self):
        # RDNA2 SOP2 uses S1[5:0] (src1 & 63). Shifts by multiples of 64 must equal shift by 0.
        val = 0x123456789ABCDEF0
        for base_shift in (0, 5, 31, 32, 63):
            for extra in (64, 128, 256, 1024, 0x10000000):
                dirty_count = base_shift + extra
                with self.subTest(base_shift=base_shift, extra=extra):
                    self.assertEqual(ref_s_lshl_b64(val, dirty_count), ref_s_lshl_b64(val, base_shift))
                    self.assertEqual(ref_s_lshr_b64(val, dirty_count), ref_s_lshr_b64(val, base_shift))
                    self.assertEqual(ref_s_ashr_i64(val, dirty_count), ref_s_ashr_i64(val, base_shift))

        # Shift by 0xFFFFFFFF (all 32 bits 1) masks to 63.
        self.assertEqual(ref_s_lshl_b64(1, MASK32), ref_s_lshl_b64(1, 63))
        self.assertEqual(ref_s_lshr_b64(0x8000000000000000, MASK32), ref_s_lshr_b64(0x8000000000000000, 63))
        self.assertEqual(ref_s_ashr_i64(0x8000000000000000, MASK32), ref_s_ashr_i64(0x8000000000000000, 63))

    def test_cross_dword_migration(self):
        # Shifting 32 bits moves low dword into high dword or vice versa.
        pattern = 0xDEADBEEFCAFEBABE
        lo, hi = split_dwords(pattern)
        self.assertEqual(lo, 0xCAFEBABE)
        self.assertEqual(hi, 0xDEADBEEF)

        # Logical left shift by 32: low dword becomes high dword, low dword zeroed.
        shl32, _ = ref_s_lshl_b64(pattern, 32)
        shl_lo, shl_hi = split_dwords(shl32)
        self.assertEqual(shl_lo, 0)
        self.assertEqual(shl_hi, 0xCAFEBABE)

        # Logical right shift by 32: high dword becomes low dword, high dword zeroed.
        shr32, _ = ref_s_lshr_b64(pattern, 32)
        shr_lo, shr_hi = split_dwords(shr32)
        self.assertEqual(shr_lo, 0xDEADBEEF)
        self.assertEqual(shr_hi, 0)

    def test_scc_condition_code_transitions(self):
        # Confirm SCC is 0 when result is 0, and 1 when result is non-zero.
        _, scc_zero = ref_s_lshl_b64(0, 5)
        self.assertEqual(scc_zero, 0)

        _, scc_nonzero = ref_s_lshl_b64(1, 5)
        self.assertEqual(scc_nonzero, 1)

        # Verify low-only and high-only non-zero words set SCC to 1.
        low_only = 0x0000000000000001
        high_only = 0x0000000100000000
        self.assertEqual(ref_s_lshl_b64(low_only, 0)[1], 1)
        self.assertEqual(ref_s_lshl_b64(high_only, 0)[1], 1)


class TestScalarAluInterpreter(unittest.TestCase):
    """Interpreter execution tests modeling SGPR register file state and SCC updates."""

    def setUp(self):
        self.interp = ScalarAluInterpreter()

    def test_initial_state_clean(self):
        self.assertEqual(self.interp.scc, 0)
        self.assertEqual(self.interp.get_sgpr_u64(0), 0)

    def test_lshl_execution_and_register_update(self):
        # Set s[4:5] = 0x0000000000000001, s6 = 16.
        # Execute s_lshl_b64 s[0:1], s[4:5], s6.
        self.interp.set_sgpr_u64(4, 1)
        self.interp.set_sgpr_u32(6, 16)

        ins = {
            "op": "s_lshl_b64",
            "args": [
                {"k": "sgpr", "r": 0, "n": 2},
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "sgpr", "r": 6},
            ],
            "mods": [],
        }
        res64, scc = self.interp.execute_ir(ins)
        self.assertEqual(res64, 0x10000)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u64(0), 0x10000)
        self.assertEqual(self.interp.scc, 1)
        # Source registers must be untouched.
        self.assertEqual(self.interp.get_sgpr_u64(4), 1)
        self.assertEqual(self.interp.get_sgpr_u32(6), 16)

    def test_lshr_execution_with_immediate_count(self):
        # Set s[8:9] = 0x1234567800000000.
        # Execute s_lshr_b64 s[10:11], s[8:9], 32.
        self.interp.set_sgpr_u64(8, 0x1234567800000000)

        ins = {
            "op": "s_lshr_b64",
            "args": [
                {"k": "sgpr", "r": 10, "n": 2},
                {"k": "sgpr", "r": 8, "n": 2},
                {"k": "imm", "v": 32},
            ],
            "mods": [],
        }
        res64, scc = self.interp.execute_ir(ins)
        self.assertEqual(res64, 0x12345678)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u64(10), 0x12345678)
        self.assertEqual(self.interp.scc, 1)

    def test_ashr_execution_sign_propagation(self):
        # Set s[0:1] = -8 (0xFFFFFFFFFFFFFFF8).
        # Execute s_ashr_i64 s[2:3], s[0:1], 1.
        # Expected: -4 (0xFFFFFFFFFFFFFFFC).
        self.interp.set_sgpr_u64(0, 0xFFFFFFFFFFFFFFF8)

        ins = {
            "op": "s_ashr_i64",
            "args": [
                {"k": "sgpr", "r": 2, "n": 2},
                {"k": "sgpr", "r": 0, "n": 2},
                {"k": "imm", "v": 1},
            ],
            "mods": [],
        }
        res64, scc = self.interp.execute_ir(ins)
        self.assertEqual(res64, 0xFFFFFFFFFFFFFFFC)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.get_sgpr_u64(2), 0xFFFFFFFFFFFFFFFC)

    def test_in_place_shift_aliasing(self):
        # Hardware supports dst aliasing src0: s[2:3] = s[2:3] << 4.
        self.interp.set_sgpr_u64(2, 0x0F)
        self.interp.step("s_lshl_b64", 2, {"k": "sgpr", "r": 2, "n": 2}, 4)
        self.assertEqual(self.interp.get_sgpr_u64(2), 0xF0)
        self.assertEqual(self.interp.scc, 1)

    def test_scc_cleared_when_result_becomes_zero(self):
        # Pre-set SCC to 1.
        self.interp.scc = 1
        # Shift 1 right by 1 -> 0, SCC must become 0.
        self.interp.set_sgpr_u64(0, 1)
        self.interp.step("s_lshr_b64", 0, {"k": "sgpr", "r": 0, "n": 2}, 1)
        self.assertEqual(self.interp.get_sgpr_u64(0), 0)
        self.assertEqual(self.interp.scc, 0)

    def test_adjacent_registers_remain_untouched(self):
        # Set canary values in s0, s1, s4, s5. Write to s[2:3].
        self.interp.set_sgpr_u32(0, 0x11111111)
        self.interp.set_sgpr_u32(1, 0x22222222)
        self.interp.set_sgpr_u32(4, 0x44444444)
        self.interp.set_sgpr_u32(5, 0x55555555)

        self.interp.step("s_lshl_b64", 2, 0x1234567890ABCDEF, 8)
        self.assertEqual(self.interp.get_sgpr_u32(0), 0x11111111)
        self.assertEqual(self.interp.get_sgpr_u32(1), 0x22222222)
        self.assertEqual(self.interp.get_sgpr_u32(4), 0x44444444)
        self.assertEqual(self.interp.get_sgpr_u32(5), 0x55555555)


class TestSpirvTranslatorIntegration(unittest.TestCase):
    """Cross-validation between interpreter reference semantics and rdna2spirv code generation."""

    def test_rdna2spirv_lshl_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_lshl_b64",
            "args": [
                {"k": "sgpr", "r": 0, "n": 2},
                {"k": "sgpr", "r": 2, "n": 2},
                {"k": "imm", "v": 12},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        # Verify key SPIR-V instructions are present per translation rules:
        self.assertIn("OpShiftLeftLogical %ulong", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("%c3f", emitted)
        self.assertIn("OpStore %scc", emitted)
        self.assertTrue(g.uses_psb)

    def test_rdna2spirv_lshr_b64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_lshr_b64",
            "args": [
                {"k": "sgpr", "r": 4, "n": 2},
                {"k": "sgpr", "r": 6, "n": 2},
                {"k": "sgpr", "r": 8},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("OpShiftRightLogical %ulong", emitted)
        self.assertIn("OpStore %scc", emitted)
        self.assertTrue(g.uses_psb)

    def test_rdna2spirv_ashr_i64_emission(self):
        g = rdna2spirv.Gen("ps")
        ins = {
            "op": "s_ashr_i64",
            "args": [
                {"k": "sgpr", "r": 0, "n": 2},
                {"k": "sgpr", "r": 2, "n": 2},
                {"k": "imm", "v": 4},
            ],
            "mods": [],
        }
        g.insn(ins)
        emitted = "\n".join(g.body)

        self.assertIn("OpBitcast %long", emitted)
        self.assertIn("OpShiftRightArithmetic %long", emitted)
        self.assertIn("OpBitcast %ulong", emitted)
        self.assertIn("OpBitwiseAnd %uint", emitted)
        self.assertIn("OpStore %scc", emitted)
        self.assertTrue(g.uses_psb)

    def test_spirv_expansion_logic_matches_interpreter(self):
        # Simulate the exact SPIR-V sequence emitted by rdna2spirv and verify equivalence with ref_s_lshl_b64.
        vectors = [
            (0x12345678, 0x9ABCDEF0, 4),
            (0xFFFFFFFF, 0x00000000, 32),
            (0x00000001, 0x00000000, 63),
            (0x00000000, 0x80000000, 1),
            (0x00000000, 0x00000000, 16),
            (0xDEADBEEF, 0xCAFEBABE, 70),  # 70 & 63 = 6
        ]
        for lo, hi, sh in vectors:
            with self.subTest(lo=hex(lo), hi=hex(hi), sh=sh):
                # 1. SPIR-V expansion logic from rdna2spirv.py lines 2614-2630:
                l64 = lo & MASK64
                h64 = hi & MASK64
                t = (h64 << 32) & MASK64
                v64 = t | l64
                m = sh & 63
                s64 = m & MASK64
                r = (v64 << s64) & MASK64
                rlo = r & MASK32
                rh = (r >> 32) & MASK64
                rhi = rh & MASK32
                orr = (rlo | rhi) & MASK32
                spirv_scc = 1 if orr != 0 else 0
                spirv_res = join_dwords(rlo, rhi)

                # 2. Reference interpreter output:
                ref_res, ref_scc = ref_s_lshl_b64(join_dwords(lo, hi), sh)

                self.assertEqual(spirv_res, ref_res)
                self.assertEqual(spirv_scc, ref_scc)

    def test_unsupported_64bit_shift_operands_rejected(self):
        for op in ("s_lshl_b64", "s_lshr_b64", "s_ashr_i64"):
            with self.subTest(op=op):
                g_bad_dst = rdna2spirv.Gen("ps")
                with self.assertRaises(rdna2spirv.Unsupported):
                    g_bad_dst.insn({
                        "op": op,
                        "args": [
                            {"k": "sgpr", "r": 0},
                            {"k": "sgpr", "r": 2, "n": 2},
                            {"k": "imm", "v": 4},
                        ],
                        "mods": [],
                    })

                g_bad_src = rdna2spirv.Gen("ps")
                with self.assertRaises(rdna2spirv.Unsupported):
                    g_bad_src.insn({
                        "op": op,
                        "args": [
                            {"k": "sgpr", "r": 0, "n": 2},
                            {"k": "sgpr", "r": 2},
                            {"k": "imm", "v": 4},
                        ],
                        "mods": [],
                    })


def translate_ps_shifts(lines: list[str]) -> str:
    """Translate a hand-written pixel shader listing exporting v0..v3 to mrt0."""
    return translate_lines(
        "ps_salu_shifts",
        [*lines, "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"],
    )


class EndToEndScalarShiftTranslatorTests(unittest.TestCase):
    """End-to-end rdna2spirv.translate + spirv_mini_interp execution for 64-bit and 32-bit scalar shifts."""

    def test_runtime_s_lshl_b64_cross_dword_modulo64_and_scc_transitions(self):
        # Load runtime 64-bit value s[4:5] and runtime 32-bit shift count s6 from buffer memory
        # so neither operand is in sgpr_imm, exercising 0->1, 0->0, 1->0, and 1->1 SCC transitions.
        spv = translate_ps_shifts([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_lshl_b64 s[8:9], s[4:5], s6",
            "s_cselect_b32 s10, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when shifted result is 0)
            "s_lshl_b64 s[12:13], s[4:5], s6",
            "s_cselect_b32 s14, 1, 0",
            "s_lshl_b32 s14, s14, 1",
            "s_or_b32 s10, s10, s14",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s12",
        ])

        vectors = [
            # Zero value across shifts: result = 0, SCC = 0 (clears seeded SCC = 1)
            (0x00000000, 0x00000000, 0),
            (0x00000000, 0x00000000, 1),
            (0x00000000, 0x00000000, 32),
            (0x00000000, 0x00000000, 63),
            # Complete shift-out to zero: result = 0, SCC = 0
            (0x00000000, 0x80000000, 1),
            (0x80000000, 0x00000000, 33),
            (0x00010000, 0x00000000, 48),
            # Low-dword-only non-zero (rhi == 0, rlo != 0) -> SCC = 1
            (0x00000001, 0x00000000, 0),
            (0x00000001, 0x00000000, 15),
            (0x00000001, 0x00000000, 31),
            # High-dword-only non-zero (rlo == 0, rhi != 0) -> SCC = 1
            (0x00000001, 0x00000000, 32),
            (0x00000001, 0x00000000, 63),
            (0x00000000, 0x00000001, 16),
            # Cross-dword migration across 32-bit boundary
            (0xCAFEBABE, 0xDEADBEEF, 0),
            (0xCAFEBABE, 0xDEADBEEF, 1),
            (0xCAFEBABE, 0xDEADBEEF, 4),
            (0xCAFEBABE, 0xDEADBEEF, 16),
            (0xCAFEBABE, 0xDEADBEEF, 31),
            (0xCAFEBABE, 0xDEADBEEF, 32),
            (0xCAFEBABE, 0xDEADBEEF, 33),
            (0xCAFEBABE, 0xDEADBEEF, 48),
            (0xCAFEBABE, 0xDEADBEEF, 63),
            # Modulo-64 unmasked runtime shift counts (S1 & 63)
            (0xCAFEBABE, 0xDEADBEEF, 64),          # 64 & 63 = 0
            (0xCAFEBABE, 0xDEADBEEF, 68),          # 68 & 63 = 4
            (0xCAFEBABE, 0xDEADBEEF, 96),          # 96 & 63 = 32
            (0xCAFEBABE, 0xDEADBEEF, 127),         # 127 & 63 = 63
            (0x00000001, 0x00000000, 0x80000020),  # & 63 = 32
            (0x00000001, 0x00000000, 0xFFFFFFFF),  # & 63 = 63
        ]

        for lo, hi, sh in vectors:
            with self.subTest(lo=hex(lo), hi=hex(hi), sh=hex(sh)):
                want64, want_scc = ref_s_lshl_b64(join_dwords(lo, hi), sh)
                want_lo, want_hi = split_dwords(want64)
                want_scc_mask = (want_scc << 1) | want_scc
                words = mrt0_words(
                    spv,
                    push_constants=PUSH_VSHARP,
                    memory={0x1000: lo, 0x1004: hi, 0x1008: sh, 0x100C: 0},
                )
                self.assertEqual(words, [want_lo, want_hi, want_scc_mask, want_lo])

    def test_runtime_s_lshr_b64_zero_extension_modulo64_and_scc_transitions(self):
        spv = translate_ps_shifts([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_lshr_b64 s[8:9], s[4:5], s6",
            "s_cselect_b32 s10, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when shifted result is 0)
            "s_lshr_b64 s[12:13], s[4:5], s6",
            "s_cselect_b32 s14, 1, 0",
            "s_lshl_b32 s14, s14, 1",
            "s_or_b32 s10, s10, s14",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s13",
        ])

        vectors = [
            # Zero value across shifts: result = 0, SCC = 0
            (0x00000000, 0x00000000, 0),
            (0x00000000, 0x00000000, 1),
            (0x00000000, 0x00000000, 32),
            (0x00000000, 0x00000000, 63),
            # Underflow shift-out to zero: result = 0, SCC = 0
            (0x00000001, 0x00000000, 1),
            (0xFFFFFFFF, 0x00000000, 32),
            (0x00000000, 0x00007FFF, 47),
            # MSB zero-extension (logical right shift never sign-extends bit 63)
            (0x00000000, 0x80000000, 0),
            (0x00000000, 0x80000000, 1),
            (0x00000000, 0x80000000, 31),
            (0x00000000, 0x80000000, 32),
            (0x00000000, 0x80000000, 63),
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 32),
            (0xFFFFFFFF, 0xFFFFFFFF, 63),
            # Cross-dword migration
            (0xCAFEBABE, 0xDEADBEEF, 0),
            (0xCAFEBABE, 0xDEADBEEF, 1),
            (0xCAFEBABE, 0xDEADBEEF, 8),
            (0xCAFEBABE, 0xDEADBEEF, 16),
            (0xCAFEBABE, 0xDEADBEEF, 31),
            (0xCAFEBABE, 0xDEADBEEF, 32),
            (0xCAFEBABE, 0xDEADBEEF, 33),
            (0xCAFEBABE, 0xDEADBEEF, 48),
            (0xCAFEBABE, 0xDEADBEEF, 63),
            # Modulo-64 unmasked runtime shift counts (S1 & 63)
            (0xCAFEBABE, 0xDEADBEEF, 64),          # 64 & 63 = 0
            (0xCAFEBABE, 0xDEADBEEF, 72),          # 72 & 63 = 8
            (0xCAFEBABE, 0xDEADBEEF, 96),          # 96 & 63 = 32
            (0xCAFEBABE, 0xDEADBEEF, 127),         # 127 & 63 = 63
            (0x00000000, 0x80000000, 0x80000010),  # & 63 = 16
            (0x00000000, 0x80000000, 0xFFFFFFFF),  # & 63 = 63
        ]

        for lo, hi, sh in vectors:
            with self.subTest(lo=hex(lo), hi=hex(hi), sh=hex(sh)):
                want64, want_scc = ref_s_lshr_b64(join_dwords(lo, hi), sh)
                want_lo, want_hi = split_dwords(want64)
                want_scc_mask = (want_scc << 1) | want_scc
                words = mrt0_words(
                    spv,
                    push_constants=PUSH_VSHARP,
                    memory={0x1000: lo, 0x1004: hi, 0x1008: sh, 0x100C: 0},
                )
                self.assertEqual(words, [want_lo, want_hi, want_scc_mask, want_hi])

    def test_runtime_s_ashr_i64_sign_extension_modulo64_and_scc_transitions(self):
        spv = translate_ps_shifts([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_ashr_i64 s[8:9], s[4:5], s6",
            "s_cselect_b32 s10, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1 (verifies active clear when positive value shifts to 0)
            "s_ashr_i64 s[12:13], s[4:5], s6",
            "s_cselect_b32 s14, 1, 0",
            "s_lshl_b32 s14, s14, 1",
            "s_or_b32 s10, s10, s14",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s13",
        ])

        vectors = [
            # Zero value across shifts: result = 0, SCC = 0
            (0x00000000, 0x00000000, 0),
            (0x00000000, 0x00000000, 1),
            (0x00000000, 0x00000000, 32),
            (0x00000000, 0x00000000, 63),
            # Positive values shifting out completely to 0 -> SCC = 0 (clears seeded SCC = 1)
            (0x00000001, 0x00000000, 1),
            (0xFFFFFFFF, 0x00000000, 32),
            (0xFFFFFFFF, 0x7FFFFFFF, 63),
            # Positive values (bit 63 == 0) zero-extend identically to s_lshr_b64
            (0xFFFFFFFF, 0x7FFFFFFF, 0),
            (0xFFFFFFFF, 0x7FFFFFFF, 1),
            (0xFFFFFFFF, 0x7FFFFFFF, 16),
            (0xFFFFFFFF, 0x7FFFFFFF, 31),
            (0xFFFFFFFF, 0x7FFFFFFF, 32),
            (0xFFFFFFFF, 0x7FFFFFFF, 48),
            (0x9ABCDEF0, 0x12345678, 12),
            (0x9ABCDEF0, 0x12345678, 36),
            # Negative values (bit 63 == 1) sign-extend 1s across high and low dwords -> SCC = 1
            (0x00000000, 0x80000000, 0),
            (0x00000000, 0x80000000, 1),
            (0x00000000, 0x80000000, 15),
            (0x00000000, 0x80000000, 31),
            (0x00000000, 0x80000000, 32),
            (0x00000000, 0x80000000, 33),
            (0x00000000, 0x80000000, 63),
            (0xFFFFFFFF, 0xFFFFFFFF, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 32),
            (0xFFFFFFFF, 0xFFFFFFFF, 63),
            (0xCAFEBABE, 0xDEADBEEF, 4),
            (0xCAFEBABE, 0xDEADBEEF, 32),
            (0xCAFEBABE, 0xDEADBEEF, 48),
            # Modulo-64 unmasked runtime shift counts (S1 & 63)
            (0x00000000, 0x80000000, 64),          # 64 & 63 = 0
            (0x00000000, 0x80000000, 65),          # 65 & 63 = 1
            (0x00000000, 0x80000000, 96),          # 96 & 63 = 32
            (0x00000000, 0x80000000, 127),         # 127 & 63 = 63
            (0xCAFEBABE, 0xDEADBEEF, 0x80000020),  # & 63 = 32
            (0x00000000, 0x80000000, 0xFFFFFFFF),  # & 63 = 63
        ]

        for lo, hi, sh in vectors:
            with self.subTest(lo=hex(lo), hi=hex(hi), sh=hex(sh)):
                want64, want_scc = ref_s_ashr_i64(join_dwords(lo, hi), sh)
                want_lo, want_hi = split_dwords(want64)
                want_scc_mask = (want_scc << 1) | want_scc
                words = mrt0_words(
                    spv,
                    push_constants=PUSH_VSHARP,
                    memory={0x1000: lo, 0x1004: hi, 0x1008: sh, 0x100C: 0},
                )
                self.assertEqual(words, [want_lo, want_hi, want_scc_mask, want_hi])

    def test_runtime_64bit_shifts_immediate_source_and_inplace_aliasing(self):
        # 1. Immediate 64-bit source S0 shifted by runtime shift count s4, plus in-place dst == src0 aliasing.
        spv = translate_ps_shifts([
            "s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            # Immediate source S0 = 1 shifted left by runtime s6
            "s_lshl_b64 s[8:9], 1, s6",
            # Immediate source S0 = -16 (sign-extended to 0xFFFFFFFFFFFFFFF0) shifted arithmetically right by s6
            "s_ashr_i64 s[10:11], -16, s6",
            # In-place destination aliasing: s[4:5] = s[4:5] >> s6
            "s_lshr_b64 s[4:5], s[4:5], s6",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s9",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s4",
        ])

        for sh in (0, 1, 2, 4, 32, 35, 63, 66):
            with self.subTest(sh=sh):
                shl_lo, shl_hi = split_dwords(ref_s_lshl_b64(1, sh)[0])
                ashr_lo, _ = split_dwords(ref_s_ashr_i64((-16) & MASK64, sh)[0])
                inplace_lo, _ = split_dwords(ref_s_lshr_b64(0xDEADBEEFCAFEBABE, sh)[0])
                words = mrt0_words(
                    spv,
                    push_constants=PUSH_VSHARP,
                    memory={0x1000: 0xCAFEBABE, 0x1004: 0xDEADBEEF, 0x1008: sh, 0x100C: 0},
                )
                self.assertEqual(words, [shl_lo, shl_hi, ashr_lo, inplace_lo])

    def test_runtime_s_lshl_b32_and_s_lshr_b32_modulo32_folding_and_scc(self):
        # Exercises 32-bit scalar logical shifts across:
        # 1) runtime value s4 + runtime shift count s5 (modulo-32 mask & 31, 0->1, 0->0, 1->0, 1->1 SCC)
        # 2) constant-folded s_lshl_b32 / s_lshr_b32 followed by runtime overwrite invalidating sgpr_imm
        spv = translate_ps_shifts([
            "s_buffer_load_dwordx2 s[4:5], s[0:3], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_lshl_b32 s6, s4, s5",
            "s_cselect_b32 s7, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1
            "s_lshl_b32 s8, s4, s5",
            "s_cselect_b32 s9, 1, 0",
            "s_lshl_b32 s9, s9, 1",
            "s_or_b32 s7, s7, s9",
            "s_cmp_eq_u32 0, 1",          # seed SCC = 0
            "s_lshr_b32 s10, s4, s5",
            "s_cselect_b32 s11, 1, 0",
            "s_cmp_eq_u32 1, 1",          # seed SCC = 1
            "s_lshr_b32 s12, s4, s5",
            "s_cselect_b32 s13, 1, 0",
            "s_lshl_b32 s13, s13, 1",
            "s_or_b32 s11, s11, s13",
            # Constant-folded s_lshl_b32 into s14, then overwritten by runtime s_lshr_b32 with immediate count 36 (& 31 = 4)
            "s_mov_b32 s14, 0x11",
            "s_lshl_b32 s14, s14, 4",
            "s_lshr_b32 s14, s4, 36",
            "s_xor_b32 s10, s10, s14",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, s11",
        ])

        vectors = [
            (0x00000000, 0),
            (0x00000000, 5),
            (0x00000001, 0),
            (0x00000001, 1),
            (0x00000001, 31),
            (0x80000000, 1),          # shl -> 0 (SCC=0), shr -> 0x40000000 (SCC=1)
            (0x80000000, 31),         # shl -> 0 (SCC=0), shr -> 1 (SCC=1)
            (0x12345678, 4),
            (0x12345678, 32),         # 32 & 31 = 0
            (0x12345678, 36),         # 36 & 31 = 4
            (0x12345678, 0x80000010), # & 31 = 16
            (0x80000001, 0xFFFFFFFF), # & 31 = 31
        ]

        for val, sh in vectors:
            with self.subTest(val=hex(val), sh=hex(sh)):
                m = sh & 31
                want_shl = ((val & MASK32) << m) & MASK32
                want_shl_scc = 3 if want_shl != 0 else 0
                want_shr = (val & MASK32) >> m
                want_shr_scc = 3 if want_shr != 0 else 0
                want_shr_imm36 = (val & MASK32) >> 4
                words = mrt0_words(
                    spv,
                    push_constants=PUSH_VSHARP,
                    memory={0x1000: val, 0x1004: sh},
                )
                self.assertEqual(
                    words,
                    [want_shl, want_shl_scc, want_shr ^ want_shr_imm36, want_shr_scc],
                )


if __name__ == "__main__":
    unittest.main()
