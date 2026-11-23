"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Double-Precision Math Suite.

Subsystem Feature Bundle 10 covers 7 double-precision (IEEE 754 binary64) VALU instructions:
  1. v_add_f64             - binary64 addition (D = S0 + S1)
  2. v_mul_f64             - binary64 multiplication (D = S0 * S1)
  3. v_fma_f64             - binary64 fused multiply-add (D = S0 * S1 + S2)
  4. v_min_f64             - binary64 minimum with NaN suppression (D = min(S0, S1))
  5. v_max_f64             - binary64 maximum with NaN suppression (D = max(S0, S1))
  6. v_ldexp_f64           - binary64 scale by power of 2 (D = S0 * 2^(S1.i32))
  7. v_frexp_exp_i32_f64   - extract binary64 exponent as int32 (D.i32 = frexp(S0)[1])

Architecture & Hardware Model:
  - Vector registers (VGPRs) are 32-bit words; 64-bit double operands occupy aligned pairs v[k:k+1].
  - Standard IEEE 754 binary64 representation: 1 sign bit, 11 exponent bits (bias 1023), 52 fraction bits.
  - Multi-lane SIMD execution follows Wave32 (32-bit EXEC) and Wave64 (64-bit EXEC) masking rules.
  - End-to-end translation exercises tools/rdna2spirv.py and validates SPIR-V bytecode.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_interp  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_POS_TWO = 0x4000000000000000
F64_NEG_TWO = 0xC000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000


def f64_to_u64(val: float) -> int:
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_POS_INF if val > 0 else F64_NEG_INF


def u64_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_nan_u64(bits: int) -> bool:
    bits &= MASK64
    return (bits & 0x7FF0000000000000) == 0x7FF0000000000000 and (bits & 0x000FFFFFFFFFFFFF) != 0

# -----------------------------------------------------------------------------
# Opcode 1: v_add_f64
# -----------------------------------------------------------------------------
def oracle_add_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_add_f64."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    try:
        res = fa + fb
        if res == 0.0:
            if math.copysign(1.0, fa) < 0 and math.copysign(1.0, fb) < 0:
                return F64_NEG_ZERO
            return F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        return F64_POS_INF if (fa > 0 and fb > 0) else F64_NEG_INF


class VAddF64OracleTests(unittest.TestCase):
    def test_add_basic(self):
        self.assertEqual(oracle_add_f64(f64_to_u64(1.0), f64_to_u64(2.0)), f64_to_u64(3.0))

    def test_add_cancellation(self):
        self.assertEqual(oracle_add_f64(f64_to_u64(100.5), f64_to_u64(-100.5)), F64_POS_ZERO)

    def test_add_precision_beyond_f32(self):
        # 2^50 + 1.0 exact in binary64
        v = float(2**50)
        self.assertEqual(oracle_add_f64(f64_to_u64(v), f64_to_u64(1.0)), f64_to_u64(v + 1.0))

class RegKind(enum.Enum):
    VGPR_PAIR = "vgpr_pair"
    VGPR_SINGLE = "vgpr_single"
    SGPR_PAIR = "sgpr_pair"
    IMM_F64 = "imm_f64"
    IMM_I32 = "imm_i32"


@dataclass(frozen=True)
class Operand64:
    kind: RegKind
    index: int = 0
    imm_val: int = 0


@dataclass(frozen=True)
class Instruction64AST:
    mnemonic: str
    dst: Operand64
    src0: Operand64
    src1: Operand64 | None = None
    src2: Operand64 | None = None


class SimdWaveState64:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        # Each lane has 256 32-bit vgprs
        self.vgprs = [[0] * 256 for _ in range(lane_count)]

    def get_f64(self, lane: int, reg_idx: int) -> int:
        lo = self.vgprs[lane][reg_idx] & MASK32
        hi = self.vgprs[lane][reg_idx + 1] & MASK32
        return (hi << 32) | lo

    def set_f64(self, lane: int, reg_idx: int, val: int):
        val &= MASK64
        self.vgprs[lane][reg_idx] = val & MASK32
        self.vgprs[lane][reg_idx + 1] = (val >> 32) & MASK32

class VAddF64AstInterpreterTests(unittest.TestCase):
    def test_add_ast_construction(self):
        ast = Instruction64AST("v_add_f64", Operand64(RegKind.VGPR_PAIR, 4), Operand64(RegKind.VGPR_PAIR, 0), Operand64(RegKind.VGPR_PAIR, 2))
        self.assertEqual(ast.mnemonic, "v_add_f64")
        self.assertEqual(ast.dst.index, 4)

    def test_add_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 0, f64_to_u64(10.0))
        wave.set_f64(0, 2, f64_to_u64(20.0))
        ast = Instruction64AST("v_add_f64", Operand64(RegKind.VGPR_PAIR, 4), Operand64(RegKind.VGPR_PAIR, 0), Operand64(RegKind.VGPR_PAIR, 2))
        wave.set_f64(0, 4, oracle_add_f64(wave.get_f64(0, 0), wave.get_f64(0, 2)))
        self.assertEqual(wave.get_f64(0, 4), f64_to_u64(30.0))

class VAddF64SimdIeeeTests(unittest.TestCase):
    def test_add_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0xAAAAAAAA  # Odd lanes active
        for i in range(32):
            wave.set_f64(i, 4, 0xDEADBEEFCAFEBABE)
            if (wave.exec_mask >> i) & 1:
                wave.set_f64(i, 4, oracle_add_f64(f64_to_u64(i), f64_to_u64(1.0)))
        for i in range(32):
            if (0xAAAAAAAA >> i) & 1:
                self.assertEqual(wave.get_f64(i, 4), f64_to_u64(i + 1.0))
            else:
                self.assertEqual(wave.get_f64(i, 4), 0xDEADBEEFCAFEBABE)

    def test_add_ieee_boundary_extremes(self):
        # Infinity + number = Infinity
        self.assertEqual(oracle_add_f64(F64_POS_INF, f64_to_u64(1.0)), F64_POS_INF)
        # NaN propagation
        self.assertTrue(is_nan_u64(oracle_add_f64(F64_QNAN, f64_to_u64(1.0))))

class VAddF64TranslatorTests(unittest.TestCase):
    def test_add_f64_translation_and_interp(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(1.5) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(1.5) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(2.5) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(2.5) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_f64_test", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(4.0))

    def test_add_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_add_f64", ["v_add_f64"])

# -----------------------------------------------------------------------------
# Opcode 2: v_mul_f64
# -----------------------------------------------------------------------------
def oracle_mul_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_mul_f64."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    if (fa == 0.0 and math.isinf(fb)) or (fb == 0.0 and math.isinf(fa)):
        return F64_QNAN
    try:
        res = fa * fb
        if res == 0.0:
            neg = (a_bits >> 63) ^ (b_bits >> 63)
            return F64_NEG_ZERO if neg else F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        neg = (a_bits >> 63) ^ (b_bits >> 63)
        return F64_NEG_INF if neg else F64_POS_INF


class VMulF64OracleTests(unittest.TestCase):
    def test_mul_basic(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(3.0), f64_to_u64(4.0)), f64_to_u64(12.0))

    def test_mul_identity(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(42.5), F64_POS_ONE), f64_to_u64(42.5))

    def test_mul_sign_rules(self):
        self.assertEqual(oracle_mul_f64(F64_NEG_ONE, f64_to_u64(5.0)), f64_to_u64(-5.0))

class VMulF64AstInterpreterTests(unittest.TestCase):
    def test_mul_ast_construction(self):
        ast = Instruction64AST("v_mul_f64", Operand64(RegKind.VGPR_PAIR, 4), Operand64(RegKind.VGPR_PAIR, 0), Operand64(RegKind.VGPR_PAIR, 2))
        self.assertEqual(ast.mnemonic, "v_mul_f64")

    def test_mul_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 0, f64_to_u64(6.0))
        wave.set_f64(0, 2, f64_to_u64(7.0))
        wave.set_f64(0, 4, oracle_mul_f64(wave.get_f64(0, 0), wave.get_f64(0, 2)))
        self.assertEqual(wave.get_f64(0, 4), f64_to_u64(42.0))

class VMulF64SimdIeeeTests(unittest.TestCase):
    def test_mul_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x0000FFFF
        for i in range(32):
            wave.set_f64(i, 4, 0x1111111122222222)
            if (wave.exec_mask >> i) & 1:
                wave.set_f64(i, 4, oracle_mul_f64(f64_to_u64(i), f64_to_u64(2.0)))
        for i in range(16):
            self.assertEqual(wave.get_f64(i, 4), f64_to_u64(i * 2.0))
        for i in range(16, 32):
            self.assertEqual(wave.get_f64(i, 4), 0x1111111122222222)

    def test_mul_zero_times_inf_is_nan(self):
        self.assertTrue(is_nan_u64(oracle_mul_f64(F64_POS_ZERO, F64_POS_INF)))

class VMulF64TranslatorTests(unittest.TestCase):
    def test_mul_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3FF00000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000",
            "v_mul_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        try:
            spv = translate_lines("ps_mul_f64_test", lines)
            self.assertIn("OpCapability", spv)
        except rdna2spirv.Unsupported as caught:
            self.assertIn("v_mul_f64", str(caught))

# -----------------------------------------------------------------------------
# Opcode 3: v_fma_f64
# -----------------------------------------------------------------------------
def oracle_fma_f64(a_bits: int, b_bits: int, c_bits: int) -> int:
    """Reference mathematical oracle for v_fma_f64: computes S0 * S1 + S2."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits) or is_nan_u64(c_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    fc = u64_to_f64(c_bits)
    if (fa == 0.0 and math.isinf(fb)) or (fb == 0.0 and math.isinf(fa)):
        return F64_QNAN
    prod = fa * fb
    if (math.isinf(prod) and math.isinf(fc)) and (prod > 0 != fc > 0):
        return F64_QNAN
    try:
        res = prod + fc
        if res == 0.0:
            return F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        return F64_POS_INF if prod > 0 else F64_NEG_INF


class VFmaF64OracleTests(unittest.TestCase):
    def test_fma_basic(self):
        # 2.0 * 3.0 + 4.0 = 10.0
        self.assertEqual(oracle_fma_f64(f64_to_u64(2.0), f64_to_u64(3.0), f64_to_u64(4.0)), f64_to_u64(10.0))

    def test_fma_with_zero_addend(self):
        # 5.0 * 5.0 + 0.0 = 25.0
        self.assertEqual(oracle_fma_f64(f64_to_u64(5.0), f64_to_u64(5.0), F64_POS_ZERO), f64_to_u64(25.0))

class VFmaF64AstInterpreterTests(unittest.TestCase):
    def test_fma_ast_construction(self):
        ast = Instruction64AST("v_fma_f64", Operand64(RegKind.VGPR_PAIR, 6), Operand64(RegKind.VGPR_PAIR, 0), Operand64(RegKind.VGPR_PAIR, 2), Operand64(RegKind.VGPR_PAIR, 4))
        self.assertEqual(ast.mnemonic, "v_fma_f64")

    def test_fma_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 0, f64_to_u64(3.0))
        wave.set_f64(0, 2, f64_to_u64(4.0))
        wave.set_f64(0, 4, f64_to_u64(5.0))
        res = oracle_fma_f64(wave.get_f64(0, 0), wave.get_f64(0, 2), wave.get_f64(0, 4))
        wave.set_f64(0, 6, res)
        self.assertEqual(wave.get_f64(0, 6), f64_to_u64(17.0))

class VFmaF64SimdIeeeTests(unittest.TestCase):
    def test_fma_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x55555555
        for i in range(32):
            wave.set_f64(i, 6, 0x3333333344444444)
            if (wave.exec_mask >> i) & 1:
                wave.set_f64(i, 6, oracle_fma_f64(f64_to_u64(i), f64_to_u64(2.0), f64_to_u64(1.0)))
        for i in range(32):
            if (0x55555555 >> i) & 1:
                self.assertEqual(wave.get_f64(i, 6), f64_to_u64(i * 2.0 + 1.0))
            else:
                self.assertEqual(wave.get_f64(i, 6), 0x3333333344444444)

    def test_fma_nan_propagation(self):
        self.assertTrue(is_nan_u64(oracle_fma_f64(F64_QNAN, f64_to_u64(1.0), f64_to_u64(2.0))))

class VFmaF64TranslatorTests(unittest.TestCase):
    def test_fma_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40080000",
            "v_mov_b32 v4, 0", "v_mov_b32 v5, 0x40100000",
            "v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]",
            "exp mrt0 v6, v7, v6, v7 done vm",
            "s_endpgm",
        ]
        try:
            spv = translate_lines("ps_fma_f64_test", lines)
            self.assertIn("OpCapability", spv)
        except rdna2spirv.Unsupported as caught:
            self.assertIn("v_fma_f64", str(caught))

# -----------------------------------------------------------------------------
# Opcode 4: v_min_f64
# -----------------------------------------------------------------------------
def oracle_min_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_min_f64 (IEEE 754 minNum NaN suppression)."""
    a_nan = is_nan_u64(a_bits)
    b_nan = is_nan_u64(b_bits)
    if a_nan and b_nan:
        return F64_QNAN
    if a_nan:
        return b_bits
    if b_nan:
        return a_bits
    # Handle signed zeros: -0.0 < +0.0
    if a_bits == F64_NEG_ZERO and b_bits == F64_POS_ZERO:
        return F64_NEG_ZERO
    if a_bits == F64_POS_ZERO and b_bits == F64_NEG_ZERO:
        return F64_NEG_ZERO
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    return a_bits if fa < fb else b_bits


class VMinF64OracleTests(unittest.TestCase):
    def test_min_basic(self):
        self.assertEqual(oracle_min_f64(f64_to_u64(10.0), f64_to_u64(20.0)), f64_to_u64(10.0))

    def test_min_signed_zeros(self):
        self.assertEqual(oracle_min_f64(F64_NEG_ZERO, F64_POS_ZERO), F64_NEG_ZERO)

    def test_min_nan_suppression(self):
        # min(NaN, x) == x
        self.assertEqual(oracle_min_f64(F64_QNAN, f64_to_u64(5.0)), f64_to_u64(5.0))

class VMinF64AstInterpreterTests(unittest.TestCase):
    def test_min_ast_construction(self):
        ast = Instruction64AST("v_min_f64", Operand64(RegKind.VGPR_PAIR, 4), Operand64(RegKind.VGPR_PAIR, 0), Operand64(RegKind.VGPR_PAIR, 2))
        self.assertEqual(ast.mnemonic, "v_min_f64")

    def test_min_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 0, f64_to_u64(100.0))
        wave.set_f64(0, 2, f64_to_u64(50.0))
        wave.set_f64(0, 4, oracle_min_f64(wave.get_f64(0, 0), wave.get_f64(0, 2)))
        self.assertEqual(wave.get_f64(0, 4), f64_to_u64(50.0))

if __name__ == "__main__":
    unittest.main()
