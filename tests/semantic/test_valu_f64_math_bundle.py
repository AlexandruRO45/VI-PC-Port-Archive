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

if __name__ == "__main__":
    unittest.main()
