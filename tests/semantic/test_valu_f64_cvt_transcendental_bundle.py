"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Double-Precision Conversion & Root/Reciprocal Suite.

Subsystem Feature Bundle 31 covers 6 double-precision (IEEE 754 binary64) VALU instructions:
  1. v_cvt_f32_f64         - binary64 to binary32 conversion with round-to-nearest-even (D.f32 = (float)S0.f64)
  2. v_cvt_f64_i32         - signed int32 to binary64 exact conversion (D.f64 = (double)S0.i32)
  3. v_cvt_f64_u32         - unsigned uint32 to binary64 exact conversion (D.f64 = (double)S0.u32)
  4. v_rcp_f64             - binary64 reciprocal (D.f64 = 1.0 / S0.f64)
  5. v_rsq_f64             - binary64 reciprocal square root (D.f64 = 1.0 / sqrt(S0.f64))
  6. v_sqrt_f64            - binary64 square root (D.f64 = sqrt(S0.f64))

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

# IEEE 754 binary64 constants
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

# IEEE 754 binary32 constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_QNAN = 0x7FC00000


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


def f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


def u32_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_u32(bits: int) -> bool:
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0

# -----------------------------------------------------------------------------
# Opcode 1: v_cvt_f32_f64
# -----------------------------------------------------------------------------
def oracle_cvt_f32_f64(bits_f64: int) -> int:
    """Reference mathematical oracle for v_cvt_f32_f64 (double to single float)."""
    if is_nan_u64(bits_f64):
        return F32_QNAN
    if bits_f64 == F64_POS_ZERO:
        return F32_POS_ZERO
    if bits_f64 == F64_NEG_ZERO:
        return F32_NEG_ZERO
    if bits_f64 == F64_POS_INF:
        return F32_POS_INF
    if bits_f64 == F64_NEG_INF:
        return F32_NEG_INF
    val = u64_to_f64(bits_f64)
    try:
        raw = struct.pack("<f", val)
        return struct.unpack("<I", raw)[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


class VCvtF32F64OracleTests(unittest.TestCase):
    def test_cvt_f32_f64_basic(self):
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(1.0)), F32_POS_ONE)
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(-1.0)), F32_NEG_ONE)
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(2.5)), f32_to_u32(2.5))

    def test_cvt_f32_f64_zeros_and_infinities(self):
        self.assertEqual(oracle_cvt_f32_f64(F64_POS_ZERO), F32_POS_ZERO)
        self.assertEqual(oracle_cvt_f32_f64(F64_NEG_ZERO), F32_NEG_ZERO)
        self.assertEqual(oracle_cvt_f32_f64(F64_POS_INF), F32_POS_INF)
        self.assertEqual(oracle_cvt_f32_f64(F64_NEG_INF), F32_NEG_INF)

    def test_cvt_f32_f64_nan_preservation(self):
        self.assertEqual(oracle_cvt_f32_f64(F64_QNAN), F32_QNAN)
        self.assertEqual(oracle_cvt_f32_f64(F64_SNAN), F32_QNAN)

class RegKind(enum.Enum):
    VGPR_PAIR = "vgpr_pair"
    VGPR_SINGLE = "vgpr_single"
    SGPR_PAIR = "sgpr_pair"
    SGPR_SINGLE = "sgpr_single"
    IMM_F64 = "imm_f64"
    IMM_I32 = "imm_i32"
    IMM_U32 = "imm_u32"


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
        self.vgprs = [[0] * 256 for _ in range(lane_count)]

    def get_f64(self, lane: int, reg_idx: int) -> int:
        lo = self.vgprs[lane][reg_idx] & MASK32
        hi = self.vgprs[lane][reg_idx + 1] & MASK32
        return (hi << 32) | lo

    def set_f64(self, lane: int, reg_idx: int, val: int):
        val &= MASK64
        self.vgprs[lane][reg_idx] = val & MASK32
        self.vgprs[lane][reg_idx + 1] = (val >> 32) & MASK32

    def get_u32(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

    def set_u32(self, lane: int, reg_idx: int, val: int):
        self.vgprs[lane][reg_idx] = val & MASK32


class VCvtF32F64AstInterpreterTests(unittest.TestCase):
    def test_cvt_f32_f64_ast_construction(self):
        ast = Instruction64AST(
            "v_cvt_f32_f64",
            Operand64(RegKind.VGPR_SINGLE, 2),
            Operand64(RegKind.VGPR_PAIR, 0),
        )
        self.assertEqual(ast.mnemonic, "v_cvt_f32_f64")
        self.assertEqual(ast.dst.index, 2)
        self.assertEqual(ast.src0.index, 0)

    def test_cvt_f32_f64_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 0, f64_to_u64(128.75))
        res_f32 = oracle_cvt_f32_f64(wave.get_f64(0, 0))
        wave.set_u32(0, 2, res_f32)
        self.assertEqual(wave.get_u32(0, 2), f32_to_u32(128.75))

class VCvtF32F64SimdIeeeTests(unittest.TestCase):
    def test_cvt_f32_f64_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x55555555
        for i in range(32):
            wave.set_u32(i, 2, 0xCAFEBABE)
            if (wave.exec_mask >> i) & 1:
                wave.set_u32(i, 2, oracle_cvt_f32_f64(f64_to_u64(float(i))))
        for i in range(32):
            if (0x55555555 >> i) & 1:
                self.assertEqual(wave.get_u32(i, 2), f32_to_u32(float(i)))
            else:
                self.assertEqual(wave.get_u32(i, 2), 0xCAFEBABE)

    def test_cvt_f32_f64_overflow_and_underflow(self):
        # Value exceeding float32 max (1e39 > 3.4028235e38) overflows to +inf
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(1e39)), F32_POS_INF)
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(-1e39)), F32_NEG_INF)
        # Value smaller than float32 min subnormal underflows to 0.0
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(1e-50)), F32_POS_ZERO)

class VCvtF32F64TranslatorTests(unittest.TestCase):
    def test_cvt_f32_f64_translation(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(3.141592653589793) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(3.141592653589793) >> 32) & MASK32:#010x}",
            "v_cvt_f32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f32_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFConvert %float", spv)

    def test_cvt_f32_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f32_f64", ["v_cvt_f32_f64"])

# -----------------------------------------------------------------------------
# Opcode 2: v_cvt_f64_i32
# -----------------------------------------------------------------------------
def oracle_cvt_f64_i32(i32_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_f64_i32 (signed int32 to double)."""
    i32_bits &= MASK32
    if i32_bits & 0x80000000:
        s_val = i32_bits - 0x100000000
    else:
        s_val = i32_bits
    return f64_to_u64(float(s_val))


class VCvtF64I32OracleTests(unittest.TestCase):
    def test_cvt_f64_i32_basic(self):
        self.assertEqual(oracle_cvt_f64_i32(0), F64_POS_ZERO)
        self.assertEqual(oracle_cvt_f64_i32(1), F64_POS_ONE)
        self.assertEqual(oracle_cvt_f64_i32(0xFFFFFFFF), F64_NEG_ONE)
        self.assertEqual(oracle_cvt_f64_i32(42), f64_to_u64(42.0))

    def test_cvt_f64_i32_extremes(self):
        # -2147483648 (0x80000000)
        self.assertEqual(oracle_cvt_f64_i32(0x80000000), f64_to_u64(-2147483648.0))
        # 2147483647 (0x7FFFFFFF)
        self.assertEqual(oracle_cvt_f64_i32(0x7FFFFFFF), f64_to_u64(2147483647.0))

class VCvtF64I32AstInterpreterTests(unittest.TestCase):
    def test_cvt_f64_i32_ast_construction(self):
        ast = Instruction64AST(
            "v_cvt_f64_i32",
            Operand64(RegKind.VGPR_PAIR, 0),
            Operand64(RegKind.VGPR_SINGLE, 2),
        )
        self.assertEqual(ast.mnemonic, "v_cvt_f64_i32")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 2)

    def test_cvt_f64_i32_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_u32(0, 2, 0xFFFFFFFE)  # -2 in two's complement
        res = oracle_cvt_f64_i32(wave.get_u32(0, 2))
        wave.set_f64(0, 0, res)
        self.assertEqual(wave.get_f64(0, 0), f64_to_u64(-2.0))

class VCvtF64I32SimdIeeeTests(unittest.TestCase):
    def test_cvt_f64_i32_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x0000FFFF
        for i in range(32):
            wave.set_f64(i, 0, 0x1111222233334444)
            if (wave.exec_mask >> i) & 1:
                val = i - 16
                wave.set_f64(i, 0, oracle_cvt_f64_i32(val & MASK32))
        for i in range(16):
            self.assertEqual(wave.get_f64(i, 0), f64_to_u64(float(i - 16)))
        for i in range(16, 32):
            self.assertEqual(wave.get_f64(i, 0), 0x1111222233334444)

    def test_cvt_f64_i32_exactness(self):
        # 53 mantissa bits accurately hold all 32-bit signed integers
        for val in (-1000000, -1, 0, 1, 1000000):
            res_bits = oracle_cvt_f64_i32(val & MASK32)
            self.assertEqual(u64_to_f64(res_bits), float(val))

class VCvtF64I32TranslatorTests(unittest.TestCase):
    def test_cvt_f64_i32_translation_and_interp(self):
        lines = [
            "v_mov_b32 v2, 42",
            "v_cvt_f64_i32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_i32_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertSToF %double", spv)
        res = spirv_interp.run(spv)
        got = (res.vgpr[0] & MASK32) | ((res.vgpr[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(42.0))

    def test_cvt_f64_i32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f64_i32", ["v_cvt_f64_i32"])

# -----------------------------------------------------------------------------
# Opcode 3: v_cvt_f64_u32
# -----------------------------------------------------------------------------
def oracle_cvt_f64_u32(u32_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_f64_u32 (unsigned uint32 to double)."""
    u_val = u32_bits & MASK32
    return f64_to_u64(float(u_val))


class VCvtF64U32OracleTests(unittest.TestCase):
    def test_cvt_f64_u32_basic(self):
        self.assertEqual(oracle_cvt_f64_u32(0), F64_POS_ZERO)
        self.assertEqual(oracle_cvt_f64_u32(1), F64_POS_ONE)
        self.assertEqual(oracle_cvt_f64_u32(100), f64_to_u64(100.0))

    def test_cvt_f64_u32_large_unsigned(self):
        # 0xFFFFFFFF = 4294967295.0
        self.assertEqual(oracle_cvt_f64_u32(0xFFFFFFFF), f64_to_u64(4294967295.0))
        # 0x80000000 = 2147483648.0
        self.assertEqual(oracle_cvt_f64_u32(0x80000000), f64_to_u64(2147483648.0))

class VCvtF64U32AstInterpreterTests(unittest.TestCase):
    def test_cvt_f64_u32_ast_construction(self):
        ast = Instruction64AST(
            "v_cvt_f64_u32",
            Operand64(RegKind.VGPR_PAIR, 0),
            Operand64(RegKind.VGPR_SINGLE, 2),
        )
        self.assertEqual(ast.mnemonic, "v_cvt_f64_u32")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 2)

    def test_cvt_f64_u32_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_u32(0, 2, 3000000000)
        res = oracle_cvt_f64_u32(wave.get_u32(0, 2))
        wave.set_f64(0, 0, res)
        self.assertEqual(wave.get_f64(0, 0), f64_to_u64(3000000000.0))

class VCvtF64U32SimdIeeeTests(unittest.TestCase):
    def test_cvt_f64_u32_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0xF0F0F0F0
        for i in range(32):
            wave.set_f64(i, 0, 0xAAAAAAAA55555555)
            if (wave.exec_mask >> i) & 1:
                wave.set_f64(i, 0, oracle_cvt_f64_u32(i * 1000000))
        for i in range(32):
            if (0xF0F0F0F0 >> i) & 1:
                self.assertEqual(wave.get_f64(i, 0), f64_to_u64(float(i * 1000000)))
            else:
                self.assertEqual(wave.get_f64(i, 0), 0xAAAAAAAA55555555)

    def test_cvt_f64_u32_zero_sign(self):
        # 0 u32 converts to +0.0, never -0.0
        res = oracle_cvt_f64_u32(0)
        self.assertEqual(res, F64_POS_ZERO)
        self.assertNotEqual(res, F64_NEG_ZERO)

class VCvtF64U32TranslatorTests(unittest.TestCase):
    def test_cvt_f64_u32_translation_and_interp(self):
        lines = [
            "v_mov_b32 v2, 100",
            "v_cvt_f64_u32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_u32_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertUToF %double", spv)
        res = spirv_interp.run(spv)
        got = (res.vgpr[0] & MASK32) | ((res.vgpr[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(100.0))

    def test_cvt_f64_u32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f64_u32", ["v_cvt_f64_u32"])

# -----------------------------------------------------------------------------
# Opcode 4: v_rcp_f64
# -----------------------------------------------------------------------------
def oracle_rcp_f64(bits: int) -> int:
    """Reference mathematical oracle for v_rcp_f64 (double reciprocal 1.0 / x)."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_INF
    if bits == F64_NEG_ZERO:
        return F64_NEG_INF
    if bits == F64_POS_INF:
        return F64_POS_ZERO
    if bits == F64_NEG_INF:
        return F64_NEG_ZERO
    val = u64_to_f64(bits)
    try:
        res = 1.0 / val
        return f64_to_u64(res)
    except ZeroDivisionError:
        return F64_POS_INF if val >= 0 else F64_NEG_INF
    except OverflowError:
        return F64_POS_INF if val > 0 else F64_NEG_INF


class VRcpF64OracleTests(unittest.TestCase):
    def test_rcp_f64_basic(self):
        self.assertEqual(oracle_rcp_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_rcp_f64(f64_to_u64(2.0)), f64_to_u64(0.5))
        self.assertEqual(oracle_rcp_f64(f64_to_u64(-2.0)), f64_to_u64(-0.5))
        self.assertEqual(oracle_rcp_f64(f64_to_u64(4.0)), f64_to_u64(0.25))

    def test_rcp_f64_zeros_and_infinities(self):
        self.assertEqual(oracle_rcp_f64(F64_POS_ZERO), F64_POS_INF)
        self.assertEqual(oracle_rcp_f64(F64_NEG_ZERO), F64_NEG_INF)
        self.assertEqual(oracle_rcp_f64(F64_POS_INF), F64_POS_ZERO)
        self.assertEqual(oracle_rcp_f64(F64_NEG_INF), F64_NEG_ZERO)

    def test_rcp_f64_nan_preservation(self):
        self.assertTrue(is_nan_u64(oracle_rcp_f64(F64_QNAN)))
        self.assertTrue(is_nan_u64(oracle_rcp_f64(F64_SNAN)))

class VRcpF64AstInterpreterTests(unittest.TestCase):
    def test_rcp_f64_ast_construction(self):
        ast = Instruction64AST(
            "v_rcp_f64",
            Operand64(RegKind.VGPR_PAIR, 0),
            Operand64(RegKind.VGPR_PAIR, 2),
        )
        self.assertEqual(ast.mnemonic, "v_rcp_f64")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 2)

    def test_rcp_f64_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 2, f64_to_u64(8.0))
        res = oracle_rcp_f64(wave.get_f64(0, 2))
        wave.set_f64(0, 0, res)
        self.assertEqual(wave.get_f64(0, 0), f64_to_u64(0.125))

class VRcpF64SimdIeeeTests(unittest.TestCase):
    def test_rcp_f64_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x00FF00FF
        for i in range(32):
            wave.set_f64(i, 0, 0x123456789ABCDEF0)
            if (wave.exec_mask >> i) & 1:
                denom = float(i + 1)
                wave.set_f64(i, 0, oracle_rcp_f64(f64_to_u64(denom)))
        for i in range(32):
            if (0x00FF00FF >> i) & 1:
                self.assertEqual(wave.get_f64(i, 0), f64_to_u64(1.0 / float(i + 1)))
            else:
                self.assertEqual(wave.get_f64(i, 0), 0x123456789ABCDEF0)

    def test_rcp_f64_boundary_extremes(self):
        # Very large float underflows to +0.0
        self.assertEqual(oracle_rcp_f64(F64_MAX_NORMAL), f64_to_u64(1.0 / u64_to_f64(F64_MAX_NORMAL)))
        # Very small normal inverts to large number
        res = oracle_rcp_f64(F64_MIN_NORMAL)
        self.assertTrue(u64_to_f64(res) > 1e300)

class VRcpF64TranslatorTests(unittest.TestCase):
    def test_rcp_f64_translation(self):
        lines = [
            f"v_mov_b32 v2, {F64_POS_TWO & MASK32:#010x}",
            f"v_mov_b32 v3, {(F64_POS_TWO >> 32) & MASK32:#010x}",
            "v_rcp_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rcp_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)

    def test_rcp_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_rcp_f64", ["v_rcp_f64"])

# -----------------------------------------------------------------------------
# Opcode 5: v_rsq_f64
# -----------------------------------------------------------------------------
def oracle_rsq_f64(bits: int) -> int:
    """Reference mathematical oracle for v_rsq_f64 (double reciprocal square root 1.0 / sqrt(x))."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_INF
    if bits == F64_NEG_ZERO:
        return F64_NEG_INF
    if bits == F64_POS_INF:
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_QNAN
    try:
        res = 1.0 / math.sqrt(val)
        return f64_to_u64(res)
    except ZeroDivisionError:
        return F64_POS_INF
    except OverflowError:
        return F64_POS_INF


class VRsqF64OracleTests(unittest.TestCase):
    def test_rsq_f64_basic(self):
        self.assertEqual(oracle_rsq_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_rsq_f64(f64_to_u64(4.0)), f64_to_u64(0.5))
        self.assertEqual(oracle_rsq_f64(f64_to_u64(16.0)), f64_to_u64(0.25))
        self.assertEqual(oracle_rsq_f64(f64_to_u64(0.25)), f64_to_u64(2.0))

    def test_rsq_f64_zeros_and_inf(self):
        self.assertEqual(oracle_rsq_f64(F64_POS_ZERO), F64_POS_INF)
        self.assertEqual(oracle_rsq_f64(F64_NEG_ZERO), F64_NEG_INF)
        self.assertEqual(oracle_rsq_f64(F64_POS_INF), F64_POS_ZERO)

    def test_rsq_f64_negative_nan(self):
        self.assertTrue(is_nan_u64(oracle_rsq_f64(f64_to_u64(-4.0))))
        self.assertTrue(is_nan_u64(oracle_rsq_f64(F64_NEG_INF)))
        self.assertTrue(is_nan_u64(oracle_rsq_f64(F64_QNAN)))

class VRsqF64AstInterpreterTests(unittest.TestCase):
    def test_rsq_f64_ast_construction(self):
        ast = Instruction64AST(
            "v_rsq_f64",
            Operand64(RegKind.VGPR_PAIR, 0),
            Operand64(RegKind.VGPR_PAIR, 2),
        )
        self.assertEqual(ast.mnemonic, "v_rsq_f64")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 2)

    def test_rsq_f64_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 2, f64_to_u64(100.0))
        res = oracle_rsq_f64(wave.get_f64(0, 2))
        wave.set_f64(0, 0, res)
        self.assertEqual(wave.get_f64(0, 0), f64_to_u64(0.1))

class VRsqF64SimdIeeeTests(unittest.TestCase):
    def test_rsq_f64_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x33333333
        powers_of_four = [float(4**i) for i in range(32)]
        for i in range(32):
            wave.set_f64(i, 0, 0xFEEDFACECAFEBABE)
            if (wave.exec_mask >> i) & 1:
                wave.set_f64(i, 0, oracle_rsq_f64(f64_to_u64(powers_of_four[i])))
        for i in range(32):
            if (0x33333333 >> i) & 1:
                expected = 1.0 / math.sqrt(powers_of_four[i])
                self.assertEqual(wave.get_f64(i, 0), f64_to_u64(expected))
            else:
                self.assertEqual(wave.get_f64(i, 0), 0xFEEDFACECAFEBABE)

    def test_rsq_f64_near_zero(self):
        small = 1e-100
        res = oracle_rsq_f64(f64_to_u64(small))
        self.assertEqual(u64_to_f64(res), 1e50)

class VRsqF64TranslatorTests(unittest.TestCase):
    def test_rsq_f64_translation(self):
        lines = [
            f"v_mov_b32 v2, {f64_to_u64(4.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(4.0) >> 32) & MASK32:#010x}",
            "v_rsq_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rsq_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("InverseSqrt", spv)

    def test_rsq_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_rsq_f64", ["v_rsq_f64"])

# -----------------------------------------------------------------------------
# Opcode 6: v_sqrt_f64
# -----------------------------------------------------------------------------
def oracle_sqrt_f64(bits: int) -> int:
    """Reference mathematical oracle for v_sqrt_f64 (double square root sqrt(x))."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_ZERO
    if bits == F64_NEG_ZERO:
        return F64_NEG_ZERO
    if bits == F64_POS_INF:
        return F64_POS_INF
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_QNAN
    try:
        res = math.sqrt(val)
        return f64_to_u64(res)
    except ValueError:
        return F64_QNAN


class VSqrtF64OracleTests(unittest.TestCase):
    def test_sqrt_f64_basic(self):
        self.assertEqual(oracle_sqrt_f64(F64_POS_ZERO), F64_POS_ZERO)
        self.assertEqual(oracle_sqrt_f64(F64_NEG_ZERO), F64_NEG_ZERO)
        self.assertEqual(oracle_sqrt_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(4.0)), f64_to_u64(2.0))
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(9.0)), f64_to_u64(3.0))
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(2.25)), f64_to_u64(1.5))

    def test_sqrt_f64_inf_and_negative(self):
        self.assertEqual(oracle_sqrt_f64(F64_POS_INF), F64_POS_INF)
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(f64_to_u64(-1.0))))
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(F64_NEG_INF)))
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(F64_QNAN)))

class VSqrtF64AstInterpreterTests(unittest.TestCase):
    def test_sqrt_f64_ast_construction(self):
        ast = Instruction64AST(
            "v_sqrt_f64",
            Operand64(RegKind.VGPR_PAIR, 0),
            Operand64(RegKind.VGPR_PAIR, 2),
        )
        self.assertEqual(ast.mnemonic, "v_sqrt_f64")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 2)

    def test_sqrt_f64_scalar_execution(self):
        wave = SimdWaveState64(lane_count=1)
        wave.set_f64(0, 2, f64_to_u64(144.0))
        res = oracle_sqrt_f64(wave.get_f64(0, 2))
        wave.set_f64(0, 0, res)
        self.assertEqual(wave.get_f64(0, 0), f64_to_u64(12.0))

class VSqrtF64SimdIeeeTests(unittest.TestCase):
    def test_sqrt_f64_simd_exec_masking(self):
        wave = SimdWaveState64(lane_count=32)
        wave.exec_mask = 0x0F0F0F0F
        for i in range(32):
            wave.set_f64(i, 0, 0x5555AAAA5555AAAA)
            if (wave.exec_mask >> i) & 1:
                sq = float(i * i)
                wave.set_f64(i, 0, oracle_sqrt_f64(f64_to_u64(sq)))
        for i in range(32):
            if (0x0F0F0F0F >> i) & 1:
                self.assertEqual(wave.get_f64(i, 0), f64_to_u64(float(i)))
            else:
                self.assertEqual(wave.get_f64(i, 0), 0x5555AAAA5555AAAA)

    def test_sqrt_f64_subnormal_precision(self):
        # Square of 2^-500 is 2^-1000, which is a small positive normal float64
        val = 2.0**-500
        sq_val = val * val
        res = oracle_sqrt_f64(f64_to_u64(sq_val))
        self.assertEqual(u64_to_f64(res), val)


if __name__ == "__main__":
    unittest.main()
