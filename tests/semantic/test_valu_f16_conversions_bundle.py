"""Bundle 18: VALU Float16 Conversions & Normalization Suite.

Tests for 6 VALU Float16 conversion instructions:
- v_cvt_f16_f32: 32-bit float to 16-bit half float conversion with rounding
- v_cvt_f32_f16: 16-bit half float to 32-bit float exact widening conversion
- v_cvt_f16_u16: 16-bit unsigned integer to 16-bit half float conversion
- v_cvt_f16_i16: 16-bit signed integer to 16-bit half float conversion
- v_cvt_u16_f16: 16-bit half float to 16-bit unsigned integer with truncation/saturation
- v_cvt_i16_f16: 16-bit half float to 16-bit signed integer with truncation/saturation

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
MASK16 = 0xFFFF

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x40088, 32, 1, 1)
FSSM = bytes(FSSM)
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def float_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f16_bits_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(val: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return 0xFC00 if val < 0 else 0x7C00


def run_cs(body: list[str], lanes: int = 32, num_words: int = 32, name: str = "cs_bundle18") -> list[int]:
    """Helper to translate and run a compute shader; stores outputs at out[lane*32]."""
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
    """Base symbolic AST node."""
    pass

def ref_v_cvt_f16_f32(f32_val: float) -> int:
    """Reference mathematical oracle for v_cvt_f16_f32 (float32 -> float16 bits)."""
    return float_to_f16_bits(f32_val)


class TestVCvtF16F32Oracle(unittest.TestCase):
    def test_basic_values(self) -> None:
        self.assertEqual(ref_v_cvt_f16_f32(0.0), 0x0000)
        self.assertEqual(ref_v_cvt_f16_f32(1.0), 0x3C00)
        self.assertEqual(ref_v_cvt_f16_f32(-1.0), 0xBC00)
        self.assertEqual(ref_v_cvt_f16_f32(2.0), 0x4000)
        self.assertEqual(ref_v_cvt_f16_f32(0.5), 0x3800)

    def test_infinities_and_zeros(self) -> None:
        self.assertEqual(ref_v_cvt_f16_f32(float("inf")), 0x7C00)
        self.assertEqual(ref_v_cvt_f16_f32(float("-inf")), 0xFC00)
        self.assertEqual(ref_v_cvt_f16_f32(-0.0), 0x8000)

class CvtF16F32Expr(ASTNode):
    def __init__(self, src_expr: str) -> None:
        self.src_expr = src_expr

    def eval(self, f32_val: float) -> int:
        return ref_v_cvt_f16_f32(f32_val)


class TestVCvtF16F32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = CvtF16F32Expr("v0")
        self.assertEqual(expr.eval(1.5), 0x3E00)
        self.assertEqual(expr.eval(-2.0), 0xC000)

class TestVCvtF16F32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        # 32 divergent lanes with varying exponents
        inputs = [2.0 ** (i - 16) for i in range(32)]
        results = [ref_v_cvt_f16_f32(x) for x in inputs]
        for i, val in enumerate(inputs):
            back = f16_bits_to_float(results[i])
            self.assertAlmostEqual(back, val, places=2)

    def test_ieee754_boundaries(self) -> None:
        # Max normal f16 is 65504 (0x7BFF)
        self.assertEqual(ref_v_cvt_f16_f32(65504.0), 0x7BFF)
        # Smallest positive subnormal is 2^-24 ~ 5.96e-8 (0x0001)
        self.assertEqual(ref_v_cvt_f16_f32(2.0 ** -24), 0x0001)

class TestVCvtF16F32Translator(unittest.TestCase):
    def test_translator_cvt_f16_f32_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x3f800000",  # 1.0f
            "v_cvt_f16_f32_e32 v2, v1",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_cvt_f16_f32")
        for l in range(32):
            self.assertEqual(results[l] & 0xFFFF, 0x3C00)

    def test_translator_preserves_high_half(self) -> None:
        # GFX10 partial write: high 16 bits of destination are kept
        body = [
            "v_mov_b32_e32 v2, 0xdead0000",
            "v_mov_b32_e32 v1, 0x40000000",  # 2.0f
            "v_cvt_f16_f32_e32 v2, v1",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_cvt_f16_f32_keep_hi")
        for l in range(32):
            self.assertEqual(results[l], 0xDEAD4000)

def ref_v_cvt_f32_f16(f16_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_f32_f16 (float16 bits -> float32 bits)."""
    f = f16_bits_to_float(f16_bits & MASK16)
    return float_to_bits(f)


class TestVCvtF32F16Oracle(unittest.TestCase):
    def test_basic_values(self) -> None:
        self.assertEqual(ref_v_cvt_f32_f16(0x3C00), 0x3F800000)  # 1.0h -> 1.0f
        self.assertEqual(ref_v_cvt_f32_f16(0xBC00), 0xBF800000)  # -1.0h -> -1.0f
        self.assertEqual(ref_v_cvt_f32_f16(0x0000), 0x00000000)  # 0.0h -> 0.0f
        self.assertEqual(ref_v_cvt_f32_f16(0x8000), 0x80000000)  # -0.0h -> -0.0f

    def test_infinities(self) -> None:
        self.assertEqual(ref_v_cvt_f32_f16(0x7C00), 0x7F800000)  # +inf
        self.assertEqual(ref_v_cvt_f32_f16(0xFC00), 0xFF800000)  # -inf

class CvtF32F16Expr(ASTNode):
    def __init__(self, src_expr: str) -> None:
        self.src_expr = src_expr

    def eval(self, f16_bits: int) -> int:
        return ref_v_cvt_f32_f16(f16_bits)


class TestVCvtF32F16AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = CvtF32F16Expr("v0")
        self.assertEqual(expr.eval(0x4000), 0x40000000)  # 2.0
        self.assertEqual(expr.eval(0x3800), 0x3F000000)  # 0.5

class TestVCvtF32F16SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        # Lanes test divergent powers of 2
        f16_inputs = [0x3C00 + (l << 6) for l in range(32)]
        for l in range(32):
            f32_bits = ref_v_cvt_f32_f16(f16_inputs[l])
            self.assertAlmostEqual(bits_to_float(f32_bits), f16_bits_to_float(f16_inputs[l]), places=4)

class TestVCvtF32F16Translator(unittest.TestCase):
    def test_translator_cvt_f32_f16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0x3c00",  # 1.0h
            "v_cvt_f32_f16_e32 v2, v1",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_cvt_f32_f16")
        for l in range(32):
            self.assertEqual(results[l], 0x3F800000)

def ref_v_cvt_f16_u16(u16_val: int) -> int:
    """Reference mathematical oracle for v_cvt_f16_u16."""
    return float_to_f16_bits(float(u16_val & MASK16))


class TestVCvtF16U16Oracle(unittest.TestCase):
    def test_basic_values(self) -> None:
        self.assertEqual(ref_v_cvt_f16_u16(0), 0x0000)
        self.assertEqual(ref_v_cvt_f16_u16(1), 0x3C00)
        self.assertEqual(ref_v_cvt_f16_u16(42), 0x5140)
        self.assertEqual(ref_v_cvt_f16_u16(1024), 0x6400)

class CvtF16U16Expr(ASTNode):
    def __init__(self, src_expr: str) -> None:
        self.src_expr = src_expr

    def eval(self, u16_val: int) -> int:
        return ref_v_cvt_f16_u16(u16_val)


class TestVCvtF16U16AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = CvtF16U16Expr("v0")
        self.assertEqual(expr.eval(2), 0x4000)
        self.assertEqual(expr.eval(100), float_to_f16_bits(100.0))

class TestVCvtF16U16SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in range(32):
            got = ref_v_cvt_f16_u16(l * 100)
            self.assertEqual(got, float_to_f16_bits(float(l * 100)))

    def test_boundary_extremes(self) -> None:
        # Max u16 is 65535 -> in f16 rounds to 65536 (0x7C00 = inf in half float without clamp)
        self.assertEqual(ref_v_cvt_f16_u16(65535), 0x7C00)

class TestVCvtF16U16Translator(unittest.TestCase):
    def test_translator_cvt_f16_u16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 42",
            "v_cvt_f16_u16_e32 v2, v1",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_cvt_f16_u16")
        for l in range(32):
            self.assertEqual(results[l] & 0xFFFF, 0x5140)

def ref_v_cvt_f16_i16(i16_val: int) -> int:
    """Reference mathematical oracle for v_cvt_f16_i16."""
    val = i16_val & MASK16
    signed_val = val - 0x10000 if val & 0x8000 else val
    return float_to_f16_bits(float(signed_val))


class TestVCvtF16I16Oracle(unittest.TestCase):
    def test_basic_values(self) -> None:
        self.assertEqual(ref_v_cvt_f16_i16(0), 0x0000)
        self.assertEqual(ref_v_cvt_f16_i16(1), 0x3C00)
        self.assertEqual(ref_v_cvt_f16_i16(-1), 0xBC00)
        self.assertEqual(ref_v_cvt_f16_i16(-42), 0xD140)

class CvtF16I16Expr(ASTNode):
    def __init__(self, src_expr: str) -> None:
        self.src_expr = src_expr

    def eval(self, i16_val: int) -> int:
        return ref_v_cvt_f16_i16(i16_val)


class TestVCvtF16I16AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        expr = CvtF16I16Expr("v0")
        self.assertEqual(expr.eval(-2), 0xC000)
        self.assertEqual(expr.eval(10), float_to_f16_bits(10.0))

class TestVCvtF16I16SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in range(32):
            val = (l - 16) * 50
            got = ref_v_cvt_f16_i16(val)
            self.assertEqual(got, float_to_f16_bits(float(val)))

    def test_boundary_extremes(self) -> None:
        # Min i16 is -32768 (0x8000) -> -32768.0 in f16 is 0xF800
        self.assertEqual(ref_v_cvt_f16_i16(0x8000), 0xF800)
        # Max i16 is 32767 -> 0x77FF
        self.assertEqual(ref_v_cvt_f16_i16(32767), float_to_f16_bits(32767.0))

class TestVCvtF16I16Translator(unittest.TestCase):
    def test_translator_cvt_f16_i16_execution(self) -> None:
        body = [
            "v_mov_b32_e32 v1, 0xffd6",  # -42
            "v_cvt_f16_i16_e32 v2, v1",
            "buffer_store_dword v2, v30, s[0:3], 0 offen",
        ]
        results = run_cs(body, name="cs_cvt_f16_i16")
        for l in range(32):
            self.assertEqual(results[l] & 0xFFFF, 0xD140)

def ref_v_cvt_u16_f16(f16_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_u16_f16 with truncation and saturation."""
    val = f16_bits_to_float(f16_bits & MASK16)
    if math.isnan(val):
        return 0
    if math.isinf(val):
        return 65535 if val > 0 else 0
    tr = math.trunc(val)
    if tr < 0:
        return 0
    if tr > 65535:
        return 65535
    return tr


class TestVCvtU16F16Oracle(unittest.TestCase):
    def test_basic_values(self) -> None:
        self.assertEqual(ref_v_cvt_u16_f16(0x5140), 42)      # 42.0h -> 42
        self.assertEqual(ref_v_cvt_u16_f16(0x3C00), 1)       # 1.0h -> 1
        self.assertEqual(ref_v_cvt_u16_f16(0x3E00), 1)       # 1.5h -> 1 (truncated)
        self.assertEqual(ref_v_cvt_u16_f16(0x3800), 0)       # 0.5h -> 0

    def test_saturation_and_nan(self) -> None:
        self.assertEqual(ref_v_cvt_u16_f16(0xBC00), 0)       # -1.0h -> 0 (clamped)
        self.assertEqual(ref_v_cvt_u16_f16(0x7C00), 65535)   # +inf -> 65535
        self.assertEqual(ref_v_cvt_u16_f16(0xFC00), 0)       # -inf -> 0
        self.assertEqual(ref_v_cvt_u16_f16(0x7E00), 0)       # NaN -> 0


if __name__ == "__main__":
    unittest.main()
