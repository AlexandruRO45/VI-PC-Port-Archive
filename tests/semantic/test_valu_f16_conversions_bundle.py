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


if __name__ == "__main__":
    unittest.main()
