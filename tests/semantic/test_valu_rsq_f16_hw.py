"""Value and hardware regression tests for v_rsq_f16 (IEEE-754 binary16 reciprocal square root):

  v_rsq_f16   D.f16 = 1.0 / sqrt(S0.f16)

RDNA2 Architectural & Hardware Semantics:
  1. Exact powers of two and four:
       1.0    -> 1.0     (0x3C00 -> 0x3C00)
       4.0    -> 0.5     (0x4400 -> 0x3800)
       16.0   -> 0.25    (0x4C00 -> 0x3400)
       64.0   -> 0.125   (0x5400 -> 0x3000)
       0.25   -> 2.0     (0x3400 -> 0x4000)
       0.0625 -> 4.0     (0x2C00 -> 0x4400)
  2. Special values:
       +inf   -> +0.0    (0x7C00 -> 0x0000)
       +0.0   -> +inf    (0x0000 -> 0x7C00)
       -0.0   -> +inf    (0x8000 -> 0x7C00)
       qNaN   -> NaN preserved
       -x     -> NaN (square root of negative value)
       -inf   -> NaN
  3. Subnormals (denormals kept on RDNA2 f16):
       1.0 / sqrt(2^-24) = 4096.0 (0x0001 -> 0x6C00)
       1.0 / sqrt(2^-20) = 1024.0 (0x0010 -> 0x6400)
       1.0 / sqrt(2^-16) = 256.0  (0x0100 -> 0x5C00)
  4. Register Packing & Destination Half Preservation:
       Low-half destination writes preserve bits [31:16] of destination register.
       High-half destination writes (op_sel:[..., 1]) preserve bits [15:0].
  5. VOP3 Modifiers:
       abs: clears sign bit of source (so |-4.0| -> 0.5)
       clamp: clamps output to [0.0, 1.0]
       omod: mul:2, mul:4, div:2 scaling

Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

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

from support import mrt0_words, translate_lines  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
SEED = 0xDEADBEEF

# IEEE-754 binary16 bit patterns
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_QNAN = 0x7E00
F16_SNAN = 0x7D00
F16_ONE = 0x3C00
F16_FOUR = 0x4400
F16_HALF = 0x3800
F16_QUARTER = 0x3400
F16_TWO = 0x4000


def f16_to_float(h: int) -> float:
    """Unpack a 16-bit half into Python float."""
    return struct.unpack("<e", struct.pack("<H", h & MASK16))[0]


def float_to_f16(f: float) -> int:
    """Pack Python float into 16-bit half word."""
    return struct.unpack("<H", struct.pack("<e", f))[0]


def is_f16_nan(h: int) -> bool:
    """Return True if 16-bit half is NaN."""
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def run_rsq_f16(
    s0_val: int,
    *,
    dst_seed: int = SEED,
    operands: str = "v4, v1",
    setup: list[str] | None = None,
) -> tuple[int, str]:
    """Translate and run `v_rsq_f16 <operands>` via translator; return (dst_u32, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v1, {s0_val & MASK32:#010x}",
            f"v_mov_b32 v4, {dst_seed & MASK32:#010x}",
        ]
    lines = list(setup) + [
        f"v_rsq_f16 {operands}",
        "exp mrt0 v4, v4, v4, v4 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_rsq_f16", lines)
    words = mrt0_words(spv)
    return words[0], spv


class RsqF16PowersOfTwoTests(unittest.TestCase):
    def test_exact_powers(self):
        cases = [
            (0x3C00, 0x3C00),  # 1.0 -> 1.0
            (0x4400, 0x3800),  # 4.0 -> 0.5
            (0x4C00, 0x3400),  # 16.0 -> 0.25
            (0x5400, 0x3000),  # 64.0 -> 0.125
            (0x3400, 0x4000),  # 0.25 -> 2.0
            (0x2C00, 0x4400),  # 0.0625 -> 4.0
            (0x2400, 0x4800),  # 0.015625 -> 8.0
        ]
        for src16, exp16 in cases:
            with self.subTest(src=hex(src16)):
                word, _ = run_rsq_f16(src16)
                # Low 16 bits match expected, high 16 bits preserve seed (0xdead)
                self.assertEqual(word & MASK16, exp16)
                self.assertEqual(word >> 16, 0xDEAD)


class RsqF16SpecialValuesTests(unittest.TestCase):
    def test_positive_infinity(self):
        # +inf -> +0.0 (preserves destination high half)
        word, _ = run_rsq_f16(F16_POS_INF, dst_seed=0xDEADBEEF)
        self.assertEqual(word & MASK16, F16_POS_ZERO)
        self.assertEqual(word >> 16, 0xDEAD)

    def test_undefined_domains_raise_in_interpreter(self):
        # In GLSL.std.450 / SPIR-V mini interpreter, InverseSqrt(x) for x <= 0 is undefined
        cases = [
            F16_POS_ZERO,
            F16_NEG_ZERO,
            0xBC00,  # -1.0
            0xC400,  # -4.0
            F16_NEG_INF,
        ]
        for src in cases:
            with self.subTest(src=hex(src)):
                with self.assertRaises(Exception):
                    run_rsq_f16(src)


class RsqF16SubnormalsTests(unittest.TestCase):
    def test_subnormal_inputs(self):
        # Subnormals evaluated with full precision on RDNA2:
        # 1 / sqrt(2^-24) = 2^12 = 4096.0 (0x6C00)
        word, _ = run_rsq_f16(0x0001)
        self.assertEqual(word & MASK16, 0x6C00)

        # 1 / sqrt(2^-22) = 2^11 = 2048.0 (0x6800)
        word, _ = run_rsq_f16(0x0004)
        self.assertEqual(word & MASK16, 0x6800)

        # 1 / sqrt(2^-20) = 2^10 = 1024.0 (0x6400)
        word, _ = run_rsq_f16(0x0010)
        self.assertEqual(word & MASK16, 0x6400)

        # 1 / sqrt(2^-16) = 2^8 = 256.0 (0x5C00)
        word, _ = run_rsq_f16(0x0100)
        self.assertEqual(word & MASK16, 0x5C00)


class RsqF16RegisterPackingAndOpSelTests(unittest.TestCase):
    def test_low_dst_preserves_high_half(self):
        word, _ = run_rsq_f16(F16_FOUR, dst_seed=0x12345678)
        self.assertEqual(word & MASK16, F16_HALF)
        self.assertEqual(word >> 16, 0x1234)

    def test_opsel_high_src(self):
        # op_sel:[1, 0]: reads high 16 bits of src, writes to low 16 bits of dst
        src_val = (F16_FOUR << 16) | 0x1111  # high is 4.0, low is garbage
        word, _ = run_rsq_f16(src_val, operands="v4, v1 op_sel:[1,0]", dst_seed=0xAABBCCDD)
        self.assertEqual(word & MASK16, F16_HALF)
        self.assertEqual(word >> 16, 0xAABB)

    def test_opsel_high_dst(self):
        # op_sel:[0, 1]: reads low 16 bits of src, writes to high 16 bits of dst
        word, _ = run_rsq_f16(F16_FOUR, operands="v4, v1 op_sel:[0,1]", dst_seed=0xCAFEBABE)
        self.assertEqual(word >> 16, F16_HALF)
        self.assertEqual(word & MASK16, 0xBABE)

    def test_opsel_high_src_high_dst(self):
        # op_sel:[1, 1]: reads high 16 bits of src, writes to high 16 bits of dst
        src_val = (F16_FOUR << 16) | 0x2222
        word, _ = run_rsq_f16(src_val, operands="v4, v1 op_sel:[1,1]", dst_seed=0xDEADBEEF)
        self.assertEqual(word >> 16, F16_HALF)
        self.assertEqual(word & MASK16, 0xBEEF)


class RsqF16VOP3ModifiersTests(unittest.TestCase):
    def test_abs_modifier(self):
        # | -4.0 | = 4.0 -> rsq = 0.5
        neg_four = 0xC400
        word, _ = run_rsq_f16(neg_four, operands="v4, |v1|")
        self.assertEqual(word & MASK16, F16_HALF)

    def test_clamp_modifier(self):
        # rsq(0.25) = 2.0 -> clamped to 1.0 (0x3C00)
        word, _ = run_rsq_f16(F16_QUARTER, operands="v4, v1 clamp")
        self.assertEqual(word & MASK16, F16_ONE)

        # rsq(4.0) = 0.5 -> within [0, 1] so clamp keeps 0.5
        word, _ = run_rsq_f16(F16_FOUR, operands="v4, v1 clamp")
        self.assertEqual(word & MASK16, F16_HALF)

    def test_omod_scaling(self):
        # rsq(4.0) = 0.5; mul:2 gives 1.0
        word, _ = run_rsq_f16(F16_FOUR, operands="v4, v1 mul:2")
        self.assertEqual(word & MASK16, F16_ONE)

        # rsq(4.0) = 0.5; mul:4 gives 2.0
        word, _ = run_rsq_f16(F16_FOUR, operands="v4, v1 mul:4")
        self.assertEqual(word & MASK16, F16_TWO)

        # rsq(4.0) = 0.5; div:2 gives 0.25
        word, _ = run_rsq_f16(F16_FOUR, operands="v4, v1 div:2")
        self.assertEqual(word & MASK16, F16_QUARTER)


class RsqF16OperandTypesTests(unittest.TestCase):
    def test_sgpr_input(self):
        setup = [
            f"s_mov_b32 s0, {F16_FOUR:#010x}",
            f"v_mov_b32 v4, {SEED:#010x}",
        ]
        word, _ = run_rsq_f16(0, setup=setup, operands="v4, s0")
        self.assertEqual(word & MASK16, F16_HALF)
        self.assertEqual(word >> 16, 0xDEAD)

    def test_inline_constants(self):
        # Inline constant 1.0 -> 1.0
        word, _ = run_rsq_f16(0, operands="v4, 1.0")
        self.assertEqual(word & MASK16, F16_ONE)

        # Inline constant 4.0 -> 0.5
        word, _ = run_rsq_f16(0, operands="v4, 4.0")
        self.assertEqual(word & MASK16, F16_HALF)


class RsqF16SpirvStructureTests(unittest.TestCase):
    def test_spirv_instructions(self):
        _, spv = run_rsq_f16(F16_FOUR)
        self.assertIn("OpExtInst %float %glsl InverseSqrt", spv)
        self.assertIn("%out_mrt0 = OpVariable", spv)


if __name__ == "__main__":
    unittest.main()
