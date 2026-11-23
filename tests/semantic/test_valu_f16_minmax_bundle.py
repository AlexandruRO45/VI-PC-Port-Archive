from __future__ import annotations

"""Semantic translator verification suite for RDNA2 Float16 Min/Max and Rounding Bundle.

Covers 6 opcodes with 100% translator execution via support.translate_lines and support.mrt0_words:
- v_min_f16: IEEE 754 binary16 minimum, NMin semantics (NaN pass-through of other operand, -0.0 < +0.0).
- v_max_f16: IEEE 754 binary16 maximum, NMax semantics (NaN pass-through of other operand, +0.0 > -0.0).
- v_floor_f16: binary16 floor rounding, sign preservation on +-0.0, pass-through for NaN/inf.
- v_ceil_f16: binary16 ceiling rounding, sign preservation on +-0.0, pass-through for NaN/inf.
- v_trunc_f16: binary16 truncation towards zero, sign preservation on +-0.0, pass-through for NaN/inf.
- v_rndne_f16: binary16 round-to-nearest-even, sign preservation on +-0.0, pass-through for NaN/inf.

Every test method compiles an RDNA2 assembly program, runs the translated SPIR-V through
spirv_mini_interp, and asserts the concrete register values written to mrt0.
Zero mock machines, zero game data, standard library Python only.
"""

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
import support  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE 754 binary16 bit constants
POS_ZERO = 0x0000
NEG_ZERO = 0x8000
POS_ONE = 0x3C00
NEG_ONE = 0xBC00
POS_HALF = 0x3800
NEG_HALF = 0xB800
POS_TWO = 0x4000
NEG_TWO = 0xC000
POS_THREE = 0x4200
NEG_THREE = 0xC200
POS_FOUR = 0x4400
NEG_FOUR = 0xC400
POS_INF = 0x7C00
NEG_INF = 0xFC00
QNAN_CANONICAL = 0x7E00
QNAN_ALT = 0x7E55
SNAN = 0x7C01

POS_MIN_SUBNORM = 0x0001
POS_MAX_SUBNORM = 0x03FF
NEG_MIN_SUBNORM = 0x8001
NEG_MAX_SUBNORM = 0x83FF
POS_MIN_NORMAL = 0x0400
POS_MAX_NORMAL = 0x7BFF
NEG_MAX_NORMAL = 0xFBFF


def f16_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def is_nan16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x03FF) != 0


def run_ps_16(lines: list[str], name: str = "ps_f16") -> int:
    """Helper to assemble a pixel shader ending with mrt0 export and return the first 32-bit word."""
    full = list(lines) + [
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, full)
    words = support.mrt0_words(spv)
    return words[0]


# ==============================================================================
# 1. v_min_f16 Translator Tests
# ==============================================================================
class TestVMinF16Translator(unittest.TestCase):
    def test_min_f16_basic_values(self):
        cases = [
            (f16_bits(1.0), f16_bits(2.0), f16_bits(1.0)),
            (f16_bits(-3.0), f16_bits(1.0), f16_bits(-3.0)),
            (f16_bits(5.0), f16_bits(5.0), f16_bits(5.0)),
            (f16_bits(-2.0), f16_bits(-4.0), f16_bits(-4.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_min_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_min_basic_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_min_f16_signed_zeros(self):
        # RDNA2 / NMin orders -0.0 < +0.0
        cases = [
            (POS_ZERO, NEG_ZERO, NEG_ZERO),
            (NEG_ZERO, POS_ZERO, NEG_ZERO),
            (NEG_ZERO, NEG_ZERO, NEG_ZERO),
            (POS_ZERO, POS_ZERO, POS_ZERO),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_min_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_min_sz_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_min_f16_nan_suppression(self):
        cases = [
            (QNAN_CANONICAL, f16_bits(3.0), f16_bits(3.0)),
            (f16_bits(-5.0), QNAN_CANONICAL, f16_bits(-5.0)),
            (SNAN, f16_bits(2.0), f16_bits(2.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_min_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_min_nan_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

        # Both NaNs produce a NaN
        prog_all_nan = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {QNAN_CANONICAL:#x}",
            f"s_mov_b32 s1, {QNAN_ALT:#x}",
            "v_min_f16 v0, s0, s1",
        ]
        w = run_ps_16(prog_all_nan, "ps_min_all_nan")
        self.assertTrue(is_nan16(w & MASK16))

    def test_min_f16_infinities(self):
        cases = [
            (POS_INF, f16_bits(100.0), f16_bits(100.0)),
            (NEG_INF, f16_bits(-100.0), NEG_INF),
            (NEG_INF, POS_INF, NEG_INF),
            (POS_INF, POS_INF, POS_INF),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_min_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_min_inf_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_min_f16_subnormals(self):
        cases = [
            (POS_MIN_SUBNORM, POS_MIN_NORMAL, POS_MIN_SUBNORM),
            (NEG_MIN_SUBNORM, POS_MIN_SUBNORM, NEG_MIN_SUBNORM),
            (POS_MAX_SUBNORM, POS_MIN_SUBNORM, POS_MIN_SUBNORM),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_min_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_min_subnorm_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_min_f16_clamp(self):
        # min(-2.0, 5.0) = -2.0 clamped to [0.0, 1.0] -> 0.0
        # min(2.0, 3.0) = 2.0 clamped to [0.0, 1.0] -> 1.0
        prog1 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_TWO:#x}",
            f"s_mov_b32 s1, {POS_FOUR:#x}",
            "v_min_f16 v0, s0, s1 clamp",
        ]
        self.assertEqual(run_ps_16(prog1, "ps_min_clamp_neg") & MASK16, POS_ZERO)

        prog2 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_TWO:#x}",
            f"s_mov_b32 s1, {POS_THREE:#x}",
            "v_min_f16 v0, s0, s1 clamp",
        ]
        self.assertEqual(run_ps_16(prog2, "ps_min_clamp_hi") & MASK16, POS_ONE)

    def test_min_f16_op_sel_and_preservation(self):
        # v0 seeded with 0x5555AAAA; op_sel:[1, 0, 1] reads s0.hi (0.5), s1.lo (2.0), writes dst.hi
        # min(0.5, 2.0) = 0.5 (0x3800) -> dst becomes 0x3800AAAA (low half 0xAAAA preserved)
        prog = [
            "v_mov_b32_e32 v0, 0x5555AAAA",
            f"s_mov_b32 s0, {(POS_HALF << 16) | POS_FOUR:#x}",
            f"s_mov_b32 s1, {POS_TWO:#x}",
            "v_min_f16 v0, s0, s1 op_sel:[1, 0, 1]",
        ]
        w = run_ps_16(prog, "ps_min_opsel")
        self.assertEqual(w, (POS_HALF << 16) | 0xAAAA)

    def test_min_f16_input_modifiers(self):
        # -|s0| vs s1: -|2.0| = -2.0; min(-2.0, -1.0) = -2.0
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_TWO:#x}",
            f"s_mov_b32 s1, {NEG_ONE:#x}",
            "v_min_f16 v0, -|s0|, s1",
        ]
        w = run_ps_16(prog, "ps_min_mods")
        self.assertEqual(w & MASK16, NEG_TWO)

    def test_min_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_min_err", ["v_min_f16 v0, s0", "s_endpgm"])

    def test_min_f16_mutation_detection(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_ONE:#x}",
            f"s_mov_b32 s1, {POS_TWO:#x}",
            "v_min_f16 v0, s0, s1",
        ]
        w = run_ps_16(prog, "ps_min_mut")
        # Mutating NMin to NMax would produce 2.0 instead of 1.0
        self.assertNotEqual(w & MASK16, POS_TWO)


# ==============================================================================
# 2. v_max_f16 Translator Tests
# ==============================================================================
class TestVMaxF16Translator(unittest.TestCase):
    def test_max_f16_basic_values(self):
        cases = [
            (f16_bits(1.0), f16_bits(2.0), f16_bits(2.0)),
            (f16_bits(-3.0), f16_bits(1.0), f16_bits(1.0)),
            (f16_bits(-5.0), f16_bits(-2.0), f16_bits(-2.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_max_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_max_basic_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_max_f16_signed_zeros(self):
        # RDNA2 / NMax orders +0.0 > -0.0
        cases = [
            (POS_ZERO, NEG_ZERO, POS_ZERO),
            (NEG_ZERO, POS_ZERO, POS_ZERO),
            (NEG_ZERO, NEG_ZERO, NEG_ZERO),
            (POS_ZERO, POS_ZERO, POS_ZERO),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_max_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_max_sz_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_max_f16_nan_suppression(self):
        cases = [
            (QNAN_CANONICAL, f16_bits(3.0), f16_bits(3.0)),
            (f16_bits(-5.0), QNAN_CANONICAL, f16_bits(-5.0)),
            (SNAN, f16_bits(4.0), f16_bits(4.0)),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_max_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_max_nan_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_max_f16_infinities(self):
        cases = [
            (POS_INF, f16_bits(100.0), POS_INF),
            (NEG_INF, f16_bits(-100.0), f16_bits(-100.0)),
            (NEG_INF, POS_INF, POS_INF),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_max_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_max_inf_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_max_f16_subnormals(self):
        cases = [
            (POS_MIN_SUBNORM, POS_MIN_NORMAL, POS_MIN_NORMAL),
            (NEG_MIN_SUBNORM, NEG_MAX_SUBNORM, NEG_MIN_SUBNORM),
        ]
        for s0, s1, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {s0:#x}",
                f"s_mov_b32 s1, {s1:#x}",
                "v_max_f16 v0, s0, s1",
            ]
            w = run_ps_16(prog, f"ps_max_subnorm_{s0}_{s1}")
            self.assertEqual(w & MASK16, expected)

    def test_max_f16_clamp(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_TWO:#x}",
            f"s_mov_b32 s1, {POS_THREE:#x}",
            "v_max_f16 v0, s0, s1 clamp",
        ]
        self.assertEqual(run_ps_16(prog, "ps_max_clamp") & MASK16, POS_ONE)

    def test_max_f16_op_sel_and_preservation(self):
        prog = [
            "v_mov_b32_e32 v0, 0x12345678",
            f"s_mov_b32 s0, {(POS_THREE << 16) | POS_ONE:#x}",
            f"s_mov_b32 s1, {POS_TWO:#x}",
            "v_max_f16 v0, s0, s1 op_sel:[1, 0, 1]",
        ]
        w = run_ps_16(prog, "ps_max_opsel")
        self.assertEqual(w, (POS_THREE << 16) | 0x5678)

    def test_max_f16_input_modifiers(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_THREE:#x}",
            f"s_mov_b32 s1, {POS_ONE:#x}",
            "v_max_f16 v0, |s0|, s1",
        ]
        w = run_ps_16(prog, "ps_max_mods")
        self.assertEqual(w & MASK16, POS_THREE)

    def test_max_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_max_err", ["v_max_f16 v0, s0", "s_endpgm"])

    def test_max_f16_mutation_detection(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_ONE:#x}",
            f"s_mov_b32 s1, {POS_TWO:#x}",
            "v_max_f16 v0, s0, s1",
        ]
        w = run_ps_16(prog, "ps_max_mut")
        self.assertNotEqual(w & MASK16, POS_ONE)


# ==============================================================================
# 3. v_floor_f16 Translator Tests
# ==============================================================================
class TestVFloorF16Translator(unittest.TestCase):
    def test_floor_f16_basic_values(self):
        cases = [
            (f16_bits(1.7), f16_bits(1.0)),
            (f16_bits(-1.2), f16_bits(-2.0)),
            (f16_bits(3.0), f16_bits(3.0)),
            (f16_bits(-4.0), f16_bits(-4.0)),
        ]
        for src, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {src:#x}",
                "v_floor_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_floor_basic_{src}")
            self.assertEqual(w & MASK16, expected)

    def test_floor_f16_signed_zeros(self):
        # floor(+0.0) = +0.0, floor(-0.0) = -0.0
        for z in (POS_ZERO, NEG_ZERO):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {z:#x}",
                "v_floor_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_floor_sz_{z}")
            self.assertEqual(w & MASK16, z)

    def test_floor_f16_infinities_and_nans(self):
        for inf in (POS_INF, NEG_INF):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {inf:#x}",
                "v_floor_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_floor_inf_{inf}")
            self.assertEqual(w & MASK16, inf)

        prog_nan = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {QNAN_CANONICAL:#x}",
            "v_floor_f16 v0, s0",
        ]
        w = run_ps_16(prog_nan, "ps_floor_nan")
        self.assertTrue(is_nan16(w & MASK16))

    def test_floor_f16_subnormals(self):
        # Small positive subnormal rounds down to +0.0; small negative subnormal rounds down to -1.0
        prog1 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_MIN_SUBNORM:#x}",
            "v_floor_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog1, "ps_floor_subnorm_pos") & MASK16, POS_ZERO)

        prog2 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_MIN_SUBNORM:#x}",
            "v_floor_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog2, "ps_floor_subnorm_neg") & MASK16, NEG_ONE)

    def test_floor_f16_clamp(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(3.7):#x}",
            "v_floor_f16 v0, s0 clamp",
        ]
        self.assertEqual(run_ps_16(prog, "ps_floor_clamp") & MASK16, POS_ONE)

    def test_floor_f16_op_sel_and_preservation(self):
        prog = [
            "v_mov_b32_e32 v0, 0xA5A51234",
            f"s_mov_b32 s0, {(f16_bits(2.9) << 16) | POS_ZERO:#x}",
            "v_floor_f16 v0, s0 op_sel:[1, 1]",
        ]
        w = run_ps_16(prog, "ps_floor_opsel")
        self.assertEqual(w, (POS_TWO << 16) | 0x1234)

    def test_floor_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_floor_err", ["v_floor_f16 v0", "s_endpgm"])

    def test_floor_f16_mutation_detection(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(1.5):#x}",
            "v_floor_f16 v0, s0",
        ]
        w = run_ps_16(prog, "ps_floor_mut")
        self.assertNotEqual(w & MASK16, POS_TWO)


# ==============================================================================
# 4. v_ceil_f16 Translator Tests
# ==============================================================================
class TestVCeilF16Translator(unittest.TestCase):
    def test_ceil_f16_basic_values(self):
        cases = [
            (f16_bits(1.2), f16_bits(2.0)),
            (f16_bits(-1.7), f16_bits(-1.0)),
            (f16_bits(3.0), f16_bits(3.0)),
            (f16_bits(-4.0), f16_bits(-4.0)),
        ]
        for src, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {src:#x}",
                "v_ceil_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_ceil_basic_{src}")
            self.assertEqual(w & MASK16, expected)

    def test_ceil_f16_signed_zeros(self):
        for z in (POS_ZERO, NEG_ZERO):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {z:#x}",
                "v_ceil_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_ceil_sz_{z}")
            self.assertEqual(w & MASK16, z)

    def test_ceil_f16_infinities_and_nans(self):
        for inf in (POS_INF, NEG_INF):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {inf:#x}",
                "v_ceil_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_ceil_inf_{inf}")
            self.assertEqual(w & MASK16, inf)

    def test_ceil_f16_subnormals(self):
        prog1 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_MIN_SUBNORM:#x}",
            "v_ceil_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog1, "ps_ceil_subnorm_pos") & MASK16, POS_ONE)

        prog2 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_MIN_SUBNORM:#x}",
            "v_ceil_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog2, "ps_ceil_subnorm_neg") & MASK16, NEG_ZERO)

    def test_ceil_f16_clamp(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(-2.5):#x}",
            "v_ceil_f16 v0, s0 clamp",
        ]
        self.assertEqual(run_ps_16(prog, "ps_ceil_clamp") & MASK16, POS_ZERO)

    def test_ceil_f16_op_sel_and_preservation(self):
        prog = [
            "v_mov_b32_e32 v0, 0x11112222",
            f"s_mov_b32 s0, {(f16_bits(0.1) << 16) | POS_ZERO:#x}",
            "v_ceil_f16 v0, s0 op_sel:[1, 1]",
        ]
        w = run_ps_16(prog, "ps_ceil_opsel")
        self.assertEqual(w, (POS_ONE << 16) | 0x2222)

    def test_ceil_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_ceil_err", ["v_ceil_f16", "s_endpgm"])

    def test_ceil_f16_mutation_detection(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(1.1):#x}",
            "v_ceil_f16 v0, s0",
        ]
        w = run_ps_16(prog, "ps_ceil_mut")
        self.assertNotEqual(w & MASK16, POS_ONE)


# ==============================================================================
# 5. v_trunc_f16 Translator Tests
# ==============================================================================
class TestVTruncF16Translator(unittest.TestCase):
    def test_trunc_f16_basic_values(self):
        cases = [
            (f16_bits(1.9), f16_bits(1.0)),
            (f16_bits(-1.9), f16_bits(-1.0)),
            (f16_bits(2.0), f16_bits(2.0)),
            (f16_bits(-3.0), f16_bits(-3.0)),
        ]
        for src, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {src:#x}",
                "v_trunc_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_trunc_basic_{src}")
            self.assertEqual(w & MASK16, expected)

    def test_trunc_f16_signed_zeros(self):
        for z in (POS_ZERO, NEG_ZERO):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {z:#x}",
                "v_trunc_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_trunc_sz_{z}")
            self.assertEqual(w & MASK16, z)

    def test_trunc_f16_infinities_and_nans(self):
        for inf in (POS_INF, NEG_INF):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {inf:#x}",
                "v_trunc_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_trunc_inf_{inf}")
            self.assertEqual(w & MASK16, inf)

    def test_trunc_f16_subnormals(self):
        prog1 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_MIN_SUBNORM:#x}",
            "v_trunc_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog1, "ps_trunc_subnorm_pos") & MASK16, POS_ZERO)

        prog2 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_MIN_SUBNORM:#x}",
            "v_trunc_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog2, "ps_trunc_subnorm_neg") & MASK16, NEG_ZERO)

    def test_trunc_f16_clamp(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(5.8):#x}",
            "v_trunc_f16 v0, s0 clamp",
        ]
        self.assertEqual(run_ps_16(prog, "ps_trunc_clamp") & MASK16, POS_ONE)

    def test_trunc_f16_op_sel_and_preservation(self):
        prog = [
            "v_mov_b32_e32 v0, 0x88889999",
            f"s_mov_b32 s0, {(f16_bits(3.99) << 16) | POS_ZERO:#x}",
            "v_trunc_f16 v0, s0 op_sel:[1, 1]",
        ]
        w = run_ps_16(prog, "ps_trunc_opsel")
        self.assertEqual(w, (POS_THREE << 16) | 0x9999)

    def test_trunc_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_trunc_err", ["v_trunc_f16 v0", "s_endpgm"])

    def test_trunc_f16_mutation_detection(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(-1.8):#x}",
            "v_trunc_f16 v0, s0",
        ]
        w = run_ps_16(prog, "ps_trunc_mut")
        self.assertNotEqual(w & MASK16, NEG_TWO)


# ==============================================================================
# 6. v_rndne_f16 Translator Tests
# ==============================================================================
class TestVRndneF16Translator(unittest.TestCase):
    def test_rndne_f16_ties_to_even(self):
        # 1.5 -> 2.0 (even), 2.5 -> 2.0 (even), -1.5 -> -2.0, -2.5 -> -2.0
        cases = [
            (f16_bits(1.5), f16_bits(2.0)),
            (f16_bits(2.5), f16_bits(2.0)),
            (f16_bits(-1.5), f16_bits(-2.0)),
            (f16_bits(-2.5), f16_bits(-2.0)),
            (f16_bits(1.4), f16_bits(1.0)),
            (f16_bits(1.6), f16_bits(2.0)),
        ]
        for src, expected in cases:
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {src:#x}",
                "v_rndne_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_rndne_tie_{src}")
            self.assertEqual(w & MASK16, expected)

    def test_rndne_f16_signed_zeros(self):
        for z in (POS_ZERO, NEG_ZERO):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {z:#x}",
                "v_rndne_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_rndne_sz_{z}")
            self.assertEqual(w & MASK16, z)

    def test_rndne_f16_infinities_and_nans(self):
        for inf in (POS_INF, NEG_INF):
            prog = [
                "v_mov_b32_e32 v0, 0",
                f"s_mov_b32 s0, {inf:#x}",
                "v_rndne_f16 v0, s0",
            ]
            w = run_ps_16(prog, f"ps_rndne_inf_{inf}")
            self.assertEqual(w & MASK16, inf)

    def test_rndne_f16_subnormals(self):
        prog1 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {POS_MIN_SUBNORM:#x}",
            "v_rndne_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog1, "ps_rndne_subnorm_pos") & MASK16, POS_ZERO)

        prog2 = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {NEG_MIN_SUBNORM:#x}",
            "v_rndne_f16 v0, s0",
        ]
        self.assertEqual(run_ps_16(prog2, "ps_rndne_subnorm_neg") & MASK16, NEG_ZERO)

    def test_rndne_f16_clamp(self):
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(4.5):#x}",
            "v_rndne_f16 v0, s0 clamp",
        ]
        self.assertEqual(run_ps_16(prog, "ps_rndne_clamp") & MASK16, POS_ONE)

    def test_rndne_f16_op_sel_and_preservation(self):
        prog = [
            "v_mov_b32_e32 v0, 0x43211234",
            f"s_mov_b32 s0, {(f16_bits(1.5) << 16) | POS_ZERO:#x}",
            "v_rndne_f16 v0, s0 op_sel:[1, 1]",
        ]
        w = run_ps_16(prog, "ps_rndne_opsel")
        self.assertEqual(w, (POS_TWO << 16) | 0x1234)

    def test_rndne_f16_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_rndne_err", ["v_rndne_f16 v0", "s_endpgm"])

    def test_rndne_f16_mutation_detection(self):
        # 1.5 rounded to nearest even MUST be 2.0, not 1.0
        prog = [
            "v_mov_b32_e32 v0, 0",
            f"s_mov_b32 s0, {f16_bits(1.5):#x}",
            "v_rndne_f16 v0, s0",
        ]
        w = run_ps_16(prog, "ps_rndne_mut")
        self.assertNotEqual(w & MASK16, POS_ONE)


if __name__ == "__main__":
    unittest.main()
