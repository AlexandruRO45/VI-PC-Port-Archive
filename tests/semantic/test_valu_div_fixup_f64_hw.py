from __future__ import annotations

"""Semantic reference oracle and architectural model for v_div_fixup_f64.

RDNA2 / GFX10 VOP3 Double-Precision Floating-Point Division Fixup:
Performs the final IEEE-754 special-case resolution and sign determination for 64-bit float division.
Operands:
  S0 = quot (estimated quotient)
  S1 = denom (denominator)
  S2 = num (numerator)

Syntax:
    v_div_fixup_f64 v[dst:dst+1], v[src0:src0+1], v[src1:src1+1], v[src2:src2+1] [clamp]

Semantics:
    sign_out = sign(S1) ^ sign(S2)
    NaN(S2) -> quiet(S2)
    NaN(S1) -> quiet(S1)
    (0/0 or inf/inf) -> -qNaN (0xFFF80000_00000000)
    (x/inf or 0/y or underflow) -> +-0 (sign_out)
    (x/0 or inf/y or NaN(S0)) -> +-inf (sign_out | 0x7FF00000_00000000)
    otherwise -> copysign(S0, sign_out)
"""

import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si
import support

# IEEE 754 Binary64 Constants
F64_POS_ZERO: int = 0x00000000_00000000
F64_NEG_ZERO: int = 0x80000000_00000000
F64_ONE: int = 0x3FF00000_00000000
F64_TWO: int = 0x40000000_00000000
F64_FOUR: int = 0x40100000_00000000
F64_HALF: int = 0x3FE00000_00000000
F64_POS_INF: int = 0x7FF00000_00000000
F64_NEG_INF: int = 0xFFF00000_00000000
F64_QNAN: int = 0x7FF80000_00000000
F64_NEG_QNAN: int = 0xFFF80000_00000000

MASK_32: int = 0xFFFF_FFFF
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF


def u32(val: int) -> int:
    return val & MASK_32


def u64(val: int) -> int:
    return val & MASK_64


def oracle_div_fixup_f64(s0_u64: int, s1_u64: int, s2_u64: int, clamp: bool = False) -> int:
    """Bit-accurate IEEE 754 double-precision division fixup matching RDNA2 hardware semantics."""
    s0_u64, s1_u64, s2_u64 = u64(s0_u64), u64(s1_u64), u64(s2_u64)

    lo0, hi0 = s0_u64 & MASK_32, (s0_u64 >> 32) & MASK_32
    lo1, hi1 = s1_u64 & MASK_32, (s1_u64 >> 32) & MASK_32
    lo2, hi2 = s2_u64 & MASK_32, (s2_u64 >> 32) & MASK_32

    def cls64(lo: int, hi: int) -> Tuple[int, bool, bool, bool]:
        mag = hi & 0x7FFF_FFFF
        iz = ((mag | lo) == 0)
        exp_all = ((mag & 0x7FF0_0000) == 0x7FF0_0000)
        iinf = exp_all and (((mag & 0x000F_FFFF) | lo) == 0)
        inan = exp_all and (((mag & 0x000F_FFFF) | lo) != 0)
        return mag, iz, iinf, inan

    m0, z0, inf0, nan0 = cls64(lo0, hi0)
    _, z1, inf1, nan1 = cls64(lo1, hi1)
    _, z2, inf2, nan2 = cls64(lo2, hi2)

    e1 = (hi1 >> 20) & 0x7FF
    e2 = (hi2 >> 20) & 0x7FF
    ediff = (e2 - e1) & MASK_32
    ediff_i = ediff - 0x1_0000_0000 if (ediff & 0x8000_0000) else ediff
    uf = (ediff_i <= -1076)

    sgn = (hi1 ^ hi2) & 0x8000_0000
    inf_hi = sgn | 0x7FF0_0000
    norm_hi = sgn | m0

    b_zz = z1 and z2
    b_ii = inf1 and inf2
    b_ind = b_zz or b_ii
    b_zero = inf1 or z2 or uf
    b_inf = z1 or inf2 or nan0

    if nan2:
        r_hi, r_lo = hi2, lo2
    elif nan1:
        r_hi, r_lo = hi1, lo1
    elif b_ind:
        r_hi, r_lo = 0xFFF8_0000, 0
    elif b_zero:
        r_hi, r_lo = sgn, 0
    elif b_inf:
        r_hi, r_lo = inf_hi, 0
    else:
        r_hi, r_lo = norm_hi, lo0

    if clamp:
        val_f = struct.unpack("<d", struct.pack("<II", r_lo, r_hi))[0]
        if math.isnan(val_f):
            r_lo, r_hi = 0, 0
        else:
            val_f = max(0.0, min(1.0, val_f))
            r_lo, r_hi = struct.unpack("<II", struct.pack("<d", val_f))

    return u64((r_hi << 32) | r_lo)


def run_translator_div_fixup(
    s0: int,
    s1: int,
    s2: int,
    mods: str = "",
    name: str = "ps_div_fixup",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result 64-bit integer."""
    lo0, hi0 = s0 & MASK_32, (s0 >> 32) & MASK_32
    lo1, hi1 = s1 & MASK_32, (s1 >> 32) & MASK_32
    lo2, hi2 = s2 & MASK_32, (s2 >> 32) & MASK_32
    lines = [
        f"v_mov_b32 v2, {lo0:#x}",
        f"v_mov_b32 v3, {hi0:#x}",
        f"v_mov_b32 v4, {lo1:#x}",
        f"v_mov_b32 v5, {hi1:#x}",
        f"v_mov_b32 v6, {lo2:#x}",
        f"v_mov_b32 v7, {hi2:#x}",
        f"v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] {mods}".strip(),
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return u64((words[1] << 32) | words[0])


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_normal_quotient(self) -> None:
        """Verify normal division fixup passes quotient with correct sign."""
        got = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_TWO)
        self.assertEqual(got, F64_ONE)

    def test_division_by_zero_produces_infinity(self) -> None:
        """Verify x / 0 produces signed infinity."""
        got = oracle_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_TWO)
        self.assertEqual(got, F64_POS_INF)

    def test_zero_divided_by_y_produces_zero(self) -> None:
        """Verify 0 / y produces signed zero."""
        got = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_ZERO)
        self.assertEqual(got, F64_POS_ZERO)

    def test_indeterminate_0_div_0_produces_neg_qnan(self) -> None:
        """Verify 0 / 0 produces canonical negative quiet NaN."""
        got = oracle_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO)
        self.assertEqual(got, F64_NEG_QNAN)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        expected = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_TWO)
        translated = run_translator_div_fixup(F64_ONE, F64_TWO, F64_TWO, name="ps_div_oracle")
        self.assertEqual(translated, expected)
