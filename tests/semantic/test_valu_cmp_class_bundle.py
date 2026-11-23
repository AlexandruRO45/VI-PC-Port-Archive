from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 VALU IEEE 754 Class Testing Suite.

Covers:
- v_cmp_class_f16: Test class of 16-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmp_class_f32: Test class of 32-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmp_class_f64: Test class of 64-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmpx_class_f16: Test class of 16-bit float, narrowing EXEC active mask.
- v_cmpx_class_f32: Test class of 32-bit float, narrowing EXEC active mask.
- v_cmpx_class_f64: Test class of 64-bit float, narrowing EXEC active mask.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF
CLASS_ALL: int = 0x3FF

# 10 IEEE 754 Class Bits:
# 0: sNaN, 1: qNaN, 2: -inf, 3: -normal, 4: -denorm, 5: -0, 6: +0, 7: +denorm, 8: +normal, 9: +inf
CLASS_SNAN = 1 << 0
CLASS_QNAN = 1 << 1
CLASS_NEG_INF = 1 << 2
CLASS_NEG_NORMAL = 1 << 3
CLASS_NEG_DENORM = 1 << 4
CLASS_NEG_ZERO = 1 << 5
CLASS_POS_ZERO = 1 << 6
CLASS_POS_DENORM = 1 << 7
CLASS_POS_NORMAL = 1 << 8
CLASS_POS_INF = 1 << 9


def run_cmp_shader(ty: str, x: int, mask: int, is_cmpx: bool = False, test_id: str = "") -> Tuple[int, str]:
    if ty == "f64":
        x_lo = x & 0xFFFF_FFFF
        x_hi = (x >> 32) & 0xFFFF_FFFF
        if is_cmpx:
            lines = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v2, {hex(x_lo)}",
                f"v_mov_b32 v3, {hex(x_hi)}",
                f"v_cmpx_class_f64_e32 v[2:3], {hex(mask)}",
                "v_mov_b32 v0, 1",
                "s_mov_b32 exec_lo, -1",
            ]
        else:
            lines = [
                f"v_mov_b32 v2, {hex(x_lo)}",
                f"v_mov_b32 v3, {hex(x_hi)}",
                f"v_cmp_class_f64_e32 vcc_lo, v[2:3], {hex(mask)}",
                "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
            ]
    else:
        if is_cmpx:
            lines = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {hex(x)}",
                f"v_cmpx_class_{ty}_e32 v1, {hex(mask)}",
                "v_mov_b32 v0, 1",
                "s_mov_b32 exec_lo, -1",
            ]
        else:
            lines = [
                f"v_mov_b32 v1, {hex(x)}",
                f"v_cmp_class_{ty}_e32 vcc_lo, v1, {hex(mask)}",
                "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
            ]
    full = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(f"ps_cmp_{ty}_{is_cmpx}_{test_id}", full)
    words = support.mrt0_words(spv)
    return words[0], spv


# ------------------------------------------------------------------------------
# v_cmp_class_f16 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def classify_f16(u: int) -> int:
    u = u & 0xFFFF
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    man = u & 0x3FF
    if exp == 0:
        if man == 0:
            return 5 if sign else 6
        return 4 if sign else 7
    elif exp == 31:
        if man == 0:
            return 2 if sign else 9
        return 1 if (man & 0x200) else 0
    return 3 if sign else 8


def ref_cmp_class_f16(u: int, mask: int) -> bool:
    c = classify_f16(u)
    return bool(mask & (1 << c))


class TestCmpClassF16Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        samples = [
            (0x7C01, CLASS_SNAN),
            (0x7E00, CLASS_QNAN),
            (0xFC00, CLASS_NEG_INF),
            (0xBC00, CLASS_NEG_NORMAL),
            (0x8001, CLASS_NEG_DENORM),
            (0x8000, CLASS_NEG_ZERO),
            (0x0000, CLASS_POS_ZERO),
            (0x0001, CLASS_POS_DENORM),
            (0x3C00, CLASS_POS_NORMAL),
            (0x7C00, CLASS_POS_INF),
        ]
        for idx, (val, mask) in enumerate(samples):
            res_hw, _ = run_cmp_shader("f16", val, mask, is_cmpx=False, test_id=f"oracle_{idx}")
            self.assertEqual(res_hw, 1 if ref_cmp_class_f16(val, mask) else 0)

    def test_oracle_mask_combinations_hardware(self):
        # Full mask matches any float
        res_all, _ = run_cmp_shader("f16", 0x3C00, CLASS_ALL, is_cmpx=False, test_id="all_mask")
        self.assertEqual(res_all, 1)

        # Zero mask never matches
        res_zero, _ = run_cmp_shader("f16", 0x3C00, 0, is_cmpx=False, test_id="zero_mask")
        self.assertEqual(res_zero, 0)
