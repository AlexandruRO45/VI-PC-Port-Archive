"""Helpers for the binary64 semantic tests: run one VALU instruction through the translator and tests/spirv_interp.py.

spirv_mini_interp.py is 32-bit float only; tests/spirv_interp.py also evaluates OpTypeFloat 64 (Python floats are
binary64). Standard library only.
"""
import math
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spirv_interp  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_INF, NEG_INF = 0x7FF0000000000000, 0xFFF0000000000000
POS_ZERO, NEG_ZERO = 0, 0x8000000000000000
QNAN = 0x7FF8000000000000
DBL_MAX = 0x7FEFFFFFFFFFFFFF
MIN_SUBNORMAL = 1
MIN_NORMAL = 0x0010000000000000


def bits(x):
    """The binary64 word of a Python float."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def value(b):
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFFFFFFFFFFFFFF))[0]


def is_nan(b):
    return (b & 0x7FF0000000000000) == 0x7FF0000000000000 and (b & 0x000FFFFFFFFFFFFF) != 0


def same(a, b):
    """Equal words, or both NaN (the payload of a NaN result is not pinned)."""
    return a == b or (is_nan(a) and is_nan(b))


def ulp_distance(a, b):
    """Distance in representable doubles between two finite words of the same sign (a NaN or a sign flip: huge)."""
    if is_nan(a) or is_nan(b) or (a ^ b) >> 63:
        return 1 << 64
    return abs((a & 0x7FFFFFFFFFFFFFFF) - (b & 0x7FFFFFFFFFFFFFFF))


def run(instruction, x=None, setup=(), dst="v[4:5]", name="ps_f64"):
    """Translate and run `instruction` on a one-lane pixel shader; return the destination (a 64-bit pair, or the
    32-bit v4 when dst is 'v4') as an int.

    `x`, a binary64 word, is loaded into v[0:1] first; `setup` adds more instructions (SGPR pairs and so on).
    """
    lines = []
    if x is not None:
        lines += [f"v_mov_b32 v0, {x & MASK32:#x}", f"v_mov_b32 v1, {(x >> 32) & MASK32:#x}"]
    lines += list(setup) + [instruction, "exp mrt0 v4, v5, v4, v5 done vm", "s_endpgm"]
    res = spirv_interp.run(support.translate_lines(name, lines))
    lo, hi = res.vgpr[4] & MASK32, res.vgpr[5] & MASK32
    return lo if dst == "v4" else lo | (hi << 32)
