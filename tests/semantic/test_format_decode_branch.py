"""The run-time FORMAT decode of buffer_load_format gives every format's values (a code outside FORMATS gives zeros).

Every code goes through the translator and the mini interpreter and is compared with a reference decoded here from
the raw bits with exact quotients. The shape of the translation (no branch per format) is in test_typed_load_flat.py.
"""
import os
import struct
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
from spirv_mini_interp import half_to_float, run_fragment

import rdna2spirv

BASE = 0x1000
WORDS = (0x8123C4F7, 0x3E9A7F01, 0xC0490FDB, 0x5AA5FF80)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def reference(code):
    if code not in rdna2spirv.Gen.FORMATS:
        return (0, 0, 0, 0)
    size, kind, widths = rdna2spirv.Gen.FORMATS[code]
    raw, bit, out = sum(w << (32 * i) for i, w in enumerate(WORDS)), 0, []
    for w in widths:
        v = raw >> bit & ((1 << w) - 1)
        bit += w
        signed = v - (1 << w) if v >> (w - 1) else v
        if kind in ("f32", "u32", "i32", "uint"):
            out.append(v)
        elif kind == "sint":
            out.append(signed & 0xFFFFFFFF)
        elif kind == "f16":
            # The interpreter's pinned f16 -> f32 rule (a NaN half widens to the canonical quiet NaN with its sign),
            # not struct's '<e', whose NaN payload depends on the Python version (CPython gh-130317).
            out.append(f32_bits(half_to_float(v)))
        elif kind == "unorm":
            out.append(f32_bits(float(Fraction(v, (1 << w) - 1))))
        elif kind == "snorm":
            out.append(f32_bits(float(max(Fraction(-1), Fraction(signed, (1 << (w - 1)) - 1)))))
        else:
            return None                 # a kind this reference doesn't model: covered by its own tests
    return tuple(out[c % len(out)] for c in range(4))


def load(code):
    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
             f"s_mov_b32 s3, {code << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0",
             "buffer_load_format_xyzw v[2:5], v1, s[0:3], 0 offen", "exp mrt0 v2, v3, v4, v5 done vm", "s_endpgm"]
    spv = support.translate_lines("ps_fmt_branch", lines)
    return tuple(run_fragment(spv, memory={BASE + 4 * i: w for i, w in enumerate(WORDS)})["%out_mrt0"])


class FormatDecodeBranchTests(unittest.TestCase):
    def test_every_format_and_the_codes_between(self):
        for code in range(max(rdna2spirv.Gen.FORMATS) + 2):
            want = reference(code)
            if want is not None:
                self.assertEqual(load(code), want, f"format {code}")


if __name__ == "__main__":
    unittest.main()
