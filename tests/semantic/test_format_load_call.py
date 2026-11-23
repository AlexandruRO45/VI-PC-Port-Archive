"""A buffer_load_format whose format comes from the V# at run time is one call to %fmt_load_fn (SHADER-774).

Inlined, each such load was about 160 SPIR-V lines, and the largest compute module has 3,281 of them. The module
keeps one copy of the load in %fmt_load_fn, and each load is a call plus the extracts of the channels it writes. The
values are checked by test_buffer_formats_all and test_format_decode_branch, which go through the same function.
"""
import re
import unittest

import support
from spirv_mini_interp import run_fragment

BASE = 0x1000
WORDS = (0x3F800000, 0x40000000, 0x40400000, 0x40800000)


def program(loads):
    lines = [f"s_mov_b32 s0, {BASE:#x}", "s_mov_b32 s1, 0", "s_mov_b32 s2, 0x100",
             f"s_mov_b32 s3, {77 << 12 | 0x30000FAC:#x}", "v_mov_b32 v1, 0"]
    for k in range(loads):
        lines.append(f"buffer_load_format_xyzw v[{2 + 4 * k}:{5 + 4 * k}], v1, s[0:3], 0 offen")
    last = 2 + 4 * (loads - 1)
    lines += [f"exp mrt0 v{last}, v{last + 1}, v{last + 2}, v{last + 3} done vm", "s_endpgm"]
    return lines


class FormatLoadCallTests(unittest.TestCase):
    def test_each_load_is_one_call(self):
        one = support.translate_lines("ps_fmt_call1", program(1))
        three = support.translate_lines("ps_fmt_call3", program(3))
        self.assertEqual(len(re.findall(r"%fmt_load_fn = OpFunction ", three)), 1)
        self.assertEqual(len(re.findall(r"OpFunctionCall %\w+ %fmt_load_fn ", three)), 3)
        # Two more loads add a call and a few extracts and register writes each, not another copy of the load.
        growth = (len(three.splitlines()) - len(one.splitlines())) / 2
        self.assertLess(growth, 40, f"{growth} lines per extra load")

    def test_the_call_reads_the_element(self):
        # 32_32_32_32_FLOAT (77) with the identity DST_SEL: the four dwords as they are.
        spv = support.translate_lines("ps_fmt_call_value", program(1))
        out = run_fragment(spv, memory={BASE + 4 * i: w for i, w in enumerate(WORDS)})["%out_mrt0"]
        self.assertEqual(tuple(out), WORDS)


if __name__ == "__main__":
    unittest.main()
