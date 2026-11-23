"""buffer_load_format_* through the V# FORMAT stays small and keeps its values (SHADER-774 compile time).

A typed load whose format is only known at run time used to decode the element in every one of the 41 formats in
Gen.FORMATS and pick one, by a binary search of branches (and, before that, a select chain): about 900 SPIR-V
instructions behind each load, plus a d16 tail of about 500 more inline. The NVIDIA driver inlines every function and
took about 0.8 s to compile each of them, so the compute modules with 33-45 of them were the slowest to compile. The
module is now one function per (components, d16) that reads each channel it writes from a per-format table with
straight-line arithmetic and no branch.

The shape tests (counts) fail on the translation that came before. The value test is the proof it computes the same
values: a digest over every format code x width x d16 x DST_SEL x alignment x bounds case on synthetic V#s and
memory (no game data), recorded from the translation before this change, which this one must reproduce.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402
from spirv_mini_interp import run_fragment  # noqa: E402

BASE = 0x1000
XYZW = 4 | 5 << 3 | 6 << 6 | 7 << 9


def _program(n, d16, count=1):
    nregs = (n + 1) // 2 if d16 else n
    vdata = "v2" if nregs == 1 else f"v[2:{1 + nregs}]"
    regs = [f"v{2 + min(i, nregs - 1)}" for i in range(4)]
    inst = f"buffer_load_format_{'d16_' if d16 else ''}{'xyzw'[:n]}"
    # the V# and the offset come from user SGPRs, so one translation serves every format and alignment
    return ["v_mov_b32 v1, s4", "v_mov_b32 v2, s5", "v_mov_b32 v3, s6"] + \
        [f"{inst} {vdata}, v1, s[0:3], 0 offen"] * count + \
        ["s_waitcnt vmcnt(0)", f"exp mrt0 {', '.join(regs)} done vm", "s_endpgm"]


def _translate(n, d16, count=1):
    return support.translate_lines(f"ps_load_flat_{n}{'h' if d16 else 'w'}{count}", _program(n, d16, count))


def _function(spv, marker):
    """The lines of the function whose name contains `marker`."""
    lines, out, inside = spv.splitlines(), [], False
    for ln in lines:
        if "= OpFunction " in ln and marker in ln:
            inside = True
        if inside:
            out.append(ln)
            if ln.strip() == "OpFunctionEnd":
                break
    return out


class Shape(unittest.TestCase):
    def test_one_branch_free_function_per_form(self):
        lines = _program(4, False, 6)[:-3] + _program(4, True, 3)[3:-3] + _program(2, False, 2)[3:]
        spv = support.translate_lines("ps_load_flat_forms", lines)
        self.assertEqual(len(re.findall(r"= OpFunction %v4uint_fmt None %fn_fmt_load", spv)), 3,
                         "one function per (components, d16) form")
        self.assertEqual(len(re.findall(r"= OpFunctionCall %v4uint_fmt %fmt_load_fn_", spv)), 11)
        self.assertEqual(spv.count("DontFlatten"), 0, "no branch per format")
        for marker in ("%fmt_load_fn_4w", "%fmt_load_fn_4h", "%fmt_load_fn_2w"):
            body = "\n".join(_function(spv, marker))
            self.assertTrue(body, marker)
            self.assertNotIn("OpBranchConditional", body, marker)
            self.assertNotIn("OpSwitch", body, marker)
            self.assertLess(body.count("\n"), 900, marker)

    def test_forms_share_one_function_type(self):
        lines = _program(3, False)[:-3] + _program(2, True)[3:]
        spv = support.translate_lines("ps_load_flat_types", lines)
        self.assertEqual(len(re.findall(r"= OpFunction %v4uint_fmt None %fn_fmt_load", spv)), 2)
        types = [ln.split(" = ", 1)[1] for ln in spv.splitlines() if " = OpTypeFunction " in ln]
        self.assertEqual(len(types), len(set(types)), "duplicate OpTypeFunction")

    def test_each_further_load_is_a_call(self):
        for n, d16 in ((1, False), (4, False), (4, True), (3, True)):
            with self.subTest(n=n, d16=d16):
                one, nine = _translate(n, d16, 1).count("\n"), _translate(n, d16, 9).count("\n")
                self.assertLess(nine - one, 8 * 150)       # was 185 to 640 lines per load


class Values(unittest.TestCase):
    """What a load returns, for every format code: the digest of every result (see the module docstring)."""

    WORDSETS = ((0x8123C4F7, 0x3E9A7F01, 0xC0490FDB, 0x5AA5FF80, 0x13579BDF, 0x2468ACE0),
                (0xFFFFFFFF, 0x7F800001, 0x80000000, 0x00000000, 0x7FC00000, 0x0000FFFF),
                (0x7F7F8080, 0x0001FFFF, 0x3C00BC00, 0x7BFF0001, 0xFF80007F, 0x01020304))
    # (word set, V# DST_SEL, byte offset into the 0x100-byte buffer)
    CASES = ((0, XYZW, 0), (1, 4 | 0 << 3 | 1 << 6 | 1 << 9, 3), (2, 7 | 6 << 3 | 5 << 6 | 4 << 9, 1),
             (0, 1 | 1 << 3 | 0 << 6 | 4 << 9, 0x100 - 12), (1, 2 | 3 << 3 | 4 << 6 | 5 << 9, 0x100 - 4))
    DIGEST = "4a48be30854465792ba692e2733e920b8cca0ebcf92953acf18da7d89601be73"

    def digest(self):
        h = hashlib.sha256()
        for n in (1, 2, 3, 4):
            for d16 in (False, True):
                spv = _translate(n, d16)
                for ws, sel, off in self.CASES:
                    for code in range(128):
                        mem = {BASE + 4 * i: w for i, w in enumerate(self.WORDSETS[ws])}
                        pc = [BASE, 0, 0x100, code << 12 | sel | 0x30000000, off, 0x12345678, 0xA9BCDEF0]
                        out = run_fragment(spv, push_constants=pc, memory=mem)["%out_mrt0"]
                        h.update(b"".join(int(x).to_bytes(4, "little") for x in out))
        return h.hexdigest()

    def test_every_format_width_d16_selector_and_alignment(self):
        self.assertEqual(self.digest(), self.DIGEST)


if __name__ == "__main__":
    unittest.main()
