"""v_ffbl_b32 / v_ffbh_u32 (RDNA2 VOP3) through tools/rdna2spirv.py.

v_ffbl_b32 D.u32 = index of the lowest set bit of S0.u32, or 0xFFFFFFFF if S0 == 0.
v_ffbh_u32 D.u32 = count of leading zero bits from the MSB side of S0.u32
                   (31 - index of the highest set bit), or 0xFFFFFFFF if S0 == 0.

Every case is a synthetic one-lane pixel shader translated by the translator and
executed by tests/semantic/spirv_mini_interp.py, compared against a hand-computed
reference. Standard library only: no GPU, no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402

M32 = 0xFFFFFFFF


def ref_ffbl(src):
    """Reference for v_ffbl_b32: index of lowest set bit, 0xFFFFFFFF if zero."""
    src &= M32
    if src == 0:
        return M32
    return (src & -src).bit_length() - 1


def ref_ffbh(src):
    """Reference for v_ffbh_u32: 31 - index of highest set bit, 0xFFFFFFFF if zero."""
    src &= M32
    if src == 0:
        return M32
    return 31 - (src.bit_length() - 1)


def run(op, src, form="v", dst="v0"):
    """v_ffbl_b32 / v_ffbh_u32 on one lane. Source form: v (VGPR), s (SGPR), i (inline literal)."""
    setup, srcs = [], []
    if form == "v":
        setup.append(f"v_mov_b32 v1, {src & M32:#x}")
        srcs.append("v1")
    elif form == "s":
        setup.append(f"s_mov_b32 s1, {src & M32:#x}")
        srcs.append("s1")
    else:
        srcs.append(f"{src & M32:#x}")
    lines = setup + [
        f"{op} {dst}, {', '.join(srcs)}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    out = support.mrt0_words(support.translate_lines(f"ps_{op}", lines))
    return out[0] & M32


# (opcode, src, expected), written by hand
CASES = [
    ("v_ffbl_b32", 0x00000000, M32),      # zero -> all-ones
    ("v_ffbl_b32", 0x00000001, 0),        # bit 0 set
    ("v_ffbl_b32", 0x00000002, 1),        # bit 1 set
    ("v_ffbl_b32", 0x00000008, 3),        # bit 3 set
    ("v_ffbl_b32", 0x80000000, 31),       # bit 31 set
    ("v_ffbl_b32", 0xDEADBEEF, 0),        # bit 0 set (odd)
    ("v_ffbl_b32", 0xFFFFFFFF, 0),        # bit 0 set
    ("v_ffbh_u32", 0x00000000, M32),      # zero -> all-ones
    ("v_ffbh_u32", 0x00000001, 31),       # bit 0 is highest -> 31 leading zeros
    ("v_ffbh_u32", 0x00000002, 30),       # bit 1 -> 30
    ("v_ffbh_u32", 0x00000008, 28),       # bit 3 -> 28
    ("v_ffbh_u32", 0x80000000, 0),        # bit 31 -> 0 leading zeros
    ("v_ffbh_u32", 0xDEADBEEF, 0),        # highest bit 31 -> 0 leading zeros
    ("v_ffbh_u32", 0xFFFFFFFF, 0),        # bit 31 -> 0
]


class FfbhFfblU32(unittest.TestCase):
    def test_hand_written_cases_match_reference(self):
        for op, src, expected in CASES:
            self.assertEqual(ref_ffbl(src) if op == "v_ffbl_b32" else ref_ffbh(src),
                             expected, (op, hex(src)))

    def test_translated_vgpr_source(self):
        for op, src, expected in CASES:
            with self.subTest(op=op, src=hex(src)):
                ref = ref_ffbl(src) if op == "v_ffbl_b32" else ref_ffbh(src)
                self.assertEqual(run(op, src, "v"), ref)

    def test_sgpr_and_inline_literal_sources(self):
        for op, src, expected in CASES[:8]:
            for form in ("s", "i"):
                with self.subTest(op=op, src=hex(src), form=form):
                    ref = ref_ffbl(src) if op == "v_ffbl_b32" else ref_ffbh(src)
                    self.assertEqual(run(op, src, form), ref)

    def test_destination_may_alias_a_source(self):
        # v1 = v_ffbh_u32 v1, v1 reads the source before writing.
        lines = ["v_mov_b32 v1, 0x00000008",
                 "v_ffbh_u32 v1, v1",
                 "exp mrt0 v1, v1, v1, v1 done vm",
                 "s_endpgm"]
        out = support.mrt0_words(support.translate_lines("ps_ffbh_alias", lines))
        self.assertEqual(out[0] & M32, 28)

    def test_random_operands(self):
        rng = random.Random(0xBEEF)
        for _ in range(150):
            src = rng.getrandbits(32)
            op = rng.choice(("v_ffbl_b32", "v_ffbh_u32"))
            with self.subTest(op=op, src=hex(src)):
                ref = ref_ffbl(src) if op == "v_ffbl_b32" else ref_ffbh(src)
                self.assertEqual(run(op, src), ref)


if __name__ == "__main__":
    unittest.main()
