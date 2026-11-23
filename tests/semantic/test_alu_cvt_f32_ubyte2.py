"""v_cvt_f32_ubyte2 (RDNA2 VOP3) through tools/rdna2spirv.py.

v_cvt_f32_ubyte2 D.f32 = (float)((S0.u32 >> 16) & 0xFF): it extracts byte 2 (bits [16:23])
of the source register, zero-extends it to 32 bits, and converts unsigned int to float.
The plain (non-SDWA, non-_e64) form encodes the byte index in the opcode name (0..3).

Every case is a synthetic one-lane pixel shader translated by the translator and executed
by tests/semantic/spirv_mini_interp.py, compared against a hand-computed reference.
Standard library only: no GPU, no game data.
"""
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF


def ref_cvt_f32_ubyte2(src):
    """Reference for v_cvt_f32_ubyte2: float((src >> 16) & 0xFF)."""
    return float((src >> 16) & 0xFF)


def run_cvt_f32_ubyte2(src, form="v", dst="v0"):
    """v_cvt_f32_ubyte2 on one lane. Source form: v (VGPR), s (SGPR), i (inline literal)."""
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
        f"v_cvt_f32_ubyte2 {dst}, {', '.join(srcs)}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    out = support.mrt0_words(support.translate_lines("ps_cvt_f32_ubyte2", lines))
    return struct.unpack("<f", struct.pack("<I", out[0] & M32))[0]


# (src, expected byte-2 value as float), written by hand
CASES = [
    (0x00000000, 0.0),            # byte 2 is 0
    (0x000000FF, 0.0),            # byte 2 is 0 (byte 0 is 0xFF)
    (0x0000FF00, 0.0),            # byte 2 is 0 (byte 1 is 0xFF)
    (0x00FF0000, 255.0),          # byte 2 is 0xFF
    (0x00AABBCC, 170.0),          # byte 2 is 0xAA
    (0x00112233, 17.0),           # byte 2 is 0x11
    (0xFF000000, 0.0),            # byte 2 is 0 (byte 3 is 0xFF)
    (0xDEADBEEF, 173.0),          # byte 2 is 0xAD
    (0xFFFFFFFF, 255.0),          # byte 2 is 0xFF
    (0x7FFF7FFF, 255.0),          # byte 2 is 0xFF
    (0x80008000, 0.0),            # byte 2 is 0x00
]


class CvtF32Ubyte2(unittest.TestCase):
    def test_hand_written_cases_match_reference(self):
        for src, expected in CASES:
            self.assertEqual(ref_cvt_f32_ubyte2(src), expected, hex(src))

    def test_translated_vgpr_source(self):
        for src, expected in CASES:
            with self.subTest(src=hex(src)):
                self.assertEqual(run_cvt_f32_ubyte2(src, "v"), expected)

    def test_sgpr_and_inline_literal_sources(self):
        for src, expected in CASES[:6]:
            for form in ("s", "i"):
                with self.subTest(src=hex(src), form=form):
                    self.assertEqual(run_cvt_f32_ubyte2(src, form), expected)

    def test_destination_may_alias_a_source(self):
        # v1 = v_cvt_f32_ubyte2 v1, v1 reads the source before writing.
        lines = ["v_mov_b32 v1, 0x00AABBCC",
                 "v_cvt_f32_ubyte2 v1, v1",
                 "exp mrt0 v1, v1, v1, v1 done vm",
                 "s_endpgm"]
        out = support.mrt0_words(support.translate_lines("ps_cvt_f32_ubyte2_alias", lines))
        self.assertEqual(struct.unpack("<f", struct.pack("<I", out[0] & M32))[0], 170.0)

    def test_random_operands(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(120):
            src = rng.getrandbits(32)
            with self.subTest(src=hex(src)):
                self.assertEqual(run_cvt_f32_ubyte2(src), ref_cvt_f32_ubyte2(src))

    def test_all_byte_index_forms(self):
        # The plain v_cvt_f32_ubyte{N} forms for N = 0..3 all extract byte N.
        cases = [
            ("v_cvt_f32_ubyte0", 0),
            ("v_cvt_f32_ubyte1", 1),
            ("v_cvt_f32_ubyte2", 2),
            ("v_cvt_f32_ubyte3", 3),
        ]
        src = 0x00AABBCC            # bytes: CC(0) BB(1) AA(2) 00(3)
        expected = {0: 204.0, 1: 187.0, 2: 170.0, 3: 0.0}
        for op, n in cases:
            lines = [f"v_mov_b32 v1, {src:#x}",
                     f"{op} v2, v1",
                     "exp mrt0 v2, v2, v2, v2 done vm",
                     "s_endpgm"]
            out = support.mrt0_words(support.translate_lines("ps_cvt_f32_ubyte_n", lines))
            self.assertEqual(
                struct.unpack("<f", struct.pack("<I", out[0] & M32))[0],
                expected[n],
                op,
            )


if __name__ == "__main__":
    unittest.main()
