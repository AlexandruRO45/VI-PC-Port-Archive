"""Semantic translation tests for RDNA2 32-bit bitwise NOT v_not_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python reference: (~x) & 0xFFFFFFFF.

Covers:
  - Basic NOT: 0 -> all-ones, all-ones -> 0
  - Reference comparison across varied bit patterns
  - Involution: NOT(NOT(x)) == x
  - Negative control: result differs from input for non-trivial cases
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_not32(x: int) -> int:
    """Plain-Python reference for v_not_b32: bitwise NOT masked to 32 bits."""
    return (~x) & M32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_not_b32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_not_b32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x: int) -> int:
    """Translate and run a shader that computes v_not_b32 v1, v0 with v0=x."""
    asm = (
        "; test_not_b32\n"
        f"\tv_mov_b32_e32 v0, {x & M32:#x}\n"
        "\tv_not_b32 v1, v0\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v1, v1, v1 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    w = si.run_fragment(spv)["%out_mrt0"]
    return w[0] & M32


class ALUVNotB32Tests(unittest.TestCase):
    def test_not_b32_basic(self):
        # ~0 = all-ones
        self.assertEqual(run_op(0), M32)
        # ~all-ones = 0
        self.assertEqual(run_op(M32), 0)
        # ~single bit
        self.assertEqual(run_op(0x80000000), 0x7FFFFFFF)
        self.assertEqual(run_op(0x00000001), 0xFFFFFFFE)

    def test_not_b32_reference(self):
        cases = [
            (0x00000000, M32),
            (M32, 0x00000000),
            (0x12345678, 0xEDCBA987),
            (0xDEADBEEF, 0x21524110),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0xFFFF0000, 0x0000FFFF),
        ]
        for x, expected in cases:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(x), expected)
                self.assertEqual(run_op(x), ref_not32(x))

    def test_not_b32_random_vs_reference(self):
        rng = random.Random(0xDEADBEEF)
        for _ in range(50):
            x = rng.getrandbits(32)
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(x), ref_not32(x))

    def test_not_b32_involution(self):
        # NOT(NOT(x)) == x
        for x in [0, M32, 0x12345678, 0xDEADBEEF, 0x80000000]:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(run_op(x)), x)

    def test_not_b32_negative_control(self):
        # Result differs from input for non-trivial values.
        x = 0x12345678
        res = run_op(x)
        expected = ref_not32(x)
        self.assertEqual(res, expected)
        self.assertNotEqual(res, x)


if __name__ == "__main__":
    unittest.main()
