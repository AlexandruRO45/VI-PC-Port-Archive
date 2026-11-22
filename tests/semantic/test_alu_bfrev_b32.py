"""Semantic translation tests for RDNA2 32-bit bit reverse v_bfrev_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and compares the result against
a plain-Python bit-reverse reference.

Covers:
  - Basic bit reverse (all 32 bits)
  - Zero, all-ones, single-bit, and boundary values
  - Negative control: result must not equal the input for non-palindromic bit patterns
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

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


def ref_bfrev32(x: int) -> int:
    """Plain-Python reference for v_bfrev_b32: reverse all 32 bits of x."""
    return int("{:032b}".format(x & M32)[::-1], 2)


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bfrev_b32.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bfrev_b32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_op(x: int) -> int:
    """Translate and run a shader that computes v_bfrev_b32 v1, v0 with v0=x."""
    asm = (
        "; test_bfrev_b32\n"
        f"\tv_mov_b32_e32 v0, {x & M32:#x}\n"
        "\tv_bfrev_b32 v1, v0\n"
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


class ALUVBitRevB32Tests(unittest.TestCase):
    def test_bfrev_b32_basic(self):
        # Reverse 0x80000000 -> 0x00000001 (only bit 31 set -> only bit 0 set)
        self.assertEqual(run_op(0x80000000), 0x00000001)
        # Reverse 0x00000001 -> 0x80000000 (only bit 0 set -> only bit 31 set)
        self.assertEqual(run_op(0x00000001), 0x80000000)

    def test_bfrev_b32_reference(self):
        cases = [0, 1, M32, 0x12345678, 0xDEADBEEF, 0xAAAAAAAA, 0x55555555,
                 0x0F0F0F0F, 0xF0F0F0F0, 0x00FF00FF, 0xFF00FF00]
        for x in cases:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(x), ref_bfrev32(x))

    def test_bfrev_b32_involution(self):
        # bfrev(bfrev(x)) == x
        for x in [0x12345678, 0xDEADBEEF, 0x00000001, 0xFFFFFFFF]:
            with self.subTest(x=hex(x)):
                self.assertEqual(run_op(run_op(x)), x)

    def test_bfrev_b32_negative_control(self):
        # For a non-palindromic pattern, result must differ from input.
        x = 0x12345678
        res = run_op(x)
        self.assertNotEqual(res, x)
        self.assertEqual(res, ref_bfrev32(x))


if __name__ == "__main__":
    unittest.main()
