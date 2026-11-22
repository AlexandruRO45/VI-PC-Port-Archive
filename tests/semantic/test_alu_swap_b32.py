"""Semantic translation tests for RDNA2 32-bit VGPR swap v_swap_b32.

Translates a synthetic pixel-shader listing through tools/rdna2spirv.translate()
-> SPIR-V -> tests/semantic/spirv_mini_interp.py and verifies that v_swap_b32
exchanges two VGPRs: after `v_swap_b32 v1, v2`, v1 holds v2's old value and
v2 holds v1's old value.

Covers:
  - Basic swap of two distinct values
  - Swap with identical values (no-op effectively)
  - Swap with zero
  - Swap with all-ones
  - Randomized swap vs Python reference
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


def run_swap(x: int, y: int):
    """Translate and run a shader that does v_swap_b32 v1, v2 with v1=x, v2=y.

    Returns (v1_after, v2_after) as 32-bit ints.
    """
    asm = (
        "; test_swap_b32\n"
        f"\tv_mov_b32_e32 v1, {x & M32:#x}\n"
        f"\tv_mov_b32_e32 v2, {y & M32:#x}\n"
        "\tv_swap_b32 v1, v2\n"
        "\ts_waitcnt vmcnt(0)\n"
        "\texp mrt0 v1, v2, v1, v2 done vm\n"
        "\ts_endpgm\n"
    )
    src_path = None
    tmp_dir = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(tmp_dir.name, "disasm")
    os.mkdir(disasm_dir)
    src_path = os.path.join(disasm_dir, "ps_swap_b32.s")
    with open(os.path.join(tmp_dir.name, "ps_swap_b32.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src_path)
    out = si.run_fragment(spv)["%out_mrt0"]
    tmp_dir.cleanup()
    return out[0] & M32, out[1] & M32


class ALUVSwapB32Tests(unittest.TestCase):
    def test_swap_b32_basic(self):
        v1, v2 = run_swap(0x12345678, 0x9ABCDEF0)
        self.assertEqual(v1, 0x9ABCDEF0)
        self.assertEqual(v2, 0x12345678)

    def test_swap_b32_identical(self):
        v1, v2 = run_swap(0xDEADBEEF, 0xDEADBEEF)
        self.assertEqual(v1, 0xDEADBEEF)
        self.assertEqual(v2, 0xDEADBEEF)

    def test_swap_b32_zero(self):
        v1, v2 = run_swap(0, 0xFFFFFFFF)
        self.assertEqual(v1, M32)
        self.assertEqual(v2, 0)

        v1, v2 = run_swap(M32, 0)
        self.assertEqual(v1, 0)
        self.assertEqual(v2, M32)

    def test_swap_b32_random(self):
        rng = random.Random(0xC0FFEE)
        for _ in range(50):
            x = rng.getrandbits(32)
            y = rng.getrandbits(32)
            with self.subTest(x=hex(x), y=hex(y)):
                v1, v2 = run_swap(x, y)
                self.assertEqual(v1, y)
                self.assertEqual(v2, x)


if __name__ == "__main__":
    unittest.main()
