"""Semantic tests for 64-bit integer bitwise VALU instructions in rdna2spirv.

Covers v_and_b64, v_or_b64, and v_xor_b64 lowering:
  - Pure 32-bit dword operations: v[d:d+1] = src0 & / | / ^ src1
  - Operates on each 32-bit dword independently with OpBitwiseAnd, OpBitwiseOr, OpBitwiseXor
  - Eliminates dependency on OpTypeInt 64 / Int64 capability for bitwise ops
  - Supports VGPR-pair, SGPR-pair, and 64-bit sign-extended/literal immediate operands
  - Correct execution verified against bit-level Python reference in spirv_mini_interp

Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest

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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bitwise64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_bitwise64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None


def translate_lines(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_valu_bitwise_b64\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    return rdna2spirv.translate(_SRC_PATH)


def run_ps_64(lines: list[str]) -> tuple[int, int]:
    """Execute shader and return (out_lo, out_hi) written to mrt0 (v0, v1)."""
    spv = translate_lines(lines + ["exp mrt0 v0, v1, v0, v1 done vm"])
    out = list(si.run_fragment(spv)["%out_mrt0"])
    return out[0], out[1]


class TestValuBitwiseB64(unittest.TestCase):
    def test_v_and_b64_vgpr(self):
        lo, hi = run_ps_64([
            "v_mov_b32 v2, 0x12345678",
            "v_mov_b32 v3, 0xABCDEF01",
            "v_mov_b32 v4, 0x0000FFFF",
            "v_mov_b32 v5, 0x000000FF",
            "v_and_b64 v[0:1], v[2:3], v[4:5]",
        ])
        self.assertEqual(lo, 0x5678)
        self.assertEqual(hi, 0x01)

    def test_v_or_b64_vgpr(self):
        lo, hi = run_ps_64([
            "v_mov_b32 v2, 0x12340000",
            "v_mov_b32 v3, 0x0000EF01",
            "v_mov_b32 v4, 0x00005678",
            "v_mov_b32 v5, 0xABCD0000",
            "v_or_b64 v[0:1], v[2:3], v[4:5]",
        ])
        self.assertEqual(lo, 0x12345678)
        self.assertEqual(hi, 0xABCDEF01)

    def test_v_xor_b64_vgpr(self):
        lo, hi = run_ps_64([
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_mov_b32 v3, 0xAAAAAAAA",
            "v_mov_b32 v4, 0x12345678",
            "v_mov_b32 v5, 0x55555555",
            "v_xor_b64 v[0:1], v[2:3], v[4:5]",
        ])
        self.assertEqual(lo, 0xFFFFFFFF ^ 0x12345678)
        self.assertEqual(hi, 0xAAAAAAAA ^ 0x55555555)

    def test_v_xor_b64_self_zero(self):
        lo, hi = run_ps_64([
            "v_mov_b32 v2, 0xDEADBEEF",
            "v_mov_b32 v3, 0xCAFEBABE",
            "v_xor_b64 v[0:1], v[2:3], v[2:3]",
        ])
        self.assertEqual(lo, 0)
        self.assertEqual(hi, 0)

    def test_v_and_b64_sgpr_source(self):
        lo, hi = run_ps_64([
            "s_mov_b32 s4, 0x0000FFFF",
            "s_mov_b32 s5, 0xFF000000",
            "v_mov_b32 v2, 0x12345678",
            "v_mov_b32 v3, 0xABCDEF01",
            "v_and_b64 v[0:1], v[2:3], s[4:5]",
        ])
        self.assertEqual(lo, 0x5678)
        self.assertEqual(hi, 0xAB000000)

    def test_v_and_b64_sign_extended_immediate(self):
        # -1 in 64-bit VALU sign-extends to 0xFFFFFFFFFFFFFFFF
        lo, hi = run_ps_64([
            "v_mov_b32 v2, 0x12345678",
            "v_mov_b32 v3, 0xABCDEF01",
            "v_and_b64 v[0:1], -1, v[2:3]",
        ])
        self.assertEqual(lo, 0x12345678)
        self.assertEqual(hi, 0xABCDEF01)

    def test_spirv_structure_no_int64(self):
        spv = translate_lines([
            "v_and_b64 v[0:1], v[2:3], v[4:5]",
            "v_or_b64 v[6:7], v[2:3], v[4:5]",
            "v_xor_b64 v[8:9], v[2:3], v[4:5]",
        ])
        self.assertIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        self.assertIn("OpBitwiseXor %uint", spv)
        # Verify it operates on pairs of %uint without introducing OpTypeInt 64
        self.assertNotIn("OpTypeInt 64", spv)


if __name__ == "__main__":
    unittest.main()
