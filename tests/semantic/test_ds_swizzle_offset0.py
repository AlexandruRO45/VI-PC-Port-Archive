"""Tests for ds_swizzle_b32 with offset:0 / omitted offset (SHADER-716)."""

import os
import pathlib
import struct
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
import rdna2spirv  # noqa: E402

FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)
FSSM = bytes(FSSM)


def _translate_cs(body: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "fssm"))
        os.makedirs(os.path.join(tmp, "disasm"))
        with open(os.path.join(tmp, "fssm", "cs_swz.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_swz.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_swz\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


class DsSwizzleOffset0Tests(unittest.TestCase):
    def test_omitted_and_zero_offset_match_quad_perm_0(self):
        spv_ref = _translate_cs(["ds_swizzle_b32 v1, v0 offset:swizzle(QUAD_PERM,0,0,0,0)"])
        spv_none = _translate_cs(["ds_swizzle_b32 v1, v0"])
        spv_zero = _translate_cs(["ds_swizzle_b32 v1, v0 offset:0"])
        self.assertEqual(spv_none, spv_ref)
        self.assertEqual(spv_zero, spv_ref)

    def test_nonzero_raw_offset_still_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            _translate_cs(["ds_swizzle_b32 v1, v0 offset:0x1234"])


if __name__ == "__main__":
    unittest.main()
