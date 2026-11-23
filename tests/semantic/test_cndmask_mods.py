"""Tests for v_cndmask_b32 with VOP3 neg/abs source modifiers (SHADER-738)."""

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
        with open(os.path.join(tmp, "fssm", "cs_cnd.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", "cs_cnd.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write("; cs_cnd\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


class CndmaskModsTests(unittest.TestCase):
    def test_vop3_abs_and_neg_modifiers_emitted(self):
        spv = _translate_cs([
            "v_mov_b32 v1, 0xbf800000",
            "v_mov_b32 v2, 0x3f800000",
            "v_cndmask_b32_e64 v0, |v1|, -v2, vcc_lo",
        ])
        self.assertIn("FAbs", spv)
        self.assertIn("OpFNegate %float", spv)
        self.assertIn("OpBitcast %uint", spv)

    def test_vop3_neg_abs_combined(self):
        spv = _translate_cs([
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 0x40000000",
            "v_cndmask_b32_e64 v0, -|v1|, v2, vcc_lo",
        ])
        self.assertIn("FAbs", spv)
        self.assertIn("OpFNegate %float", spv)


if __name__ == "__main__":
    unittest.main()
