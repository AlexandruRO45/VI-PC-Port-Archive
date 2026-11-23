"""Regression tests for 32-bit buffer_atomic_* and buffer_atomic_cmpswap vdata width validation."""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(line):
    return support.translate_lines(
        "cs_buf_atomic_vdata",
        [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 1",
            "v_mov_b32 v3, 2",
            line,
            "s_endpgm",
        ],
        fssm=CS_FSSM,
    )


class BufferAtomicOperandTests(unittest.TestCase):
    def test_valid_vdata_widths_translate(self):
        spv_add = translate("buffer_atomic_add v2, v1, s[8:11], 0 offen glc")
        self.assertIn("OpAtomicIAdd %uint", spv_add)
        spv_cmp = translate("buffer_atomic_cmpswap v[2:3], v1, s[8:11], 0 offen glc")
        self.assertIn("OpAtomicCompareExchange %uint", spv_cmp)

    def test_wrong_vdata_width_is_rejected(self):
        for bad in (
            "buffer_atomic_add v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_sub v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_swap v[2:3], v1, s[8:11], 0 offen glc",
            "buffer_atomic_cmpswap v2, v1, s[8:11], 0 offen glc",
            "buffer_atomic_add off, v1, s[8:11], 0 offen",
        ):
            with self.subTest(line=bad):
                with self.assertRaises(support.rdna2spirv.Unsupported):
                    translate(bad)


if __name__ == "__main__":
    unittest.main()
