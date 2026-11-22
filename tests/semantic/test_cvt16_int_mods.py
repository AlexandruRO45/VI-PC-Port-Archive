"""v_cvt_i16_f16 / v_cvt_u16_f16 with VOP3 clamp and omod against the words measured natively on RDNA2
(cs_cvt16_int_mods.asm): the modifiers leave the integer result unchanged."""
import math
import os
import struct
import sys
import unittest

import support

sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
from check import CVT16_INT_MODS_WORDS
from compute_fixture import compute_header

OUT_V = [0, 0, 256, 0x31016FAC]   # the output V# in s[0:3]: base 0, 256 bytes, FORMAT 32_FLOAT, DST_SEL xyzw
# (source half after neg/abs, signed) for the eight cases; clamp and omod play no part
CASES = [(0x3E00, True), (0xC100, True), (0x5640, False), (0x4100, True), (0x5640, False), (0xFC00, True),
         (0xBC00, False), (0x7BFF, False)]


def reference(half, signed):
    """Truncate toward zero and saturate to the 16-bit range (NaN gives 0); D's high half stays 0xdead."""
    x = struct.unpack("<e", struct.pack("<H", half))[0]
    lo, hi = (-32768, 32767) if signed else (0, 65535)
    v = 0 if x != x else (hi if x == math.inf else lo if x == -math.inf else max(lo, min(hi, int(x))))
    return 0xDEAD0000 | (v & 0xFFFF)


class Cvt16IntMods(unittest.TestCase):
    def test_reference_gives_the_words(self):
        self.assertEqual([f"{reference(*c):#010x}" for c in CASES], [f"{w:#010x}" for w in CVT16_INT_MODS_WORDS])

    def test_translated_fixture(self):
        asm = os.path.join(support.ROOT, "runtime", "test", "shaders", "cs_cvt16_int_mods.asm")
        spv = support.translate_fixture("cs_cvt16_int_mods", fssm=bytes(compute_header(asm)))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=OUT_V + [0] * 28, memory=memory)
        got = [memory[4 * i] for i in range(len(CVT16_INT_MODS_WORDS))]
        self.assertEqual([f"{w:#010x}" for w in got], [f"{w:#010x}" for w in CVT16_INT_MODS_WORDS])
        self.assertEqual({memory[a] for a in range(32, 0x100, 4)}, {0xEEEEEEEE})


if __name__ == "__main__":
    unittest.main()
