"""cs_w32_s64_data: 64-bit scalar bit counters and v_mbcnt_hi in wave32 code, run here on the CPU too.

Standard library only; no GPU, no LLVM, no game files.

LLVM's wave32 output has s_bcnt1_i32_b64 and s_ff1_i32_b64 (a 64-bit popcount / ctz of a uniform value) and a lane-id
chain may end in v_mbcnt_hi_u32_b32. The translator treated them as wave64 markers: with a wave32 FSSM header it refused
the shader ("FSSM wave32 header contradicts wave64 code"). The header decides now; the markers that wave32 code cannot
contain (exec_hi, _saveexec_b64, _wrexec_b64, s_wqm_b64, s_bfm_b64, a _b64 op on exec or vcc) are still refused, and
without any header the markers still imply wave64.

runtime/test/shaders/cs_w32_s64_data.asm stores eight words (derivations in its header); this runs the fixture with a
wave32 FSSM header through spirv_mini_interp as one 32-lane group and checks them against hand constants and a
plain-Python model, and that check.py and scripts/run_gpu_tests.py expect the same words.
"""
import os
import re
import struct
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
M32 = 0xFFFFFFFF
VSHARP = [0, 0, 0x1000, 0x31016FAC]


def fssm(wave):
    """An FSSM header with the magic, the wave size flag (0xd4 bit 5: set = wave32) and a 32x1x1 compute header."""
    h = bytearray(0x148)
    h[0:4] = b"FSSM"
    struct.pack_into("<I", h, 0xD4, 0x20 if wave == 32 else 0)
    struct.pack_into("<4I", h, 0x138, 0x88, 32, 1, 1)
    return bytes(h)


def ff(x, bit):
    """Index of the lowest bit of x equal to `bit`, -1 (as 32 bits) if none."""
    for i in range(64):
        if (x >> i) & 1 == bit:
            return i
    return M32


X1, X2, X3 = 0x0000010080000000, (1 << 64) - 1, 0xFFFF0000FFFFFFFF
EXPECTED = [bin(X1).count("1"), ff(X1, 1), 64 - bin(X3).count("1"), ff(X3, 0), bin(X3).count("1"), ff(X2, 0), 5,
            bin(0x0F0F0F0F & ((1 << 20) - 1)).count("1")]
HAND = [2, 31, 16, 32, 48, 0xFFFFFFFF, 5, 12]


class S64Data(unittest.TestCase):
    def test_model_matches_hand_constants(self):
        self.assertEqual(EXPECTED, HAND)

    def test_translated_shader_with_wave32_header(self):
        spv = support.translate_fixture("cs_w32_s64_data", fssm=fssm(32))
        mem = si.run_workgroup(spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                                   "%sg_lane": i},
                               push_constants=VSHARP + [0] * 28, memory={})
        self.assertFalse(support.rdna2spirv.translate.last_gen.wave64)
        self.assertEqual(sorted(mem), [4 * k for k in range(8)])
        for k, want in enumerate(HAND):
            with self.subTest(word=k):
                self.assertEqual(f"{mem[4 * k]:#010x}", f"{want:#010x}")

    def test_wave32_header_still_refuses_what_wave32_cannot_contain(self):
        for line in ("s_bfm_b64 s[10:11], s20, s21", "s_wqm_b64 s[10:11], s[12:13]", "s_mov_b32 exec_hi, -1",
                     "s_and_saveexec_b64 s[10:11], vcc", "s_mov_b64 s[10:11], exec"):
            with self.subTest(op=line):
                with self.assertRaisesRegex(Exception, "contradicts wave64 code"):
                    support.translate_lines("cs_w32_bad", [line, "s_endpgm"], fssm=fssm(32))

    def test_wave64_header_takes_the_same_markers(self):
        for line in ("s_bcnt1_i32_b64 s20, s[10:11]", "s_ff1_i32_b64 s20, s[10:11]", "v_mbcnt_hi_u32_b32 v1, -1, v1"):
            with self.subTest(op=line):
                support.translate_lines("cs_w64_ok", [line, "s_endpgm"], fssm=fssm(64))
                self.assertTrue(support.rdna2spirv.translate.last_gen.wave64)

    def test_without_a_header_the_markers_still_mean_wave64(self):
        zero = bytearray(0x148)
        struct.pack_into("<4I", zero, 0x138, 0x88, 32, 1, 1)
        for line in ("s_bcnt1_i32_b64 s20, s[10:11]", "v_mbcnt_hi_u32_b32 v1, -1, v1"):
            with self.subTest(op=line):
                support.translate_lines("cs_nohdr", [line, "s_endpgm"], fssm=bytes(zero))
                self.assertTrue(support.rdna2spirv.translate.last_gen.wave64)
        support.translate_lines("cs_nohdr32", ["s_mov_b32 s20, 1", "s_endpgm"], fssm=bytes(zero))
        self.assertFalse(support.rdna2spirv.translate.last_gen.wave64)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_s64_data"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) & M32 for v in m.group(1).split(",")], HAND)


if __name__ == "__main__":
    unittest.main()
