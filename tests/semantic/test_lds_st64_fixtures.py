"""runtime/test/shaders/cs_w32_lds_st64_b64.asm and cs_w32_lds_st64_b32.asm: ds_write2st64_b64, ds_read2st64_b64,
ds_write2st64_b32 and ds_read2st64_b32 with distinct offsets (also offset0 > offset1).

The GPU words (check.py CS_DUMPS, run_gpu_tests.py) come from all 32 lanes of one wave32 group, which fill the whole
LDS with 0xEEEEEEEE and count the words that are not the fill. spirv_mini_interp keeps LDS per invocation and rejects
OpControlBarrier, so the CPU run is one invocation (lane 0) on the fixture without its two s_barrier lines. Lane 0 fills only the
words at byte 128 * k, so out[7] (the high dword of the last read, or the count when the count is off) differs from the
GPU's: the written words that lie on a multiple of 128 are the count lane 0 sees. The plain-Python reference models
the ISA pseudocode of the four ops and gives both word lists.
"""
import os
import re
import sys
import unittest

import spirv_mini_interp as si
import support

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")
FILL = 0xEEEEEEEE

GPU = {
    "cs_w32_lds_st64_b64": [0x11110001, 0x22220002, 0x33330003, 0x44440004,
                            0x77770007, 0x88880008, 0x55550005, 0x66660006],
    "cs_w32_lds_st64_b32": [0x99990009, 0xAAAA000A, 0xBBBB000B, 0xCCCC000C,
                            0xCCCC000C, 0xBBBB000B, 0xDDDD000D, 0x1234ABCD],
}
# Lane 0 alone: the count it sees is the number of written words at a multiple of 128 (b64: 0x480 and 0xA80, b32: none).
LANE0 = {"cs_w32_lds_st64_b64": GPU["cs_w32_lds_st64_b64"][:7] + [2],
         "cs_w32_lds_st64_b32": GPU["cs_w32_lds_st64_b32"][:7] + [0]}


class Lds:
    """The DS two-address forms from the RDNA2 ISA: element size 4 or 8 bytes, offsets in units of 64 elements."""

    def __init__(self):
        self.m = {}

    def word(self, a):
        return self.m.get(a, FILL)

    def write2st64(self, size, addr, offset0, offset1, d0, d1):
        for off, data in ((offset0, d0), (offset1, d1)):
            for i in range(size // 4):
                self.m[addr + off * size * 64 + 4 * i] = (data >> (32 * i)) & 0xFFFFFFFF

    def read2st64(self, size, addr, offset0, offset1):
        return [self.word(addr + off * size * 64 + 4 * i) for off in (offset0, offset1) for i in range(size // 4)]


def reference_b64():
    lds = Lds()
    for a, v in ((0x480, 0x55550005), (0x484, 0x66660006), (0xA80, 0x77770007), (0xA84, 0x88880008)):
        lds.m[a] = v
    lds.write2st64(8, 0x40, 1, 3, 0x22220002_11110001, 0x44440004_33330003)
    out = [lds.word(a) for a in (0x240, 0x244, 0x640, 0x644)] + lds.read2st64(8, 0x80, 5, 2)
    return out, lds


def reference_b32():
    lds = Lds()
    lds.m[0x4C0], lds.m[0x9C0] = 0xBBBB000B, 0xCCCC000C
    lds.write2st64(4, 0x20, 3, 7, 0x99990009, 0xAAAA000A)
    out = [lds.word(0x320), lds.word(0x720)] + lds.read2st64(4, 0xC0, 4, 9) + lds.read2st64(4, 0xC0, 9, 4)
    lds.write2st64(4, 0x30, 11, 2, 0xDDDD000D, 0x1234ABCD)
    return out + [lds.word(0xB30), lds.word(0x230)], lds


class LdsSt64(unittest.TestCase):
    REF = {"cs_w32_lds_st64_b64": reference_b64, "cs_w32_lds_st64_b32": reference_b32}
    STORED = {"cs_w32_lds_st64_b64": 8, "cs_w32_lds_st64_b32": 6}

    def test_reference(self):
        for name, ref in self.REF.items():
            with self.subTest(shader=name):
                out, lds = ref()
                self.assertEqual(out, GPU[name])
                self.assertEqual(len(lds.m), self.STORED[name])   # no word but the stored ones left the fill
                lane0 = sum(1 for a in lds.m if a % 128 == 0)
                self.assertEqual(LANE0[name][7], GPU[name][7] if lane0 == self.STORED[name] else lane0)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            for name, want in GPU.items():
                m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % name, text)
                with self.subTest(file="/".join(rel), shader=name):
                    self.assertIsNotNone(m)
                    self.assertEqual([int(v, 0) for v in m.group(1).split(",")], want)

    def test_translated_shader_lane0(self):
        sys.path.insert(0, SHADERS)
        from compute_fixture import compute_header
        for name, want in LANE0.items():
            header = compute_header(os.path.join(SHADERS, name + ".asm"))
            with open(os.path.join(support.ROOT, "tests", "semantic", "fixtures", "disasm", name + ".s"),
                      encoding="utf-8") as f:
                lines = [ln for ln in f.read().splitlines()[1:] if ln.strip() and ln.strip() != "s_barrier"]
            with self.subTest(shader=name):
                spv = support.translate_lines(name, lines, fssm=bytes(header))
                memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
                si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                                 push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
                self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])
                self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")


if __name__ == "__main__":
    unittest.main()
