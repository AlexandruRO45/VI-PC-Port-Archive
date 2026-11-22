"""runtime/test/shaders/cs_w32_ds_subword_write.asm, cs_w32_ds_subword_read.asm and cs_w32_ds_wide.asm: LDS sub-dword
writes and reads, ds_mskor_b32 / _rtn_, ds_add_rtn_u32 and the 96/128-bit accesses.

The GPU words (check.py CS_DUMPS, run_gpu_tests.py) come from one wave32 group. spirv_mini_interp keeps LDS per
invocation and rejects OpControlBarrier, so the CPU run is one invocation (lane 0) on the fixture without its s_barrier
lines. The two fixtures that run on lane 0 alone give the GPU words; cs_w32_ds_subword_write has 32 lanes share dwords on
the GPU, so for lane 0 alone the plain-Python model below (one lane, memory not written reads 0) gives the words.
"""
import os
import re
import sys
import unittest

import spirv_mini_interp as si
import support

SHADERS = os.path.join(support.ROOT, "runtime", "test", "shaders")

GPU = {
    "cs_w32_ds_subword_write": [0x44434241, 0x605F5E5D, 0xEEEEEEEE, 0xFF81FF80, 0xFF9FFF9E, 0xEEEE22EE, 0x556622EE, 0xEEEE9ABC],
    "cs_w32_ds_subword_read": [0xF2, 0xFFFFFFF2, 0x8081, 0xFFFF8081, 0xAAAA8001, 0x7FFEBBBB, 0x58C57884, 0xFFFFFFE0],
    "cs_w32_ds_wide": [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10, 0x11111111, 0x22222222, 0x33333333, 0x11110003],
}


class Lds:
    """Byte-addressed LDS; a byte nobody wrote reads 0."""

    def __init__(self):
        self.b = {}

    def write(self, addr, value, nbytes):
        for i in range(nbytes):
            self.b[addr + i] = (value >> (8 * i)) & 0xFF

    def read(self, addr, nbytes):
        return sum(self.b.get(addr + i, 0) << (8 * i) for i in range(nbytes))


def sext(v, bits):
    return v | (~((1 << bits) - 1) & 0xFFFFFFFF) if v >> (bits - 1) else v


def model_write(lanes):
    m = Lds()
    for lane in range(lanes):
        m.write(4 * lane, 0xEEEEEEEE, 4)
    for lane in range(lanes):
        m.write(lane, 0x12340041 + lane, 1)                    # ds_write_b8: DATA[7:0]
        m.write(0x40 + 2 * lane, 0xAAAAFF80 + lane, 2)         # ds_write_b16: DATA[15:0]
    out = [m.read(a, 4) for a in (0x00, 0x1C, 0x20, 0x40, 0x7C)]
    m.write(0xC0, 0xEEEEEEEE, 4)
    m.write(0xD0, 0xEEEEEEEE, 4)
    m.write(0xC1, (0x11223344 >> 16) & 0xFF, 1)                # ds_write_b8_d16_hi: DATA[23:16]
    out.append(m.read(0xC0, 4))
    m.write(0xC2, 0x5566AABB >> 16, 2)                         # ds_write_b16_d16_hi: DATA[31:16]
    out.append(m.read(0xC0, 4))
    m.write(0xD0, 0x9ABC1234 >> 16, 2)
    out.append(m.read(0xD0, 4))
    return out


def model_read():
    m = Lds()
    for a, v in ((0x00, 0x8081F2F3), (0x04, 0x7FFE8001), (0x20, 0x12345678), (0x24, 0xFFFFFFFF), (0x28, 0xFFFFFFF0)):
        m.write(a, v, 4)
    out = [m.read(1, 1), sext(m.read(1, 1), 8), m.read(2, 2), sext(m.read(2, 2), 16),
           0xAAAA0000 | m.read(4, 2), (m.read(6, 2) << 16) | 0xBBBB]
    old = m.read(0x20, 4)                                      # ds_mskor_rtn_b32: (old & ~DATA) | DATA2, returns old
    m.write(0x20, (old & ~0x00FF00F0 & 0xFFFFFFFF) | 0xAB00CD05, 4)
    m.write(0x24, (m.read(0x24, 4) & ~0x0F0F0F0F & 0xFFFFFFFF) | 0x01010101, 4)
    old_add = m.read(0x28, 4)                                  # ds_add_rtn_u32
    m.write(0x28, (old_add + 0x20) & 0xFFFFFFFF, 4)
    out.append(old ^ m.read(0x20, 4) ^ m.read(0x24, 4))
    out.append(old_add ^ m.read(0x28, 4))
    return out


def model_wide():
    m = Lds()
    m.write(0x5C, 0xAAAA0001, 4)
    m.write(0x6C, 0xBBBB0002, 4)
    for i, v in enumerate((0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10)):
        m.write(0x40 + 4 * i, v, 4)
    for i, v in enumerate((0x11111111, 0x22222222, 0x33333333)):
        m.write(0x60 + 4 * i, v, 4)
    return [m.read(0x40 + 4 * i, 4) for i in range(4)] + [m.read(0x60 + 4 * i, 4) for i in range(3)] + \
        [m.read(0x5C, 4) ^ m.read(0x6C, 4)]


class LdsSubwordWide(unittest.TestCase):
    def test_models_give_the_gpu_words(self):
        self.assertEqual(model_write(32), GPU["cs_w32_ds_subword_write"])
        self.assertEqual(model_read(), GPU["cs_w32_ds_subword_read"])
        self.assertEqual(model_wide(), GPU["cs_w32_ds_wide"])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            for name, want in GPU.items():
                m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % name, text)
                with self.subTest(file="/".join(rel), shader=name):
                    self.assertIsNotNone(m)
                    self.assertEqual([int(v, 0) for v in m.group(1).split(",")], want)

    def test_translated_shaders_lane0(self):
        sys.path.insert(0, SHADERS)
        from compute_fixture import compute_header
        want_all = {"cs_w32_ds_subword_write": model_write(1), "cs_w32_ds_subword_read": GPU["cs_w32_ds_subword_read"],
                    "cs_w32_ds_wide": GPU["cs_w32_ds_wide"]}
        for name, want in want_all.items():
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
