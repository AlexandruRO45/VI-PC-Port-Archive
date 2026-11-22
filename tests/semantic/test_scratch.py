"""Value tests for RDNA2 scratch_load_* / scratch_store_*, run through spirv_mini_interp, plus the twin of the GPU fixture
runtime/test/shaders/cs_w32_scratch.asm.

RDNA2 ISA (FLAT, scratch segment): each lane has its own private memory; the byte address is vaddr (unless `off`) +
saddr (unless `off`) + the signed 12-bit instruction offset. The translator gives each invocation a Private array of
SCRATCH_DWORDS dwords (zero at the start): dword accesses are dword aligned, byte and short accesses touch only their
bytes, a store past the array is dropped and a load past it reads 0 (hardware behaviour there is not measured).
Each case is a compute shader run as one invocation (lane 0). Standard library only; no GPU, no LLVM, no game files.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_listing  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF


def run(lines, regs):
    """Run `lines`, then store the VGPRs in `regs` to buffer words 0, 1, ...; return them."""
    body = list(lines)
    for k, r in enumerate(regs):
        body += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    mem = run_listing(body)
    return [mem.get(4 * k) for k in range(len(regs))]


class Scratch(unittest.TestCase):
    def test_dword_and_multi_dword_roundtrip(self):
        got = run(["v_mov_b32_e32 v1, 0x20", "v_mov_b32_e32 v4, 1", "v_mov_b32_e32 v5, 2", "v_mov_b32_e32 v6, 3",
                   "v_mov_b32_e32 v7, 4", "scratch_store_dwordx4 v1, v[4:7], off", "s_waitcnt_vscnt null, 0x0",
                   "scratch_load_dword v10, v1, off offset:4", "scratch_load_dwordx2 v[11:12], v1, off offset:8",
                   "scratch_load_dwordx3 v[13:15], v1, off", "scratch_load_dwordx4 v[16:19], v1, off",
                   "s_waitcnt vmcnt(0)"], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        self.assertEqual(got, [2, 3, 4, 1, 2, 3, 1, 2, 3, 4])

    def test_address_forms(self):
        got = run(["v_mov_b32_e32 v1, 0x24", "s_mov_b32 s5, 0x10", "v_mov_b32_e32 v2, 0xAAAA5555",
                   "scratch_store_dword v1, v2, off offset:-4",            # vaddr + negative offset: 0x20
                   "scratch_store_dword off, v2, s5 offset:4",              # saddr + offset: 0x14
                   "scratch_load_dword v10, off, s5 offset:16",             # saddr + offset: 0x20
                   "v_mov_b32_e32 v3, 0x14", "scratch_load_dword v11, v3, off",
                   "scratch_load_dword v12, v3, s5 offset:-16",              # vaddr + saddr + offset: 0x14
                   "s_waitcnt vmcnt(0)"], [10, 11, 12])
        self.assertEqual(got, [0xAAAA5555, 0xAAAA5555, 0xAAAA5555])

    def test_byte_and_short_accesses_keep_the_neighbours(self):
        got = run(["v_mov_b32_e32 v1, 0x20", "v_mov_b32_e32 v2, 0x11111111", "scratch_store_dword v1, v2, off",
                   "v_mov_b32_e32 v1, 0x21", "v_mov_b32_e32 v3, 0xffffffab", "scratch_store_byte v1, v3, off",
                   "v_mov_b32_e32 v1, 0x22", "v_mov_b32_e32 v4, 0xffffbeef", "scratch_store_short v1, v4, off",
                   "v_mov_b32_e32 v5, 0x12345678", "scratch_store_byte_d16_hi v1, v5, off offset:1",
                   "v_mov_b32_e32 v1, 0x20", "scratch_load_dword v10, v1, off", "scratch_load_ubyte v11, v1, off offset:1",
                   "scratch_load_sbyte v12, v1, off offset:1", "scratch_load_ushort v13, v1, off offset:2",
                   "scratch_load_sshort v14, v1, off offset:2", "s_waitcnt vmcnt(0)"], [10, 11, 12, 13, 14])
        # 0x11111111, byte 0x21 = 0xAB -> 0x1111AB11, short 0x22 = 0xBEEF -> 0xBEEFAB11, byte 0x23 = 0x34 (bits 23:16 of
        # 0x12345678) -> 0x34EFAB11
        self.assertEqual(got, [0x34EFAB11, 0xAB, 0xFFFFFFAB, 0x34EF, 0x34EF])

    def test_d16_forms_keep_the_other_half(self):
        got = run(["v_mov_b32_e32 v1, 0x40", "v_mov_b32_e32 v2, 0x8081f2f3", "scratch_store_dword v1, v2, off",
                   "v_mov_b32_e32 v10, 0xcafe0000", "v_mov_b32_e32 v11, 0x0000cafe", "v_mov_b32_e32 v12, 0xcafe0000",
                   "v_mov_b32_e32 v13, 0x0000cafe", "v_mov_b32_e32 v14, 0xcafe0000", "v_mov_b32_e32 v15, 0x0000cafe",
                   "scratch_load_ubyte_d16 v10, v1, off offset:1", "scratch_load_ubyte_d16_hi v11, v1, off offset:1",
                   "scratch_load_sbyte_d16 v12, v1, off offset:1", "scratch_load_sbyte_d16_hi v13, v1, off offset:1",
                   "scratch_load_short_d16 v14, v1, off offset:2", "scratch_load_short_d16_hi v15, v1, off offset:2",
                   "s_waitcnt vmcnt(0)"], [10, 11, 12, 13, 14, 15])
        self.assertEqual(got, [0xCAFE00F2, 0x00F2CAFE, 0xCAFEFFF2, 0xFFF2CAFE, 0xCAFE8081, 0x8081CAFE])

    def test_out_of_range(self):
        top = 4 * rdna2spirv.SCRATCH_DWORDS
        got = run([f"v_mov_b32_e32 v1, {top - 4:#x}", "v_mov_b32_e32 v2, 0x77", "scratch_store_dword v1, v2, off",
                   f"v_mov_b32_e32 v3, {top:#x}", "scratch_store_dword v3, v2, off",     # dropped
                   "v_mov_b32_e32 v10, 0x99", "scratch_load_dword v10, v3, off",         # reads 0
                   "scratch_load_dword v11, v1, off", "v_mov_b32_e32 v4, -4", "v_mov_b32_e32 v12, 0x99",
                   "scratch_load_dword v12, v4, off", "s_waitcnt vmcnt(0)"], [10, 11, 12])
        self.assertEqual(got, [0, 0x77, 0])

    def test_inactive_lane_keeps_vdst(self):
        got = run(["v_mov_b32_e32 v1, 0", "v_mov_b32_e32 v10, 0x1234", "s_mov_b32 exec_lo, 0",
                   "scratch_load_dword v10, v1, off", "s_waitcnt vmcnt(0)", "s_mov_b32 exec_lo, 1"], [10])
        self.assertEqual(got, [0x1234])

    def test_refused_forms(self):
        for line in ("scratch_load_dword v1, v2, off offset:4096", "scratch_load_dword v1, v2, off lds",
                     "scratch_load_dword v1, v[2:3], off", "scratch_load_dword v1, v2, s[2:3]"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                run_listing(["v_mov_b32_e32 v2, 0", line])


NAME = "cs_w32_scratch"
GPU = [0x20, 0x05060708, 0x090A0B0C, 0x1111AB11, 0xBEEF0000, 0xFFFFFFAB, 0x0000BEEF, 0xFFFFBEEF]


class ScratchFixture(unittest.TestCase):
    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(support.ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % NAME, text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], GPU)

    def test_translated_shader_lane0(self):
        # One lane: out[0] counts the lanes that read their own value back, which is 1 here; the rest is lane 0's.
        shaders = os.path.join(support.ROOT, "runtime", "test", "shaders")
        sys.path.insert(0, shaders)
        from compute_fixture import compute_header
        header = compute_header(os.path.join(shaders, NAME + ".asm"))
        with open(os.path.join(support.ROOT, "tests", "semantic", "fixtures", "disasm", NAME + ".s"),
                  encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines()[1:] if ln.strip() and ln.strip() != "s_barrier"]
        spv = support.translate_lines(NAME, lines, fssm=bytes(header))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        want = [1] + GPU[1:]
        self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])


if __name__ == "__main__":
    unittest.main()
