"""flat_load_*, flat_store_*, flat_atomic_*: translated as the global_* access at the 64-bit address in a VGPR pair.

Standard library only; no GPU, no LLVM, no game files.

Four GPU fixtures (runtime/test/shaders/cs_w32_flat_{load,load_d16,store,atomic}.asm, derivations in their headers,
words measured on gfx1036 and shared with check.py / scripts/run_gpu_tests.py) run through spirv_mini_interp. Then
the refusals: a flat offset outside the 11-bit unsigned range llvm-mc accepts, the forms global_* also refuses.
"""
import os
import re
import struct
import sys
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

WANT = {
    "cs_w32_flat_load": [0x11223344, 0xAB, 0xFFFFFFAB, 0xCDEF, 0xFFFFCDEF, 0x0102, 0x00A1A2A3, 0xB1B2B3B4],
    "cs_w32_flat_load_d16": [0xDEAD00AB, 0xDEADFFAB, 0xDEADCDEF, 0x00ABABCD, 0xFFABABCD, 0xCDEFABCD, 0x11223344,
                             0x5A5A5A5A],
    "cs_w32_flat_store": [0x11223344, 0xAB, 0xCDEF, 0x0102, 0x00A1A2A3, 0xB1B2B3B4, 0x5678, 0x78],
    "cs_w32_flat_atomic": [0x0005000C, 0x11112222, 0x44443333, 0x55555555, 0x0F0FFFFF, 0x0F0F000E, 0xFFFFFFFB,
                           0xFFFFFFF0],
}
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)


def run_fixture(name):
    memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
    spv = support.translate_fixture(name, fssm=FSSM)
    # One lane (a partial subgroup): the shader cuts EXEC to lane 0.
    si.run_workgroup(spv, 1, 32,
                     inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
    assert set(memory) == set(range(0, 0x100, 4)), "nothing written outside the heap"
    return memory


class FlatFixtures(unittest.TestCase):
    def test_gpu_fixture_words(self):
        for name, want in WANT.items():
            with self.subTest(shader=name):
                mem = run_fixture(name)
                self.assertEqual([f"{mem[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            for name, want in WANT.items():
                m = re.search(r'"%s"\)?[,:]\s*\[([^\]]*)\]' % name, text)
                with self.subTest(file="/".join(rel), shader=name):
                    self.assertIsNotNone(m)
                    self.assertEqual([int(v, 0) for v in m.group(1).split(",")], want)

    def test_flat_is_the_global_access(self):
        # The same program with every flat_ mnemonic spelled global_ ... off must give the same words.
        def words(prefix):
            lines = ["v_mov_b32 v10, 0x80", "v_mov_b32 v11, 0", "v_mov_b32 v20, 0x7766", "v_mov_b32 v21, 0xcafe",
                     f"{prefix}_store_dword v[10:11], v20" + (", off" if prefix == "global" else "") + " offset:4",
                     f"{prefix}_load_dword v1, v[10:11]" + (", off" if prefix == "global" else "") + " offset:4",
                     "s_waitcnt vmcnt(0)", "v_mov_b32 v25, 0",
                     "buffer_store_dword v1, v25, s[0:3], 0 offen", "s_endpgm"]
            spv = support.translate_lines("cs_flatcmp", lines, fssm=FSSM)
            memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
            si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                              "%sg_lane": i},
                             push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
            return memory
        flat, glob = words("flat"), words("global")
        self.assertEqual(flat, glob)
        self.assertEqual(flat[0], 0x7766)


class FlatRefusals(unittest.TestCase):
    def refused(self, line):
        with self.assertRaises(support.rdna2spirv.Unsupported) as cm:
            support.translate_lines("cs_flatref", ["v_mov_b32 v10, 0x80", "v_mov_b32 v11, 0", line, "s_endpgm"],
                                    fssm=FSSM)
        return str(cm.exception)

    def test_offset_outside_the_flat_range(self):
        # llvm-mc takes 0..2047 for a FLAT offset; global_* takes a signed 12-bit one, so these must not slip through.
        for off in ("-4", "2048", "4095"):
            with self.subTest(offset=off):
                self.assertIn("flat_load_dword", self.refused(f"flat_load_dword v1, v[10:11] offset:{off}"))

    def test_other_refusals_name_the_flat_op(self):
        for line in ("flat_load_dword v1, v[10:11] lds", "flat_load_dword v1, v[10:11] tfe",
                     "flat_atomic_add_x2 v[1:2], v[10:11], v[2:3] glc"):
            with self.subTest(line=line):
                self.assertIn("flat_", self.refused(line))
                self.assertNotIn("global_", self.refused(line))


if __name__ == "__main__":
    unittest.main()