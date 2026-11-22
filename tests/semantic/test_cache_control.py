"""buffer_gl0_inv, buffer_gl1_inv, s_dcache_inv, s_gl1_inv, s_dcache_wb: cache control as a device-scope memory barrier.

Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game files.

runtime/test/shaders/cs_w32_cache_inv.asm (words in its header, shared with check.py and scripts/run_gpu_tests.py) runs
through spirv_mini_interp. The translation of each op is one OpMemoryBarrier at Device scope with AcquireRelease |
UniformMemory, in a compute, a pixel and a vertex shader, and the module still validates.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support
import spirv_mini_interp as si

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OPS = ["buffer_gl0_inv", "buffer_gl1_inv", "s_dcache_inv", "s_gl1_inv", "s_dcache_wb"]
WANT = [0x11, 0x22, 0x33, 0x44, 0x55, 0x77, 0x88, 0x1234]
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x8088, 32, 1, 1)
FSSM = bytes(FSSM)


def barriers(spv):
    """(count, scope id, semantics id) of the OpMemoryBarrier lines of a translation, constants resolved."""
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(%\S+) = OpConstant %uint (\d+)$", spv, re.M)}
    found = re.findall(r"^\s*OpMemoryBarrier (%\S+) (%\S+)$", spv, re.M)
    return [(consts[s], consts[m]) for s, m in found]


class CacheControl(unittest.TestCase):
    def test_gpu_fixture_words(self):
        spv = support.translate_fixture("cs_w32_cache_inv", fssm=FSSM)
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual([memory[4 * i] for i in range(8)], WANT)

    def test_gpu_runners_expect_the_same_words(self):
        for rel in (("runtime", "test", "shaders", "check.py"), ("scripts", "run_gpu_tests.py")):
            with open(os.path.join(ROOT, *rel), encoding="utf-8") as f:
                text = f.read()
            m = re.search(r'"cs_w32_cache_inv"\)?[,:]\s*\[([^\]]*)\]', text)
            with self.subTest(file="/".join(rel)):
                self.assertIsNotNone(m)
                self.assertEqual([int(v, 0) for v in m.group(1).split(",")], WANT)

    def test_each_op_is_one_device_acquire_release_barrier(self):
        for op in OPS:
            with self.subTest(op=op):
                spv = support.translate_lines("cs_cacheop", [op, "s_endpgm"], fssm=FSSM)
                self.assertEqual(barriers(spv), [(1, 0x48)])      # Device, AcquireRelease (0x8) | UniformMemory (0x40)
        spv = support.translate_lines("cs_cacheop", OPS + ["s_endpgm"], fssm=FSSM)
        self.assertEqual(barriers(spv), [(1, 0x48)] * len(OPS))

    def test_no_barrier_without_the_op(self):
        spv = support.translate_lines("cs_cacheop", ["s_waitcnt_vscnt null, 0x0", "s_icache_inv", "s_endpgm"], fssm=FSSM)
        self.assertEqual(barriers(spv), [])

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate_in_every_stage(self):
        for name, fssm in (("cs_cacheval", FSSM), ("ps_cacheval", None), ("vs_cacheval", None)):
            with self.subTest(shader=name):
                spv = support.translate_lines(name, ["v_mov_b32 v1, 0", *OPS, "s_endpgm"], fssm=fssm)
                with tempfile.TemporaryDirectory() as tmp:
                    asm_path, bin_path = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                    with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv)
                    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", asm_path, "-o", bin_path], check=True)
                    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", bin_path], check=True)


if __name__ == "__main__":
    unittest.main()