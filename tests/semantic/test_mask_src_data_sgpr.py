"""Gen.mask_src on an SGPR that is plain data, not a lane mask (SHADER-200, from the finding on #1218).

The translator's per-lane mask form is 0xFFFFFFFF or 0. An SGPR that lane_masks.py does not mark holds one value for
the whole wave; bit N of it is lane N's mask bit (ISA: s_and_saveexec_b32 with s4 = 0xff00ff leaves 16 lanes on, lanes
0-7 and 16-23). Until this change every lane read the raw value, so any non-zero SGPR meant "all lanes on". For a wave64
pair s[r:r+1], lanes 32-63 take their bit from r+1 at bit lane-32. A register certified as a mask is read as before,
and a value already in per-lane form (a mask ANDed with data, say) gives the same result through the bit test.

* The GPU tests runtime/test/shaders/cs_w32_mask_sgpr.asm and cs_w64_mask_sgpr.asm (fixtures from regen_fixtures.py)
  run here on one 32-wide subgroup (wave32) and one 64-wide subgroup (wave64): every word equals the GPU check's value
  (check.py CS_DUMPS). On an NVIDIA GeForce RTX 4090 (driver 617.42) run_gpu_tests.py gives the same words, for wave64
  with LEONIDA_W64_EMULATE=1, and the pre-fix translator gives 32 32 32 4294967040 32 32 16 2072 (wave32).
* --wave64-emulate is only translated here, not run: the interpreter has no OpControlBarrier and does not share
  Workgroup memory between invocations. The test checks that the emulated pair read is the same bit test.
* A few more cases as short hand-written programs, checked lane by lane: 0xff00ff gives 16 lanes, 0 none, all ones
  all, a certified mask unchanged, a mask ANDed with data, and the wave64 pairs.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

rdna2spirv = support.rdna2spirv

VSHARP = [0, 0, 0x1000, 0x31016FAC]
W32_WORDS = [16, 1028, 24, 4294967040, 32, 1040, 8, 4104]
W64_WORDS = [32, 8193, 4294967040, 64, 1048, 4120, 24, 25]


def fssm(threads):
    h = bytearray(0x148)
    struct.pack_into("<4I", h, 0x138, 0x8088, threads, 1, 1)
    return bytes(h)


def translate(name, lanes, emulate=False, lines=None):
    prev = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        if lines is None:
            return support.translate_fixture(name, fssm=fssm(lanes))
        return support.translate_lines(name, lines, fssm=fssm(lanes))
    finally:
        rdna2spirv.WAVE64_EMULATE = prev


def run(spv, lanes, subgroup):
    return si.run_workgroup(
        spv, lanes, subgroup,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % subgroup},
        push_constants=VSHARP + [0] * 28, memory={})


class GpuTestFixtures(unittest.TestCase):
    def check(self, mem, want):
        self.assertEqual([mem.get(4 * k) for k in range(8)], want)

    def test_wave32(self):
        self.check(run(translate("cs_w32_mask_sgpr", 32), 32, 32), W32_WORDS)

    def test_wave64_native_subgroup(self):
        self.check(run(translate("cs_w64_mask_sgpr", 64), 64, 64), W64_WORDS)

    def test_wave64_emulated_translation(self):
        # Same per-lane bit test, lane = local index & 63, the pair's half chosen by lane < 32.
        spv = translate("cs_w64_mask_sgpr", 64, emulate=True)
        plain = translate("cs_w64_mask_sgpr", 64)
        for text in (spv, plain):
            self.assertIn("OpShiftRightLogical", text)
            self.assertIn("OpULessThan %bool", text)
        self.assertIn("OpControlBarrier", spv)
        self.assertNotIn("OpControlBarrier", plain)

    def test_mask_anded_with_unmarked_data(self):
        # out[6] of both fixtures: a lane mask (v_cmp) ANDed with a data register (per-lane form), then saveexec.
        self.assertEqual(W32_WORDS[6], 8)       # 0xffff & 0xff00ff = lanes 0-7
        self.assertEqual(W64_WORDS[6], 24)      # lanes 0-39 & (0xff_00ff00ff) = lanes 0-7, 16-23, 32-39
        self.assertEqual(run(translate("cs_w32_mask_sgpr", 32), 32, 32).get(24), 8)
        self.assertEqual(run(translate("cs_w64_mask_sgpr", 64), 64, 64).get(24), 24)


TAIL = ["v_mov_b32 v8, 1", "v_lshlrev_b32 v10, 2, v0", "buffer_store_dword v8, v10, s[0:3], 0 offen"]


def flags(name, lanes, body, emulate=False, subgroup=None):
    """Run `body`; the list of per-lane flags (1 = the lane is on after the saveexec)."""
    spv = translate(name, lanes, emulate, lines=list(body) + TAIL)
    mem = run(spv, lanes, subgroup or (32 if emulate else lanes))
    return [mem.get(4 * i, 0) for i in range(lanes)]


def bits(v, lo, n):
    return [(v >> (lo + i)) & 1 for i in range(n)]


class HandWrittenCases(unittest.TestCase):
    def w32(self, value):
        return flags("cs_ms_w32", 32, ["s_mov_b32 exec_lo, -1", f"s_mov_b32 s4, {value}", "s_and_saveexec_b32 s6, s4"])

    def test_wave32_partial_data(self):
        got = self.w32("0xff00ff")
        self.assertEqual(sum(got), 16)
        self.assertEqual(got, bits(0xFF00FF, 0, 32))

    def test_wave32_zero_and_all_ones(self):
        self.assertEqual(self.w32("0"), [0] * 32)
        self.assertEqual(self.w32("-1"), [1] * 32)

    def test_wave32_certified_mask_unchanged(self):
        got = flags("cs_ms_cert32", 32, ["s_mov_b32 exec_lo, -1", "v_cmp_eq_u32_e64 s4, v0, 5",
                                         "s_and_saveexec_b32 s6, s4"])
        self.assertEqual(got, [1 if i == 5 else 0 for i in range(32)])

    def w64(self, lo, hi, emulate=False):
        return flags("cs_ms_w64", 64, ["s_mov_b64 exec, -1", f"s_mov_b32 s4, {lo}", f"s_mov_b32 s5, {hi}",
                                       "s_and_saveexec_b64 s[6:7], s[4:5]"], emulate=False)

    def test_wave64_pair_data(self):
        for emulate in (False,):
            with self.subTest(emulate=False):
                got = self.w64("0xff00ff", "0xf0f0f0f0", emulate)
                self.assertEqual(got, bits(0xFF00FF, 0, 32) + bits(0xF0F0F0F0, 0, 32))

    def test_wave64_high_only_zero_and_all_ones(self):
        for emulate in (False,):
            with self.subTest(emulate=False):
                self.assertEqual(self.w64("0", "1", emulate), [0] * 32 + [1] + [0] * 31)
                self.assertEqual(self.w64("0", "0", emulate), [0] * 64)
                self.assertEqual(self.w64("-1", "-1", emulate), [1] * 64)

    def test_wave64_certified_mask_unchanged(self):
        for emulate in (False,):
            with self.subTest(emulate=False):
                got = flags("cs_ms_cert64", 64, ["s_mov_b64 exec, -1", "v_cmp_eq_u32_e64 s[4:5], v0, 40",
                                                 "s_and_saveexec_b64 s[6:7], s[4:5]"], emulate=False)
                self.assertEqual(got, [1 if i == 40 else 0 for i in range(64)])

    def test_wave64_pair_anded_with_mask(self):
        # A per-lane-form result (mask & data) is read back through the bit test: same lanes as the ISA's AND.
        for emulate in (False,):
            with self.subTest(emulate=False):
                got = flags("cs_ms_and64", 64, ["s_mov_b64 exec, -1", "v_cmp_gt_u32_e64 s[10:11], 40, v0",
                                                "s_mov_b32 s4, 0xff00ff", "s_mov_b32 s5, 0xff",
                                                "s_and_b64 s[12:13], s[10:11], s[4:5]",
                                                "s_and_saveexec_b64 s[6:7], s[12:13]"], emulate=False)
                want = bits(0xFF00FF, 0, 32) + bits(0xFF, 0, 32)
                self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
