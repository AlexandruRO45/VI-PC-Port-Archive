"""Packed f16 sources are widened with integer ops (Gen.f16_widen), not GLSL.std.450 UnpackHalf2x16.

The AMD Windows driver (26.10.1, RX 9060 XT / RDNA4, wave32) gave wrong halves out of UnpackHalf2x16 when its input's
high half had just been written through op_sel and the register was written again later (SHADER-777 follow-up). The
integer widening must give the same f32 for every f16 class, so these tests run v_pk_mul_f16 by 1.0 through the
translator and the mini interpreter: each half comes back unchanged (+-0, denormals, normals, +-inf), a NaN stays a
NaN, op_sel picks the right half, and no packed f16 op emits UnpackHalf2x16 any more. Synthetic values only.
"""
import random
import unittest

import support

ONE2 = 0x3C003C00   # 1.0 in both halves

EDGE = [0x0000, 0x8000, 0x0001, 0x8001, 0x0002, 0x01FF, 0x0200, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x0401, 0x3555,
        0x3C00, 0xBC00, 0x3C01, 0x7BFE, 0x7BFF, 0xFBFF, 0x5000, 0x7C00, 0xFC00]
NANS = [0x7C01, 0x7E00, 0x7FFF, 0xFC01, 0xFE00, 0xFFFF]


def is_nan(h):
    return (h & 0x7C00) == 0x7C00 and (h & 0x3FF) != 0


def run(words, mods=""):
    """v_pk_mul_f16 of 1.0 by each word (four per shader), returned as written to mrt0."""
    lines = []
    for i, w in enumerate(words):
        lines += [f"v_mov_b32 v{i}, 0", f"v_mov_b32 v{4 + i}, {w:#x}", f"v_mov_b32 v{8 + i}, {ONE2:#x}",
                  f"v_pk_mul_f16 v{i}, v{8 + i}, v{4 + i}{mods}"]
    lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_pk_f16_widen", lines))


def values():
    rng = random.Random(0xF16)
    finite = [h for h in range(0x10000) if (h & 0x7C00) != 0x7C00]
    return EDGE + rng.sample(finite, 120)


class PkF16WidenTests(unittest.TestCase):
    def check(self, mods, swap):
        vals = values()
        pairs = [(vals[i], vals[(i * 7 + 3) % len(vals)]) for i in range(len(vals))]
        pairs += [(0, 0)] * (-len(pairs) % 4)
        for k in range(0, len(pairs), 4):
            chunk = pairs[k:k + 4]
            got = run([lo | hi << 16 for lo, hi in chunk], mods)
            for (lo, hi), g in zip(chunk, got):
                want = (hi | lo << 16) if swap else (lo | hi << 16)
                self.assertEqual(g, want, f"1.0 * {lo | hi << 16:#010x}{mods} = {g:#010x}, want {want:#010x}")

    def test_each_half_unchanged(self):
        self.check("", swap=False)

    def test_op_sel_swaps_the_halves(self):
        self.check(" op_sel:[0,1] op_sel_hi:[1,0]", swap=True)

    def test_nan_stays_nan(self):
        for k in range(0, len(NANS), 2):
            lo, hi = NANS[k], NANS[k + 1]
            got = run([lo | hi << 16] + [0] * 3)[0]
            self.assertTrue(is_nan(got & 0xFFFF) and is_nan(got >> 16), f"1.0 * {lo | hi << 16:#010x} = {got:#010x}")

    def test_no_unpackhalf2x16(self):
        forms = ["v_pk_mul_f16 v0, v4, v5", "v_pk_add_f16 v0, v4, v5 op_sel:[1,0] op_sel_hi:[0,1]",
                 "v_pk_max_f16 v0, v4, v5", "v_pk_min_f16 v0, v4, v5", "v_pk_fma_f16 v0, v4, v5, v6"]
        for form in forms:
            spv = support.translate_lines("ps_pk_f16_form", [
                "v_mov_b32 v4, 0x3c004000", "v_mov_b32 v5, 0x38003c00", "v_mov_b32 v6, 0x3c003c00", form,
                "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
            self.assertNotIn("UnpackHalf2x16", spv, form)


if __name__ == "__main__":
    unittest.main()
