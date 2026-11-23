"""v_cmp(x)_lg_{f32,f16}: ordered not-equal, emitted as OpFOrdLessThan || OpFOrdGreaterThan (Gen.ford_ne).

LG is S0 <> S1: true when the sources are ordered and differ, false when either is NaN (RDNA2 ISA VOPC). The
translator used OpFOrdNotEqual, the same predicate, but MoltenVK 1.4.2's SPIRV-Cross writes it to MSL as a plain
`!=`, which is true on NaN: on an Apple M5 ps_vcmp_f32_nan, ps_vcmp_f16(_e64) and ps_vcmpx_f16(_e64) each read
their `lg` nibble wrong. `x < y || x > y` is the same predicate (false on NaN, false for +0 against -0, false for an
infinity against itself), and SPIRV-Cross keeps it ordered. Here every lg form is checked by value with NaN, signed
zeros, infinities and ordinary operands, and no translation of an lg compare or of v_div_fmas_f64 (whose exact
residual step tests "not zero" three times) contains OpFOrdNotEqual.
"""
import struct
import unittest

import support

QNAN, SNAN, PINF, NINF, NZERO = 0x7FC00001, 0x7F800001, 0x7F800000, 0xFF800000, 0x80000000
H_QNAN, H_SNAN, H_ONE, H_TWO, H_NZERO, H_PINF = 0x7E00, 0x7C01, 0x3C00, 0x4000, 0x8000, 0x7C00


def f32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def lg(a, b):
    """The ISA's LG on two f32 bit patterns."""
    fa, fb = (struct.unpack("<f", struct.pack("<I", w))[0] for w in (a, b))
    return int(fa == fa and fb == fb and fa != fb)


# (S0, S1) f32 pairs: NaN on either side (quiet and signalling), equal, unequal, signed zeros, infinities
PAIRS = [(QNAN, f32(1.0)), (f32(1.0), QNAN), (QNAN, QNAN), (SNAN, f32(2.0)), (f32(1.0), f32(2.0)),
         (f32(2.0), f32(1.0)), (f32(1.0), f32(1.0)), (0, NZERO), (PINF, PINF), (PINF, NINF), (NINF, f32(-3.5)),
         (QNAN, PINF)]


def pixel_shader(pairs, form):
    """mrt0.x = LG of each pair as a bit (bit i = pair i), computed with `form` (e32 VCC, e64 SGPR or cmpx EXEC)."""
    lines = ["s_mov_b32 s20, exec_lo", "v_mov_b32_e32 v40, 0"]
    for i, (a, b) in enumerate(pairs):
        lines += [f"v_mov_b32_e32 v1, {a:#x}", f"v_mov_b32_e32 v2, {b:#x}"]
        if form == "e32":
            lines += ["v_cmp_lg_f32_e32 vcc_lo, v1, v2", "v_cndmask_b32_e64 v3, 0, 1, vcc_lo"]
        elif form == "e64":
            lines += ["v_cmp_lg_f32_e64 s4, v1, v2", "v_cndmask_b32_e64 v3, 0, 1, s4"]
        else:   # v_cmpx: EXEC = LG; the lanes still on write 1, then EXEC comes back
            lines += ["v_mov_b32_e32 v3, 0", "v_cmpx_lg_f32_e32 v1, v2", "v_mov_b32_e32 v3, 1",
                      "s_mov_b32 exec_lo, s20"]
        lines += [f"v_lshlrev_b32_e32 v3, {i}, v3", "v_or_b32_e32 v40, v40, v3"]
    return lines + ["exp mrt0 v40, v40, v40, v40 done vm", "s_endpgm"]


class OrderedNotEqual(unittest.TestCase):
    def test_f32_forms(self):
        want = sum(lg(a, b) << i for i, (a, b) in enumerate(PAIRS))
        self.assertEqual(want, 0b011000110000)        # only (1,2), (2,1), (+inf,-inf), (-inf,-3.5)
        for form in ("e32", "e64", "cmpx"):
            with self.subTest(form=form):
                spv = support.translate_lines(f"ps_lg_{form}", pixel_shader(PAIRS, form))
                self.assertNotIn("OpFOrdNotEqual", spv)
                self.assertEqual(support.mrt0_words(spv)[0], want)

    def test_f16(self):
        """v_cmp_lg_f16 and v_cmpx_lg_f16, low and high halves (op_sel), NaN quiet and signalling."""
        pairs16 = [(H_QNAN, H_ONE), (H_ONE, H_QNAN), (H_SNAN, H_TWO), (H_ONE, H_TWO), (H_ONE, H_ONE), (0, H_NZERO),
                   (H_PINF, H_PINF)]
        want = 0b0001000                                   # only (1, 2)
        for hi in (False, True):
            lines = ["s_mov_b32 s20, exec_lo", "v_mov_b32_e32 v40, 0", "v_mov_b32_e32 v41, 0"]
            for i, (a, b) in enumerate(pairs16):
                wa, wb = (a << 16 | 0x3C00, b << 16 | 0x3C00) if hi else (0x3C00 << 16 | a, 0x3C00 << 16 | b)
                sel = " op_sel:[1,1]" if hi else ""
                lines += [f"v_mov_b32_e32 v1, {wa:#x}", f"v_mov_b32_e32 v2, {wb:#x}",
                          f"v_cmp_lg_f16_e64 s4, v1, v2{sel}", "v_cndmask_b32_e64 v3, 0, 1, s4",
                          f"v_lshlrev_b32_e32 v3, {i}, v3", "v_or_b32_e32 v40, v40, v3"]
                if not hi:
                    lines += ["v_mov_b32_e32 v3, 0", "v_cmpx_lg_f16_e32 v1, v2", "v_mov_b32_e32 v3, 1",
                              "s_mov_b32 exec_lo, s20", f"v_lshlrev_b32_e32 v3, {i}, v3", "v_or_b32_e32 v41, v41, v3"]
            lines += ["exp mrt0 v40, v41, v40, v41 done vm", "s_endpgm"]
            with self.subTest(high=hi):
                spv = support.translate_lines("ps_lg_f16", lines)
                self.assertNotIn("OpFOrdNotEqual", spv)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], want)
                if not hi:
                    self.assertEqual(words[1], want)

    def test_div_fmas_f64_has_no_ordered_not_equal(self):
        """v_div_fmas_f64's exact-residual step tests three doubles for "not zero" (ford_ne); its values are checked
        by test_alu_div_fmas.py."""
        lines = ["v_div_fmas_f64 v[0:1], v[2:3], v[4:5], v[6:7]", "exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_div_fmas_f64", lines)
        self.assertNotIn("OpFOrdNotEqual", spv)
        self.assertGreaterEqual(spv.count("OpLogicalOr %bool"), 3)

if __name__ == "__main__":
    unittest.main()
