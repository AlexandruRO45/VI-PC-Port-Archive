"""Validation tests for shared Gen.opsel_bits in tools/rdna2spirv.py.

Verifies that duplicate op_sel: modifiers, malformed syntax, non-binary entries, and over-long
lists are rejected with rdna2spirv.Unsupported (both in unit calls to Gen.opsel_bits and through
representative 1-, 2-, and 3-source RDNA2 instructions), while valid full and short op_sel:[...]
forms still translate, pass spirv-val --target-env vulkan1.2, and execute identically in
spirv_mini_interp.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

TAIL = ["exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"]


def _translate_ps(body_lines):
    return translate_lines("ps_opsel_val", list(body_lines) + TAIL)


def _validate_spirv(testcase, spvasm):
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv_path = os.path.join(tmp, "mod.spv")
        p_as = subprocess.run(
            [SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", spv_path],
            input=spvasm.encode("utf-8"),
            capture_output=True,
        )
        testcase.assertEqual(p_as.returncode, 0, p_as.stderr.decode("utf-8", "replace"))
        p_val = subprocess.run(
            [SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
            capture_output=True,
        )
        testcase.assertEqual(p_val.returncode, 0, p_val.stderr.decode("utf-8", "replace"))


class OpselBitsUnitTests(unittest.TestCase):
    def test_absent_and_valid_forms(self):
        self.assertEqual(rdna2spirv.Gen.opsel_bits([], 1), ([0], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["clamp", "mul:2"], 2), ([0, 0], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel_hi:[1,1]"], 2), ([0, 0], 0))

        # Full-length forms
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1,1]"], 1), ([1], 1))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1,0,1]"], 2), ([1, 0], 1))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1,0,1,1]"], 3), ([1, 0, 1], 1))

        # Short forms zero-padded
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1]"], 1), ([1], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1]"], 2), ([1, 0], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[0,1]"], 2), ([0, 1], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1]"], 3), ([1, 0, 0], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[0,1]"], 3), ([0, 1, 0], 0))
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1,0,1]"], 3), ([1, 0, 1], 0))

        # Optional whitespace inside brackets
        self.assertEqual(rdna2spirv.Gen.opsel_bits(["op_sel:[1, 0, 1]"], 2), ([1, 0], 1))

    def test_rejects_duplicate_opsel(self):
        for mods in (
            ["op_sel:[0,0]", "op_sel:[1,0]"],
            ["op_sel:[1,0]", "clamp", "op_sel:[1,0]"],
            ["op_sel:[0]", "op_sel:[0]"],
        ):
            with self.subTest(mods=mods), self.assertRaises(rdna2spirv.Unsupported):
                rdna2spirv.Gen.opsel_bits(mods, 2)

    def test_rejects_malformed_syntax(self):
        bad_tokens = (
            "op_sel:",
            "op_sel:foo",
            "op_sel:[]",
            "op_sel:[ ]",
            "op_sel:[1,0",
            "op_sel:1,0]",
            "op_sel:[1,]",
            "op_sel:[,0]",
            "op_sel:[1,,0]",
            "op_sel:[1,0]]",
            "op_sel:[[1,0]]",
            "op_sel:[x,0]",
            "op_sel:[0,1a]",
        )
        for tok in bad_tokens:
            with self.subTest(tok=tok), self.assertRaises(rdna2spirv.Unsupported):
                rdna2spirv.Gen.opsel_bits([tok], 2)

    def test_rejects_non_binary_values(self):
        bad_tokens = (
            "op_sel:[2]",
            "op_sel:[0,2]",
            "op_sel:[-1,0]",
            "op_sel:[+1,0]",
            "op_sel:[01,0]",
            "op_sel:[0,-0]",
            "op_sel:[1,0,3]",
        )
        for tok in bad_tokens:
            with self.subTest(tok=tok), self.assertRaises(rdna2spirv.Unsupported):
                rdna2spirv.Gen.opsel_bits([tok], 2)

    def test_rejects_overlong_lists(self):
        cases = (
            (1, "op_sel:[0,0,0]"),
            (1, "op_sel:[1,0,1]"),
            (2, "op_sel:[0,0,0,0]"),
            (2, "op_sel:[1,0,0,1]"),
            (3, "op_sel:[0,0,0,0,0]"),
            (3, "op_sel:[1,0,1,0,1]"),
        )
        for nsrc, tok in cases:
            with self.subTest(nsrc=nsrc, tok=tok), self.assertRaises(rdna2spirv.Unsupported):
                rdna2spirv.Gen.opsel_bits([tok], nsrc)


class OpselBitsCallerValidationTests(unittest.TestCase):
    CALLERS = (
        # (mnemonic_and_operands, nsrc)
        ("v_rsq_f16_e64 v10, v1", 1),
        ("v_cvt_f16_u16_e64 v10, v1", 1),
        ("v_cvt_norm_i16_f16_e64 v10, v1", 1),
        ("v_frexp_mant_f16_e64 v10, v1", 1),
        ("v_add_f16_e64 v10, v1, v2", 2),
        ("v_mul_f16_e64 v10, v1, v2", 2),
        ("v_ldexp_f16 v10, v1, v2", 2),
        ("v_cmp_lt_f16_e64 s4, v1, v2", 2),
        ("v_permlane16_b32 v10, v1, s4, s5", 2),
        ("v_fma_f16 v10, v1, v2, v3", 3),
        ("v_med3_f16 v10, v1, v2, v3", 3),
        ("v_mad_u16 v10, v1, v2, v3", 3),
        ("v_mad_u32_u16 v10, v1, v2, v3", 3),
        ("v_div_fixup_f16 v10, v1, v2, v3", 3),
    )

    def test_callers_reject_invalid_opsel(self):
        setup = [
            "v_mov_b32 v1, 0x40003c00",
            "v_mov_b32 v2, 0x44004200",
            "v_mov_b32 v3, 0x38003c00",
            "v_mov_b32 v10, 0",
            "v_mov_b32 v11, 0",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
            "s_mov_b32 s4, 0",
            "s_mov_b32 s5, 0",
        ]
        for insn, nsrc in self.CALLERS:
            overlong = "op_sel:[" + ",".join(["0"] * (nsrc + 2)) + "]"
            bad_mods = (
                "op_sel:[0] op_sel:[1]",
                "op_sel:[]",
                "op_sel:[x]",
                "op_sel:[1,]",
                "op_sel:[2]",
                "op_sel:[-1]",
                overlong,
            )
            for mod in bad_mods:
                with self.subTest(insn=insn, mod=mod), self.assertRaises(rdna2spirv.Unsupported):
                    _translate_ps(setup + [f"{insn} {mod}"])


class OpselBitsPositiveControlTests(unittest.TestCase):
    # Packed f16 pairs:
    #   v1 = (lo=1.0 [0x3c00], hi=2.0 [0x4000])
    #   v2 = (lo=3.0 [0x4200], hi=4.0 [0x4400])
    #   v3 = (lo=0.5 [0x3800], hi=8.0 [0x4800])
    #   v10 preloaded with 0xbeefcafe to verify preserved half
    SETUP = [
        "v_mov_b32 v1, 0x40003c00",
        "v_mov_b32 v2, 0x44004200",
        "v_mov_b32 v3, 0x48003800",
        "v_mov_b32 v10, 0xbeefcafe",
        "v_mov_b32 v11, 0",
        "v_mov_b32 v12, 0",
        "v_mov_b32 v13, 0",
    ]

    def _run_and_validate(self, insn):
        spv = _translate_ps(self.SETUP + [insn])
        _validate_spirv(self, spv)
        return mrt0_words(spv)[0]

    def test_valid_full_and_short_opsel_forms(self):
        cases = (
            # (short_form, full_form, expected_v10_word)
            # v_add_f16: hi(v1)=2.0 + lo(v2)=3.0 -> 5.0 (0x4500) in lo, preserving hi 0xbeef
            (
                "v_add_f16_e64 v10, v1, v2 op_sel:[1]",
                "v_add_f16_e64 v10, v1, v2 op_sel:[1,0,0]",
                0xBEEF4500,
            ),
            # v_mul_f16: lo(v1)=1.0 * hi(v2)=4.0 -> 4.0 (0x4400) in lo, preserving hi 0xbeef
            (
                "v_mul_f16_e64 v10, v1, v2 op_sel:[0,1]",
                "v_mul_f16_e64 v10, v1, v2 op_sel:[0,1,0]",
                0xBEEF4400,
            ),
            # v_mul_f16 full with dst=1: hi(v1)=2.0 * hi(v2)=4.0 -> 8.0 (0x4800) in hi, preserving lo 0xcafe
            (
                "v_mul_f16_e64 v10, v1, v2 op_sel:[1,1,1]",
                "v_mul_f16_e64 v10, v1, v2 op_sel:[1,1,1]",
                0x4800CAFE,
            ),
            # v_fma_f16: hi(v1)=2.0 * lo(v2)=3.0 + hi(v3)=8.0 -> 14.0 (0x4b00) in lo, preserving hi 0xbeef
            (
                "v_fma_f16 v10, v1, v2, v3 op_sel:[1,0,1]",
                "v_fma_f16 v10, v1, v2, v3 op_sel:[1,0,1,0]",
                0xBEEF4B00,
            ),
            # v_fma_f16 short 1-entry: hi(v1)=2.0 * lo(v2)=3.0 + lo(v3)=0.5 -> 6.5 (0x4680) in lo
            (
                "v_fma_f16 v10, v1, v2, v3 op_sel:[1]",
                "v_fma_f16 v10, v1, v2, v3 op_sel:[1,0,0,0]",
                0xBEEF4680,
            ),
            # v_med3_f16: median(hi(v1)=2.0, hi(v2)=4.0, lo(v3)=0.5) -> 2.0 (0x4000) in lo
            (
                "v_med3_f16 v10, v1, v2, v3 op_sel:[1,1]",
                "v_med3_f16 v10, v1, v2, v3 op_sel:[1,1,0,0]",
                0xBEEF4000,
            ),
            # v_rsq_f16: rsq(hi(v2)=4.0) -> 0.5 (0x3800) in lo
            (
                "v_rsq_f16_e64 v10, v2 op_sel:[1]",
                "v_rsq_f16_e64 v10, v2 op_sel:[1,0]",
                0xBEEF3800,
            ),
            # v_rsq_f16 full with dst=1: rsq(hi(v2)=4.0) -> 0.5 (0x3800) in hi, preserving lo 0xcafe
            (
                "v_rsq_f16_e64 v10, v2 op_sel:[1,1]",
                "v_rsq_f16_e64 v10, v2 op_sel:[1,1]",
                0x3800CAFE,
            ),
        )
        for short_insn, full_insn, want in cases:
            with self.subTest(short=short_insn, full=full_insn):
                got_short = self._run_and_validate(short_insn)
                got_full = self._run_and_validate(full_insn)
                self.assertEqual(got_short, want, f"{got_short:#010x} != {want:#010x}")
                self.assertEqual(got_short, got_full)

    def test_valid_cvt_f16_u16_and_mad_u16(self):
        # v1 = (lo=3, hi=7), v2 = (lo=5, hi=11), v3 = (lo=2, hi=13)
        setup = [
            "v_mov_b32 v1, 0x00070003",
            "v_mov_b32 v2, 0x000b0005",
            "v_mov_b32 v3, 0x000d0002",
            "v_mov_b32 v10, 0xbeefcafe",
            "v_mov_b32 v11, 0",
            "v_mov_b32 v12, 0",
            "v_mov_b32 v13, 0",
        ]
        # v_cvt_f16_u16: hi(v1)=7 -> 7.0 (0x4700) in lo
        spv_short = _translate_ps(setup + ["v_cvt_f16_u16_e64 v10, v1 op_sel:[1]"])
        spv_full = _translate_ps(setup + ["v_cvt_f16_u16_e64 v10, v1 op_sel:[1,0]"])
        _validate_spirv(self, spv_short)
        _validate_spirv(self, spv_full)
        self.assertEqual(mrt0_words(spv_short)[0], 0xBEEF4700)
        self.assertEqual(mrt0_words(spv_short)[0], mrt0_words(spv_full)[0])

        # v_mad_u16: hi(v1)=7 * lo(v2)=5 + hi(v3)=13 = 48 (0x0030) in hi, preserving lo 0xcafe
        spv_mad = _translate_ps(setup + ["v_mad_u16 v10, v1, v2, v3 op_sel:[1,0,1,1]"])
        _validate_spirv(self, spv_mad)
        self.assertEqual(mrt0_words(spv_mad)[0], 0x0030CAFE)


if __name__ == "__main__":
    unittest.main()
