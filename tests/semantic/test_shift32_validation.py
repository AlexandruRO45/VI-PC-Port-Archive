"""Validation and semantic tests for 32-bit VALU shifts (v_lshlrev_b32, v_lshrrev_b32, v_ashrrev_i32).

Covers operand arity, single-VGPR destination, single-dword source widths, rejection of float
neg/abs modifiers on all forms (including _dpp via original parsed args), rejection of sext on
non-SDWA forms (including _dpp), rejection of clamp/mul:*/div:*/op_sel*/unknown/invalid/duplicate
modifiers, plus positive controls across bare, _e32, _e64, _sdwa (incl. sext and known-SGPR
src0_sel), _dpp, and plain negative integer immediates.

Standard library only (plus spirv-as / spirv-val when on PATH); no game files.
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
from support import mrt0_words, translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
SHIFT_OPS = ("v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32")


def _s32(x):
    x &= MASK32
    return x - (1 << 32) if (x & 0x80000000) else x


def ref_shift(op, sh, val):
    sh &= 31
    val &= MASK32
    if op == "v_lshlrev_b32":
        return (val << sh) & MASK32
    if op == "v_lshrrev_b32":
        return val >> sh
    if op == "v_ashrrev_i32":
        return (_s32(val) >> sh) & MASK32
    raise ValueError(op)


def validate_spirv(spv_text):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise AssertionError("spirv-as and spirv-val must be on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        bin_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", bin_path],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", bin_path],
            capture_output=True, text=True,
        )
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def run_ps(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = translate_lines(name, lines)
    validate_spirv(spv)
    return mrt0_words(spv)[0], spv


def translate_only(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return translate_lines(name, lines)


class Shift32PositiveTests(unittest.TestCase):
    def test_bare_e32_e64_and_negative_immediates(self):
        for op in SHIFT_OPS:
            for suffix in ("", "_e32", "_e64"):
                with self.subTest(op=op, suffix=suffix):
                    word, _ = run_ps(
                        f"ps_{op}{suffix}_reg",
                        [
                            "v_mov_b32 v1, 5",
                            "v_mov_b32 v2, 0x80001234",
                            f"{op}{suffix} v0, v1, v2",
                        ],
                    )
                    self.assertEqual(word, ref_shift(op, 5, 0x80001234))

            # Plain negative integer immediate as shift amount (-1 -> 0xFFFFFFFF -> & 31 == 31)
            with self.subTest(op=op, case="neg_imm_src0"):
                word, _ = run_ps(
                    f"ps_{op}_neg_imm_src0",
                    [
                        "v_mov_b32 v2, 0x80000001",
                        f"{op}_e64 v0, -1, v2",
                    ],
                )
                self.assertEqual(word, ref_shift(op, 31, 0x80000001))

            # Plain negative integer immediate as value (-1 -> 0xFFFFFFFF)
            with self.subTest(op=op, case="neg_imm_src1"):
                word, _ = run_ps(
                    f"ps_{op}_neg_imm_src1",
                    [
                        f"{op}_e64 v0, 4, -1",
                    ],
                )
                self.assertEqual(word, ref_shift(op, 4, 0xFFFFFFFF))

    def test_valid_sdwa_and_sext(self):
        # src1 BYTE_0 = 0x84: sext -> 0xFFFFFF84, zero-ext -> 0x00000084
        # src0 BYTE_1 = 0x04: shift by 4
        for op in SHIFT_OPS:
            with self.subTest(op=op, case="sdwa_sext_src1_pad"):
                word, _ = run_ps(
                    f"ps_{op}_sdwa_sext_src1",
                    [
                        "v_mov_b32 v1, 0x00000400",
                        "v_mov_b32 v2, 0x12345684",
                        f"{op}_sdwa v0, v1, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD "
                        "src0_sel:BYTE_1 src1_sel:BYTE_0",
                    ],
                )
                self.assertEqual(word, ref_shift(op, 4, 0xFFFFFF84))

            with self.subTest(op=op, case="sdwa_sext_src0_preserve"):
                # v1 BYTE_0 = 0xFE -> sext = 0xFFFFFFFE -> & 31 = 30
                # v2 WORD_1 = 0x8001 -> zero-ext = 0x00008001
                # dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE keeps upper 16 bits of v0 (0xDEAD)
                word, _ = run_ps(
                    f"ps_{op}_sdwa_sext_src0",
                    [
                        "v_mov_b32 v0, 0xDEAD0000",
                        "v_mov_b32 v1, 0x000000FE",
                        "v_mov_b32 v2, 0x80015555",
                        f"{op}_sdwa v0, sext(v1), v2 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE "
                        "src0_sel:BYTE_0 src1_sel:WORD_1",
                    ],
                )
                expected_lo = ref_shift(op, 30, 0x00008001) & 0xFFFF
                self.assertEqual(word, 0xDEAD0000 | expected_lo)

            with self.subTest(op=op, case="sdwa_unused_sext"):
                word, _ = run_ps(
                    f"ps_{op}_sdwa_unused_sext",
                    [
                        "v_mov_b32 v1, 1",
                        "v_mov_b32 v2, 0x00000040",
                        f"{op}_sdwa v0, v1, v2 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT "
                        "src0_sel:DWORD src1_sel:DWORD",
                    ],
                )
                raw = ref_shift(op, 1, 0x00000040) & 0xFF
                expected = (raw | 0xFFFFFF00) if (raw & 0x80) else raw
                self.assertEqual(word, expected)

            with self.subTest(op=op, case="sdwa_known_sgpr_src0_sel"):
                # s0 is a known SGPR constant 0x00000300; src0_sel:BYTE_1 must extract 3, not 0!
                word, _ = run_ps(
                    f"ps_{op}_sdwa_sgpr_src0",
                    [
                        "s_mov_b32 s0, 0x00000300",
                        "v_mov_b32 v2, 0x80000010",
                        f"{op}_sdwa v0, s0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD "
                        "src0_sel:BYTE_1 src1_sel:DWORD",
                    ],
                )
                self.assertEqual(word, ref_shift(op, 3, 0x80000010))

    def test_valid_dpp(self):
        dpp_forms = (
            "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
            "dpp8:[1,0,3,2,5,4,7,6] fi:1",
        )
        for op in SHIFT_OPS:
            for idx, dpp_mod in enumerate(dpp_forms):
                with self.subTest(op=op, dpp=dpp_mod):
                    spv = translate_only(
                        f"ps_{op}_dpp_{idx}",
                        [
                            "v_mov_b32 v0, 0xCAFEBABE",
                            "v_mov_b32 v1, 4",
                            "v_mov_b32 v2, 0x800000F0",
                            f"{op}_dpp v0, v1, v2 {dpp_mod}",
                        ],
                    )
                    validate_spirv(spv)


class Shift32ValidationTests(unittest.TestCase):
    def assert_rejected(self, lines, *, name="ps_bad_shift32"):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_only(name, lines)

    def test_arity_rejected(self):
        for op in SHIFT_OPS:
            for bad in (
                f"{op} v0, v1",
                f"{op} v0, v1, v2, v3",
                f"{op}_e32 v0, v1",
                f"{op}_e64 v0, v1, v2, v3",
                f"{op}_sdwa v0, v1 dst_sel:DWORD",
                f"{op}_sdwa v0, v1, v2, v3 dst_sel:DWORD",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected([bad])

    def test_destination_kind_width_and_mods_rejected(self):
        for op in SHIFT_OPS:
            for bad in (
                f"{op} s0, v1, v2",
                f"{op} v[0:1], v1, v2",
                f"{op} null, v1, v2",
                f"{op} vcc_lo, v1, v2",
                f"{op} exec_lo, v1, v2",
                f"{op} 1, v1, v2",
                f"{op} -v0, v1, v2",
                f"{op} |v0|, v1, v2",
                f"{op} sext(v0), v1, v2",
                f"{op}_sdwa sext(v0), v1, v2 dst_sel:DWORD",
                f"{op}_sdwa -v0, v1, v2 dst_sel:DWORD",
                f"{op}_sdwa |v0|, v1, v2 dst_sel:DWORD",
                f"{op}_dpp -v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp |v0|, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp sext(v0), v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected([bad])

    def test_source_width_rejected(self):
        for op in SHIFT_OPS:
            for bad in (
                f"{op} v0, v[1:2], v3",
                f"{op} v0, s[0:1], v3",
                f"{op} v0, v1, v[2:3]",
                f"{op} v0, v1, s[2:3]",
                f"{op}_sdwa v0, v[1:2], v3 dst_sel:DWORD",
                f"{op}_sdwa v0, v1, v[2:3] dst_sel:DWORD",
                f"{op}_dpp v0, v[1:2], v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, v1, v[2:3] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected([bad])

    def test_neg_abs_rejected_on_all_forms(self):
        for op in SHIFT_OPS:
            for bad in (
                f"{op} v0, -v1, v2",
                f"{op} v0, |v1|, v2",
                f"{op} v0, v1, -v2",
                f"{op} v0, v1, |v2|",
                f"{op}_e32 v0, -v1, v2",
                f"{op}_e32 v0, v1, |v2|",
                f"{op}_e64 v0, -v1, v2",
                f"{op}_e64 v0, |v1|, v2",
                f"{op}_e64 v0, v1, -v2",
                f"{op}_e64 v0, v1, |v2|",
                f"{op}_e64 v0, neg(1), v2",
                f"{op}_e64 v0, abs(1), v2",
                f"{op}_e64 v0, v1, neg(1)",
                f"{op}_e64 v0, v1, abs(1)",
                f"{op}_sdwa v0, -v1, v2 dst_sel:DWORD src0_sel:BYTE_0",
                f"{op}_sdwa v0, |v1|, v2 dst_sel:DWORD src0_sel:BYTE_0",
                f"{op}_sdwa v0, v1, -v2 dst_sel:DWORD src1_sel:BYTE_0",
                f"{op}_sdwa v0, v1, |v2| dst_sel:DWORD src1_sel:BYTE_0",
                f"{op}_sdwa v0, sext(-v1), v2 dst_sel:DWORD src0_sel:BYTE_0",
                f"{op}_sdwa v0, v1, -sext(v2) dst_sel:DWORD src1_sel:BYTE_0",
                f"{op}_dpp v0, -v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, |v1|, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, v1, -v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, v1, |v2| quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected([bad])

    def test_sext_rejected_on_non_sdwa_forms(self):
        for op in SHIFT_OPS:
            for bad in (
                f"{op} v0, sext(v1), v2",
                f"{op} v0, v1, sext(v2)",
                f"{op}_e32 v0, sext(v1), v2",
                f"{op}_e32 v0, v1, sext(v2)",
                f"{op}_e64 v0, sext(v1), v2",
                f"{op}_e64 v0, v1, sext(v2)",
                f"{op}_dpp v0, sext(v1), v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v0, v1, sext(v2) quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected([bad])

    def test_instruction_modifiers_rejected(self):
        for op in SHIFT_OPS:
            # Bare / _e32 / _e64: reject clamp, mul:*, div:*, op_sel*, SDWA/DPP/unknown mods
            for suffix in ("", "_e32", "_e64"):
                for mod in (
                    "clamp",
                    "mul:2",
                    "mul:4",
                    "div:2",
                    "op_sel:[0,0,0]",
                    "op_sel_hi:[0,0,0]",
                    "dst_sel:WORD_0",
                    "src0_sel:BYTE_0",
                    "quad_perm:[1,0,3,2]",
                    "glc",
                ):
                    insn = f"{op}{suffix} v0, v1, v2 {mod}"
                    with self.subTest(insn=insn):
                        self.assert_rejected([insn])

            # _sdwa: reject clamp/omod/op_sel/unknown, invalid selectors/dst_unused, duplicates
            for mod in (
                "clamp",
                "mul:2",
                "div:2",
                "op_sel:[0,0,0]",
                "glc",
                "dst_sel:BOGUS",
                "src0_sel:BOGUS",
                "src1_sel:BOGUS",
                "dst_unused:BOGUS",
                "src0_sel:BYTE_0 src0_sel:BYTE_1",
                "src1_sel:WORD_0 src1_sel:WORD_1",
                "dst_sel:BYTE_0 dst_sel:WORD_0",
                "dst_unused:UNUSED_PAD dst_unused:UNUSED_SEXT",
            ):
                insn = f"{op}_sdwa v0, v1, v2 {mod}"
                with self.subTest(insn=insn):
                    self.assert_rejected([insn])

            # _dpp: reject non-DPP modifiers
            for mod in (
                "clamp",
                "mul:2",
                "div:2",
                "op_sel:[0,0,0]",
                "src0_sel:BYTE_0",
                "glc",
            ):
                insn = f"{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf {mod}"
                with self.subTest(insn=insn):
                    self.assert_rejected([insn])


if __name__ == "__main__":
    unittest.main()
