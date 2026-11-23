"""Operand, encoding, and modifier validation for 14 binary64 VALU ops in tools/rdna2spirv.py.

Covers:
  - v_cvt_f64_i32, v_cvt_f64_u32
  - v_trunc_f64, v_ceil_f64, v_rndne_f64, v_floor_f64
  - v_add_f64, v_cvt_f32_f64, v_rcp_f64, v_rsq_f64, v_sqrt_f64
  - v_frexp_mant_f64, v_frexp_exp_i32_f64
  - v_ldexp_f64

Checks arity, destination register kind/width/modifiers, integer-source width and float neg/abs
rejection, sext rejection, _sdwa/_dpp and invalid _e32 rejection, unknown/duplicate/conflicting
modifiers, plus positive controls with SPIR-V assembly and validation when spirv-as/spirv-val
are available on PATH. Standard library only; no game files.
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

import support  # noqa: E402,F401
from support import translate_lines  # noqa: E402
import rdna2spirv  # noqa: E402

VOP1_F64_DST2_SRC_F64 = (
    "v_trunc_f64",
    "v_ceil_f64",
    "v_rndne_f64",
    "v_floor_f64",
    "v_rcp_f64",
    "v_rsq_f64",
    "v_sqrt_f64",
    "v_frexp_mant_f64",
)
VOP1_F64_DST2_SRC_INT = ("v_cvt_f64_i32", "v_cvt_f64_u32")
VOP1_F64_DST1_SRC_F64 = ("v_cvt_f32_f64", "v_frexp_exp_i32_f64")
VOP3_ONLY_F64 = ("v_add_f64", "v_ldexp_f64")
ALL_14_OPS = (
    *VOP1_F64_DST2_SRC_F64,
    *VOP1_F64_DST2_SRC_INT,
    *VOP1_F64_DST1_SRC_F64,
    *VOP3_ONLY_F64,
)
OMOD_STRICT_OPS = (
    "v_trunc_f64",
    "v_ceil_f64",
    "v_rndne_f64",
    "v_floor_f64",
    "v_add_f64",
    "v_cvt_f32_f64",
    "v_rcp_f64",
    "v_rsq_f64",
    "v_sqrt_f64",
    "v_cvt_f64_i32",
    "v_cvt_f64_u32",
)


def _validate_spirv(spv_text):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
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


def _translate_ps(name, body_lines):
    lines = list(body_lines) + ["exp mrt0 v0, v1, v0, v1 done vm", "s_endpgm"]
    return translate_lines(name, lines)


def _valid_insn(op, suffix="", mods=""):
    tail = f" {mods}" if mods else ""
    if op == "v_add_f64":
        return f"{op}{suffix} v[0:1], v[2:3], v[4:5]{tail}"
    if op == "v_ldexp_f64":
        return f"{op}{suffix} v[0:1], v[2:3], v4{tail}"
    if op in VOP1_F64_DST2_SRC_INT:
        return f"{op}{suffix} v[0:1], v2{tail}"
    if op in VOP1_F64_DST1_SRC_F64:
        return f"{op}{suffix} v0, v[2:3]{tail}"
    return f"{op}{suffix} v[0:1], v[2:3]{tail}"


class F64ValuPositiveTests(unittest.TestCase):
    def test_valid_forms_and_modifiers_translate_and_validate(self):
        for op in ALL_14_OPS:
            suffixes = ("", "_e64") if op in VOP3_ONLY_F64 else ("", "_e32", "_e64")
            for suffix in suffixes:
                with self.subTest(op=op, suffix=suffix):
                    spv = _translate_ps(f"ps_{op}{suffix}", [_valid_insn(op, suffix)])
                    _validate_spirv(spv)

        # Valid VOP3 float neg/abs on f64 sources, clamp, and omod
        valid_cases = [
            "v_add_f64 v[0:1], -v[2:3], |v[4:5]| clamp mul:2",
            "v_cvt_f32_f64_e64 v0, -|v[2:3]| clamp div:2",
            "v_rcp_f64_e64 v[0:1], -v[2:3] clamp mul:4",
            "v_rsq_f64_e64 v[0:1], |v[2:3]| mul:2",
            "v_sqrt_f64_e64 v[0:1], |v[2:3]| clamp",
            "v_trunc_f64_e64 v[0:1], -v[2:3] clamp div:2",
            "v_ceil_f64_e64 v[0:1], |v[2:3]| mul:2",
            "v_rndne_f64_e64 v[0:1], -v[2:3] clamp",
            "v_floor_f64_e64 v[0:1], -|v[2:3]| mul:4",
            "v_frexp_mant_f64_e64 v[0:1], -|v[2:3]| clamp",
            "v_frexp_exp_i32_f64_e64 v0, -|v[2:3]|",
            "v_ldexp_f64 v[0:1], -|v[2:3]|, -5 clamp",
            "v_cvt_f64_i32_e64 v[0:1], -1 clamp mul:2",
            "v_cvt_f64_u32_e64 v[0:1], 7 clamp div:2",
            "v_floor_f64_e32 v[0:1], -0.5",
            "v_frexp_mant_f64_e32 v[0:1], -4.0",
        ]
        for idx, insn in enumerate(valid_cases):
            with self.subTest(insn=insn):
                spv = _translate_ps(f"ps_f64_pos_{idx}", [insn])
                _validate_spirv(spv)


class F64ValuValidationTests(unittest.TestCase):
    def assert_rejected(self, insn):
        with self.assertRaises(rdna2spirv.Unsupported):
            _translate_ps("ps_f64_bad", [insn])

    def test_arity_rejected(self):
        bad_insns = []
        for op in (*VOP1_F64_DST2_SRC_F64, *VOP1_F64_DST2_SRC_INT, *VOP1_F64_DST1_SRC_F64):
            dst = "v0" if op in VOP1_F64_DST1_SRC_F64 else "v[0:1]"
            src = "v2" if op in VOP1_F64_DST2_SRC_INT else "v[2:3]"
            bad_insns.extend([
                f"{op} {dst}",
                f"{op} {dst}, {src}, v4",
            ])
        for op in VOP3_ONLY_F64:
            s1 = "v4" if op == "v_ldexp_f64" else "v[4:5]"
            bad_insns.extend([
                f"{op} v[0:1], v[2:3]",
                f"{op} v[0:1], v[2:3], {s1}, v6",
            ])
        for insn in bad_insns:
            with self.subTest(insn=insn):
                self.assert_rejected(insn)

    def test_destination_kind_width_and_modifiers_rejected(self):
        for op in ALL_14_OPS:
            if op == "v_add_f64":
                srcs = "v[2:3], v[4:5]"
                wrong_dsts = ("s[0:1]", "v0", "v[0:3]", "null", "-v[0:1]", "|v[0:1]|", "sext(v[0:1])")
            elif op == "v_ldexp_f64":
                srcs = "v[2:3], v4"
                wrong_dsts = ("s[0:1]", "v0", "v[0:3]", "null", "-v[0:1]", "|v[0:1]|", "sext(v[0:1])")
            elif op in VOP1_F64_DST2_SRC_INT:
                srcs = "v2"
                wrong_dsts = ("s[0:1]", "v0", "v[0:3]", "null", "-v[0:1]", "|v[0:1]|", "sext(v[0:1])")
            elif op in VOP1_F64_DST1_SRC_F64:
                srcs = "v[2:3]"
                wrong_dsts = ("s0", "v[0:1]", "null", "vcc_lo", "-v0", "|v0|", "sext(v0)")
            else:
                srcs = "v[2:3]"
                wrong_dsts = ("s[0:1]", "v0", "v[0:3]", "null", "-v[0:1]", "|v[0:1]|", "sext(v[0:1])")
            for dst in wrong_dsts:
                insn = f"{op} {dst}, {srcs}"
                with self.subTest(insn=insn):
                    self.assert_rejected(insn)

    def test_integer_source_width_and_float_modifiers_rejected(self):
        for bad in (
            "v_cvt_f64_i32 v[0:1], v[2:3]",
            "v_cvt_f64_i32 v[0:1], s[2:3]",
            "v_cvt_f64_i32 v[0:1], -v2",
            "v_cvt_f64_i32 v[0:1], |v2|",
            "v_cvt_f64_i32 v[0:1], neg(1)",
            "v_cvt_f64_i32 v[0:1], abs(1)",
            "v_cvt_f64_i32 v[0:1], sext(v2)",
            "v_cvt_f64_u32 v[0:1], v[2:3]",
            "v_cvt_f64_u32 v[0:1], -v2",
            "v_cvt_f64_u32 v[0:1], |v2|",
            "v_cvt_f64_u32 v[0:1], sext(v2)",
            "v_ldexp_f64 v[0:1], v[2:3], v[4:5]",
            "v_ldexp_f64 v[0:1], v[2:3], s[4:5]",
            "v_ldexp_f64 v[0:1], v[2:3], -v4",
            "v_ldexp_f64 v[0:1], v[2:3], |v4|",
            "v_ldexp_f64 v[0:1], v[2:3], neg(2)",
            "v_ldexp_f64 v[0:1], v[2:3], abs(2)",
            "v_ldexp_f64 v[0:1], v[2:3], sext(v4)",
        ):
            with self.subTest(insn=bad):
                self.assert_rejected(bad)

    def test_sext_on_f64_sources_rejected(self):
        for op in ALL_14_OPS:
            if op in VOP1_F64_DST2_SRC_INT:
                continue
            dst = "v0" if op in VOP1_F64_DST1_SRC_F64 else "v[0:1]"
            if op == "v_add_f64":
                cases = [
                    f"{op} {dst}, sext(v[2:3]), v[4:5]",
                    f"{op} {dst}, v[2:3], sext(v[4:5])",
                ]
            elif op == "v_ldexp_f64":
                cases = [f"{op} {dst}, sext(v[2:3]), v4"]
            else:
                cases = [f"{op} {dst}, sext(v[2:3])"]
            for insn in cases:
                with self.subTest(insn=insn):
                    self.assert_rejected(insn)

    def test_sdwa_dpp_and_invalid_e32_encodings_rejected(self):
        for op in ALL_14_OPS:
            for enc, extra in (
                ("_sdwa", "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD"),
                ("_dpp", "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"),
            ):
                insn = _valid_insn(op, enc, extra)
                with self.subTest(insn=insn):
                    self.assert_rejected(insn)

        # VOP3-only f64 ops have no _e32 encoding
        for op in VOP3_ONLY_F64:
            with self.subTest(op=op, enc="_e32"):
                self.assert_rejected(_valid_insn(op, "_e32"))

        # VOP1 _e32 rejects clamp, omod, and neg/abs source modifiers
        for op in (*VOP1_F64_DST2_SRC_F64, *VOP1_F64_DST1_SRC_F64):
            dst = "v0" if op in VOP1_F64_DST1_SRC_F64 else "v[0:1]"
            for bad in (
                f"{op}_e32 {dst}, v[2:3] clamp",
                f"{op}_e32 {dst}, v[2:3] mul:2",
                f"{op}_e32 {dst}, -v[2:3]",
                f"{op}_e32 {dst}, |v[2:3]|",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected(bad)
        for op in VOP1_F64_DST2_SRC_INT:
            for bad in (
                f"{op}_e32 v[0:1], v2 clamp",
                f"{op}_e32 v[0:1], v2 mul:2",
            ):
                with self.subTest(insn=bad):
                    self.assert_rejected(bad)

    def test_unknown_duplicate_and_disallowed_modifiers_rejected(self):
        for op in ALL_14_OPS:
            for mod in ("op_sel:[0,0]", "op_sel_hi:[0,0]", "dst_sel:DWORD", "glc", "mul:8", "clamp clamp"):
                insn = _valid_insn(op, "", mod)
                with self.subTest(insn=insn):
                    self.assert_rejected(insn)

        for op in OMOD_STRICT_OPS:
            for mod in ("mul:2 mul:4", "mul:2 div:2", "mul:2 mul:2"):
                insn = _valid_insn(op, "", mod)
                with self.subTest(insn=insn):
                    self.assert_rejected(insn)

        # v_frexp_exp_i32_f64 disallows clamp and omod; v_ldexp_f64 disallows omod
        for bad in (
            "v_frexp_exp_i32_f64 v0, v[2:3] clamp",
            "v_frexp_exp_i32_f64 v0, v[2:3] mul:2",
            "v_frexp_exp_i32_f64 v0, v[2:3] div:2",
            "v_ldexp_f64 v[0:1], v[2:3], v4 mul:2",
            "v_ldexp_f64 v[0:1], v[2:3], v4 mul:4",
            "v_ldexp_f64 v[0:1], v[2:3], v4 div:2",
        ):
            with self.subTest(insn=bad):
                self.assert_rejected(bad)


if __name__ == "__main__":
    unittest.main()
