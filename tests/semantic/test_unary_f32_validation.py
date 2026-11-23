"""Validation and positive-control tests for the 12 unary f32 VOP1/VOP3 ops.

Covers:
  v_rcp_f32, v_rsq_f32, v_sqrt_f32, v_exp_f32, v_log_f32, v_floor_f32,
  v_fract_f32, v_ceil_f32, v_trunc_f32, v_rndne_f32, v_sin_f32, v_cos_f32

Verifies that:
- Malformed arity, non-VGPR / 64-bit / modified destinations, 64-bit or sext()
  sources, input/output modifiers on _e32, output/non-DPP modifiers on _dpp,
  duplicate clamp, duplicate or conflicting omod, invalid/duplicate SDWA
  selectors, VOP3P modifiers (op_sel/op_sel_hi/neg_lo/neg_hi), and unknown
  modifiers are rejected with rdna2spirv.Unsupported across all 12 ops.
- Valid bare, _e32, _e64, VOP3 neg/abs, clamp, mul:2/mul:4/div:2, _sdwa, and
  _dpp forms translate cleanly, assemble and validate with spirv-as / spirv-val
  --target-env vulkan1.2 (when available on PATH), and compute expected values
  in spirv_mini_interp.
"""
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

UNARY_F32_OPS = (
    "v_rcp_f32",
    "v_rsq_f32",
    "v_sqrt_f32",
    "v_exp_f32",
    "v_log_f32",
    "v_floor_f32",
    "v_fract_f32",
    "v_ceil_f32",
    "v_trunc_f32",
    "v_rndne_f32",
    "v_sin_f32",
    "v_cos_f32",
)

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None
_SPIRV_AS = shutil.which("spirv-as")
_SPIRV_VAL = shutil.which("spirv-val")


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def u2f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_unary_f32_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_unary_f32_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines, export="v0, v0, v0, v0"):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_unary_f32_validation\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + f"\texp mrt0 {export} done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spvasm):
    if not (_SPIRV_AS and _SPIRV_VAL):
        return
    spvasm_path = os.path.join(_TMP_DIR.name, "out.spvasm")
    spv_path = os.path.join(_TMP_DIR.name, "out.spv")
    with open(spvasm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(spvasm)
    res_as = subprocess.run(
        [_SPIRV_AS, "--target-env", "vulkan1.2", spvasm_path, "-o", spv_path],
        capture_output=True,
        text=True,
    )
    if res_as.returncode != 0:
        raise AssertionError(f"spirv-as failed:\n{res_as.stderr}")
    res_val = subprocess.run(
        [_SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
        capture_output=True,
        text=True,
    )
    if res_val.returncode != 0:
        raise AssertionError(f"spirv-val failed:\n{res_val.stderr}")


def run_f32(lines, export="v0, v0, v0, v0", check_spirv=False):
    spvasm = translate(lines, export=export)
    if check_spirv:
        validate_spirv(spvasm)
    return list(si.run_fragment(spvasm)["%out_mrt0"])


def expected_unary(op, x):
    if op == "v_rcp_f32":
        return 1.0 / x
    if op == "v_rsq_f32":
        return 1.0 / math.sqrt(x)
    if op == "v_sqrt_f32":
        return math.sqrt(x)
    if op == "v_exp_f32":
        return 2.0 ** x
    if op == "v_log_f32":
        return math.log2(x)
    if op == "v_floor_f32":
        return float(math.floor(x))
    if op == "v_fract_f32":
        return x - math.floor(x)
    if op == "v_ceil_f32":
        return float(math.ceil(x))
    if op == "v_trunc_f32":
        return float(math.trunc(x))
    if op == "v_rndne_f32":
        return float(round(x))
    if op == "v_sin_f32":
        return math.sin((x - round(x)) * 2.0 * math.pi)
    if op == "v_cos_f32":
        return math.cos((x - round(x)) * 2.0 * math.pi)
    raise ValueError(op)


class UnaryF32NegativeValidationTests(unittest.TestCase):
    def assert_rejected(self, line):
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {line}"):
            translate(["v_mov_b32 v1, 0x3f800000", line])

    def test_arity_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for form in (
                f"{op}",
                f"{op} v0",
                f"{op} v0, v1, v2",
                f"{op}_e32 v0, v1, v2",
                f"{op}_e64 v0, v1, v2",
                f"{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2]",
            ):
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)

    def test_invalid_destination_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for form in (
                f"{op} s0, v1",
                f"{op} exec_lo, v1",
                f"{op} null, v1",
                f"{op} v[0:1], v2",
                f"{op} -v0, v1",
                f"{op} |v0|, v1",
                f"{op} sext(v0), v1",
                f"{op}_dpp -v0, v1 quad_perm:[1,0,3,2]",
            ):
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)

    def test_invalid_source_width_and_sext_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for form in (
                f"{op} v0, v[1:2]",
                f"{op} v0, s[0:1]",
                f"{op} v0, sext(v1)",
                f"{op}_e64 v0, sext(v1)",
                f"{op}_sdwa v0, sext(v1) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
                f"{op}_dpp v0, sext(v1) quad_perm:[1,0,3,2]",
            ):
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)

    def test_e32_modifiers_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for form in (
                f"{op}_e32 v0, -v1",
                f"{op}_e32 v0, |v1|",
                f"{op}_e32 v0, -|v1|",
                f"{op}_e32 v0, neg(1.0)",
                f"{op}_e32 v0, |1.0|",
                f"{op}_e32 v0, v1 clamp",
                f"{op}_e32 v0, v1 mul:2",
                f"{op}_e32 v0, v1 mul:4",
                f"{op}_e32 v0, v1 div:2",
                f"{op}_e32 v0, v1 op_sel:[0,0]",
                f"{op}_e32 v0, v1 dst_sel:DWORD",
            ):
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)

    def test_dpp_output_and_unknown_modifiers_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for form in (
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] clamp",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] mul:2",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] mul:4",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] div:2",
                f"{op}_dpp v0, v1 row_shr:1 op_sel:[0,0]",
                f"{op}_dpp v0, v1 row_shr:1 dst_sel:DWORD",
                f"{op}_dpp v0, v1 dpp8:[1,0,3,2,5,4,7,6] glc",
            ):
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)

    def test_duplicate_and_conflicting_output_modifiers_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for suffix in ("", "_e64", "_sdwa"):
                for mods in (
                    "clamp clamp",
                    "mul:2 mul:2",
                    "mul:2 mul:4",
                    "mul:2 div:2",
                    "mul:4 div:2",
                    "div:2 mul:2 clamp",
                ):
                    form = f"{op}{suffix} v0, v1 {mods}"
                    with self.subTest(op=op, form=form):
                        self.assert_rejected(form)

    def test_vop3p_and_unknown_modifiers_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for suffix in ("", "_e64"):
                for mod in (
                    "op_sel:[0,0]",
                    "op_sel_hi:[0,0]",
                    "neg_lo:[0,0]",
                    "neg_hi:[0,0]",
                    "mul:8",
                    "div:4",
                    "glc",
                    "high",
                    "dst_sel:DWORD",
                    "src0_sel:WORD_0",
                    "quad_perm:[1,0,3,2]",
                ):
                    form = f"{op}{suffix} v0, v1 {mod}"
                    with self.subTest(op=op, form=form):
                        self.assert_rejected(form)

    def test_sdwa_invalid_and_duplicate_selectors_rejected_all_ops(self):
        for op in UNARY_F32_OPS:
            for mods in (
                "dst_sel:BAD dst_unused:UNUSED_PAD src0_sel:DWORD",
                "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BAD",
                "dst_sel:DWORD dst_unused:BOGUS src0_sel:DWORD",
                "dst_sel:DWORD src0_sel:DWORD src1_sel:DWORD",
                "dst_sel:DWORD dst_sel:WORD_0 src0_sel:DWORD",
                "dst_sel:DWORD src0_sel:DWORD src0_sel:WORD_0",
                "dst_sel:DWORD dst_unused:UNUSED_PAD dst_unused:UNUSED_PRESERVE src0_sel:DWORD",
                "dst_sel:DWORD src0_sel:DWORD op_sel:[0,0]",
                "dst_sel:DWORD src0_sel:DWORD op_sel_hi:[0,0]",
                "dst_sel:DWORD src0_sel:DWORD neg_lo:[0,0]",
                "dst_sel:DWORD src0_sel:DWORD neg_hi:[0,0]",
                "dst_sel:DWORD src0_sel:DWORD mul:8",
                "dst_sel:DWORD src0_sel:DWORD glc",
            ):
                form = f"{op}_sdwa v0, v1 {mods}"
                with self.subTest(op=op, form=form):
                    self.assert_rejected(form)


class UnaryF32PositiveControlTests(unittest.TestCase):
    def test_plain_e32_e64_and_vop3_input_output_mods_all_ops(self):
        # Choose x = 4.0 so all 12 ops are finite and well-behaved, and -4.0 for |v1| / -v1.
        x = 4.0
        x_bits = f2u(x)
        neg_x_bits = f2u(-x)
        tol = 1e-5
        for op in UNARY_F32_OPS:
            want = expected_unary(op, x)
            lines = [
                f"v_mov_b32 v1, {x_bits:#x}",
                f"v_mov_b32 v2, {neg_x_bits:#x}",
                f"{op} v0, v1",
                f"{op}_e32 v3, v1",
                f"{op}_e64 v4, |v2|",
                f"{op}_e64 v5, -v2",
            ]
            with self.subTest(op=op, group="basic_and_input_mods"):
                got = run_f32(lines, export="v0, v3, v4, v5", check_spirv=True)
                for idx, bits in enumerate(got):
                    self.assertAlmostEqual(u2f(bits), want, delta=tol, msg=f"{op} lane {idx}")

            # Output modifiers: clamp, mul:2, mul:4, div:2, and combined mul:2 clamp.
            want_clamp = min(1.0, max(0.0, want))
            want_mul2 = want * 2.0
            want_mul4 = want * 4.0
            want_div2 = want * 0.5
            want_mul2_clamp = min(1.0, max(0.0, want * 2.0))
            out_lines = [
                f"v_mov_b32 v1, {x_bits:#x}",
                f"{op}_e64 v0, v1 clamp",
                f"{op}_e64 v3, v1 mul:2",
                f"{op}_e64 v4, v1 mul:4",
                f"{op}_e64 v5, v1 div:2",
            ]
            with self.subTest(op=op, group="output_mods"):
                got = run_f32(out_lines, export="v0, v3, v4, v5", check_spirv=True)
                for got_u, exp_f in zip(got, (want_clamp, want_mul2, want_mul4, want_div2)):
                    self.assertAlmostEqual(u2f(got_u), exp_f, delta=tol * 4.0)

            combo_lines = [
                f"v_mov_b32 v1, {x_bits:#x}",
                f"{op} v0, v1 mul:2 clamp",
                f"{op}_e32 v3, 4.0",
                f"s_mov_b32 s4, {x_bits:#x}",
                f"{op}_e32 v4, s4",
                f"{op}_e32 v5, -1.0" if op not in ("v_rsq_f32", "v_sqrt_f32", "v_log_f32") else f"{op}_e32 v5, 1.0",
            ]
            with self.subTest(op=op, group="combo_and_e32_const_sgpr"):
                got = run_f32(combo_lines, export="v0, v3, v4, v5", check_spirv=True)
                self.assertAlmostEqual(u2f(got[0]), want_mul2_clamp, delta=tol * 4.0)
                self.assertAlmostEqual(u2f(got[1]), want, delta=tol)
                self.assertAlmostEqual(u2f(got[2]), want, delta=tol)
                want5 = expected_unary(op, -1.0 if op not in ("v_rsq_f32", "v_sqrt_f32", "v_log_f32") else 1.0)
                self.assertAlmostEqual(u2f(got[3]), want5, delta=tol)

    def test_sdwa_valid_forms_all_ops(self):
        x = 4.0
        x_bits = f2u(x)
        neg_x_bits = f2u(-x)
        tol = 1e-5
        for op in UNARY_F32_OPS:
            want = expected_unary(op, x)
            want_mul2_clamp = min(1.0, max(0.0, want * 2.0))
            lines = [
                f"v_mov_b32 v1, {x_bits:#x}",
                f"v_mov_b32 v2, {neg_x_bits:#x}",
                "v_mov_b32 v4, 0xdeadbeef",
                f"{op}_sdwa v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
                f"{op}_sdwa v3, |v2| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD mul:2 clamp",
                f"{op}_sdwa v4, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD",
            ]
            with self.subTest(op=op):
                got = run_f32(lines, export="v0, v3, v4, v4", check_spirv=True)
                self.assertAlmostEqual(u2f(got[0]), want, delta=tol)
                self.assertAlmostEqual(u2f(got[1]), want_mul2_clamp, delta=tol * 4.0)
                # High 16 bits of v4 preserved from 0xdeadbeef; low 16 bits match low 16 bits of result.
                self.assertEqual(got[2] >> 16, 0xDEAD)
                self.assertEqual(got[2] & 0xFFFF, got[0] & 0xFFFF)

    def test_dpp_valid_forms_all_ops(self):
        x_bits = f2u(4.0)
        neg_x_bits = f2u(-4.0)
        for op in UNARY_F32_OPS:
            lines = [
                f"v_mov_b32 v1, {x_bits:#x}",
                f"v_mov_b32 v2, {neg_x_bits:#x}",
                f"{op}_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_dpp v3, |v2| row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1",
                f"{op}_dpp v4, v1 dpp8:[1,0,3,2,5,4,7,6]",
            ]
            with self.subTest(op=op):
                spvasm = translate(lines, export="v0, v3, v4, v4")
                validate_spirv(spvasm)


if __name__ == "__main__":
    unittest.main()
