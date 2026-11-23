"""Validation and positive-control tests for the 11 binary/ternary f32 VALU ops.

Covers:
  * Binary VOP2/VOP3/SDWA/DPP (6): v_add_f32, v_sub_f32, v_subrev_f32, v_mul_f32,
    v_min_f32, v_max_f32
  * Binary VOP2/VOP3/DPP, no SDWA (1): v_fmac_f32
  * Ternary VOP3-only (2): v_fma_f32, v_med3_f32
  * Ternary VOP2-only literal K (2): v_fmaak_f32, v_fmamk_f32

Verifies that illegal encodings, operand counts/shapes/modifiers, duplicate or
conflicting output modifiers, VOP3P modifiers, invalid SDWA selectors, and
unknown modifiers all raise ``rdna2spirv.Unsupported``, while valid RDNA2 forms
translate, assemble/validate with ``spirv-as`` + ``spirv-val --target-env vulkan1.2``,
and produce exact binary32 results in ``spirv_mini_interp``.
"""
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

BINARY_SDWA_OPS = (
    "v_add_f32",
    "v_sub_f32",
    "v_subrev_f32",
    "v_mul_f32",
    "v_min_f32",
    "v_max_f32",
)
BINARY_DPP_OPS = BINARY_SDWA_OPS + ("v_fmac_f32",)
TERNARY_VOP3_OPS = ("v_fma_f32", "v_med3_f32")
TERNARY_LIT_OPS = ("v_fmaak_f32", "v_fmamk_f32")
ALL_11_OPS = BINARY_DPP_OPS + TERNARY_VOP3_OPS + TERNARY_LIT_OPS

_TMP_DIR = None
_SRC_PATH = None


def f32(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_f32_bin_tern_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f32_bin_tern_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate(lines, export="v0, v1, v2, v3"):
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_f32_bin_tern_validation\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + f"\texp mrt0 {export} done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spvasm_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise unittest.SkipTest("spirv-as / spirv-val not found on PATH")
    spvasm_path = os.path.join(_TMP_DIR.name, "out.spvasm")
    spv_path = os.path.join(_TMP_DIR.name, "out.spv")
    with open(spvasm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(spvasm_text)
    subprocess.run(
        [spirv_as, "--target-env", "vulkan1.2", spvasm_path, "-o", spv_path],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [spirv_val, "--target-env", "vulkan1.2", spv_path],
        check=True,
        capture_output=True,
        text=True,
    )


def run_and_validate(lines, export="v0, v1, v2, v3"):
    spvasm = translate(lines, export=export)
    validate_spirv(spvasm)
    return list(si.run_fragment(spvasm)["%out_mrt0"])


def _base_args(op: str) -> str:
    if op in BINARY_DPP_OPS:
        return "v0, v1, v2"
    if op == "v_fmaak_f32":
        return "v0, v1, v2, 1.0"
    if op == "v_fmamk_f32":
        return "v0, v1, 1.0, v2"
    return "v0, v1, v2, v3"


def _setup_regs():
    return [
        "v_mov_b32 v0, 1.0",
        "v_mov_b32 v1, 2.0",
        "v_mov_b32 v2, 3.0",
        "v_mov_b32 v3, 4.0",
    ]


class F32BinTernValidationNegativeTests(unittest.TestCase):
    def assertUnsupported(self, insn_line: str) -> None:
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {insn_line}"):
            translate(_setup_regs() + [insn_line])

    def test_arity_rejected_all_11(self):
        for op in BINARY_DPP_OPS:
            with self.subTest(op=op, kind="too_few"):
                self.assertUnsupported(f"{op} v0, v1")
            with self.subTest(op=op, kind="too_many"):
                self.assertUnsupported(f"{op} v0, v1, v2, v3")
        for op in TERNARY_VOP3_OPS + TERNARY_LIT_OPS:
            with self.subTest(op=op, kind="too_few"):
                self.assertUnsupported(f"{op} v0, v1, v2")
            with self.subTest(op=op, kind="too_many"):
                self.assertUnsupported(f"{op} v0, v1, v2, v3, v4")

    def test_invalid_destinations_all_11(self):
        bad_dsts = ("s0", "v[0:1]", "null", "exec_lo", "-v0", "|v0|", "sext(v0)")
        for op in ALL_11_OPS:
            srcs = _base_args(op).split(", ", 1)[1]
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    self.assertUnsupported(f"{op} {dst}, {srcs}")

    def test_invalid_sources_all_11(self):
        for op in BINARY_DPP_OPS:
            for bad in ("v0, v[1:2], v2", "v0, v1, s[0:1]", "v0, sext(v1), v2", "v0, v1, sext(v2)"):
                with self.subTest(op=op, args=bad):
                    self.assertUnsupported(f"{op} {bad}")
        for op in TERNARY_VOP3_OPS:
            for bad in (
                "v0, v[1:2], v2, v3",
                "v0, v1, v[2:3], v3",
                "v0, v1, v2, s[0:1]",
                "v0, sext(v1), v2, v3",
                "v0, v1, sext(v2), v3",
                "v0, v1, v2, sext(v3)",
            ):
                with self.subTest(op=op, args=bad):
                    self.assertUnsupported(f"{op} {bad}")
        for op in TERNARY_LIT_OPS:
            for bad in (
                "v0, v[1:2], v2, 1.0",
                "v0, v1, s[0:1], 1.0",
                "v0, sext(v1), v2, 1.0",
                "v0, v1, sext(v2), 1.0",
            ):
                with self.subTest(op=op, args=bad):
                    self.assertUnsupported(f"{op} {bad}")

    def test_illegal_encodings(self):
        illegal = [
            "v_fmac_f32_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_fma_f32_e32 v0, v1, v2, v3",
            "v_fma_f32_sdwa v0, v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_fma_f32_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_med3_f32_e32 v0, v1, v2, v3",
            "v_med3_f32_sdwa v0, v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_med3_f32_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_fmaak_f32_e64 v0, v1, v2, 1.0",
            "v_fmaak_f32_sdwa v0, v1, v2, 1.0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_fmaak_f32_dpp v0, v1, v2, 1.0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_fmamk_f32_e64 v0, v1, 1.0, v2",
            "v_fmamk_f32_sdwa v0, v1, 1.0, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_fmamk_f32_dpp v0, v1, 1.0, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        ]
        for line in illegal:
            with self.subTest(line=line):
                self.assertUnsupported(line)

    def test_e32_rejects_input_and_output_modifiers(self):
        for op in BINARY_DPP_OPS:
            for bad in (
                f"{op}_e32 v0, -v1, v2",
                f"{op}_e32 v0, |v1|, v2",
                f"{op}_e32 v0, v1, -v2",
                f"{op}_e32 v0, neg(1.0), v2",
                f"{op}_e32 v0, v1, v2 clamp",
                f"{op}_e32 v0, v1, v2 mul:2",
                f"{op}_e32 v0, v1, v2 div:2",
                f"{op}_e32 v0, v1, v2 bogus",
            ):
                with self.subTest(line=bad):
                    self.assertUnsupported(bad)

    def test_fmaak_fmamk_reject_all_modifiers(self):
        for op in TERNARY_LIT_OPS:
            for suf in ("", "_e32"):
                args = _base_args(op)
                for bad in (
                    f"{op}{suf} {args} clamp",
                    f"{op}{suf} {args} mul:2",
                    f"{op}{suf} {args} mul:4",
                    f"{op}{suf} {args} div:2",
                    f"{op}{suf} {args} op_sel:[0,0,0,0]",
                    f"{op}{suf} {args} bogus",
                    f"{op}{suf} v0, -v1, v2, 1.0" if op == "v_fmaak_f32" else f"{op}{suf} v0, -v1, 1.0, v2",
                    f"{op}{suf} v0, v1, |v2|, 1.0" if op == "v_fmaak_f32" else f"{op}{suf} v0, v1, 1.0, |v2|",
                    f"{op}{suf} v0, v1, v2, neg(1.0)" if op == "v_fmaak_f32" else f"{op}{suf} v0, v1, neg(1.0), v2",
                ):
                    with self.subTest(line=bad):
                        self.assertUnsupported(bad)

    def test_dpp_rejects_sext_and_non_dpp_modifiers(self):
        dpp_tail = "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        for op in BINARY_DPP_OPS:
            for bad in (
                # Must catch sext on src0 via self.cur_ins["args"] even after dpp_insn rewrites args[1].
                f"{op}_dpp v0, sext(v1), v2 {dpp_tail}",
                f"{op}_dpp v0, v1, sext(v2) {dpp_tail}",
                f"{op}_dpp v0, v1, v2 {dpp_tail} clamp",
                f"{op}_dpp v0, v1, v2 {dpp_tail} mul:2",
                f"{op}_dpp v0, v1, v2 {dpp_tail} div:2",
                f"{op}_dpp v0, v1, v2 {dpp_tail} op_sel:[0,0,0]",
                f"{op}_dpp v0, v1, v2 {dpp_tail} src0_sel:DWORD",
                f"{op}_dpp v0, v1, v2 {dpp_tail} bogus",
            ):
                with self.subTest(line=bad):
                    self.assertUnsupported(bad)

    def test_vop3_duplicate_conflicting_vop3p_and_unknown_mods(self):
        vop3_ops = BINARY_DPP_OPS + TERNARY_VOP3_OPS
        bad_mod_sets = (
            "clamp clamp",
            "mul:2 mul:2",
            "mul:2 div:2",
            "mul:4 div:2",
            "op_sel:[0,0,0]",
            "op_sel_hi:[1,1]",
            "neg_lo:[1,0]",
            "neg_hi:[0,1]",
            "src0_sel:DWORD",
            "dst_sel:DWORD",
            "quad_perm:[1,0,3,2]",
            "bogus_mod",
        )
        for op in vop3_ops:
            args = _base_args(op)
            for suf in ("", "_e64"):
                for mods in bad_mod_sets:
                    with self.subTest(op=op + suf, mods=mods):
                        self.assertUnsupported(f"{op}{suf} {args} {mods}")

    def test_sdwa_invalid_selectors_duplicates_and_sext(self):
        for op in BINARY_SDWA_OPS:
            for bad in (
                f"{op}_sdwa v0, sext(v1), v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD",
                f"{op}_sdwa v0, v1, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0",
                f"{op}_sdwa v0, v1, v2 dst_sel:BOGUS",
                f"{op}_sdwa v0, v1, v2 src0_sel:BOGUS",
                f"{op}_sdwa v0, v1, v2 src1_sel:BOGUS",
                f"{op}_sdwa v0, v1, v2 src2_sel:DWORD",
                f"{op}_sdwa v0, v1, v2 dst_unused:BOGUS",
                f"{op}_sdwa v0, v1, v2 clamp clamp",
                f"{op}_sdwa v0, v1, v2 mul:2 div:2",
                f"{op}_sdwa v0, v1, v2 src0_sel:DWORD src0_sel:WORD_0",
                f"{op}_sdwa v0, v1, v2 dst_sel:DWORD dst_sel:WORD_0",
                f"{op}_sdwa v0, v1, v2 dst_unused:UNUSED_PAD dst_unused:UNUSED_PRESERVE",
                f"{op}_sdwa v0, v1, v2 op_sel:[0,0,0]",
                f"{op}_sdwa v0, v1, v2 neg_lo:[1,0]",
                f"{op}_sdwa v0, v1, v2 bogus",
            ):
                with self.subTest(line=bad):
                    self.assertUnsupported(bad)


class F32BinTernValidationPositiveTests(unittest.TestCase):
    def test_binary_vop2_vop3_controls(self):
        got = run_and_validate([
            "v_mov_b32 v4, 2.0",
            "v_mov_b32 v5, 0.5",
            "v_add_f32_e32 v0, v4, v5",               # 2.5
            "v_sub_f32_e64 v1, v4, -v5 mul:2",        # (2.0 - (-0.5)) * 2 = 5.0
            "v_subrev_f32_e64 v2, v4, v5 div:2",      # (0.5 - 2.0) * 0.5 = -0.75
            "v_mul_f32 v3, -v4, |v5| clamp",          # clamp(-1.0, 0, 1) = 0.0
        ])
        self.assertEqual(got, [f32(2.5), f32(5.0), f32(-0.75), f32(0.0)])

        got2 = run_and_validate([
            "v_mov_b32 v4, -3.0",
            "v_mov_b32 v5, 2.0",
            "v_min_f32_e32 v0, v4, v5",               # -3.0
            "v_min_f32_e64 v1, |v4|, v5 mul:4 clamp", # clamp(min(3, 2) * 4) = 1.0
            "v_max_f32_e32 v2, v4, v5",               # 2.0
            "v_max_f32 v3, -v4, v5 div:2",            # max(3.0, 2.0) * 0.5 = 1.5
        ])
        self.assertEqual(got2, [f32(-3.0), f32(1.0), f32(2.0), f32(1.5)])

    def test_binary_sdwa_and_dpp_controls(self):
        got_sdwa = run_and_validate([
            "v_mov_b32 v4, 0.5",
            "v_mov_b32 v5, 0.25",
            "v_add_f32_sdwa v0, v4, v5 mul:2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_sub_f32_sdwa v1, v4, -v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_subrev_f32_sdwa v2, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_mul_f32_sdwa v3, |v4|, v5 mul:4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
        ])
        self.assertEqual(got_sdwa, [f32(1.0), f32(0.75), f32(-0.25), f32(0.5)])

        got_sdwa_mm = run_and_validate([
            "v_mov_b32 v4, -2.0",
            "v_mov_b32 v5, 0.5",
            "v_min_f32_sdwa v0, |v4|, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_max_f32_sdwa v1, -v4, v5 div:2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(got_sdwa_mm[:2], [f32(0.5), f32(1.0)])

        # DPP positive controls: validate with spirv-as/spirv-val and execute in a 4-lane subgroup.
        dpp_spvasm = translate([
            "v_mov_b32 v4, 2.0",
            "v_mov_b32 v5, 0.5",
            "v_add_f32_dpp v0, -v4, v5 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",     # -2.0 + 0.5 = -1.5
            "v_subrev_f32_dpp v1, v4, |v5| quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf", # 0.5 - 2.0 = -1.5
            "v_mul_f32_dpp v2, v4, v5 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",      # 1.0
            "v_mov_b32 v3, 1.0",
            "v_fmac_f32_dpp v3, -v4, v5 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",    # -2.0*0.5 + 1.0 = 0.0
        ])
        validate_spirv(dpp_spvasm)
        _, outs = si.run_workgroup(dpp_spvasm, invocations=4, subgroup_size=4, return_outputs=True)
        self.assertEqual(list(outs[0]["%out_mrt0"]), [f32(-1.5), f32(-1.5), f32(1.0), f32(0.0)])

    def test_fmac_fma_med3_fmaak_fmamk_controls(self):
        got = run_and_validate([
            "v_mov_b32 v4, 2.0",
            "v_mov_b32 v5, 0.5",
            "v_mov_b32 v6, 3.0",
            "v_mov_b32 v0, 1.0",
            "v_fmac_f32_e32 v0, v4, v5",               # 2.0 * 0.5 + 1.0 = 2.0
            "v_mov_b32 v1, 3.0",
            "v_fmac_f32_e64 v1, -v4, v5 div:2",        # (-2.0 * 0.5 + 3.0) * 0.5 = 1.0
            "v_fma_f32_e64 v2, v4, v5, -v6 clamp",     # clamp(2*0.5 - 3 = -2) = 0.0
            "v_med3_f32_e64 v3, -v4, |v5|, v6 mul:2",  # med3(-2.0, 0.5, 3.0) * 2 = 1.0
        ])
        self.assertEqual(got, [f32(2.0), f32(1.0), f32(0.0), f32(1.0)])

        got_lit = run_and_validate([
            "v_mov_b32 v4, 2.0",
            "v_mov_b32 v5, 3.0",
            "v_fmaak_f32 v0, v4, v5, -1.0",            # 2.0 * 3.0 + (-1.0) = 5.0
            "v_fmaak_f32_e32 v1, v4, v5, 0x3f800000",  # 2.0 * 3.0 + 1.0 = 7.0
            "v_fmamk_f32 v2, v4, -1.0, v5",            # 2.0 * (-1.0) + 3.0 = 1.0
            "v_fmamk_f32_e32 v3, v4, 0x40000000, v5",  # 2.0 * 2.0 + 3.0 = 7.0
        ])
        self.assertEqual(got_lit, [f32(5.0), f32(7.0), f32(1.0), f32(7.0)])


if __name__ == "__main__":
    unittest.main()
