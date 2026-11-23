"""Validation and semantic tests for 10 32-bit RDNA2 SALU integer/bit ops.

Covers SOP1 unary ops:
  - s_brev_b32, s_abs_i32, s_sext_i32_i8, s_sext_i32_i16
and SOP2 binary ops:
  - s_ashr_i32, s_lshl1_add_u32, s_lshl2_add_u32, s_lshl3_add_u32,
    s_lshl4_add_u32, s_absdiff_i32

Negative tests verify that tools/rdna2spirv.py raises Unsupported for invalid
destinations (VGPR, 64-bit SGPR/vcc/exec pairs, register lists, immediates,
read-only specials), invalid sources (VGPR, 64-bit pairs, register lists),
operand modifiers (neg/abs/sext), invalid encoding suffixes (_e32/_e64/_dpp/_sdwa),
instruction modifiers (clamp, mul:*, div:*, op_sel:*, unknown), and wrong arity.

Positive controls verify SGPR, special 32-bit scalar registers (m0, vcc_lo,
vcc_hi, null, scc), plain negative integer immediates, and SCC semantics with
spirv_mini_interp and spirv-as + spirv-val --target-env vulkan1.2.
"""
from __future__ import annotations

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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

SOP1_OPS = (
    "s_brev_b32",
    "s_abs_i32",
    "s_sext_i32_i8",
    "s_sext_i32_i16",
)
SOP2_OPS = (
    "s_ashr_i32",
    "s_lshl1_add_u32",
    "s_lshl2_add_u32",
    "s_lshl3_add_u32",
    "s_lshl4_add_u32",
    "s_absdiff_i32",
)
ALL_OPS = SOP1_OPS + SOP2_OPS

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_salu32_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu32_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_lines(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_salu32_val\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v1, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_lines(lines: list[str]) -> tuple[int, int]:
    spv = translate_lines(lines)
    out = si.run_fragment(spv)["%out_mrt0"]
    return out[0], out[1]


def validate_spirv(spv_text: str) -> None:
    if not SPIRV_AS or not SPIRV_VAL:
        raise unittest.SkipTest("spirv-as/spirv-val not on PATH")
    spvasm_path = os.path.join(_TMP_DIR.name, "val.spvasm")
    spv_path = os.path.join(_TMP_DIR.name, "val.spv")
    with open(spvasm_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(spv_text)
    res_as = subprocess.run(
        [SPIRV_AS, "--target-env", "vulkan1.2", spvasm_path, "-o", spv_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_as.returncode != 0:
        raise AssertionError(f"spirv-as failed:\n{res_as.stderr}")
    res_val = subprocess.run(
        [SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_val.returncode != 0:
        raise AssertionError(f"spirv-val failed:\n{res_val.stderr}")


def s32(x: int) -> int:
    x &= MASK32
    return x - (1 << 32) if x >> 31 else x


def ref_brev32(x: int) -> int:
    x &= MASK32
    out = 0
    for i in range(32):
        if (x >> i) & 1:
            out |= 1 << (31 - i)
    return out


class Salu32NegativeValidationTests(unittest.TestCase):
    """Negative validation tests across all 10 32-bit SALU ops."""

    def _assert_rejected(self, line: str) -> None:
        lines = [
            "s_mov_b32 s2, 0x12345678",
            "s_mov_b32 s3, 3",
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 1",
            line,
        ]
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"Expected Unsupported for: {line}"):
            translate_lines(lines)

    def _valid_srcs(self, op: str) -> str:
        return "s2" if op in SOP1_OPS else "s2, s3"

    def test_reject_vgpr_dst_and_src(self):
        for op in ALL_OPS:
            with self.subTest(op=op, case="vgpr_dst"):
                self._assert_rejected(f"{op} v0, {self._valid_srcs(op)}")
            with self.subTest(op=op, case="vgpr_dst_imm_src"):
                imm_srcs = "-5" if op in SOP1_OPS else "1, 2"
                self._assert_rejected(f"{op} v0, {imm_srcs}")
            with self.subTest(op=op, case="vgpr_src0"):
                srcs = "v1" if op in SOP1_OPS else "v1, s3"
                self._assert_rejected(f"{op} s0, {srcs}")
            if op in SOP2_OPS:
                with self.subTest(op=op, case="vgpr_src1"):
                    self._assert_rejected(f"{op} s0, s2, v1")

    def test_reject_64bit_pairs_and_reg_lists(self):
        bad_regs = ("s[0:1]", "v[0:1]", "vcc", "exec", "[s0,s1]", "[v0,v1]")
        for op in ALL_OPS:
            for bad in bad_regs:
                with self.subTest(op=op, pos="dst", bad=bad):
                    self._assert_rejected(f"{op} {bad}, {self._valid_srcs(op)}")
                with self.subTest(op=op, pos="src0", bad=bad):
                    srcs = bad if op in SOP1_OPS else f"{bad}, s3"
                    self._assert_rejected(f"{op} s0, {srcs}")
                if op in SOP2_OPS:
                    with self.subTest(op=op, pos="src1", bad=bad):
                        self._assert_rejected(f"{op} s0, s2, {bad}")

    def test_reject_non_writable_dst(self):
        bad_dsts = ("1", "-1", "0.5", "scc", "src_scc", "off")
        for op in ALL_OPS:
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    self._assert_rejected(f"{op} {dst}, {self._valid_srcs(op)}")

    def test_reject_operand_modifiers(self):
        bad_src_mods = ("neg(s2)", "-s2", "abs(s2)", "|s2|", "sext(s2)", "neg(1)", "abs(1)", "|1|", "sext(1)")
        for op in ALL_OPS:
            with self.subTest(op=op, pos="dst_neg"):
                self._assert_rejected(f"{op} neg(s0), {self._valid_srcs(op)}")
            with self.subTest(op=op, pos="dst_abs"):
                self._assert_rejected(f"{op} abs(s0), {self._valid_srcs(op)}")
            for mod_src in bad_src_mods:
                with self.subTest(op=op, pos="src0", src=mod_src):
                    srcs = mod_src if op in SOP1_OPS else f"{mod_src}, s3"
                    self._assert_rejected(f"{op} s0, {srcs}")
                if op in SOP2_OPS:
                    with self.subTest(op=op, pos="src1", src=mod_src):
                        self._assert_rejected(f"{op} s0, s2, {mod_src}")

    def test_reject_invalid_encodings(self):
        for op in ALL_OPS:
            srcs = self._valid_srcs(op)
            with self.subTest(op=op, enc="_e32"):
                self._assert_rejected(f"{op}_e32 s0, {srcs}")
            with self.subTest(op=op, enc="_e64"):
                self._assert_rejected(f"{op}_e64 s0, {srcs}")
            with self.subTest(op=op, enc="_sdwa"):
                self._assert_rejected(f"{op}_sdwa s0, {srcs} dst_sel:DWORD")
            with self.subTest(op=op, enc="_dpp_scalar"):
                self._assert_rejected(f"{op}_dpp s0, {srcs} quad_perm:[0,1,2,3]")
            with self.subTest(op=op, enc="_dpp_vgpr"):
                v_srcs = "v1" if op in SOP1_OPS else "v1, s3"
                self._assert_rejected(f"{op}_dpp v0, {v_srcs} quad_perm:[0,1,2,3]")

    def test_reject_instruction_modifiers(self):
        bad_mods = ("clamp", "mul:2", "div:2", "op_sel:[0,0]", "glc", "foo:1")
        for op in ALL_OPS:
            srcs = self._valid_srcs(op)
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self._assert_rejected(f"{op} s0, {srcs} {mod}")

    def test_reject_wrong_arity(self):
        for op in SOP1_OPS:
            with self.subTest(op=op, arity=0):
                self._assert_rejected(f"{op}")
            with self.subTest(op=op, arity=1):
                self._assert_rejected(f"{op} s0")
            with self.subTest(op=op, arity=3):
                self._assert_rejected(f"{op} s0, s2, s3")
        for op in SOP2_OPS:
            with self.subTest(op=op, arity=1):
                self._assert_rejected(f"{op} s0")
            with self.subTest(op=op, arity=2):
                self._assert_rejected(f"{op} s0, s2")
            with self.subTest(op=op, arity=4):
                self._assert_rejected(f"{op} s0, s2, s3, s4")


class Salu32PositiveControlTests(unittest.TestCase):
    """Positive controls verified via spirv_mini_interp and spirv-val."""

    def test_all_10_ops_sgpr_and_scc_with_spirv_val(self):
        cases = [
            # (op, setup_lines, insn, scc_in, expected_dst, expected_scc)
            ("s_brev_b32", ["s_mov_b32 s2, 0x12345678"], "s_brev_b32 s0, s2", 1, ref_brev32(0x12345678), 1),
            ("s_abs_i32", ["s_mov_b32 s2, -42"], "s_abs_i32 s0, s2", 0, 42, 1),
            ("s_sext_i32_i8", ["s_mov_b32 s2, 0x1280"], "s_sext_i32_i8 s0, s2", 1, 0xFFFFFF80, 1),
            ("s_sext_i32_i16", ["s_mov_b32 s2, 0x12348001"], "s_sext_i32_i16 s0, s2", 1, 0xFFFF8001, 1),
            ("s_ashr_i32", ["s_mov_b32 s2, 0x80000000", "s_mov_b32 s3, 4"], "s_ashr_i32 s0, s2, s3", 0, 0xF8000000, 1),
            ("s_lshl1_add_u32", ["s_mov_b32 s2, 0x80000001", "s_mov_b32 s3, 5"], "s_lshl1_add_u32 s0, s2, s3", 0, 7, 1),
            ("s_lshl2_add_u32", ["s_mov_b32 s2, 3", "s_mov_b32 s3, 5"], "s_lshl2_add_u32 s0, s2, s3", 1, 17, 0),
            ("s_lshl3_add_u32", ["s_mov_b32 s2, 3", "s_mov_b32 s3, 5"], "s_lshl3_add_u32 s0, s2, s3", 1, 29, 0),
            ("s_lshl4_add_u32", ["s_mov_b32 s2, 3", "s_mov_b32 s3, 5"], "s_lshl4_add_u32 s0, s2, s3", 1, 53, 0),
            ("s_absdiff_i32", ["s_mov_b32 s2, 3", "s_mov_b32 s3, 10"], "s_absdiff_i32 s0, s2, s3", 0, 7, 1),
        ]
        for op, setup, insn, scc_in, want_d, want_scc in cases:
            with self.subTest(op=op):
                lines = setup + [
                    f"s_mov_b32 s4, {scc_in}",
                    "s_cmp_eq_u32 s4, 1",
                    insn,
                    "s_cselect_b32 s1, 1, 0",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, s1",
                ]
                spv = translate_lines(lines)
                validate_spirv(spv)
                out = si.run_fragment(spv)["%out_mrt0"]
                self.assertEqual((out[0], out[1]), (want_d, want_scc))

    def test_plain_negative_immediates_preserved(self):
        cases = [
            ("s_brev_b32", "s_brev_b32 s0, -1", 0xFFFFFFFF),
            ("s_abs_i32", "s_abs_i32 s0, -17", 17),
            ("s_sext_i32_i8", "s_sext_i32_i8 s0, -1", 0xFFFFFFFF),
            ("s_sext_i32_i16", "s_sext_i32_i16 s0, -2", 0xFFFFFFFE),
            ("s_ashr_i32", "s_ashr_i32 s0, -16, 2", 0xFFFFFFFC),
            ("s_lshl1_add_u32", "s_lshl1_add_u32 s0, 4, -1", 7),
            ("s_lshl2_add_u32", "s_lshl2_add_u32 s0, 4, -1", 15),
            ("s_lshl3_add_u32", "s_lshl3_add_u32 s0, 4, -1", 31),
            ("s_lshl4_add_u32", "s_lshl4_add_u32 s0, 4, -1", 63),
            ("s_absdiff_i32", "s_absdiff_i32 s0, -10, 5", 15),
        ]
        for op, insn, want in cases:
            with self.subTest(op=op):
                lines = [insn, "v_mov_b32 v0, s0", "v_mov_b32 v1, 0"]
                spv = translate_lines(lines)
                validate_spirv(spv)
                got, _ = si.run_fragment(spv)["%out_mrt0"][:2]
                self.assertEqual(got, want)

    def test_special_scalar_destinations_and_sources(self):
        # m0 destination & source
        lines_m0 = [
            "s_mov_b32 s2, -9",
            "s_abs_i32 m0, s2",
            "s_lshl1_add_u32 s0, m0, 1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ]
        spv_m0 = translate_lines(lines_m0)
        validate_spirv(spv_m0)
        self.assertEqual(si.run_fragment(spv_m0)["%out_mrt0"][0], 19)

        # vcc_lo destination & source
        lines_vcc = [
            "s_mov_b32 s2, 0x00000080",
            "s_sext_i32_i8 vcc_lo, s2",
            "s_absdiff_i32 s0, vcc_lo, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
        ]
        spv_vcc = translate_lines(lines_vcc)
        validate_spirv(spv_vcc)
        self.assertEqual(si.run_fragment(spv_vcc)["%out_mrt0"][0], 128)

        # null destination discards result but still updates SCC; null source reads 0
        lines_null = [
            "s_mov_b32 s0, 0xCAFEBABE",
            "s_mov_b32 s4, 0",
            "s_cmp_eq_u32 s4, 1",
            "s_absdiff_i32 null, 7, null",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
        ]
        spv_null = translate_lines(lines_null)
        validate_spirv(spv_null)
        self.assertEqual(tuple(si.run_fragment(spv_null)["%out_mrt0"][:2]), (0xCAFEBABE, 1))


if __name__ == "__main__":
    unittest.main()
