"""Regression tests for operand/modifier validation on RDNA2 16-bit integer min3/max3/med3 ops:
v_min3_i16, v_min3_u16, v_max3_i16, v_max3_u16, v_med3_i16, v_med3_u16.

Verifies:
- Rejection of non-VGPR and multi-register destinations for both dhi=0 and dhi=1.
- Rejection of multi-register sources, operand neg/abs/sext modifiers, _dpp/_sdwa encodings,
  unsupported/duplicate instruction modifiers, and wrong operand arity.
- Positive controls for dhi=0, dhi=1, _e64, clamp, and plain negative integer immediates,
  validated with spirv-as, spirv-val --target-env vulkan1.2, and spirv_mini_interp.
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK16 = 0xFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
SENTINEL = 0xA55A5AA5

INT16_OPS = (
    "v_min3_i16",
    "v_min3_u16",
    "v_max3_i16",
    "v_max3_u16",
    "v_med3_i16",
    "v_med3_u16",
)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_int16_minmax_med3_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_int16_minmax_med3_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_asm(asm: str) -> str:
    if _TMP_DIR is None:
        setUpModule()
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spvasm: str) -> None:
    sa = shutil.which("spirv-as")
    sv = shutil.which("spirv-val")
    if not (sa and sv):
        if os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS"):
            raise AssertionError("spirv-as / spirv-val required but not found on PATH")
        return
    with tempfile.TemporaryDirectory() as td:
        asm_p = os.path.join(td, "mod.spvasm")
        spv_p = os.path.join(td, "mod.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], check=True)
        subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], check=True)


def make_ps(insn_line: str, setup_lines: tuple[str, ...] = ()) -> str:
    lines = [
        "; test_int16_minmax_med3_validation",
        f"\tv_mov_b32 v0, {SENTINEL:#x}",
        "\tv_mov_b32 v1, 0x80007fff",
        "\tv_mov_b32 v2, 0xffff0001",
        "\tv_mov_b32 v3, 0x0005fffe",
        "\tv_mov_b32 v4, 0x00070003",
        *setup_lines,
        f"\t{insn_line}",
        "\texp mrt0 v0, v0, v0, v0 done vm",
        "\ts_endpgm",
    ]
    return "\n".join(lines) + "\n"


def s16(h: int) -> int:
    h &= MASK16
    return h - 0x10000 if h & 0x8000 else h


def ref_int16(op: str, a: int, b: int, c: int) -> int:
    key = s16 if op.endswith("_i16") else (lambda v: v & MASK16)
    xs = [a & MASK16, b & MASK16, c & MASK16]
    if "min3" in op:
        return min(xs, key=key) & MASK16
    if "max3" in op:
        return max(xs, key=key) & MASK16
    if "med3" in op:
        return sorted(xs, key=key)[1] & MASK16
    raise ValueError(op)


def merge_half(res16: int, dst_hi: int, old: int = SENTINEL) -> int:
    res16 &= MASK16
    return ((res16 << 16) | (old & MASK16)) if dst_hi else ((old & 0xFFFF0000) | res16)


class Int16MinMaxMed3ValidationTests(unittest.TestCase):
    """Negative validation tests across all 6 16-bit integer min3/max3/med3 ops."""

    def _assert_rejected(self, insn_line: str, setup_lines: tuple[str, ...] = ()) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_asm(make_ps(insn_line, setup_lines=setup_lines))

    def test_reject_non_vgpr_and_multireg_dst(self):
        cases = (
            ("sgpr_dst_default", "s0, v1, v2, v3"),
            ("sgpr_dst_dhi0", "s0, v1, v2, v3 op_sel:[0,0,0,0]"),
            ("sgpr_dst_dhi1", "s0, v1, v2, v3 op_sel:[0,0,0,1]"),
            ("vcc_lo_dst_dhi0", "vcc_lo, v1, v2, v3"),
            ("m0_dst_dhi1", "m0, v1, v2, v3 op_sel:[0,0,0,1]"),
            ("multivgpr_dst_dhi0", "v[0:1], v1, v2, v3"),
            ("multivgpr_dst_dhi1", "v[0:1], v1, v2, v3 op_sel:[0,0,0,1]"),
            ("multisgpr_dst_dhi0", "s[0:1], v1, v2, v3"),
        )
        for op in INT16_OPS:
            for label, tail in cases:
                with self.subTest(op=op, case=label):
                    self._assert_rejected(f"{op} {tail}")

    def test_reject_multireg_sources(self):
        cases = (
            ("src0_vgpr2", "v0, v[1:2], v2, v3"),
            ("src1_vgpr2", "v0, v1, v[2:3], v3"),
            ("src2_vgpr2", "v0, v1, v2, v[3:4]"),
            ("src0_sgpr2", "v0, s[0:1], v2, v3"),
            ("src1_sgpr2", "v0, v1, s[2:3], v3"),
            ("src2_sgpr2", "v0, v1, v2, s[4:5]"),
        )
        for op in INT16_OPS:
            for label, tail in cases:
                with self.subTest(op=op, case=label):
                    self._assert_rejected(f"{op} {tail}")

    def test_reject_operand_neg_abs_sext(self):
        cases = (
            ("dst_neg", "-v0, v1, v2, v3"),
            ("dst_abs", "|v0|, v1, v2, v3"),
            ("dst_sext", "sext(v0), v1, v2, v3"),
            ("src0_neg", "v0, -v1, v2, v3"),
            ("src1_neg", "v0, v1, -v2, v3"),
            ("src2_neg", "v0, v1, v2, -v3"),
            ("src0_neg_imm", "v0, neg(16), v2, v3"),
            ("src0_abs", "v0, |v1|, v2, v3"),
            ("src1_abs", "v0, v1, |v2|, v3"),
            ("src2_abs", "v0, v1, v2, |v3|"),
            ("src1_abs_imm", "v0, v1, |16|, v3"),
            ("src0_sext", "v0, sext(v1), v2, v3"),
            ("src1_sext", "v0, v1, sext(v2), v3"),
            ("src2_sext", "v0, v1, v2, sext(v3)"),
        )
        for op in INT16_OPS:
            for label, tail in cases:
                with self.subTest(op=op, case=label):
                    self._assert_rejected(f"{op} {tail}")

    def test_reject_dpp_and_sdwa(self):
        for op in INT16_OPS:
            with self.subTest(op=op, form="dpp"):
                self._assert_rejected(f"{op}_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf")
            with self.subTest(op=op, form="sdwa"):
                self._assert_rejected(
                    f"{op}_sdwa v0, v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
                )

    def test_reject_unsupported_and_duplicate_modifiers(self):
        bad_mods = (
            "mul:2",
            "mul:4",
            "div:2",
            "op_sel_hi:[0,0,0]",
            "neg_lo:[1,0,0]",
            "neg_hi:[0,1,0]",
            "clamp clamp",
            "op_sel:[0,0,0,0] op_sel:[0,0,0,1]",
            "glc",
            "high",
            "dst_sel:DWORD",
            "foo:1",
        )
        for op in INT16_OPS:
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self._assert_rejected(f"{op} v0, v1, v2, v3 {mod}")

    def test_reject_wrong_arity(self):
        for op in INT16_OPS:
            with self.subTest(op=op, arity=3):
                self._assert_rejected(f"{op} v0, v1, v2")
            with self.subTest(op=op, arity=5):
                self._assert_rejected(f"{op} v0, v1, v2, v3, v4")


class Int16MinMaxMed3PositiveControlsTests(unittest.TestCase):
    """Positive controls verified with spirv-as, spirv-val --target-env vulkan1.2, and spirv_mini_interp."""

    def test_valid_forms_spirv_val_and_interp(self):
        # v1 = 0x80007fff (lo=0x7fff, hi=0x8000)
        # v2 = 0xffff0001 (lo=0x0001, hi=0xffff)
        # v3 = 0x0005fffe (lo=0xfffe, hi=0x0005)
        for op in INT16_OPS:
            cases = (
                ("dhi0_default", f"{op} v0, v1, v2, v3", (0x7FFF, 0x0001, 0xFFFE), 0),
                ("dhi0_opsel", f"{op} v0, v1, v2, v3 op_sel:[0,1,0,0]", (0x7FFF, 0xFFFF, 0xFFFE), 0),
                ("dhi1_opsel", f"{op} v0, v1, v2, v3 op_sel:[1,0,1,1]", (0x8000, 0x0001, 0x0005), 1),
                ("e64_opsel", f"{op}_e64 v0, v1, v2, v3 op_sel:[1,1,0,1]", (0x8000, 0xFFFF, 0xFFFE), 1),
                ("clamp_dhi0", f"{op} v0, v1, v2, v3 clamp", (0x7FFF, 0x0001, 0xFFFE), 0),
                ("clamp_dhi1", f"{op} v0, v1, v2, v3 op_sel:[0,1,1,1] clamp", (0x7FFF, 0xFFFF, 0x0005), 1),
                ("neg_imm_dhi0", f"{op} v0, -16, v2, 5", (0xFFF0, 0x0001, 0x0005), 0),
                ("neg_imm_dhi1", f"{op} v0, v1, -1, 10 op_sel:[1,0,0,1]", (0x8000, 0xFFFF, 0x000A), 1),
            )
            for label, insn_line, (h0, h1, h2), dhi in cases:
                with self.subTest(op=op, case=label):
                    spv = translate_asm(make_ps(insn_line))
                    validate_spirv(spv)
                    got = si.run_fragment(spv)["%out_mrt0"][0]
                    want = merge_half(ref_int16(op, h0, h1, h2), dhi)
                    self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
