"""Validation and positive-control tests for integer/bitwise VALU ops in tools/rdna2spirv.py.

Covers:
  - v_mad_u64_u32, v_mad_i64_i32 (fixes silent write to v[0:1] when dst is s[0:1],
    plus arity, operand width, neg/abs/sext, _dpp/_sdwa, and VOP3 modifier rejection)
  - v_bfe_u32, v_bfe_i32, v_perm_b32, v_xad_u32 (arity, VGPR dst, single-register sources,
    neg/abs/sext, _dpp/_sdwa, and VOP3 modifier rejection)
  - v_xnor_b32 (arity, VGPR dst, single-register sources, neg/abs/sext, _dpp/_sdwa,
    and VOP3 modifier rejection)

Positive controls translate valid forms (including _e32/_e64, plain negative integer
immediates, and carry-out SGPR pairs on v_mad_u64_u32/v_mad_i64_i32), validate the
generated SPIR-V with spirv-as + spirv-val --target-env vulkan1.2, and verify exact
results with spirv_mini_interp.
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

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_int_vop3_misc_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_int_vop3_misc_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def _translate_ps(body_lines: list[str]) -> str:
    asm = "; test_int_vop3_misc_validation\n" + "\n".join(f"\t{ln}" for ln in body_lines) + "\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def _validate_spirv(spvasm: str) -> None:
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
        r1 = subprocess.run([sa, "--target-env", "vulkan1.2", asm_p, "-o", spv_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r1.stderr}")
        r2 = subprocess.run([sv, "--target-env", "vulkan1.2", spv_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r2.stderr}")


def _export_u32_halves(reg: str) -> list[str]:
    """Pack reg into 16-bit halves ORed with 0x40000000 so exp mrt0 preserves raw bits."""
    return [
        f"v_and_b32_e32 v20, {reg}, 0xffff",
        "v_or_b32_e32 v20, 0x40000000, v20",
        f"v_lshrrev_b32_e32 v21, 16, {reg}",
        "v_or_b32_e32 v21, 0x40000000, v21",
        "exp mrt0 v20, v21, v20, v21 done vm",
    ]


def _run_u32(setup_lines: list[str], result_reg: str = "v0") -> int:
    spv = _translate_ps(setup_lines + _export_u32_halves(result_reg))
    _validate_spirv(spv)
    out = si.run_fragment(spv)["%out_mrt0"]
    return (out[0] & 0xFFFF) | ((out[1] & 0xFFFF) << 16)


def _run_mad64(
    insn: str,
    s0: int,
    s1: int,
    s2: int,
    *,
    carry_reg: str = "s20",
) -> tuple[int, int]:
    """Run a v_mad_u64_u32 / v_mad_i64_i32 shader and return (dst_u64, carry_bit)."""
    c_lo = s2 & M32
    c_hi = (s2 >> 32) & M32
    lines = [
        f"v_mov_b32_e32 v2, {s0 & M32:#x}",
        f"v_mov_b32_e32 v3, {s1 & M32:#x}",
        f"v_mov_b32_e32 v4, {c_lo:#x}",
        f"v_mov_b32_e32 v5, {c_hi:#x}",
        insn,
    ]
    if carry_reg == "null":
        lines.append("v_mov_b32_e32 v6, 0")
    else:
        lines.append(f"v_cndmask_b32_e64 v6, 0, 1, {carry_reg}")
    # Export v0 (lo 32) in mrt0[0:2], v1 (hi 32) in mrt0[2:4], carry in mrt1[0]
    lines += [
        "v_and_b32_e32 v20, v0, 0xffff",
        "v_or_b32_e32 v20, 0x40000000, v20",
        "v_lshrrev_b32_e32 v21, 16, v0",
        "v_or_b32_e32 v21, 0x40000000, v21",
        "v_and_b32_e32 v22, v1, 0xffff",
        "v_or_b32_e32 v22, 0x40000000, v22",
        "v_lshrrev_b32_e32 v23, 16, v1",
        "v_or_b32_e32 v23, 0x40000000, v23",
        "v_or_b32_e32 v24, 0x40000000, v6",
        "exp mrt0 v20, v21, v22, v23 vm",
        "exp mrt1 v24, v24, v24, v24 done vm",
    ]
    spv = _translate_ps(lines)
    _validate_spirv(spv)
    res = si.run_fragment(spv)
    m0 = res["%out_mrt0"]
    m1 = res["%out_mrt1"]
    lo = (m0[0] & 0xFFFF) | ((m0[1] & 0xFFFF) << 16)
    hi = (m0[2] & 0xFFFF) | ((m0[3] & 0xFFFF) << 16)
    carry = m1[0] & 1
    return (lo | (hi << 32)) & M64, carry


class TestMad64Validation(unittest.TestCase):
    """Negative validation tests for v_mad_u64_u32 and v_mad_i64_i32."""

    def test_reject_non_vgpr_or_bad_width_dst(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            bad_insns = [
                f"{op} s[0:1], s20, v2, v3, v[4:5]",
                f"{op} vcc, s20, v2, v3, v[4:5]",
                f"{op} null, s20, v2, v3, v[4:5]",
                f"{op} v0, s20, v2, v3, v[4:5]",
                f"{op} v[0:3], s20, v2, v3, v[4:5]",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_bad_arity_and_operand_widths(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            bad_insns = [
                f"{op} v[0:1], s20, v2, v3",
                f"{op} v[0:1], s20, v2, v3, v[4:5], v6",
                f"{op} v[0:1], s20, v[2:3], v3, v[4:5]",
                f"{op} v[0:1], s20, v2, s[2:3], v[4:5]",
                f"{op} v[0:1], s20, v2, v3, v4",
                f"{op} v[0:1], s20, v2, v3, s4",
                f"{op} v[0:1], s20, v2, v3, v[4:7]",
                f"{op} v[0:1], v10, v2, v3, v[4:5]",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_operand_modifiers(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            bad_insns = [
                f"{op} v[0:1], s20, -v2, v3, v[4:5]",
                f"{op} v[0:1], s20, v2, |v3|, v[4:5]",
                f"{op} v[0:1], s20, sext(v2), v3, v[4:5]",
                f"{op} v[0:1], s20, neg(1), v3, v[4:5]",
                f"{op} v[0:1], s20, v2, |1|, v[4:5]",
                f"{op} v[0:1], s20, v2, v3, -v[4:5]",
                f"{op} v[0:1], s20, v2, v3, |v[4:5]|",
                f"{op} v[0:1], s20, v2, v3, neg(1)",
                f"{op} |v[0:1]|, s20, v2, v3, v[4:5]",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_mods_and_dpp_sdwa(self):
        for op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            bad_insns = [
                f"{op} v[0:1], s20, v2, v3, v[4:5] clamp",
                f"{op} v[0:1], s20, v2, v3, v[4:5] mul:2",
                f"{op} v[0:1], s20, v2, v3, v[4:5] div:2",
                f"{op} v[0:1], s20, v2, v3, v[4:5] op_sel:[1,0,0,0]",
                f"{op}_dpp v[0:1], s20, v2, v3, v[4:5] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_sdwa v[0:1], s20, v2, v3, v[4:5]",
                f"{op}_sdwa v[0:1], s20, v2, v3, v[4:5] dst_sel:DWORD src0_sel:DWORD",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])


class TestFourOpValuValidation(unittest.TestCase):
    """Negative validation tests for v_bfe_u32, v_bfe_i32, v_perm_b32, and v_xad_u32."""

    OPS = ("v_bfe_u32", "v_bfe_i32", "v_perm_b32", "v_xad_u32")

    def test_reject_non_vgpr_or_bad_width_dst(self):
        for op in self.OPS:
            for dst in ("s0", "vcc_lo", "null", "v[0:1]"):
                insn = f"{op} {dst}, v1, v2, v3"
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_bad_arity_and_wide_sources(self):
        for op in self.OPS:
            bad_insns = [
                f"{op} v0, v1, v2",
                f"{op} v0, v1, v2, v3, v4",
                f"{op} v0, v[1:2], v2, v3",
                f"{op} v0, v1, s[2:3], v3",
                f"{op} v0, v1, v2, v[3:4]",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_operand_modifiers(self):
        for op in self.OPS:
            bad_insns = [
                f"{op} v0, -v1, v2, v3",
                f"{op} v0, v1, |v2|, v3",
                f"{op} v0, v1, v2, sext(v3)",
                f"{op} v0, neg(1), 4, 8",
                f"{op} v0, 0x1234, |4|, 8",
                f"{op} v0, v1, v2, neg(1)",
                f"{op} |v0|, v1, v2, v3",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_mods_and_dpp_sdwa(self):
        for op in self.OPS:
            bad_insns = [
                f"{op} v0, v1, v2, v3 clamp",
                f"{op} v0, v1, v2, v3 mul:2",
                f"{op} v0, v1, v2, v3 op_sel:[1,0,0,0]",
                f"{op}_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_sdwa v0, v1, v2, v3",
                f"{op}_sdwa v0, v1, v2, v3 src0_sel:BYTE_0 dst_sel:DWORD",
            ]
            for insn in bad_insns:
                with self.subTest(insn=insn):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])


class TestXnorValidation(unittest.TestCase):
    """Negative validation tests for v_xnor_b32."""

    def test_reject_bad_dst_arity_and_widths(self):
        bad_insns = [
            "v_xnor_b32 s0, v1, v2",
            "v_xnor_b32 vcc_lo, v1, v2",
            "v_xnor_b32 null, v1, v2",
            "v_xnor_b32 v[0:1], v1, v2",
            "v_xnor_b32 v0, v1",
            "v_xnor_b32 v0, v1, v2, v3",
            "v_xnor_b32 v0, v[1:2], v2",
            "v_xnor_b32 v0, v1, s[2:3]",
        ]
        for insn in bad_insns:
            with self.subTest(insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])

    def test_reject_operand_modifiers_mods_and_dpp_sdwa(self):
        bad_insns = [
            "v_xnor_b32 v0, -v1, v2",
            "v_xnor_b32 v0, v1, |v2|",
            "v_xnor_b32 v0, sext(v1), v2",
            "v_xnor_b32 v0, neg(1), v2",
            "v_xnor_b32 v0, v1, |1|",
            "v_xnor_b32 |v0|, v1, v2",
            "v_xnor_b32 v0, v1, v2 clamp",
            "v_xnor_b32 v0, v1, v2 mul:2",
            "v_xnor_b32 v0, v1, v2 op_sel:[1,0,0]",
            "v_xnor_b32_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_xnor_b32_sdwa v0, v1, v2",
            "v_xnor_b32_sdwa v0, v1, v2 src0_sel:WORD_0 dst_sel:DWORD",
        ]
        for insn in bad_insns:
            with self.subTest(insn=insn):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate_ps([insn, "exp mrt0 v0, v0, v0, v0 done vm"])


class TestPositiveControls(unittest.TestCase):
    """Valid forms of all 7 ops: translate, spirv-val --target-env vulkan1.2, and interpret."""

    def test_v_mad_u64_u32_valid_forms(self):
        # Wave32 single SGPR carry-out with overflow
        val, carry = _run_mad64(
            "v_mad_u64_u32 v[0:1], s20, v2, v3, v[4:5]",
            0xFFFFFFFF, 0xFFFFFFFF, 0x00000001FFFFFFFF,
            carry_reg="s20",
        )
        full = 0xFFFFFFFF * 0xFFFFFFFF + 0x00000001FFFFFFFF
        self.assertEqual((val, carry), (full & M64, (full >> 64) & 1))

        # Wave64 SGPR pair carry-out + _e64 suffix + plain negative integer immediate
        val, carry = _run_mad64(
            "v_mad_u64_u32_e64 v[0:1], s[20:21], -1, v3, -1",
            0, 2, 0,
            carry_reg="s[20:21]",
        )
        full = 0xFFFFFFFF * 2 + 0xFFFFFFFFFFFFFFFF
        self.assertEqual((val, carry), (full & M64, (full >> 64) & 1))

        # null carry-out
        val, carry = _run_mad64(
            "v_mad_u64_u32 v[0:1], null, v2, v3, v[4:5]",
            100, 200, 300,
            carry_reg="null",
        )
        self.assertEqual((val, carry), (20300, 0))

    def test_v_mad_i64_i32_valid_forms(self):
        # Wave32 with plain negative immediate -3 and negative 64-bit accumulator
        val, sign = _run_mad64(
            "v_mad_i64_i32 v[0:1], s20, -3, v3, v[4:5]",
            0, 10, (-5) & M64,
            carry_reg="s20",
        )
        exact = (-3) * 10 + (-5)
        self.assertEqual((val, sign), (exact & M64, 1))

        # Wave64 SGPR pair carry-out + _e64 suffix
        val, sign = _run_mad64(
            "v_mad_i64_i32_e64 v[0:1], s[20:21], -1, -1, 0",
            0, 0, 0,
            carry_reg="s[20:21]",
        )
        self.assertEqual((val, sign), (1, 0))

    def test_v_bfe_valid_forms(self):
        # Unsigned register form + _e64
        r = _run_u32([
            "v_mov_b32_e32 v1, 0xabcd1234",
            "v_mov_b32_e32 v2, 8",
            "v_mov_b32_e32 v3, 12",
            "v_bfe_u32_e64 v0, v1, v2, v3",
        ])
        self.assertEqual(r, 0xD12)

        # Signed constant-folded form with plain negative immediate -1
        r = _run_u32([
            "v_bfe_i32 v0, -1, 4, 8",
        ])
        self.assertEqual(r, 0xFFFFFFFF)

    def test_v_perm_b32_valid_forms(self):
        # Known immediate selector + _e64
        r = _run_u32([
            "v_mov_b32_e32 v1, 0x11223344",
            "v_mov_b32_e32 v2, 0x55667788",
            "v_perm_b32_e64 v0, v1, v2, 0x07000c0d",
        ])
        self.assertEqual(r, 0x118800FF)

        # Dynamic VGPR selector + plain negative immediate
        r = _run_u32([
            "v_mov_b32_e32 v1, 0x11223344",
            "v_mov_b32_e32 v3, 0x07060100",
            "v_perm_b32 v0, v1, -1, v3",
        ])
        self.assertEqual(r, 0x1122FFFF)

    def test_v_xad_u32_valid_forms(self):
        # Plain negative immediate -1 (matches ps_ops8.s pattern) + _e64
        r = _run_u32([
            "v_mov_b32_e32 v1, 0x12345678",
            "v_xad_u32_e64 v0, v1, -1, 1",
        ])
        self.assertEqual(r, ((0x12345678 ^ M32) + 1) & M32)

    def test_v_xnor_b32_valid_forms(self):
        for form in ("v_xnor_b32", "v_xnor_b32_e32", "v_xnor_b32_e64"):
            with self.subTest(form=form):
                r = _run_u32([
                    "v_mov_b32_e32 v1, 0x12345678",
                    f"{form} v0, -1, v1",
                ])
                # ~((-1) ^ x) == x
                self.assertEqual(r, 0x12345678)


if __name__ == "__main__":
    unittest.main()
