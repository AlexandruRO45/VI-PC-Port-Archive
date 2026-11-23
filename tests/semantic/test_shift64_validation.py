"""Validation and semantic tests for RDNA2 64-bit VALU shifts:
v_lshlrev_b64, v_lshrrev_b64, v_ashrrev_i64.

Verifies:
- Positive controls across all three ops (VGPR/SGPR pairs, 64-bit immediates including plain
  negative integers on both shift count and value, _e64 suffix, special 32-bit shift source)
  validated with spirv-as, spirv-val --target-env vulkan1.2, and spirv_mini_interp.
- Rejection with rdna2spirv.Unsupported for:
  * non-VGPR destinations (e.g. s[0:1], [s0, s1], vcc, exec, null, imm), preventing silent
    miscompilation into v[0:1];
  * wrong destination width (1, 3, 4 registers) and wrong operand arity (0, 1, 2, 4 args);
  * multi-register shift amounts on src0 (a[1]);
  * wrong-width or non-pair/non-imm src1 (a[2]);
  * neg/abs/sext operand modifiers on a[0..2];
  * _dpp and _sdwa suffixes and any non-empty instruction modifiers (clamp, mul:*, div:*,
    op_sel:*, unknown modifiers).
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
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M64 = (1 << 64) - 1
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def s64(x: int) -> int:
    x &= M64
    return x - (1 << 64) if x >> 63 else x


OPS = {
    "v_lshlrev_b64": lambda n, x: (x << (n & 63)) & M64,
    "v_lshrrev_b64": lambda n, x: (x & M64) >> (n & 63),
    "v_ashrrev_i64": lambda n, x: (s64(x) >> (n & 63)) & M64,
}

EXPORT_HALVES = [
    "v_and_b32 v5, 0xffff, v0",
    "v_lshrrev_b32 v6, 16, v0",
    "v_and_b32 v7, 0xffff, v1",
    "v_lshrrev_b32 v8, 16, v1",
    "v_or_b32 v5, 0x40000000, v5",
    "v_or_b32 v6, 0x40000000, v6",
    "v_or_b32 v7, 0x40000000, v7",
    "v_or_b32 v8, 0x40000000, v8",
]

_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_shift64_val.s")
    with open(os.path.join(_TMP_DIR.name, "ps_shift64_val.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    body = list(lines) + EXPORT_HALVES
    asm = "; test_shift64_val\n" + "".join(f"\t{ln}\n" for ln in body) + "\texp mrt0 v5, v6, v7, v8 done vm\n\ts_endpgm\n"
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def validate_spirv(spv_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as d:
        asm_p = os.path.join(d, "s.spvasm")
        bin_p = os.path.join(d, "s.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", bin_p], capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed: {r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", bin_p], capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed: {r2.stderr}")


def run_and_validate(lines: list[str]) -> int:
    spv = translate_ps(lines)
    validate_spirv(spv)
    w = si.run_fragment(spv)["%out_mrt0"]
    assert all(x >> 16 == 0x4000 for x in w), [hex(x) for x in w]
    return sum((x & 0xFFFF) << (16 * i) for i, x in enumerate(w))


class Shift64ValidationTests(unittest.TestCase):
    def test_positive_controls_vgpr_sgpr_imm_and_e64(self):
        val = 0x81234567FEDCBA98
        val_lo, val_hi = val & 0xFFFFFFFF, val >> 32
        for op, fn in OPS.items():
            cases = [
                (
                    "vgpr_pair_vgpr_shift",
                    [f"v_mov_b32 v4, 5", f"v_mov_b32 v2, {val_lo:#x}", f"v_mov_b32 v3, {val_hi:#x}",
                     f"{op} v[0:1], v4, v[2:3]"],
                    fn(5, val),
                ),
                (
                    "sgpr_pair_sgpr_shift",
                    [f"s_mov_b32 s4, 33", f"s_mov_b32 s2, {val_lo:#x}", f"s_mov_b32 s3, {val_hi:#x}",
                     f"{op} v[0:1], s4, s[2:3]"],
                    fn(33, val),
                ),
                (
                    "e64_suffix",
                    [f"v_mov_b32 v4, 12", f"v_mov_b32 v2, {val_lo:#x}", f"v_mov_b32 v3, {val_hi:#x}",
                     f"{op}_e64 v[0:1], v4, v[2:3]"],
                    fn(12, val),
                ),
                (
                    "positive_imm_src_and_shift",
                    [f"{op} v[0:1], 4, 0x12345678"],
                    fn(4, 0x12345678),
                ),
                (
                    "negative_imm_src",
                    [f"{op} v[0:1], 4, -16"],
                    fn(4, (-16) & M64),
                ),
                (
                    "negative_imm_shift",
                    [f"v_mov_b32 v2, {val_lo:#x}", f"v_mov_b32 v3, {val_hi:#x}",
                     f"{op} v[0:1], -1, v[2:3]"],
                    fn(63, val),
                ),
                (
                    "null_shift_source",
                    [f"v_mov_b32 v2, {val_lo:#x}", f"v_mov_b32 v3, {val_hi:#x}",
                     f"{op} v[0:1], null, v[2:3]"],
                    fn(0, val),
                ),
            ]
            for label, lines, expected in cases:
                with self.subTest(op=op, case=label):
                    self.assertEqual(run_and_validate(lines), expected)

    def test_reject_non_vgpr_and_bad_width_destinations(self):
        bad_dsts = [
            "s[0:1]",      # SGPR pair: previously miscompiled silently into v[0:1]
            "[s0, s1]",    # slist pair
            "vcc",
            "exec",
            "null",
            "0",
            "v0",          # 1-register VGPR
            "v[0:2]",      # 3-register VGPR
            "v[0:3]",      # 4-register VGPR
        ]
        for op in OPS:
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_ps([f"{op} {dst}, v4, v[2:3]"])

    def test_reject_bad_arity(self):
        for op in OPS:
            for line in (
                op,
                f"{op} v[0:1]",
                f"{op} v[0:1], v4",
                f"{op} v[0:1], v4, v[2:3], v5",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_ps([line])

    def test_reject_bad_shift_and_src_widths(self):
        for op in OPS:
            for line in (
                # Multi-register shift amount (a[1])
                f"{op} v[0:1], v[4:5], v[2:3]",
                f"{op} v[0:1], s[4:5], v[2:3]",
                f"{op} v[0:1], [s4, s5], v[2:3]",
                # Wrong-width or unsupported 64-bit source (a[2])
                f"{op} v[0:1], v4, v2",
                f"{op} v[0:1], v4, v[2:4]",
                f"{op} v[0:1], v4, s2",
                f"{op} v[0:1], v4, s[2:4]",
                f"{op} v[0:1], v4, null",
                f"{op} v[0:1], v4, 1.0",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_ps([line])

    def test_reject_operand_neg_abs_sext_modifiers(self):
        for op in OPS:
            for line in (
                # Modifiers on a[0] (dst)
                f"{op} -v[0:1], v4, v[2:3]",
                f"{op} |v[0:1]|, v4, v[2:3]",
                f"{op} sext(v[0:1]), v4, v[2:3]",
                # Modifiers on a[1] (shift amount)
                f"{op} v[0:1], -v4, v[2:3]",
                f"{op} v[0:1], |v4|, v[2:3]",
                f"{op} v[0:1], neg(4), v[2:3]",
                f"{op} v[0:1], abs(4), v[2:3]",
                f"{op} v[0:1], sext(v4), v[2:3]",
                # Modifiers on a[2] (64-bit source)
                f"{op} v[0:1], v4, -v[2:3]",
                f"{op} v[0:1], v4, |v[2:3]|",
                f"{op} v[0:1], v4, -s[2:3]",
                f"{op} v[0:1], v4, |s[2:3]|",
                f"{op} v[0:1], v4, neg(1)",
                f"{op} v[0:1], v4, abs(1)",
                f"{op} v[0:1], v4, sext(v[2:3])",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_ps([line])

    def test_reject_dpp_sdwa_and_instruction_mods(self):
        bad_mods = [
            "clamp",
            "mul:2",
            "mul:4",
            "div:2",
            "op_sel:[0,0,0]",
            "op_sel:[1,0,0]",
            "op_sel_hi:[1,1,1]",
            "neg_lo:[0,0,0]",
            "unknown_mod",
        ]
        for op in OPS:
            for line in (
                f"{op}_dpp v[0:1], v4, v[2:3] quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                f"{op}_sdwa v[0:1], v4, v[2:3] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        translate_ps([line])
            for suffix in ("", "_e64"):
                for mod in bad_mods:
                    line = f"{op}{suffix} v[0:1], v4, v[2:3] {mod}"
                    with self.subTest(op=op, suffix=suffix, mod=mod):
                        with self.assertRaises(rdna2spirv.Unsupported):
                            translate_ps([line])


if __name__ == "__main__":
    unittest.main()
