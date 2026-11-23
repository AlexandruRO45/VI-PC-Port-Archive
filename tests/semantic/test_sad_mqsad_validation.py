"""Validation and positive-control tests for the 8 RDNA2 SAD/MSAD/QSAD/MQSAD ops.

Covers `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_sad_u8`, `v_sad_hi_u8`,
`v_sad_u16`, `v_sad_u32`, `v_msad_u8`, and `v_mqsad_u32_u8`:
- Rejects non-VGPR destinations (including `s[0:1]` on 64-bit QSAD/MQSAD and
  `s[0:3]` on 128-bit `v_mqsad_u32_u8`, which previously miscompiled to VGPRs).
- Rejects wrong destination widths, wrong operand counts, wrong source widths,
  `neg`/`abs`/`sext` operand modifiers, `_dpp`/`_sdwa` encodings, and any
  non-empty instruction modifier (`clamp`, `mul:2`, `div:2`, `op_sel:...`, etc.).
- Verifies valid forms for all 8 ops with `spirv-as`, `spirv-val --target-env vulkan1.2`,
  and `spirv_mini_interp`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M16 = 0xFFFF
M32 = 0xFFFFFFFF

SAD32_OPS = ("v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8")
QSAD64_OPS = ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8")
ALL_OPS = SAD32_OPS + QSAD64_OPS + ("v_mqsad_u32_u8",)


def _validate_spv(spvasm: str) -> None:
    """Assemble and validate SPIR-V when spirv-as and spirv-val are on PATH."""
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        res_as = subprocess.run(
            [spirv_as, "--target-env", "spv1.5", asm_path, "-o", spv_path],
            capture_output=True,
            text=True,
        )
        if res_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{res_as.stderr}\n{res_as.stdout}")
        res_val = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", spv_path],
            capture_output=True,
            text=True,
        )
        if res_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{res_val.stderr}\n{res_val.stdout}")


def _translate(ins_line: str, setup: list[str] | None = None) -> str:
    lines = list(setup or ()) + [ins_line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.translate_lines("ps_sad_val", lines)


def _export_u32_tail(reg: int) -> list[str]:
    return [
        f"v_and_b32 v20, 0xffff, v{reg}",
        f"v_lshrrev_b32 v21, 16, v{reg}",
        "v_or_b32 v20, 0x40000000, v20",
        "v_or_b32 v21, 0x40000000, v21",
        "exp mrt0 v20, v21, v20, v20 done vm",
        "s_endpgm",
    ]


def _run_u32(lines: list[str], dst_reg: int = 0) -> int:
    spvasm = support.translate_lines("ps_sad_ok32", lines + _export_u32_tail(dst_reg))
    _validate_spv(spvasm)
    lo, hi = si.run_fragment(spvasm)["%out_mrt0"][:2]
    assert lo >> 16 == 0x4000 and hi >> 16 == 0x4000, (hex(lo), hex(hi))
    return (lo & M16) | ((hi & M16) << 16)


def _run_u64(lines: list[str], dst_lo: int = 0) -> int:
    tail = [
        f"v_and_b32 v20, 0xffff, v{dst_lo}",
        f"v_lshrrev_b32 v21, 16, v{dst_lo}",
        f"v_and_b32 v22, 0xffff, v{dst_lo + 1}",
        f"v_lshrrev_b32 v23, 16, v{dst_lo + 1}",
        "v_or_b32 v20, 0x40000000, v20",
        "v_or_b32 v21, 0x40000000, v21",
        "v_or_b32 v22, 0x40000000, v22",
        "v_or_b32 v23, 0x40000000, v23",
        "exp mrt0 v20, v21, v22, v23 done vm",
        "s_endpgm",
    ]
    spvasm = support.translate_lines("ps_sad_ok64", lines + tail)
    _validate_spv(spvasm)
    w = si.run_fragment(spvasm)["%out_mrt0"]
    assert all(v >> 16 == 0x4000 for v in w), [hex(v) for v in w]
    return sum((v & M16) << (16 * i) for i, v in enumerate(w))


def _run_u128(lines: list[str], dst_lo: int = 8) -> list[int]:
    tail = []
    for k in range(4):
        lo, hi = 20 + 2 * k, 21 + 2 * k
        tail += [
            f"v_and_b32 v{lo}, 0xffff, v{dst_lo + k}",
            f"v_lshrrev_b32 v{hi}, 16, v{dst_lo + k}",
            f"v_or_b32 v{lo}, 0x40000000, v{lo}",
            f"v_or_b32 v{hi}, 0x40000000, v{hi}",
        ]
    tail += [
        "exp mrt0 v20, v21, v22, v23 vm",
        "exp mrt1 v24, v25, v26, v27 done vm",
        "s_endpgm",
    ]
    spvasm = support.translate_lines("ps_sad_ok128", lines + tail)
    _validate_spv(spvasm)
    out = si.run_fragment(spvasm)
    h = list(out["%out_mrt0"]) + list(out["%out_mrt1"])
    assert all(v >> 16 == 0x4000 for v in h), [hex(v) for v in h]
    return [(h[2 * k] & M16) | ((h[2 * k + 1] & M16) << 16) for k in range(4)]


def _ref_sad32(op: str, x: int, y: int, acc: int) -> int:
    if op == "v_sad_u32":
        return (abs(x - y) + acc) & M32
    if op == "v_sad_u16":
        s = sum(abs(((x >> off) & M16) - ((y >> off) & M16)) for off in (0, 16))
        return (s + acc) & M32
    masked = op == "v_msad_u8"
    s = 0
    for off in (0, 8, 16, 24):
        xb, yb = (x >> off) & 0xFF, (y >> off) & 0xFF
        if masked and yb == 0:
            continue
        s += abs(xb - yb)
    if op == "v_sad_hi_u8":
        s <<= 16
    return (s + acc) & M32


def _ref_qsad64(op: str, x: int, y: int, acc: int) -> int:
    masked = op == "v_mqsad_pk_u16_u8"
    out = 0
    for k in range(4):
        win = (x >> (8 * k)) & M32
        s = 0
        for off in (0, 8, 16, 24):
            xb, yb = (win >> off) & 0xFF, (y >> off) & 0xFF
            if masked and yb == 0:
                continue
            s += abs(xb - yb)
        out |= ((((acc >> (16 * k)) & M16) + s) & M16) << (16 * k)
    return out


def _ref_mqsad128(x: int, y: int, accs: list[int]) -> list[int]:
    out = []
    for k in range(4):
        win = (x >> (8 * k)) & M32
        s = 0
        for off in (0, 8, 16, 24):
            xb, yb = (win >> off) & 0xFF, (y >> off) & 0xFF
            if yb == 0:
                continue
            s += abs(xb - yb)
        out.append(((accs[k] & M32) + s) & M32)
    return out


def _valid_args(op: str) -> tuple[str, str, str, str]:
    if op in SAD32_OPS:
        return ("v0", "v1", "v2", "v3")
    if op in QSAD64_OPS:
        return ("v[0:1]", "v[2:3]", "v4", "v[6:7]")
    return ("v[8:11]", "v[2:3]", "v1", "v[4:7]")


class TestSadMqsadValidationNegative(unittest.TestCase):
    """Negative validation tests across all 8 SAD/MSAD/QSAD/MQSAD ops."""

    def test_non_vgpr_dst_rejected(self) -> None:
        cases = []
        for op in SAD32_OPS:
            for dst in ("s0", "null", "exec_lo", "vcc_lo", "m0"):
                cases.append((op, f"{op} {dst}, v1, v2, v3"))
        for op in QSAD64_OPS:
            for dst in ("s[0:1]", "s0", "null", "exec", "vcc"):
                cases.append((op, f"{op} {dst}, v[2:3], v4, v[6:7]"))
        for dst in ("s[0:3]", "s[0:1]", "s0", "null"):
            cases.append(("v_mqsad_u32_u8", f"v_mqsad_u32_u8 {dst}, v[2:3], v1, v[4:7]"))
        for op, line in cases:
            with self.subTest(op=op, line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate(line)

    def test_wrong_dst_vgpr_width_rejected(self) -> None:
        cases = []
        for op in SAD32_OPS:
            for dst in ("v[0:1]", "v[0:3]"):
                cases.append((op, f"{op} {dst}, v1, v2, v3"))
        for op in QSAD64_OPS:
            for dst in ("v0", "v[0:2]", "v[0:3]"):
                cases.append((op, f"{op} {dst}, v[2:3], v4, v[6:7]"))
        for dst in ("v8", "v[8:9]", "v[8:10]"):
            cases.append(("v_mqsad_u32_u8", f"v_mqsad_u32_u8 {dst}, v[2:3], v1, v[4:7]"))
        for op, line in cases:
            with self.subTest(op=op, line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate(line)

    def test_wrong_operand_count_rejected(self) -> None:
        for op in ALL_OPS:
            d, s0, s1, s2 = _valid_args(op)
            for line in (
                f"{op} {d}, {s0}, {s1}",
                f"{op} {d}, {s0}, {s1}, {s2}, v12",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate(line)

    def test_wrong_source_widths_rejected(self) -> None:
        cases = []
        for op in SAD32_OPS:
            cases += [
                (op, f"{op} v0, v[1:2], v2, v3"),
                (op, f"{op} v0, s[1:2], v2, v3"),
                (op, f"{op} v0, v1, v[2:3], v3"),
                (op, f"{op} v0, v1, s[2:3], v3"),
                (op, f"{op} v0, v1, v2, v[3:4]"),
                (op, f"{op} v0, v1, v2, s[3:4]"),
            ]
        for op in QSAD64_OPS:
            cases += [
                (op, f"{op} v[0:1], v2, v4, v[6:7]"),
                (op, f"{op} v[0:1], s2, v4, v[6:7]"),
                (op, f"{op} v[0:1], v[2:5], v4, v[6:7]"),
                (op, f"{op} v[0:1], v[2:3], v[4:5], v[6:7]"),
                (op, f"{op} v[0:1], v[2:3], s[4:5], v[6:7]"),
                (op, f"{op} v[0:1], v[2:3], v4, v6"),
                (op, f"{op} v[0:1], v[2:3], v4, s6"),
                (op, f"{op} v[0:1], v[2:3], v4, v[6:9]"),
            ]
        cases += [
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v2, v1, v[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], s2, v1, v[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:5], v1, v[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v[1:2], v[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], s[1:2], v[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v4"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:5]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v1, s[4:7]"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v1, null"),
            ("v_mqsad_u32_u8", "v_mqsad_u32_u8 v[8:11], v[2:3], v1, 0"),
        ]
        for op, line in cases:
            with self.subTest(op=op, line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    _translate(line)

    def test_operand_modifiers_neg_abs_sext_rejected(self) -> None:
        for op in ALL_OPS:
            d, s0, s1, s2 = _valid_args(op)
            base = [d, s0, s1, s2]
            for idx in range(4):
                for wrap in (lambda x: f"-{x}", lambda x: f"|{x}|", lambda x: f"sext({x})"):
                    args = list(base)
                    args[idx] = wrap(args[idx])
                    line = f"{op} {', '.join(args)}"
                    with self.subTest(op=op, idx=idx, line=line):
                        with self.assertRaises(rdna2spirv.Unsupported):
                            _translate(line)

    def test_dpp_and_sdwa_encodings_rejected(self) -> None:
        for op in ALL_OPS:
            d, s0, s1, s2 = _valid_args(op)
            for line in (
                f"{op}_dpp {d}, {s0}, {s1}, {s2} row_shl:1",
                f"{op}_sdwa {d}, {s0}, {s1}, {s2} dst_sel:DWORD",
            ):
                with self.subTest(op=op, line=line):
                    with self.assertRaises(rdna2spirv.Unsupported):
                        _translate(line)

    def test_instruction_modifiers_rejected(self) -> None:
        bad_mods = ("clamp", "mul:2", "div:2", "op_sel:[0,0,0,0]", "unknown_mod")
        for op in ALL_OPS:
            d, s0, s1, s2 = _valid_args(op)
            for mod in bad_mods:
                line = f"{op} {d}, {s0}, {s1}, {s2} {mod}"
                with self.subTest(op=op, mod=mod):
                    with self.assertRaises(rdna2spirv.Unsupported) as cm:
                        _translate(line)
                    if op in SAD32_OPS and mod == "clamp":
                        self.assertIn(f"{op} clamp", str(cm.exception))


class TestSadMqsadPositiveControls(unittest.TestCase):
    """Positive controls verified by spirv-as, spirv-val (vulkan1.2), and spirv_mini_interp."""

    def test_sad32_ops_valid_forms(self) -> None:
        x, y, acc = 0x10203040, 0x00253035, 0x00000100
        for op in SAD32_OPS:
            want = _ref_sad32(op, x, y, acc)
            # VGPR sources
            with self.subTest(op=op, form="vgpr"):
                got = _run_u32([
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    f"v_mov_b32 v3, {acc:#x}",
                    f"{op} v0, v1, v2, v3",
                ])
                self.assertEqual(got, want)
            # Explicit _e64 + SGPR + immediate
            with self.subTest(op=op, form="e64_sgpr_imm"):
                got = _run_u32([
                    f"s_mov_b32 s1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    f"{op}_e64 v0, s1, v2, {acc:#x}",
                ])
                self.assertEqual(got, want)

    def test_qsad64_ops_valid_forms(self) -> None:
        x = 0x0807060504030201
        y = 0x04000201
        acc = 0x0004000300020001
        for op in QSAD64_OPS:
            want = _ref_qsad64(op, x, y, acc)
            with self.subTest(op=op, form="vgpr"):
                got = _run_u64([
                    f"v_mov_b32 v2, {x & M32:#x}",
                    f"v_mov_b32 v3, {x >> 32:#x}",
                    f"v_mov_b32 v4, {y:#x}",
                    f"v_mov_b32 v6, {acc & M32:#x}",
                    f"v_mov_b32 v7, {acc >> 32:#x}",
                    f"{op} v[0:1], v[2:3], v4, v[6:7]",
                ])
                self.assertEqual(got, want)
            with self.subTest(op=op, form="e64_sgpr_literal64"):
                got = _run_u64([
                    f"s_mov_b32 s2, {x & M32:#x}",
                    f"s_mov_b32 s3, {x >> 32:#x}",
                    f"v_mov_b32 v4, {y:#x}",
                    f"{op}_e64 v[0:1], s[2:3], v4, {acc:#x}",
                ])
                self.assertEqual(got, want)
            with self.subTest(op=op, form="literal64_src0_null_acc"):
                got = _run_u64([
                    f"v_mov_b32 v4, {y:#x}",
                    f"{op} v[0:1], {x:#x}, v4, null",
                ])
                self.assertEqual(got, _ref_qsad64(op, x, y, 0))

    def test_mqsad_u32_u8_valid_forms(self) -> None:
        x = 0x0807060504030201
        y = 0x04000201
        accs = [10, 20, 30, 0xFFFFFFFF]
        want = _ref_mqsad128(x, y, accs)
        acc_setup = [f"v_mov_b32 v{4 + k}, {accs[k]:#x}" for k in range(4)]

        with self.subTest(form="vgpr"):
            got = _run_u128(
                [f"v_mov_b32 v2, {x & M32:#x}", f"v_mov_b32 v3, {x >> 32:#x}", f"v_mov_b32 v1, {y:#x}"]
                + acc_setup
                + ["v_mqsad_u32_u8 v[8:11], v[2:3], v1, v[4:7]"]
            )
            self.assertEqual(got, want)

        with self.subTest(form="e64_sgpr_pair"):
            got = _run_u128(
                [f"s_mov_b32 s2, {x & M32:#x}", f"s_mov_b32 s3, {x >> 32:#x}", f"v_mov_b32 v1, {y:#x}"]
                + acc_setup
                + ["v_mqsad_u32_u8_e64 v[8:11], s[2:3], v1, v[4:7]"]
            )
            self.assertEqual(got, want)

        with self.subTest(form="literal64_src0"):
            got = _run_u128(
                [f"v_mov_b32 v1, {y:#x}"]
                + acc_setup
                + [f"v_mqsad_u32_u8 v[8:11], {x:#x}, v1, v[4:7]"]
            )
            self.assertEqual(got, want)


if __name__ == "__main__":
    unittest.main()
