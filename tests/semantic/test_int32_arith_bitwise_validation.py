"""Validation and positive-control tests for 7 32-bit integer arithmetic/bitwise VALU ops.

Covers:
  - VOP3-only: v_add_nc_i32, v_sub_nc_i32
  - VOP2/VOP3/SDWA/DPP: v_add_nc_u32, v_sub_nc_u32, v_and_b32, v_or_b32, v_xor_b32

Verifies that malformed operands, destinations, encodings, and modifiers raise
rdna2spirv.Unsupported, while valid plain/_e32/_e64/clamp/_sdwa/_dpp forms and
plain negative integer immediates translate to valid SPIR-V (checked with
spirv-as + spirv-val --target-env vulkan1.2 when available) and compute exact
results in spirv_mini_interp / spirv_wave_interp.
"""
from __future__ import annotations

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
import rdna2spirv  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

M32 = 0xFFFFFFFF
VOP3_ONLY_OPS = ("v_add_nc_i32", "v_sub_nc_i32")
VOP2_ARITH_OPS = ("v_add_nc_u32", "v_sub_nc_u32")
VOP2_BIT_OPS = ("v_and_b32", "v_or_b32", "v_xor_b32")
VOP2_OPS = VOP2_ARITH_OPS + VOP2_BIT_OPS
CLAMP_OPS = VOP3_ONLY_OPS + VOP2_ARITH_OPS
ALL_OPS = VOP3_ONLY_OPS + VOP2_OPS

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _validate_spvasm(spvasm: str) -> None:
    """Assemble and validate SPIR-V with spirv-as + spirv-val --target-env vulkan1.2 if on PATH."""
    if not SPIRV_AS or not SPIRV_VAL:
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r_as = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
                              capture_output=True, text=True, check=False)
        if r_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r_as.stderr}\n{r_as.stdout}")
        r_val = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", spv_path],
                               capture_output=True, text=True, check=False)
        if r_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r_val.stderr}\n{r_val.stdout}")


def _ps_single_op(op_line: str, init_lines: list[str] | None = None) -> str:
    """Translate a pixel shader that initializes v1..v3/s0..s1, runs op_line -> v0, and exports v0 as two halves."""
    lines = list(init_lines) if init_lines is not None else [
        "v_mov_b32_e32 v0, 0x11223344",
        "v_mov_b32_e32 v1, 0x00000010",
        "v_mov_b32_e32 v2, 0x00000003",
        "s_mov_b32 s0, 0x00000007",
        "s_mov_b32 s1, 0x00000005",
    ]
    lines += [
        op_line,
        "v_and_b32_e32 v4, 0xffff, v0",
        "v_or_b32_e32 v4, 0x40000000, v4",
        "v_lshrrev_b32_e32 v5, 16, v0",
        "v_or_b32_e32 v5, 0x40000000, v5",
        "v_mov_b32_e32 v6, 0",
        "exp mrt0 v4, v5, v6, v6 done vm",
        "s_endpgm",
    ]
    return support.translate_lines("ps_int32_val", lines)


def _run_ps_u32(op_line: str, v1: int = 0x10, v2: int = 0x03, v0_init: int = 0x11223344, s0: int = 7) -> int:
    init = [
        f"v_mov_b32_e32 v0, {v0_init & M32:#x}",
        f"v_mov_b32_e32 v1, {v1 & M32:#x}",
        f"v_mov_b32_e32 v2, {v2 & M32:#x}",
        f"s_mov_b32 s0, {s0 & M32:#x}",
    ]
    spvasm = _ps_single_op(op_line, init)
    _validate_spvasm(spvasm)
    words = support.mrt0_words(spvasm)
    return (words[0] & 0xFFFF) | ((words[1] & 0xFFFF) << 16)


class TestInt32ArithBitwiseNegative(unittest.TestCase):
    """Negative validation tests across all 7 ops."""

    def _assert_rejected(self, line: str) -> None:
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {line}"):
            _ps_single_op(line)

    def test_wrong_operand_count(self) -> None:
        for op in ALL_OPS:
            with self.subTest(op=op, kind="too_few"):
                self._assert_rejected(f"{op} v0, v1")
            with self.subTest(op=op, kind="too_many"):
                self._assert_rejected(f"{op} v0, v1, v2, v3")

    def test_invalid_destination(self) -> None:
        bad_dsts = ("s0", "v[0:1]", "s[0:1]", "vcc_lo", "exec_lo", "-v0", "|v0|", "sext(v0)")
        for op in ALL_OPS:
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    self._assert_rejected(f"{op} {dst}, v1, v2")

    def test_64bit_source_operands(self) -> None:
        for op in ALL_OPS:
            for srcs in ("v[1:2], v2", "v1, v[2:3]", "s[0:1], v2", "v1, s[0:1]"):
                with self.subTest(op=op, srcs=srcs):
                    self._assert_rejected(f"{op} v0, {srcs}")

    def test_source_neg_abs_sext_on_non_sdwa(self) -> None:
        bad_srcs = (
            "-v1, v2",
            "v1, -v2",
            "|v1|, v2",
            "v1, abs(v2)",
            "neg(5), v2",
            "v1, neg(5)",
            "|5|, v2",
            "sext(v1), v2",
            "v1, sext(v2)",
        )
        for op in ALL_OPS:
            for srcs in bad_srcs:
                with self.subTest(op=op, srcs=srcs):
                    self._assert_rejected(f"{op}_e64 v0, {srcs}")
        for op in VOP2_OPS:
            for srcs in ("-v1, v2", "|v1|, v2", "sext(v1), v2"):
                with self.subTest(op=op, enc="_e32", srcs=srcs):
                    self._assert_rejected(f"{op}_e32 v0, {srcs}")

    def test_source_neg_abs_on_sdwa_and_dpp(self) -> None:
        sdwa_tail = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD"
        dpp_tail = "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        for op in VOP2_OPS:
            for srcs in ("-v1, v2", "v1, -v2", "|v1|, v2", "v1, |v2|", "-sext(v1), v2"):
                with self.subTest(op=op, enc="_sdwa", srcs=srcs):
                    self._assert_rejected(f"{op}_sdwa v0, {srcs} {sdwa_tail}")
            for srcs in ("-v1, v2", "v1, -v2", "|v1|, v2", "v1, |v2|", "sext(v1), v2", "v1, sext(v2)"):
                with self.subTest(op=op, enc="_dpp", srcs=srcs):
                    self._assert_rejected(f"{op}_dpp v0, {srcs} {dpp_tail}")

    def test_vop3_only_rejects_e32_dpp_sdwa(self) -> None:
        for op in VOP3_ONLY_OPS:
            with self.subTest(op=op, enc="_e32"):
                self._assert_rejected(f"{op}_e32 v0, v1, v2")
            with self.subTest(op=op, enc="_dpp"):
                self._assert_rejected(f"{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf")
            with self.subTest(op=op, enc="_sdwa"):
                self._assert_rejected(
                    f"{op}_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
                )

    def test_clamp_rejected_on_bitwise_and_invalid_encodings(self) -> None:
        sdwa_tail = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
        dpp_tail = "quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        for op in VOP2_BIT_OPS:
            for line in (
                f"{op} v0, v1, v2 clamp",
                f"{op}_e64 v0, v1, v2 clamp",
                f"{op}_e32 v0, v1, v2 clamp",
                f"{op}_sdwa v0, v1, v2 clamp {sdwa_tail}",
                f"{op}_dpp v0, v1, v2 clamp {dpp_tail}",
            ):
                with self.subTest(line=line):
                    self._assert_rejected(line)
        for op in VOP2_ARITH_OPS:
            for line in (
                f"{op}_e32 v0, v1, v2 clamp",
                f"{op}_dpp v0, v1, v2 clamp {dpp_tail}",
            ):
                with self.subTest(line=line):
                    self._assert_rejected(line)

    def test_duplicate_clamp_rejected(self) -> None:
        sdwa_tail = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
        for op in CLAMP_OPS:
            with self.subTest(op=op, enc="_e64"):
                self._assert_rejected(f"{op}_e64 v0, v1, v2 clamp clamp")
        for op in VOP2_ARITH_OPS:
            with self.subTest(op=op, enc="_sdwa"):
                self._assert_rejected(f"{op}_sdwa v0, v1, v2 clamp clamp {sdwa_tail}")

    def test_invalid_modifiers_rejected_on_all_ops(self) -> None:
        bad_mods = ("mul:2", "mul:4", "div:2", "op_sel:[0,0,0]", "glc", "unknown_mod")
        for op in ALL_OPS:
            for mod in bad_mods:
                with self.subTest(op=op, mod=mod):
                    self._assert_rejected(f"{op}_e64 v0, v1, v2 {mod}")
        for op in VOP2_OPS:
            for mod in ("mul:2", "div:2", "op_sel:[0,0,0]", "glc"):
                with self.subTest(op=op, enc="_e32", mod=mod):
                    self._assert_rejected(f"{op}_e32 v0, v1, v2 {mod}")
                with self.subTest(op=op, enc="_dpp", mod=mod):
                    self._assert_rejected(
                        f"{op}_dpp v0, v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf {mod}"
                    )
                with self.subTest(op=op, enc="_sdwa", mod=mod):
                    self._assert_rejected(
                        f"{op}_sdwa v0, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD {mod}"
                    )

    def test_invalid_sdwa_modifiers_rejected(self) -> None:
        bad_sdwa = (
            "dst_sel:BOGUS",
            "src0_sel:BOGUS",
            "src1_sel:BOGUS",
            "dst_unused:BOGUS",
            "dst_sel:BYTE_0 dst_sel:BYTE_1",
        )
        for op in VOP2_OPS:
            for mod in bad_sdwa:
                with self.subTest(op=op, mod=mod):
                    self._assert_rejected(f"{op}_sdwa v0, v1, v2 {mod}")


class TestInt32ArithBitwisePositive(unittest.TestCase):
    """Positive controls across all 7 ops, encodings, clamp, SDWA, DPP, and negative immediates."""

    def test_plain_e64_and_negative_immediates(self) -> None:
        cases = [
            ("v_add_nc_i32 v0, v1, v2", 0x7FFFFFF0, 0x20, (0x7FFFFFF0 + 0x20) & M32),
            ("v_add_nc_i32_e64 v0, v1, -5", 10, 0, 5),
            ("v_sub_nc_i32 v0, v1, v2", 5, 12, (-7) & M32),
            ("v_sub_nc_i32_e64 v0, v1, -5", 10, 0, 15),
            ("v_add_nc_u32 v0, v1, v2", 0xFFFFFFF0, 0x20, 0x10),
            ("v_add_nc_u32_e64 v0, v1, -5", 10, 0, 5),
            ("v_sub_nc_u32 v0, v1, v2", 5, 12, (-7) & M32),
            ("v_sub_nc_u32_e64 v0, v1, -5", 10, 0, 15),
            ("v_and_b32 v0, v1, v2", 0xF0F01234, 0x0FF0FFFF, 0x00F01234),
            ("v_and_b32_e64 v0, v1, -16", 0x1234567F, 0, 0x12345670),
            ("v_or_b32 v0, v1, v2", 0xF0001200, 0x0F000034, 0xFF001234),
            ("v_or_b32_e64 v0, v1, -1", 0x12345678, 0, 0xFFFFFFFF),
            ("v_xor_b32 v0, v1, v2", 0xAA551234, 0xFFFF00FF, 0x55AA12CB),
            ("v_xor_b32_e64 v0, v1, -1", 0x12345678, 0, (~0x12345678) & M32),
        ]
        for line, v1, v2, expected in cases:
            with self.subTest(line=line):
                got = _run_ps_u32(line, v1=v1, v2=v2)
                self.assertEqual(got, expected)

    def test_e32_vop2_forms(self) -> None:
        cases = [
            ("v_add_nc_u32_e32 v0, s0, v1", 100, 0, 7, 107),
            ("v_sub_nc_u32_e32 v0, s0, v1", 3, 0, 10, 7),
            ("v_and_b32_e32 v0, 0xff, v1", 0x12345678, 0, 0, 0x78),
            ("v_or_b32_e32 v0, s0, v1", 0x100, 0, 0x55, 0x155),
            ("v_xor_b32_e32 v0, -1, v1", 0x0F0F0F0F, 0, 0, 0xF0F0F0F0),
        ]
        for line, v1, v2, s0, expected in cases:
            with self.subTest(line=line):
                got = _run_ps_u32(line, v1=v1, v2=v2, s0=s0)
                self.assertEqual(got, expected)

    def test_clamp_forms(self) -> None:
        cases = [
            ("v_add_nc_i32_e64 v0, v1, v2 clamp", 0x7FFFFFF0, 0x20, 0x7FFFFFFF),
            ("v_add_nc_i32 v0, v1, v2 clamp", 0x80000010, 0xFFFFFFE0, 0x80000000),
            ("v_sub_nc_i32_e64 v0, v1, v2 clamp", 0x80000010, 0x20, 0x80000000),
            ("v_sub_nc_i32 v0, v1, v2 clamp", 0x7FFFFFF0, 0xFFFFFFE0, 0x7FFFFFFF),
            ("v_add_nc_u32_e64 v0, v1, v2 clamp", 0xFFFFFFF0, 0x20, 0xFFFFFFFF),
            ("v_sub_nc_u32_e64 v0, v1, v2 clamp", 5, 12, 0),
        ]
        for line, v1, v2, expected in cases:
            with self.subTest(line=line):
                got = _run_ps_u32(line, v1=v1, v2=v2)
                self.assertEqual(got, expected)

    def test_sdwa_forms(self) -> None:
        cases = [
            # sext(BYTE_0) = 0xFFFFFF80 (-128) + 0x00000001 = 0xFFFFFF81
            (
                "v_add_nc_u32_sdwa v0, sext(v1), v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD",
                0x00000080, 1, 0, 0xFFFFFF81,
            ),
            # clamped SDWA add: 0xFFFFFFFF + 1 -> 0xFFFFFFFF
            (
                "v_add_nc_u32_sdwa v0, v1, v2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                0xFFFFFFFF, 1, 0, 0xFFFFFFFF,
            ),
            # clamped SDWA sub: 1 - 2 -> 0
            (
                "v_sub_nc_u32_sdwa v0, v1, v2 clamp dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                1, 2, 0, 0,
            ),
            # SDWA dst_sel:BYTE_1 with UNUSED_PRESERVE
            (
                "v_and_b32_sdwa v0, v1, v2 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:BYTE_0",
                0x000000F5, 0x0000003F, 0xAA00CCDD, 0xAA0035DD,
            ),
            # SDWA sext + UNUSED_SEXT on v_or_b32
            (
                "v_or_b32_sdwa v0, sext(v1), v2 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:BYTE_0 src1_sel:BYTE_0",
                0x00000080, 0x00000005, 0, 0xFFFFFF85,
            ),
            # SDWA WORD_1 on v_xor_b32
            (
                "v_xor_b32_sdwa v0, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0",
                0x12340000, 0x000000FF, 0, 0x000012CB,
            ),
        ]
        for line, v1, v2, v0_init, expected in cases:
            with self.subTest(line=line):
                got = _run_ps_u32(line, v1=v1, v2=v2, v0_init=v0_init)
                self.assertEqual(got, expected)

    def test_dpp_forms(self) -> None:
        # Vertex shader running 32 lanes: v0 = lane id (0..31), v1 = 10, quad_perm:[1,0,3,2] swaps adjacent lanes
        dpp_cases = [
            ("v_add_nc_u32_dpp v6, v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", lambda s, b: (s + b) & M32),
            ("v_sub_nc_u32_dpp v6, v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", lambda s, b: (s - b) & M32),
            ("v_and_b32_dpp v6, v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", lambda s, b: s & b),
            ("v_or_b32_dpp v6, v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf", lambda s, b: s | b),
            ("v_xor_b32_dpp v6, v0, v1 dpp8:[1,0,3,2,5,4,7,6]", lambda s, b: s ^ b),
        ]
        for line, fn in dpp_cases:
            with self.subTest(line=line):
                spvasm = support.translate_lines("vs_int32_dpp", [
                    "v_mbcnt_lo_u32_b32 v0, -1, 0",
                    "v_mov_b32_e32 v1, 10",
                    "v_mov_b32_e32 v6, 0",
                    line,
                    "v_mov_b32_e32 v5, 0",
                    "exp pos0 v5, v5, v5, v5 done",
                    "exp param0 v6, v5, v5, v5",
                    "s_endpgm",
                ])
                _validate_spvasm(spvasm)
                res = wi.run_wave(
                    spvasm,
                    wave=32,
                    builtins={"VertexIndex": lambda lane: lane, "InstanceIndex": lambda lane: 0},
                )
                got = [o["%out_param0"][0] for o in res["outputs"]]
                expected = [fn(lane ^ 1, 10) for lane in range(32)]
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
