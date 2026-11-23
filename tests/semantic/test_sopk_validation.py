"""Validation and positive-control tests for RDNA2 SOPK 16-bit immediate SALU ops.

Covers all 16 SOPK ALU/compare instructions in tools/rdna2spirv.py:
  - s_movk_i32, s_cmovk_i32, s_addk_i32, s_mulk_i32
  - s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32}

Negative tests verify that malformed forms raise rdna2spirv.Unsupported (never
KeyError/IndexError or silent truncation). Positive controls verify valid
16-bit immediate boundaries (-32768..65535), valid 32-bit scalar operands
(sgpr, m0, vcc_lo, vcc_hi, exec_lo, exec_hi, null), and SCC semantics with
spirv-as + spirv-val --target-env vulkan1.2 + spirv_mini_interp.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

ALL_CMPK_OPS = [
    f"s_cmpk_{cond}_{ty}"
    for cond in ("eq", "lg", "gt", "ge", "lt", "le")
    for ty in ("i32", "u32")
]
ALL_SOPK_OPS = ["s_movk_i32", "s_cmovk_i32", "s_addk_i32", "s_mulk_i32"] + ALL_CMPK_OPS


def _translate_one(name: str, inst_line: str) -> str:
    return support.translate_lines(name, [
        "s_mov_b32 s0, 1",
        "s_mov_b32 s1, 2",
        "v_mov_b32 v0, 0",
        inst_line,
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])


def _validate_spirv(spvasm: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        raise AssertionError("spirv-as and spirv-val must be on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "shader.spvasm")
        spv_path = os.path.join(tmp, "shader.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        r_as = subprocess.run(
            [spirv_as, "--target-env", "vulkan1.2", asm_path, "-o", spv_path],
            capture_output=True, text=True, check=False,
        )
        if r_as.returncode != 0:
            raise AssertionError(f"spirv-as failed:\n{r_as.stderr}\n{r_as.stdout}")
        r_val = subprocess.run(
            [spirv_val, "--target-env", "vulkan1.2", spv_path],
            capture_output=True, text=True, check=False,
        )
        if r_val.returncode != 0:
            raise AssertionError(f"spirv-val failed:\n{r_val.stderr}\n{r_val.stdout}")


class TestSopkNegativeValidation(unittest.TestCase):
    def _assert_unsupported(self, line: str) -> None:
        with self.assertRaises(rdna2spirv.Unsupported, msg=f"expected Unsupported for: {line}"):
            _translate_one("ps_sopk_neg", line)

    def test_rejects_non_immediate_second_operand(self):
        bad_srcs = ["s1", "v1", "m0", "null", "vcc_lo", "1.0", "[s1,s2]"]
        for op in ALL_SOPK_OPS:
            for src in bad_srcs:
                with self.subTest(op=op, src=src):
                    self._assert_unsupported(f"{op} s0, {src}")

    def test_rejects_out_of_range_immediates(self):
        bad_imms = ["0x10000", "65536", "-32769", "-65536", "0x100000000"]
        for op in ALL_SOPK_OPS:
            for imm in bad_imms:
                with self.subTest(op=op, imm=imm):
                    self._assert_unsupported(f"{op} s0, {imm}")

    def test_rejects_invalid_first_operand(self):
        bad_dsts = [
            "v0", "v[0:1]", "s[0:1]", "vcc", "exec",
            "scc", "src_scc", "1", "1.0", "[s0,s1]", "[v0,v1]",
        ]
        for op in ALL_SOPK_OPS:
            for dst in bad_dsts:
                with self.subTest(op=op, dst=dst):
                    self._assert_unsupported(f"{op} {dst}, 1")

    def test_rejects_wrong_arity(self):
        for op in ALL_SOPK_OPS:
            for args in ("", "s0", "s0, 1, 2"):
                line = f"{op} {args}".strip()
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_rejects_encoding_suffixes(self):
        for op in ALL_SOPK_OPS:
            cases = [
                f"{op}_e32 s0, 1",
                f"{op}_e64 s0, 1",
                f"{op}_sdwa s0, 1 dst_sel:DWORD",
                f"{op}_dpp s0, 1 row_shl:1",
                f"{op}_dpp v0, 1 row_shl:1",
            ]
            for line in cases:
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_rejects_instruction_modifiers(self):
        for op in ALL_SOPK_OPS:
            for mod in ("clamp", "mul:2", "op_sel:[0,0]"):
                line = f"{op} s0, 1 {mod}"
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_rejects_operand_modifiers(self):
        for op in ALL_SOPK_OPS:
            cases = [
                f"{op} -s0, 1",
                f"{op} |s0|, 1",
                f"{op} sext(s0), 1",
                f"{op} s0, neg(1)",
                f"{op} s0, abs(1)",
                f"{op} s0, |1|",
                f"{op} s0, sext(1)",
            ]
            for line in cases:
                with self.subTest(line=line):
                    self._assert_unsupported(line)

    def test_rejects_invalid_cmpk_suffixes(self):
        bad_ops = [
            "s_cmpk_ne_i32",
            "s_cmpk_ne_u32",
            "s_cmpk_eq_f32",
            "s_cmpk_eq_u64",
            "s_cmpk_foo_i32",
            "s_cmpk_bad",
            "s_cmpk_eq_i32_extra",
        ]
        for op in bad_ops:
            with self.subTest(op=op):
                self._assert_unsupported(f"{op} s0, 1")


class TestSopkPositiveControls(unittest.TestCase):
    def test_movk_cmovk_addk_mulk_edges_and_scc(self):
        spv = support.translate_lines("ps_sopk_pos_alu", [
            # s_movk_i32: -32768 -> 0xFFFF8000, SCC unchanged (pre-set to 1).
            "s_cmp_eq_u32 1, 1",
            "s_movk_i32 s0, -32768",
            "s_cselect_b32 s1, 1, 0",
            # s_addk_i32: 0x7FFFFFFF + 1 -> 0x80000000, signed overflow sets SCC = 1.
            "s_cmp_eq_u32 0, 1",
            "s_mov_b32 s2, 0x7fffffff",
            "s_addk_i32 s2, 1",
            "s_cselect_b32 s3, 1, 0",
            # s_mulk_i32: 3 * 0xffff (-1) -> 0xFFFFFFFD, preserves SCC = 1.
            "s_mov_b32 s4, 3",
            "s_mulk_i32 s4, 0xffff",
            "s_cselect_b32 s5, 1, 0",
            # s_cmovk_i32 with SCC = 1 -> writes signext(0xffff) = 0xFFFFFFFF; SCC = 0 keeps old.
            "s_mov_b32 s6, 0x1234",
            "s_cmovk_i32 s6, 0xffff",
            "s_cmp_eq_u32 0, 1",
            "s_cmovk_i32 s6, 0x7fff",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s2",
            "v_mov_b32 v2, s4",
            "v_mov_b32 v3, s6",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv)
        self.assertEqual(
            support.mrt0_words(spv),
            [0xFFFF8000, 0x80000000, 0xFFFFFFFD, 0xFFFFFFFF],
        )

    def test_special_32bit_scalar_operands(self):
        spv = support.translate_lines("ps_sopk_pos_special", [
            # m0, vcc_lo, vcc_hi, null with movk/cmovk/addk/mulk
            "s_movk_i32 m0, -32768",
            "s_addk_i32 m0, 0x7fff",        # -32768 + 32767 = -1 (0xFFFFFFFF), SCC = 0
            "s_movk_i32 vcc_lo, 0x10",
            "s_mulk_i32 vcc_lo, -2",        # 16 * -2 = -32 (0xFFFFFFE0)
            "s_cmp_eq_u32 1, 1",
            "s_mov_b32 vcc_hi, 0x55",
            "s_cmovk_i32 vcc_hi, 0xffff",   # SCC=1 -> 0xFFFFFFFF
            "s_movk_i32 null, 0xffff",
            "s_cmovk_i32 null, -1",
            "s_addk_i32 null, 1",           # 0 + 1, SCC = 0
            "s_mulk_i32 null, 2",
            # s_cmpk_* with m0 and null
            "s_cmpk_eq_i32 m0, -1",         # -1 == -1 -> SCC = 1
            "s_cselect_b32 s0, 1, 0",
            "s_cmpk_eq_u32 null, 0",        # 0 == 0 -> SCC = 1
            "s_cselect_b32 s1, 1, 0",
            "s_add_u32 s0, s0, s1",         # 2
            "v_mov_b32 v0, m0",
            "v_mov_b32 v1, vcc_lo",
            "v_mov_b32 v2, vcc_hi",
            "v_mov_b32 v3, s0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv)
        self.assertEqual(
            support.mrt0_words(spv),
            [0xFFFFFFFF, 0xFFFFFFE0, 0xFFFFFFFF, 2],
        )

    def test_cmpk_signed_and_unsigned_boundaries(self):
        spv = support.translate_lines("ps_sopk_pos_cmpk", [
            # -32768 == -32768 (i32) -> 1
            "s_mov_b32 s0, 0xffff8000",
            "s_cmpk_eq_i32 s0, -32768",
            "s_cselect_b32 s1, 1, 0",
            # 0xFFFF == 65535 (u32) -> 1
            "s_mov_b32 s0, 0x0000ffff",
            "s_cmpk_eq_u32 s0, 0xffff",
            "s_cselect_b32 s2, 1, 0",
            # 0xFFFFFFFF == 0xffff (i32: -1 == -1) -> 1, (u32: 0xFFFFFFFF == 65535) -> 0
            "s_mov_b32 s0, 0xffffffff",
            "s_cmpk_eq_i32 s0, 0xffff",
            "s_cselect_b32 s3, 1, 0",
            "s_cmpk_eq_u32 s0, 0xffff",
            "s_cselect_b32 s4, 1, 0",
            "v_mov_b32 v0, s1",
            "v_mov_b32 v1, s2",
            "v_mov_b32 v2, s3",
            "v_mov_b32 v3, s4",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv)
        self.assertEqual(support.mrt0_words(spv), [1, 1, 1, 0])

    def test_exec_lo_valid_sopk_operand(self):
        spv_write = support.translate_lines("ps_sopk_pos_exec_write", [
            "s_cmp_eq_u32 1, 1",
            "s_movk_i32 exec_lo, -1",
            "s_cmovk_i32 exec_lo, 0xffff",
            "v_mov_b32 v0, 42",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv_write)
        self.assertEqual(support.mrt0_words(spv_write)[0], 42)

        spv_cmp = support.translate_lines("ps_sopk_pos_exec_cmp", [
            "s_cmpk_lg_u32 exec_lo, 0",
            "s_cselect_b32 s0, 1, 0",
            "v_mov_b32 v0, s0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv_cmp)


if __name__ == "__main__":
    unittest.main()
