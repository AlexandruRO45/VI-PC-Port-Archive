"""Bounds checks for 32-bit literal dwords on binary64 (f64_literal) sources.

RDNA2 encodes a floating-point literal operand as a single unsigned 32-bit trailing dword
placed in bits [63:32] of a binary64 source (bits [31:0] = 0). Values outside
[0, 0xFFFFFFFF] cannot be encoded as a single literal dword and must be refused with
Unsupported("<op> literal") rather than silently masked with & 0xFFFFFFFF.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

F64_SOURCE_WORDS_OPS = (
    ("v_add_f64", "v[4:5], {lit}, v[0:1]"),
    ("v_add_f64", "v[4:5], v[0:1], {lit}"),
    ("v_cvt_f32_f64", "v4, {lit}"),
    ("v_rcp_f64", "v[4:5], {lit}"),
    ("v_rsq_f64", "v[4:5], {lit}"),
    ("v_sqrt_f64", "v[4:5], {lit}"),
    ("v_frexp_mant_f64", "v[4:5], {lit}"),
    ("v_frexp_exp_i32_f64", "v4, {lit}"),
)

OUT_OF_RANGE_LITERALS = (
    "0x100000000",
    "0x0994940e82458cc8",
    "0xffffffffffffffff",
    "-17",
    "-0x1",
    "neg(0x100000000)",
)


def _translate_single(line: str) -> str:
    return support.translate_lines("ps_f64_lit_bounds", [
        "v_mov_b32 v0, 0",
        "v_mov_b32 v1, 0x3ff00000",
        "v_mov_b32 v4, 0",
        "v_mov_b32 v5, 0",
        line,
        "exp mrt0 v4, v5, v4, v5 done vm",
        "s_endpgm",
    ])


def _translate_ldexp_with_f64_literal(value: int, exp_val: int = 2) -> str:
    """Translate a pixel shader with v_ldexp_f64 carrying an f64_literal IR source."""
    orig_parse = support.rdna2spirv.parse_file

    def patched_parse(path):
        info = orig_parse(path)
        for ins in info["insns"]:
            if ins["op"] == "v_ldexp_f64":
                ins["args"][1] = {"k": "imm", "v": value, "f64_literal": True}
        return info

    with mock.patch.object(support.rdna2spirv, "parse_file", side_effect=patched_parse):
        return support.translate_lines("ps_ldexp_f64_lit_bounds", [
            f"v_mov_b32 v2, {exp_val}",
            "v_ldexp_f64 v[4:5], 0x3ff00000, v2",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ])


def _validate_spirv(spv_text: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        return
    with tempfile.TemporaryDirectory() as directory:
        asm = Path(directory) / "shader.spvasm"
        binary = asm.with_suffix(".spv")
        asm.write_text(spv_text, encoding="ascii")
        subprocess.run([spirv_as, "--target-env", "vulkan1.2", str(asm), "-o", str(binary)],
                       check=True, capture_output=True, text=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.2", str(binary)],
                       check=True, capture_output=True, text=True)


class F64LiteralBoundsTests(unittest.TestCase):
    def test_f64_source_words_rejects_out_of_range_literals(self):
        for op, template in F64_SOURCE_WORDS_OPS:
            for lit in OUT_OF_RANGE_LITERALS:
                line = f"{op} {template.format(lit=lit)}"
                with self.subTest(line=line):
                    with self.assertRaises(support.rdna2spirv.Unsupported) as ctx:
                        _translate_single(line)
                    self.assertEqual(str(ctx.exception), f"{op} literal")

    def test_ldexp_f64_rejects_out_of_range_f64_literal(self):
        for value in (-17, -1, 0x100000000, 0x0994940E82458CC8, 0xFFFFFFFFFFFFFFFF):
            with self.subTest(value=hex(value)):
                with self.assertRaises(support.rdna2spirv.Unsupported) as ctx:
                    _translate_ldexp_with_f64_literal(value)
                self.assertEqual(str(ctx.exception), "v_ldexp_f64 literal")

    def test_valid_32bit_boundary_literals_accepted(self):
        for lit, expected_hi in (("0x0", 0), ("0x1", 1), ("0x3ff80000", 0x3FF80000),
                                 ("0xffffffff", 0xFFFFFFFF), ("neg(0x3ff80000)", 0x3FF80000)):
            with self.subTest(lit=lit):
                spv = _translate_single(f"v_sqrt_f64_e64 v[4:5], {lit}")
                self.assertIn(f"OpConstant %uint {expected_hi}", spv)
                _validate_spirv(spv)

    def test_valid_inline_negative_integers_remain_accepted(self):
        for inline in ("-16", "-1", "0", "1", "64"):
            with self.subTest(inline=inline):
                spv = _translate_single(f"v_add_f64 v[4:5], {inline}, v[0:1]")
                self.assertIn("OpFAdd %double", spv)
                _validate_spirv(spv)

    def test_valid_frexp_and_ldexp_literals_execute_exact(self):
        # frexp(0x3ff80000_00000000 = 1.5) -> mant = 0.75 (0x3fe80000_00000000), exp = 1
        spv_frexp = support.translate_lines("ps_frexp_lit_ok", [
            "v_frexp_mant_f64 v[4:5], 0x3ff80000",
            "v_frexp_exp_i32_f64 v6, 0x3ff80000",
            "v_mov_b32 v7, 0",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ])
        _validate_spirv(spv_frexp)
        self.assertEqual(support.mrt0_words(spv_frexp), [0, 0x3FE80000, 1, 0])

        # ldexp(0x3ff00000_00000000 = 1.0, 2) -> 4.0 (0x40100000_00000000)
        spv_ldexp = _translate_ldexp_with_f64_literal(0x3FF00000, exp_val=2)
        _validate_spirv(spv_ldexp)
        self.assertEqual(support.mrt0_words(spv_ldexp)[:2], [0, 0x40100000])

        # ldexp boundary 0xFFFFFFFF (quiet NaN payload) passes through unchanged
        spv_ldexp_max = _translate_ldexp_with_f64_literal(0xFFFFFFFF, exp_val=2)
        _validate_spirv(spv_ldexp_max)
        self.assertEqual(support.mrt0_words(spv_ldexp_max)[:2], [0, 0xFFFFFFFF])
        self.assertRegex(spv_ldexp_max, r"OpConstant %uint 4294967295")


if __name__ == "__main__":
    unittest.main()
