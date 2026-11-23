"""LLVM's lit(...) operand on top of #6808, which reads lit(V) as the constant V.

tools/shader_ir.py marks that imm "literal": it is the literal dword even when V is in the inline-constant range,
so a 64-bit float source reads it as bits 63:32 with zero low bits, as the bytes LLVM's assembler writes for it.
The f64 readers that decide by value or by a 0x spelling otherwise took lit(1) (and, on v_cmp_*_f64 and
v_ldexp_f64, lit(0x1) too) as the inline integer 1 in bits 31:0. Spellings LLVM's gfx10 printer never writes and
whose meaning depends on the operand type (lit(-1), lit(1.0)), a value past one dword (which llvm-mc rejects) and
lit64() stay "special", which the translator refuses.

No llvm-mc, SPIRV-Tools, GPU or game data is needed here.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from support import TOOLS_DIR, mrt0_words, translate_lines  # noqa: E402

sys.path.insert(0, TOOLS_DIR)
import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
EXPORT = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]


def fixture_lines(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return [line for line in f.read().splitlines() if line and not line.startswith(";")]


def f64_words(spv):
    """The (lo, hi) constant word pairs the module builds a double from."""
    words = {name: int(value) for name, value in re.findall(r"(%\w+) = OpConstant %uint (\d+)", spv)}
    return [(words[lo], words[hi]) for lo, hi in
            re.findall(r"= OpCompositeConstruct %v2uint_d (%c\w+) (%c\w+)", spv)]


class LitOperandParse(unittest.TestCase):
    def test_lit_is_a_marked_literal(self):
        self.assertEqual(shader_ir.operand("lit(0x40100000)"), {"k": "imm", "v": 0x40100000, "literal": True})
        self.assertEqual(shader_ir.operand("lit(0x0)"), {"k": "imm", "v": 0, "literal": True})
        self.assertEqual(shader_ir.operand("lit(1)"), {"k": "imm", "v": 1, "literal": True})
        self.assertEqual(shader_ir.operand("lit(0xffffffff)"), {"k": "imm", "v": 0xFFFFFFFF, "literal": True})

    def test_modifiers_keep_the_mark(self):
        self.assertEqual(shader_ir.operand("-lit(0x3f800000)"),
                         {"k": "imm", "v": 0x3F800000, "literal": True, "neg": True})
        self.assertEqual(shader_ir.operand("|lit(0xbf800000)|"),
                         {"k": "imm", "v": 0xBF800000, "literal": True, "abs": True})
        self.assertEqual(shader_ir.operand("-|lit(0x40000000)|"),
                         {"k": "imm", "v": 0x40000000, "literal": True, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("neg(lit(0x1))"), {"k": "imm", "v": 1, "literal": True, "neg": True})

    def test_decimal_lit_on_an_f64_source_is_a_literal_word(self):
        ins = shader_ir.parse_line("\tv_add_f64 v[6:7], lit(1), -lit(64)")
        self.assertTrue(all(src.get("f64_literal") for src in ins["args"][1:]))

    def test_what_is_not_a_gfx10_dword_stays_unparsed(self):
        for text in ("lit64(0x40100000)", "lit(0x100000000)", "lit(1.0)", "lit(-1)", "-lit(2.5e1)"):
            with self.subTest(text=text):
                self.assertEqual(shader_ir.operand(text)["k"], "special")


class LitTranslation(unittest.TestCase):
    def test_inline_range_literal_on_f64_compare_and_ldexp(self):
        spv = translate_lines("ps_lit_f64_inline_range", fixture_lines("lit_f64_inline_range.asm") + EXPORT)
        self.assertEqual(mrt0_words(spv), [0x3F800000, 0x41F00000, 0, 0])

    def test_inline_range_literal_on_f64_arithmetic_is_the_high_word(self):
        # Float64 arithmetic doesn't run in the interpreter: check the words the source is built from.
        # llvm-mc 23.1.3 gfx1030 assembles these to the literal dword 1 / 0x40 after the instruction.
        for line, word in (("v_add_f64 v[0:1], lit(1), v[2:3]", 1), ("v_cvt_f32_f64_e32 v0, lit(64)", 0x40),
                           ("v_rcp_f64_e32 v[0:1], lit(2)", 2)):
            with self.subTest(line=line):
                spv = translate_lines("ps_lit_f64_word", ["v_mov_b32 v2, 0", "v_mov_b32 v3, 0", line] + EXPORT)
                self.assertIn((0, word), f64_words(spv))
                self.assertNotIn((word, 0), f64_words(spv))

    def test_unparsed_spellings_are_refused(self):
        for line in ("v_add_f32_e32 v0, lit(1.0), v1", "v_add_f32_e32 v0, lit(-1), v1",
                     "v_add_f32_e32 v0, lit(0x100000000), v1", "v_add_f32_e32 v0, lit64(0x1), v1"):
            with self.subTest(line=line), self.assertRaises(rdna2spirv.Unsupported):
                translate_lines("ps_lit_refused", ["v_mov_b32 v1, 0", line, "v_mov_b32 v1, 0",
                                                   "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"] + EXPORT)


if __name__ == "__main__":
    unittest.main()
