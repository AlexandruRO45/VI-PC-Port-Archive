"""v_cvt_f32_f64 (RDNA2 VOP1 / VOP3): D.f = (float)S0.d, the binary64 in a register pair narrowed to binary32; and
v_cvt_f64_f32, which the translator has no lowering for and refuses.

Every case is translated by tools/rdna2spirv.py. tests/semantic/spirv_mini_interp.py does not execute OpFConvert
yet, so the translated SPIR-V is checked along its SSA chain: the one OpFConvert %float reads a %double bitcast from
the pair loaded from the named registers (low word first), a literal dword lands in bits 63:32 of the operand
(docs/design.md "f64 literal sources"), and the VOP3 modifiers reach the double as FAbs then FNegate. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402


def translate(lines):
    return support.translate_lines("ps_cvt_f32_f64", list(lines) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


def doubles(spv):
    """{%id: 64-bit value} of every %double built from two constant words (the literal operands)."""
    words = {name: int(value) for name, value in re.findall(r"(%\w+) = OpConstant %uint (\d+)", spv)}
    pairs = {name: (words[high] << 32) | words[low]
             for name, low, high in re.findall(r"(%\w+) = OpCompositeConstruct %v2uint_d (%\w+) (%\w+)", spv)
             if low in words and high in words}
    return {name: pairs[pair] for name, pair in re.findall(r"(%\w+) = OpBitcast %double (%\w+)", spv) if pair in pairs}


def defs(spv):
    """{%id: (opcode, [operands])} for every SSA definition."""
    return {name: (op, rest.split()) for name, op, rest in re.findall(r"^\s*(%\w+) = (Op\w+) (.*)$", spv, re.M)}


def chain_to_convert(spv):
    """Follow the one OpFConvert %float back to its source: the (opcode, operands) steps from the convert down to the
    OpCompositeConstruct of the dword pair, and where the pair's two words were loaded from ("%vgpr %c2")."""
    d = defs(spv)
    converts = [name for name, (op, args) in d.items() if op == "OpFConvert" and args[0] == "%float"]
    assert len(converts) == 1, converts
    steps, cur = [], d[converts[0]][1][1]
    while True:
        op, args = d[cur]
        steps.append((op, args))
        if op == "OpCompositeConstruct":
            break
        cur = args[-1]                       # OpFNegate %double %x / OpExtInst %double %glsl FAbs %x / OpBitcast %double %x
    words = []
    for word in steps[-1][1][1:]:            # %v2uint_d %lo %hi
        op, args = d[word]
        words.append(" ".join(d[args[1]][1][1:]) if op == "OpLoad" else word)   # the register behind an OpLoad
    return steps, words


class CvtF32F64(unittest.TestCase):
    def test_register_pair_is_narrowed_with_fconvert(self):
        # The convert reads a double bitcast from the pair, v2 low and v3 high: a swapped pair fails here.
        for form in ("v_cvt_f32_f64", "v_cvt_f32_f64_e32", "v_cvt_f32_f64_e64"):
            with self.subTest(form):
                spv = translate(["v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40080000", f"{form} v0, v[2:3]"])
                steps, words = chain_to_convert(spv)
                self.assertEqual([op for op, _ in steps], ["OpBitcast", "OpCompositeConstruct"])
                self.assertEqual(words, ["%vgpr %c2", "%vgpr %c3"])

    def test_literal_and_inline_sources(self):
        # 1.5 is the literal word 0x3ff80000 in bits 63:32; inline 64 is the integer 64; inline -1 sign-extends.
        for src, want in (("0x3ff80000", 0x3FF8000000000000), ("64", 64), ("-1", 0xFFFFFFFFFFFFFFFF), ("1.0", 0x3FF0000000000000)):
            with self.subTest(src):
                spv = translate([f"v_cvt_f32_f64 v0, {src}"])
                steps, words = chain_to_convert(spv)
                self.assertEqual([op for op, _ in steps], ["OpBitcast", "OpCompositeConstruct"])
                self.assertEqual(set(doubles(spv).values()), {want})      # the one double built from constant words

    def test_modifiers_apply_to_the_double_abs_then_neg(self):
        # -|x|: the convert reads FNegate of FAbs of the bitcast double; |-x| (neg first) or either alone fails here.
        spv = translate(["v_mov_b32 v2, 0", "v_mov_b32 v3, 0xc0080000", "v_cvt_f32_f64_e64 v0, -|v[2:3]|"])
        steps, words = chain_to_convert(spv)
        self.assertEqual([op for op, _ in steps], ["OpFNegate", "OpExtInst", "OpBitcast", "OpCompositeConstruct"])
        self.assertEqual(steps[1][1][2], "FAbs")
        self.assertEqual(words, ["%vgpr %c2", "%vgpr %c3"])
        for mods, ops in (("|v[2:3]|", ["OpExtInst", "OpBitcast", "OpCompositeConstruct"]),
                          ("-v[2:3]", ["OpFNegate", "OpBitcast", "OpCompositeConstruct"])):
            with self.subTest(mods):
                steps, _ = chain_to_convert(translate(["v_mov_b32 v2, 0", "v_mov_b32 v3, 0xc0080000", f"v_cvt_f32_f64_e64 v0, {mods}"]))
                self.assertEqual([op for op, _ in steps], ops)

    def test_sgpr_pair_source(self):
        spv = translate(["s_mov_b32 s2, 0", "s_mov_b32 s3, 0x40080000", "v_cvt_f32_f64 v0, s[2:3]"])
        steps, words = chain_to_convert(spv)
        self.assertEqual([op for op, _ in steps], ["OpBitcast", "OpCompositeConstruct"])
        self.assertEqual(words, ["%sgpr %c2", "%sgpr %c3"])

    def test_widening_is_refused_not_guessed(self):
        with self.assertRaises(Unsupported) as cm:
            translate(["v_mov_b32 v2, 0x40400000", "v_cvt_f64_f32 v[0:1], v2"])
        self.assertIn("v_cvt_f64_f32", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
