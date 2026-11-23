"""Gen.fconst declares f32 constants once at module scope, with exact bits.

A finite normal value or +0 becomes one `OpConstant %float` written as repr() of its exact value; -0, subnormals, inf
and NaN keep the `OpBitcast %float %c<bits>` in the body. The decimal literal must give the same bits through both
assemblers the project uses: tools/spirv_asm.py, which reads decimal f32 via binary64 (#3911), and spirv-as, which
reads it straight to binary32 (checked when it is on PATH). Synthetic only.

    python -I tests/semantic/test_module_float_constants.py -v
"""
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from support import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402  (tools/, on the path through support)

Gen = rdna2spirv.Gen


def constant_words(binary):
    """The value word of every OpConstant in a module, in order."""
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    out, i = [], 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        if op == 43:
            out.append(words[i + 3])
        i += count
    return out


def module(bit_patterns):
    lines = ["OpCapability Shader", "OpMemoryModel Logical GLSL450", "%float = OpTypeFloat 32"]
    lines += [f"%k{i} = OpConstant %float {Gen.fconst_value(b)}" for i, b in enumerate(bit_patterns)]
    return "\n".join(lines) + "\n"


def sample_patterns():
    rng = random.Random(771)
    edges = [0x00000000, 0x00800000, 0x80800000, 0x00800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000,
             0x33800000, 0x3B808081, 0x4B800000, 0x3EAAAAAB, 0x3DCCCCCD]
    rand = []
    while len(rand) < 4000:
        b = rng.getrandbits(32)
        e = b & 0x7F800000
        if e and e != 0x7F800000:
            rand.append(b)
    return edges + rand


class ModuleFloatConstantTests(unittest.TestCase):
    def test_decimal_literal_gives_the_exact_bits_in_spirv_asm(self):
        patterns = sample_patterns()
        got = constant_words(spirv_asm.assemble(module(patterns), target_env="vulkan1.2"))
        self.assertEqual([hex(w) for w in got], [hex(b) for b in patterns])

    @unittest.skipUnless(shutil.which("spirv-as"), "spirv-as not on PATH")
    def test_decimal_literal_gives_the_exact_bits_in_spirv_as(self):
        patterns = sample_patterns()
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "k.spvasm"), os.path.join(d, "k.spv")
            with open(src, "w", encoding="utf-8", newline="\n") as f:
                f.write(module(patterns))
            subprocess.run(["spirv-as", "--target-env", "vulkan1.2", src, "-o", dst], check=True)
            with open(dst, "rb") as f:
                got = constant_words(f.read())
        self.assertEqual([hex(w) for w in got], [hex(b) for b in patterns])

    def test_which_values_are_hoisted(self):
        g = Gen.__new__(Gen)
        g.fconsts, g.fconsts_frozen, g.consts, g.body, g.n = {}, False, {}, [], 0
        g.emulate = False
        self.assertEqual(g.fconst(2.0 ** -126), "%cf_00800000")
        self.assertEqual(g.fconst(1.5), "%cf_3fc00000")
        # +0 and 1 are the %f0 and %f1 every module declares, and 0.5 is %fhalf_g: no second constant of the same value.
        self.assertEqual(g.fconst(0.0), "%f0")
        self.assertEqual(g.fconst(1.0), "%f1")
        g.uses_fhalf = False
        self.assertEqual(g.fconst(0.5), "%fhalf_g")
        self.assertTrue(g.uses_fhalf)
        self.assertEqual(sorted(g.fconsts), [0x00800000, 0x3FC00000])
        self.assertIs(g.fconst(2.0 ** -126), g.fconsts[0x00800000])      # one declaration per value
        for x in (-0.0, 2.0 ** -149, float("inf"), float("-inf"), float("nan")):
            with self.subTest(x=x):
                self.assertFalse(g.fconst(x).startswith("%cf_"))
        g.fconsts_frozen = True                                           # after the module's constants are written
        self.assertFalse(g.fconst(0.75).startswith("%cf_"))
        self.assertEqual(g.fconst(1.0), "%f1")                           # declared in every module, so still usable

    def test_a_translated_module_declares_flt_min_once(self):
        lines = ["v_mov_b32 v0, 0x00800001", "v_mov_b32 v1, 0x3f000000",
                 "v_mad_f32 v2, v0, v1, v0", "v_mad_f32 v3, v2, v1, v0",
                 "exp mrt0 v2, v3, v3, v3 done vm", "s_endpgm"]
        spv = support.translate_lines("ps_module_float_constants", lines)
        self.assertEqual(len(re.findall(r"^%cf_00800000 = OpConstant %float ", spv, re.M)), 1)
        self.assertNotIn("OpBitcast %float %c800000", spv)
        values = re.findall(r"^%\S+ = OpConstant %float (\S+)$", spv, re.M)
        self.assertEqual(len(values), len({float(v) for v in values}), "two float constants of the same value")
        support.mrt0_words(spv)     # the module still runs in the interpreter with the module-scope constants


if __name__ == "__main__":
    unittest.main()
