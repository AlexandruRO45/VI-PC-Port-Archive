"""Check the SPIR-V data flow for output modifiers on v_cvt_f32_f64."""

import re
import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

from listing import translate_listing

FIXTURE = TEST_DIR / "fixtures" / "f64_conversion.asm"


class F64ConversionTest(unittest.TestCase):
    def test_output_modifiers(self):
        body = "\n".join(
            line for line in FIXTURE.read_text(encoding="utf-8").splitlines()
            if not line.startswith(";")
        )
        cases = (
            ("", None, False),
            ("mul:2", 0x40000000, False),
            ("mul:4", 0x40800000, False),
            ("div:2", 0x3F000000, False),
            ("clamp", None, True),
            ("clamp div:2", 0x3F000000, True),
        )
        with tempfile.TemporaryDirectory() as directory:
            for modifiers, scale_bits, clamp in cases:
                with self.subTest(modifiers=modifiers):
                    spv = translate_listing(Path(directory), f"{body} {modifiers}".rstrip())
                    self.check_conversion(spv, scale_bits, clamp)

    def check_conversion(self, spv, scale_bits, clamp):
        def result(instruction):
            match = re.search(rf"^ *(%\w+) = {instruction}$", spv, re.M)
            self.assertIsNotNone(match, f"Missing SPIR-V instruction: {instruction}")
            return match.group(1)

        value = result(r"OpFConvert %float %\w+")
        if scale_bits is not None:
            constant = result(rf"OpConstant %uint {scale_bits}")
            scale = result(rf"OpBitcast %float {constant}")
            value = result(rf"OpFMul %float {value} {scale}")
        if clamp:
            value = result(rf"OpExtInst %float %glsl FClamp {value} %f0 %f1")
        word = result(rf"OpBitcast %uint {value}")
        # A block with no inactive lanes stores the word directly (#4473); one whose EXEC can be partial
        # predicates the write on a select of the old word. Both are correct, and neither may write another
        # register or write this one from a different value.
        predicated = re.search(rf"(?m)^ *(%\w+) = OpSelect %uint %\w+ {word} (%\w+)$", spv)
        stored = word if predicated is None else predicated.group(1)
        written = re.findall(r"(?m)^ *OpStore %reg_v3 (\S+)$", spv)
        self.assertIn(stored, written, "nothing stores the converted word to v3")
        self.assertEqual([w for w in written if w != stored], [], "v3 is also written from another value")


if __name__ == "__main__":
    unittest.main()
