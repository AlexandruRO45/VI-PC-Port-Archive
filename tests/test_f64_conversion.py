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

from f64_store_shape import PREFIXES, assert_stored_to_vgpr  # noqa: E402


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
            for prefix_name, prefix in PREFIXES:
                for modifiers, scale_bits, clamp in cases:
                    with self.subTest(exec=prefix_name, modifiers=modifiers):
                        spv = translate_listing(Path(directory), f"{prefix}{body} {modifiers}".rstrip())
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
        assert_stored_to_vgpr(self, spv, word, 3)


if __name__ == "__main__":
    unittest.main()
