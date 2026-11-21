"""tools/shader_ir.py must ignore assembler directives instead of treating them as instructions.

LLVM 21's llvm-mc output starts at the first instruction. LLVM 18 (Ubuntu's /usr/lib/llvm-18) emits a
tab-prefixed `.text` first. That line has no `; encoding:` comment, so it reached rdna2spirv.py as an
instruction with enc=None and crashed every shard on `bytes.fromhex(None)`.

    python -I tests/test_shader_ir.py
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_ir  # noqa: E402


class DirectiveTest(unittest.TestCase):
    def test_text_directive_is_not_an_instruction(self):
        self.assertIsNone(shader_ir.parse_line("\t.text\n"))
        self.assertIsNone(shader_ir.parse_line("\t.amdgcn_next_shader\n"))

    def test_real_instruction_still_parses(self):
        ins = shader_ir.parse_line("\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
        self.assertIsNotNone(ins)
        self.assertEqual(ins["op"], "s_endpgm")
        self.assertEqual(ins["enc"], "000081bf")

    def test_parse_file_skips_a_directive(self):
        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False, encoding="utf-8") as f:
            f.write("; header\n\t.text\n\ts_endpgm ; encoding: [0x00,0x00,0x81,0xbf]\n")
            path = f.name
        try:
            info = shader_ir.parse_file(path)
            self.assertEqual(len(info["insns"]), 1)
            self.assertEqual(info["insns"][0]["op"], "s_endpgm")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
