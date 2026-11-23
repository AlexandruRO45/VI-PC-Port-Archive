"""Quoted SPIR-V strings retain physical line endings and embedded source text."""
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from spirv_asm import SpirvAsmError, assemble  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

HEADER = """OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
"""
BODY = """%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""


def literal_strings(binary, opcode, leading_operands):
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    result = []
    at = 5
    while at < len(words):
        count = words[at] >> 16
        if words[at] & 0xffff == opcode:
            payload = words[at + 1 + leading_operands:at + count]
            raw = struct.pack(f"<{len(payload)}I", *payload)
            result.append(raw.partition(b"\0")[0].decode("utf-8"))
        at += count
    return result


class MultilineStringTests(unittest.TestCase):
    def checked_assembly(self, source, *, sdk_parity=True):
        binary = assemble(source)
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory, "input.spvasm")
            ours = Path(directory, "python.spv")
            reference = Path(directory, "sdk.spv")
            src.write_bytes(source.encode("utf-8"))
            ours.write_bytes(binary)
            if SPIRV_AS and sdk_parity:
                result = subprocess.run(
                    [SPIRV_AS, "--target-env", "vulkan1.2", str(src), "-o", str(reference)],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = reference.read_bytes()
                # The tool-specific generator word is the only intentional difference.
                self.assertEqual(binary[:8] + binary[12:], expected[:8] + expected[12:])
            if SPIRV_VAL:
                for path in (ours, reference) if SPIRV_AS and sdk_parity else (ours,):
                    result = subprocess.run(
                        [SPIRV_VAL, "--target-env", "vulkan1.2", str(path)],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
        return binary

    def test_in_string_line_characters_are_preserved(self):
        for separator in ("\n", "\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\u2029"):
            with self.subTest(separator=repr(separator)):
                value = "first" + separator + "last"
                # Windows spirv-as uses text-mode file input, folding CRLF before parsing.
                # The library API preserves its raw input; validate those literal bytes here.
                parity = not (sys.platform == "win32" and separator == "\r\n")
                binary = self.checked_assembly(HEADER + f'%text = OpString "{value}"\n' + BODY,
                                               sdk_parity=parity)
                self.assertEqual(literal_strings(binary, 7, 1), [value])
                if separator == "\r\n":
                    self.assertIn(b"first\r\nlast\0", binary)

    def test_debug_source_and_continued_source(self):
        source = "#version 450\n\nlayout(local_size_x = 1) in;\nvoid main() { }\n"
        continuation = "// retained source\n// second line\n"
        assembly = (HEADER + '%file = OpString "shader.comp"\n'
                    + f'OpSource GLSL 450 %file "{source}"\n'
                    + f'OpSourceContinued "{continuation}"\n' + BODY)
        binary = self.checked_assembly(assembly)
        self.assertEqual(literal_strings(binary, 3, 3), [source])
        self.assertEqual(literal_strings(binary, 2, 0), [continuation])

    def test_multiline_entry_name_keeps_following_interface_ids(self):
        header = HEADER.replace('"main"', '"first\nsecond" %value')
        declarations = """OpDecorate %value Location 0
%float = OpTypeFloat 32
%pointer = OpTypePointer Input %float
%value = OpVariable %pointer Input
"""
        # A vertex input supplies a legal trailing interface operand.
        header = header.replace("GLCompute", "Vertex").replace(
            "OpExecutionMode %main LocalSize 1 1 1\n", "")
        binary = self.checked_assembly(header + declarations + BODY)
        self.assertEqual(literal_strings(binary, 15, 2), ["first\nsecond"])

    def test_comments_escaped_quotes_and_backslashes(self):
        # Comments can contain unmatched quotes; in-string semicolons are literal data.
        assembly = (HEADER + '; " ignored comment\n'
                    + '%text = OpString "first;\n\\"quoted\\" \\\\ path\nlast" ; " ignored\n'
                    + 'OpName %main "left\\\nright"\n' + BODY)
        binary = self.checked_assembly(assembly)
        self.assertEqual(literal_strings(binary, 7, 1), ['first;\n"quoted" \\ path\nlast'])
        self.assertEqual(literal_strings(binary, 5, 1), ["left\nright"])

    def test_physical_line_numbers_after_multiline_strings(self):
        for newline in ("\n", "\r\n"):
            with self.subTest(newline=repr(newline)):
                source = '%text = OpString "first\nsecond\nlast"\n; " ignored\nOpDoesNotExist\n'
                source = source.replace("\n", newline)
                with self.assertRaisesRegex(SpirvAsmError, r"line 5: unknown opcode 'OpDoesNotExist'"):
                    assemble(source)

    def test_unterminated_strings_keep_the_starting_line(self):
        for tail in ('"first\nsecond', '"first\n\\"still quoted\\"', '"first\nlast\\'):
            with self.subTest(tail=tail):
                with self.assertRaisesRegex(SpirvAsmError, r"line 3: unterminated string literal"):
                    assemble('; comment with a "\n\n%text = OpString ' + tail)


if __name__ == "__main__":
    unittest.main()
