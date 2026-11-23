"""The assembler CLI must preserve its source when output aliases it. Synthetic inputs only."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


ASSEMBLER = Path(__file__).resolve().parents[1] / "tools" / "spirv_asm.py"
SOURCE = (
    "OpCapability Shader\n"
    "OpMemoryModel Logical GLSL450\n"
    'OpEntryPoint GLCompute %main "main"\n'
    "OpExecutionMode %main LocalSize 1 1 1\n"
    "%void = OpTypeVoid\n"
    "%fn = OpTypeFunction %void\n"
    "%main = OpFunction %void None %fn\n"
    "%entry = OpLabel\n"
    "OpReturn\n"
    "OpFunctionEnd\n"
).encode("utf-8")


class SpirvAsmPathTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "input.spvasm"
        self.source.write_bytes(SOURCE)

    def run_assembler(self, output):
        return subprocess.run(
            [sys.executable, "-I", str(ASSEMBLER), str(self.source), "-o", str(output)],
            capture_output=True, text=True, errors="replace", timeout=30,
        )

    def assert_refused(self, output):
        result = self.run_assembler(output)
        self.assertEqual(self.source.read_bytes(), SOURCE, "assembly source was overwritten")
        self.assertEqual(Path(output).read_bytes(), SOURCE, "source alias was overwritten")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("output would overwrite input", result.stderr)
        self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_same_path_preserves_source(self):
        self.assert_refused(self.source)

    def test_normalized_path_preserves_source(self):
        child = self.root / "child"
        child.mkdir()
        self.assert_refused(child / ".." / self.source.name)

    def test_hard_link_preserves_source(self):
        output = self.root / "output.spv"
        try:
            os.link(self.source, output)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"hard links unavailable: {exc}")
        self.assert_refused(output)

    def test_symbolic_link_preserves_source(self):
        output = self.root / "output.spv"
        try:
            output.symlink_to(self.source)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")
        self.assert_refused(output)

    def test_distinct_output_assembles_and_preserves_source(self):
        output = self.root / "output.spv"
        result = self.run_assembler(output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.source.read_bytes(), SOURCE)
        binary = output.read_bytes()
        self.assertEqual(len(binary) % 4, 0)
        self.assertEqual(struct.unpack_from("<I", binary)[0], 0x07230203)


if __name__ == "__main__":
    unittest.main()
