"""Check the handoff from the shader builder to the benchmark without a GPU."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_wave64  # noqa: E402


class BuildArtifacts(unittest.TestCase):
    def test_skipped_shader_keeps_the_builders_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench = subprocess.run(
                [sys.executable, "-I", "-B", ROOT / "scripts/bench_wave64.py",
                 "--shader", "ps_ij_computed", "--build", tmp],
                cwd=tmp, env={**os.environ, "LEONIDA_BARY_FALLBACK": "1"},
                capture_output=True, timeout=30,
            )
        self.assertNotEqual(bench.returncode, 0)
        self.assertNotIn(b"Traceback", bench.stderr)
        self.assertIn(b"ps_ij_computed", bench.stderr)
        self.assertIn(b"PerVertexKHR", bench.stderr)

    def test_spv_alone_suffices_only_for_native(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch.object(bench_wave64.subprocess, "run", return_value=result):
                for mode in ("native", "emu"):
                    spirv = Path(tmp, mode, "spirv")
                    spirv.mkdir(parents=True)
                    (spirv / "sample.spv").touch()
                native = bench_wave64.build(tmp, "sample", False, "llvm-mc")
                self.assertEqual(Path(native), Path(tmp, "native/spirv/sample.spv"))
                with self.assertRaises(SystemExit) as caught:
                    bench_wave64.build(tmp, "sample", True, "llvm-mc")
                self.assertIn("sample.spvasm", str(caught.exception))
                emulated = Path(tmp, "emu/spirv/sample.spv")
                emulated.with_suffix(".spvasm").touch()
                self.assertEqual(Path(bench_wave64.build(tmp, "sample", True, "llvm-mc")), emulated)


if __name__ == "__main__":
    unittest.main()
