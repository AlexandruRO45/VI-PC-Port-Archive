"""Test the Python SPIR-V assembler's CLI output and failure handling."""
import contextlib
import errno
import io
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import spirv_asm  # noqa: E402


class SpirvAsmOutputTests(unittest.TestCase):
    def test_cli_write_failure_preserves_the_last_complete_output(self):
        def interrupt_binary_writes(opener):
            def open_file(*args, **kwargs):
                handle = opener(*args, **kwargs)
                if handle.mode == "wb":
                    write = handle.write

                    def interrupted(data):
                        write(data[:4])
                        raise OSError(errno.ENOSPC, "simulated full disk")

                    handle.write = interrupted
                return handle

            return open_file

        for has_previous in (True, False):
            with self.subTest(has_previous=has_previous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "input.spvasm"
                output = root / "output.spv"
                source.write_bytes(b"OpNop\n")
                previous = spirv_asm.assemble("OpReturn\n")
                if has_previous:
                    output.write_bytes(previous)
                original_paths = set(root.iterdir())
                error = io.StringIO()
                with (mock.patch("builtins.open", interrupt_binary_writes(open)),
                      mock.patch.object(io, "open", interrupt_binary_writes(io.open)),
                      contextlib.redirect_stderr(error)):
                    result = spirv_asm.main([str(source), "-o", str(output)])
                self.assertEqual(result, 1)
                self.assertIn("simulated full disk", error.getvalue())
                self.assertEqual(set(root.iterdir()), original_paths)
                if has_previous:
                    self.assertEqual(output.read_bytes(), previous)

    def test_cli_publishes_through_an_output_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.spvasm"
            output = root / "output.spv"
            target = root / "target.spv"
            source.write_bytes(b"OpNop\n")
            target.write_bytes(spirv_asm.assemble("OpReturn\n"))
            target.chmod(0o640)
            previous_mode = stat.S_IMODE(target.stat().st_mode)
            try:
                output.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable")
            self.assertEqual(spirv_asm.main([str(source), "-o", str(output)]), 0)
            self.assertTrue(output.is_symlink())
            self.assertEqual(target.read_bytes(), spirv_asm.assemble("OpNop\n"))
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), previous_mode)

    def test_cli_respects_a_read_only_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.spvasm", root / "output.spv"
            source.write_bytes(b"OpNop\n")
            previous = spirv_asm.assemble("OpReturn\n")
            output.write_bytes(previous)
            output.chmod(0o444)
            try:
                with contextlib.suppress(PermissionError):
                    os.close(os.open(output, os.O_WRONLY))
                    self.skipTest("this process can write read-only files")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(spirv_asm.main([str(source), "-o", str(output)]), 1)
                self.assertEqual(output.read_bytes(), previous)
                self.assertEqual(set(root.iterdir()), {source, output})
            finally:
                output.chmod(0o600)

    def test_cli_still_writes_to_the_null_device(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.spvasm"
            source.write_bytes(b"OpNop\n")
            self.assertEqual(spirv_asm.main([str(source), "-o", os.devnull]), 0)
            self.assertTrue(stat.S_ISCHR(os.stat(os.devnull).st_mode))


if __name__ == "__main__":
    unittest.main()
