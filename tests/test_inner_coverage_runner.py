"""CPU-only checks of the synthetic InnerCoverage runner's exit-status contract."""
import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / 'runtime/test/inner_coverage/harness_result.py'
spec = importlib.util.spec_from_file_location('inner_result', path)
result = importlib.util.module_from_spec(spec)
spec.loader.exec_module(result)


class RunnerStatus(unittest.TestCase):
    def completed(self, code, stderr=''):
        return subprocess.CompletedProcess(['fixture'], code, stderr=stderr)

    def test_supported_success(self):
        result.check_harness_result(self.completed(0))

    def test_unsupported_hardware_skips_with_reason(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as error:
                result.check_harness_result(self.completed(77, 'primitiveUnderestimation=0'))
        self.assertEqual(error.exception.code, 77)
        self.assertIn('primitiveUnderestimation=0', output.getvalue())

    def test_validation_is_failure_even_with_skip_status(self):
        for code in (0, 77):
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                result.check_harness_result(self.completed(code, 'PROBE_VALIDATION: error'))

    def test_real_failure_is_not_skip(self):
        with self.assertRaises(RuntimeError):
            result.check_harness_result(self.completed(1, 'readback mismatch'))

    def test_matrix_propagates_child_skip(self):
        with patch('subprocess.run', return_value=self.completed(77)):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                result.run_suite_command(['fixture'], {})
        self.assertEqual(error.exception.code, 77)

    def test_real_child_skip_reaches_parent(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            result.run_suite_command([sys.executable, '-c', 'raise SystemExit(77)'], os.environ.copy())
        self.assertEqual(error.exception.code, 77)

    def test_matrix_preserves_child_failure(self):
        with patch('subprocess.run', return_value=self.completed(1)):
            with self.assertRaises(subprocess.CalledProcessError):
                result.run_suite_command(['fixture'], {})


if __name__ == '__main__':
    unittest.main()
