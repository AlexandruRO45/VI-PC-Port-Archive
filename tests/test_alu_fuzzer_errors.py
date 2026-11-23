"""ALU differential fuzzers must fail on invalid modules and interpreter crashes.

Synthetic inputs only; python -I tests/test_alu_fuzzer_errors.py needs no GPU or LLVM.
"""
import contextlib
import importlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests" / "semantic"))

import rdna2spirv  # noqa: E402
import spirv_mini_interp as interp  # noqa: E402

TOOLS = tuple(importlib.import_module(name) for name in
              ("fmisc_diff", "pk16_diff", "salu_diff", "sdwa_diff", "vint_diff"))


class InterpreterErrors(unittest.TestCase):
    def cli(self, tool, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tool.main(["--programs", "2", "--seed", "1", "--max-ops", "1", *extra])
        return rc, out.getvalue()

    def assert_error(self, rc, output, kind):
        self.assertEqual(rc, 2, output)
        self.assertIn("harness error: mini_interp: " + kind, output)
        self.assertNotIn("skipped", output)

    def test_invalid_module_after_a_passing_program(self):
        translate = rdna2spirv.translate
        for tool in TOOLS:
            for instruction in ("OpSelectionMerge %bad None", "%bad = OpPhi %uint %uint_0 %entry"):
                with self.subTest(tool=tool.__name__, instruction=instruction):
                    calls = 0

                    def corrupt_second(path):
                        nonlocal calls
                        calls += 1
                        text = translate(path)
                        if calls == 2:
                            self.assertIn("OpFunctionEnd", text)
                            text = text.replace("OpFunctionEnd", instruction + "\nOpFunctionEnd", 1)
                        return text

                    with mock.patch.object(rdna2spirv, "translate", side_effect=corrupt_second):
                        rc, output = self.cli(tool)
                    self.assertEqual(calls, 2)
                    self.assert_error(rc, output, "InvalidModule:")

    def test_crashes_after_a_passing_program(self):
        run_fragment = interp.run_fragment
        for tool in TOOLS:
            for error_type in (RuntimeError, KeyError, ValueError):
                with self.subTest(tool=tool.__name__, error=error_type.__name__):
                    calls = 0

                    def fail_second(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 2:
                            raise error_type("synthetic interpreter failure")
                        return run_fragment(*args, **kwargs)

                    with mock.patch.object(interp, "run_fragment", side_effect=fail_second):
                        rc, output = self.cli(tool)
                    self.assertEqual(calls, 2)
                    self.assert_error(rc, output, error_type.__name__ + ":")

    def test_explicit_unsupported_remains_a_counted_skip(self):
        run_fragment = interp.run_fragment
        for tool in TOOLS:
            with self.subTest(tool=tool.__name__):
                calls = 0

                def skip_second(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise interp.UnsupportedConstruct("unmodelled operation")
                    return run_fragment(*args, **kwargs)

                with mock.patch.object(interp, "run_fragment", side_effect=skip_second):
                    rc, output = self.cli(tool)
                self.assertEqual(rc, 0, output)
                self.assertIn("1 pass, 0 mismatch, 1 skipped", output)
                self.assertIn("mini_interp: UnsupportedConstruct: unmodelled operation", output)

    def test_hardware_match_does_not_hide_error_and_oracle_is_closed(self):
        from hw_oracle import runner
        for tool in TOOLS:
            with self.subTest(tool=tool.__name__), contextlib.ExitStack() as stack:
                suffix = "" if tool.__name__ in ("salu_diff", "sdwa_diff") else "_e32"
                lines = [f"v_mov_b32{suffix} v{i}, 0" for i in range(8, 12)]
                self.assertEqual(tool.reference(lines), [0] * 4)
                oracle = mock.Mock(arch="gfx1030", device="synthetic")
                oracle.run.return_value = [[0] * 4, [0] * 4]
                stack.enter_context(mock.patch.object(runner, "open_oracle", return_value=oracle))
                stack.enter_context(mock.patch.object(tool, "generate", return_value=lines))
                if hasattr(tool, "hw_prepare"):
                    stack.enter_context(mock.patch.object(tool, "hw_prepare", return_value=oracle.run.return_value))
                else:
                    stack.enter_context(mock.patch.object(tool, "hw_kernel", side_effect=lambda lines, wave: lines))
                stack.enter_context(mock.patch.object(interp, "run_fragment",
                                                     side_effect=interp.InvalidModule("invalid merge")))
                rc, output = self.cli(tool, "--oracle", "translator,hw")
                self.assert_error(rc, output, "InvalidModule:")
                oracle.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
