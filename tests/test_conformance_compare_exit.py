"""tools/conformance/compare.py: an unexpected exception is exit 2, never the mismatch code 1.

main() promises that a caller can tell "the pixels differ" (1) from "this program crashed" (2). The
unexpected-failure path printed the traceback through a module that was never imported, so it raised
NameError instead and Python exited 1. Standard library only; no GPU and no game files.
"""
import contextlib
import importlib.util
import io
import os
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(os.path.dirname(HERE), "tools", "conformance", "compare.py")


def load_compare():
    spec = importlib.util.spec_from_file_location("conformance_compare_under_test", COMPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Parser:
    """Stands in for compare.py's argument parser: every run calls `handler`."""

    def __init__(self, handler):
        self.handler = handler

    def parse_args(self, argv):
        return types.SimpleNamespace(handler=self.handler)


class UnexpectedFailureExit(unittest.TestCase):
    def setUp(self):
        self.compare = load_compare()

    def run_main(self, handler):
        self.compare.build_argument_parser = lambda: Parser(handler)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = self.compare.main([])
        return code, stderr.getvalue()

    def test_unexpected_exception_is_exit_error_with_a_traceback(self):
        def crash(arguments):
            raise ZeroDivisionError("synthetic crash")

        code, stderr = self.run_main(crash)
        self.assertEqual(code, self.compare.EXIT_ERROR)
        self.assertEqual(code, 2)
        self.assertIn("Traceback", stderr)
        self.assertIn("ZeroDivisionError: synthetic crash", stderr)
        self.assertIn("this is a bug in compare.py, not a pixel mismatch", stderr)

    def test_tool_error_is_exit_error_without_a_traceback(self):
        def refuse(arguments):
            raise self.compare.ToolError("synthetic refusal")

        code, stderr = self.run_main(refuse)
        self.assertEqual(code, 2)
        self.assertIn("error: synthetic refusal", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_handler_result_is_returned(self):
        self.assertEqual(self.run_main(lambda arguments: self.compare.EXIT_FAILED)[0], 1)
        self.assertEqual(self.run_main(lambda arguments: self.compare.EXIT_PASSED)[0], 0)


if __name__ == "__main__":
    unittest.main()
