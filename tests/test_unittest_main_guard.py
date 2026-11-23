"""Every test file's `unittest.main()` guard comes after all of its test classes.

    python -I tests/test_unittest_main_guard.py

The project runs each test file with `python -I tests/test_<name>.py`. A class defined below the guard is never
registered in that run, so its tests pass unseen in CI's standard-library step and only appear under
`unittest discover`. Two files hid seven tests this way; this test reads the test sources themselves and fails
if it happens again. Standard library only; no game data, no GPU.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def calls_unittest_main(statement):
    """True when a statement calls unittest.main, not merely mentions it in a string."""
    for child in ast.walk(statement):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "main" and ast.unparse(child.func.value) == "unittest":
                return True
    return False


def is_main_guard(node):
    """An `if __name__ == "__main__":` block that actually calls unittest.main."""
    if not isinstance(node, ast.If):
        return False
    test = ast.unparse(node.test)
    if "__name__" not in test or "__main__" not in test:
        return False
    return any(calls_unittest_main(statement) for statement in node.body)


def is_test_case(node):
    """A class deriving from unittest.TestCase, imported either way."""
    if not isinstance(node, ast.ClassDef):
        return False
    return any(ast.unparse(base).split(".")[-1] == "TestCase" for base in node.bases)


class MainGuardDetectionTests(unittest.TestCase):
    """The guard detector follows calls, so it does not miss an argument or mistake a string for a call."""

    @staticmethod
    def guard(body):
        return ast.parse('if __name__ == "__main__":\n' + body).body[0]

    def test_main_with_arguments_is_a_guard(self):
        self.assertTrue(is_main_guard(self.guard("    unittest.main(argv=['x'])\n")))

    def test_multi_statement_block_is_a_guard(self):
        self.assertTrue(is_main_guard(self.guard("    print('start')\n    unittest.main()\n")))

    def test_main_wrapped_in_another_call_is_a_guard(self):
        self.assertTrue(is_main_guard(self.guard("    sys.exit(unittest.main())\n")))

    def test_a_string_that_mentions_main_is_not_a_guard(self):
        self.assertFalse(is_main_guard(self.guard("    print('unittest.main(argv=[])')\n")))


class MainGuardOrderTests(unittest.TestCase):
    def test_no_test_case_is_defined_below_the_guard(self):
        hidden = []
        for path in sorted((ROOT / "tests").rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            guards = [node for node in tree.body if is_main_guard(node)]
            if not guards:
                continue
            guard_line = min(node.lineno for node in guards)
            for node in tree.body:
                if is_test_case(node) and node.lineno > guard_line:
                    hidden.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
        self.assertEqual(hidden, [], "test classes below unittest.main() do not run under python -I")


if __name__ == "__main__":
    unittest.main()
