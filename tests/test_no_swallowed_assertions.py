"""No test wraps its own assertion in an except clause that throws the failure away.

    python -I tests/test_no_swallowed_assertions.py

``try: ...; self.assertEqual(...)`` followed by ``except Exception: pass`` can never fail: AssertionError is an
Exception, so the handler takes the failure along with the error it was written for. The sabotage control of
tests/semantic/test_sdwa_dst_mask.py did this, and its sabotaged module raised before the assertion, so the control
checked nothing. This reads every Python file under tests/ (it imports none) and fails on a ``try`` whose body asserts
and that has a handler for AssertionError (bare ``except:``, ``Exception``, ``BaseException``, ``AssertionError``)
which neither raises, asserts nor skips. Standard library only; no game data.
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# What an except clause names when it also catches a failed assertion.
CATCHES_FAILURES = {"Exception", "BaseException", "AssertionError", "self.failureException"}


def _called(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _asserts(node):
    """An assert statement, or a call of assert* / fail (unittest's, or a helper named like them)."""
    return any(isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and _called(n).startswith(("assert", "fail")))
               for n in ast.walk(node))


def _catches_failures(handler):
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(ast.unparse(name) in CATCHES_FAILURES for name in names)


def _keeps_the_failure(handler):
    """The handler raises (again, or something else), asserts, or skips the test."""
    for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
        if isinstance(node, ast.Raise) or (isinstance(node, ast.Call) and "skip" in _called(node).lower()):
            return True
    return _asserts(ast.Module(body=handler.body, type_ignores=[]))


def swallowed(source):
    """Line numbers of the try statements in `source` whose handlers drop a failed assertion of their body."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            continue
        if not _asserts(ast.Module(body=node.body, type_ignores=[])):
            continue
        if any(_catches_failures(h) and not _keeps_the_failure(h) for h in node.handlers):
            found.append(node.lineno)
    return sorted(found)


def test_sources(root):
    """Every .py file under tests/, as a path relative to `root` with / separators."""
    found = []
    for base, dirs, files in os.walk(os.path.join(root, "tests")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                found.append(os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/"))
    return found


class SwallowedAssertionTest(unittest.TestCase):
    def test_no_test_swallows_its_own_assertion(self):
        files = test_sources(ROOT)
        self.assertGreater(len(files), 300, "the scan found almost no test files; is the layout different?")
        bad = []
        for rel in files:
            with open(os.path.join(ROOT, rel), encoding="utf-8-sig") as f:
                bad += [f"{rel}:{line}" for line in swallowed(f.read())]
        self.assertEqual(bad, [], "these try statements assert in their body and have an except clause that also "
                         "catches the AssertionError and drops it, so the assertion can never fail: " + ", ".join(bad))

    def test_the_scan_flags_only_handlers_that_drop_the_failure(self):
        source = (
            "def t(self):\n"
            "    try:\n"                                    # 2: the failure goes with the error
            "        self.assertEqual(run(), 1)\n"
            "    except Exception:\n"
            "        pass\n"
            "    try:\n"                                    # 6: a bare except is the same
            "        assert run() == 1\n"
            "    except:\n"
            "        note()\n"
            "    try:\n"                                    # 10: a tuple that names one of them
            "        self.fail('x')\n"
            "    except (ValueError, AssertionError) as e:\n"
            "        print(e)\n"
            "    try:\n"                                    # 14: narrow: an AssertionError still fails the test
            "        self.assertEqual(run(), 1)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    try:\n"                                    # 18: raised again
            "        self.assertEqual(run(), 1)\n"
            "    except Exception:\n"
            "        cleanup()\n"
            "        raise\n"
            "    try:\n"                                    # 23: skipped, with the reason
            "        self.assertTrue(probe())\n"
            "    except Exception as e:\n"
            "        self.skipTest(str(e))\n"
            "    try:\n"                                    # 27: the handler asserts instead
            "        assert run() == 1\n"
            "    except AssertionError:\n"
            "        self.assertEqual(fallback(), 1)\n"
            "    try:\n"                                    # 31: nothing asserted in the body
            "        run()\n"
            "    except Exception:\n"
            "        pass\n"
        )
        self.assertEqual(swallowed(source), [2, 6, 10])


if __name__ == "__main__":
    unittest.main()
