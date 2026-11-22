"""No Python file defines the same function twice in one scope or repeats a key in a dict literal.

    python -I tests/test_no_duplicate_definitions.py

A merge that keeps both sides of a hunk leaves the second definition silently replacing the first. In a test class
the first test method then never runs, and nothing reports it (TEST-893: tests/semantic/test_salu_and_saveexec_b64.py
carried its whole header block twice, and tools/rdna2spirv.py's FORMATS listed key 5 twice). This parses every
tracked .py file with `ast` and reports a function or method defined twice directly in the same module or class body,
and a constant key repeated in one dict literal. Definitions in different branches of an `if` or `try` are not
compared, and property setters/getters/deleters, `typing.overload` and `singledispatch` registrations are allowed.
Reads tracked files only (`git ls-files`, with a directory walk when git is not available); no game files.
"""
import ast
import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Submodules and vendored code are not this repository's to fix.
SKIP_PREFIXES = ("third_party/",)
ALLOWED_DECORATORS = {"setter", "getter", "deleter", "overload", "register"}


def tracked_python_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
        files = [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for root, dirs, names in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__", "third_party")]
            files += [os.path.relpath(os.path.join(root, n), ROOT).replace(os.sep, "/")
                      for n in names if n.endswith(".py")]
    return sorted(p for p in files if not p.startswith(SKIP_PREFIXES))


def _allowed(func):
    for d in func.decorator_list:
        if isinstance(d, ast.Call):
            d = d.func
        name = d.attr if isinstance(d, ast.Attribute) else d.id if isinstance(d, ast.Name) else None
        if name in ALLOWED_DECORATORS:
            return True
    return False


def duplicates(path, source):
    """[(path:line, message)] for functions defined twice in one module/class body and repeated dict keys."""
    out = []
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef)):
            seen = {}
            for item in node.body:
                if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) or _allowed(item):
                    continue
                if item.name in seen:
                    scope = f"class {node.name}" if isinstance(node, ast.ClassDef) else "module"
                    out.append((f"{path}:{item.lineno}",
                                f"{scope} defines {item.name}() again (first at line {seen[item.name]})"))
                else:
                    seen[item.name] = item.lineno
        elif isinstance(node, ast.Dict):
            keys = {}
            for key in node.keys:
                if not isinstance(key, ast.Constant):
                    continue
                k = (type(key.value), key.value)
                if k in keys:
                    out.append((f"{path}:{key.lineno}",
                                f"dict key {key.value!r} repeated (first at line {keys[k]})"))
                else:
                    keys[k] = key.lineno
    return out


class NoDuplicateDefinitions(unittest.TestCase):
    def test_checker_reports_only_real_duplicates(self):
        source = (
            "def f(): pass\n"
            "def f(): pass\n"                      # 2: reported
            "if x:\n    def g(): pass\n"
            "else:\n    def g(): pass\n"            # branches: not compared
            "class C:\n"
            "    def test_a(self): pass\n"
            "    def test_a(self): pass\n"          # 9: reported
            "    @property\n    def p(self): pass\n"
            "    @p.setter\n    def p(self, v): pass\n"
            "d = {1: 'a', True: 'b', '1': 'c', 1: 'd', k: 0, k: 1}\n"  # 14: only the second 1 is reported
        )
        self.assertEqual([where for where, _ in duplicates("t.py", source)], ["t.py:2", "t.py:9", "t.py:14"])

    def test_repository_has_no_duplicate_definitions(self):
        files = tracked_python_files()
        self.assertTrue(files)
        found = []
        for p in files:
            with open(os.path.join(ROOT, p), encoding="utf-8") as f:
                found += duplicates(p, f.read())
        self.assertEqual(found, [], "duplicate definitions: " + "; ".join(f"{w} {m}" for w, m in found))


if __name__ == "__main__":
    unittest.main()
