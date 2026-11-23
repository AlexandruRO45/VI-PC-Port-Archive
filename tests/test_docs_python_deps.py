"""AGENTS.md names every tool that needs numpy or Pillow, and CONTRIBUTING.md says how pytest-based tests run.

    python -I tests/test_docs_python_deps.py

Both documents promise that Python tools use only the standard library "unless a file says otherwise" and that
`python -I tests/test_<name>.py` runs a unit test. The exceptions drift: `calibrated_texture.py` and `skinned_glb.py`
took numpy without joining the list, and 24 test files are pytest-based and fail (or run nothing) under `python -I`.
Reads tracked files only; no game files.
"""
import ast
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = {"numpy", "PIL"}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return [p for p in out.splitlines() if p]
    except (OSError, subprocess.CalledProcessError):
        found = []
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in (".git", "build", "__pycache__")]
            found += [os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/") for f in files]
        return found


def _is_third_party_import(node, modules=THIRD_PARTY):
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] in modules for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        return (node.module or "").split(".")[0] in modules
    return False


def imports_third_party(source):
    """True if the module imports numpy or PIL anywhere (top level, in a try block or inside a function)."""
    return any(_is_third_party_import(node) for node in ast.walk(ast.parse(source)))


def imports_third_party_at_module_scope(source):
    """True if the module has an unguarded top-level import of numpy or PIL."""
    return any(_is_third_party_import(node) for node in ast.parse(source).body)


def has_numpy_stdlib_fallback(source):
    """True if the module guards its numpy import with try/except ImportError and has no top-level unguarded import."""
    if imports_third_party_at_module_scope(source):
        return False
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Try) and any(_is_third_party_import(stmt, {"numpy"}) for stmt in node.body)
        for node in ast.walk(tree)
    )


def _convention_bullet(agents_text):
    bullet = re.search(r"^- Python tools run with `python -I`.*?(?=^- |^\Z)", agents_text, re.S | re.M)
    return bullet.group(0) if bullet else ""


def listed_tools(agents_text):
    """Basenames of the `*.py` tools in the AGENTS.md convention bullet that names the numpy/Pillow exceptions."""
    return set(re.findall(r"`([\w]+\.py)`", _convention_bullet(agents_text)))


def listed_tool_groups(agents_text):
    """Return (module_scope, deferred, fallback) tool sets parsed from the AGENTS.md convention bullet."""
    bullet = _convention_bullet(agents_text)
    if "require numpy at import time" not in bullet or "fall back to the standard library" not in bullet:
        return set(), set(), set(re.findall(r"`([\w]+\.py)`", bullet))
    before_require, rest = bullet.split("require numpy at import time", 1)
    if ", and only " in rest:
        middle, fallback_clause = rest.split(", and only ", 1)
    elif ", and " in rest:
        middle, fallback_clause = rest.split(", and ", 1)
    else:
        middle, fallback_clause = "", rest
    return (
        set(re.findall(r"`([\w]+\.py)`", before_require)),
        set(re.findall(r"`([\w]+\.py)`", middle)),
        set(re.findall(r"`([\w]+\.py)`", fallback_clause)),
    )


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


class DocsPythonDeps(unittest.TestCase):
    def test_listing_checker_sees_a_missing_and_a_stale_name(self):
        text = "- Python tools run with `python -I`; the exceptions are `a.py` and `gone.py`.\n- Next bullet `b.py`.\n"
        self.assertEqual(listed_tools(text), {"a.py", "gone.py"})
        self.assertTrue(imports_third_party("try:\n    import numpy as np\nexcept ImportError:\n    np = None\n"))
        self.assertTrue(imports_third_party("def f():\n    from PIL import Image\n"))
        self.assertFalse(imports_third_party("import json\nfrom os import path\n"))
        self.assertTrue(imports_third_party_at_module_scope("import numpy as np\n"))
        self.assertFalse(imports_third_party_at_module_scope(
            "try:\n    import numpy as np\nexcept ImportError:\n    np = None\n"))
        self.assertFalse(imports_third_party_at_module_scope("def f():\n    import numpy as np\n"))
        self.assertTrue(has_numpy_stdlib_fallback(
            "try:\n    import numpy as np\nexcept ImportError:\n    np = None\n"))
        self.assertFalse(has_numpy_stdlib_fallback("import numpy as np\n"))
        self.assertFalse(has_numpy_stdlib_fallback("def f():\n    import numpy as np\n"))
        old_lumped = (
            "- Python tools run with `python -I`; the exceptions are `bc_feature.py`, `world_grid_atlas.py` "
            "and `skinned_glb.py` (which fall back to the standard library when numpy is absent).\n"
        )
        self.assertEqual(listed_tool_groups(old_lumped),
                         (set(), set(), {"bc_feature.py", "world_grid_atlas.py", "skinned_glb.py"}))

    def test_agents_lists_the_tools_that_need_numpy_or_pillow(self):
        files = tracked_files()
        needing = {os.path.basename(p) for p in files
                   if p.endswith(".py") and (p.startswith(("tools/", "scripts/")) or p == "make_game.py")
                   and imports_third_party(read(p))}
        listed = listed_tools(read("AGENTS.md"))
        self.assertTrue(needing and listed)
        self.assertEqual(sorted(needing - listed), [], "tools that import numpy/Pillow but are not in AGENTS.md")
        self.assertEqual(sorted(listed - needing), [], "tools named in AGENTS.md that no longer import numpy/Pillow")

    def test_agents_distinguishes_module_scope_and_fallback_numpy_tools(self):
        files = [p for p in tracked_files()
                 if p.endswith(".py") and (p.startswith(("tools/", "scripts/")) or p == "make_game.py")]
        actual_module_scope = set()
        actual_deferred = set()
        actual_fallback = set()
        for p in files:
            src = read(p)
            if not imports_third_party(src):
                continue
            name = os.path.basename(p)
            if imports_third_party_at_module_scope(src):
                actual_module_scope.add(name)
            elif has_numpy_stdlib_fallback(src):
                actual_fallback.add(name)
            else:
                actual_deferred.add(name)
        doc_module_scope, doc_deferred, doc_fallback = listed_tool_groups(read("AGENTS.md"))
        self.assertEqual(sorted(doc_module_scope), sorted(actual_module_scope),
                         "AGENTS.md module-scope numpy tools do not match tools with top-level unguarded imports")
        self.assertEqual(sorted(doc_deferred), sorted(actual_deferred),
                         "AGENTS.md deferred function-level numpy/Pillow tools do not match actual sources")
        self.assertEqual(sorted(doc_fallback), sorted(actual_fallback),
                         "AGENTS.md stdlib-fallback numpy tools do not match tools with try/except ImportError")

    def test_contributing_says_pytest_files_do_not_run_under_python_i(self):
        pytest_only = [p for p in tracked_files()
                       if re.fullmatch(r"tests/test_[\w]+\.py", p) and "__main__" not in read(p)]
        if pytest_only:
            self.assertIn("pytest", read("CONTRIBUTING.md"),
                          f"{len(pytest_only)} test files have no __main__ block (e.g. {pytest_only[0]}) but "
                          "CONTRIBUTING.md never mentions pytest")


if __name__ == "__main__":
    unittest.main()
