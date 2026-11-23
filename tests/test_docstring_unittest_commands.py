"""A `python -m unittest` command in a test file's docstring can import that file.

    python -I tests/test_docstring_unittest_commands.py

Test files say in their docstring how to run them. `python -m unittest <file>` imports the file as a module of the
folder it is started from, so it only works when the folders around it are on sys.path: under `python -I` the
checkout is not (No module named 'tests'), and a tests/semantic file named that way cannot import the support.py next
to it even without -I (No module named 'support'). `unittest discover -s <folder>` puts the folder on sys.path
itself, so a docstring that runs one file through unittest uses `discover -s <its folder> -p <its name>` (or plain
`discover -s <its folder>` for the whole folder); a file with a `__main__` block can also say
`python -I tests/test_x.py`, which this test leaves alone.

Reads the .py files in tests/ and tests/semantic only; no game files, no subprocesses.
"""
import ast
import fnmatch
import os
import re
import shlex
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ("tests", "tests/semantic")
COMMAND = re.compile(r"\bpy(?:thon[\d.]*)?(?:\s+-[A-Za-z]+)*\s+-m\s+unittest\b([^\n`#]*)")


def files_to_check():
    for folder in FOLDERS:
        for name in sorted(os.listdir(os.path.join(ROOT, folder))):
            if name.endswith(".py"):
                yield f"{folder}/{name}"


def docstring_commands(path):
    """The argument text after `-m unittest` of every unittest command in path's module docstring."""
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        source = f.read()
    if "-m unittest" not in source:
        return []
    try:
        doc = ast.get_docstring(ast.parse(source), clean=False)
    except SyntaxError:
        return []
    # a bare mention, or a placeholder such as <file>, describes the form; it is not a command to run
    found = (m.group(1).strip() for m in COMMAND.finditer(doc or ""))
    return [arguments for arguments in found if arguments and "<" not in arguments]


def problem(path, arguments):
    """Why `python -m unittest <arguments>`, as path's docstring writes it, cannot import path; None if it can."""
    try:
        words = shlex.split(arguments)
    except ValueError:
        words = arguments.split()
    if not words or words[0] != "discover":
        return "names the file instead of discovering it, so its folder is not on sys.path"
    start, pattern = ".", "test*.py"
    for option, value in zip(words, words[1:]):
        if option in ("-s", "--start-directory"):
            start = value
        elif option in ("-p", "--pattern"):
            pattern = value
    folder, name = path.rsplit("/", 1)
    if os.path.normpath(start).replace(os.sep, "/") != folder:
        return f"discovers in {start}, not in {folder}"
    if not fnmatch.fnmatch(name, pattern):
        return f"pattern {pattern} does not match {name}"
    return None


class ProblemRuleTest(unittest.TestCase):
    def test_a_named_file_is_refused(self):
        self.assertIsNotNone(problem("tests/test_x.py", "tests/test_x.py -v"))
        self.assertIsNotNone(problem("tests/semantic/test_x.py", "tests/semantic/test_x.py -v"))
        self.assertIsNotNone(problem("tests/test_x.py", "tests.test_x"))

    def test_discover_must_reach_the_file(self):
        self.assertIsNone(problem("tests/semantic/test_x.py", "discover -s tests/semantic -p test_x.py -v"))
        self.assertIsNone(problem("tests/semantic/test_x.py", "discover -s tests/semantic -p 'test_x.py'"))
        self.assertIsNone(problem("tests/test_x.py", "discover -s tests -v"))
        self.assertIsNotNone(problem("tests/semantic/test_x.py", "discover -s tests -p test_x.py"))
        self.assertIsNotNone(problem("tests/semantic/test_x.py", "discover -s tests/semantic -p test_y.py"))


class DocstringCommandsTest(unittest.TestCase):
    def test_the_scan_sees_commands(self):
        # the docstrings of this folder hold dozens of discover lines; finding none means the scan broke
        found = sum(len(docstring_commands(path)) for path in files_to_check())
        self.assertGreater(found, 10)

    def test_every_docstring_command_imports_its_file(self):
        bad = []
        for path in files_to_check():
            for arguments in docstring_commands(path):
                why = problem(path, arguments)
                if why:
                    bad.append(f"{path}: python -m unittest {arguments}: {why}")
        self.assertEqual(bad, [], "\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
