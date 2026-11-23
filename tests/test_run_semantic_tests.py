"""scripts/run_semantic_tests.py on a throwaway suite: it finds what unittest discover finds, and a failing, crashing
or unimportable file always makes it exit 1 with that file named, in parallel and with -j 1.

    python -I tests/test_run_semantic_tests.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_semantic_tests as r  # noqa: E402

PASS = "import unittest\nimport helper\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(helper.X, 1)\n" \
       "    def test_b(self): pass\n"
FAIL = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(1, 2, 'boom-msg')\n"
BROKEN = "import no_such_module_anywhere\n"


def suite(tmp, files):
    for rel, text in files.items():
        path = os.path.join(tmp, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return tmp


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = r.main(list(argv))
    return code, out.getvalue()


@contextlib.contextmanager
def denied(folder):
    """Hold `folder` so this process cannot list it: chmod 000 on POSIX, an exclusive handle on Windows.

    Both are undone on exit, so the temporary folder can still be removed; no ACL is written.
    """
    if os.name == "nt":
        handle = _open_without_sharing(folder)
        try:
            yield
        finally:
            _close(handle)
    else:
        os.chmod(folder, 0)
        try:
            yield
        finally:
            os.chmod(folder, 0o700)


def _open_without_sharing(folder):
    """A directory handle held with dwShareMode 0, so os.scandir reports a sharing violation (PermissionError)."""
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    generic_read, open_existing, file_flag_backup_semantics = 0x80000000, 3, 0x02000000
    handle = kernel32.CreateFileW(folder, generic_read, 0, None, open_existing, file_flag_backup_semantics, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise unittest.SkipTest("cannot open the folder without sharing it")
    return handle


def _close(handle):
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(handle)


class RunnerTest(unittest.TestCase):
    def test_finds_what_discover_finds(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_one.py": "", "helper.py": "", "pkg/__init__.py": "", "pkg/test_two.py": "",
                        "plain_dir/test_three.py": "", "testx.py": ""})
            # discover enters packages only, and its pattern is test*.py
            self.assertEqual(r.test_modules(tmp), ["test_one", "testx", "pkg.test_two"])

    def test_all_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p1.py": PASS, "test_p2.py": PASS})
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 0, out)
            self.assertIn("2 files, 4 tests, 0 files failed", out)

    def test_a_failure_or_import_error_is_reported_and_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "test_f.py": FAIL, "test_b.py": BROKEN})
            code, out = run(tmp, "-j", "3")
            self.assertEqual(code, 1, out)
            self.assertIn("==== FAILED: test_f", out)
            self.assertIn("boom-msg", out)                         # the failing file's own output is shown
            self.assertIn("==== FAILED: test_b", out)
            self.assertIn("2 files failed", out)
            self.assertTrue(out.rstrip().endswith("test_b, test_f"), out)

    def test_one_job_runs_discover_and_keeps_its_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS, "test_f.py": FAIL})
            self.assertEqual(run(tmp, "-j", "1")[0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"helper.py": "X = 1\n", "test_p.py": PASS})
            self.assertEqual(run(tmp, "-j", "1")[0], 0)

    def test_an_empty_suite_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1)
            self.assertIn("no test*.py files", out)

    def test_an_unreadable_package_fails_the_run_instead_of_being_skipped(self):
        """A package the user cannot list may hold failing files: it is exit 1 and one line, as under -j 1."""
        with tempfile.TemporaryDirectory() as tmp:
            suite(tmp, {"test_outer.py": PASS, "pkg/__init__.py": "", "pkg/test_inner.py": FAIL})
            folder = os.path.join(tmp, "pkg")
            with denied(folder):
                try:
                    os.scandir(folder).close()
                except PermissionError:
                    pass
                else:
                    self.skipTest("this platform lets the owner list the folder anyway")
                code, out = run(tmp, "-j", "2")
            self.assertEqual(code, 1, out)
            self.assertIn("cannot read", out)
            self.assertIn(folder, out)
            self.assertNotIn("boom-msg", out)   # the hidden file did not run; the run is not green either


if __name__ == "__main__":
    unittest.main()
