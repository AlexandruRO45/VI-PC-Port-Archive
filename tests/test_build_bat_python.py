"""Build.bat starts make_game.py with the python.exe it found by that file's full path (sys.executable).

    python -I tests/test_build_bat_python.py

The Python install manager's `py` and `python` are app execution aliases in WindowsApps. A Python started through
either one, and every program that Python starts, does not see folders that other programs create under
%LOCALAPPDATA% or %APPDATA%; the same python.exe started by its full path does (measured on Windows 11 26200 with
PythonManager 26.4.240.0 and Python 3.14.4; #1233 measured it for `python`). Build.bat's `:resolve_python` swaps PY
for sys.executable before it starts make_game.py. This test runs that block as written, with small launchers standing
in for the aliases, so it needs no install manager, compiler or game files. Windows only (cmd.exe).
"""
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r"^:resolve_python\r?\n.*?^exit /b 0\r?$", re.M | re.S)


def resolve_block():
    """Build.bat's `:resolve_python` label through its `exit /b 0`, as written."""
    found = BLOCK.findall((ROOT / "Build.bat").read_text(encoding="utf-8"))
    if len(found) != 1:
        raise AssertionError(f"Build.bat: expected one `:resolve_python` block ending in `exit /b 0`, found "
                             f"{len(found)}; update tests/test_build_bat_python.py together with Build.bat")
    return found[0].replace("\r\n", "\n")


@unittest.skipUnless(os.name == "nt", "Build.bat runs in cmd.exe")
class ResolvePythonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Spaces in the launcher folder, as in "C:\Users\First Last\...".
        self.bin = self.tmp / "launch dir"
        self.bin.mkdir()
        self.driver = self.tmp / "driver.bat"
        # The driver sets PY, calls the block and prints PY. Paths come in through the environment so the batch file
        # itself stays ASCII; cmd.exe reads a batch file in the console code page.
        with open(self.driver, "w", encoding="ascii", newline="\r\n") as f:
            f.write("@echo off\nsetlocal\nset PY=%LEONIDA_TEST_PY%\ncall :resolve_python\n"
                    "\"%LEONIDA_TEST_REPORTER%\" -I -c \"import os; print(ascii(os.environ['PY']))\"\n"
                    "exit /b 0\n\n" + resolve_block() + "\n")

    def launcher(self, name, body):
        path = self.bin / name
        with open(path, "w", encoding="ascii", newline="\r\n") as f:
            f.write("@echo off\n" + body + "\n")
        return path

    def resolve(self, py, target=sys.executable):
        """PY after `call :resolve_python`, starting from `py`; launchers run `target`."""
        env = {**os.environ, "LEONIDA_TEST_PY": py, "LEONIDA_TEST_REPORTER": sys.executable,
               "LEONIDA_TEST_TARGET": str(target)}
        result = subprocess.run(["cmd.exe", "/d", "/c", str(self.driver)], env=env, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return ast.literal_eval(result.stdout.decode("ascii").strip())

    def assert_names(self, py, executable):
        """PY is `executable`, quoted, as `:find_user_python` sets it."""
        self.assertTrue(py.startswith('"') and py.endswith('"'), py)
        self.assertTrue(os.path.samefile(py[1:-1], executable), (py, executable))

    def test_a_launcher_is_replaced_by_the_interpreter_it_runs(self):
        # A quoted launcher path, as `:find_user_python` sets PY, and a bare command with an argument, as `py -3.14`.
        quoted = self.launcher("py alias.cmd", '"%LEONIDA_TEST_TARGET%" %*')
        self.assert_names(self.resolve(f'"{quoted}"'), sys.executable)
        bare_dir = self.tmp / "bare"
        bare_dir.mkdir()
        bare = bare_dir / "pylaunch.cmd"
        with open(bare, "w", encoding="ascii", newline="\r\n") as f:
            f.write('@"%LEONIDA_TEST_TARGET%" %2 %3 %4 %5 %6 %7 %8 %9\n')
        self.assert_names(self.resolve(f"{bare} -3.14"), sys.executable)

    def test_the_interpreter_by_its_full_path_stays(self):
        self.assert_names(self.resolve(f'"{sys.executable}"'), sys.executable)

    def test_a_non_ascii_interpreter_path_comes_back_intact(self):
        # for /f reads the child's output in the console code page; a path printed in another one would be garbled
        # and the swap silently skipped. A venv gives an interpreter whose sys.executable has such a path.
        venv = self.tmp / "Pé ñ"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, capture_output=True,
                       timeout=300)
        venv_python = venv / "Scripts" / "python.exe"
        alias = self.launcher("python.cmd", '"%LEONIDA_TEST_TARGET%" %*')
        self.assert_names(self.resolve(f'"{alias}"', target=venv_python), venv_python)

    def test_py_is_kept_when_no_usable_path_comes_back(self):
        for name, body in (("silent.cmd", "exit /b 1"),
                           ("elsewhere.cmd", r"echo C:\leonida-no-such-folder\python.exe")):
            with self.subTest(launcher=name):
                py = f'"{self.launcher(name, body)}"'
                self.assertEqual(self.resolve(py), py)

    def test_build_bat_resolves_before_it_starts_make_game(self):
        text = (ROOT / "Build.bat").read_text(encoding="utf-8")
        call = text.find("\ncall :resolve_python")
        start = text.find('%PY% -I "%~dp0make_game.py"')
        self.assertGreater(call, 0, "Build.bat no longer calls :resolve_python")
        self.assertGreater(start, call, "Build.bat must call :resolve_python before it starts make_game.py")


if __name__ == "__main__":
    unittest.main()
