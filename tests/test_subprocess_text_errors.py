"""Every child process whose output a tool decodes as text says what to do with bytes it can't decode (BUILD-1265).

`subprocess.run(..., text=True)` decodes with the locale's code page unless Python runs in UTF-8 mode, and with
`errors="strict"`. A Windows console tool prints in the OEM code page ("non \x8a riconosciuto" from cmd.exe in
Italian), `wsl.exe` prints its errors in UTF-16LE, and llvm-mc or spirv-val can echo a path in another encoding. The
tool then loses that output: on POSIX run() raises UnicodeDecodeError, and on Windows (capture_output reads through
threads) the reader thread dies and stdout comes back as None, with no exception in the tool.

This parses the tools the user runs (tools/, scripts/, make_game.py, runtime/test/shaders/) with ast and fails on a
`run`/`check_output`/`Popen`/`call`/`check_call` call that decodes text (`text=` or `universal_newlines=` not False,
or `encoding=`) without an `errors=` argument. Standard library only, no game files.

    python -I tests/test_subprocess_text_errors.py -v
"""
import ast
import os
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKED = ("tools/", "scripts/", "make_game.py", "runtime/test/shaders/")
# Fixed by its own PR (#4395, BUILD-1245); drop this entry when that lands.
PENDING = {"tools/shader_disasm.py"}
CALLS = {"run", "check_output", "Popen", "call", "check_call"}


def tracked_python(root=ROOT):
    # NUL-separated and as bytes: a path with a space or a non-ASCII letter stays one whole path (git quotes those
    # in its default output, and splitting on whitespace cuts them).
    out = subprocess.run(["git", "-C", root, "ls-files", "-z", "*.py"], capture_output=True, check=True).stdout
    files = [f.decode("utf-8", "surrogateescape") for f in out.split(b"\0") if f]
    return [f for f in files if f.startswith(CHECKED) and f not in PENDING]


def decodes_text(call):
    for k in call.keywords:
        if k.arg == "encoding":
            return True
        if k.arg in ("text", "universal_newlines"):
            if not (isinstance(k.value, ast.Constant) and k.value.value in (False, None)):
                return True
    return False


def missing_errors(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else None
        if name in CALLS and decodes_text(node) and not any(k.arg == "errors" for k in node.keywords):
            found.append(f"{path}:{node.lineno}: {ast.unparse(fn)}(...)")
    return found


class SubprocessTextErrorsTest(unittest.TestCase):
    def test_text_mode_child_output_names_errors(self):
        try:
            files = tracked_python()
        except (OSError, subprocess.CalledProcessError) as e:
            self.skipTest(f"git ls-files unavailable: {e}")
        self.assertTrue(files, "no tracked Python files found under " + ", ".join(CHECKED))
        bad = [hit for f in files for hit in missing_errors(f)]
        self.assertEqual(bad, [], "text-mode calls without errors=:\n" + "\n".join(bad))

    def test_paths_with_spaces_and_non_ascii_letters_are_listed_whole(self):
        names = ["tools/plain.py", "tools/with space.py", "tools/é.py"]
        with tempfile.TemporaryDirectory() as root:
            try:
                subprocess.run(["git", "-C", root, "init", "-q"], check=True, capture_output=True)
            except (OSError, subprocess.CalledProcessError) as e:
                self.skipTest(f"git unavailable: {e}")
            for name in names:
                os.makedirs(os.path.join(root, os.path.dirname(name)), exist_ok=True)
                with open(os.path.join(root, name), "w", encoding="utf-8") as f:
                    f.write("")
            subprocess.run(["git", "-C", root, "add", "--", *names], check=True, capture_output=True)
            self.assertEqual(sorted(tracked_python(root)), sorted(names))

    def test_the_check_sees_a_missing_errors(self):
        # Negative control: the checker flags exactly the call that lacks errors=.
        src = ("import subprocess\n"
               "subprocess.run(['x'], capture_output=True, text=True)\n"
               "subprocess.run(['x'], capture_output=True, text=True, errors='replace')\n"
               "subprocess.run(['x'], capture_output=True)\n"
               "subprocess.check_output(['x'], encoding='utf-8')\n")
        tree = ast.parse(src)
        flagged = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None)
                   in CALLS and decodes_text(n) and not any(k.arg == "errors" for k in n.keywords)]
        self.assertEqual(sorted(flagged), [2, 5])

    def test_undecodable_child_output_is_replaced_not_lost(self):
        # What the fix buys: a child that prints a byte invalid in UTF-8 (0x8a, an OEM-code-page letter).
        child = [sys.executable, "-I", "-c", "import sys; sys.stdout.buffer.write(b'non \\x8a riconosciuto')"]
        r = subprocess.run(child, capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.stdout, "non � riconosciuto")
        # Without errors= the output is lost. On POSIX run() raises UnicodeDecodeError. On Windows capture_output reads
        # through threads, so the error kills the reader thread (its traceback goes to stderr), and stdout comes back
        # as None with no exception at all, so the tool reads "no output".
        hook, threading.excepthook = threading.excepthook, lambda args: None
        try:
            try:
                lost = subprocess.run(child, capture_output=True, text=True, encoding="utf-8").stdout is None
            except UnicodeDecodeError:
                lost = True
        finally:
            threading.excepthook = hook
        self.assertTrue(lost)


if __name__ == "__main__":
    unittest.main()
