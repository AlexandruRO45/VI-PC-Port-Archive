"""The two hand-written text writers that only a run reaches end their lines in LF (BUILD-468).

`tools/wave_diff.py` writes a `.asm` case and its store prologue, `tools/leonida_tui.py` appends a stage log line by
line. Neither form shows up in `test_text_write_line_endings.py`'s `ast` scan (`newline=` is simply missing), so only
that test failed: a Windows run of either tool wrote a CRLF `.asm` case and a CRLF log where every other tool writes
LF. On Linux the same two writes produce LF whether or not `newline=` is passed, so the test replaces the builtin
`open` with Windows' text-mode rules (`newline=None` means `\r\n`) and compares raw bytes here.

    python -I tests/test_text_write_bytes.py

Standard library only; in-memory files, no game data, no GPU.
"""
import builtins
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)

import leonida_tui  # noqa: E402
import wave_diff  # noqa: E402


class WindowsTextFile:
    """A file object that translates like Windows: an unset `newline` is `\r\n`, and `\r\n` becomes `\n` on read."""

    def __init__(self, path, mode, encoding, errors, newline, written):
        self.path = path
        self.newline = "\r\n" if newline is None else newline
        self.encoding = encoding or "utf-8"
        self.errors = errors or "strict"
        self.written = written
        self.buffer = bytearray(b"")
        if any(c in mode for c in "ar+") and os.path.exists(path):
            with open(path, "rb") as f:
                existing = f.read().decode(self.encoding, self.errors)
            self.buffer = bytearray(existing.replace("\r\n", "\n").encode(self.encoding))

    def write(self, text):
        self.buffer += text.replace("\n", self.newline).encode(self.encoding, self.errors)
        return len(text)

    def flush(self):
        pass

    def close(self):
        self.written[self.path] = bytes(self.buffer)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def windows_open(written):
    """A stand-in for the builtin `open`: same signatures, Windows' line-ending rules, bytes kept in `written`."""

    def opener(path, mode="r", *args, **kwargs):
        path = os.fspath(path)
        if "b" in mode:
            with open(path, mode, *args, **kwargs) as f:
                written[path] = f.read()
            return io.BytesIO(written[path])
        return WindowsTextFile(path, mode, kwargs.get("encoding"), kwargs.get("errors"), kwargs.get("newline"),
                               written)

    return opener


class RawBytes(unittest.TestCase):
    def setUp(self):
        self.written = {}
        patcher = mock.patch.object(builtins, "open", windows_open(self.written))
        patcher.start()
        self.addCleanup(patcher.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = os.path.realpath(tmp.name)

    def read(self, path):
        self.assertIn(path, self.written, "the writer closed nothing")
        return self.written[path]

    def test_a_wave_diff_case_keeps_lf(self):
        path = os.path.join(self.dir, "cases", "case.asm")
        wave_diff._write(path, ["v_mov_b32 v0, 0"], stores=True)
        text = self.read(path)
        self.assertNotIn(b"\r\n", text)
        self.assertTrue(text.startswith(b"; wave_diff case"))
        self.assertIn(b"\tv_mov_b32 v0, 0\n", text)
        self.assertTrue(text.endswith(b"\ts_endpgm\n"))

    def test_a_stage_log_line_keeps_lf(self):
        log = os.path.join(self.dir, "logs", "shaders.log")
        console = leonida_tui.StageConsole("shaders", log)
        self.addCleanup(console.close)
        console.write_line("started")
        console.write_line("done")
        console.close()   # the log is only read back once it is closed; the cleanup close is then a no-op
        self.assertEqual(self.read(log), b"started\ndone\n")


if __name__ == "__main__":
    unittest.main()
