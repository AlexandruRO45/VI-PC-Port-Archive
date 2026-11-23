"""runtime/test/shaders/check.py: a skipped test is reported and counted as a skip, never as "ok".

    python -I tests/test_shader_check_skips.py

check.py's helpers return a Skip (a str) when the device can't run a test. Without a Vulkan driver
vk_texture_test exits 77 and prints nothing, and table_fetch_test used to return the program's output as that
Skip: an empty string, which main() printed as "ps_table_fetch ok" and which took one failure off the count, so
one real failure next to it ended the run with exit 0. No GPU, no build and no game data: the test programs are
replaced by stand-ins.
"""
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runtime", "test", "shaders"))
import check  # noqa: E402

UNSUPPORTED_INT64 = "device: X\nunsupported: shaderInt64 (a stage declares Int64)\n"


def program(code, output):
    """A stand-in for check.run: every run exits `code` with `output`, and one that exits 0 writes the readback it
    was given (run_readback fails a run that leaves none)."""
    def run(*cmd, env=None, cwd=None):
        if code == 0:
            for arg in cmd:
                if str(arg).endswith(".raw"):
                    with open(arg, "wb") as f:
                        f.write(bytes(64 * 64 * 16))
        return code, output
    return run


def table_fetch(code, output):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(check, "run", side_effect=program(code, output)):
        return check.table_fetch_test(tmp, tmp)


class TableFetchSkip(unittest.TestCase):
    def test_a_silent_exit_77_is_a_skip_with_text(self):
        result = table_fetch(77, "")
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported device)")

    def test_the_unsupported_line_is_quoted(self):
        result = table_fetch(77, UNSUPPORTED_INT64)
        self.assertIsInstance(result, check.Skip)
        self.assertEqual(result, "SKIP (unsupported: shaderInt64 (a stage declares Int64))")

    def test_other_exits_still_fail(self):
        result = table_fetch(1, "fetch-sint pixel 0 channel 0: got 0, expected -126\n")
        self.assertNotIsInstance(result, check.Skip)
        self.assertIn("exit 1", result)


class MainCountsSkips(unittest.TestCase):
    """check.py's main() over stand-in helpers: every one passes except those a test sets."""

    def run_main(self, overrides, code=77, output=""):
        helpers = {name: (lambda *args, **kwargs: "") for name, value in vars(check).items()
                   if isinstance(value, types.FunctionType) and name.endswith("_test") and name != "table_fetch_test"}
        helpers.update(overrides)
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            for name, value in helpers.items():
                stack.enter_context(mock.patch.object(check, name, value))
            # vk_store_test and vk_texture_test (table_fetch_test) are the programs main() and the real
            # table_fetch_test start themselves.
            stack.enter_context(mock.patch.object(check, "run", side_effect=program(code, output)))
            stack.enter_context(mock.patch.object(check, "pick_build_dir", side_effect=lambda build, programs: build))
            stack.enter_context(mock.patch.object(sys, "argv", ["check.py", tmp, tmp]))
            stack.enter_context(contextlib.redirect_stdout(out))
            status = check.main()
        return status, out.getvalue()

    def test_a_silent_skip_does_not_hide_a_failure(self):
        status, out = self.run_main({"fmt_1010102_test": lambda *args, **kwargs: "word 15: expected 0x1, got 0x2"})
        self.assertEqual(status, 1, out)
        self.assertIn("word 15: expected 0x1, got 0x2", out)
        self.assertRegex(out, r"(?m)^ps_table_fetch +SKIP \(unsupported device\)$")
        self.assertNotRegex(out, r"(?m)^ps_table_fetch +ok$")

    def test_an_empty_skip_from_any_helper_is_a_skip(self):
        status, out = self.run_main({"fmt_1010102_test": lambda *args, **kwargs: check.Skip("")}, 0,
                                    "fetch-sint: 4096 exact pixels\nfetch-uint: 4096 exact pixels\n"
                                    "fetch-miss: 4096 exact pixels\n0 bad\n0 bad\n0 bad\n")
        self.assertEqual(status, 0, out)
        self.assertRegex(out, r"(?m)^cs_fmt_1010102 +SKIP \(no reason given\)$")
        self.assertRegex(out, r"(?m)^ps_table_fetch +ok$")
        total = len([line for line in out.splitlines() if line.strip()]) - 1
        self.assertIn(f"{total - 1} of {total} pass, 1 skipped", out)

    def test_every_result_is_counted_once(self):
        status, out = self.run_main({"fmt_1010102_test": lambda *args, **kwargs: "word 0: expected 0x0, got 0x1",
                                     "fmt_sint32_test": lambda *args, **kwargs: check.Skip("SKIP (no Int64)")})
        rows = [line for line in out.splitlines() if line.strip()][:-1]
        skipped = sum(" SKIP (" in line for line in rows)
        self.assertEqual(status, 1, out)
        self.assertIn(f"{len(rows) - 1 - skipped} of {len(rows)} pass, {skipped} skipped", out)


if __name__ == "__main__":
    unittest.main()
