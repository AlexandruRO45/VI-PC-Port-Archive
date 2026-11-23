"""CPU-only tests for tools/buffer_format_fuzz.py's command line: an explicit --formats is checked before the GPU.

    python -I tests/test_buffer_format_fuzz_cli.py

A typo in --formats used to be found only after runner.open_oracle(), so a machine without the oracle answered
"hardware oracle unavailable" (77) and one with it raised SystemExit (1). It is a usage error (2) before the oracle.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "buffer_format_fuzz.py")


def run(*args):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, check=False)


def load_tool():
    spec = importlib.util.spec_from_file_location("buffer_format_fuzz", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BufferFormatFuzzFormatsTests(unittest.TestCase):
    def test_unknown_format_is_a_usage_error_before_the_oracle(self):
        proc = run("--oracle", "hw", "--formats", "8_8_8_8_UNORM_typo")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '8_8_8_8_UNORM_typo'", proc.stderr)
        self.assertNotIn("hardware oracle", proc.stdout + proc.stderr)

    def test_a_code_outside_the_table_is_refused_too(self):
        proc = run("--oracle", "hw", "--formats", "0,8_8_8_8_UNORM")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '0'", proc.stderr)

    def test_known_names_and_codes_map_to_the_table(self):
        tool = load_tool()
        pairs = tool.select_formats("8_8_8_8_UNORM,56")
        self.assertEqual([n for _, n in pairs], ["8_8_8_8_UNORM", tool.GFX10_NAMES[56]])
        with self.assertRaises(ValueError):
            tool.select_formats("not_a_format")


class BufferFormatFuzzBlankSlotsTests(unittest.TestCase):
    """A blank slot between commas names no format: like every other comma-list parser in tools/
    (bcpack_history_calib.parse_codecs, buffer_store_probe.parse_waves) it is dropped, and a spec
    that held nothing but commas is a usage error -- not "unknown format ''"."""

    def test_blank_slots_give_the_same_selection_as_the_plain_list(self):
        tool = load_tool()
        want = tool.select_formats("56")
        for spec in ("56,", ",56", "56,,", " 56 ", "56, ,"):
            self.assertEqual(tool.select_formats(spec), want, spec)

    def test_a_trailing_comma_is_not_a_usage_error(self):
        proc = run("--oracle", "hw", "--formats", "8_8_8_8_UNORM,")
        self.assertNotEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertNotIn("unknown format", proc.stderr)

    def test_a_spec_of_only_commas_is_a_usage_error_before_the_oracle(self):
        proc = run("--oracle", "hw", "--formats", ",,")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("--formats must not be empty", proc.stderr)
        self.assertNotIn("hardware oracle", proc.stdout + proc.stderr)

    def test_the_empty_default_and_a_bad_neighbour_still_behave(self):
        tool = load_tool()
        # "" is the default: every code the family has (gfx10: all 77, gfx11: the 63 with a GFX11 spelling)
        self.assertEqual(len(tool.select_formats("")), 63)
        self.assertEqual(len(tool.select_formats("", "gfx10")), 77)
        self.assertEqual(tool.select_formats("56,"), [(56, tool.GFX10_NAMES[56])])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            tool.select_formats(" , ")
        with self.assertRaisesRegex(ValueError, "unknown format 'nope'"):
            tool.select_formats("56,nope,")


if __name__ == "__main__":
    unittest.main()
