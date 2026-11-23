"""Tests for tools/shader_stage_manifest.py shader_stems.

Covers the RecursionError guard: a deeply nested index.json must raise a clean
ValueError instead of an uncaught traceback. CPU-only, no game files, no GPU.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from shader_stage_manifest import shader_stems  # noqa: E402


def write_deep_json(path, depth):
    """Write a JSON array nested `depth` levels deep."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("[" * depth + "]" * depth)


class ShaderStemsRecursionError(unittest.TestCase):
    """Deeply nested JSON must raise ValueError, not RecursionError."""

    def test_deeply_nested_json_raises_valueerror(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.json")
            write_deep_json(path, 100000)
            with self.assertRaises(ValueError) as ctx:
                shader_stems(path, "file")
            self.assertIn("too deeply nested", str(ctx.exception))

    def test_normal_json_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump([{"file": "ps_main.s"}, {"file": "vs_main.s"}], f)
            self.assertEqual(shader_stems(path, "file", ".s"), {"ps_main", "vs_main"})


if __name__ == "__main__":
    unittest.main()
