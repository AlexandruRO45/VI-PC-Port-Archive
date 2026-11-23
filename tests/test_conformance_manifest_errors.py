"""Input diagnostics from the conformance comparator CLI, using synthetic JSON."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


COMPARE = Path(__file__).resolve().parents[1] / "tools" / "conformance" / "compare.py"


class ManifestErrors(unittest.TestCase):
    def test_integer_beyond_the_decimal_limit_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_bytes(
                b'{"renderer":"fixture","adapter":"cpu","scenes":[{"name":"pixel",'
                b'"png":"image.png","checks":[{"x":' + b"1" * 641 + b',"y":0,"r":1,"g":2,"b":3,"a":255}]}]}'
            )
            result = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", "-X", "int_max_str_digits=640",
                 str(COMPARE), "expect", "--manifest", str(manifest)],
                capture_output=True, encoding="utf-8", timeout=15,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(str(manifest), result.stderr)
        self.assertIn("cannot decode a JSON value", result.stderr)
        self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)


if __name__ == "__main__":
    unittest.main()
