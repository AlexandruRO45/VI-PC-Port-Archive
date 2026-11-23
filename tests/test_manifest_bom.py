"""Both integrity CLIs accept Windows UTF-8 BOMs without weakening verification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1] / "tools"


class ManifestBomTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.package.mkdir()
        self.payload = self.package / "sample.bin"
        self.payload.write_bytes(b"synthetic payload")
        self.manifest = self.root / "manifest.json"
        self.document = {"files": {"sample.bin": {
            "size": self.payload.stat().st_size,
            "sha256": hashlib.sha256(self.payload.read_bytes()).hexdigest(),
        }}}

    def run_case(self, document, expected_code):
        for encoding in ("utf-8", "utf-8-sig"):
            self.manifest.write_text(document, encoding=encoding, newline="\n")
            before = (self.manifest.read_bytes(), self.payload.read_bytes())
            for tool in ("manifest.py", "manifest_verifier.py"):
                for jobs in (1, 2):
                    with self.subTest(encoding=encoding, tool=tool, jobs=jobs):
                        result = subprocess.run(
                            [sys.executable, "-I", str(TOOLS / tool), "verify", str(self.package),
                             "--manifest", str(self.manifest), "--jobs", str(jobs)],
                            capture_output=True, text=True, errors="replace", timeout=30)
                        self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                        self.assertNotIn("Traceback", result.stderr)
                        if expected_code == 0:
                            self.assertIn("OK", result.stdout)
                        elif expected_code == 1:
                            self.assertIn("CHANGED", result.stdout)
                        else:
                            self.assertIn("ERROR:", result.stderr)
                        self.assertEqual(before, (self.manifest.read_bytes(), self.payload.read_bytes()))

    def test_matching_files_verify_with_or_without_bom(self):
        self.run_case(json.dumps(self.document), 0)

    def test_same_size_corruption_still_fails(self):
        self.payload.write_bytes(b"Synthetic payload")
        self.run_case(json.dumps(self.document), 1)

    def test_malformed_json_still_errors(self):
        self.run_case('{"files":', 2)

    def test_duplicate_keys_still_error(self):
        self.run_case('{"files": {}, "files": {}}', 2)


if __name__ == "__main__":
    unittest.main()
