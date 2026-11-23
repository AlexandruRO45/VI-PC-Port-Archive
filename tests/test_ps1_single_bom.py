"""PowerShell scripts start with at most one UTF-8 byte-order mark.

    python -I tests/test_ps1_single_bom.py

Windows PowerShell 5.1 reads a .ps1 with one BOM as UTF-8, but a second BOM after it becomes the text of the
first token: `powershell -File scripts\\build_runtime.ps1` then stops at line 1 with "'ï»¿#' is not recognized"
and the build never starts. A merge brought that second BOM back into build_runtime.ps1 (561b65a8). Standard
library only.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BOM = b"\xef\xbb\xbf"


class SingleBom(unittest.TestCase):
    def test_no_script_has_a_bom_after_its_first_bytes(self):
        scripts = sorted(ROOT.glob("**/*.ps1"))
        self.assertTrue(scripts, "no .ps1 files found")
        for path in scripts:
            if any(part in (".git", "build", "node_modules") for part in path.relative_to(ROOT).parts):
                continue
            with self.subTest(script=path.relative_to(ROOT).as_posix()):
                data = path.read_bytes()
                body = data[len(BOM):] if data.startswith(BOM) else data
                self.assertNotIn(BOM, body, "a UTF-8 BOM after the start of the file")


if __name__ == "__main__":
    unittest.main()
