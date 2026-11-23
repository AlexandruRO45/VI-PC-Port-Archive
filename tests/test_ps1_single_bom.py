"""PowerShell scripts start with at most one UTF-8 byte-order mark.

A second BOM is read as text: line 1 becomes the command named "\\ufeff#", so `param(...)` is no longer the
script's first statement and its parameters are lost. `scripts/build_runtime.ps1` shipped this way after 35616ce67,
so `-Debug`, `-FetchSdl3` and `-jobs` stopped working and every build printed two errors.
"""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BOM = b"\xef\xbb\xbf"
SKIP = {".git", "build", "third_party", "node_modules"}


def powershell_files():
    for path in sorted(ROOT.rglob("*.ps1")):
        if not SKIP.intersection(path.relative_to(ROOT).parts):
            yield path


class SingleBomTest(unittest.TestCase):
    def test_no_script_starts_with_two_boms(self):
        paths = list(powershell_files())
        self.assertTrue(paths, "found no .ps1 files")
        doubled = [p.relative_to(ROOT).as_posix() for p in paths if p.read_bytes().startswith(BOM * 2)]
        self.assertEqual(doubled, [], "remove the extra UTF-8 BOM at the start of these files")


if __name__ == "__main__":
    unittest.main()
