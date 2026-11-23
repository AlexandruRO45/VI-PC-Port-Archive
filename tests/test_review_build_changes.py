"""Exercise the build-script report against a synthetic two-commit repository."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import quote


REPORT = Path(__file__).resolve().parents[1] / "scripts" / "review_build_changes.py"


class BuildReviewTest(unittest.TestCase):
    def test_only_added_build_lines_are_linked_and_source_is_never_echoed(self):
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(["git", "-c", "user.name=Test", "-c",
                                                "user.email=test@example.invalid", "-c",
                                                "commit.gpgsign=false", *args], cwd=directory)

            root = Path(directory)
            tools = root / "tools"
            tools.mkdir()
            name = "odd [review](1).py"
            path = tools / name
            path.write_bytes(b"old = Path.home()\nkeep = 1\n")
            git("init", "-q")
            git("add", ".")
            git("commit", "-qm", "base")
            path.write_bytes(b"keep = 1\nurlopen('PRIVATE_SOURCE')\nvalue = Path.home()\n")
            (root / "Build.bat").write_bytes(b"curl.exe example.invalid\n")
            (root / "notes.md").write_bytes(b"curl example.invalid\n")
            git("add", ".")
            git("commit", "-qm", "head")
            head = git("rev-parse", "HEAD").decode().strip()
            url = "https://example.invalid/project"

            def report(base):
                return subprocess.check_output([sys.executable, "-I", os.fspath(REPORT),
                                                base, head, url], cwd=root, text=True)

            text = report("HEAD^")
            link = f"{url}/blob/{head}/tools/{quote(name)}#L"
            self.assertIn(link + "2)", text)
            self.assertIn(link + "3)", text)
            self.assertIn("Build.bat#L1)", text)
            self.assertEqual(text.count("\n- "), 3)
            self.assertNotIn("PRIVATE_SOURCE", text)
            self.assertNotIn("notes.md", text)
            self.assertIn("No added lines matched", report(head))


if __name__ == "__main__":
    unittest.main()
