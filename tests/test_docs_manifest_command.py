"""A `manifest.py verify` command a docs page shows must name a manifest that is in the repository.

    python -I tests/test_docs_manifest_command.py

`docs/design.md` gives one command for checking a game copy against the package manifest, and it named no
manifest: the tool then falls back to `reports/manifest.json`, which `.gitignore` keeps out of the repository,
so on a fresh clone the reader's own check failed before it read a file. The page now names the tracked
`game/xbox-1.0.2.1.json`, and this keeps any page from going back to a path only a local run has.

The manifest is sizes and SHA-256 only, which the repository already ships, so nothing from a game copy is read
here.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -I keeps the script directory off sys.path.
sys.path.insert(0, os.path.join(ROOT, "tools"))
import manifest  # noqa: E402

VERIFY = re.compile(r"manifest\.py\s+verify\b")
MANIFEST_ARG = re.compile(r"--manifest\s+(\S+)")


def pages():
    """The prose files a reader is sent to: every docs note, then the two top-level pages."""
    docs = os.path.join(ROOT, "docs")
    for name in sorted(os.listdir(docs)):
        if name.endswith(".md"):
            yield os.path.join("docs", name)
    for name in ("README.md", "CONTRIBUTING.md"):
        if os.path.isfile(os.path.join(ROOT, name)):
            yield name


def manifest_of(line):
    """The manifest a `manifest.py verify` command on `line` reads, or None when the line is not one."""
    if not VERIFY.search(line):
        return None
    match = MANIFEST_ARG.search(line)
    return match.group(1) if match else manifest.DEFAULT


class DocsManifestCommandTest(unittest.TestCase):
    def test_every_documented_verify_command_names_a_repository_manifest(self):
        commands = 0
        for rel in pages():
            with open(os.path.join(ROOT, rel), encoding="utf-8") as page:
                lines = page.read().split("\n")
            for number, line in enumerate(lines, 1):
                named = manifest_of(line)
                if named is None:
                    continue
                commands += 1
                named = named.strip("`'\"")
                where = named if os.path.isabs(named) else os.path.join(ROOT, named)
                with self.subTest(page=f"{rel}:{number}", manifest=named):
                    self.assertFalse(named.startswith("<"), f"{rel}:{number} names a placeholder")
                    self.assertTrue(os.path.isfile(where),
                                    f"{rel}:{number} names {named}, which a fresh clone does not have")
                    self.assertTrue(manifest.load_manifest(where),
                                    f"{rel}:{number}: {named} lists no files")
        self.assertGreater(commands, 0, "no page shows a manifest.py verify command")


if __name__ == "__main__":
    unittest.main()
