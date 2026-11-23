"""A PowerShell script's usage line names every parameter whose default is a placeholder.

    python -I tests/test_ps1_usage_placeholders.py

A default such as "<ref>\\codewalker-gen9" is no real path, so a run that leaves that parameter out fails:
scripts/meta_to_xml.ps1's documented command stopped at its first Add-Type that way. This reads the scripts only, so no
PowerShell is needed.
"""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER_PARAM = re.compile(r'^\s*\[[A-Za-z]+\]\$(\w+)\s*=\s*"<[^"<>]*>', re.M)
USAGE = re.compile(r"^#\s+\.\\(\S+\.ps1)\b(.*)$", re.M)


def unnamed_placeholders(text, name):
    """The placeholder-default parameters that none of the script's own usage lines pass."""
    usage = [args for script, args in USAGE.findall(text) if script.lower() == name.lower()]
    if not usage:
        return []
    return [p for p in PLACEHOLDER_PARAM.findall(text)
            if not any(re.search(rf"(?<![\w-])-{p}\b", args, re.I) for args in usage)]


class Ps1UsagePlaceholders(unittest.TestCase):
    def test_the_rule(self):
        old = ('#   .\\meta_to_xml.ps1 -Carve <work>\\carve -Out <work>\\meta [-Limit N]\n'
               'param(\n    [string]$Carve = "<work>\\carve",\n    [string]$CodeWalker = "<ref>\\codewalker-gen9",\n'
               '    [int]$Limit = 0\n)\n')
        self.assertEqual(unnamed_placeholders(old, "meta_to_xml.ps1"), ["CodeWalker"])
        fixed = old.replace("[-Limit N]", "-CodeWalker <folder> [-Limit N]")
        self.assertEqual(unnamed_placeholders(fixed, "meta_to_xml.ps1"), [])

    def test_every_script(self):
        scripts = sorted(glob.glob(os.path.join(ROOT, "**", "*.ps1"), recursive=True))
        self.assertTrue(any(os.path.basename(p) == "meta_to_xml.ps1" for p in scripts))
        for path in scripts:
            with open(path, encoding="utf-8-sig") as f:
                text = f.read()
            with self.subTest(script=os.path.relpath(path, ROOT)):
                self.assertEqual(unnamed_placeholders(text, os.path.basename(path)), [])


if __name__ == "__main__":
    unittest.main()
