"""The xgameruntime environment-override table lists exactly the variables the shim reads.

    python -I tests/test_docs_runtime_env_table.py

`tests/test_docs_env_vars.py` checks that every `LEONIDA_*` variable is named in *some* tracked Markdown page.
That still lets `runtime/README.md`'s "environment overrides" table drift from the code: a new override can be
read by `runtime/shims/xgameruntime.cpp` without a row, or a row can outlive the variable it documents. This
compares the table's first column with the string literals the shim reads, in both directions.

`LEONIDA_GAME_DIR` is read in the same file but sets the game folder rather than changing an answer, so it is
documented further down the page and is not part of the table. Standard library only; no game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = "runtime/README.md"
SHIM = "runtime/shims/xgameruntime.cpp"
HEADING = "### What `xgameruntime.dll` reports: environment overrides"
# A variable name as a string literal: "LEONIDA_X", L"LEONIDA_X" or 'LEONIDA_X'.
LITERAL = re.compile(r"""["']((?:LEONIDA|GTA6RT)_[A-Z0-9_]+)["']""")
ROW = re.compile(r"^\|\s*`((?:LEONIDA|GTA6RT)_[A-Z0-9_]+)`\s*\|", re.M)
# Read by the shim, but its job is the game folder: documented under "What xgameruntime.dll reports" below, not
# in the override table.
NOT_AN_ANSWER = {"LEONIDA_GAME_DIR"}


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as f:
        return f.read()


def override_section(text):
    """The override section's text, from its heading to the next `##`/`###` heading."""
    start = text.find(HEADING)
    if start < 0:
        return ""
    start += len(HEADING)
    end = re.search(r"\n#{2,3} ", text[start:])
    return text[start:start + end.start()] if end else text[start:]


def table_variables(text):
    """The variables named in the first column of the override table, in order."""
    found = ROW.findall(override_section(text))
    return list(dict.fromkeys(found))


def shim_variables(text):
    return sorted(set(LITERAL.findall(text)) - NOT_AN_ANSWER)


def mismatches(page_text, code_text):
    """(read but not tabulated, tabulated but not read), sorted."""
    table = set(table_variables(page_text))
    read_by_shim = set(shim_variables(code_text))
    return sorted(read_by_shim - table), sorted(table - read_by_shim)


class OverrideTableTest(unittest.TestCase):
    def test_the_checker_reports_both_directions(self):
        page = (HEADING + "\n\n| variable | default |\n|---|---|\n"
                "| `LEONIDA_ONE` | `1` | x |\n"
                "| `LEONIDA_TWO` | unset | y |\n\n## Next section\n")
        code = ('GetEnvironmentVariableA("LEONIDA_ONE", s, n);\n'
                'GetEnvironmentVariableA("LEONIDA_TWO", s, n);\n'
                'GetEnvironmentVariableA("LEONIDA_THREE", s, n);\n'
                'GetEnvironmentVariableW(L"LEONIDA_GAME_DIR", s, n);\n')
        self.assertEqual(mismatches(page, code), (["LEONIDA_THREE"], []))

        page_extra = page.replace("`LEONIDA_TWO`", "`LEONIDA_FOUR`")
        self.assertEqual(mismatches(page_extra, code), (["LEONIDA_THREE", "LEONIDA_TWO"], ["LEONIDA_FOUR"]))

        # A row outside the section does not count.
        outside = page.replace(HEADING, "### Something else") + "| `LEONIDA_THREE` | x |\n"
        self.assertEqual(table_variables(outside), [])

    def test_the_section_and_scan_find_the_known_rows(self):
        table = table_variables(read(README))
        self.assertEqual(table, ["LEONIDA_DEVICE_TYPE", "LEONIDA_ANALYTICS_FORM", "LEONIDA_PACKAGE_ID",
                                 "LEONIDA_CONTENT_PACKAGES", "LEONIDA_PACKAGED"])
        self.assertGreaterEqual(len(shim_variables(read(SHIM))), 5)

    def test_the_table_matches_the_shim(self):
        missing, extra = mismatches(read(README), read(SHIM))
        self.assertEqual((missing, extra), ([], []),
                         "read by runtime/shims/xgameruntime.cpp but not in the runtime/README.md table: "
                         + ", ".join(missing) + "; in the table but not read by the shim: " + ", ".join(extra))


if __name__ == "__main__":
    unittest.main()
