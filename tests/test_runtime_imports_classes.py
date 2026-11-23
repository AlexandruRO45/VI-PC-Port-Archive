"""The class table in docs/runtime_imports.md says which GDK runtime classes the shims serve.

    python -I tests/test_runtime_imports_classes.py

The "The retail executable's runtime classes" section marks each class the executable reaches as served or not, and
the paragraph under the table names the classes still unserved. Both are answers about `runtime/shims`, which changes
without the page, so this compares them with the same reading `tools/gdk_runtime_calls.py` uses: every
`constexpr GUID k...Class` in `runtime/shims/xgameruntime*.cpp` is a class `QueryApiImpl` serves. The checker is also
exercised on a table with one wrong cell, so a check that accepts everything cannot pass.

Reads tracked files only; no game files, no GPU, standard library only.
"""
import json
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIMS = os.path.join(ROOT, "runtime", "shims")
PAGE = os.path.join(ROOT, "docs", "runtime_imports.md")
SECTION = "The retail executable's runtime classes"

ROW = re.compile(r"^\|\s*`(X\w+Impl)`\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(yes|no)\s*\|", re.M)
UNSERVED = re.compile(r"The (?:one|two|three|four|five) that (?:is|are) not:([^.]*)\.", re.S)
CLASS_NAME = re.compile(r"`(X\w+Impl)`")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def section(text):
    """The part of the page from the class-table heading to the next heading."""
    start = text.find(SECTION)
    if start < 0:
        return ""
    end = text.find("\n### ", start + 1)
    return text[start:end if end >= 0 else len(text)]


def table(text):
    """{class name: the Served cell is yes} for every row of the class table."""
    return {m.group(1): m.group(2) == "yes" for m in ROW.finditer(text)}


def shim_classes():
    """The class ids the shim sources serve, read exactly as tools/gdk_runtime_calls.py reads them."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from gdk_runtime_calls import served_classes  # noqa: E402  (the tool this page's numbers come from)
    ids = set()
    for name in sorted(os.listdir(SHIMS)):
        if name.startswith("xgameruntime") and name.endswith(".cpp"):
            ids.update(served_classes(read(os.path.join(SHIMS, name))))
    return ids


def class_ids():
    """{class name: class id} from the class table the tool uses."""
    with open(os.path.join(ROOT, "tools", "gdk_runtime_classes.json"), encoding="utf-8") as f:
        return {name: cls["id"] for name, cls in json.load(f)["classes"].items()}


def problems(rows, ids, served):
    """One line per row whose Served cell contradicts the shims, or that names no known class."""
    out = []
    for name, marked in rows.items():
        if name not in ids:
            out.append(f"{name}: not a class in tools/gdk_runtime_classes.json")
        elif marked != (ids[name] in served):
            out.append(f"{name}: the page says {'served' if marked else 'not served'}, the shims "
                       f"{'serve' if ids[name] in served else 'do not serve'} it")
    return out


class RuntimeClassTableTest(unittest.TestCase):
    def setUp(self):
        self.rows = table(section(read(PAGE)))
        self.ids = class_ids()
        self.served = shim_classes()

    def test_the_page_has_its_table(self):
        self.assertTrue(self.rows, f"{SECTION}: no class table rows found in docs/runtime_imports.md")
        self.assertTrue(self.served, "runtime/shims serves no class at all; the shim scan found nothing")

    def test_the_served_column_matches_the_shims(self):
        self.assertEqual(problems(self.rows, self.ids, self.served), [])

    def test_the_unserved_sentence_names_the_unserved_rows(self):
        sentence = UNSERVED.search(section(read(PAGE)))
        self.assertIsNotNone(sentence, "the class table needs its 'The one that is not: ...' sentence")
        named = sorted(set(CLASS_NAME.findall(sentence.group(1))))
        unserved = sorted(name for name, marked in self.rows.items() if not marked)
        self.assertEqual(named, unserved)

    def test_the_checker_catches_a_wrong_cell(self):
        ids = {"XServedImpl": "11111111-1111-1111-1111-111111111111"}
        self.assertEqual(problems({"XServedImpl": True}, ids, set(ids.values())), [])
        self.assertNotEqual(problems({"XServedImpl": False}, ids, set(ids.values())), [])
        self.assertNotEqual(problems({"XUnknownImpl": True}, ids, set(ids.values())), [])


if __name__ == "__main__":
    unittest.main()
