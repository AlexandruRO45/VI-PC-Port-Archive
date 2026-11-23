"""Tests for tools/gen_gdk_runtime_classes.py, driven by a hand-written docs tree (no clone, no game files).

    python -I tests/test_gen_gdk_runtime_classes.py
"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_README = """\
# CXTest

Class Id: `11111111-2222-3333-4444-555555555555`

Implements: [IXTest](IXTest.md)
"""

INTERFACE_PAGE = """\
# IXTest

Interface Id: `66666666-7777-8888-9999-AAAAAAAAAAAA`

Inherits: IUnknown

Layout:

- DoThing
- *unknown*
- DoOther

The paragraph after the list is not a method.
"""


def load_module():
    path = os.path.join(ROOT, "tools", "gen_gdk_runtime_classes.py")
    spec = importlib.util.spec_from_file_location("gen_gdk_runtime_classes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_module()


def write_docs(root):
    """A two-file COM tree: one class folder with its README and one interface page."""
    folder = os.path.join(root, "COM", "CXTest")
    os.makedirs(folder)
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(CLASS_README)
    with open(os.path.join(folder, "IXTest.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(INTERFACE_PAGE)
    return root


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.docs = write_docs(os.path.join(self.dir, "docs"))

    def test_class_and_interface_facts(self):
        table = gen.build(self.docs)
        self.assertEqual(table["classes"]["CXTest"], {
            "id": "11111111-2222-3333-4444-555555555555",
            "implements": ["IXTest"],
        })
        self.assertEqual(table["interfaces"]["IXTest"], {
            "id": "66666666-7777-8888-9999-aaaaaaaaaaaa",
            "class": "CXTest",
            "inherits": None,
            "methods": ["DoThing", None, "DoOther"],
        })

    def test_a_folder_without_com_is_an_oserror(self):
        with self.assertRaises(OSError):
            gen.build(self.dir)


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.docs = write_docs(os.path.join(self.dir, "docs"))

    def run_args(self, *args):
        err = io.StringIO()
        with mock.patch("sys.argv", ["gen_gdk_runtime_classes.py", *args]), \
                contextlib.redirect_stderr(err):
            code = gen.main()
        return code, err.getvalue()

    def test_out_parent_folder_is_created(self):
        out = os.path.join(self.dir, "build", "nested", "classes.json")
        code, err = self.run_args(self.docs, "--out", out)
        self.assertEqual((code, err), (0, ""))
        with open(out, encoding="utf-8", newline="") as f:
            text = f.read()
        self.assertNotIn("\r\n", text)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text)["classes"]["CXTest"]["implements"], ["IXTest"])

    def test_a_folder_named_out_is_refused(self):
        folder = os.path.join(self.dir, "named.json")
        os.makedirs(folder)
        with self.assertRaises(SystemExit):
            self.run_args(self.docs, "--out", folder)
        self.assertEqual(os.listdir(folder), [])

    def test_unwritable_out_is_one_error_line(self):
        blocker = os.path.join(self.dir, "not_a_folder")
        with open(blocker, "wb") as f:
            f.write(b"keep me")
        code, err = self.run_args(self.docs, "--out", os.path.join(blocker, "classes.json"))
        self.assertEqual(code, 2)
        self.assertIn("ERROR: cannot write", err)
        self.assertNotIn("Traceback", err)
        with open(blocker, "rb") as f:
            self.assertEqual(f.read(), b"keep me")

    def test_unreadable_docs_is_one_line(self):
        code, err = self.run_args(self.dir, "--out", os.path.join(self.dir, "classes.json"))
        self.assertEqual(code, 1)
        self.assertIn("cannot read the docs", err)
        self.assertNotIn("Traceback", err)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "classes.json")))


if __name__ == "__main__":
    unittest.main()
