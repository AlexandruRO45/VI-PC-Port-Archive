"""Tests for tools/gen_gdk_runtime_classes.py, driven by a hand-written docs tree (no clone, no game files).

    python -I tests/test_gen_gdk_runtime_classes.py
"""
import importlib.util
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
