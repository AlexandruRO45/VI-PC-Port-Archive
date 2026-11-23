"""tools/gen_gdk_runtime_classes.py on a hand-written xgameruntime-docs tree (no network, no game files).

    python -I tests/test_gen_gdk_runtime_classes.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gen_gdk_runtime_classes as gen  # noqa: E402

CLASS_ID = "9e1b2d0c-2f3a-4b6d-9a1e-2b6d5a7c9f10"
USER_ID = "1f8f0c62-0d8b-4b4a-9e6d-5c2b1a0f3e77"
GAMERTAG_ID = "3b6a1d84-7c55-4b0e-8a2f-6d1c9b8e4a02"


def page(text):
    return f"# Page\n\n{text}\n"


class GdkRuntimeClasses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.docs = os.path.join(self.tmp.name, "xgameruntime-docs")
        com = os.path.join(self.docs, "COM", "XUser")
        os.makedirs(com)
        files = {
            "README.md": page(f"Class Id: `{CLASS_ID}`\n\nImplements: [IXUserImpl](IXUserImpl.md), "
                              "[IXUserGamertagImpl](IXUserGamertagImpl.md)"),
            "IXUserImpl.md": page(f"Interface Id: `{USER_ID}`\n\nInherits: IUnknown\n\nLayout:\n"
                                  "- [GetId]\n- *unknown*\n- GetGamertag\n\nThe paragraph after the list."),
            "IXUserGamertagImpl.md": page(f"Interface Id: `{GAMERTAG_ID}`\n\nInherits: [IXUserImpl](IXUserImpl.md)\n\n"
                                          "Layout changes unknown, all methods listed under its base."),
        }
        for name, text in files.items():
            with open(os.path.join(com, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_keeps_the_slot_of_an_unknown_method(self):
        with open(os.path.join(self.docs, "COM", "XUser", "IXUserImpl.md"), encoding="utf-8") as f:
            self.assertEqual(gen.parse_layout(f.read()), ["GetId", None, "GetGamertag"])

    def test_interface_without_a_layout_has_none_and_keeps_its_base(self):
        classes, interfaces = gen.parse_docs(self.docs)
        self.assertEqual(interfaces["IXUserGamertagImpl"]["inherits"], "IXUserImpl")
        self.assertIsNone(interfaces["IXUserGamertagImpl"]["methods"])

    def test_class_lists_the_interfaces_it_implements(self):
        classes, _ = gen.parse_docs(self.docs)
        self.assertEqual(classes["XUser"], {"id": CLASS_ID,
                                            "implements": ["IXUserImpl", "IXUserGamertagImpl"]})

    def test_check_reports_a_link_with_no_page(self):
        classes, interfaces = gen.parse_docs(self.docs)
        classes["XUser"]["implements"].append("IXUserMissingImpl")
        self.assertEqual(gen.check(classes, interfaces), ["XUser implements IXUserMissingImpl, which has no page"])

    def test_missing_com_folder_says_what_it_wanted(self):
        with self.assertRaises(FileNotFoundError) as raised:
            gen.parse_docs(self.tmp.name)
        self.assertIn("COM folder", str(raised.exception))

    def test_out_creates_a_missing_folder(self):
        out = os.path.join(self.tmp.name, "reports", "deep", "classes.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen.main([self.docs, "--out", out]), 0)
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        table = json.loads(raw)
        self.assertEqual(table["interfaces"]["IXUserImpl"]["id"], USER_ID)
        self.assertEqual(table["interfaces"]["IXUserImpl"]["methods"], ["GetId", None, "GetGamertag"])


if __name__ == "__main__":
    unittest.main()
