"""Synthetic unit tests for leading text/value records outside an open element in tools/rbf0_index.py."""
from __future__ import annotations

import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import rbf0_index  # noqa: E402
import rbf0_tree  # noqa: E402


def _desc(index: int, kind: int, name: bytes) -> bytes:
    return bytes((index, kind)) + struct.pack("<H", len(name)) + name


def _elem(index: int, name: bytes | None, attrs: int = 0) -> bytes:
    head = _desc(index, 0x00, name) if name is not None else bytes((index, 0x00))
    return head + struct.pack("<HHH", 0, 0, attrs)


def _text(payload: bytes) -> bytes:
    return b"\xfd\xff" + struct.pack("<I", len(payload)) + payload


def _valid_root(wide: bool = False) -> bytes:
    prefix = rbf0_index.MAGIC + (rbf0_index.PROLOGUE if wide else b"")
    int_bytes = struct.pack("<Q" if wide else "<I", 42)
    return (
        prefix
        + _elem(0, b"Root", attrs=1)
        + _desc(1, 0x10, b"count") + int_bytes
        + _text(b"hello\x00")
        + b"\xff\xff"
    )


class TestRbf0IndexRootBounds(unittest.TestCase):
    def test_text_outside_element_rejected_like_rbf0_tree(self):
        blob = rbf0_index.MAGIC + _text(b"stray\x00") + _elem(0, b"Root") + b"\xff\xff"
        idx = rbf0_index.parse_rbf0(blob)
        tree = rbf0_tree.parse_resource(blob)
        self.assertEqual(idx["status"], "bad")
        self.assertFalse(idx["complete"])
        self.assertIn("text outside an element", idx["error"])
        self.assertEqual(idx["status"], tree["status"])

    def test_value_outside_element_rejected_like_rbf0_tree(self):
        cases = (
            ("int", _desc(0, 0x10, b"x") + struct.pack("<I", 7)),
            ("bool", _desc(0, 0x20, b"flag")),
            ("string", _desc(0, 0x60, b"s") + struct.pack("<H", 2) + b"ok"),
        )
        for label, record in cases:
            with self.subTest(kind=label):
                blob = rbf0_index.MAGIC + record + _elem(1, b"Root") + b"\xff\xff"
                idx = rbf0_index.parse_rbf0(blob)
                tree = rbf0_tree.parse_resource(blob)
                self.assertEqual(idx["status"], "bad")
                self.assertFalse(idx["complete"])
                self.assertIn("value outside an element", idx["error"])
                self.assertEqual(idx["status"], tree["status"])

    def test_valid_narrow_and_wide_resources_still_parse(self):
        for wide in (False, True):
            with self.subTest(wide=wide):
                res = rbf0_index.parse_rbf0(_valid_root(wide=wide))
                self.assertEqual(res["status"], "ok")
                self.assertTrue(res["complete"])
                self.assertEqual(res["root"], "Root")
                self.assertEqual(res["ints"], [42])
                self.assertEqual(res["texts"], 1)


if __name__ == "__main__":
    unittest.main()
