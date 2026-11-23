"""Synthetic value-validation tests for tools/rbf0_encode.py. No game data."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import rbf0_encode  # noqa: E402
import rbf0_tree  # noqa: E402


def _doc_with_child(child: dict) -> dict:
    return {
        "status": "ok",
        "layout": "narrow-32",
        "descriptors": [
            {"index": 0, "name": "Root"},
            {"index": 1, "name": child["name"]},
        ],
        "tree": {
            "kind": "element",
            "name": "Root",
            "descriptor": 0,
            "attrs_declared": 0,
            "attributes": [],
            "content": [child],
        },
    }


class TestRbf0EncodeValueValidation(unittest.TestCase):
    def test_malformed_vec3_float_special_rejected_as_encode_error(self):
        for bad_special in (123, "bad", [], ["inf"], {"0": "inf", "1": "inf", "2": "inf"}):
            child = {
                "kind": "value",
                "name": "pos",
                "descriptor": 1,
                "type": "vec3",
                "value": [1.0, 2.0, 3.0],
                "float_special": bad_special,
            }
            with self.subTest(float_special=bad_special):
                with self.assertRaises(rbf0_encode.EncodeError) as ctx:
                    rbf0_encode.encode(_doc_with_child(child))
                self.assertEqual(ctx.exception.reason, "value")

    def test_valid_vec3_float_special_roundtrips(self):
        child = {
            "kind": "value",
            "name": "pos",
            "descriptor": 1,
            "type": "vec3",
            "value": [1.0, None, -2.0],
            "float_special": [None, "inf", None],
        }
        blob = rbf0_encode.encode(_doc_with_child(child))
        parsed = rbf0_tree.parse_resource(blob)
        self.assertEqual(parsed["status"], "ok")
        item = parsed["tree"]["content"][0]
        self.assertEqual(item["float_special"], ["1.0", "inf", "-2.0"])

    def test_bool_and_non_int_string_byte_length_rejected(self):
        for bad_len in (True, False, "1", 1.0):
            child = {
                "kind": "value",
                "name": "tag",
                "descriptor": 1,
                "type": "string",
                "value": "a",
                "byte_length": bad_len,
            }
            with self.subTest(byte_length=bad_len):
                with self.assertRaises(rbf0_encode.EncodeError) as ctx:
                    rbf0_encode.encode(_doc_with_child(child))
                self.assertEqual(ctx.exception.reason, "length")


if __name__ == "__main__":
    unittest.main()
