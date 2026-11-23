"""The fast paths in gamedata_index.py must give the same answers as the plain code they replaced.

    python -I tests/test_gamedata_index_fast_paths.py

Each test compares the tool with a reference written out in this file, on synthetic bytes only.
"""
import importlib.util
import os
import random
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gamedata_index.py")


def load_tool():
    spec = importlib.util.spec_from_file_location("gamedata_index_fast_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_printable(data):
    sample = data[:4096]
    if not sample:
        return 0.0
    good = sum(1 for c in sample if c in (9, 10, 13) or 32 <= c < 127)
    return good / len(sample)


def reference_xml_complete(tool, root, data):
    if not root:
        return False
    text = tool.XML_NON_TAG.sub(b"", data)
    name = re.escape(root.encode("ascii", "replace"))
    opens = re.findall(rb"<" + name + rb"(?=[\s/>])(?:\"[^\"]*\"|'[^']*'|[^'\">])*>", text)
    closes = re.findall(rb"</" + name + rb"\s*>", text)
    self_closed = sum(1 for tag in opens if tag.rstrip().endswith(b"/>"))
    if len(opens) - self_closed > len(closes):
        return False
    return bool(closes or self_closed)


XML_CASES = [
    b"",
    b"<a/>",
    b"<a />",
    b"<a></a>",
    b"<a><a></a>",
    b"<a><a></a></a>",
    b"<a x='1'/><a y=\"2\"></a>",
    b"<a x=\"1>2\"/>",
    b"<a x=\"1>2\">",
    b"<a x=\"1>2\"></a>",
    b"<a x='unterminated>",
    b"<a x=\"unterminated>",
    b"<ab></ab><a>",
    b"<a/ >",
    b"<a\n/>",
    b"<a\t></a  >",
    b"<a><b/><a/></a>",
    b"<a/><a/><a/>",
    b"</a>",
    b"<a",
    b"<a>",
    b"<a/>></a>",
    b"<?xml version='1.0'?><a><!-- <a> --></a>",
]


class FastPathParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = load_tool()

    def test_printable_matches_reference_on_every_byte_value(self):
        for value in range(256):
            data = bytes([value]) * 7
            self.assertEqual(self.tool.printable(data), reference_printable(data), value)
        self.assertEqual(self.tool.printable(b""), 0.0)

    def test_printable_matches_reference_on_random_and_long_input(self):
        rng = random.Random(20240601)
        for size in (1, 2, 17, 4095, 4096, 4097, 9000):
            for density in (0.0, 0.3, 0.5, 0.97, 1.0):
                data = bytes(
                    rng.randrange(32, 127) if rng.random() < density else rng.randrange(256)
                    for _ in range(size)
                )
                self.assertEqual(self.tool.printable(data), reference_printable(data), (size, density))

    def test_printable_only_looks_at_the_first_4096_bytes(self):
        data = b"A" * 4096 + b"\x00" * 100
        self.assertEqual(self.tool.printable(data), 1.0)

    def test_xml_complete_matches_reference_on_fixed_cases(self):
        for root in ("a", "ab", "b", ""):
            for case in XML_CASES:
                with self.subTest(root=root, case=case):
                    self.assertEqual(
                        self.tool.xml_complete(root, case), reference_xml_complete(self.tool, root, case)
                    )

    def test_xml_complete_matches_reference_on_random_documents(self):
        rng = random.Random(7)
        pieces = [
            b"<a>", b"</a>", b"<a/>", b"<a />", b"<a x=\"1>2\"/>", b"<a x='>'>", b"<ab>", b"</ab>",
            b"<b/>", b" ", b"\n", b"text", b"<a", b">", b"\"", b"'", b"/", b"<?xml?>", b"\x00\xff",
        ]
        for _ in range(400):
            data = b"".join(rng.choice(pieces) for _ in range(rng.randrange(0, 14)))
            for root in ("a", "ab"):
                self.assertEqual(
                    self.tool.xml_complete(root, data), reference_xml_complete(self.tool, root, data), data
                )

    def test_xml_complete_handles_many_distinct_roots(self):
        # The compiled-pattern cache is bounded; more roots than it holds must still give right answers.
        for i in range(1500):
            root = f"r{i}"
            doc = f"<{root}></{root}>".encode()
            self.assertTrue(self.tool.xml_complete(root, doc))
            self.assertFalse(self.tool.xml_complete(root, f"<{root}>".encode()))


if __name__ == "__main__":
    unittest.main()
