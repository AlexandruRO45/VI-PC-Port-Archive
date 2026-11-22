"""Self-authored XML only; no game files, GPU or third-party packages."""
import io
import json
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from xml.parsers import expat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from gamedata_reader import XmlReadError, file_chunks, read_xml


class ReaderTests(unittest.TestCase):
    def test_routes_and_mixed_content(self):
        events = []
        def sink(route):
            return lambda e: events.append((route, e))
        doc = b'<Config><Item type="number">12<Part/>tail</Item><Item type=""/><Other/></Config>'
        result = read_xml([doc], {("Config", "number"): sink("number"),
                                 ("Config", ""): sink("empty"),
                                 ("Config", None): sink("root")}, sink("default"))
        self.assertEqual((result.elements, result.typed_elements, result.max_depth), (5, 2, 3))
        self.assertEqual([(r, e.tag) for r, e in events if e.kind == "start"],
                         [("root", "Config"), ("number", "Item"), ("root", "Part"),
                          ("empty", "Item"), ("root", "Other")])
        self.assertEqual([(r, e.text) for r, e in events if e.kind == "text"],
                         [("number", "12"), ("number", "tail")])
        self.assertEqual([(r, e.tag) for r, e in events if e.kind == "end"],
                         [("root", "Part"), ("number", "Item"), ("empty", "Item"),
                          ("root", "Other"), ("root", "Config")])
        with self.assertRaises(TypeError):
            events[0][1].attributes["new"] = "value"

    def test_default_and_unknown_type(self):
        events = []
        read_xml([b'<Other type="unknown"/>'], {("Config", None): self.fail}, events.append)
        self.assertEqual([e.kind for e in events], ["start", "end"])
        self.assertEqual(events[0].attributes, {"type": "unknown"})
        # A type with no specialised route falls back to its document handler.
        events.clear()
        read_xml([b'<Config type="unknown"/>'], {("Config", None): events.append})
        self.assertEqual(len(events), 2)

    def test_arbitrary_boundaries_and_encodings(self):
        xml = '<Config xmlns="urn:test"><Item type="word">café &amp; <![CDATA[<x>]]></Item></Config>'
        for encoding in ("utf-8", "utf-16"):
            raw = ('<?xml version="1.0" encoding="' + encoding + '"?>' + xml).encode(encoding)
            for size in (1, 2, 7, 65536):
                with self.subTest(encoding=encoding, size=size):
                    events = []
                    result = read_xml((raw[i:i+size] for i in range(0, len(raw), size)),
                                      default=events.append)
                    self.assertEqual(result.root, "{urn:test}Config")
                    self.assertEqual(result.bytes_read, len(raw))
                    self.assertEqual("".join(e.text for e in events), "café & <x>")

    def test_does_not_resolve_includes_or_convert_values(self):
        events = []
        read_xml([b'<Config xmlns:xi="http://www.w3.org/2001/XInclude">'
                  b'<xi:include href="file:///not-opened"/><Item value="001" ref="$(fixture)"/>'
                  b'</Config>'], default=events.append)
        self.assertEqual(events[-2].attributes, {"value": "001", "ref": "$(fixture)"})

    def test_invalid_documents_fail_without_echoing_input(self):
        for doc in (b'', b'<Config>', b'<Config></Wrong>', b'<A/><B/>',
                    b'<Config secret="private" secret="private"/>', b'<Config>\xff</Config>',
                    b'<?xml version="1.0" encoding="private"?><Config/>',
                    b'<?xml version="1.0" encoding="UTF-32"?><Config/>'):
            with self.subTest(doc=doc), self.assertRaises(XmlReadError) as caught:
                read_xml([doc])
            self.assertNotIn("private", str(caught.exception))

    def test_dtd_is_refused_including_external_entities(self):
        for doc in (b'<!DOCTYPE Config [<!ENTITY x "private">]><Config>&x;</Config>',
                    b'<!DOCTYPE Config SYSTEM "file:///private"><Config/>'):
            with self.assertRaisesRegex(XmlReadError, "DTD"):
                read_xml([doc])

    def test_depth_limit(self):
        self.assertEqual(read_xml([b'<A><B/></A>'], max_depth=2).max_depth, 2)
        with self.assertRaisesRegex(XmlReadError, "depth"):
            read_xml([b'<A><B><C/></B></A>'], max_depth=2)

    def test_callback_errors_propagate_unchanged(self):
        for error in (RuntimeError("consumer"), LookupError("consumer"), ValueError("consumer"),
                      expat.ExpatError("consumer")):
            def fail(event):
                raise error
            with self.assertRaises(type(error)) as caught:
                read_xml([b'<A/>'], default=fail)
            self.assertIs(caught.exception, error)

    def test_callbacks_are_incremental_and_partial_on_error(self):
        events = []
        def chunks():
            yield b'<Config><Item/>'
            self.assertEqual([e.kind for e in events], ["start", "start", "end"])
            yield b'</Wrong>'
        with self.assertRaises(XmlReadError):
            read_xml(chunks(), default=events.append)

    def test_memory_does_not_grow_with_sibling_count(self):
        def peak(count):
            def chunks():
                yield b'<Config>'
                for _ in range(count):
                    yield b'<Item value="fixture">text</Item>'
                yield b'</Config>'
            tracemalloc.start()
            try:
                read_xml(chunks())
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
        small = peak(100)
        self.assertLess(peak(20000), small + 100000)

    def test_stream_ownership_and_cli_counts(self):
        stream = io.BytesIO(b'<Config/>')
        self.assertEqual(read_xml(file_chunks(stream, 2)).elements, 1)
        self.assertFalse(stream.closed)
        with tempfile.TemporaryDirectory() as directory:
            good, bad = [os.path.join(directory, name) for name in ("good.xml", "private.xml")]
            for path, content in ((good, b'<Config><Item type="x"/></Config>'), (bad, b'<Secret>')):
                with open(path, "wb") as out:
                    out.write(content)
            run = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "tools", "gamedata_reader.py"),
                                  good, bad, os.path.join(directory, "missing.xml")],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout), dict(documents=3, well_formed=1, invalid=1,
                             unreadable=1, elements=2, typed_elements=1, max_depth=2))
            self.assertEqual(run.stderr, "")
            self.assertNotIn("Secret", run.stdout)
            self.assertNotIn(directory, run.stdout)


if __name__ == "__main__":
    unittest.main()
