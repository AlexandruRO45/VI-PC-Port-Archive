"""Exhaustive tests for tools/manifest_stream.py: every read size, valid and invalid documents (no game data)."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import manifest_stream as ms  # noqa: E402

VALID = [
    '{"Items":[]}',
    '{"Count":0,"Items":[]}',
    '{"Items":[{}],"Count":1}',
    '{"src":"a\\"b\\\\c\\u00e9 ]}{,:","Count":2,"Items":[{"a":1.25},{"b":-1E-2}],'
    '"Scan":{"x":[1e24,true,false,null,"]}{,"],"y":{}}}',
    '{"m":{"Count":99},"Items":[{"n":[1,[2,{"c":"d"}]]}],"Count":1}',
    ' \n{ "Items" : [ { } , { "k" : 0.5e+3 } ] , "Fail" : [ { } , { } ] , "Count" : 2 } \n',
    '{"Items":[{"s":"x"}],"f":12345678901234567890,"g":-0.0,"h":[],"i":""}',
    '{"Count":3,"Items":[1,"two",[3]],"tail":[[[[]]]]}',
]

INVALID = [
    '', ' ', '[]', '{}', '{"Count":1}', 'null',
    '{"Items":[{}{}]}', '{ "Count":2,"Items":[{}{}]', '{"Items":[{},]}', '{"Items":[,{}]}', '{"Items":[{},,{}]}',
    '{"Items":[{}]', '{"Items":[{}]} x', '{"Items":[{}]}{}', '{"Items":[{}] "Count":1}', '{"Count":1 "Items":[]}',
    '{"Items":[{}],}', '{,"Items":[]}', '{"Items" [{}]}', '{Items:[]}',
    '{"Items":[],"F":[{} {}]}', '{"Count":1,"Items":[{}],"ModelDecodeFailures":[{} {}]}',
    '{"Items":[],"F":{"a":1 "b":2}}', '{"Items":[],"F":{"a" 1}}', '{"Items":[],"F":{1:2}}', '{"Items":[],"F":[1,2}',
    '{"Items":[],"F":{"a":[}}', '{"Items":[],"F":[1,]}', '{"Items":[],"F":{"a":1,}}', '{"Items":[],"F":[[]]]}',
    '{"Items":[],"F":tru}', '{"Items":[],"F":nul}', '{"Items":[],"F":1.}', '{"Items":[],"F":1e}',
    '{"Items":[],"F":-}', '{"Items":[],"F":"abc}', '{"Items":[],"F":01}', '{"Items":[{}x]}', '{"Items":[],"F":}',
    '{"Count":1.5e,"Items":[]}', '{"Items":[],"Count":"1"', '{"Items":[]',
    '{"Items":[],"F":[NaN]}', '{"Items":[],"F":Infinity}', '{"Items":[{"a":-Infinity}]}', '{"Count":NaN,"Items":[]}',
    '{"Items":[],"F":{"a":[1,NaN]}}', '{"Items":[NaN]}',
]


def run(path, chunk):
    gen = ms.iter_items(path, chunk)
    items = []
    while True:
        try:
            items.append(next(gen))
        except StopIteration as stop:
            return items, stop.value


def expected(text):
    doc = json.loads(text)
    c = doc.get("Count")
    return doc["Items"], (c if isinstance(c, int) and not isinstance(c, bool) else None)


class EveryReadSize(unittest.TestCase):
    def test_valid_documents_match_json_loads_at_every_read_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in VALID:
                want = expected(text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in range(1, len(text) + 2):
                    self.assertEqual(run(path, chunk), want, (text, chunk))

    def test_invalid_documents_are_rejected_at_every_read_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for text in INVALID:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                for chunk in range(1, len(text) + 2):
                    with self.assertRaises(ValueError, msg=(text, chunk)):
                        run(path, chunk)

    def test_numbers_split_at_the_default_read_size(self):
        # the reviewers' shapes: the default-size read ends inside a top-level number, after "1.", "1e" or "1E-",
        # with the member before and after Items. The padding is computed from the exact prefix that is written, and
        # the test checks that the boundary really falls inside the number.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            for number, cut in (("1.25", 2), ("1e24", 2), ("1E-2", 3), ("12", 1), ("1e3", 2), ("-0.5e+3", 5)):
                for before in (True, False):
                    prefix = "{" if before else '{"Items":[{}],"Count":1,'
                    key = '"scale":'
                    pad = ms.CHUNK - len(prefix) - len(key) - cut
                    rest = ',"Items":[{}],"Count":1}' if before else "}"
                    text = prefix + " " * pad + key + number + rest
                    self.assertEqual(text[ms.CHUNK - cut:ms.CHUNK], number[:cut], (number, before))
                    self.assertEqual(text[ms.CHUNK:ms.CHUNK + len(number) - cut], number[cut:], (number, before))
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(text)
                    self.assertEqual(run(path, ms.CHUNK), ([{}], 1), (number, before))


class Memory(unittest.TestCase):
    def test_skipped_members_are_not_built(self):
        import tracemalloc
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.json")
            records = ",".join(['{"Offset":1234,"Reason":"synthetic ] } [ \\" x"}'] * 40000)
            peaks = {}
            for last in (False, True):
                fields = ['"Items":[{}]', '"ModelDecodeFailures":[' + records + ']']
                fields.insert(2 if last else 0, '"Count":1')
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{" + ",".join(fields) + "}")
                tracemalloc.start()
                self.assertEqual(run(path, 1 << 16), ([{}], 1))
                peaks[last] = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
            size = os.path.getsize(path)
            self.assertLess(max(peaks.values()), size // 4, (peaks, size))   # far below the decoded size

    def test_syntax_error_fails_without_reading_the_rest(self):
        # tedin7 on #4946: a syntax error early in a large manifest read the whole file before it was reported
        import io

        class Counting(io.StringIO):
            reads = 0

            def read(self, n=-1):
                Counting.reads += 1
                return super().read(n)

        tail = ',{"a":1}' * 200000                     # about 1.6 MB of valid items after the error
        for bad in ('x', 'tru e', '1.x', '"a" "b"', '{"a" 1}', '[1 2]', '-x', 'nulL'):
            for chunk in (64, 4096):
                Counting.reads = 0
                text = '{"Items":[{"a":1},' + bad + tail + ']}'
                s = ms.Stream(Counting(text), chunk)
                s.expect("{", "")
                s.value("key")
                s.expect(":", "")
                s.expect("[", "")
                with self.assertRaises(ValueError, msg=(bad, chunk)):
                    while True:
                        s.value("in Items")
                        s.expect(",", "between items")
                self.assertLess(Counting.reads * chunk, 3 * 4096 + 64, (bad, chunk, Counting.reads))


if __name__ == "__main__":
    unittest.main()
