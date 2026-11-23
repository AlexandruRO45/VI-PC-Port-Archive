"""Parity tests for the fused record loop in tools/rbf0_tree.py.

    python -I tests/test_rbf0_tree_parity.py

rbf0_tree.Parser reads the record stream in one loop with the open element's state in locals. ReferenceParser below
is the record-at-a-time parser it replaced, kept as the oracle: every stream here, including each truncated prefix
and seeded byte mutations, must give an identical document (same values, same key order, same error and offset) from
both. Every byte is built by the test. No game file, game asset path or game name is used, and nothing is pinned to
a hash or to a platform.
"""
import collections
import json
import math
import os
import random
import struct
import sys
import tracemalloc
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_tree  # noqa: E402
from rbf0_tree import MAGIC, PROLOGUE, TYPE_NAMES, VALUE_TYPES, TreeError, _render  # noqa: E402

WIDE = MAGIC + PROLOGUE
CLOSE = b"\xff\xff"


def _utf8(raw):
    """Return (text, valid). Invalid bytes become U+FFFD and are counted, never silently accepted."""
    try:
        return raw.decode("utf-8"), True
    except UnicodeDecodeError:
        return raw.decode("utf-8", "replace"), False


class ReferenceParser:
    """The record-at-a-time parser rbf0_tree.Parser replaced, kept verbatim as the oracle."""

    def __init__(self, data, source="", offset=0, chunks=1):
        self.data = data
        self.source = source
        self.offset = offset
        self.chunks = chunks
        self.wide = data[4:8] == PROLOGUE
        self.p = 8 if self.wide else 4
        self.scalar = 8 if self.wide else 4
        self.descs = []
        self.stack = []
        self.root = None
        self.status = "ok"
        self.error = ""
        self.error_offset = None
        self.counts = collections.Counter()
        self.descriptor_names = set()
        self.attribute_mismatches = 0
        self.attribute_order_violations = 0
        self.text_after_child = 0
        self.text_before_child = 0
        self.value_after_child = 0
        self.invalid_utf8 = 0
        self.max_depth = 0

    def take(self, n, what):
        if self.p + n > len(self.data):
            raise TreeError(f"{what} truncated", self.p, truncated=True)
        start = self.p
        self.p += n
        return self.data[start:self.p]

    def _descriptor(self, index, kind, at):
        if index == len(self.descs):
            n = struct.unpack("<H", self.take(2, "descriptor name length"))[0]
            raw = self.take(n, "descriptor name")
            name, valid = _utf8(raw)
            if not valid:
                self.invalid_utf8 += 1
            self.descs.append({
                "index": index,
                "name": name,
                "first_type": TYPE_NAMES.get(kind, f"0x{kind:02x}"),
                "offset": at,
                "name_offset": self.p - n,
                "uses": 0,
                "types": {},
                "utf8_valid": valid,
            })
            self.descriptor_names.add(name)
        elif index > len(self.descs):
            raise TreeError(f"descriptor index {index} of {len(self.descs)}", at)
        desc = self.descs[index]
        desc["uses"] += 1
        key = TYPE_NAMES.get(kind, f"0x{kind:02x}")
        desc["types"][key] = desc["types"].get(key, 0) + 1
        return desc

    def _attach(self, frame, node):
        node["depth"] = frame["depth"] + 1
        if frame["node"]["kind"] != "element":
            raise TreeError("record outside an element", node["offset"])
        if node["kind"] == "value" and frame["attrs_seen"] < frame["attrs_declared"]:
            if frame["content_started"]:
                frame["node"]["attribute_order_violation"] = True
                self.attribute_order_violations += 1
            frame["node"]["attributes"].append(node)
            frame["node"]["attrs_actual"] += 1
            frame["attrs_seen"] += 1
            return
        if node["kind"] in ("element", "text") and frame["attrs_seen"] < frame["attrs_declared"]:
            frame["node"]["attribute_order_violation"] = True
            self.attribute_order_violations += 1
        frame["content_started"] = True
        frame["node"]["content"].append(node)
        frame["node"]["content_count"] += 1
        if node["kind"] == "element":
            frame["node"]["children"] += 1
            if frame["text_seen"]:
                frame["node"]["text_before_child"] = True
                self.text_before_child += 1
            frame["child_seen"] = True
        elif node["kind"] == "value":
            if frame["child_seen"]:
                frame["node"]["value_after_child"] = True
                self.value_after_child += 1
        elif node["kind"] == "text":
            if frame["child_seen"]:
                frame["node"]["text_after_child"] = True
                self.text_after_child += 1
            if frame["text_seen"]:
                frame["node"]["multiple_text"] = True
            frame["text_seen"] = True

    def parse(self):
        data = self.data
        if len(data) < 4 or data[:4] != MAGIC:
            raise TreeError("not an RBF0 stream", 0)
        try:
            while self.p < len(data):
                self.record_offset = self.p
                at = self.p
                index = self.take(1, "record index")[0]
                kind = self.take(1, "record type")[0]
                if index == 0xFF:
                    if kind != 0xFF:
                        raise TreeError("close tag is not FF FF", at)
                    if not self.stack:
                        raise TreeError("close without an open element", at)
                    frame = self.stack.pop()
                    frame["node"]["end"] = self.p
                    if frame["attrs_seen"] < frame["attrs_declared"]:
                        frame["node"]["attribute_mismatch"] = True
                        self.attribute_mismatches += 1
                    if not self.stack:
                        self.root = frame["node"]
                        break
                    continue
                if index == 0xFD:
                    if kind != 0xFF:
                        raise TreeError("text tag is not FD FF", at)
                    n = struct.unpack("<I", self.take(4, "text length"))[0]
                    raw = self.take(n, "text")
                    payload_at = self.p - n
                    body = raw[:-1] if raw.endswith(b"\x00") else raw
                    text, valid = _utf8(body)
                    if not valid:
                        self.invalid_utf8 += 1
                    node = {
                        "kind": "text",
                        "offset": at,
                        "payload_offset": payload_at,
                        "length": n,
                        "nul_terminated": raw.endswith(b"\x00"),
                        "text": text,
                        "utf8_valid": valid,
                    }
                    if not self.stack:
                        raise TreeError("text outside an element", at)
                    self._attach(self.stack[-1], node)
                    self.counts["texts"] += 1
                    continue
                desc = self._descriptor(index, kind, at)
                name = desc["name"]
                if kind == 0x00:
                    raw = self.take(6, "element header")
                    payload_at = self.p - 6
                    word0, word1, attrs = struct.unpack("<HHH", raw)
                    if word0 or word1:
                        raise TreeError(f"element header word is not zero ({word0}, {word1})", at)
                    node = {
                        "kind": "element",
                        "name": name,
                        "descriptor": index,
                        "offset": at,
                        "header_offset": payload_at,
                        "attrs_declared": attrs,
                        "attrs_actual": 0,
                        "depth": len(self.stack) + 1,
                        "children": 0,
                        "content_count": 0,
                        "attributes": [],
                        "content": [],
                        "end": None,
                    }
                    self.counts["elements"] += 1
                    self.max_depth = max(self.max_depth, len(self.stack) + 1)
                    if self.stack:
                        self._attach(self.stack[-1], node)
                    else:
                        if self.root is not None:
                            raise TreeError("second root element", at)
                        self.root = node
                    self.stack.append({
                        "node": node,
                        "attrs_declared": attrs,
                        "attrs_seen": 0,
                        "content_started": False,
                        "child_seen": False,
                        "text_seen": False,
                        "depth": len(self.stack) + 1,
                    })
                    continue
                if kind not in VALUE_TYPES:
                    raise TreeError(f"unknown record type {kind:#04x}", at)
                node = self._value(kind, name, index, at)
                if not self.stack:
                    raise TreeError("value outside an element", at)
                self._attach(self.stack[-1], node)
                self.counts["values"] += 1
        except TreeError as exc:
            if not self.error:
                self.status = "truncated" if exc.truncated else "bad"
                self.error = str(exc)
                self.error_offset = exc.offset
        if self.status == "ok":
            if self.stack:
                self.status = "truncated"
                self.error = f"{len(self.stack)} unclosed element(s) at 0x{self.p:x}"
                self.error_offset = self.p
            elif self.root is None:
                self.status = "bad"
                self.error = "no root element"
                self.error_offset = self.p
            elif self.root.get("end") != len(data):
                self.status = "trailing"
                self.error = f"{len(data) - self.root['end']} bytes after the root closed"
                self.error_offset = self.root["end"]
        return {
            "source": self.source,
            "offset": self.offset,
            "size": len(data),
            "chunks": self.chunks,
            "layout": "wide-64" if self.wide else "narrow-32",
            "prologue": PROLOGUE.hex() if self.wide else "",
            "root_name": self.root["name"] if self.root else "",
            "status": self.status,
            "error": self.error,
            "error_offset": self.error_offset,
            "end": self.root["end"] if self.root and self.root.get("end") is not None else self.p,
            "descriptors": self.descs,
            "descriptor_count": len(self.descs),
            "distinct_descriptor_names": len(self.descriptor_names),
            "counts": dict(self.counts),
            "attribute_mismatches": self.attribute_mismatches,
            "attribute_order_violations": self.attribute_order_violations,
            "text_after_child": self.text_after_child,
            "text_before_child": self.text_before_child,
            "value_after_child": self.value_after_child,
            "invalid_utf8": self.invalid_utf8,
            "max_depth": self.max_depth,
            "tree": self.root,
        }

    def _value(self, kind, name, index, at):
        payload_at = self.p
        extra = {}
        if kind == 0x10:
            raw = self.take(self.scalar, "integer")
            value = struct.unpack("<Q" if self.wide else "<I", raw)[0]
            self.counts["ints"] += 1
        elif kind == 0x20 or kind == 0x30:
            value = kind == 0x20
            self.counts["bools"] += 1
        elif kind == 0x40:
            raw = self.take(self.scalar, "float")
            value = struct.unpack("<d" if self.wide else "<f", raw)[0]
            self.counts["floats"] += 1
        elif kind == 0x50:
            raw = self.take(3 * self.scalar, "vector")
            value = list(struct.unpack("<ddd" if self.wide else "<fff", raw))
            if any(not math.isfinite(v) for v in value):
                extra["float_special"] = [_render(v) for v in value]
                value = [v if math.isfinite(v) else None for v in value]
            self.counts["vec3"] += 1
        else:  # 0x60
            n = struct.unpack("<H", self.take(2, "string length"))[0]
            raw = self.take(n, "string")
            payload_at = self.p - n
            # The measured string form is length-prefixed, not NUL-terminated. Keep every byte even if the last
            # one is zero; only the text record (whose u32 length includes a terminator in this corpus) is trimmed.
            value, valid = _utf8(raw)
            if not valid:
                self.invalid_utf8 += 1
            extra = {"byte_length": n, "nul_terminated": raw.endswith(b"\x00"), "utf8_valid": valid}
            self.counts["strings"] += 1
        if isinstance(value, float) and not math.isfinite(value):
            extra["float_special"] = _render(value)
            value = None
        return {
            "kind": "value",
            "name": name,
            "descriptor": index,
            "type": VALUE_TYPES[kind],
            "offset": at,
            "payload_offset": payload_at,
            "size": self.p - payload_at,
            "value": value,
            **extra,
        }


def reference(data, source="", offset=0, chunks=1):
    return ReferenceParser(data, source, offset, chunks).parse()


def dumped(doc):
    """JSON text keeps the key order, which == on dicts ignores and the report files do not."""
    return json.dumps(doc, allow_nan=False)


class Stream:
    """Builds a record stream; a descriptor name is written once, on its first use."""

    def __init__(self, wide=True):
        self.wide = wide
        self.out = bytearray(WIDE if wide else MAGIC)
        self.names = {}

    def _head(self, name, kind):
        index = self.names.get(name)
        if index is None:
            index = self.names[name] = len(self.names)
            return bytes([index, kind]) + struct.pack("<H", len(name)) + name
        return bytes([index, kind])

    def open(self, name, attrs=0):
        self.out += self._head(name, 0x00) + struct.pack("<HHH", 0, 0, attrs)
        return self

    def close(self):
        self.out += CLOSE
        return self

    def int(self, name, value):
        self.out += self._head(name, 0x10) + struct.pack("<Q" if self.wide else "<I", value)
        return self

    def float(self, name, value):
        self.out += self._head(name, 0x40) + struct.pack("<d" if self.wide else "<f", value)
        return self

    def vec3(self, name, a, b, c):
        self.out += self._head(name, 0x50) + struct.pack("<ddd" if self.wide else "<fff", a, b, c)
        return self

    def string(self, name, raw):
        self.out += self._head(name, 0x60) + struct.pack("<H", len(raw)) + raw
        return self

    def flag(self, name, value):
        self.out += self._head(name, 0x20 if value else 0x30)
        return self

    def text(self, raw, nul=True):
        raw += b"\x00" if nul else b""
        self.out += b"\xfd\xff" + struct.pack("<I", len(raw)) + raw
        return self

    def bytes(self):
        return bytes(self.out)


def everything(wide):
    """Every record kind, every flag the parser raises, nesting, and text and values in odd places."""
    s = Stream(wide)
    s.open(b"Root", attrs=0)
    s.open(b"Item", attrs=3).int(b"id", 7).float(b"ratio", 0.5).string(b"label", b"alpha")
    s.vec3(b"pos", 1.0, -2.5, 3.25).text(b"body").open(b"Leaf").text(b"inner").close()
    s.int(b"late", 9).text(b"after child").text(b"second text").flag(b"on", True).flag(b"off", False)
    s.close()
    s.open(b"Short", attrs=2).int(b"id", 1).close()                       # attribute count mismatch
    s.open(b"Order", attrs=2).text(b"first").int(b"id", 2).int(b"n", 3).close()  # text before attributes
    s.open(b"Order2", attrs=2).open(b"Child").close().int(b"id", 1).int(b"n", 2).close()
    s.open(b"Empty").close()
    s.open(b"Bytes").string(b"nul", b"a\x00").string(b"bad", b"\xff\xfe").text(b"\xc3\x28").text(b"x", nul=False)
    s.close()
    s.open(b"Deep")
    for _ in range(40):
        s.open(b"Deep", attrs=1).int(b"d", 1)
    for _ in range(40):
        s.close()
    s.close()
    s.float(b"nan", float("nan")).float(b"inf", float("inf")).float(b"ninf", float("-inf"))
    s.vec3(b"vnan", float("nan"), 1.0, float("-inf"))
    s.close()
    return s.bytes()


def bad_name_stream(wide):
    s = Stream(wide)
    s.open(b"Root")
    s.out += bytes([1, 0x10]) + struct.pack("<H", 2) + b"\xff\xfe" + struct.pack("<Q" if wide else "<I", 1)
    s.close()
    return s.bytes()


def synthetic(kind, wide, size, seed):
    """A seeded stream of about `size` bytes. 'dense' reuses a few dozen descriptors; 'sparse' reaches for the
    253 that one index byte can address, so the descriptor table is as wide as it gets."""
    rnd = random.Random(seed)
    s = Stream(wide)
    s.open(b"Root")
    depth = 1
    k = 0
    while len(s.out) < size:
        k += 1
        mod = 8 if kind == "dense" else 40
        s.open(b"E%d" % (k % mod), attrs=rnd.choice((0, 2, 3, 4)))
        for j in range(rnd.choice((0, 2, 3, 4))):
            name = b"a%d_%d" % (k % mod, j)
            choice = rnd.randrange(5)
            if choice == 0:
                s.int(name, rnd.randrange(1 << 20))
            elif choice == 1:
                s.float(name, rnd.random() * 100)
            elif choice == 2:
                s.string(name, b"s%d" % rnd.randrange(1000))
            elif choice == 3:
                s.vec3(name, rnd.random(), rnd.random(), rnd.random())
            else:
                s.flag(name, rnd.random() < 0.5)
        if rnd.random() < 0.3:
            s.text(b"text %d" % k)
        if rnd.random() < 0.25 and depth < 8:
            depth += 1
            continue
        s.close()
        if depth > 1 and rnd.random() < 0.4:
            s.close()
            depth -= 1
    while depth:
        s.close()
        depth -= 1
    return s.bytes()


class ParityTest(unittest.TestCase):
    def assertSame(self, data, label=""):
        """Both parsers give the same document, or both refuse the stream (a bad magic) with the same error."""
        try:
            want = reference(data)
        except TreeError as expected:
            with self.assertRaises(TreeError, msg=label) as raised:
                rbf0_tree.parse_resource(data)
            self.assertEqual(str(raised.exception), str(expected), label)
            return {"status": "refused"}
        got = rbf0_tree.parse_resource(data)
        self.assertEqual(got, want, label)
        self.assertEqual(dumped(got), dumped(want), label)
        return want

    def test_every_record_kind_wide_and_narrow(self):
        for wide in (True, False):
            doc = self.assertSame(everything(wide), f"wide={wide}")
            self.assertEqual(doc["status"], "ok")
            for key in ("elements", "values", "texts", "ints", "floats", "vec3", "strings", "bools"):
                self.assertGreater(doc["counts"][key], 0, key)
            for key in ("attribute_mismatches", "attribute_order_violations", "text_after_child",
                        "text_before_child", "value_after_child", "invalid_utf8"):
                self.assertGreater(doc[key], 0, key)
            self.assertGreater(doc["max_depth"], 40)

    def test_non_finite_values_are_nulled_and_named(self):
        doc = rbf0_tree.parse_resource(everything(False))
        values = {c["name"]: c for c in doc["tree"]["content"] if c["kind"] == "value"}
        self.assertIsNone(values["nan"]["value"])
        self.assertEqual(values["inf"]["float_special"], "inf")
        self.assertEqual(values["ninf"]["float_special"], "-inf")
        self.assertEqual(values["vnan"]["value"], [None, 1.0, None])
        self.assertEqual(values["vnan"]["float_special"], ["nan", "1.0", "-inf"])

    def test_invalid_utf8_descriptor_name(self):
        for wide in (True, False):
            doc = self.assertSame(bad_name_stream(wide))
            self.assertEqual(doc["invalid_utf8"], 1)
            self.assertFalse(doc["descriptors"][1]["utf8_valid"])

    def test_every_truncated_prefix(self):
        for wide in (True, False):
            blob = everything(wide)
            seen = set()
            for cut in range(len(blob) + 1):
                seen.add(self.assertSame(blob[:cut], f"wide={wide} cut={cut}")["status"])
            self.assertIn("truncated", seen)
            self.assertIn("bad", seen)
            self.assertIn("ok", seen)

    def test_every_single_byte_corruption_of_a_small_stream(self):
        s = Stream(True)
        s.open(b"R", attrs=1).int(b"a", 1).string(b"s", b"hi").text(b"t").vec3(b"v", 1.0, 2.0, 3.0)
        s.open(b"C").flag(b"f", True).close().close()
        blob = s.bytes()
        for at in range(len(blob)):
            for value in (0x00, 0x01, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x7F, 0xFC, 0xFD, 0xFE, 0xFF):
                if blob[at] != value:
                    self.assertSame(blob[:at] + bytes([value]) + blob[at + 1:], f"byte {at} -> {value:#x}")

    def test_seeded_mutations(self):
        rnd = random.Random(20240607)
        statuses = collections.Counter()
        for wide in (True, False):
            for kind in ("dense", "sparse"):
                base = synthetic(kind, wide, 600, 11)
                self.assertSame(base)
                for _ in range(120):
                    m = bytearray(base)
                    how = rnd.randrange(5)
                    if how == 0:
                        m = m[:rnd.randrange(len(m))]
                    elif how == 1:
                        for _ in range(rnd.randrange(1, 4)):
                            m[rnd.randrange(len(m))] = rnd.randrange(256)
                    elif how == 2:
                        at = rnd.randrange(8, len(m))
                        del m[at:at + rnd.randrange(1, 9)]
                    elif how == 3:
                        at = rnd.randrange(8, len(m))
                        m[at:at] = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 8)))
                    else:
                        m += bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 20)))
                    statuses[self.assertSame(bytes(m))["status"]] += 1
        self.assertGreaterEqual(len(statuses), 3, statuses)

    def test_random_record_soup(self):
        rnd = random.Random(99)
        choices = (0, 1, 2, 3, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0xFD, 0xFF, 0x41)
        for _ in range(1500):
            body = bytes(rnd.choice(choices) if rnd.random() < 0.9 else rnd.randrange(256)
                         for _ in range(rnd.randrange(0, 48)))
            self.assertSame(MAGIC + (PROLOGUE if rnd.random() < 0.5 else b"") + body)

    def test_large_streams(self):
        for kind in ("dense", "sparse"):
            for wide in (True, False):
                doc = self.assertSame(synthetic(kind, wide, 40000, 5), f"{kind} wide={wide}")
                self.assertEqual(doc["status"], "ok")
        self.assertGreater(rbf0_tree.parse_resource(synthetic("sparse", True, 40000, 5))["descriptor_count"], 150)

    def test_not_rbf0_still_raises(self):
        for data in (b"", b"RBF", b"NOPE0000"):
            with self.assertRaises(TreeError):
                rbf0_tree.parse_resource(data)
            with self.assertRaises(TreeError):
                reference(data)

    def test_peak_memory_does_not_grow(self):
        """The fused loop keeps no table that grows with the stream or the descriptors: its peak stays within a
        few KiB of the oracle's, for a stream of few descriptors and for one with many."""
        for kind in ("dense", "sparse"):
            for wide in (True, False):
                blob = synthetic(kind, wide, 60000, 3)
                peaks = []
                for parse in (reference, rbf0_tree.parse_resource):
                    tracemalloc.start()
                    try:
                        doc = parse(blob)
                        peaks.append(tracemalloc.get_traced_memory()[1])
                    finally:
                        tracemalloc.stop()
                    del doc
                self.assertLessEqual(peaks[1], peaks[0] + 4096, f"{kind} wide={wide} {peaks}")


if __name__ == "__main__":
    unittest.main()
