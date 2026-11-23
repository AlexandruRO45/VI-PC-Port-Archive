"""gamedata_reader: the fast paths of ``parse_scalar``, ``parse_content`` and ``Reader`` give exactly the values, the
statistics and the errors of the plain conversion (no game files, no pinned hashes: every expectation is computed here
by a reference implementation written from the conversion rules in the module docstring).

    python -I tests/test_gamedata_reader_fast_paths.py
"""
import math
import os
import random
import re
import sys
import tracemalloc
import unittest
from collections import Counter

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
sys.path.insert(0, TOOLS)
import gamedata_reader as R  # noqa: E402

_INT = re.compile(r"[-+]?\d+")
_HEX = re.compile(r"0x[0-9A-Fa-f]+")
_FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?f?|[-+]?(inf|nan)", re.I)
_VECTOR_KEYS = ("x", "y", "z", "w")
_BASE_KEYS = frozenset(("type", "content", "value", "ref", "key"))
_ALL_KEYS = _BASE_KEYS | frozenset(_VECTOR_KEYS)
_GROUPS = {"vector2_array": 2, "vec3v_array": 3, "matrix33": 3, "matrix34": 3}


def ref_float(text):
    try:
        return float(text)
    except ValueError:
        return float(text[:-1])


def ref_scalar(text):
    """parse_scalar as the module docstring states it: bool, int (decimal or 0x hex), float, else the text."""
    if text in ("true", "false"):
        return text == "true"
    if _HEX.fullmatch(text):
        return int(text, 16)
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return ref_float(text)
    return text


def ref_content(kind, text):
    nums = [ref_scalar(t) for t in (text or "").split()]
    width = _GROUPS.get(kind)
    if not width:
        return nums
    if len(nums) % width:
        raise ValueError(f"{kind}: {len(nums)} numbers is not a multiple of {width}")
    return [tuple(nums[i:i + width]) for i in range(0, len(nums), width)]


def ref_is_vector(attrib):
    return "x" in attrib and all(_FLOAT.fullmatch(attrib[k]) for k in _VECTOR_KEYS if k in attrib)


class ReferenceReader(R.Reader):
    """The element conversion without any shortcut: every element takes the whole ladder of conventions."""

    def convert(self, el, use_type_handler=True):
        in_handler = id(el) in self._active_type_elements
        value = self._convert(el, use_type_handler)
        return self.with_attributes(el.attrib, value, count_stats=not in_handler)

    def with_attributes(self, attrib, value, count_stats=True):
        uses_vector = (attrib.get("type") != "NULL" and "content" not in attrib and "value" not in attrib
                       and ref_is_vector(attrib))
        consumed = _ALL_KEYS if uses_vector else _BASE_KEYS
        extra = {f"@{k}": ref_scalar(v) for k, v in attrib.items() if k not in consumed}
        if not extra:
            return value
        if count_stats:
            self.stats["extra attributes"] += 1
        if isinstance(value, dict):
            return {**value, **extra}
        return {"$value": value, **extra} if value not in ("", None) else extra

    def _convert(self, el, use_type_handler=True):
        a = el.attrib
        t = a.get("type")
        if t == "NULL":
            self.stats["null"] += 1
            return None
        if t is not None:
            if id(el) not in self._active_type_elements:
                self.types[t] += 1
            handler = self.type_handlers.get(t) if use_type_handler else None
            if handler:
                self.stats["type handler"] += 1
                self._active_type_elements.add(id(el))
                try:
                    return handler(el, self)
                finally:
                    self._active_type_elements.discard(id(el))
        if "content" in a:
            self.stats["array"] += 1
            return ref_content(a["content"], el.text)
        if "value" in a:
            self.stats["scalar"] += 1
            return ref_scalar(a["value"])
        if ref_is_vector(a):
            self.stats["vector"] += 1
            return tuple(ref_float(a[k]) for k in _VECTOR_KEYS if k in a)
        if "ref" in a:
            self.stats["reference"] += 1
            return {"$ref": a["ref"]}
        kids = list(el)
        if not kids:
            self.stats["text"] += 1
            return (el.text or "").strip()
        return self._typed_object(t, self.children(kids))

    def assemble(self, parts):
        if all(tag == "Item" for tag, _, _ in parts):
            if all(key is not None for _, key, _ in parts):
                self.stats["map"] += 1
                return {key: value for _, key, value in parts}
            self.stats["list"] += 1
            return [value for _, _, value in parts]
        self.stats["struct"] += 1
        out, repeated = {}, set()
        for tag, _, value in parts:
            if tag in out:
                if tag not in repeated:
                    out[tag], repeated = [out[tag]], repeated | {tag}
                out[tag].append(value)
            else:
                out[tag] = value
        return out


def same(a, b):
    """Equality that also tells ``1`` from ``1.0`` from ``True`` and ``-0.0`` from ``0.0``; NaN equals NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return a == b


def outcome(fn, *args):
    try:
        return ("ok", fn(*args))
    except Exception as e:  # the error type and message are part of the contract (run() reports the message)
        return ("error", type(e).__name__, str(e))


def same_outcome(a, b):
    return a[0] == b[0] and (a[1:] == b[1:] if a[0] == "error" else same(a[1], b[1]))


ALPHABET = list("0123456789+-.eEfFxXaAbBcCdD_ \t") + [
    "inf", "nan", "INF", "NaN", "0x", "0X", "٣", "½", "ı", "İ", "ſ", "true", "false",
    " ", " ", "1e5", "Infinity", "1_000"]
KNOWN_TOKENS = ["0", "1", "-1", "+7", "007", "12345678901234567890", "0x0", "0xFF", "0xff", "0Xff", "0x", "0xG",
                "1.5", "-1.5", "+1.5", ".5", "5.", "-.5", "+.5e3", "1e5", "1E-5", "1.5f", "1.5F", "1f", "f", "-f",
                "inf", "-inf", "+INF", "nan", "-nan", "NaN", "inff", "infinity", "1.5.2", "--1", "1-", "", "e5",
                "true", "false", "True", "TRUE", "-0", "-0.0", "0.0", "00.10", "1_0", "١٢", "١.٥",
                "1 2", " 1", "1\n"]


class ParseScalarParity(unittest.TestCase):
    def test_known_tokens(self):
        for t in KNOWN_TOKENS:
            with self.subTest(token=t):
                self.assertTrue(same_outcome(outcome(ref_scalar, t), outcome(R.parse_scalar, t)), t)

    def test_random_tokens(self):
        rng = random.Random(20260101)
        for _ in range(60000):
            t = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 6)))
            self.assertTrue(same_outcome(outcome(ref_scalar, t), outcome(R.parse_scalar, t)), repr(t))

    def test_random_numbers(self):
        rng = random.Random(7)
        for _ in range(20000):
            kind = rng.randrange(4)
            if kind == 0:
                t = str(rng.randint(-10 ** rng.randint(1, 30), 10 ** rng.randint(1, 30)))
            elif kind == 1:
                t = f"{rng.uniform(-1e3, 1e3):.{rng.randint(0, 8)}f}"
            elif kind == 2:
                t = f"{rng.uniform(-1, 1):.6e}"
            else:
                t = f"{rng.uniform(-9, 9):.3f}f"
            self.assertTrue(same_outcome(outcome(ref_scalar, t), outcome(R.parse_scalar, t)), t)


class ParseContentParity(unittest.TestCase):
    KINDS = ["short_array", "float_array", "float16_array", "vector2_array", "vec3v_array", "matrix33", "matrix34",
             "unknown_kind"]

    def check(self, kind, text):
        self.assertTrue(same_outcome(outcome(ref_content, kind, text), outcome(R.parse_content, kind, text)),
                        (kind, text))

    def test_bodies_that_take_the_bulk_paths_and_their_neighbours(self):
        bodies = [None, "", " ", "\n\t ", "1", "1.5", "0.5 1.5 2.5", "\n 0.5\n1.5\r\n2.5 \n", "1 2 3 4 5 6",
                  "+1 -2 003", "-.5 5. +.5", "1.5 2", "1 2.5", "1.5 1.5f", "1e5 2.0", "0x10 1.0", "1.5 true",
                  "1.5 2.5", "1.5 2.5", "١.٥ 2.5", "١٢ 3", "1_0 2.5", "1.5 x 2.5",
                  "1.5 2.5 3.5 4.5 5.5 6.5", "1 2 3 4 5", "1 2", "inf nan -inf", "1.5\x0b2.5", "1.5\x0c2.5",
                  "1.5\x1c2.5", "1.5\x852.5", "1. .5", "1..5", "1.5.", "- 1.5", "+ 2"]
        for kind in self.KINDS:
            for body in bodies:
                with self.subTest(kind=kind, body=body):
                    self.check(kind, body)

    def test_random_bodies(self):
        rng = random.Random(99)
        for _ in range(40000):
            mode = rng.random()
            n = rng.randint(0, 10)
            if mode < 0.35:
                toks = [f"{rng.uniform(-9, 9):.{rng.randint(0, 4)}f}" for _ in range(n)]
            elif mode < 0.55:
                toks = [str(rng.randint(-99, 99)) for _ in range(n)]
            elif mode < 0.8:
                toks = [rng.choice(KNOWN_TOKENS[1:]) or "9" for _ in range(n)]
            else:
                toks = ["".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 4))) for _ in range(n)]
            seps = [rng.choice([" ", "\n", "  ", "\t", "\r\n", " "]) for _ in range(n + 1)]
            text = "".join(seps[i] + toks[i] for i in range(n)) + (seps[n] if rng.random() < 0.5 else "")
            self.check(rng.choice(self.KINDS), text)


def make_document(rng, items):
    """A generated document using every convention, odd attribute combinations and bodies included."""
    def number():
        return rng.choice(["0", "1", "-3", "0x1F", "2.5", "-0.25", "1e3", "7f", "1.5f", "inf", "true", "false",
                           "text", "", " 4 ", "1_0"])

    def vec_attrs():
        keys = rng.choice(["x", "xy", "xyz", "xyzw", "xz", "yz", "x"])
        return " ".join(f'{k}="{rng.choice(["0", "1.5", "-2", "3.25f", "abc", "1e2"])}"' for k in keys)

    def body():
        return " ".join(rng.choice(["0", "1", "2.5", "-0.5", "0x2", "7.0"]) for _ in range(rng.randint(0, 9)))

    def element(depth):
        pick = rng.randrange(14)
        tag = rng.choice(["a", "b", "Item", "pos", "size"])
        if pick == 0:
            return f'<{tag} value="{number()}"/>'
        if pick == 1:
            return f'<{tag} value="{number()}" name="n{rng.randint(0, 3)}" userFlags="{number()}"/>'
        if pick == 2:
            return f"<{tag} {vec_attrs()}/>"
        if pick == 3:
            return f'<{tag} {vec_attrs()} extra="{number()}"/>'
        if pick == 4:
            kind = rng.choice(["short_array", "float_array", "vector2_array", "vec3v_array", "matrix33", "matrix34"])
            return f'<{tag} content="{kind}">{body()}</{tag}>'
        if pick == 5:
            return f'<{tag} ref="r{rng.randint(0, 9)}"/>'
        if pick == 6:
            return f'<{tag} type="NULL"/>'
        if pick == 7:
            return f"<{tag}>text{rng.randint(0, 9)}</{tag}>"
        if pick == 8:
            return f"<{tag}/>"
        if pick == 9:
            return f'<{tag} x="1.0" value="3"/>'
        if depth > 2:
            return f'<{tag} value="{number()}"/>'
        kids = "".join(element(depth + 1) for _ in range(rng.randint(0, 4)))
        if pick == 10:
            return f'<{tag} type="CHandled">{kids}</{tag}>'
        if pick == 11:
            return f'<{tag} type="CPlain" name="p">{kids}</{tag}>'
        if pick == 12:
            return "<{0}>{1}</{0}>".format(tag, "".join(f'<Item key="k{i}">{element(depth + 1)}</Item>'
                                                      for i in range(rng.randint(0, 3))))
        return f"<{tag}>{kids}</{tag}>"

    root_attrs = rng.choice(["", ' type="CRoot"', ' version="3"', ' type="CHandledRoot"'])
    return ('<?xml version="1.0"?>\n<Root%s>%s</Root>' % (root_attrs, "".join(element(0) for _ in range(items)))
            ).encode()


def handled(el, reader):
    return {"handled": [reader.convert(k) for k in el], "n": len(el)}


def as_list_handler(el, reader):
    return reader.convert(el, use_type_handler=False)


class ReaderParity(unittest.TestCase):
    def read_both(self, data, handlers=None):
        out = []
        for cls in (ReferenceReader, R.Reader):
            reader = cls(type_handlers=dict(handlers or {}), root_handlers={})
            out.append((outcome(reader.read, data), dict(reader.stats), dict(reader.types)))
        return out

    def check(self, data, handlers=None):
        ref, new = self.read_both(data, handlers)
        self.assertTrue(same_outcome(ref[0], new[0]), data)
        self.assertEqual(ref[1], new[1], data)
        self.assertEqual(ref[2], new[2], data)

    def test_generated_documents(self):
        rng = random.Random(4242)
        for i in range(250):
            self.check(make_document(rng, rng.randint(1, 8)))

    def test_generated_documents_with_type_handlers(self):
        rng = random.Random(777)
        for i in range(250):
            self.check(make_document(rng, rng.randint(1, 8)),
                       {"CHandled": handled, "CPlain": as_list_handler, "CHandledRoot": handled})

    def test_vector_shortcut_edge_cases(self):
        docs = [b'<R><v x="1"/></R>', b'<R><v x="1.5" y="2" z="3" w="4"/></R>', b'<R><v x="1" y="a"/></R>',
                b'<R><v y="1"/></R>', b'<R><v x="1" q="2"/></R>', b'<R><v x="1" ref="r"/></R>',
                b'<R><v x="1" type="NULL"/></R>', b'<R><v x="1" type="CThing"/></R>', b'<R><v x="1" y="2f"/></R>',
                b'<R><v x="1e400" y="-inf" z="nan"/></R>', b'<R><v x="1" value="2"/></R>',
                b'<R><v value="1"/><v value="x" type="NULL"/><v type="NULL" value="1"/></R>',
                b'<R><v value=" 1"/><v value="0x"/></R>', b'<R><v content="short_array" value="3"/></R>']
        for d in docs:
            self.check(d)
            self.check(d, {"CThing": handled})


class MemoryParity(unittest.TestCase):
    """The fast paths keep no state between calls and allocate no more than the plain conversion."""

    @staticmethod
    def peak(fn, *args):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            fn(*args)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_dense_and_sparse_arrays_peak_no_higher(self):
        rng = random.Random(5)
        dense = " ".join(f"{rng.random():.6f}" for _ in range(60000))
        sparse_ints = " ".join(str(rng.randint(0, 9)) for _ in range(60000))
        mixed = " ".join(rng.choice(["0", "1.5", "7", "0x2", "2.5f"]) for _ in range(60000))
        one = "1.5"
        for text in (dense, sparse_ints, mixed, one, ""):
            ref = self.peak(ref_content, "float_array", text)
            new = self.peak(R.parse_content, "float_array", text)
            self.assertLessEqual(new, ref * 1.05 + 4096, (len(text), ref, new))

    def test_module_keeps_no_growing_cache(self):
        before = sorted(k for k, v in vars(R).items() if isinstance(v, (dict, list, set, Counter)))
        for i in range(10000):
            R.parse_scalar(f"{i}.{i}")
            R.parse_scalar(str(i))
        R.parse_content("float_array", " ".join(f"{i}.5" for i in range(10000)))
        sizes = {k: len(getattr(R, k)) for k in before}
        for i in range(10000, 20000):
            R.parse_scalar(f"{i}.{i}")
        self.assertEqual(sizes, {k: len(getattr(R, k)) for k in before})


if __name__ == "__main__":
    unittest.main()
