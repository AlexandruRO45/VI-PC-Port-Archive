"""Stream the ``Items`` array of a large JSON manifest (``rpf9dump`` writes several-GB ones) without loading it.

``iter_items(path)`` yields each element of the top-level ``"Items"`` array and returns (as the generator's value) the
top-level ``Count``, wherever it stands in the object, or None. Everything is validated as JSON: the top-level object,
its separators, and every member's value. Members other than ``Items`` and ``Count`` (``ModelDecodeFailures`` can be
large) are checked without building them. Invalid input raises ``ValueError``, and so do ``NaN``, ``Infinity`` and
``-Infinity``, which Python's ``json`` would otherwise accept.

The one rule that makes a fixed-size read buffer safe: a decoded value is accepted only when a terminator (``,``,
``}``, ``]``, ``:`` after a key, or whitespace) is visible after it, or the file has ended. Otherwise more is read and the same value is
decoded again from its start, so a number, literal or string cut by a read boundary (``1|.25``, ``1e|24``, ``tru|e``)
is never accepted as a prefix. Standard library only.
"""
import json
import re

CHUNK = 8 << 20

_WS_RUN = re.compile(r"[ \t\r\n]*")


def _reject_constant(name):
    """``json`` accepts ``NaN``, ``Infinity`` and ``-Infinity`` by default; they are not JSON."""
    raise ValueError(f"{name} is not valid JSON")
_TERMINATORS = frozenset(",}]: \t\r\n")   # what may follow a complete value (":" only a key string)
_CUT_MARGIN = 8   # a cut value's decode error lies at most this far before the buffer's end (longest: `\uXXXX`)


class Stream:
    """A JSON text read in chunks: whitespace, single characters and whole values, reading more as needed."""

    def __init__(self, f, chunk=CHUNK):
        self.f, self.chunk = f, chunk
        self.dec = json.JSONDecoder(parse_constant=_reject_constant)
        self.buf, self.pos, self.eof = "", 0, False

    def fill(self):
        """Read the next chunk, keeping what is not consumed yet. False at the end of the file."""
        if self.eof:
            return False
        more = self.f.read(self.chunk)
        if not more:
            self.eof = True
            return False
        self.buf, self.pos = self.buf[self.pos:] + more, 0
        return True

    def peek(self):
        """The next non-whitespace character (not consumed), or None at the end of the file."""
        while True:
            self.pos = _WS_RUN.match(self.buf, self.pos).end()
            if self.pos < len(self.buf):
                return self.buf[self.pos]
            if not self.fill():
                return None

    def expect(self, ch, where):
        got = self.peek()
        if got != ch:
            raise ValueError(f"expected {ch!r} {where}, found {got!r}")
        self.pos += 1

    def value(self, where):
        """Decode one complete JSON value at the current position (see the module docstring for when it is accepted)."""
        if self.peek() is None:
            raise ValueError(f"file ends {where}")
        while True:
            try:
                val, end = self.dec.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError as e:
                # A value cut by the read buffer fails within a few characters of the buffer's end (`tru|`, `1.|`,
                # `\u00|`), or as an unterminated string. Any other error is a syntax error, so it is raised at once
                # instead of reading on to the end of a several-GB file (tedin7, #4946).
                if not self.eof and len(self.buf) - e.pos > _CUT_MARGIN and not e.msg.startswith("Unterminated"):
                    raise ValueError(f"invalid JSON value {where}: {e.msg}") from None
                val, end = None, None
            if end is not None and end < len(self.buf) and self.buf[end] in _TERMINATORS:
                self.pos = end
                return val
            if end is not None and len(self.buf) - end > _CUT_MARGIN:
                # a value followed by a non-terminator, with enough read after it that it cannot grow (`1.x`)
                raise ValueError(f"invalid JSON after a value {where}: {self.buf[end]!r}")
            if not self.fill():  # at the end of the file: what decoded is all there is
                if end is None:
                    raise ValueError(f"invalid or truncated JSON value {where}")
                self.pos = end
                return val

    def skip_value(self, where):
        """Check one JSON value, separators and all, without building it: containers are walked with a stack, keys
        and scalars go through value()."""
        first = self.peek()
        if first not in ("[", "{"):
            self.value(where)
            return
        close = {"[": "]", "{": "}"}
        self.pos += 1
        stack = [[first, "start"]]        # per open container: its kind and what may come next
        while stack:
            kind, state = stack[-1]
            ch = self.peek()
            if ch is None:
                raise ValueError(f"file ends inside a value {where}")
            if state == "after":
                if ch == ",":
                    self.pos += 1
                    stack[-1][1] = "element"
                elif ch == close[kind]:
                    self.pos += 1
                    stack.pop()
                else:
                    raise ValueError(f"expected ',' or {close[kind]!r} {where}, found {ch!r}")
                continue
            if state == "start" and ch == close[kind]:
                self.pos += 1
                stack.pop()
                continue
            if kind == "{":
                if ch != '"':
                    raise ValueError(f"expected a string key {where}, found {ch!r}")
                self.value(where)
                self.expect(":", where)
                ch = self.peek()
            stack[-1][1] = "after"
            if ch in ("[", "{"):
                self.pos += 1
                stack.append([ch, "start"])
            else:
                if ch is None:
                    raise ValueError(f"file ends inside a value {where}")
                self.value(where)


def _is_count(key, val):
    return key == "Count" and isinstance(val, int) and not isinstance(val, bool)


def iter_items(path, chunk=CHUNK):
    """Yield the elements of the top-level ``"Items"`` array; return the top-level ``Count`` (or None)."""
    with open(path, "r", encoding="utf-8") as f:
        s = Stream(f, chunk)
        s.expect("{", "at the start")
        count, seen_items = None, False
        if s.peek() == "}":
            raise ValueError("no top-level \"Items\" array")
        while True:
            if s.peek() != '"':
                raise ValueError(f"expected a top-level key, found {s.peek()!r}")
            key = s.value("in the top-level object")
            s.expect(":", f"after the top-level key {key!r}")
            if key == "Items" and not seen_items:
                seen_items = True
                s.expect("[", "to open \"Items\"")
                if s.peek() == "]":
                    s.pos += 1
                else:
                    while True:
                        yield s.value("in \"Items\"")
                        nxt = s.peek()
                        if nxt == "]":
                            s.pos += 1
                            break
                        if nxt != ",":
                            raise ValueError(f"expected ',' or ']' in \"Items\", found {nxt!r}")
                        s.pos += 1
            elif key == "Count":
                val = s.value("for the top-level key 'Count'")
                if _is_count(key, val):
                    count = val
            else:
                s.skip_value(f"for the top-level key {key!r}")
            nxt = s.peek()
            if nxt == "}":
                s.pos += 1
                break
            if nxt != ",":
                raise ValueError(f"expected ',' or '}}' between top-level members, found {nxt!r}")
            s.pos += 1
        if s.peek() is not None:
            raise ValueError("data after the top-level object")
        if not seen_items:
            raise ValueError("no top-level \"Items\" array")
        return count
