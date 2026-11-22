"""Streaming game-data XML dispatch (stdlib only; no schema or RTTI guesses).

read_xml accepts an iterable of byte chunks. Handlers are keyed by (document root,
element's type attribute); (root, None) is the fallback for that root. A default
handler receives otherwise unhandled events. Text callbacks may split anywhere;
concatenate them if a consumer needs a complete value. Tail text belongs to the
parent. Namespaced names use ElementTree's {uri}local notation.

Callbacks are provisional until read_xml returns successfully. Consumers must
discard partial state after an error; do not commit runtime configuration in a
callback. No XML values, references, includes or paths are interpreted or opened.
DTD declarations are refused. Caller-owned streams are never closed.

CLI: python -I tools/gamedata_reader.py <file.xml> [<file.xml> ...]
Only aggregate counts are printed, including on failure; inputs remain local.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
import json
from types import MappingProxyType
from xml.parsers import expat


class XmlReadError(ValueError):
    """A malformed or unsupported document; never includes input text."""


@dataclass(frozen=True)
class Event:
    kind: str
    root: str
    tag: str
    type: str | None
    depth: int
    attributes: Mapping[str, str]
    text: str = ""


@dataclass(frozen=True)
class Summary:
    root: str
    elements: int
    typed_elements: int
    max_depth: int
    bytes_read: int


Handler = Callable[[Event], None]


def _name(name: str) -> str:
    return "{" + name if "}" in name else name


def read_xml(chunks: Iterable[bytes],
             handlers: Mapping[tuple[str, str | None], Handler] | None = None,
             default: Handler | None = None, *, max_depth: int = 256) -> Summary:
    """Dispatch one complete document, retaining only open-element metadata.

    An exact type handler takes precedence over the root fallback. Missing type
    and an explicitly empty type are distinct. Types are not inherited. Handler
    exceptions propagate unchanged. Input must be byte chunks, not a bytes object.
    Memory is proportional to open-element metadata plus Expat's current token
    and the input chunk; consumer-retained events are the consumer's responsibility.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    if isinstance(chunks, (bytes, str)):
        raise TypeError("pass an iterable of byte chunks")
    routes = dict(handlers or {})
    stack: list[tuple[Event, Handler | None]] = []
    root = ""
    elements = typed = deepest = total = 0
    parser = expat.ParserCreate(namespace_separator="}")

    def start(name, attrs):
        nonlocal root, elements, typed, deepest
        if len(stack) >= max_depth:
            raise XmlReadError("XML depth limit exceeded")
        tag = _name(name)
        if not stack:
            root = tag
        attributes = MappingProxyType({_name(k): v for k, v in attrs.items()})
        kind = attributes.get("type")
        handler = routes.get((root, kind), routes.get((root, None), default))
        event = Event("start", root, tag, kind, len(stack) + 1, attributes)
        stack.append((event, handler))
        elements += 1
        typed += "type" in attributes
        deepest = max(deepest, len(stack))
        if handler is not None:
            handler(event)

    def end(name):
        event, handler = stack.pop()
        if handler is not None:
            handler(replace(event, kind="end"))

    def text(value):
        if stack:
            event, handler = stack[-1]
            if handler is not None:
                handler(replace(event, kind="text", text=value))

    def doctype(*args):
        raise XmlReadError("DTD declarations are unsupported")

    # Keep callback exceptions distinct from Expat's own parse errors, even when
    # a consumer happens to raise an ExpatError itself.
    callback_error = None

    def guarded(callback):
        def call(*args):
            nonlocal callback_error
            try:
                return callback(*args)
            except Exception as exc:
                callback_error = exc
                raise
        return call

    parser.StartElementHandler = guarded(start)
    parser.EndElementHandler = guarded(end)
    parser.CharacterDataHandler = guarded(text)
    parser.StartDoctypeDeclHandler = guarded(doctype)
    try:
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("XML chunks must be bytes")
            total += len(chunk)
            parser.Parse(chunk, False)
        parser.Parse(b"", True)
    except (expat.ExpatError, LookupError, ValueError) as exc:
        if callback_error is exc:
            raise
        raise XmlReadError(f"invalid XML at line {parser.ErrorLineNumber}, "
                           f"column {parser.ErrorColumnNumber}") from None
    return Summary(root, elements, typed, deepest, total)


def file_chunks(stream, chunk_size: int = 65536):
    """Yield bounded reads from a caller-owned binary file."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    while chunk := stream.read(chunk_size):
        yield chunk


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("files", nargs="+")
    args = cli.parse_args()
    counts = dict(documents=0, well_formed=0, invalid=0, unreadable=0,
                  elements=0, typed_elements=0, max_depth=0)
    for path in args.files:
        counts["documents"] += 1
        try:
            with open(path, "rb") as stream:
                result = read_xml(file_chunks(stream))
        except OSError:
            counts["unreadable"] += 1
        except XmlReadError:
            counts["invalid"] += 1
        else:
            counts["well_formed"] += 1
            counts["elements"] += result.elements
            counts["typed_elements"] += result.typed_elements
            counts["max_depth"] = max(counts["max_depth"], result.max_depth)
    print(json.dumps(counts, sort_keys=True))
    return int(bool(counts["invalid"] or counts["unreadable"]))


if __name__ == "__main__":
    raise SystemExit(main())
