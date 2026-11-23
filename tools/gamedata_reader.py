"""Stream the game-data XML of a carved RPF9 archive into Python values, dispatching on root tag and ``type``.

The XML in ``common.rpf`` is RAGE parser metadata (docs/gamedata.md): 118 root tags and 5,293 element tags, written
with a handful of conventions. This reader turns any of it into plain values without a schema per tag, and lets a
caller plug in a schema where one is known:

- ``<x value="1.5"/>``: a scalar (bool for ``true``/``false``, int for decimal or ``0x`` hex, float, else the text);
- ``<v x=".." y=".." [z=".." [w=".."]]/>``: a vector, as a tuple of floats;
- ``<a content="short_array|float_array|float16_array|vector2_array|vec3v_array|matrix33|matrix34">..</a>``: the
  whitespace-separated numbers of the body (pairs, triples or matrix rows where the content says so);
- ``<Item type="CSomeClass">..</Item>``: a polymorphic object, ``{"$type": "CSomeClass", ...}``; ``type="NULL"`` is
  ``None``;
- ``<x ref="name"/>``: ``{"$ref": "name"}``;
- children that are all ``<Item>``: a list; children that all carry ``key``: a dict by key; repeated tags: a list;
- an element with only text: the text.

Dispatch: ``register_root(tag)`` registers a handler for a whole document's root element, ``register_type(name)``
one for every element with ``type="name"`` (both called as ``handler(element, reader)``; ``reader.convert`` gives the
generic value of a sub-element). Everything else takes the generic conversion. The documents are read with
``iterparse`` and each top-level field is converted and dropped as soon as it ends, so memory is bounded by the largest
field, not the document.

    python -I tools/gamedata_reader.py <carve_dir or .xml> [...] [--root TAG] [--out DIR] [--recover]

prints counts only: documents read, roots, the elements each convention covered, ``type`` classes seen, and
documents that are not well-formed XML (tail-only or truncated carves), with the reason. ``--recover`` also extracts
complete trailing sibling sections and ``<Item>`` subtrees from payloads that fail whole-document parsing without
fabricating broken elements. ``--out DIR`` writes one JSON file per document: those files hold the game's data and
stay on the local machine (CONTRIBUTING.md). Standard library only.
Usage: `python -I tools/gamedata_reader.py --help` lists the options.
"""
import argparse
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

BOM = b"\xef\xbb\xbf"
VECTOR_KEYS = ("x", "y", "z", "w")
BASE_CONVENTION_KEYS = frozenset(("type", "content", "value", "ref", "key"))
CONVENTION_KEYS = BASE_CONVENTION_KEYS | frozenset(VECTOR_KEYS)
GROUPS = {"vector2_array": 2, "vec3v_array": 3, "matrix33": 3, "matrix34": 3}
INT = re.compile(r"[-+]?\d+")
HEX = re.compile(r"0x[0-9A-Fa-f]+")
FLOAT = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?f?|[-+]?(inf|nan)", re.I)
_XML_TOKEN_RE = re.compile(
    rb"<!--.*?-->|<!\[CDATA\[.*?\]\]>|<\?.*?\?>|<(/?)([A-Za-z_][\w.:-]*)([^<>]*)>",
    re.S,
)
_TAIL_CLOSE_RE = re.compile(rb"</([A-Za-z_][\w.:-]*)>\s*$")

ROOT_HANDLERS = {}
TYPE_HANDLERS = {}


def _is_vector(attrib):
    """True when ``attrib`` carries ``x`` and every present ``x``/``y``/``z``/``w`` attribute is a float."""
    return "x" in attrib and all(FLOAT.fullmatch(attrib[k]) for k in VECTOR_KEYS if k in attrib)


def _uses_vector_keys(attrib):
    """True when ``attrib`` resolves via the vector convention rather than another attribute convention."""
    return (attrib.get("type") != "NULL"
            and "content" not in attrib
            and "value" not in attrib
            and _is_vector(attrib))


def register_root(tag):
    """Decorator: ``handler(element, reader)`` for every document whose root element is ``tag``."""
    def wrap(fn):
        ROOT_HANDLERS[tag] = fn
        return fn
    return wrap


def register_type(name):
    """Decorator: ``handler(element, reader)`` for every element with ``type="name"``."""
    def wrap(fn):
        TYPE_HANDLERS[name] = fn
        return fn
    return wrap


def parse_scalar(text):
    """A ``value`` attribute or array token: bool, int (decimal or 0x hex), float, else the text unchanged."""
    if text in ("true", "false"):
        return text == "true"
    if HEX.fullmatch(text):
        return int(text, 16)
    if INT.fullmatch(text):
        return int(text)
    if FLOAT.fullmatch(text):
        return _float(text)
    return text


def _float(text):
    """A float with an optional C-style ``f`` suffix ("1.5f"); "inf" keeps its f."""
    try:
        return float(text)
    except ValueError:
        return float(text[:-1])


def parse_content(kind, text):
    """The numbers of a ``content=`` body, grouped into tuples for vector arrays and matrices."""
    nums = [parse_scalar(t) for t in (text or "").split()]
    width = GROUPS.get(kind)
    if not width:
        return nums
    if len(nums) % width:
        raise ValueError(f"{kind}: {len(nums)} numbers is not a multiple of {width}")
    return [tuple(nums[i:i + width]) for i in range(0, len(nums), width)]


class Reader:
    """Converts elements, counting which convention each one took (``stats``) and the ``type`` classes seen."""

    def __init__(self, root_handlers=None, type_handlers=None):
        self.root_handlers = ROOT_HANDLERS if root_handlers is None else root_handlers
        self.type_handlers = TYPE_HANDLERS if type_handlers is None else type_handlers
        self.stats = Counter()
        self.types = Counter()
        self._active_type_elements = set()

    def convert(self, el, use_type_handler=True):
        """The generic value of element ``el`` (a registered ``type`` handler first, unless told not to). Attributes
        no convention uses (``name``, ``userFlags``, ...) are kept as ``"@name"`` entries."""
        in_handler = id(el) in self._active_type_elements
        value = self._convert(el, use_type_handler)
        return self.with_attributes(el.attrib, value, count_stats=not in_handler)

    def with_attributes(self, attrib, value, count_stats=True):
        consumed = CONVENTION_KEYS if _uses_vector_keys(attrib) else BASE_CONVENTION_KEYS
        extra = {f"@{k}": parse_scalar(v) for k, v in attrib.items() if k not in consumed}
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
            return parse_content(a["content"], el.text)
        if "value" in a:
            self.stats["scalar"] += 1
            return parse_scalar(a["value"])
        if _is_vector(a):
            self.stats["vector"] += 1
            return tuple(_float(a[k]) for k in VECTOR_KEYS if k in a)
        if "ref" in a:
            self.stats["reference"] += 1
            return {"$ref": a["ref"]}
        kids = list(el)
        if not kids:
            self.stats["text"] += 1
            return (el.text or "").strip()
        return self._typed_object(t, self.children(kids))

    def _typed_object(self, t, value):
        """Wrap an assembled child container in a polymorphic ``$type`` dict when ``t`` is set."""
        if t is None:
            return value
        self.stats["object"] += 1
        if not isinstance(value, dict):
            value = {"$items": value}
        return {"$type": t, **value}

    def _streams_children(self, root):
        """True when top-level children can be converted and dropped piecemeal instead of keeping the whole root."""
        if root.tag in self.root_handlers:
            return False
        a = root.attrib
        t = a.get("type")
        if t == "NULL" or (t is not None and t in self.type_handlers):
            return False
        return not ("content" in a or "value" in a or _is_vector(a) or "ref" in a)

    def children(self, kids):
        """Child elements as a list (all ``Item``), a dict by ``key`` (all keyed) or a dict by tag."""
        return self.assemble([(k.tag, k.attrib.get("key"), self.convert(k)) for k in kids])

    def assemble(self, parts):
        """``(tag, key or None, value)`` of each child, in order -> list, dict by key or dict by tag (a tag that occurs
        more than once becomes the list of its values)."""
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

    def read(self, data):
        """One document (bytes) -> ``(root tag, root type, value)``. Fields are converted as their end tag arrives
        and then dropped from the tree. Raises ``ET.ParseError`` for a document that is not well-formed."""
        stream = io.BytesIO(data[len(BOM):] if data.startswith(BOM) else data)
        depth, root, parts, stream_kids = 0, None, [], False
        for event, el in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                depth += 1
                if root is None:
                    root = el
                    stream_kids = self._streams_children(root)
                continue
            depth -= 1
            if depth == 1 and stream_kids:
                # a top-level field is complete: convert it now and drop its sub-tree and root reference
                parts.append((el.tag, el.attrib.get("key"), self.convert(el)))
                el.clear()
                del root[:]
        handler = self.root_handlers.get(root.tag)
        if handler:
            self.stats["root handler"] += 1
            return root.tag, root.attrib.get("type"), handler(root, self)
        if not parts:
            return root.tag, root.attrib.get("type"), self.convert(root)
        attrib = dict(root.attrib)
        value = self.assemble(parts)
        t = attrib.get("type")
        if t is not None:
            self.types[t] += 1
            value = self._typed_object(t, value)
        return root.tag, t, self.with_attributes(attrib, value)

    def recover(self, data, root_hint=""):
        """Extract complete sibling sections and ``<Item>`` subtrees from truncated/fragment XML."""
        return recover_fragments(data, reader=self, root_hint=root_hint)


def _trailing_root_span(body: bytes, root_hint: str) -> tuple[str, int]:
    """Return ``(root_tag, body_end)`` when ``body`` ends with an unmatched closing root tag."""
    if root_hint:
        m_close = re.search(rb"</" + re.escape(root_hint.encode("ascii", "replace")) + rb">\s*$", body)
        if m_close:
            return root_hint, m_close.start()
        return root_hint, len(body)
    m_tail = _TAIL_CLOSE_RE.search(body)
    if m_tail:
        tag_bytes = m_tail.group(1)
        if not re.search(rb"<" + re.escape(tag_bytes) + rb"(?:[\s>/])", body):
            return tag_bytes.decode("ascii", "replace"), m_tail.start()
    return "", len(body)


def _trailing_siblings(body: bytes, body_end: int, tokens: list, reader: Reader, root_tag: str = "") -> tuple[dict, int]:
    """Recover complete top-level sibling elements immediately preceding ``</Root>``."""
    root_bytes = root_tag.encode("ascii", "replace") if root_tag else b""
    stack = []
    counts = Counter()
    trailing_start = None
    prev_end = 0
    saw_root_open = False

    for tok in tokens:
        if b"<" in body[prev_end:tok.start()]:
            if stack:
                stack[-1][2] = True
            else:
                trailing_start = None
        prev_end = tok.end()
        name = tok.group(2)
        if name is None:
            continue
        slash, rest = tok.group(1), tok.group(3)
        if rest.rstrip().endswith(b"/"):
            if not stack and trailing_start is None:
                trailing_start = tok.start()
        elif not slash:
            if not stack and not saw_root_open and root_bytes and name == root_bytes and trailing_start is None:
                saw_root_open = True
                continue
            if not stack and trailing_start is None:
                trailing_start = tok.start()
            stack.append([name, tok.start(), False])
            counts[name] += 1
        else:
            if counts[name] > 0:
                tainted = False
                while stack:
                    top_name, _, top_tainted = stack.pop()
                    counts[top_name] -= 1
                    tainted = tainted or top_tainted
                    if top_name == name:
                        break
                    tainted = True
                if tainted:
                    if stack:
                        stack[-1][2] = True
                    else:
                        trailing_start = None
            else:
                stack.clear()
                counts.clear()
                trailing_start = None

    if b"<" in body[prev_end:body_end] or stack or trailing_start is None or trailing_start >= body_end:
        return {}, body_end
    span = body[trailing_start:body_end]
    try:
        wrap = ET.fromstring(b"<_recover>" + span + b"</_recover>")
    except ET.ParseError:
        return {}, body_end
    sections = reader.children(list(wrap))
    return (sections if isinstance(sections, dict) else {"$items": sections}), trailing_start


def _recover_prefix_items(body: bytes, prefix_end: int, tokens: list, reader: Reader, root_tag: str = "") -> list:
    """Recover outermost well-formed ``<Item>...</Item>`` subtrees within ``body[:prefix_end]`` in linear time."""
    root_bytes = root_tag.encode("ascii", "replace") if root_tag else b""
    stack = []
    counts = Counter()
    non_item_depth = 0
    candidates = []
    prev_end = 0
    saw_root_open = False

    for tok in tokens:
        if tok.end() > prefix_end:
            break
        if b"<" in body[prev_end:tok.start()] and stack:
            stack[-1][2] = True
        prev_end = tok.end()
        name = tok.group(2)
        if name is None:
            continue
        slash, rest = tok.group(1), tok.group(3)
        if rest.rstrip().endswith(b"/"):
            if name == b"Item" and non_item_depth == 0:
                candidates.append((tok.start(), tok.end()))
        elif not slash:
            if not stack and not saw_root_open and root_bytes and name == root_bytes and not candidates:
                saw_root_open = True
                continue
            stack.append([name, tok.start(), False])
            counts[name] += 1
            if name != b"Item":
                non_item_depth += 1
        else:
            if counts[name] > 0:
                tainted = False
                start_pos = 0
                while stack:
                    top_name, top_start, top_tainted = stack.pop()
                    counts[top_name] -= 1
                    if top_name != b"Item":
                        non_item_depth -= 1
                    tainted = tainted or top_tainted
                    if top_name == name:
                        start_pos = top_start
                        break
                    tainted = True
                if name == b"Item" and non_item_depth == 0 and not tainted:
                    while candidates and candidates[-1][0] >= start_pos:
                        candidates.pop()
                    candidates.append((start_pos, tok.end()))
                elif tainted and stack:
                    stack[-1][2] = True
            else:
                stack.clear()
                counts.clear()
                non_item_depth = 0

    items = []
    for start_pos, end_pos in candidates:
        try:
            items.append(reader.convert(ET.fromstring(body[start_pos:end_pos])))
        except ET.ParseError:
            pass
    return items


def recover_fragments(data: bytes, reader: Reader | None = None, root_hint: str = "") -> tuple[str, dict, list]:
    """Recover complete trailing sibling sections and ``<Item>`` subtrees from non-well-formed XML."""
    reader = reader or Reader()
    body = data[len(BOM):] if data.startswith(BOM) else data
    root_tag, body_end = _trailing_root_span(body, root_hint)
    tokens = list(_XML_TOKEN_RE.finditer(body, 0, body_end))
    sections, prefix_end = ({}, body_end)
    if body_end < len(body):
        sections, prefix_end = _trailing_siblings(body, body_end, tokens, reader, root_tag)
    items = _recover_prefix_items(body, prefix_end, tokens, reader, root_tag)
    return root_tag, sections, items


def _unique_json_path(out_dir: str, stem: str, written_names: set) -> str:
    """Choose a case-insensitively unique JSON output filename inside ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{stem}.json"
    suffix = 2
    while filename.casefold() in written_names:
        filename = f"{stem}_{suffix}.json"
        suffix += 1
    written_names.add(filename.casefold())
    return os.path.join(out_dir, filename)


def documents(targets):
    """XML payloads of carve directories (index.json) or .xml files, as gamedata_xml.load_documents yields them."""
    import gamedata_xml as gx
    return gx.load_documents(targets)


def run(targets, root_filter=None, out_dir=None, reader=None, recover=False):
    """Read every document; returns a summary of counts (no names of entries, no values)."""
    reader = reader or Reader()
    roots, failed, read = Counter(), Counter(), 0
    recovered = recovered_sections = recovered_items = 0
    written_names = set()
    for n, doc in enumerate(documents(targets)):
        hint = doc.get("root_hint")
        if root_filter and hint != root_filter:
            continue
        name = doc.get("offset_hex") or doc.get("name") or str(n)
        try:
            tag, _, value = reader.read(doc["data"])
        except (ET.ParseError, LookupError, ValueError) as e:
            # Unsupported encodings and malformed content arrays fail outside ET.ParseError too.
            failed[str(e).split(":")[0]] += 1
            if recover:
                rtag, sections, items = recover_fragments(doc["data"], reader=reader, root_hint=hint or "")
                if sections or items:
                    recovered += 1
                    recovered_sections += len(sections)
                    recovered_items += len(items)
                    if out_dir:
                        path = _unique_json_path(out_dir, f"recovered_{rtag or 'fragment'}_{name}", written_names)
                        with open(path, "w", encoding="utf-8", newline="\n") as f:
                            json.dump({"$recovered": True, "root": rtag or None,
                                       "sections": sections, "items": items}, f, indent=1, default=str)
            continue
        read += 1
        roots[tag] += 1
        if out_dir:
            path = _unique_json_path(out_dir, f"{tag}_{name}", written_names)
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(value, f, indent=1, default=str)
    return {"documents": read, "roots": len(roots), "per_root_documents": dict(roots),
            "not_well_formed": sum(failed.values()), "parse_errors": dict(failed),
            "recovered_documents": recovered, "recovered_sections": recovered_sections,
            "recovered_items": recovered_items,
            "elements_by_convention": dict(reader.stats), "type_classes": len(reader.types),
            "typed_elements": sum(reader.types.values()),
            "registered": {"roots": len(reader.root_handlers), "types": len(reader.type_handlers)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+", help="rpf9_carve.py directories (with index.json) or .xml files")
    ap.add_argument("--root", help="only documents with this root tag")
    ap.add_argument("--out", help="write one JSON file per document here (game data: keep it local)")
    ap.add_argument("--recover", action="store_true",
                    help="extract complete sibling sections and <Item> subtrees from non-well-formed XML")
    a = ap.parse_args(argv)
    if a.out and os.path.exists(a.out) and not os.path.isdir(a.out):
        print(f"ERROR: --out {a.out} exists and is not a directory", file=sys.stderr)
        return 2
    try:
        s = run(a.targets, a.root, a.out, recover=a.recover)
    except FileNotFoundError as e:
        print(f"ERROR: {e} does not exist", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as e:
        # an unreadable file, or a carve folder whose index.json is not JSON or not the rpf9_carve.py shape
        where = getattr(e, "filename", None) or ", ".join(a.targets)
        print(f"ERROR: cannot read {where}: {getattr(e, 'strerror', None) or e!r}", file=sys.stderr)
        return 2
    print(f"{s['documents']} documents read, {s['roots']} roots, {s['not_well_formed']} not well-formed "
          f"{s['parse_errors'] or ''}".rstrip())
    if a.recover:
        print(f"recovered: {s['recovered_documents']} payloads ({s['recovered_sections']} sections, "
              f"{s['recovered_items']} items)")
    print("elements by convention:", ", ".join(f"{k} {v}" for k, v in sorted(s["elements_by_convention"].items())))
    print(f"type classes: {s['type_classes']} ({s['typed_elements']} typed elements); registered handlers: "
          f"{s['registered']['roots']} roots, {s['registered']['types']} types")
    total_read = s["documents"] + (s["recovered_documents"] if a.recover else 0)
    if a.out:
        print(f"wrote {total_read} JSON files to {a.out} (game data: keep them local)")
    if not total_read and s["not_well_formed"]:
        # Nothing parsed (or recovered with --recover). A corpus with some truncated tails still reads the rest
        # and stays 0; a run that read nothing at all exits 1.
        print(f"ERROR: no document could be read; {s['not_well_formed']} not well-formed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
