"""Census of the exported texture dictionaries and of where map-material texture references resolve.

Reads an ``make_game.py`` export folder (``<work>/assets``), listing folders with ``os.listdir`` only:

* texture dictionaries: ``<archive>/textures/manifest.json`` (``rpf9dump textures``: ``Items`` with ``Offset`` and
  ``Textures`` carrying ``Hash``, ``Name``, ``Format``, ``Width``, ``Height``, ``MipCount``, ``Key``) for every archive
  folder that has one. Per archive: dictionaries (checked against the manifest's ``Count``), textures, empty and
  single-texture dictionaries, distinct texture hashes, formats, the offset band and how the offsets sit in a 4 KiB
  page, and whether ``Hash`` is the joaat of ``Name``;
* texture pools: ``ttpf/<pool>/manifest.json`` and ``ttpf_pack/<pack>/<pool>/manifest.json`` (``Textures`` with a
  64-bit ``AssetKey``), each checked against its ``Count``;
* material references: every ``<archive>/models/manifest.json`` (streamed item by item; the levels manifest is several
  GB), each material texture's ``Id64`` and ``Name``, and the textures embedded in models (``Embedded``).

Each distinct material reference (``Id64``) is looked up in three places: the pools by ``AssetKey == Id64`` (exact
64-bit), the embedded textures by ``Key == Id64`` or ``joaat(Name) == Hash``, and the dictionaries by
``joaat(Name) == Hash`` (32-bit, so a name-hash match, not proof of the same asset). It is counted for every source it
resolves in, and as unresolved when it resolves in none, per referencing archive and overall.

    python -I tools/texture_dicts.py <export dir> [--json FILE]

Prints counts only: no texture, model or shader names. ``--json`` may not overwrite an input. Standard library only.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hashdict import joaat  # noqa: E402
from report_paths import same_report_path  # noqa: E402

CHUNK = 8 << 20


_WS_RUN = re.compile(r"[ \t\r\n]*")


class _Stream:
    """Incremental reader over a JSON text: whitespace, single characters and whole values, reading more as needed."""

    def __init__(self, f, chunk, dec):
        self.f, self.chunk, self.dec = f, chunk, dec
        self.buf, self.pos = "", 0

    def fill(self):
        more = self.f.read(self.chunk)
        if not more:
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
        """Decode one JSON value. A value that reaches the end of the buffer may go on in the next read (the digits of
        a number split between reads), so it is accepted only when something follows it or the file has ended."""
        if self.peek() is None:
            raise ValueError(f"file ends {where}")
        while True:
            try:
                val, end = self.dec.raw_decode(self.buf, self.pos)
            except json.JSONDecodeError:
                val, end = None, None
            if end is not None and end < len(self.buf):
                self.pos = end
                return val
            if not self.fill():
                if end is None:
                    raise ValueError(f"invalid or truncated JSON value {where}")
                self.pos = end
                return val


def _is_count(key, val):
    return key == "Count" and isinstance(val, int) and not isinstance(val, bool)


def iter_items(path, chunk=CHUNK):
    """Yield the objects of the top-level ``"Items"`` array of a JSON object without loading the file whole.

    The top level is parsed member by member, so ``Count`` is the top-level member wherever it stands (before or after
    ``Items``; a ``Count`` nested in another value is not it), and separators are checked: a missing, doubled, leading
    or trailing comma in ``Items``, an unclosed object or data after it raise ``ValueError``. Returns (as the
    generator's value) the top-level ``Count``, or None."""
    with open(path, "r", encoding="utf-8") as f:
        s = _Stream(f, chunk, json.JSONDecoder())
        s.expect("{", "at the start")
        count = None
        while True:
            if s.peek() == "}":
                raise ValueError("no top-level \"Items\" array")
            key = s.value("in the top-level object")
            if not isinstance(key, str):
                raise ValueError("a top-level key is not a string")
            s.expect(":", f"after the top-level key {key!r}")
            if key != "Items":
                val = s.value(f"for the top-level key {key!r}")
                if _is_count(key, val):
                    count = val
                s.expect(",", "between top-level members")
                continue
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
                    if s.peek() in (",", "]"):
                        raise ValueError("misplaced ',' in \"Items\"")
            while True:  # the members after Items, then the end of the object
                nxt = s.peek()
                if nxt == "}":
                    s.pos += 1
                    break
                if nxt != ",":
                    raise ValueError(f"expected ',' or '}}' after \"Items\", found {nxt!r}")
                s.pos += 1
                key = s.value("in the top-level object")
                if not isinstance(key, str):
                    raise ValueError("a top-level key is not a string")
                s.expect(":", f"after the top-level key {key!r}")
                val = s.value(f"for the top-level key {key!r}")
                if _is_count(key, val):
                    count = val
            if s.peek() is not None:
                raise ValueError("data after the top-level object")
            return count


def read_all(path):
    gen, items = iter_items(path), []
    while True:
        try:
            items.append(next(gen))
        except StopIteration as stop:
            return items, stop.value


def key64(value):
    """A 64-bit key as lowercase hex without a 0x prefix, or None."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value & 0xFFFFFFFFFFFFFFFF:016x}"
    if isinstance(value, str) and value.strip():
        v = value.strip().lower()
        v = v[2:] if v.startswith("0x") else v
        if re.fullmatch(r"[0-9a-f]{1,16}", v):
            return v.rjust(16, "0")
    return None


def archives(export):
    return sorted(e for e in os.listdir(export) if os.path.isdir(os.path.join(export, e)))


def pool_manifests(export):
    found = []
    base = os.path.join(export, "ttpf")
    if os.path.isdir(base):
        for pool in sorted(os.listdir(base)):
            m = os.path.join(base, pool, "manifest.json")
            if os.path.isfile(m):
                found.append(("ttpf/" + pool, m))
    base = os.path.join(export, "ttpf_pack")
    if os.path.isdir(base):
        for pack in sorted(os.listdir(base)):
            if not os.path.isdir(os.path.join(base, pack)):
                continue
            for pool in sorted(os.listdir(os.path.join(base, pack))):
                m = os.path.join(base, pack, pool, "manifest.json")
                if os.path.isfile(m):
                    found.append((f"ttpf_pack/{pack}/{pool}", m))
    return found


def census(export):
    problems = []
    report = {"dictionaries": {}, "pools": {"manifests": 0, "textures": 0, "distinct_keys": 0},
              "references": {}, "problems": problems}

    # texture dictionaries
    dict_hashes = {}                  # archive -> set of 32-bit hashes
    for arc in archives(export):
        m = os.path.join(export, arc, "textures", "manifest.json")
        if not os.path.isfile(m):
            continue
        with open(m, encoding="utf-8") as f:
            doc = json.load(f)
        items = doc.get("Items") or []
        if doc.get("Count") is not None and doc["Count"] != len(items):
            problems.append(f"{arc}/textures: {len(items)} dictionaries, manifest Count {doc['Count']}")
        textures = [t for it in items for t in (it.get("Textures") or [])]
        hashes, formats, named, joaat_ok, with_key = set(), Counter(), 0, 0, 0
        for t in textures:
            h = t.get("Hash")
            try:
                hv = int(h, 16) if isinstance(h, str) else int(h)
            except (TypeError, ValueError):
                hv = None
            if hv is not None:
                hashes.add(hv)
            formats[str(t.get("Format"))] += 1
            if t.get("Name"):
                named += 1
                if hv is not None and hv == joaat(t["Name"]):
                    joaat_ok += 1
            if t.get("Key") is not None:
                with_key += 1
        sizes = Counter(len(it.get("Textures") or []) for it in items)
        offs = [it["Offset"] for it in items if isinstance(it.get("Offset"), int)]
        page = Counter(o % 4096 for o in offs)
        dict_hashes[arc] = hashes
        report["dictionaries"][arc] = {
            "dictionaries": len(items), "declared_count": doc.get("Count"), "textures": len(textures),
            "empty": sizes.get(0, 0), "single_texture": sizes.get(1, 0), "largest": max(sizes) if sizes else 0,
            "distinct_hashes": len(hashes), "hash_is_joaat_of_name": joaat_ok, "named": named, "with_64bit_key": with_key,
            "formats": dict(formats.most_common()),
            "offset_gib": [round(min(offs) / 2**30, 2), round(max(offs) / 2**30, 2)] if offs else None,
            "offset_mod_4096": dict(page.most_common(5)),
            "scan": doc.get("Scan"),
        }
    all_dict_hashes = set().union(*dict_hashes.values()) if dict_hashes else set()

    # texture pools
    pool_keys = set()
    for label, m in pool_manifests(export):
        with open(m, encoding="utf-8") as f:
            doc = json.load(f)
        tex = doc.get("Textures") or []
        if doc.get("Count") is not None and doc["Count"] != len(tex):
            problems.append(f"{label}: {len(tex)} textures, manifest Count {doc['Count']}")
        report["pools"]["manifests"] += 1
        report["pools"]["textures"] += len(tex)
        for t in tex:
            k = key64(t.get("AssetKey"))
            if k:
                pool_keys.add(k)
    report["pools"]["distinct_keys"] = len(pool_keys)

    # embedded textures (in model items), then material references, per referencing archive
    emb_keys, emb_hashes, emb_textures = set(), set(), 0
    overall_refs = {}
    for arc in archives(export):
        m = os.path.join(export, arc, "models", "manifest.json")
        if not os.path.isfile(m):
            continue
        refs = {}                     # Id64 -> set of name hashes seen with it
        bindings = no_key = models = 0
        gen = iter_items(m)
        declared = None
        while True:
            try:
                item = next(gen)
            except StopIteration as stop:
                declared = stop.value
                break
            models += 1
            emb = item.get("Embedded")
            if isinstance(emb, dict):
                emb = emb.get("Textures") or []
            for t in emb if isinstance(emb, list) else []:
                if isinstance(t, dict):
                    emb_textures += 1
                    k = key64(t.get("Key"))
                    if k:
                        emb_keys.add(k)
                    if t.get("Name"):
                        emb_hashes.add(joaat(t["Name"]))
            for mat in item.get("Materials") or []:
                for t in mat.get("Textures") or []:
                    bindings += 1
                    k = key64(t.get("Id64"))
                    if not k:
                        no_key += 1
                        continue
                    nh = joaat(t["Name"]) if t.get("Name") else None
                    refs.setdefault(k, set()).add(nh)
        if declared is not None and declared != models:
            problems.append(f"{arc}/models: {models} models, manifest Count {declared}")
        report["references"][arc] = (refs, bindings, no_key, models)
        for k, names in refs.items():
            overall_refs.setdefault(k, set()).update(names)
    report["embedded"] = {"textures": emb_textures, "distinct_keys": len(emb_keys),
                          "distinct_name_hashes": len(emb_hashes)}
    emb = (emb_keys, emb_hashes)
    for arc, (refs, bindings, no_key, models) in list(report["references"].items()):
        report["references"][arc] = summarise(refs, pool_keys, emb, all_dict_hashes, dict_hashes, bindings, no_key,
                                              models)
    report["references"]["total"] = summarise(overall_refs, pool_keys, emb, all_dict_hashes, dict_hashes,
                                              sum(r["bindings"] for r in report["references"].values()),
                                              sum(r["bindings_without_key"] for r in report["references"].values()),
                                              sum(r["models"] for r in report["references"].values()))
    return report


def summarise(refs, pool_keys, emb, all_dict_hashes, dict_hashes, bindings, no_key, models):
    emb_keys, emb_hashes = emb
    found = Counter()
    by_dict_archive = Counter()
    for k, names in refs.items():
        hashes = {h for h in names if h is not None}
        in_pool = k in pool_keys
        in_emb = k in emb_keys or bool(hashes & emb_hashes)
        hit = hashes & all_dict_hashes
        found["pool"] += in_pool
        found["embedded"] += in_emb
        found["dictionary"] += bool(hit)
        found["unresolved"] += not (in_pool or in_emb or hit)
        for arc, hs in dict_hashes.items():
            if hit & hs:
                by_dict_archive[arc] += 1
    return {"models": models, "bindings": bindings, "bindings_without_key": no_key, "distinct_references": len(refs),
            "resolution": {c: found.get(c, 0) for c in ("pool", "embedded", "dictionary", "unresolved")},
            "dictionary_by_archive": dict(by_dict_archive)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="the make_game assets folder (<work>/assets)")
    ap.add_argument("--json", metavar="FILE", help="write the full report as JSON")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.export):
        ap.error(f"{a.export}: not a folder")
    if a.json and os.path.exists(a.json) and os.path.commonpath(
            [os.path.realpath(a.json), os.path.realpath(a.export)]) == os.path.realpath(a.export):
        ap.error(f"--json {a.json} is inside the export folder; write the report elsewhere")
    if a.json and same_report_path(a.json, a.export):
        ap.error(f"--json {a.json} would overwrite an input")
    r = census(a.export)
    for arc, d in r["dictionaries"].items():
        print(f"dictionaries {arc}: {d['dictionaries']} (Count {d['declared_count']}), {d['textures']} textures, "
              f"{d['single_texture']} single-texture, {d['empty']} empty, {d['distinct_hashes']} distinct hashes, "
              f"Hash = joaat(Name) {d['hash_is_joaat_of_name']}/{d['named']}, 64-bit key {d['with_64bit_key']}, "
              f"offsets {d['offset_gib']} GiB, offset % 4096 {d['offset_mod_4096']}")
        print(f"  formats: {d['formats']}")
    p = r["pools"]
    print(f"texture pools: {p['manifests']} manifests, {p['textures']} textures, {p['distinct_keys']} distinct keys")
    e = r["embedded"]
    print(f"embedded textures: {e['textures']}, {e['distinct_keys']} distinct keys, "
          f"{e['distinct_name_hashes']} distinct name hashes")
    for arc, s in r["references"].items():
        print(f"references {arc}: {s['models']} models, {s['bindings']} bindings ({s['bindings_without_key']} without key), "
              f"{s['distinct_references']} distinct keys: {s['resolution']}"
              + (f", dictionary matches by archive {s['dictionary_by_archive']}" if s["dictionary_by_archive"] else ""))
    for msg in r["problems"]:
        print("problem:", msg, file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        print("wrote", a.json)
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
