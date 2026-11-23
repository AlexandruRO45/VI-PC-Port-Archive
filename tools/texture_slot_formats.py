"""Texture formats per material parameter slot, over every model manifest of an export.

Reads an ``make_game.py`` export folder (``<work>/assets``), listing folders with ``os.listdir`` only. Each material
texture binding in ``<archive>/models/manifest.json`` (streamed item by item; the levels manifest is several GB)
carries ``Param`` (the stored 32-bit parameter name hash as eight hex digits), ``Id64`` and ``Name`` (the stored
32-bit texture name hash). Per parameter, each distinct texture (``Id64``) is counted once and resolved as metadata:

* exact: the 64-bit key is a texture pool ``AssetKey`` (``ttpf``/``ttpf_pack`` manifests, ``DxgiFormat``) and/or an
  embedded texture ``Key`` (``Format``) of any model. The formats from both sources are pooled: a key with one format
  is tallied, a key with more than one (two pools, or a pool and an embedded copy, disagreeing) is counted as
  conflicting and left out of the tally; a key whose rows give no format at all is still exact and counted as
  without format. Exact keys found in both sources, and pool matches whose row has no
  exported file (``File`` null or missing on disk), are counted too: an exact match is a metadata match, not proof
  that a texture file is available;
* candidate: otherwise the stored name hash is on exactly one distinct embedded or dictionary texture (``Hash``; the
  same usable ``Key``, or without one the same format and size, is one texture); not proof of the same asset, so
  its formats are listed apart. A hash on more than one distinct texture is ambiguous; a candidate whose one
  metadata group holds several keyless rows (which can still be different textures) is counted as uncertain too;
* unresolved: neither.

An all-zero key or name hash is a sentinel for "none" (as in ``world_build.py``): it is never joined, and a binding
whose ``Id64`` is zero counts as a binding without a usable key. A manifest whose ``Count`` is above the rows it holds
is reported as export coverage; more rows than ``Count`` is a problem (exit 1). Formats are DXGI numbers (names from
``tex_headers.DXGI`` are mapped back).

    python -I tools/texture_slot_formats.py <export dir> [--archive NAME ...] [--top N] [--json FILE]

``--archive`` (the folder name exactly as listed) limits the bindings to those archives' models; the pools,
dictionaries and embedded textures used for resolving are always the whole export. Prints counts and parameter
hashes, keyed by the export's archive and pool folder names: no texture, model or shader names. ``--json`` may not be
inside the export folder. Standard library only.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_stream import iter_items  # noqa: E402
from report_paths import same_report_path  # noqa: E402
from tex_headers import DXGI  # noqa: E402
from texture_dicts import (archives, check_count, embedded_textures, inside, key64, load_object,  # noqa: E402
                           objects, pool_manifests, stored_hash)

DXGI_BY_NAME = {name: num for num, name in DXGI.items()}
ZERO_KEY = "0" * 16


def dxgi(value):
    """A format as its DXGI number, or the string as written when it is not a known DXGI name; None when absent."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        v = value.strip()
        if v.isdigit():
            return int(v)
        v = v.upper()
        v = v[len("DXGI_FORMAT_"):] if v.startswith("DXGI_FORMAT_") else v
        return DXGI_BY_NAME.get(v, v)
    return None


def param_hash(value):
    """The parameter as eight upper-case hex digits (as ``rpf9dump`` writes it), an int as such, else None."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value & 0xFFFFFFFF:08X}"
    if isinstance(value, str) and re.fullmatch(r"(0x)?[0-9a-fA-F]{1,8}", value.strip()):
        return f"{int(value.strip(), 16):08X}"
    return None


def usable_key(value):
    """A 64-bit key as 16 lowercase hex digits, or None when absent, malformed or the all-zero sentinel."""
    k = key64(value)
    return None if k is None or k == ZERO_KEY else k


def usable_hash(row, **kw):
    """The stored 32-bit name hash, or None when absent or zero."""
    hv, _ = stored_hash(row, **kw)
    return hv or None


def add(index, key, fmt):
    """Record that ``key`` exists, and its format when there is one: a row without a format is still a key match."""
    if key is not None:
        formats = index.setdefault(key, set())
        if fmt is not None:
            formats.add(fmt)


def models_archives(export):
    return [a for a in archives(export) if os.path.isfile(os.path.join(export, a, "models", "manifest.json"))]


def resolvers(export, problems, coverage):
    """Pool key -> formats, pool key -> whether any row has its exported file, stored name hash -> formats (one per
    dictionary entry)."""
    pool, pool_file, by_hash = {}, {}, {}
    for label, m in pool_manifests(export):
        doc = load_object(m)
        rows = objects(doc.get("Textures"), f"{label} Textures")
        check_count(label, len(rows), doc.get("Count"), problems, coverage, "textures")
        base = os.path.dirname(m)
        for t in rows:
            k = usable_key(t.get("AssetKey"))
            if k is None:
                continue
            add(pool, k, dxgi(t.get("DxgiFormat")))
            fn = t.get("File")
            has = isinstance(fn, str) and bool(fn) and os.path.isfile(os.path.join(base, fn))
            pool_file[k] = pool_file.get(k, False) or has
    for arc in archives(export):
        m = os.path.join(export, arc, "textures", "manifest.json")
        if os.path.isfile(m):
            doc = load_object(m)
            items = objects(doc.get("Items"), f"{arc}/textures Items")
            check_count(f"{arc}/textures", len(items), doc.get("Count"), problems, coverage, "dictionaries")
            for it in items:
                for t in objects(it.get("Textures"), f"{arc}/textures Textures"):
                    hv = usable_hash(t, name_key=None)
                    if hv is not None:
                        by_hash.setdefault(hv, {}).setdefault(identity(t), [dxgi(t.get("Format")), 0])[1] += 1
    return pool, pool_file, by_hash


def identity(t):
    """Rows that are the same texture: the same usable 64-bit ``Key``, or, without one, the same format and size (as
    ``texture_dicts.texture_identity``). One texture embedded in many models is one candidate, not an ambiguous one."""
    k = usable_key(t.get("Key"))
    if k:
        return ("key", k)
    return ("format and size", str(t.get("Format")), str(t.get("Width")), str(t.get("Height")))


def census(export, only=None):
    problems, coverage = [], {}
    pool, pool_file, by_hash = resolvers(export, problems, coverage)
    emb = {}                           # embedded Key -> formats
    refs = {}                          # param -> {Id64: set of stored name hashes}
    bindings, no_key, no_param = Counter(), 0, 0
    models = 0
    for arc in models_archives(export):
        m = os.path.join(export, arc, "models", "manifest.json")
        use = only is None or arc in only
        gen, rows = iter_items(m), 0
        while True:
            try:
                item = next(gen)
            except StopIteration as stop:
                check_count(f"{arc}/models", rows, stop.value, problems, coverage, "models")
                break
            if not isinstance(item, dict):
                raise ValueError(f"{arc}/models: model entry {rows + 1} is not an object")
            rows += 1
            # embedded textures resolve references from every archive, so they are read even when filtered out
            for t in embedded_textures(item, arc):
                add(emb, usable_key(t.get("Key")), dxgi(t.get("Format")))
                hv = usable_hash(t)
                if hv is not None:
                    by_hash.setdefault(hv, {}).setdefault(identity(t), [dxgi(t.get("Format")), 0])[1] += 1
            if not use:
                continue
            models += 1
            for mat in objects(item.get("Materials"), f"{arc}/models Materials"):
                for t in objects(mat.get("Textures"), f"{arc}/models material Textures"):
                    p = param_hash(t.get("Param"))
                    if p is None:
                        no_param += 1
                        continue
                    bindings[p] += 1
                    k = usable_key(t.get("Id64"))
                    if k is None:
                        no_key += 1
                        continue
                    refs.setdefault(p, {}).setdefault(k, set()).add(usable_hash(t, hash_key=None))

    params = {}
    for p, keys in refs.items():
        part, fmts, cand_fmts = Counter(), Counter(), Counter()
        for k, hashes in keys.items():
            in_pool, in_emb = k in pool, k in emb
            if in_pool or in_emb:
                part["exact"] += 1
                part["both"] += in_pool and in_emb
                part["no file"] += in_pool and not pool_file.get(k, False)
                found = pool.get(k, set()) | emb.get(k, set())
                if len(found) == 1:
                    fmts[next(iter(found))] += 1
                elif found:
                    part["conflicting"] += 1
                else:
                    part["no format"] += 1
                continue
            textures = {}                 # identity -> [format, rows]
            matched = [h for h in hashes if h is not None and h in by_hash]
            for h in matched:
                for ident, (fmt, rows) in by_hash[h].items():
                    textures.setdefault(ident, [fmt, 0])[1] += rows
            # as in texture_dicts: one metadata group (even reached through two hashes) is a candidate, uncertain when
            # it holds several keyless rows; more than one group is ambiguous
            if len(textures) == 1:
                part["candidate"] += 1
                (ident, (fmt, rows)), = textures.items()
                part["uncertain"] += ident[0] != "key" and rows > 1
                if fmt is not None:
                    cand_fmts[fmt] += 1
            elif textures:
                part["ambiguous"] += 1
            else:
                part["unresolved"] += 1
        params[p] = {
            "bindings": bindings[p], "distinct_textures": len(keys),
            "exact": part["exact"], "exact_conflicting": part["conflicting"], "exact_pool_and_embedded": part["both"],
            "exact_pool_without_file": part["no file"], "exact_without_format": part["no format"],
            "candidate": part["candidate"], "candidate_uncertain": part["uncertain"],
            "ambiguous": part["ambiguous"], "unresolved": part["unresolved"],
            "formats": {str(f): n for f, n in sorted(fmts.items(), key=lambda kv: (-kv[1], str(kv[0])))},
            "candidate_formats": {str(f): n for f, n in sorted(cand_fmts.items(),
                                                               key=lambda kv: (-kv[1], str(kv[0])))},
        }
    for p in bindings:                  # parameters seen only without a usable key
        params.setdefault(p, {"bindings": bindings[p], "distinct_textures": 0, "exact": 0, "exact_conflicting": 0,
                              "exact_pool_and_embedded": 0, "exact_pool_without_file": 0,
                              "exact_without_format": 0, "candidate": 0, "candidate_uncertain": 0,
                              "ambiguous": 0, "unresolved": 0, "formats": {}, "candidate_formats": {}})
    order = sorted(params, key=lambda p: (-params[p]["bindings"], p))
    return {"archives": sorted(only) if only is not None else "all", "models": models,
            "bindings": sum(bindings.values()), "bindings_without_key": no_key,
            "bindings_without_param": no_param, "parameters": {p: params[p] for p in order},
            "coverage": coverage, "problems": problems}


def fmt_list(d, n=4):
    items = list(d.items())
    s = ", ".join(f"{f}: {c:,}" for f, c in items[:n])
    return s + (f", +{len(items) - n} more" if len(items) > n else "") if items else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="the make_game assets folder (<work>/assets)")
    ap.add_argument("--archive", action="append", metavar="NAME",
                    help="count bindings from this archive's models only (the folder name as listed; repeatable)")
    ap.add_argument("--top", type=int, default=20, metavar="N", help="parameters to print (default 20; 0 = all)")
    ap.add_argument("--json", metavar="FILE", help="write the full report as JSON (not inside the export folder)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.export):
        ap.error(f"{a.export}: not a folder")
    if a.json and (inside(a.json, a.export) or same_report_path(a.json, a.export)):
        ap.error(f"--json {a.json} is inside the export folder; write the report elsewhere")
    listed = models_archives(a.export)
    if not listed:
        ap.error(f"{a.export}: no <archive>/models/manifest.json; is this a make_game assets folder?")
    only = set(a.archive) if a.archive else None
    if only is not None:
        missing = sorted(only - set(listed))   # exact names, so a case variant on Windows does not select nothing
        if missing:
            ap.error(f"no models manifest for --archive {', '.join(missing)} (archives with models: "
                     f"{', '.join(listed)})")
    try:
        r = census(a.export, only)
    except ValueError as e:
        print(f"error: invalid manifest JSON: {e}", file=sys.stderr)
        return 2
    print(f"models {r['models']:,}, bindings {r['bindings']:,} ({r['bindings_without_key']:,} without a usable Id64, "
          f"{r['bindings_without_param']:,} without Param), parameters {len(r['parameters']):,}")
    print("parameter | bindings | textures | exact | candidate | ambiguous | unresolved | formats (DXGI: textures)"
          " | candidate formats")
    shown = list(r["parameters"].items())
    for p, d in (shown[:a.top] if a.top else shown):
        notes = [f"{d[k]:,} {label}" for k, label in (("exact_conflicting", "conflicting"),
                                                      ("exact_pool_and_embedded", "pool and embedded"),
                                                      ("exact_pool_without_file", "pool row without file"),
                                                      ("exact_without_format", "without format"))
                 if d[k]]
        note = f" ({', '.join(notes)})" if notes else ""
        unc = f" ({d['candidate_uncertain']:,} uncertain)" if d["candidate_uncertain"] else ""
        print(f"0x{p} | {d['bindings']:,} | {d['distinct_textures']:,} | {d['exact']:,}{note} | "
              f"{d['candidate']:,}{unc} | {d['ambiguous']:,} | {d['unresolved']:,} | {fmt_list(d['formats'])} | "
              f"{fmt_list(d['candidate_formats'])}")
    for label, c in r["coverage"].items():
        print(f"coverage {label}: {c['rows']:,} of {c['declared']:,} rows ({c['missing']:,} not exported)")
    for msg in r["problems"]:
        print("problem:", msg, file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1)
            f.write("\n")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
