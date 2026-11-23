"""Texture formats per material parameter slot, over every model manifest of an export.

Reads an ``make_game.py`` export folder (``<work>/assets``), listing folders with ``os.listdir`` only. Each material
texture binding in ``<archive>/models/manifest.json`` (streamed item by item; the levels manifest is several GB)
carries ``Param`` (the stored 32-bit parameter name hash as eight hex digits), ``Id64`` and ``Name`` (the stored
32-bit texture name hash). Per parameter, each distinct texture (``Id64``) is counted once and resolved the way
``texture_dicts.py`` does it:

* exact: the 64-bit key is a texture pool ``AssetKey`` (``ttpf``/``ttpf_pack`` manifests, ``DxgiFormat``) or an
  embedded texture ``Key`` (``Format``);
* candidate: otherwise the stored name hash is on exactly one embedded or dictionary entry (``Hash``); not proof of
  the same asset, so its formats are listed apart. A hash on more than one entry is ambiguous and not counted;
* unresolved: neither.

Formats are DXGI numbers (names from ``tex_headers.DXGI`` are mapped back). A key found with more than one format is
counted as conflicting and left out of the format tally.

    python -I tools/texture_slot_formats.py <export dir> [--archive NAME ...] [--top N] [--json FILE]

``--archive`` limits the bindings to those archives' models (the pools and dictionaries used for resolving are
always the whole export). Prints counts and parameter hashes only: no texture, model or shader names. ``--json``
may not be inside the export folder. Standard library only.
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
from texture_dicts import archives, inside, key64, pool_manifests, stored_hash  # noqa: E402

DXGI_BY_NAME = {name: num for num, name in DXGI.items()}


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


def add(index, key, fmt):
    if key is not None and fmt is not None:
        index.setdefault(key, set()).add(fmt)


def resolvers(export):
    """Exact 64-bit key -> formats (pools and embedded keys); stored name hash -> formats, one per entry."""
    exact, by_hash = {}, {}
    for _label, m in pool_manifests(export):
        with open(m, encoding="utf-8") as f:
            doc = json.load(f)
        for t in doc.get("Textures") or []:
            add(exact, key64(t.get("AssetKey")), dxgi(t.get("DxgiFormat")))
    for arc in archives(export):
        m = os.path.join(export, arc, "textures", "manifest.json")
        if os.path.isfile(m):
            with open(m, encoding="utf-8") as f:
                doc = json.load(f)
            for it in doc.get("Items") or []:
                for t in it.get("Textures") or []:
                    hv, _ = stored_hash(t, name_key=None)
                    if hv is not None:
                        by_hash.setdefault(hv, []).append(dxgi(t.get("Format")))
    return exact, by_hash


def embedded(item):
    emb = item.get("Embedded")
    if isinstance(emb, dict):
        emb = emb.get("Textures") or []
    return [t for t in emb if isinstance(t, dict)] if isinstance(emb, list) else []


def census(export, only=None):
    exact, by_hash = resolvers(export)
    refs = {}                          # param -> {Id64: set of stored name hashes}
    bindings, no_key, no_param = Counter(), 0, 0
    models = 0
    for arc in archives(export):
        m = os.path.join(export, arc, "models", "manifest.json")
        if not os.path.isfile(m):
            continue
        use = only is None or arc in only
        for item in iter_items(m):
            # embedded textures resolve references from every archive, so they are read even when filtered out
            for t in embedded(item):
                add(exact, key64(t.get("Key")), dxgi(t.get("Format")))
                hv, _ = stored_hash(t)
                if hv is not None:
                    by_hash.setdefault(hv, []).append(dxgi(t.get("Format")))
            if not use:
                continue
            models += 1
            for mat in item.get("Materials") or []:
                for t in mat.get("Textures") or []:
                    p = param_hash(t.get("Param"))
                    if p is None:
                        no_param += 1
                        continue
                    bindings[p] += 1
                    k = key64(t.get("Id64"))
                    if not k:
                        no_key += 1
                        continue
                    hv, _ = stored_hash(t, hash_key=None)
                    refs.setdefault(p, {}).setdefault(k, set()).add(hv)

    params = {}
    for p, keys in refs.items():
        part, fmts, cand_fmts = Counter(), Counter(), Counter()
        for k, hashes in keys.items():
            if k in exact:
                part["exact"] += 1
                if len(exact[k]) == 1:
                    fmts[next(iter(exact[k]))] += 1
                else:
                    part["exact, conflicting formats"] += 1
                continue
            entries = [f for h in hashes if h is not None for f in by_hash.get(h, [])]
            if len(entries) == 1:
                part["candidate"] += 1
                cand_fmts[entries[0]] += 1
            elif entries:
                part["ambiguous"] += 1
            else:
                part["unresolved"] += 1
        params[p] = {
            "bindings": bindings[p], "distinct_textures": len(keys),
            "exact": part["exact"], "exact_conflicting": part["exact, conflicting formats"],
            "candidate": part["candidate"], "ambiguous": part["ambiguous"], "unresolved": part["unresolved"],
            "formats": {str(f): n for f, n in sorted(fmts.items(), key=lambda kv: (-kv[1], str(kv[0])))},
            "candidate_formats": {str(f): n for f, n in sorted(cand_fmts.items(),
                                                               key=lambda kv: (-kv[1], str(kv[0])))},
        }
    for p in bindings:                  # parameters seen only without a key
        params.setdefault(p, {"bindings": bindings[p], "distinct_textures": 0, "exact": 0, "exact_conflicting": 0,
                              "candidate": 0, "ambiguous": 0, "unresolved": 0, "formats": {},
                              "candidate_formats": {}})
    order = sorted(params, key=lambda p: (-params[p]["bindings"], p))
    return {"archives": sorted(only) if only is not None else "all", "models": models,
            "bindings": sum(bindings.values()), "bindings_without_key": no_key,
            "bindings_without_param": no_param, "parameters": {p: params[p] for p in order}}


def fmt_list(d, n=4):
    items = list(d.items())
    s = ", ".join(f"{f}: {c:,}" for f, c in items[:n])
    return s + (f", +{len(items) - n} more" if len(items) > n else "") if items else "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="the make_game assets folder (<work>/assets)")
    ap.add_argument("--archive", action="append", metavar="NAME",
                    help="count bindings from this archive's models only (repeatable)")
    ap.add_argument("--top", type=int, default=20, metavar="N", help="parameters to print (default 20; 0 = all)")
    ap.add_argument("--json", metavar="FILE", help="write the full report as JSON (not inside the export folder)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.export):
        ap.error(f"{a.export}: not a folder")
    if a.json and (inside(a.json, a.export) or same_report_path(a.json, a.export)):
        ap.error(f"--json {a.json} is inside the export folder; write the report elsewhere")
    only = set(a.archive) if a.archive else None
    if only is not None:
        missing = sorted(n for n in only if not os.path.isfile(os.path.join(a.export, n, "models", "manifest.json")))
        if missing:
            ap.error(f"no models manifest for --archive {', '.join(missing)}")
    try:
        r = census(a.export, only)
    except ValueError as e:
        print(f"error: invalid manifest JSON: {e}", file=sys.stderr)
        return 2
    print(f"models {r['models']:,}, bindings {r['bindings']:,} ({r['bindings_without_key']:,} without Id64, "
          f"{r['bindings_without_param']:,} without Param), parameters {len(r['parameters']):,}")
    print("parameter | bindings | textures | exact | candidate | ambiguous | unresolved | formats (DXGI: textures)"
          " | candidate formats")
    shown = list(r["parameters"].items())
    for p, d in (shown[:a.top] if a.top else shown):
        conflict = f" ({d['exact_conflicting']:,} conflicting)" if d["exact_conflicting"] else ""
        print(f"0x{p} | {d['bindings']:,} | {d['distinct_textures']:,} | {d['exact']:,}{conflict} | "
              f"{d['candidate']:,} | {d['ambiguous']:,} | {d['unresolved']:,} | {fmt_list(d['formats'])} | "
              f"{fmt_list(d['candidate_formats'])}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
