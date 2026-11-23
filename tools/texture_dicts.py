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

``rpf9dump`` writes a material binding's ``Name`` as its stored 32-bit name hash in eight hex digits; dictionary and
embedded rows carry a stored ``Hash``. Those stored values are compared as they are (a name that is not eight hex
digits, in a hand-written file, is hashed with joaat). Each distinct material reference (``Id64``) is then an exact
64-bit key match (pool ``AssetKey`` or embedded ``Key``), else a 32-bit name-hash candidate (embedded or dictionary
``Hash``; not proof of the same asset, and ambiguous when the hash is on more than one entry), else unresolved; the
three add up to the total, and per-source counts (which can overlap) are listed too. A manifest whose ``Count`` is
above the rows it holds is reported as export coverage (failed rows can be left out); more rows than ``Count`` is a
problem.

    python -I tools/texture_dicts.py <export dir> [--json FILE]

Prints counts only: no texture, model or shader names. ``--json`` may not be inside the export folder (drives other
than the export's count as outside). Standard library only.
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
from manifest_stream import CHUNK, iter_items  # noqa: E402,F401  (one shared, validated reader)


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


def stored_hash(t, name_key="Name", hash_key="Hash"):
    """The 32-bit name hash a row stores, and how it was found.

    ``rpf9dump`` writes a material binding's ``Name`` as the stored name hash in eight hex digits
    (``NameHash.ToString("X8")``), and dictionary and embedded rows carry a separate stored ``Hash``. So the stored
    value is used as is; only a name that is not eight hex digits (a hand-written file) is hashed with joaat."""
    h = t.get(hash_key) if hash_key else None
    if isinstance(h, int) and not isinstance(h, bool):
        return h & 0xFFFFFFFF, "stored"
    if isinstance(h, str) and re.fullmatch(r"(0x)?[0-9a-fA-F]{1,8}", h.strip()):
        return int(h.strip(), 16), "stored"
    name = t.get(name_key)
    if isinstance(name, str) and name.strip():
        n = name.strip()
        if re.fullmatch(r"[0-9a-fA-F]{8}", n):
            return int(n, 16), "stored"
        return joaat(n), "joaat"
    return None, None


def inside(path, folder):
    """True when ``path`` lies in ``folder`` (or is it). Paths on different Windows drives are separate trees."""
    p, f = os.path.realpath(os.path.abspath(path)), os.path.realpath(os.path.abspath(folder))
    if os.path.splitdrive(p)[0].lower() != os.path.splitdrive(f)[0].lower():
        return False
    try:
        return os.path.normcase(os.path.commonpath([p, f])) == os.path.normcase(f)
    except ValueError:
        return False


def check_count(label, rows, declared, problems, coverage, unit):
    """A manifest ``Count`` above the rows read is export coverage (``rpf9dump`` can leave out rows that failed to
    decode); more rows than ``Count`` cannot happen and is a problem."""
    if declared is None or declared == rows:
        return
    if rows > declared:
        problems.append(f"{label}: {rows} {unit}, more than the manifest Count {declared}")
    else:
        coverage[label] = {"declared": declared, "rows": rows, "missing": declared - rows}


def census(export):
    problems, coverage = [], {}
    report = {"dictionaries": {}, "pools": {"manifests": 0, "textures": 0, "distinct_keys": 0},
              "references": {}, "coverage": coverage, "problems": problems}

    # texture dictionaries: stored hashes, and how many entries carry each (more than one = an ambiguous candidate)
    dict_hashes = {}                  # archive -> set of 32-bit hashes
    entries_per_hash = Counter()      # dictionary and embedded entries per stored hash
    for arc in archives(export):
        m = os.path.join(export, arc, "textures", "manifest.json")
        if not os.path.isfile(m):
            continue
        with open(m, encoding="utf-8") as f:
            doc = json.load(f)
        items = doc.get("Items") or []
        check_count(f"{arc}/textures", len(items), doc.get("Count"), problems, coverage, "dictionaries")
        textures = [t for it in items for t in (it.get("Textures") or [])]
        hashes, formats, named, joaat_ok, with_key = set(), Counter(), 0, 0, 0
        for t in textures:
            hv, _ = stored_hash(t, name_key=None)
            if hv is not None:
                hashes.add(hv)
                entries_per_hash[hv] += 1
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

    # texture pools: exact 64-bit keys
    pool_keys = set()
    for label, m in pool_manifests(export):
        with open(m, encoding="utf-8") as f:
            doc = json.load(f)
        tex = doc.get("Textures") or []
        check_count(label, len(tex), doc.get("Count"), problems, coverage, "textures")
        report["pools"]["manifests"] += 1
        report["pools"]["textures"] += len(tex)
        for t in tex:
            k = key64(t.get("AssetKey"))
            if k:
                pool_keys.add(k)
    report["pools"]["distinct_keys"] = len(pool_keys)

    # embedded textures (in model items), then material references, per referencing archive
    emb_keys, emb_hashes, emb_textures = set(), set(), 0
    name_forms = Counter()
    overall_refs = {}
    for arc in archives(export):
        m = os.path.join(export, arc, "models", "manifest.json")
        if not os.path.isfile(m):
            continue
        refs = {}                     # Id64 -> set of stored name hashes seen with it
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
                    hv, _ = stored_hash(t)
                    if hv is not None:
                        emb_hashes.add(hv)
                        entries_per_hash[hv] += 1
            for mat in item.get("Materials") or []:
                for t in mat.get("Textures") or []:
                    bindings += 1
                    k = key64(t.get("Id64"))
                    if not k:
                        no_key += 1
                        continue
                    hv, form = stored_hash(t, hash_key=None)
                    name_forms[form or "none"] += 1
                    refs.setdefault(k, set()).add(hv)
        check_count(f"{arc}/models", models, declared, problems, coverage, "models")
        report["references"][arc] = (refs, bindings, no_key, models)
        for k, hashes in refs.items():
            overall_refs.setdefault(k, set()).update(hashes)
    report["embedded"] = {"textures": emb_textures, "distinct_keys": len(emb_keys),
                          "distinct_name_hashes": len(emb_hashes)}
    report["material_name_hashes"] = dict(name_forms)
    sources = (pool_keys, emb_keys, emb_hashes, all_dict_hashes, dict_hashes, entries_per_hash)
    for arc, (refs, bindings, no_key, models) in list(report["references"].items()):
        report["references"][arc] = summarise(refs, sources, bindings, no_key, models)
    report["references"]["total"] = summarise(overall_refs, sources,
                                              sum(r["bindings"] for r in report["references"].values()),
                                              sum(r["bindings_without_key"] for r in report["references"].values()),
                                              sum(r["models"] for r in report["references"].values()))
    return report


def summarise(refs, sources, bindings, no_key, models):
    """Per distinct reference: an exact 64-bit key match (pool ``AssetKey`` or embedded ``Key``), else a 32-bit
    name-hash candidate (embedded or dictionary stored ``Hash``), else unresolved. The three add up to the total;
    the per-source counts below them can overlap."""
    pool_keys, emb_keys, emb_hashes, all_dict_hashes, dict_hashes, entries_per_hash = sources
    part, source, by_dict_archive = Counter(), Counter(), Counter()
    for k, hashes in refs.items():
        hashes = {h for h in hashes if h is not None}
        in_pool, in_emb_key = k in pool_keys, k in emb_keys
        emb_cand, dict_cand = hashes & emb_hashes, hashes & all_dict_hashes
        source["pool key"] += in_pool
        source["embedded key"] += in_emb_key
        source["pool and embedded key"] += in_pool and in_emb_key
        source["embedded name hash"] += bool(emb_cand)
        source["dictionary name hash"] += bool(dict_cand)
        if in_pool or in_emb_key:
            part["exact 64-bit key"] += 1
        elif emb_cand or dict_cand:
            part["32-bit name-hash candidate only"] += 1
            if any(entries_per_hash[h] > 1 for h in emb_cand | dict_cand) or len(emb_cand | dict_cand) > 1:
                part["  of which ambiguous (hash on more than one entry)"] += 1
        else:
            part["unresolved"] += 1
        if dict_cand:  # most references have no candidate: skip the per-archive loop (shdann, #4980)
            for arc, hs in dict_hashes.items():
                if dict_cand & hs:
                    by_dict_archive[arc] += 1
    order = ("exact 64-bit key", "32-bit name-hash candidate only", "  of which ambiguous (hash on more than one entry)",
             "unresolved")
    src_order = ("pool key", "embedded key", "pool and embedded key", "embedded name hash", "dictionary name hash")
    return {"models": models, "bindings": bindings, "bindings_without_key": no_key, "distinct_references": len(refs),
            "resolution": {c: part.get(c, 0) for c in order},
            "by_source": {c: source.get(c, 0) for c in src_order},
            "dictionary_candidates_by_archive": dict(by_dict_archive)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="the make_game assets folder (<work>/assets)")
    ap.add_argument("--json", metavar="FILE", help="write the full report as JSON (not inside the export folder)")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.export):
        ap.error(f"{a.export}: not a folder")
    if a.json and (inside(a.json, a.export) or same_report_path(a.json, a.export)):
        ap.error(f"--json {a.json} is inside the export folder; write the report elsewhere")
    try:
        r = census(a.export)
    except ValueError as e:
        print(f"error: invalid manifest JSON: {e}", file=sys.stderr)
        return 2
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
          f"{e['distinct_name_hashes']} distinct stored hashes")
    print(f"material name hashes: {r['material_name_hashes']}")
    for arc, s in r["references"].items():
        print(f"references {arc}: {s['models']} models, {s['bindings']} bindings ({s['bindings_without_key']} without key), "
              f"{s['distinct_references']} distinct keys: {s['resolution']}; by source {s['by_source']}"
              + (f"; dictionary candidates by archive {s['dictionary_candidates_by_archive']}"
                 if s["dictionary_candidates_by_archive"] else ""))
    for label, c in r["coverage"].items():
        print(f"coverage {label}: {c['rows']} of {c['declared']} rows ({c['missing']} not exported)")
    for msg in r["problems"]:
        print("problem:", msg, file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(r, f, indent=1)
        print("wrote", a.json)
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
