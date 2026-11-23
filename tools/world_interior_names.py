"""Resolve the interior keys of the location-name table to interior archetypes and their placements.

``GameShape__Data__LocationNameLoopkup`` (``common.rpf``, see docs/world.md) lists ``InteriorNames/Item key=...``
with one or more inner items, each a display-name key and a world position. This tool tests what a key names:

- every ``CMloArchetypeDef`` (10506455) in ``levels.rpf`` is read with its ``name`` hash, its 64-bit keys (the
  ``0x4f`` members), its box and its rooms (``CMloRoomDef`` 0b1811f1);
- every ``CMloInstanceDef`` (09cc28be), the record that places an interior in a map, is read with its
  ``archetypeName``, position, rotation and scale; ``world_index.py`` reads only ``CEntityDef``, so interiors
  never appear in its entity rows;
- each key is hashed (lower-cased joaat) and looked up among the interior names; a match counts as confirmed
  only when the lower-cased key's MurmurHash3 x64 low 64 equals one of that interior's 64-bit keys;
- each position is compared with the placements of its own key's interior: how many placements there are, and
  how far the position lies from the placed interior box (0 = inside), with the stored and the conjugated
  quaternion.

Every field is found by the case-sensitive joaat of its name in the resource's member table.

Usage:
    python -I world_interior_names.py <levels.rpf|carved folder> --names <LocationNameLoopkup JSON>
                                      [--start N] [--limit BYTES] [--jobs N] [--out FILE]

``--names`` is the JSON file ``gamedata_reader.py --root GameShape__Data__LocationNameLoopkup --out DIR`` writes
for ``common.rpf``. The printed summary holds counts only. ``--out`` also lists every key's hash, match and
distances: game data, keep it local.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import world_archetypes as WA  # noqa: E402
import world_index as WI  # noqa: E402
from pool_workers import process_workers  # noqa: E402
from report_paths import path_key, same_report_path  # noqa: E402
from world_names import hash64, joaat  # noqa: E402

J = WA.joaat_cs
MLO = J("CMloArchetypeDef")
ROOM = J("CMloRoomDef")
INSTANCE = J("CMloInstanceDef")
NAME, BBMIN, BBMAX = J("name"), J("bbMin"), J("bbMax")
U64, ARRAY = 0x4F, 0x52
LIMBO = joaat("limbo")
OVERLAP = 64 << 20  # a worker reads this far past its window so a resource that straddles the edge is whole
THRESHOLDS = (0, 1, 2, 5, 10, 25, 100)


def _vec3(b, base, member):
    if member is None:
        return None
    p = base + member[0]
    return list(struct.unpack_from("<3f", b, p)) if p + 12 <= len(b) else None


def _instances(b, structs, blocks, h):
    if h not in structs:
        return
    ssize, members = structs[h]
    for name, size, off in blocks:
        if name != h or off is None or not ssize or size % ssize or off + size > len(b):
            continue
        for k in range(size // ssize):
            yield off + k * ssize, members


def _array(b, blocks, base, member):
    """(block hash, first byte, count) of an array member: a u64 (low 12 bits: 1-based block index, bits 20+:
    byte offset in that block) and a u16 count at +8, as docs/world.md describes for instance lists."""
    p = base + member[0]
    if p + 10 > len(b):
        return None
    ref, count = struct.unpack_from("<Q", b, p)[0], struct.unpack_from("<H", b, p + 8)[0]
    index = ref & 0xFFF
    if not count or not 1 <= index <= len(blocks):
        return None
    h, size, off = blocks[index - 1]
    start = ref >> 20
    if off is None or start >= size:
        return None
    return h, off + start, count


def _rooms(b, structs, blocks, base, members):
    """[[name hash, bbMin, bbMax]] of the CMloRoomDef array an interior points at."""
    if ROOM not in structs:
        return []
    rsize, rm = structs[ROOM]
    for mm in members.values():
        if mm[1] != ARRAY:
            continue
        found = _array(b, blocks, base, mm)
        if found and found[0] == ROOM and rsize:
            h, first, count = found
            return [[WI._value(b, o, rm.get(NAME)), _vec3(b, o, rm.get(BBMIN)), _vec3(b, o, rm.get(BBMAX))]
                    for o in range(first, first + rsize * count, rsize) if o + rsize <= len(b)]
    return []


def parse(b):
    """(interiors, placements) of one validated 0x7a resource."""
    if WI.parse_resource(b) is None:
        return [], []
    structs, blocks = WA.schema(b)
    interiors = []
    for base, m in _instances(b, structs, blocks, MLO):
        keys = {"%08x" % mh: struct.unpack_from("<Q", b, base + mm[0])[0]
                for mh, mm in m.items() if mm[1] == U64 and base + mm[0] + 8 <= len(b)}
        interiors.append({"name": WI._value(b, base, m.get(NAME)), "keys": keys,
                          "bb_min": _vec3(b, base, m.get(BBMIN)), "bb_max": _vec3(b, base, m.get(BBMAX)),
                          "rooms": _rooms(b, structs, blocks, base, m)})
    placements = []
    for base, m in _instances(b, structs, blocks, INSTANCE):
        placements.append({"archetype": WI._value(b, base, m.get(WI.FIELD["archetypeName"])),
                           "position": WI._value(b, base, m.get(WI.FIELD["position"])),
                           "rotation": WI._value(b, base, m.get(WI.FIELD["rotation"])),
                           "scale_xy": WI._value(b, base, m.get(WI.FIELD["scaleXY"])),
                           "scale_z": WI._value(b, base, m.get(WI.FIELD["scaleZ"]))})
    return interiors, placements


def scan_window(job):
    """Resources whose stream starts in [start, end) of an RPF9 archive (read up to end + OVERLAP)."""
    path, start, end, size = job
    out = []
    limit = min(size, end + OVERLAP) - start
    for row, data in WA.iter_resources(path, start, limit):
        if row["offset"] < end:
            out.append((row["offset"], parse(data)))
    return out


def scan(path, start=None, limit=None, jobs=1):
    """{offset or file: (interiors, placements)} for an RPF9 archive window or a carved file/folder."""
    magic = b""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            magic = f.read(8)
    if magic[:4] != b"9FPR":
        return {row.get("file", i): parse(data) for i, (row, data) in enumerate(WA.iter_resources(path, start, limit))}
    size = os.path.getsize(path)
    data_start = WI._u32(magic, 4)
    lo = start if start is not None else data_start
    # A window that starts before the data area or past the end of the archive held no resource, so the report was an
    # empty one and the run still exited 0. instance_lists.py rejects the same window.
    if start is not None and not data_start <= lo < size:
        raise ValueError("--start 0x%x is outside the data area of %s (0x%x..0x%x)"
                         % (lo, path, data_start, size))
    hi = min(size, lo + limit) if limit is not None else size
    n = max(1, jobs)
    step = max(1, -(-(hi - lo) // n))
    windows = [(path, s, min(hi, s + step), size) for s in range(lo, hi, step)]
    found = {}
    if n == 1:
        results = [scan_window(w) for w in windows]
    else:
        # Windows refuses more than 61 worker processes (pool_workers.py); --jobs is lowered there. The windows are
        # still all scanned, so the result does not depend on the cap.
        with ProcessPoolExecutor(max_workers=process_workers(n)) as ex:
            results = list(ex.map(scan_window, windows))
    for rows in results:
        for off, parsed in rows:
            found[off] = parsed
    return found


def load_names(path):
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except ValueError as e:      # json.JSONDecodeError and UnicodeDecodeError are both ValueErrors
        raise ValueError("cannot be read as JSON: %s" % e) from None
    if not isinstance(doc, dict):
        raise ValueError("expected a JSON object with InteriorNames, found %s" % type(doc).__name__)
    section = doc.get("InteriorNames", doc)
    if not isinstance(section, dict):
        raise ValueError("InteriorNames must be an object, found %s" % type(section).__name__)
    out = {}
    for key, items in section.items():
        items = items if isinstance(items, list) else [items]
        out[key] = [it.get("Pos") for it in items if isinstance(it, dict) and it.get("Pos")]
    return out


def _local(q, p, r):
    d = [q[i] - p[i] for i in range(3)]
    return [sum(r[j][i] * d[j] for j in range(3)) for i in range(3)]  # R^T (q - p)


def _outside(v, lo, hi):
    return math.sqrt(sum(max(lo[i] - v[i], 0.0, v[i] - hi[i]) ** 2 for i in range(3)))


def _buckets(values):
    values = sorted(values)
    return {"n": len(values), "median_m": round(statistics.median(values), 3) if values else None,
            "within_m": {str(t): bisect.bisect_right(values, t) for t in THRESHOLDS}}


def resolve(names, found):
    interiors, placements = {}, {}
    n_interiors = n_placements = 0
    for parsed_interiors, parsed_placements in found.values():
        for d in parsed_interiors:
            n_interiors += 1
            interiors.setdefault(d["name"], d)
        for p in parsed_placements:
            n_placements += 1
            placements.setdefault(p["archetype"], []).append(p)
    summary = {"interior_definitions": n_interiors, "interior_names": len(interiors),
               "placements": n_placements, "placed_interior_names": len(placements),
               "placements_of_unknown_interior": sum(len(v) for k, v in placements.items() if k not in interiors),
               "keys": len(names), "positions": sum(map(len, names.values())),
               "keys_naming_an_interior": 0, "confirmed_by_64bit_key": {}, "keys_not_confirmed": 0,
               "keys_with_placements": 0, "positions_vs_placements": {"equal": 0, "more_positions": 0,
                                                                      "fewer_positions": 0},
               "positions_in_a_room_box": {"stored": 0, "conjugate": 0}}
    rows, dist = {}, {"stored": [], "conjugate": []}
    for key, positions in names.items():
        h = joaat(key)
        row = rows[key] = {"hash": "%08x" % h, "positions": len(positions)}
        d = interiors.get(h)
        if d is None:
            continue
        summary["keys_naming_an_interior"] += 1
        h64 = hash64(key.lower())
        member = next((mh for mh, v in d["keys"].items() if v == h64), None)
        row["confirmed_by"] = member
        if member is None:
            summary["keys_not_confirmed"] += 1
        else:
            c = summary["confirmed_by_64bit_key"]
            c[member] = c.get(member, 0) + 1
        pl = placements.get(h, [])
        row["placements"] = len(pl)
        if not pl:
            continue
        summary["keys_with_placements"] += 1
        pv = summary["positions_vs_placements"]
        pv["equal" if len(positions) == len(pl) else "more_positions" if len(positions) > len(pl)
           else "fewer_positions"] += 1
        lo, hi = d["bb_min"], d["bb_max"]
        if lo is None or hi is None:
            continue
        for tag, conj in (("stored", False), ("conjugate", True)):
            rots = [(p["position"], WA._rotation(p["rotation"], conj)) for p in pl if p["position"]]
            out = []
            for q in positions:
                locs = [_local(q, p, r) for p, r in rots]
                out.append(min(_outside(v, lo, hi) for v in locs))
                if any(rn != LIMBO and rlo and rhi and _outside(v, rlo, rhi) == 0.0
                       for v in locs for rn, rlo, rhi in d["rooms"]):
                    summary["positions_in_a_room_box"][tag] += 1
            row["outside_box_m_" + tag] = [round(x, 3) for x in out]
            dist[tag].extend(out)
    summary["distance_outside_own_interior_box"] = {k: _buckets(v) for k, v in dist.items()}
    return summary, rows


def _inside(path, folder):
    """True when `path` resolves inside `folder`, whose files the scan reads. A folder is scanned whole, not only
    its .bin files, so an --out under it replaces one of the inputs that the next run would read."""
    folder = path_key(folder)
    parent = path_key(os.path.dirname(os.path.abspath(path)))
    return parent == folder or parent.startswith(folder.rstrip(os.sep) + os.sep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="levels.rpf, one carved 0x7a file, or a folder of carved files")
    ap.add_argument("--names", required=True, help="GameShape__Data__LocationNameLoopkup JSON (gamedata_reader.py)")
    ap.add_argument("--start", type=int, help="first byte of an RPF9 data-area window")
    ap.add_argument("--limit", type=int, help="window size in bytes (default: to the end of the archive)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel windows over an RPF9 archive (default 1)")
    ap.add_argument("--out", default=os.path.join("reports", "world_interior_names.json"))
    a = ap.parse_args(argv)
    for p in (a.path, a.names):
        if not os.path.exists(p):
            ap.error("no such file or folder: %s" % p)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if a.start is not None and a.start < 0:
        ap.error("--start must not be negative")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    # Check the report path before the scan, as world_archetypes.py does: --out naming the archive or the names JSON
    # replaced that input (the run exited 0 and the names were gone), --out naming a folder ended in a PermissionError,
    # and a parent that is a regular file failed in os.makedirs, all after the work was done.
    for source in (a.path, a.names):
        if same_report_path(a.out, source):
            ap.error("--out would overwrite the input %s" % source)
    if os.path.isdir(a.out):
        ap.error("--out %s is a folder, not a file" % a.out)
    if os.path.isdir(a.path) and _inside(a.out, a.path):
        ap.error("--out %s is inside the scanned folder %s" % (a.out, a.path))
    out_dir = os.path.dirname(os.path.abspath(a.out))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        ap.error("--out %s: cannot create %s: %s" % (a.out, out_dir, e.strerror or e))
    try:
        names = load_names(a.names)
    except OSError as e:
        ap.error("%s: %s" % (a.names, e.strerror or e))
    except ValueError as e:
        ap.error("%s %s" % (a.names, e))
    try:
        found = scan(a.path, a.start, a.limit, a.jobs)
    except ValueError as exc:
        ap.error(str(exc))
    summary, rows = resolve(names, found)
    try:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "keys": rows}, f, indent=1, sort_keys=True)
    except OSError as e:
        ap.error("--out %s: %s" % (a.out, e.strerror or e))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
