#!/usr/bin/env python3
"""Turn every stored SGD5 pass into the graphics pipeline it names.

    python -I tools/sgd5_pass_pipelines.py <shaders.rpf> <carve index.json> --out manifest.json \
        [--spirv-dir DIR] [--passes passes.json] [--states states.json]

Each pass selects programs per group (0 = VS or MS, 1 = PS, 2 = GS, 5 = CS); a program joins to the carve index
(shader_carve's index.json) by its carve offset. The manifest lists every distinct graphics program set that has
a PS, in the format ``tools/shader_link_check.py batch`` reads. ``--spirv-dir`` drops sets whose .spv is absent.
``--passes`` writes one row per pass with its programs, stages, the raster/depth/blend row bytes (6, 7, 8) and
the decoded rows as raster_state/depth_state/blend_state (null for an empty table); ``--states`` writes the
distinct decoded rows with the number of stored passes using each.
A selected row outside the measured value ranges is an error even without ``--passes``: the tool exits 1 rather
than decode a layout it has not seen.
The archive is opened read-only and read in place, one database at a time.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sgd5_effects as S  # noqa: E402
from shader_link_check import CHAINS  # noqa: E402  (the stage sets `batch` accepts)

PASS_SIZE = 24


def stage_key(group, entry):
    """Manifest key for a program: group 0 is vs or ms by the shader's own stage."""
    if group == 0:
        return "ms" if entry["stage"] == "ms" else "vs"
    return {1: "ps", 2: "gs", 5: "cs"}.get(group, "g%d" % group)


CULL = {1: "none", 2: "front", 3: "back"}
COMPARE = {1: "never", 2: "less", 3: "equal", 4: "less_equal", 5: "greater", 6: "not_equal", 7: "greater_equal",
           8: "always"}
KINDS = ("raster", "depth", "blend")


def decode_raster(row):
    """16 bytes: fill mode (3 in every stored row), cull mode 1..3; the rest stays raw."""
    if row[0] != 3 or row[1] not in CULL:
        raise ValueError("rasterizer row outside the measured ranges")
    return {"fill_mode": row[0], "cull_mode": row[1], "cull": CULL[row[1]], "raw_from_2": list(row[2:])}


def decode_depth(row):
    """24 bytes: depth enable, depth write mask, depth compare, stencil enable; the rest stays raw."""
    if max(row[0], row[1], row[3]) > 1 or row[2] not in COMPARE:
        raise ValueError("depth/stencil row outside the measured ranges")
    return {"depth_enable": row[0], "depth_write": row[1], "depth_func": row[2], "depth_compare": COMPARE[row[2]],
            "stencil_enable": row[3], "raw_from_4": list(row[4:])}


def decode_blend(row):
    """68 bytes: a 4-byte prefix and eight 8-byte target entries (enable, six raw bytes, write mask)."""
    targets = []
    for i in range(8):
        entry = row[4 + 8 * i:12 + 8 * i]
        if entry[0] > 1 or entry[7] > 15:
            raise ValueError("blend row outside the measured ranges")
        targets.append({"enable": entry[0], "write_mask": entry[7], "raw": list(entry[1:7])})
    return {"prefix_raw": list(row[:4]), "targets": targets}


DECODERS = (decode_raster, decode_depth, decode_blend)


def pass_states(data, effect, raw, cache=None):
    """Decode the rows bytes 6, 7, 8 of a pass select; None for an empty table, ValueError out of range.

    ``cache``, a dict kept for one effect, makes passes that select the same row share one decoded object."""
    cache = {} if cache is None else cache
    out = []
    for kind, decode, selector, (count, first, width) in zip(KINDS, DECODERS, raw[6:9], effect.tables):
        if selector >= max(count, 1):
            raise ValueError(f"{kind} selector {selector} is outside the effect's {count} rows")
        if (kind, selector) not in cache:
            row = data[first + selector * width:first + (selector + 1) * width]
            cache[(kind, selector)] = decode(row) if count else None
        out.append(cache[(kind, selector)])
    return out


def census(archive, index, databases=None, states=None):
    """Return (rows, counts); one row per stored pass with a program in the carve index.

    ``states``, a Counter, gets one count per stored pass for each (kind, decoded row as JSON)."""
    by_offset = {}
    for entry in index:
        # Two different entries at one carve offset would make the result depend on the index order.
        if by_offset.setdefault(entry["offset"], entry) != entry:
            raise ValueError(f"carve index has two different entries at offset {entry['offset']:#x}")
    counts = collections.Counter()
    rows = []
    with open(archive, "rb") as stream:
        for database, first_group, end in (S.DATABASES if databases is None else databases):
            stream.seek(database)
            if stream.read(4) != b"SGD5":
                raise ValueError(f"expected an SGD5 database at {database:#x}")
            stream.seek(first_group)
            groups = S.program_groups(stream, end)
            count = S.u32(stream.read(4))
            base = stream.tell()
            data = stream.read(end - base)
            if len(data) != end - base:
                raise ValueError("truncated effect region")
            carve = {}
            for effect in S.effect_records(data, groups, count):
                cursor = effect.first_pass
                decoded_rows = {}  # (kind, selector) -> decoded row, shared by the effect's passes
                as_json = {}
                for tindex, (tname, passes) in enumerate(S.techniques(data, effect)):
                    for programs in passes:
                        raw = data[cursor:cursor + PASS_SIZE]
                        cursor += PASS_SIZE
                        counts["passes"] += 1
                        decoded = pass_states(data, effect, raw, decoded_rows)
                        for kind, selector, state in zip(KINDS, raw[6:9], decoded):
                            if state is None:
                                counts[f"passes with an empty {kind} table"] += 1
                            if states is not None:
                                if (kind, selector) not in as_json:
                                    as_json[(kind, selector)] = json.dumps(state, sort_keys=True)
                                states[(kind, as_json[(kind, selector)])] += 1
                        if not programs:
                            counts["skipped: no program"] += 1
                            continue
                        entries = {}
                        for group, ordinal in programs:
                            if (group, ordinal) not in carve:
                                ident = S.program_identity(stream, group, ordinal, groups[group][ordinal])
                                carve[(group, ordinal)] = by_offset.get(ident["carve_offset"])
                            entries[group] = carve[(group, ordinal)]
                        if any(e is None for e in entries.values()):
                            counts["skipped: program not in carve index"] += 1
                            continue
                        keys = {stage_key(g, e): os.path.splitext(e["file"])[0] for g, e in entries.items()}
                        rows.append({"db": database, "effect_start": base + effect.start, "technique": tindex,
                                     "programs": {str(g): os.path.splitext(e["file"])[0] for g, e in entries.items()},
                                     "stages": {str(g): e["stage"] for g, e in entries.items()},
                                     "raster": raw[6], "depth": raw[7], "blend": raw[8],
                                     "raster_state": decoded[0], "depth_state": decoded[1],
                                     "blend_state": decoded[2], "keys": keys})
                if cursor != effect.pass_end:
                    raise ValueError("pass cursor does not reach the end of the effect's passes")
    return rows, counts


def pipelines_from(rows, counts, spirv_dir=None):
    """Distinct graphics sets with a PS, sorted; updates counts with the skip reasons."""
    spv = None
    if spirv_dir is not None:
        spv = {f[:-4] for f in os.listdir(spirv_dir) if f.endswith(".spv")}
    sets = set()
    for row in rows:
        keys = row["keys"]
        counts["stages " + "+".join(sorted(keys))] += 1
        if "cs" in keys:
            counts["skipped: compute"] += 1
        elif "ps" not in keys:
            counts["skipped: no PS"] += 1
        elif tuple(s for s in ("vs", "ms", "gs", "ps") if s in keys) not in CHAINS:
            counts["skipped: a stage set shader_link_check does not take"] += 1
        else:
            sets.add(tuple(sorted(keys.items())))
    out = []
    for item in sorted(sets):
        keys = dict(item)
        if spv is not None and not all(name in spv for name in keys.values()):
            counts["skipped: no .spv"] += 1
            continue
        first = keys.get("vs") or keys["ms"]
        pipe = {"name": "p%05d" % (len(out) + 1)}
        for stage in ("vs", "ms", "gs", "ps"):
            if stage in keys:
                pipe[stage] = keys[stage]
        pipe["name"] += "_" + first
        out.append(pipe)
    counts["distinct graphics sets with a PS"] = len(sets)
    counts["pipelines written"] = len(out)
    return out


def write_json(path, doc):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", help="shaders.rpf (read only)")
    ap.add_argument("index", help="carve index.json")
    ap.add_argument("--out", required=True, help="shader_link_check batch manifest to write")
    ap.add_argument("--spirv-dir", help="leave out sets whose .spv is missing here")
    ap.add_argument("--passes", help="write per-pass rows here")
    ap.add_argument("--states", help="write the distinct decoded render-state rows with pass use counts here")
    args = ap.parse_args(argv)
    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    states = collections.Counter() if args.states else None
    try:
        rows, counts = census(args.archive, index, states=states)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    pipelines = pipelines_from(rows, counts, args.spirv_dir)
    if pipelines:
        # `shader_link_check.py batch` refuses an empty list, so an empty manifest is never written.
        write_json(args.out, {"version": 1, "pipelines": pipelines})
    if args.passes:
        write_json(args.passes, [{k: v for k, v in r.items() if k != "keys"} for r in rows])
    if args.states:
        write_json(args.states, {kind: [{"state": json.loads(text), "passes": n}
                                        for (k, text), n in sorted(states.items()) if k == kind]
                                 for kind in KINDS})
    for key, value in sorted(counts.items()):
        print(f"{value:8d}  {key}")
    if not pipelines:
        print(f"error: no pipeline left to write; {args.out} not written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
