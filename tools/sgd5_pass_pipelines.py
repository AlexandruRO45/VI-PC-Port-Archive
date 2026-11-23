#!/usr/bin/env python3
"""Turn every stored SGD5 pass into the graphics pipeline it names.

    python -I tools/sgd5_pass_pipelines.py <shaders.rpf> <carve index.json> --out manifest.json \
        [--spirv-dir DIR] [--passes passes.json]

Each pass selects programs per group (0 = VS or MS, 1 = PS, 2 = GS, 5 = CS); a program joins to the carve index
(shader_carve's index.json) by its carve offset. The manifest lists every distinct graphics program set that has
a PS, in the format ``tools/shader_link_check.py batch`` reads. ``--spirv-dir`` drops sets whose .spv is absent.
``--passes`` writes one row per pass with its programs, stages and the raster/depth/blend row bytes (6, 7, 8).
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


def census(archive, index, databases=None):
    """Return (rows, counts); one row per stored pass with a program in the carve index."""
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
                for tindex, (tname, passes) in enumerate(S.techniques(data, effect)):
                    for programs in passes:
                        raw = data[cursor:cursor + PASS_SIZE]
                        cursor += PASS_SIZE
                        counts["passes"] += 1
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
                                     "raster": raw[6], "depth": raw[7], "blend": raw[8], "keys": keys})
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
    args = ap.parse_args(argv)
    with open(args.index, encoding="utf-8") as f:
        index = json.load(f)
    try:
        rows, counts = census(args.archive, index)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    pipelines = pipelines_from(rows, counts, args.spirv_dir)
    if pipelines:
        # `shader_link_check.py batch` refuses an empty list, so an empty manifest is never written.
        write_json(args.out, {"version": 1, "pipelines": pipelines})
    if args.passes:
        write_json(args.passes, [{k: v for k, v in r.items() if k != "keys"} for r in rows])
    for key, value in sorted(counts.items()):
        print(f"{value:8d}  {key}")
    if not pipelines:
        print(f"error: no pipeline left to write; {args.out} not written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
