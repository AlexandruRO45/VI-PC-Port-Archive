#!/usr/bin/env python3
"""Count stored graphics passes by shader chain and raster, depth and blend selections.

Read the private --passes JSON from sgd5_pass_pipelines.py. A decoded row is
compared by its complete value, including uninterpreted bytes. An empty table
has an unknown runtime default. Older exports only identify local selectors.
Both are scoped to database and effect. Groups describe stored selections,
not runtime pipeline objects. The detailed output is for local use only.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shader_link_check import CHAINS


def selection(row, kind):
    index = row[kind]
    if type(index) is not int or not 0 <= index <= 255:
        raise ValueError(f"invalid {kind} selector")
    field = kind + "_state"
    if field in row and row[field] is not None:
        if not isinstance(row[field], dict):
            raise ValueError(f"invalid decoded {kind} row")
        return {"value": row[field]}
    local = {"db": row["db"], "effect_start": row["effect_start"]}
    if any(type(offset) is not int or offset < 0 for offset in local.values()):
        raise ValueError("invalid effect provenance")
    if field in row:
        if index != 0:
            raise ValueError(f"empty {kind} table with nonzero selector")
        return {"empty_table": local}
    return {"local_row": dict(local, selector=index)}


def inventory(rows):
    chains = collections.defaultdict(collections.Counter)
    for row in rows:
        programs = row["programs"]
        stages = row["stages"]
        if programs.keys() != stages.keys():
            raise ValueError("program and stage groups differ")
        groups = sorted(programs)
        if groups not in (["0", "1"], ["0", "1", "2"]):
            continue
        if stages["1"] != "ps" or ("2" in stages and stages["2"] != "gs"):
            raise ValueError("unexpected graphics program stage")
        ordered = ["0", "2", "1"] if "2" in stages else ["0", "1"]
        chain = tuple(stages[g] for g in ordered)
        if chain not in CHAINS:
            continue
        shaders = tuple((stages[g], programs[g]) for g in ordered)
        states = {k: selection(row, k) for k in ("raster", "depth", "blend")}
        chains[shaders][json.dumps(states, sort_keys=True)] += 1

    result = []
    for shaders, counts in sorted(chains.items()):
        variants = [{"states": json.loads(key), "passes": n} for key, n in sorted(counts.items())]
        result.append({"shaders": dict(shaders), "variants": variants})
    return {"chains": result, "summary": {
        "graphics_passes": sum(sum(c.values()) for c in chains.values()),
        "shader_chains": len(chains),
        "selection_groups": sum(map(len, chains.values())),
        "chains_with_multiple_selections": sum(len(c) > 1 for c in chains.values()),
    }}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("passes", help="cached per-pass JSON")
    parser.add_argument("--out", help="write detailed private inventory (default: print counts only)")
    args = parser.parse_args(argv)
    try:
        with open(args.passes, encoding="utf-8") as f:
            report = inventory(json.load(f))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.error(f"invalid pass metadata: {exc}")
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
