#!/usr/bin/env python3
"""Build a private replay manifest from one null-usage trace.

Share only --aggregate output. The rank is a lower bound from the reported
top rows, not an exhaustive list or proof of render dependency.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.triage_scarlett_shaders import requests

HEX = r"[0-9a-f]{64}"
STAGE = rf"(?:VS|PS|DS|HS|GS|CS|AS|MS)=(?:{HEX}|unknown)"
ROW = re.compile(
    rf"null usage rank kind=(compute|raster) ((?:{STAGE})(?: {STAGE})*) "
    r"recorded=([0-9]{1,20}) display_bound=([0-9]{1,20}) "
    r"interval=([0-9]{1,20}) trigger=(?:present_sample|recording_checkpoint)$")
SUMMARY = re.compile(
    r"null usage interval recorded=([0-9]{1,20}) display_bound=([0-9]{1,20}) "
    r"tracked=([0-9]{1,4}) overflow=([0-9]{1,20}) overflow_display_bound=([0-9]{1,20}) "
    r"compute=([0-9]{1,20}) raster=([0-9]{1,20}) interval=([0-9]{1,20}) "
    r"trigger=(?:present_sample|recording_checkpoint)$")


def build(lines, sources, limit=64):
    summaries, ranked = {}, {}
    for line in lines:
        # Only the provider's diagnostic records are meaningful.
        marker = "TuxBox d3d12_xs: "
        if marker not in line:
            continue
        message = line.rstrip().split(marker, 1)[1]
        if m := SUMMARY.fullmatch(message):
            total, display, tracked, overflow, overflow_display, compute, raster, interval = map(int, m.groups())
            if interval == 0 or interval in summaries:
                raise ValueError("duplicate/invalid interval; supply one process trace")
            if display > total or overflow > total or overflow_display > min(display, overflow) or compute+raster != total or tracked > 1024:
                raise ValueError("inconsistent usage summary")
            summaries[interval] = (total, display, tracked, overflow, overflow_display)
        elif m := ROW.fullmatch(message):
            kind, fields, count, display, interval = m.groups()
            count, display, interval = map(int, (count, display, interval))
            stages = tuple(sorted(tuple(field.split("=", 1)) for field in fields.split()))
            if count == 0 or display > count or (kind == "compute" and display) or interval == 0 or len({s for s,_ in stages}) != len(stages):
                raise ValueError("invalid usage rank")
            key = (interval, kind, stages)
            if key in ranked:
                raise ValueError("duplicate usage rank; supply one process trace")
            ranked[key] = (count, display)
    if not summaries:
        raise ValueError("no null usage interval summaries")
    by_interval = defaultdict(list)
    for (interval, kind, stages), (count, display) in ranked.items():
        by_interval[interval].append((kind, stages, count, display))
    scores = defaultdict(lambda: [0,0])
    pairs = defaultdict(lambda: [0,0])
    unknown_rows = 0
    incomplete = 0
    for interval, (total, display, tracked, overflow, overflow_display) in summaries.items():
        rows = by_interval.pop(interval, [])
        if len(rows) > min(tracked, 8) or sum(row[2] for row in rows) > total-overflow or sum(row[3] for row in rows) > display-overflow_display:
            raise ValueError("rank rows exceed interval summary")
        if len(rows) < min(tracked,8):
            incomplete += 1
        for kind, stages, count, bound in rows:
            ids = {digest for _,digest in stages if digest != "unknown"}
            if any(digest == "unknown" for _, digest in stages):
                unknown_rows += 1
            for digest in ids:
                scores[digest][0] += count
                scores[digest][1] += bound
            stage_map = dict(stages)
            producer = stage_map.get("GS") or stage_map.get("VS")
            pixel = stage_map.get("PS")
            if producer and pixel and producer != "unknown" and pixel != "unknown":
                pairs[(producer,pixel)][0] += count
                pairs[(producer,pixel)][1] += bound
    # Unpaired rank rows can be a truncated trace; don't silently count them.
    orphan_rows = sum(map(len, by_interval.values()))
    candidates = []
    matched = set()
    for path in sources:
        parts = path.stem.split("_")
        if path.suffix == ".link":
            key = tuple(parts[:2])
            score = pairs.get(key)
            ids = set(key)
        else:
            score = scores.get(parts[0])
            ids = {parts[0]}
        if score:
            candidates.append((score[1],score[0],path.name))
            matched.update(ids & scores.keys())
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    chosen = candidates[:limit]
    manifest = {"schema":1,"scope":"Private replay selection from sampled null usage; stage association, not causal attribution",
                "requests":[name for _,_,name in chosen]}
    aggregate = {"schema":1,"scope":"Sampled null usage; lower-bound stage association; no identifiers",
                 "intervals":len(summaries),"incomplete_intervals":incomplete,"orphan_rank_rows":orphan_rows,
                 "unknown_identity_rows":unknown_rows,"recorded":sum(v[0] for v in summaries.values()),
                 "display_bound":sum(v[1] for v in summaries.values()),
                 "overflow":sum(v[3] for v in summaries.values()),
                 "observed_stage_identities":len(scores),"matched_stage_identities":len(matched),
                 "candidate_requests":len(candidates),"selected_requests":len(chosen)}
    return manifest, aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 10000 or not args.requests.is_dir():
        parser.error("limit must be 1..10000 and request directory must exist")
    outputs = (args.manifest,args.aggregate)
    if len({p.resolve() for p in outputs}) != 2 or any(p.exists() or p.resolve().is_relative_to(args.requests.resolve()) for p in outputs):
        parser.error("choose fresh, distinct outputs outside the request archive")
    try:
        with args.log.open(encoding="utf-8", errors="replace") as stream:
            manifest, aggregate = build(stream, requests(args.requests), args.limit)
        if not manifest["requests"]:
            # No local replay candidates; still publish a useful aggregate.
            with args.aggregate.open("x",encoding="utf-8") as f:
                json.dump(aggregate,f,indent=2); f.write("\n")
            print(json.dumps(aggregate))
            return 2
        for path, obj in zip(outputs,(manifest,aggregate)):
            with path.open("x",encoding="utf-8") as f:
                json.dump(obj,f,indent=2); f.write("\n")
    except (OSError,ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(aggregate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
