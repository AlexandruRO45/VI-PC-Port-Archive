"""Join deferred SV_Depth declarations to one translation inventory and optional sweep tag.

New rows identify the FSSM payload directly. Legacy rows require the original sample.json
and translation inventory for that tag. Readback size and worker status qualify both forms.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ps_sweep import source_identity
from feature_scan import parts, signature_offsets
from fssm_unpack import unpack
from spirv_interface import DEC_BUILTIN, OP_DECORATE, instructions, parse_words


def declarations(directory):
    payloads = []
    counts = Counter()
    for row in json.loads((directory / "index.json").read_bytes()):
        if row["stage"] != "ps" or "deferred" not in row["name"].casefold():
            continue
        counts["deferred_entries"] += 1
        try:
            data = (directory / row["file"]).read_bytes()
            sig = parts(data)["OSG1"]
            if not any(struct.unpack_from("<I", sig, off + 12)[0] == 65
                       for off in signature_offsets(sig)):
                continue
            counts["declared_depth"] += 1
            blob, _ = unpack(data)
            if blob is None:
                raise ValueError("missing payload")
            payloads.append(hashlib.sha256(blob).hexdigest())
        except (OSError, ValueError, KeyError, zlib.error, struct.error):
            counts["unreadable_evidence"] += 1
    return payloads, counts


def sweep_cases(results, index, counts):
    measured = {}
    receipts = {}
    legacy = False
    for log in results.glob("results_*.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            i = row["i"]
            if type(i) is not int or i < 0 or i in measured:
                raise ValueError("duplicate or out-of-range sweep result")
            if "source_identity" in row:
                source = row["source_identity"]
                digest = source_identity(source).get("fssm_sha256") if isinstance(source, dict) else None
                receipts[i] = digest
                if digest is None:
                    counts["rows_without_fssm_identity"] += 1
            else:
                legacy = True
            planes = [results / f"{i}.raw.depth"]
            if row.get("two_clear"):
                planes.append(results / f"{i}.raw.zero.depth")
            measured[i] = (row["code"] == 0 and row.get("depth") is True
                           and all(p.is_file() and p.stat().st_size == 32 * 32 * 4 for p in planes))
    cases = defaultdict(list)
    for i, digest in receipts.items():
        if digest:
            cases[digest].append(i)
    if legacy or (not measured and index is not None):
        if index is None:
            raise ValueError("legacy rows require the original translation inventory")
        sample = json.loads((results.parent / "sample.json").read_bytes())
        if any(i >= len(sample) for i in measured if i not in receipts):
            raise ValueError("out-of-range legacy sweep result")
        for i, row in enumerate(sample):
            if i in receipts:
                continue
            meta = index.get(row["name"], {})
            if meta.get("stage") == "ps":
                digest = source_identity(meta).get("fssm_sha256")
                if digest:
                    cases[digest].append(i)
    return cases, measured


def coverage(payloads, translation=None, results=None):
    index = json.loads((translation / "index.json").read_bytes()) if translation is not None else None
    names = defaultdict(list)
    counts = Counter()
    for name, meta in (index or {}).items():
        if meta["stage"] == "ps":
            names[source_identity(meta).get("fssm_sha256")].append(name)
    multiplicity = Counter(payloads)
    if results is not None:
        cases, measured = sweep_cases(results, index, counts)
        counts["successful_depth_cases_in_tag"] = sum(measured.values())
    for digest in payloads:
        matches = names[digest]
        if multiplicity[digest] > 1 or len(matches) > 1:
            counts["ambiguous_identity"] += 1
            continue
        if results is not None:
            ids = cases[digest]
            if len(ids) > 1:
                counts["ambiguous_sample"] += 1
            elif not ids and counts["rows_without_fssm_identity"]:
                counts["unknown_sample_identity"] += 1
            elif not ids:
                counts["unsampled"] += 1
            elif measured.get(ids[0]):
                counts["depth_measured"] += 1
            else:
                counts["sample_without_depth_measurement"] += 1
        if index is None:
            continue
        if not matches:
            counts["unresolved_identity" if names[None] else "absent_from_index"] += 1
            continue
        name, = matches
        counts["matched_index"] += 1
        try:
            words = parse_words((translation / (name + ".spv")).read_bytes())
            depth = any(op == OP_DECORATE and operands[1:] == (DEC_BUILTIN, 22)
                        for op, operands, _ in instructions(words))
        except (OSError, ValueError):
            counts["unreadable_module"] += 1
            continue
        counts["frag_depth" if depth else "no_frag_depth"] += 1
        if index[name].get("bindings"):
            counts["requires_descriptors"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxbc", type=Path)
    parser.add_argument("translation", type=Path, nargs="?", help="inventory for module checks or legacy rows")
    parser.add_argument("--results", type=Path, help="sweep tag directory (legacy logs also need the adjacent original sample.json)")
    args = parser.parse_args()
    if args.translation is None and args.results is None:
        parser.error("supply a translation inventory or --results")
    payloads, counts = declarations(args.dxbc)
    counts.update(coverage(payloads, args.translation, args.results))
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
