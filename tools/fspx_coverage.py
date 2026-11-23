"""Count per-program FSPX translation outcomes without reading shader contents."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_index(path, missing):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return missing


def coverage(unpacked, output):
    inventory = read_index(unpacked / "index.json", None)
    translated = read_index(output / "index.json", {})
    refused = read_index(output / "refusals.json", [])
    if not isinstance(inventory, list):
        raise ValueError("expected an unpacked program list in the input index.json")
    if not isinstance(translated, dict):
        raise ValueError("expected a translator object in the output index.json")
    if not isinstance(refused, list) or not all(isinstance(name, str) for name in refused):
        raise ValueError("expected a list of program names in refusals.json")
    refused = set(refused)
    seen = set()
    counts = dict.fromkeys(("programs", "eligible", "translated", "refused", "unaccounted"), 0)
    gaps = Counter()
    for row in inventory:
        file = row.get("file") if isinstance(row, dict) else None
        if (not isinstance(file, str) or not file.endswith(".dxbc") or len(file) == 5
                or any(c in file for c in "/\\:")):
            raise ValueError("expected an unpacked .dxbc filename")
        name = file[:-5]
        if name.casefold() in seen:
            raise ValueError("duplicate unpacked program name")
        seen.add(name.casefold())
        counts["programs"] += 1
        listing = (unpacked / "disasm" / (name + ".s")).is_file()
        counts["eligible"] += int(listing)
        indexed = name in translated
        module = (output / (name + ".spv")).is_file()
        if not listing:
            reason = "missing_listing"
        elif name in refused and (indexed or module):
            reason = "conflicting_results"
        elif name in refused:
            counts["refused"] += 1
            continue
        elif indexed and module:
            counts["translated"] += 1
            continue
        elif indexed:
            reason = "missing_module"
        elif module:
            reason = "unindexed_module"
        else:
            reason = "no_result"
        gaps[reason] += 1
        counts["unaccounted"] += 1
    return {**counts, "unaccounted_by_reason": dict(sorted(gaps.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unpacked", type=Path, help="fspx_unpack.py output directory")
    parser.add_argument("output", type=Path, help="matching rdna2spirv.py output directory")
    args = parser.parse_args()
    try:
        result = coverage(args.unpacked, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"cannot report coverage: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
