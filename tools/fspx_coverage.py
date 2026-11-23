"""Count per-program FSPX translation outcomes and compare validated runs by code identity."""
import argparse
from collections import Counter
import hashlib
import json
import subprocess
from pathlib import Path


def read_index(path, missing):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return missing


def outcomes(unpacked, output, source):
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
    for row in inventory:
        file = row.get("file") if isinstance(row, dict) else None
        if (not isinstance(file, str) or not file.endswith(".dxbc") or len(file) == 5
                or any(c in file for c in "/\\:")):
            raise ValueError("expected an unpacked .dxbc filename")
        name = file[:-5]
        if name.casefold() in seen:
            raise ValueError("duplicate unpacked program name")
        seen.add(name.casefold())
        if source == "listing":
            eligible = (unpacked / "disasm" / (name + ".s")).is_file()
        else:
            eligible = (unpacked / "fspx" / (name + ".code")).is_file()
        indexed = name in translated
        module = (output / (name + ".spv")).is_file()
        if not eligible:
            reason = "missing_" + source
        elif name in refused and (indexed or module):
            reason = "conflicting_results"
        elif name in refused:
            reason = "refused"
        elif indexed and module:
            reason = "translated"
        elif indexed:
            reason = "missing_module"
        elif module:
            reason = "unindexed_module"
        else:
            reason = "no_result"
        yield name, eligible, reason


def coverage(unpacked, output):
    counts = dict.fromkeys(("programs", "eligible", "translated", "refused", "unaccounted"), 0)
    gaps = Counter()
    for _, eligible, result in outcomes(unpacked, output, "listing"):
        counts["programs"] += 1
        counts["eligible"] += int(eligible)
        if result in ("translated", "refused"):
            counts[result] += 1
        else:
            counts["unaccounted"] += 1
            gaps[result] += 1
    return {**counts, "unaccounted_by_reason": dict(sorted(gaps.items()))}


def validated(module):
    result = subprocess.run(["spirv-val", "--target-env", "vulkan1.2",
                             "--allow-offset-texture-operand", str(module)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def by_identity(unpacked, output, source):
    groups = {}
    missing = 0
    for name, _, result in outcomes(unpacked, output, source):
        code = unpacked / "fspx" / (name + ".code")
        try:
            data = code.read_bytes()
        except FileNotFoundError:
            data = b""
        if not data:
            missing += 1
            continue
        if result == "translated":
            result = "validated" if validated(output / (name + ".spv")) else "validation_failed"
        digest = hashlib.sha256(data).digest()
        groups[digest] = "duplicate_identity" if digest in groups else result
    return groups, missing


def compare(before, after):
    """Count transitions once per unique kind-6 code identity, without guessing alias pairings."""
    left, left_missing = by_identity(*before)
    right, right_missing = by_identity(*after)
    transitions = {}
    for identity in left.keys() | right.keys():
        previous = left.get(identity, "absent")
        current = right.get(identity, "absent")
        transitions.setdefault(previous, Counter())[current] += 1
    return {"missing_code_before": left_missing, "missing_code_after": right_missing,
            "transitions": transitions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unpacked", type=Path, help="fspx_unpack.py output directory")
    parser.add_argument("output", type=Path, help="matching rdna2spirv.py output directory")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("UNPACKED", "OUTPUT"),
                        help="compare before to after by SHA-256 of complete .code files; validate indexed modules")
    parser.add_argument("--sources", nargs=2, choices=("listing", "code"),
                        help="input used by each translator run (default: listing listing)")
    args = parser.parse_args()
    if args.sources and not args.compare:
        parser.error("--sources requires --compare")
    try:
        if args.compare:
            left_source, right_source = args.sources or ("listing", "listing")
            result = compare((args.unpacked, args.output, left_source), (*args.compare, right_source))
        else:
            result = coverage(args.unpacked, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"cannot report coverage: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=bool(args.compare)))


if __name__ == "__main__":
    main()
