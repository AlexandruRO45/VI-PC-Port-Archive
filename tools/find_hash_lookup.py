"""Rank candidate hash-lookup functions in a PE image (docs/modding.md, "Day one: finding the hook").

    python -I tools/find_hash_lookup.py game_scarlett_master.exe [--hash player_story000_00 ...] [--window 48]
                                        [--top 15] [--json out.json]

Reads the file only; nothing runs. For each --hash (a model name, hashed with joaat as the game does, or 0x...;
default: Lucia's player model and cs_mandy), it finds every place the executable sections hold that 32-bit value
(an immediate such as `mov ecx, 0xE472B687`) and the direct calls (E8 rel32) in the next --window bytes. Each call
target is a candidate; candidates are ranked by how many of the hashes lead to them, then by how many direct calls
the whole image makes to them. RAGE's archetype lookup by hash takes the hash and is called from hundreds of places,
so it should rank first; pass its RVA to `gta6rt --archetype-lookup`.

Standard library only. No game data is written: the report holds RVAs and counts.
"""
import argparse
import json
import struct
import sys
from bisect import bisect_right
from collections import Counter, defaultdict

IMAGE_SCN_MEM_EXECUTE = 0x20000000
HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def joaat(name):
    """RAGE's joaat: Jenkins one-at-a-time over the lower-cased name (joaat('cs_mandy') == 0xD7C0CDFF)."""
    h = 0
    for c in name.lower().encode("utf-8"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def parse_hash(text):
    if not text:
        raise ValueError("hash must not be empty")
    if text.lower().startswith("0x"):
        digits = text[2:]
        if not digits or any(c not in HEX_DIGITS for c in digits):
            raise ValueError(f"invalid hex hash: {text!r}")
        value = int(digits, 16)
        if value > 0xFFFFFFFF:
            raise ValueError(f"{text} is wider than 32 bits")
        return value
    return joaat(text)


def sections(data):
    """[(name, rva, raw offset, size, executable)] of a PE file (32- or 64-bit)."""
    if data[:2] != b"MZ":
        raise ValueError("not a PE file (no MZ header): an encrypted or packed executable can't be scanned")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("no PE signature")
    count, opt_size = struct.unpack_from("<H12xH", data, pe + 6)
    first = pe + 24 + opt_size
    out = []
    for i in range(count):
        name, vsize, rva, rsize, raw = struct.unpack_from("<8sIIII", data, first + 40 * i)
        flags = struct.unpack_from("<I", data, first + 40 * i + 36)[0]
        out.append((name.rstrip(b"\0").decode("ascii", "replace"), rva, raw, min(vsize or rsize, rsize),
                    bool(flags & IMAGE_SCN_MEM_EXECUTE)))
    return out


def direct_calls(code, base_rva, start=0, end=None):
    """{target rva: [call site rva, ...]} for every E8 rel32 in `code[start:end]` (which starts at base_rva)."""
    if end is None:
        end = len(code)
    calls = defaultdict(list)
    find = code.find
    limit = end - 5
    i = find(b"\xE8", start, end)
    while 0 <= i <= limit:
        rel = struct.unpack_from("<i", code, i + 1)[0]
        site = base_rva + (i - start)
        calls[site + 5 + rel].append(site)
        i = find(b"\xE8", i + 1, end)
    return calls


def rank(data, hashes, window=48, top=15):
    if window < 0:
        raise ValueError("window must be non-negative")
    if top < 0:
        raise ValueError("top must be non-negative")
    code_sections = [s for s in sections(data) if s[4]]
    if not code_sections:
        raise ValueError("no executable section")
    data_len = len(data)
    code_spans = []
    for _, rva, raw, size, _ in code_sections:
        start = min(raw, data_len)
        end = min(raw + size, data_len)
        if end > start:
            code_spans.append((rva, start, end))
    merged = []
    for lo, hi in sorted((rva, rva + (end - start)) for rva, start, end in code_spans):
        if merged and lo <= merged[-1][1]:
            if hi > merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    if not merged:
        def in_code(_target):
            return False
    elif len(merged) == 1:
        lo, hi = merged[0]

        def in_code(target):
            return lo <= target < hi
    else:
        starts = [lo for lo, _ in merged]

        def in_code(target):
            idx = bisect_right(starts, target) - 1
            return idx >= 0 and target < merged[idx][1]
    unique_hashes = []
    seen_values = set()
    for name, value in hashes:
        if value not in seen_values:
            seen_values.add(value)
            unique_hashes.append((name, value))
    all_calls = defaultdict(list)
    for rva, start, end in code_spans:
        for target, sites in direct_calls(data, rva, start, end).items():
            if in_code(target):
                all_calls[target].extend(sites)
    leads = defaultdict(set)      # candidate rva -> hashes that lead to it
    hits = Counter()              # hash value -> occurrences found
    find = data.find
    for name, value in unique_hashes:
        needle = struct.pack("<I", value)
        for rva, start, end in code_spans:
            at = find(needle, start, end)
            while at >= 0:
                hits[value] += 1
                tail_start = at + 4
                tail_end = min(tail_start + window, end)
                tail_limit = tail_end - 5
                j = find(b"\xE8", tail_start, tail_end)
                while 0 <= j <= tail_limit:
                    rel = struct.unpack_from("<i", data, j + 1)[0]
                    target = rva + (j - start) + 5 + rel
                    if in_code(target):
                        leads[target].add(name)
                    j = find(b"\xE8", j + 1, tail_end)
                at = find(needle, at + 1, end)
    ranked = sorted(leads, key=lambda t: (-len(leads[t]), -len(all_calls.get(t, ())), t))[:top]
    return {
        "hashes": [{"name": n, "value": f"0x{v:08X}", "occurrences": hits[v]} for n, v in unique_hashes],
        "candidates": [{"rva": f"0x{t:X}", "led_by": sorted(leads[t]), "direct_callers": len(all_calls.get(t, ()))}
                       for t in ranked],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exe")
    ap.add_argument("--hash", action="append", default=None,
                    help="model name or 0x hash (repeatable; default player_story000_00 and cs_mandy)")
    ap.add_argument("--window", type=int, default=48, help="bytes after a hash searched for calls (default 48)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args(argv)
    if a.window < 0:
        print("ERROR: --window must be non-negative", file=sys.stderr)
        return 2
    if a.top < 0:
        print("ERROR: --top must be non-negative", file=sys.stderr)
        return 2
    try:
        hashes = [(h, parse_hash(h)) for h in (a.hash or ["player_story000_00", "cs_mandy"])]
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        with open(a.exe, "rb") as f:
            data = f.read()
        report = rank(data, hashes, a.window, a.top)
    except (OSError, ValueError, struct.error) as e:
        print(f"ERROR: {a.exe}: {e}", file=sys.stderr)
        return 2
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=1)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: {a.json}: {e}", file=sys.stderr)
            return 2
    for h in report["hashes"]:
        print(f"{h['name']} = {h['value']}: {h['occurrences']} occurrence(s) in code")
    if not report["candidates"]:
        print("no direct call follows any of the hashes: try a larger --window, or the lookup is reached indirectly")
    for c in report["candidates"]:
        print(f"  {c['rva']:>12}  callers {c['direct_callers']:>6}  led by {', '.join(c['led_by'])}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
