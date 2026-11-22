"""Pick a world prop for tools/first_light.py from a `rpf9dump models` export, by rule.

`rpf9dump models` writes one OBJ per drawable and a `manifest.json` next to them. For levels.rpf that manifest is
several GB, so this tool streams it one item at a time (a few KB of memory) and never loads it whole.

The rule (docs/first_light.md section 7): a static drawable (not a fragment, no bone bindings), exactly one mesh,
300 to 3,000 triangles, a bounding box that is not degenerate (every extent at least 5% of the longest, so a flat
card or a thin rod is out), and an OBJ that carries UVs. Of the candidates, the one whose triangle count is
closest to 1,000 wins, ties broken by file name, so the choice is repeatable. The tool prints the OBJ path, its
counts and its SHA-256; the name stays on the user's machine.

    python -I tools/first_light_prop.py <models-dir> [--limit 10]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

MIN_TRIS, MAX_TRIS, TARGET_TRIS = 300, 3000, 1000
MIN_EXTENT_RATIO = 0.05
SCALARS = ("File", "Meshes", "Verts", "Tris", "IsFragment", "HasBoneBindings")


def stream_items(lines):
    """Yield one dict per manifest item: File, Meshes, Verts, Tris, IsFragment, HasBoneBindings, Min, Max.

    The manifest is indented JSON (items at 4 spaces, their fields at 6, array elements at 8); only those fields
    are read, so nested material lists are skipped without being parsed.
    """
    item, array = None, None
    for line in lines:
        line = line.rstrip("\r\n")
        if item is None:
            if line == "    {":
                item = {}
            continue
        if line in ("    }", "    },"):
            yield item
            item, array = None, None
        elif array is not None:
            if line.startswith("      ]"):
                array = None
            elif line.startswith("        "):
                item[array].append(float(line.strip().rstrip(",")))
        elif line.startswith("      ") and not line.startswith("       "):
            key, _, value = line.strip().partition(": ")
            key = key.strip('"')
            value = value.rstrip(",")
            if key in ("Min", "Max") and value == "[":
                array = key
                item[key] = []
            elif key in SCALARS:
                item[key] = value.strip('"') if key == "File" else {"true": True, "false": False}.get(value, value)
                if key in ("Meshes", "Verts", "Tris"):
                    item[key] = int(value)


def qualifies(item: dict) -> bool:
    """The rule, on a manifest item."""
    if item.get("IsFragment") is not False or item.get("HasBoneBindings") is not False:
        return False
    if item.get("Meshes") != 1 or not MIN_TRIS <= item.get("Tris", 0) <= MAX_TRIS or not item.get("Verts"):
        return False
    lo, hi = item.get("Min"), item.get("Max")
    if not lo or not hi or len(lo) != 3 or len(hi) != 3 or not item.get("File"):
        return False
    extents = [b - a for a, b in zip(lo, hi)]
    return min(extents) > 0 and min(extents) >= MIN_EXTENT_RATIO * max(extents)


def rank(item: dict):
    return abs(item["Tris"] - TARGET_TRIS), item["File"]


def has_uv(path: str) -> bool:
    with open(path, encoding="utf-8", errors="replace") as f:
        return any(line.startswith("vt ") for line in f)


def _front(items: list[dict], models_dir: str, keep: int) -> list[dict]:
    """The first `keep` items in rule order whose OBJ exists and has UVs. Each OBJ is opened at most once."""
    out = []
    for item in sorted(items, key=rank):
        if "_usable" not in item:
            path = os.path.join(models_dir, item["File"])
            item["_usable"] = os.path.isfile(path) and has_uv(path)
        if item["_usable"]:
            out.append(item)
            if len(out) == keep:
                break
    return out


def pick(models_dir: str, limit: int = 1) -> list[dict]:
    """The best `limit` candidates (rule order) whose OBJ exists and has UVs."""
    best: list[dict] = []
    with open(os.path.join(models_dir, "manifest.json"), encoding="utf-8", errors="replace") as f:
        for item in stream_items(f):
            if qualifies(item):
                best.append(item)
                if len(best) > 4 * limit + 64:
                    # Bounded memory. Only usable items are kept, so an item dropped here ranks below
                    # 2 * limit + 32 usable ones and can't be among the best `limit`.
                    best = _front(best, models_dir, 2 * limit + 32)
    found = _front(best, models_dir, limit)
    for item in found:
        item.pop("_usable", None)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", help="the folder `rpf9dump models` wrote (OBJs and manifest.json)")
    ap.add_argument("--limit", type=int, default=1, help="how many candidates to list (default 1: the pick)")
    a = ap.parse_args(argv)
    found = pick(os.path.abspath(a.models), a.limit)
    if not found:
        raise SystemExit("no drawable in the manifest passes the rule")
    for item in found:
        path = os.path.join(os.path.abspath(a.models), item["File"])
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        print(f"{path}\t{item['Verts']} vertices\t{item['Tris']} triangles\tsha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
