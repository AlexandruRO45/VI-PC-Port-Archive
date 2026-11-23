"""Check manifest-listed DDS files in an existing texture pool export.

    python -I tools/check_texture_pools.py <assets-dir-or-pool-dir>

Read only the manifests and DDS headers. Report counts by pool when a check fails.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dds_png import DDS_DX10_HEADER_SIZE, DdsError, UnsupportedDdsFormat, minimum_dds_size  # noqa: E402


PROBLEMS = ("invalid_manifest", "missing", "unreadable", "invalid_dds", "truncated", "unsupported")
READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)


def pool_manifests(root: Path) -> list[Path]:
    manifest = root / "manifest.json"
    if manifest.exists():
        return [manifest]
    return sorted((*root.glob("ttpf/*/manifest.json"), *root.glob("ttpf_pack/*/*/manifest.json")))


def manifest_files(manifest: Path) -> list[str | None]:
    """Read paths from one pool. A null File marks a deliberately unexported texture."""
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    rows = doc.get("Textures") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or type(doc.get("Count")) is not int or doc["Count"] != len(rows):
        raise ValueError("Invalid texture count or rows")
    names = []
    for row in rows:
        if not isinstance(row, dict) or "File" not in row:
            raise ValueError("Missing File field")
        name = row["File"]
        if name is not None and (not isinstance(name, str) or not name.lower().endswith(".dds")
                                 or any(char in name for char in "/\\:\0")):
            raise ValueError("File must name a DDS in this pool directory")
        names.append(name)
    return names


def check_dds(path: Path) -> str:
    try:
        fd = os.open(path, READ_FLAGS)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return "unreadable"
            header = os.read(fd, DDS_DX10_HEADER_SIZE)
        finally:
            os.close(fd)
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"

    try:
        required = minimum_dds_size(header)
    except UnsupportedDdsFormat:
        return "unsupported"
    except DdsError:
        return "invalid_dds"
    return "truncated" if info.st_size < required else "valid"


def check_pool(manifest: Path) -> Counter[str]:
    try:
        names = manifest_files(manifest)
    except (OSError, ValueError, RecursionError):
        return Counter(invalid_manifest=1)

    count: Counter[str] = Counter()
    for name in names:
        result = "not_exported" if name is None else check_dds(manifest.parent / name)
        count[result] += 1
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check DDS headers and sizes against an existing texture-pool export.")
    parser.add_argument("root", type=Path, help="exported assets directory or one texture pool directory")
    root = parser.parse_args(argv).root
    manifests = pool_manifests(root)
    if not manifests:
        parser.error(f"no texture pool manifests found under {root}")

    total: Counter[str] = Counter()
    for manifest in manifests:
        count = check_pool(manifest)
        total.update(count)
        issues = ", ".join(f"{key}={count[key]}" for key in PROBLEMS if count[key])
        if issues:
            print(f"{manifest.parent.relative_to(root)}: {issues}")
    counts = ", ".join(f"{key}={total[key]}" for key in ("valid", "not_exported", *PROBLEMS))
    print(f"Totals: pools={len(manifests)}, {counts}")
    return int(any(total[key] for key in PROBLEMS))


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    raise SystemExit(main())
