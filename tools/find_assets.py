#!/usr/bin/env python3
"""Helper utility to filter, categorize, and locate extracted .obj assets."""

import argparse
import math
import re
import sys
from pathlib import Path

CATEGORIES = {
    "characters": re.compile(r"^(cs_|ig_|hc_|player_|a_c_|crm_)", re.IGNORECASE),
    "hair": re.compile(r"(hair|beard)", re.IGNORECASE),
    "minigames": re.compile(r"(minigame|bfyb|sea_tr|crashed_heli)", re.IGNORECASE),
    "map": re.compile(r"^s\d+_w\d+", re.IGNORECASE),
}


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"--top must be >= 0, got {parsed}")
    return parsed


def finite_non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            f"--min-size must be a finite number >= 0, got {value!r}"
        )
    return parsed


def scan_assets(work_dir: Path, category: str, min_kb: float, top_n: int) -> int:
    if not work_dir.is_dir():
        print(f"[!] Work directory not found or is not a directory: {work_dir}", file=sys.stderr)
        return 1

    assets_candidate = work_dir / "assets"
    if assets_candidate.exists():
        if not assets_candidate.is_dir():
            print(f"[!] Expected directory at: {assets_candidate}", file=sys.stderr)
            return 1
        assets_dir = assets_candidate
    else:
        assets_dir = work_dir

    min_bytes = int(min_kb * 1024)
    pattern = CATEGORIES.get(category)
    matches = []

    for obj_path in assets_dir.rglob("*.obj"):
        if not obj_path.is_file():
            continue
        try:
            size = obj_path.stat().st_size
        except OSError:
            continue

        if size < min_bytes:
            continue

        if pattern and not pattern.search(obj_path.name):
            continue

        matches.append((size, obj_path))

    matches.sort(key=lambda item: item[0], reverse=True)

    shown = min(top_n, len(matches))
    print(f"Found {len(matches)} matching .obj models (showing top {shown}):\n")
    for size, path in matches[:top_n]:
        size_mb = size / (1024 * 1024)
        print(f"[{size_mb:6.2f} MB] {path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter and sort extracted .obj models from Leonida asset dumps."
    )
    parser.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Path to the Leonida work directory (e.g. D:\\Leonida_work)",
    )
    parser.add_argument(
        "--category",
        choices=["all", "characters", "hair", "minigames", "map"],
        default="all",
        help="Filter models by asset category (default: all)",
    )
    parser.add_argument(
        "--min-size",
        type=finite_non_negative_float,
        default=10.0,
        help="Minimum file size in KB to skip empty/stub meshes (default: 10)",
    )
    parser.add_argument(
        "--top",
        type=non_negative_int,
        default=25,
        help="Number of largest models to display (default: 25)",
    )
    args = parser.parse_args(argv)
    return scan_assets(args.work, args.category, args.min_size, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
