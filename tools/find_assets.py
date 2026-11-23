#!/usr/bin/env python3
"""Helper utility to filter, categorize, and locate extracted .obj assets."""

import argparse
import re
from pathlib import Path

CATEGORIES = {
    "characters": re.compile(r"^(cs_|ig_|hc_|player_|a_c_|crm_)", re.IGNORECASE),
    "hair": re.compile(r"(hair|beard)", re.IGNORECASE),
    "minigames": re.compile(r"(minigame|bfyb|sea_tr|crashed_heli)", re.IGNORECASE),
    "map": re.compile(r"^s\d+_w\d+", re.IGNORECASE),
}


def scan_assets(work_dir: Path, category: str, min_kb: float, top_n: int) -> None:
    assets_dir = work_dir / "assets"
    if not assets_dir.exists():
        assets_dir = work_dir

    if not assets_dir.exists():
        print(f"[!] Directory not found: {assets_dir}")
        return

    min_bytes = int(min_kb * 1024)
    pattern = CATEGORIES.get(category)
    matches = []

    for obj_path in assets_dir.rglob("*.obj"):
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

    print(f"Found {len(matches)} matching .obj models (showing top {min(top_n, len(matches))}):\n")
    for size, path in matches[:top_n]:
        size_mb = size / (1024 * 1024)
        print(f"[{size_mb:6.2f} MB] {path}")


def main() -> None:
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
        type=float,
        default=10.0,
        help="Minimum file size in KB to skip empty/stub meshes (default: 10)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of largest models to display (default: 25)",
    )
    args = parser.parse_args()
    scan_assets(args.work, args.category, args.min_size, args.top)


if __name__ == "__main__":
    main()
