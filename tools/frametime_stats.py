#!/usr/bin/env python3
"""Compute FPS, percentile frame times, 1% lows and stutter metrics from frame logs.

Supports MangoHud CSV logs, CapFrameX/PresentMon CSVs, and raw frametime lists.
Standard library only.

Usage:
    python -I tools/frametime_stats.py run.csv
    python -I tools/frametime_stats.py --compare baseline.csv tuned.csv
    python -I tools/frametime_stats.py run.csv --json
    python -I tools/frametime_stats.py run.csv --target-fps 60 --stutter-threshold 20.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys


def parse_frametimes_ms(path: str) -> list[float]:
    """Extract a list of frame times in milliseconds from a CSV or text log."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"log file not found: {path}")

    frametimes: list[float] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        # Check first line to detect CSV headers vs bare numbers
        first_line = f.readline()
        if not first_line:
            return []

        # Reset and use csv.reader
        f.seek(0)
        reader = csv.reader(f)

        header_row = next(reader, None)
        if not header_row:
            return []

        col_idx = None
        mode = "auto"  # "ns", "ms", "fps", or "auto" (magnitude heuristic)

        # Detect column header
        lower_headers = [h.strip().lower() for h in header_row]
        for idx, h in enumerate(lower_headers):
            if h in ("frametime_ns", "frametime(ns)", "frametime_nsec"):
                col_idx = idx
                mode = "ns"
                break
            if h in (
                "frametime_ms",
                "frametime(ms)",
                "frametime",
                "frame_time",
                "frametime_msec",
                "ms",
                "msbetweenpresents",
                "msbetweenpresent",
                "msbetweendisplaychange",
            ):
                col_idx = idx
                mode = "ms"
                break
            if h in ("fps", "current_fps"):
                col_idx = idx
                mode = "fps"
                break

        if col_idx is None:
            # Maybe there are no headers and it's just raw numbers
            try:
                val = float(header_row[0].strip())
                col_idx = 0
                mode = "auto"
                if val > 100000.0:
                    frametimes.append(val / 1_000_000.0)
                else:
                    frametimes.append(val)
            except ValueError:
                # First row was a header we didn't recognize, find first numeric column in next row
                pass

        for row in reader:
            if not row:
                continue
            if col_idx is None:
                # Try to locate the numeric column
                for idx, cell in enumerate(row):
                    try:
                        v = float(cell.strip())
                        col_idx = idx
                        mode = "auto"
                        if v > 100000.0:
                            frametimes.append(v / 1_000_000.0)
                        else:
                            frametimes.append(v)
                        break
                    except ValueError:
                        continue
                continue

            if col_idx >= len(row):
                continue

            cell = row[col_idx].strip()
            if not cell:
                continue
            try:
                val = float(cell)
            except ValueError:
                continue

            if mode == "ns":
                frametimes.append(val / 1_000_000.0)
            elif mode == "fps":
                if val > 0:
                    frametimes.append(1000.0 / val)
            elif mode == "ms":
                frametimes.append(val)
            else:  # mode == "auto" (unlabeled input)
                if val > 100000.0:
                    frametimes.append(val / 1_000_000.0)
                else:
                    frametimes.append(val)

    return [ft for ft in frametimes if ft > 0 and math.isfinite(ft)]


def percentile(sorted_data: list[float], p: float) -> float:
    """Calculate the p-th percentile (0..100) using linear interpolation."""
    if not sorted_data:
        return 0.0
    if p <= 0:
        return sorted_data[0]
    if p >= 100:
        return sorted_data[-1]

    n = len(sorted_data)
    idx = (p / 100.0) * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    frac = idx - low

    if low == high:
        return sorted_data[low]
    return sorted_data[low] * (1.0 - frac) + sorted_data[high] * frac


def compute_metrics(
    frametimes: list[float],
    stutter_threshold_ms: float = 20.0,
    target_fps: float | None = None,
) -> dict:
    """Compute FPS and latency statistics from frame times in ms."""
    if not frametimes:
        res = {
            "count": 0,
            "duration_s": 0.0,
            "avg_fps": 0.0,
            "min_fps": 0.0,
            "max_fps": 0.0,
            "one_percent_low_fps": 0.0,
            "zero_point_one_percent_low_fps": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "p99_9_ms": 0.0,
            "max_ms": 0.0,
            "stutter_count": 0,
            "stutter_time_pct": 0.0,
        }
        if target_fps is not None:
            res["target_fps"] = target_fps
            res["target_hit_pct"] = 0.0
        return res

    count = len(frametimes)
    total_time_ms = sum(frametimes)
    duration_s = total_time_ms / 1000.0

    avg_fps = (count / duration_s) if duration_s > 0 else 0.0
    min_ms = min(frametimes)
    max_ms = max(frametimes)
    max_fps = (1000.0 / min_ms) if min_ms > 0 else 0.0
    min_fps = (1000.0 / max_ms) if max_ms > 0 else 0.0

    sorted_ms = sorted(frametimes)
    p50_ms = percentile(sorted_ms, 50.0)
    p95_ms = percentile(sorted_ms, 95.0)
    p99_ms = percentile(sorted_ms, 99.0)
    p99_9_ms = percentile(sorted_ms, 99.9)

    # 1% low is inverted 99th percentile frame time (or bottom 1% FPS)
    one_percent_low_fps = (1000.0 / p99_ms) if p99_ms > 0 else 0.0
    zero_point_one_percent_low_fps = (1000.0 / p99_9_ms) if p99_9_ms > 0 else 0.0

    # Stutters: frames taking longer than the specified threshold (default > 20ms or > 2x median)
    effective_thresh = max(stutter_threshold_ms, p50_ms * 1.5)
    stuttering_frames = [ft for ft in frametimes if ft >= effective_thresh]
    stutter_count = len(stuttering_frames)
    stutter_time_ms = sum(stuttering_frames)
    stutter_time_pct = (stutter_time_ms / total_time_ms * 100.0) if total_time_ms > 0 else 0.0

    metrics = {
        "count": count,
        "duration_s": round(duration_s, 2),
        "avg_fps": round(avg_fps, 2),
        "min_fps": round(min_fps, 2),
        "max_fps": round(max_fps, 2),
        "one_percent_low_fps": round(one_percent_low_fps, 2),
        "zero_point_one_percent_low_fps": round(zero_point_one_percent_low_fps, 2),
        "min_ms": round(min_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "p99_ms": round(p99_ms, 2),
        "p99_9_ms": round(p99_9_ms, 2),
        "max_ms": round(max_ms, 2),
        "stutter_count": stutter_count,
        "stutter_time_pct": round(stutter_time_pct, 2),
    }

    if target_fps is not None and target_fps > 0:
        target_ms = 1000.0 / target_fps
        on_time = [ft for ft in frametimes if ft <= target_ms * 1.05]
        metrics["target_fps"] = target_fps
        metrics["target_hit_pct"] = round(len(on_time) / count * 100.0, 2)

    return metrics


def format_report(name: str, metrics: dict) -> str:
    """Format single run metrics into human-readable text."""
    lines = [
        f"=== Frametime Analysis: {name} ===",
        f"Frames captured: {metrics['count']:,} ({metrics['duration_s']} s)",
        f"Average FPS:     {metrics['avg_fps']:>7.2f} FPS",
        f"1% Low FPS:      {metrics['one_percent_low_fps']:>7.2f} FPS",
        f"0.1% Low FPS:    {metrics['zero_point_one_percent_low_fps']:>7.2f} FPS",
        "--------------------------------------",
        f"Min Frametime:   {metrics['min_ms']:>7.2f} ms",
        f"Median (P50):    {metrics['p50_ms']:>7.2f} ms",
        f"P95 Frametime:   {metrics['p95_ms']:>7.2f} ms",
        f"P99 Frametime:   {metrics['p99_ms']:>7.2f} ms",
        f"P99.9 Frametime: {metrics['p99_9_ms']:>7.2f} ms",
        f"Max Frametime:   {metrics['max_ms']:>7.2f} ms",
        "--------------------------------------",
        f"Stutters:        {metrics['stutter_count']} frames ({metrics['stutter_time_pct']}% of runtime)",
    ]
    if metrics.get("target_fps") is not None:
        lines.append(f"Target FPS ({metrics['target_fps']:.0f}):  {metrics.get('target_hit_pct', 0.0)}% of frames on time")
    return "\n".join(lines)


def format_comparison(name1: str, m1: dict, name2: str, m2: dict) -> str:
    """Format a two-run comparative report with delta percentages."""
    def delta(v1: float, v2: float, higher_is_better: bool = True) -> str:
        if v1 == 0:
            return "N/A"
        diff = ((v2 - v1) / v1) * 100.0
        sign = "+" if diff > 0 else ""
        text = f"{sign}{diff:.1f}%"
        return f"{text:>8}"

    lines = [
        "=== Performance Comparison ===",
        f"Baseline [A]: {name1}",
        f"Tuned    [B]: {name2}",
        "",
        f"{'Metric':<24} | {'[A] Baseline':<12} | {'[B] Tuned':<12} | {'Change':<8}",
        "-" * 65,
        f"{'Average FPS':<24} | {m1['avg_fps']:>10.2f}   | {m2['avg_fps']:>10.2f}   | {delta(m1['avg_fps'], m2['avg_fps'], True)}",
        f"{'1% Low FPS':<24} | {m1['one_percent_low_fps']:>10.2f}   | {m2['one_percent_low_fps']:>10.2f}   | {delta(m1['one_percent_low_fps'], m2['one_percent_low_fps'], True)}",
        f"{'0.1% Low FPS':<24} | {m1['zero_point_one_percent_low_fps']:>10.2f}   | {m2['zero_point_one_percent_low_fps']:>10.2f}   | {delta(m1['zero_point_one_percent_low_fps'], m2['zero_point_one_percent_low_fps'], True)}",
        f"{'Median Frame Time (ms)':<24} | {m1['p50_ms']:>10.2f}   | {m2['p50_ms']:>10.2f}   | {delta(m1['p50_ms'], m2['p50_ms'], False)}",
        f"{'P95 Frame Time (ms)':<24} | {m1['p95_ms']:>10.2f}   | {m2['p95_ms']:>10.2f}   | {delta(m1['p95_ms'], m2['p95_ms'], False)}",
        f"{'P99 Frame Time (ms)':<24} | {m1['p99_ms']:>10.2f}   | {m2['p99_ms']:>10.2f}   | {delta(m1['p99_ms'], m2['p99_ms'], False)}",
        f"{'Stutter Frames':<24} | {m1['stutter_count']:>10}   | {m2['stutter_count']:>10}   | {delta(m1['stutter_count'], m2['stutter_count'], False)}",
        f"{'Stutter Time %':<24} | {m1['stutter_time_pct']:>9.1f}%  | {m2['stutter_time_pct']:>9.1f}%  | {delta(m1['stutter_time_pct'], m2['stutter_time_pct'], False)}",
    ]
    if m1.get("target_hit_pct") is not None and m2.get("target_hit_pct") is not None:
        lines.append(
            f"{'Target Hit %':<24} | {m1['target_hit_pct']:>9.1f}%  | {m2['target_hit_pct']:>9.1f}%  | {delta(m1['target_hit_pct'], m2['target_hit_pct'], True)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="CSV log file(s) to analyze")
    ap.add_argument("--compare", action="store_true", help="compare exactly two log files")
    ap.add_argument("--target-fps", type=float, default=None,
                    help="target frame rate in FPS to evaluate frame delivery (default: None)")
    ap.add_argument("--stutter-threshold", type=float, default=20.0,
                    help="frametime threshold in ms considered a stutter (default: 20.0)")
    ap.add_argument("--json", action="store_true", help="output result as JSON")
    args = ap.parse_args(argv)

    if args.compare:
        if len(args.logs) != 2:
            print("ERROR: --compare requires exactly two log files", file=sys.stderr)
            return 2
        p1, p2 = args.logs
        try:
            ft1 = parse_frametimes_ms(p1)
            ft2 = parse_frametimes_ms(p2)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        m1 = compute_metrics(ft1, args.stutter_threshold, args.target_fps)
        m2 = compute_metrics(ft2, args.stutter_threshold, args.target_fps)

        if args.json:
            print(json.dumps({
                "baseline": {"path": p1, **m1},
                "tuned": {"path": p2, **m2},
            }, indent=2))
        else:
            print(format_comparison(os.path.basename(p1), m1, os.path.basename(p2), m2))
        return 0

    results = []
    for log_path in args.logs:
        try:
            fts = parse_frametimes_ms(log_path)
            metrics = compute_metrics(fts, args.stutter_threshold, args.target_fps)
            results.append({"path": log_path, **metrics})
            if not args.json:
                print(format_report(os.path.basename(log_path), metrics))
                if len(args.logs) > 1:
                    print()
        except (OSError, ValueError) as exc:
            print(f"ERROR: {log_path}: {exc}", file=sys.stderr)
            return 1

    if args.json:
        out = results[0] if len(results) == 1 else results
        print(json.dumps(out, indent=2))

    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
