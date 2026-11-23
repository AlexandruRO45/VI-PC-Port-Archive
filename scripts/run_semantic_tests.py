"""Run the translator's semantic tests (tests/semantic) one test file per process, several files at a time.

Usage:
    python -I scripts/run_semantic_tests.py [--start DIR] [--pattern GLOB] [--jobs N] [--timeout SECONDS] [--verbose]

Each test file runs as `python -m unittest discover -s DIR -p FILE`, the same discovery and imports as
`python -m unittest discover -s tests/semantic`, so a file that passes one way passes the other. The serial run takes
about 7 minutes on Linux (Python 3.13); with one file per process on 4 workers, about 1.5 minutes.

A failing file's whole output is printed. The exit code is 1 when a file fails, errors, runs no test, or runs past
--timeout (it is stopped and reported), and when --start holds no test file; else 0. --jobs defaults to the CPU count;
--timeout (default 300 seconds per file) keeps one stuck file from holding the job until the CI limit.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)


def test_files(start, pattern):
    """The file names unittest discovery would load at the top of `start`, sorted."""
    return sorted(p.name for p in Path(start).glob(pattern) if p.is_file() and p.suffix == ".py")


def run_file(start, name, timeout):
    """(name, verdict, tests run, seconds, output) for one test file; verdict is "ok", "failed" or "timed out"."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(start), "-p", name]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.monotonic()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                           timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired as e:
        out = "".join(x if isinstance(x, str) else (x or b"").decode("utf-8", "replace") for x in (e.stdout, e.stderr))
        return name, "timed out", 0, time.monotonic() - t0, out
    out = r.stdout + r.stderr
    m = RAN.search(out)
    ran = int(m.group(1)) if m else 0
    verdict = "ok" if r.returncode == 0 and ran > 0 else "failed"
    if ran == 0:   # Python 3.12+ exits 5 here, older versions 0: either way the file checked nothing
        out += "\n(no test ran in this file)\n"
    return name, verdict, ran, time.monotonic() - t0, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default=str(ROOT / "tests" / "semantic"), help="directory to discover (tests/semantic)")
    ap.add_argument("--pattern", default="test*.py", help="test file pattern (unittest's default)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="files run at once (CPU count)")
    ap.add_argument("--timeout", type=float, default=300.0, help="seconds per file before it is stopped (0: none)")
    ap.add_argument("--verbose", action="store_true", help="print every file's output, not only the failing ones")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    names = test_files(args.start, args.pattern)
    if not names:
        print(f"run_semantic_tests: no {args.pattern} file in {args.start}", file=sys.stderr)
        return 1
    # Largest files first: they tend to run longest, so the pool does not end on one long file.
    order = sorted(names, key=lambda n: -os.path.getsize(os.path.join(args.start, n)))
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.jobs) as pool:
        results = sorted(pool.map(lambda n: run_file(args.start, n, args.timeout), order))
    bad = [r for r in results if r[1] != "ok"]
    for name, verdict, ran, secs, out in results:
        if verdict != "ok" or args.verbose:
            print(f"===== {name}: {verdict} ({ran} tests, {secs:.1f} s)")
            print(out.rstrip())
    for name, verdict, _, secs, _ in bad:
        print(f"{verdict.upper()}: {name} ({secs:.1f} s)")
    total = sum(r[2] for r in results)
    print(f"semantic tests: {len(results)} files, {total} tests, {len(bad)} files not ok, "
          f"{time.monotonic() - t0:.1f} s on {args.jobs} jobs")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
