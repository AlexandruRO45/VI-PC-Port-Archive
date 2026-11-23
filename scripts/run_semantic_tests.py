"""Run the translator's semantic tests (tests/semantic) one test file per process, several files at a time.

Usage:
    python -I scripts/run_semantic_tests.py [--start DIR] [--pattern GLOB] [--jobs N] [--timeout SECONDS] [--verbose]

Each test file (at the top of DIR and in its sub-packages) runs as `python -m unittest discover -s DIR -p FILE`, the same discovery and imports as
`python -m unittest discover -s tests/semantic`, so a file that passes one way passes the other. The serial run takes
about 7 minutes on Linux (Python 3.13); with one file per process on 4 workers, about 1.5 minutes.

A failing file's whole output is printed. The exit code is 1 when a file fails, errors, runs no test, or runs past
--timeout (it is stopped and reported), and when --start holds no test file; else 0. --jobs defaults to the CPU count;
--timeout (default 300 seconds per file) keeps one stuck file from holding the job until the CI limit.
"""
import argparse
import fnmatch
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
KILL_WAIT = 10.0   # seconds to collect a killed file's output before giving up on it


def test_files(start, pattern):
    """The test files unittest discovery would load under `start`, as sorted "/"-separated paths relative to it.

    Like unittest, it looks at the top of `start` and inside sub-folders that are packages (hold an __init__.py).
    """
    found = []
    for folder, subs, files in os.walk(start):
        rel = Path(folder).relative_to(start).as_posix()
        subs[:] = [d for d in subs if os.path.isfile(os.path.join(folder, d, "__init__.py"))]
        found += [n if rel == "." else f"{rel}/{n}" for n in files
                  if n.endswith(".py") and fnmatch.fnmatch(n, pattern)]
    return sorted(found)


def kill_tree(proc):
    """Stop the child and everything it started (they share its process group, or on Windows its tree)."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=KILL_WAIT)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        proc.kill()


def run_file(start, name, timeout):
    """(name, verdict, tests run, seconds, output) for one test file; verdict is "ok", "failed" or "timed out".

    `name` is the file's path relative to `start`; a file in a sub-package is discovered from its own folder
    with `start` as the top-level directory, so its imports resolve as in a full discovery.
    """
    folder, _, base = name.rpartition("/")
    sub = os.path.join(start, folder) if folder else str(start)
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", sub, "-t", str(start), "-p", base]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    group = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
             else {"start_new_session": True})
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", env=env, **group)
    try:
        stdout, stderr = proc.communicate(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        try:   # a descendant that escaped the kill may still hold the pipes: do not wait for it forever
            stdout, stderr = proc.communicate(timeout=KILL_WAIT)
        except subprocess.TimeoutExpired as e:
            stdout, stderr = e.stdout, e.stderr
        out = "".join(x if isinstance(x, str) else (x or b"").decode("utf-8", "replace") for x in (stdout, stderr))
        return name, "timed out", 0, time.monotonic() - t0, out
    out = stdout + stderr
    m = RAN.search(out)
    ran = int(m.group(1)) if m else 0
    verdict = "ok" if proc.returncode == 0 and ran > 0 else "failed"
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
    for stream in (sys.stdout, sys.stderr):   # a test's output may not fit a redirected stdout's code page (cp1251)
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    names = test_files(args.start, args.pattern)
    if not names:
        print(f"run_semantic_tests: no {args.pattern} file in {args.start}", file=sys.stderr)
        return 1
    # Largest files first: they tend to run longest, so the pool does not end on one long file.
    order = sorted(names, key=lambda n: -os.path.getsize(os.path.join(args.start, *n.split("/"))))
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
    note = " (timed out, tests not counted)" if any(r[1] == "timed out" for r in results) else ""
    print(f"semantic tests: {len(results)} files, {total} tests{note}, {len(bad)} files not ok, "
          f"{time.monotonic() - t0:.1f} s on {args.jobs} jobs")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
