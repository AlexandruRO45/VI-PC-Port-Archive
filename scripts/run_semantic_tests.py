r"""Run independent test files and packages from the translator regression suite in parallel.

    python -I scripts/run_semantic_tests.py tests/semantic          # one job per core
    python -I scripts/run_semantic_tests.py tests/semantic -j 4     # four at a time
    python -I scripts/run_semantic_tests.py tests/semantic -v       # also print passing jobs' output

Each top-level `test*.py` file runs as `python -m unittest <module>` from the suite folder. Each top-level package
runs with `unittest discover` from the same folder so package tests, imports, and `load_tests` hooks are honored.
Tests can run side by side because each writes only into its own temporary folder. With -j 1, the runner uses
`unittest discover` for the whole suite. Standard library only.
"""
import argparse
import concurrent.futures
import fnmatch
import os
import re
import subprocess
import sys
import time


def test_modules(directory):
    """Names of the top-level test files and package directories to schedule."""
    for _, dirs, files in os.walk(directory):
        packages = [name for name in dirs if os.path.isfile(os.path.join(directory, name, "__init__.py"))]
        return sorted(packages + [name for name in files if fnmatch.fnmatchcase(name, "test*.py")])
    return []


def run_module(directory, module, verbose):
    """Return (exit code, tests run, output). A code of None marks empty discovery."""
    path = os.path.join(directory, module)
    selection = ["discover", "-s", path, "-t", directory] if os.path.isdir(path) else [module.removesuffix(".py")]
    cmd = [sys.executable, "-m", "unittest", *selection, *(["-v"] if verbose else [])]
    p = subprocess.run(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace")
    ran = re.search(r"^Ran (\d+) tests? in", p.stdout, re.M)
    code = p.returncode
    # Older unittest versions report empty discovery as OK. Explicit skips have a different summary.
    if code in (0, 5) and re.search(r"^Ran 0 tests in [^\n]*\n\n(?:NO TESTS RAN|OK)$", p.stdout, re.M):
        code = None
    return code, int(ran.group(1)) if ran else 0, p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="the suite's folder, tests/semantic")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="files at a time (default: cores)")
    ap.add_argument("-v", "--verbose", action="store_true", help="unittest -v, and print passing files' output")
    a = ap.parse_args(argv)
    directory = os.path.abspath(a.directory)
    if a.jobs <= 1:
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", directory,
                                *(["-v"] if a.verbose else [])])
    modules = test_modules(directory)
    if not modules:
        print(f"no test*.py files in {directory}")
        return 1
    # Largest files first: they tend to run longest, and one started last would set the wall time alone.
    # Start packages before standalone files because a package can contain several test files.
    modules.sort(key=lambda m: (m.endswith(".py"), -os.path.getsize(os.path.join(directory, m))))
    scheduled = set(modules)
    shadowed = sorted(name for name in modules if name.endswith(".py") and name.removesuffix(".py") in scheduled)
    if shadowed:
        print(f"test files shadowed by packages in {directory}: {', '.join(shadowed)}", file=sys.stderr)
        return 1
    start, total, failed = time.monotonic(), 0, []
    empty = 0
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        jobs = {pool.submit(run_module, directory, m, a.verbose): m.removesuffix(".py") for m in modules}
        for job in concurrent.futures.as_completed(jobs):
            code, ran, out = job.result()
            total += ran
            if code is None:
                empty += 1
            if code:
                failed.append(jobs[job])
                print(f"==== FAILED: {jobs[job]}\n{out}", flush=True)
            elif a.verbose:
                print(f"==== {jobs[job]}\n{out}", flush=True)
    took = time.monotonic() - start
    kind = "test targets" if any(not m.endswith(".py") for m in modules) else "files"
    summary = f"{len(modules)} {kind}, {total} tests, {len(failed)} {kind} failed in {took:.0f} s with {a.jobs} jobs"
    print(summary + (": " + ", ".join(sorted(failed)) if failed else ""))
    if empty == len(modules):
        print(f"no tests discovered in {directory}")
        return 1
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
