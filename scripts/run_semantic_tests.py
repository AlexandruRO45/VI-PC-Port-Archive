r"""Run the translator regression suite (tests/semantic) one process per test file, several files at a time.

    python -I scripts/run_semantic_tests.py tests/semantic          # one file per core
    python -I scripts/run_semantic_tests.py tests/semantic -j 4     # four at a time
    python -I scripts/run_semantic_tests.py tests/semantic -v       # also print the output of the files that pass

It runs the files `python -m unittest discover -s tests/semantic` finds (`test*.py`, and packages below it), each as
`python -m unittest <module>` from that folder, so a file sees the same import path as under discover. The files can
run side by side because each one writes only into its own temporary folder; a new test must do the same. With -j 1,
or on a single core, it runs that discover command itself. Prints the output of every failing file and one summary
line, and exits 1 if a file fails or none is found. Standard library only.

A file passes only when unittest itself finished: each run goes through a small bootstrap that prints a random
token, made fresh for every run, with the outer runner's test count after `unittest.main` returns. A worker that
exits 0 before that (os._exit, an interpreter crash reported as 0, or a nested runner that printed its own passing
summary and then exited) has no token and fails.
"""
import argparse
import concurrent.futures
import fnmatch
import os
import re
import secrets
import subprocess
import sys
import time


def test_modules(directory):
    """Dotted names of the test modules unittest discover finds under `directory` (it enters packages only)."""
    names = []
    for base, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if os.path.isfile(os.path.join(base, d, "__init__.py")))
        prefix = os.path.relpath(base, directory).replace(os.sep, ".")
        for name in sorted(files):
            if fnmatch.fnmatchcase(name, "test*.py"):
                names.append(name[:-3] if prefix == "." else f"{prefix}.{name[:-3]}")
    return names


# Runs unittest with the given arguments and reports completion itself, after unittest.main returns: a line
# "<token> <tests run> <1 if successful else 0>". The token is argv[1], so a test cannot know it in advance by name.
BOOTSTRAP = r"""import sys, unittest
token = sys.argv[1]
result = unittest.main(module=None, argv=["python -m unittest", *sys.argv[2:]], exit=False).result
sys.stderr.flush()
print("\n" + f"{token} {result.testsRun} {int(result.wasSuccessful())}", flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
"""


def unittest_command(args):
    """(token, command) running `python -m unittest <args>` through BOOTSTRAP."""
    token = "leonida-semantic-done-" + secrets.token_hex(16)
    return token, [sys.executable, "-c", BOOTSTRAP, token, *args]


def completion(token, line):
    """(tests run, successful) if `line` is the bootstrap's completion line for `token`, else None."""
    m = re.fullmatch(re.escape(token) + r" (\d+) ([01])", line.strip())
    return (int(m.group(1)), m.group(2) == "1") if m else None


def run_module(directory, module, verbose):
    """(return code, tests run, output) of `python -m unittest <module>` run from `directory`. The code is 1 when
    unittest did not report completing, whatever the process's exit status."""
    token, cmd = unittest_command([*(["-v"] if verbose else []), module])
    p = subprocess.run(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       errors="replace")
    done, kept = None, []
    for line in p.stdout.splitlines(keepends=True):
        seen = completion(token, line)
        if seen is None:
            kept.append(line)
        else:
            done = seen
    out = "".join(kept)
    if done is None:
        out += f"\n(the test process exited with {p.returncode} before unittest reported completing)\n"
        return p.returncode or 1, 0, out
    ran, ok = done
    return (p.returncode or (0 if ok else 1)), ran, out


def run_discover(directory, verbose):
    """Exit code of `python -m unittest discover -s <directory>`, output streamed; 1 when unittest did not report
    completing."""
    token, cmd = unittest_command(["discover", "-s", directory, *(["-v"] if verbose else [])])
    done = None
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, errors="replace") as p:
        for line in p.stdout:
            seen = completion(token, line)
            if seen is None:
                sys.stdout.write(line)
                sys.stdout.flush()
            else:
                done = seen
    if done is None:
        print(f"the test process exited with {p.returncode} before unittest reported completing")
        return p.returncode or 1
    return p.returncode or (0 if done[1] else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directory", help="the suite's folder, tests/semantic")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1, help="files at a time (default: cores)")
    ap.add_argument("-v", "--verbose", action="store_true", help="unittest -v, and print passing files' output")
    a = ap.parse_args(argv)
    directory = os.path.abspath(a.directory)
    if a.jobs <= 1:
        return run_discover(directory, a.verbose)
    modules = test_modules(directory)
    if not modules:
        print(f"no test*.py files in {directory}")
        return 1
    # Largest files first: they tend to run longest, and one started last would set the wall time alone.
    modules.sort(key=lambda m: -os.path.getsize(os.path.join(directory, *m.split(".")) + ".py"))
    start, total, failed = time.monotonic(), 0, []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
        jobs = {pool.submit(run_module, directory, m, a.verbose): m for m in modules}
        for job in concurrent.futures.as_completed(jobs):
            code, ran, out = job.result()
            total += ran
            if code:
                failed.append(jobs[job])
                print(f"==== FAILED: {jobs[job]}\n{out}", flush=True)
            elif a.verbose:
                print(f"==== {jobs[job]}\n{out}", flush=True)
    took = time.monotonic() - start
    summary = f"{len(modules)} files, {total} tests, {len(failed)} files failed in {took:.0f} s with {a.jobs} jobs"
    print(summary + (": " + ", ".join(sorted(failed)) if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
