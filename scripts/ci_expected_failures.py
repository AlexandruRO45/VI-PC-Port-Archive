"""Judge a CI test run that has known failures, each fixed by an open pull request.

    python -I scripts/ci_expected_failures.py FAILED_LIST EXIT_STATUS [EXPECTED ...]

FAILED_LIST is a file with one failed test name per line, EXIT_STATUS the test runner's exit status, and each
EXPECTED argument one or more whitespace-separated test names (so a workflow can pass one quoted list). Exits 1 when a
test outside EXPECTED failed, or when the runner failed without naming a test (a crash, a build error). An expected
failure that passed gets a GitHub Actions notice, so the list in the workflow can shrink.
"""

import sys


def judge(failed: list[str], status: int, expected: list[str]) -> tuple[int, list[str]]:
    """(exit status, GitHub Actions workflow commands to print)."""
    unexpected = [name for name in failed if name not in expected]
    lines = [f"::error::{name} failed" for name in unexpected]
    if status != 0 and not failed:
        lines.append(f"::error::the test runner exited with {status} without naming a failed test")
    for name in expected:
        if name in failed:
            lines.append(f"::warning::{name} failed, as expected until its fix is merged")
        else:
            lines.append(f"::notice::{name} passed: drop it from the expected failures")
    return (1 if unexpected or (status != 0 and not failed) else 0), lines


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    if len(argv) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        status = int(argv[1])
    except ValueError:
        print(f"EXIT_STATUS must be an integer, not {argv[1]!r}", file=sys.stderr)
        return 2
    try:
        with open(argv[0], encoding="utf-8-sig") as f:
            failed = [line.strip() for line in f if line.strip()]
    except OSError as e:
        print(f"cannot read {argv[0]}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
        return 2
    expected = [name for arg in argv[2:] for name in arg.split()]
    code, lines = judge(failed, status, expected)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv[1:]))
