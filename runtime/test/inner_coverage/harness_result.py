"""Preserve unavailable-hardware status without hiding validation failures."""
import sys


def check_harness_result(result):
    if "PROBE_VALIDATION:" in result.stderr:
        raise RuntimeError(result.stderr)
    if result.returncode == 77:
        print("SKIP: InnerCoverage fixture prerequisites unavailable:\n" + result.stderr, file=sys.stderr)
        raise SystemExit(77)
    if result.returncode:
        raise RuntimeError(result.stderr)


def run_suite_command(command, env):
    import subprocess
    result = subprocess.run(command, env=env)
    if result.returncode == 77:
        print("SKIP: InnerCoverage matrix incomplete; a required GPU capability or validation layer is unavailable.", file=sys.stderr)
        raise SystemExit(77)
    result.check_returncode()
