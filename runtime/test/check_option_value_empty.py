"""gta6rt refuses an empty value for --trace, --game-dir, --overlay and --mods.

A command line built from variables (`--trace "%TRACE%"`) passes an empty value when the variable is unset. That was
read as the option not given, so the startup trace the first-boot runbook asks for was never created and the run went
on to stop at the image: a mistyped command line read as a clean boot attempt with no trace to read. An empty value is
not a path, so it is a missing value: the usage and exit 2, before the image is opened (a text file stands in for it).

A real value is still taken, and an empty *game* argument after `--` is still the game's own, not the loader's.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(loader, args, cwd):
    result = subprocess.run([loader] + args, cwd=str(cwd), capture_output=True, timeout=60)
    return (result.returncode,
            result.stdout.decode("utf-8", "replace"),
            result.stderr.decode("utf-8", "replace"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loader", required=True)
    parser.add_argument("--work-dir", required=True)
    a = parser.parse_args()
    root = Path(a.work_dir) / "option_value_empty"
    root.mkdir(parents=True, exist_ok=True)
    image = root / "image.exe"
    image.write_text("not a PE image", encoding="ascii")
    trace = root / "trace.jsonl"
    if trace.exists():
        trace.unlink()
    failures = []

    # The error comes before anything is opened, so the image may be a text file; a run that reached it would print
    # its loader error and exit 1 instead of the usage and exit 2.
    for option in ("--trace", "--game-dir", "--overlay", "--mods"):
        code, out, err = run(a.loader, ["--dry-run", str(image), option, ""], root)
        if code != 2 or "usage: gta6rt" not in err or "module(s) mapped" in out or "running " in out:
            failures.append(f'{option} "": exit {code}, stdout {out!r}, stderr {err!r}')
    if trace.exists():
        failures.append(f'the empty --trace value created {trace.name}')

    # A real value is still taken: --trace FILE is created (opened before the image is read) and the run stops at the
    # image with exit 1, not with the usage.
    code, out, err = run(a.loader, ["--dry-run", str(image), "--trace", str(trace)], root)
    if code != 1 or "usage: gta6rt" in err or not trace.exists():
        failures.append(f"--trace FILE was not taken as the trace file: exit {code}, stderr {err!r}")

    # `--` still takes whatever follows, including an empty argument, as the game's own.
    code, out, err = run(a.loader, ["--dry-run", str(image), "--", ""], root)
    if code != 1 or "usage: gta6rt" in err:
        failures.append(f"-- with an empty game argument was refused: exit {code}, stderr {err!r}")

    if failures:
        print("gta6rt did not treat an empty option value as a missing value", file=sys.stderr)
        for failure in failures:
            print(f"--- {failure}", file=sys.stderr)
        return 1
    print("gta6rt refuses an empty option value (exit 2) and still takes a real one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
