"""Check the terminal startup trace for a data-only DLL through each loader mode."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main(loader, image):
    with tempfile.TemporaryDirectory() as directory:
        for mode in ("--dry-run", "--init", "--run"):
            trace = Path(directory) / (mode + ".jsonl")
            result = subprocess.run([loader, mode, image, "--overlay", directory, "--trace", trace],
                                    capture_output=True, timeout=30)
            events = [json.loads(line) for line in trace.read_bytes().splitlines()]
            terminal = events[-1] if events else {}
            if mode == "--run":
                message = f"{Path(image).name} has no entry point"
                if (result.returncode != 1 or message.encode() not in result.stderr
                        or terminal.get("kind") != "error" or terminal.get("what") != message):
                    raise SystemExit(f"{mode}: rc={result.returncode}, stderr={result.stderr!r}, last={terminal}")
            elif result.returncode != 0 or terminal.get("kind") != "exit" or terminal.get("code") != 0:
                raise SystemExit(f"{mode}: rc={result.returncode}, stderr={result.stderr!r}, last={terminal}")


if __name__ == "__main__":
    main(*sys.argv[1:])
