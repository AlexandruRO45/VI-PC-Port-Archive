"""Exercise --log on native Windows without supplying a console or output pipes."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main(loader, guest):
    with tempfile.TemporaryDirectory() as directory:
        log = Path(directory, "loader.log")
        log.write_bytes(b"previous run\n")
        result = subprocess.run(
            [loader, "--run", guest, "--log", log, "--overlay", Path(directory, "overlay"),
             "--", "hello", "42", "exit-code"],
            cwd=directory, creationflags=subprocess.DETACHED_PROCESS, timeout=30,
        )
        output = log.read_bytes()
        if result.returncode != 42 or not output.startswith(b"previous run\n") or b"tiny_game: OK" not in output:
            raise SystemExit(f"Detached run exited {result.returncode}: {output!r}")

        result = subprocess.run(
            [loader, "--dry-run", Path(directory, "missing.exe"), "--log", log],
            cwd=directory, creationflags=subprocess.DETACHED_PROCESS, timeout=30,
        )
        latest = log.read_bytes()
        if result.returncode != 1 or not latest.startswith(output) or b"error:" not in latest[len(output):]:
            raise SystemExit(f"Detached error reporting exited {result.returncode}: {latest!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
