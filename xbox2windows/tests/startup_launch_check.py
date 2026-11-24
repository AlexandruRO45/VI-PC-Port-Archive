"""Check tools/startup/launch.py's containment on a source-built probe: exit code, captured
stdout/stderr and trap log, the timeout kill, the commit cap and the missing-EXE path.

launch.py runs one executable in a kill-on-close job with a timeout, a commit cap and captured
stdout/stderr. tests/startup_launch_probe.cpp is the only input the check needs: it writes known
bytes, or holds memory, spawns a child or stays alive so a job limit has to stop it. No GPU and no
game files are involved.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
LAUNCH = REPO / "tools/startup/launch.py"


def launch(probe, work, name, options, arguments):
    logdir = work / name
    logdir.mkdir()
    command = [sys.executable, "-B", str(LAUNCH), str(probe), str(work), str(logdir), *options, "--", *arguments]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    return result, logdir, time.monotonic() - started


def captured(logdir, name):
    path = logdir / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def main(probe):
    probe = Path(probe).resolve()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)

        # A finished run: the launcher reports the child's exit code and saves both streams and the trap log.
        result, logdir, _ = launch(probe, work, "report", ["--timeout", "10"], ["report", "7"])
        assert result.returncode == 0, result.stderr
        assert "exited" in result.stdout and "exit code 0x00000007" in result.stdout, result.stdout
        assert "peak job memory" in result.stdout, result.stdout
        assert captured(logdir, "stdout.txt") == "launch-probe stdout\n", captured(logdir, "stdout.txt")
        assert captured(logdir, "stderr.txt") == "launch-probe stderr\n", captured(logdir, "stderr.txt")
        assert captured(logdir, "trap.log") == "x2w-trap launch-probe\n", captured(logdir, "trap.log")

        # A child that overstays the timeout is terminated, not left running.
        result, logdir, elapsed = launch(probe, work, "timeout", ["--timeout", "1"], ["sleep", "30000"])
        assert result.returncode == 0, result.stderr
        assert "killed after 1 s timeout" in result.stdout, result.stdout
        assert "exit code 0xdead0001" in result.stdout, result.stdout
        assert elapsed < 20, f"the launcher took {elapsed:.1f} s to enforce a 1 s timeout"
        assert captured(logdir, "stdout.txt") == "launch-probe sleeping\n", captured(logdir, "stdout.txt")

        # The commit cap is a ceiling on the child; the same request fits under a larger cap.
        result, logdir, _ = launch(probe, work, "cap", ["--timeout", "10", "--memory-gib", "0.5"], ["allocate", "1024"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt").startswith("allocate failed error="), captured(logdir, "stdout.txt")
        result, logdir, _ = launch(probe, work, "cap-control", ["--timeout", "10", "--memory-gib", "4"], ["allocate", "1024"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt") == "allocate ok 1024 MiB\n", captured(logdir, "stdout.txt")

        # The process limit contains helpers the child starts; a larger limit lets the same helper run.
        result, logdir, _ = launch(probe, work, "one-process", ["--timeout", "10", "--max-processes", "1"], ["spawn"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt").startswith("spawn failed error="), captured(logdir, "stdout.txt")
        result, logdir, _ = launch(probe, work, "two-processes", ["--timeout", "10", "--max-processes", "2"], ["spawn"])
        assert result.returncode == 0, result.stderr
        assert captured(logdir, "stdout.txt") == "spawn ok\n", captured(logdir, "stdout.txt")

        # A missing executable is reported instead of looking like a finished run.
        logdir = work / "missing"
        logdir.mkdir()
        result = subprocess.run([sys.executable, "-B", str(LAUNCH), str(work / "missing.exe"), str(work), str(logdir),
                                 "--timeout", "10"], capture_output=True, text=True, timeout=60)
        assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
        assert "CreateProcess failed: error" in result.stdout, result.stdout

        print("launch: exit code, stdout/stderr/trap capture, 1 s timeout kill, 0.5 GiB commit cap "
              "(and its 4 GiB control), 1-process limit (and its 2-process control) and the missing-EXE path passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: startup_launch_check.py <startup_launch_probe.exe>")
    sys.exit(main(sys.argv[1]))