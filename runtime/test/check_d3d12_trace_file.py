"""Exercise the mapped D3D12 trace and the gta6rt CLI with synthetic programs on Windows."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from boot_trace import parse_d3d12


def run(*args, code=0):
    result = subprocess.run([str(arg) for arg in args], capture_output=True, timeout=12)
    if result.returncode != code:
        raise AssertionError(f"{args}: exit {result.returncode}, expected {code}\n"
                             f"{result.stderr.decode(errors='replace')}\n{result.stdout.decode(errors='replace')}")
    return result


def records(data):
    return data.split(b"\0", 1)[0].decode("ascii").splitlines()


def main():
    fixture, loader, guest = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="leonida_d3d12_trace_") as folder:
        root = Path(folder)
        serial = root / "serial.log"
        legacy = records(run(fixture, "stderr").stderr)
        assert len(legacy) == 3, legacy
        assert run(fixture, "serial", serial).stderr == b""
        assert records(serial.read_bytes()) == legacy
        assert serial.read_bytes().endswith(b"\n") and b"\0" not in serial.read_bytes()

        saved = serial.read_bytes()
        stderr = run(fixture, "serial", serial, code=3).stderr
        assert b"open failed: 80" in stderr or b"open failed: 183" in stderr, stderr
        assert serial.read_bytes() == saved
        run(fixture, "serial", root / "missing" / "trace.log", code=3)

        concurrent = root / "concurrent.log"
        run(fixture, "concurrent", concurrent)
        output = records(concurrent.read_bytes())
        assert output[:3] == legacy and len(output) == 3 + 4 * 128
        next_call = [0] * 4
        for seq, row in enumerate(output[3:], start=4):
            match = re.fullmatch(r"\[d3d12-trace\] #(\d+) Fixture_(\d+)_(\d+) called", row)
            assert match, row
            actual, thread, call = map(int, match.groups())
            assert actual == seq and call == next_call[thread], row
            next_call[thread] += 1
        assert next_call == [128] * 4

        full = root / "full.log"
        run(fixture, "small", full)
        truncated = parse_d3d12(str(full))
        assert 3 < truncated["calls"] < 43 and truncated["limit_reached"], truncated
        assert truncated["last_seq"] == truncated["calls"] and full.stat().st_size < 512

        crash = root / "crash.log"
        with subprocess.Popen([str(fixture), "crash", str(crash)], stdout=subprocess.PIPE) as child:
            ready = threading.Event()
            line = []

            def wait_for_publication():
                line.append(child.stdout.readline().strip())
                ready.set()

            reader = threading.Thread(target=wait_for_publication)
            reader.start()
            try:
                assert ready.wait(10) and line == [b"published"] and child.poll() is None, line
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
                reader.join(timeout=10)
        data = crash.read_bytes()
        assert len(data) == 128 * 1024 and b"\0" in data, len(data)
        assert records(data) == legacy
        assert parse_d3d12(str(crash))["calls"] == 3

        cli = root / "cli.log"
        argv = (loader, "--run", guest, "--game-dir", root, "--overlay", root / "overlay")
        run(*argv, "--trace-d3d12-file", cli, "--", "hello", "42")
        assert cli.read_bytes() == b""  # no graphics calls; the explicit ExitProcess path trims the mapping
        run(*argv, "--trace-d3d12-file", cli, code=2)
        assert cli.read_bytes() == b""
        run(*argv, "--trace-d3d12-file", code=2)
        mixed = root / "mixed.log"
        run(*argv, "--trace-d3d12", "--trace-d3d12-file", mixed, code=2)
        assert not mixed.exists()

    print("Mapped D3D12 trace: legacy parity, concurrent order, file limits, child termination and CLI passed")


if __name__ == "__main__":
    main()
