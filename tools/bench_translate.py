"""Time the shader translator over the repository's own fixtures, so a change can be measured without game data.

    python -I tools/bench_translate.py [passes]

Translates every listing in tests/semantic/fixtures/disasm once per pass and prints the fastest and the median
pass. Those listings are the translator's regression suite: hand-written and synthetic RDNA2 programs with no game
content, covering pixel, vertex, compute and the other stages, so the run exercises the whole translator rather
than a sample of it. What it does not exercise is the size of a real corpus shader: the fixtures are a few hundred
to a few thousand instructions, and the per-shader fixed cost (the container and signature reads, the module
assembly) weighs more here than it does on a 26 503-shader run.

Passes are compared, not summed: the machine is shared, so the best pass is the number to quote and the median is
the one to expect. Standard library only, no game files, no GPU.

The profiler to reach for when a change needs explaining is a sampling one (pyinstrument) over one big fixture:
`python -m pyinstrument -r 4 tools/rdna2spirv.py ...` attributes the time to the passes rather than to call counts.
"""
import os
import statistics
import struct
import sys
import tempfile
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rdna2spirv  # noqa: E402  (the folder is off sys.path under -I)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "semantic", "fixtures", "disasm")
# translate() reads <dir>/<name>.dxbc next to the disassembly folder; this one declares no inputs.
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


USAGE = """usage: python -I tools/bench_translate.py [passes]

Times tools/rdna2spirv.py over the fixtures in tests/semantic/fixtures/disasm: every listing is translated once
per pass, and the fastest and the median pass are printed. `passes` defaults to 12. No game data, no GPU.
"""


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv[1:])
    if args and args[0] in ("-h", "--help"):
        print(USAGE.strip())
        return 0
    if len(args) > 1 or (args and not args[0].isdigit()):
        print(USAGE.strip(), file=sys.stderr)
        return 2
    passes = int(args[0]) if args else 12
    try:
        names = sorted(f[:-2] for f in os.listdir(FIXTURES) if f.endswith(".s"))
    except OSError as exc:
        print(f"ERROR: cannot list {FIXTURES}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    if not names:
        print(f"ERROR: no fixtures in {FIXTURES}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        paths = {}
        for name in names:
            with open(os.path.join(FIXTURES, name + ".s"), "rb") as src:
                data = src.read()
            path = os.path.join(tmp, "disasm", name + ".s")
            with open(path, "wb") as dst:
                dst.write(data)
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as dst:
                dst.write(STUB_DXBC)
            paths[name] = path

        times, translated, chars = [], 0, 0
        for _ in range(passes):
            translated = chars = 0
            start = time.perf_counter()
            for name in names:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        text = rdna2spirv.translate(paths[name])
                    except (rdna2spirv.Unsupported, OSError, struct.error, ValueError):
                        continue          # a fixture this run cannot translate (a missing FSSM, a refused shape)
                translated += 1
                chars += len(text)
            times.append(time.perf_counter() - start)

    best, median = min(times), statistics.median(times)
    print(f"{len(names)} fixtures, {passes} passes, {translated} translated, {chars:,} chars of SPIR-V")
    print(f"best   {best * 1000:8.1f} ms   {best / max(translated, 1) * 1000:6.3f} ms/shader")
    print(f"median {median * 1000:8.1f} ms   {median / max(translated, 1) * 1000:6.3f} ms/shader")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))
