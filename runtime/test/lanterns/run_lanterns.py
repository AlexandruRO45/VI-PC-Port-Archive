"""Run Lanterns under gta6rt --run, check what it printed, and turn its frame dumps into PNGs.

    python -I runtime/test/lanterns/run_lanterns.py --gta6rt <gta6rt.exe> --guest <lanterns.exe>
        --shaders <folder fssm_pack.py wrote> --out <folder> [--stage 1] [--wine] [--timeout 240] [--png FILE]

It sets LEONIDA_D3D12=vulkan and LEONIDA_SHADERS=<shaders>/spirv and runs
"gta6rt --run lanterns.exe -- --stage N --shaders <shaders> --out <out>" (under Wine with --wine, paths turned into
Z:\ paths). The run passes only if all of these hold, which a loader that exits 0 early, a guest that prints and then
fails, or the host's PC device instead of the shim's Vulkan device would each break:
  - the exit code is 0;
  - stdout has the line "Lanterns stage N: PASS, ..." exactly once (the guest prints it after every check);
  - the output has the shim's complete factory line "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x... (Vulkan
    device 0x...)" (the guard PR #3913's run_guest_draw.cmake uses).
The timeout is a promise: when it expires the whole process group (Windows: the process tree) is killed and the run
fails with exit code 3, whatever the guest, Wine or the GPU are doing. tests/test_run_lanterns.py checks it with a
launcher that never ends.

PNGs (sRGB-encoded from the linear float target), <out>/lanterns-stage<N>.png and lanterns-stage<N>-sheet.png:
  stage 1: the GPU frame with the depth test; sheet: GPU | CPU reference / GPU without depth (the negative control) |
           mask of the depth frame: certain and exact (grey), certain and wrong (red), uncertain (yellow)
"""
import argparse
import array
import os
import re
import signal
import struct
import subprocess
import sys
import time
import zlib

WIDTH, HEIGHT = 384, 256
TIMEOUT_EXIT = 3
FACTORY = re.compile(r"^\[rt\] d3d12_xs: D3D12XboxCreateDevice -> proxy (0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]* "
                     r"\(Vulkan device (0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]*\)$", re.M)


def marker(stage):
    return re.compile(rf"^Lanterns stage {stage}: PASS, .*$", re.M)


def check_transcript(stage, exit_code, stdout, stderr):
    """The reasons a finished run does not count as a pass (empty: it passes)."""
    stdout, stderr = stdout.replace("\r\n", "\n"), stderr.replace("\r\n", "\n")
    failures = []
    if exit_code != 0:
        failures.append(f"exit code {exit_code}, not 0")
    found = marker(stage).findall(stdout)
    if len(found) != 1:
        failures.append(f"stdout has the 'Lanterns stage {stage}: PASS' line {len(found)} times, not once")
    if not FACTORY.search(stderr) and not FACTORY.search(stdout):
        failures.append("no line shows D3D12XboxCreateDevice returning the shim's Vulkan device")
    return failures


def kill_tree(proc):
    """Kill the launcher and everything it started, then reap it; never waits unboundedly."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run(cmd, env, timeout, log_base):
    """(exit code, stdout, stderr, timed out). Output goes to <log_base>.stdout/.stderr files, not pipes: a process
    the launcher leaves behind (wineserver) may hold a pipe open long after gta6rt exits, so the wait is on the
    process. The child gets its own process group, so a timeout kills all of it."""
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with open(log_base + ".stdout", "wb") as out_file, open(log_base + ".stderr", "wb") as err_file:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out_file, stderr=err_file, **kwargs)
        try:
            code, timed_out = proc.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            code, timed_out = None, True
    texts = []
    for suffix in (".stdout", ".stderr"):
        with open(log_base + suffix, "rb") as f:
            texts.append(f.read().decode("utf-8", "replace"))
    return code, texts[0], texts[1], timed_out


def wine_path(path):
    return "Z:" + os.path.abspath(path).replace("/", "\\")


# ---------------------------------------------------------------------------------------------------------- PNG

def png(path, width, height, rows):
    """rows: one bytes object of width * 3 RGB bytes per row."""
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def srgb8(v):
    v = 0.0 if v != v or v < 0 else 1.0 if v > 1 else v
    v = 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
    return int(v * 255 + 0.5)


def read_floats(path):
    data = array.array("f")
    with open(path, "rb") as f:
        data.frombytes(f.read())
    if sys.byteorder != "little":
        data.byteswap()
    if len(data) != WIDTH * HEIGHT * 4:
        raise ValueError(f"{path}: {len(data)} floats, not {WIDTH * HEIGHT * 4}")
    return data


def rgb_rows(floats):
    return [bytes(srgb8(floats[(y * WIDTH + x) * 4 + c]) for x in range(WIDTH) for c in range(3)) for y in range(HEIGHT)]


def mask_rows(gpu, ref, uncertain):
    rows = []
    for y in range(HEIGHT):
        row = bytearray()
        for x in range(WIDTH):
            i = y * WIDTH + x
            if uncertain[i]:
                row += b"\xe0\xc0\x20"
            elif gpu[i * 4:i * 4 + 4] == ref[i * 4:i * 4 + 4]:
                row += b"\x50\x50\x50"
            else:
                row += b"\xff\x00\x00"
        rows.append(bytes(row))
    return rows


# The panels of each stage's PNGs: "<dump>.gpu" or "<dump>.ref" is a frame, "<dump>:mask" the uncertainty map of a dump.
PANELS = {
    1: {"main": [["stage1-depth.gpu"]],
        "sheet": [["stage1-depth.gpu", "stage1-depth.ref"], ["stage1-nodepth.gpu", "stage1-depth:mask"]]},
}


def panel_rows(out, name, cache):
    if name not in cache:
        if name.endswith(":mask"):
            dump = os.path.join(out, name[:-5])
            with open(dump + ".uncertain.u8", "rb") as f:
                uncertain = f.read()
            cache[name] = mask_rows(read_floats(dump + ".gpu.rgba32f"), read_floats(dump + ".ref.rgba32f"), uncertain)
        else:
            cache[name] = rgb_rows(read_floats(os.path.join(out, name + ".rgba32f")))
    return cache[name]


def grid(out, layout, cache):
    rows = []
    for panel_row in layout:
        panels = [panel_rows(out, name, cache) for name in panel_row]
        rows += [b"".join(parts) for parts in zip(*panels)]
    return WIDTH * len(layout[0]), HEIGHT * len(layout), rows


def write_pngs(out, stage, main_png):
    cache = {}
    png(main_png, *grid(out, PANELS[stage]["main"], cache))
    sheet = os.path.join(out, f"lanterns-stage{stage}-sheet.png")
    png(sheet, *grid(out, PANELS[stage]["sheet"], cache))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Lanterns under gta6rt --run and check it.")
    ap.add_argument("--gta6rt", required=True)
    ap.add_argument("--guest", required=True)
    ap.add_argument("--shaders", required=True, help="the folder fssm_pack.py wrote (with spirv/index.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", type=int, default=1, choices=sorted(PANELS))
    ap.add_argument("--wine", action="store_true", help="run gta6rt.exe under wine (paths become Z:\\...)")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds before the run is killed and fails")
    ap.add_argument("--png", help="where to write the main PNG (default <out>/lanterns-stage<N>.png)")
    a = ap.parse_args(argv)
    for path in (a.gta6rt, a.guest, os.path.join(a.shaders, "spirv", "index.json")):
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            return 2
    os.makedirs(a.out, exist_ok=True)
    conv = wine_path if a.wine else os.path.abspath
    env = dict(os.environ, LEONIDA_D3D12="vulkan", LEONIDA_SHADERS=conv(os.path.join(a.shaders, "spirv")))
    cmd = (["wine"] if a.wine else []) + [a.gta6rt, "--run", conv(a.guest), "--", "--stage", str(a.stage),
                                          "--shaders", conv(a.shaders), "--out", conv(a.out)]
    started = time.monotonic()
    code, out, err, timed_out = run(cmd, env, a.timeout, os.path.join(a.out, f"gta6rt-stage{a.stage}"))
    took = time.monotonic() - started
    print(f"--- gta6rt stdout ---\n{out}\n--- gta6rt stderr ---\n{err}\n--- exit: {code} ({took:.1f} s)")
    if timed_out:
        print(f"lanterns: FAILED: no exit after {a.timeout:g} s; the process group was killed", file=sys.stderr)
        return TIMEOUT_EXIT
    failures = check_transcript(a.stage, code, out, err)
    try:
        main_png = a.png or os.path.join(a.out, f"lanterns-stage{a.stage}.png")
        sheet = write_pngs(a.out, a.stage, main_png)
        print(f"lanterns: wrote {main_png} and {sheet}")
    except (OSError, ValueError) as e:
        failures.append(f"no frame dumps to draw: {e}")
    if failures:
        print("lanterns: FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("lanterns: " + marker(a.stage).search(out.replace("\r\n", "\n")).group(0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
