"""Run Lanterns under gta6rt --run, check what it printed, and turn its frame dumps into PNGs.

    python -I runtime/test/lanterns/run_lanterns.py --gta6rt <gta6rt.exe> --guest <lanterns.exe>
        --shaders <folder fssm_pack.py wrote> --out <folder> [--stage 1] [--wine] [--timeout 240] [--png FILE]

It sets LEONIDA_D3D12=vulkan and LEONIDA_SHADERS=<shaders>/spirv and runs
"gta6rt --run lanterns.exe -- --stage N --shaders <shaders> --out <out>" (under Wine with --wine, paths turned into
Z:\\ paths). The run passes only if all of these hold, which a loader that exits 0 early, a guest that prints and then
fails, or the host's PC device instead of the shim's Vulkan device would each break:
  - the exit code is 0;
  - stdout has the stage's PASS line exactly once, whole, with numbers that fit together (PASS_LINES; the guest
    prints it after every check);
  - the output has the shim's complete factory line "[rt] d3d12_xs: D3D12XboxCreateDevice -> proxy 0x... (Vulkan
    device 0x...)" (the guard PR #3913's run_guest_draw.cmake uses).
The timeout is a promise: when it expires everything the run started is killed and the run fails with exit code 3,
whatever the guest, Wine or the GPU are doing. On Linux the runner is a child subreaper, so a process that left the
launcher's session and lost its parent still comes back to it and is killed too; on Windows the process tree is
killed with taskkill /T. If anything survives (or taskkill fails), the run fails with exit code 4 and says what is
left. tests/test_run_lanterns.py checks it with a launcher that never ends and a child that escapes its session.

With --wine the prefix's wineserver is stopped (wineserver -k, then -w, each under a 10 s timeout) before the
launch: a launch under a new virtual X server while the previous run's explorer.exe is alive hangs in winevulkan.

Each run writes into a new folder, <out>/stage<N>-<date>-<time>-<suffix>: the guest's dumps, the launcher's output and
the PNGs, so no file of an earlier run can stand in for this one's.

PNGs (sRGB-encoded from the linear float target), lanterns-stage<N>.png and lanterns-stage<N>-sheet.png in that
folder:
  stage 1: the GPU frame with the depth test; sheet: GPU | CPU reference / GPU without depth (the negative control) |
           mask of the depth frame: certain and exact (grey), certain and wrong (red), uncertain (yellow)
  stage 3: frames 1 and 2 side by side (the points rise); sheet: frame 2 | its reference / the control frame without
           the Wait | mask of frame 2
  stage 5: the fp16 scene and the frame after the bloom post-process; sheet: final | its reference / the composite
           with the blur skipped (the control) | mask of the final frame (yellow: bounded, not exact, pixels)
  stage 6: the textured frame; sheet: frame | the middle of its reference bounds / the control (solids two mips
           down) | where the control left the frame's bounds (red, as predicted)
  stage 8: the shadowed frame; sheet: frame | the middle of its reference bounds / the control (map cleared to far) |
           the control's mask against its own all-lit reference (grey exact, yellow bounded, red outside)
  stage 9: the lit frame; sheet: frame | reference / the G-buffer albedo | mask / one light moved | frame
  stage 10: the 4x resolve and the 1x render side by side; sheet: resolve | reference / 1x | mask (yellow: edge pixels
            whose samples differ, grey: uncertain)
  stage 11: the ray-traced shadows; sheet: frame | reference / the control (one instance masked) | mask /
            the second control (two instances swap BLAS references) | frame
"""
import argparse
import array
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zlib

WIDTH, HEIGHT = 384, 256
TIMEOUT_EXIT = 3
CLEANUP_EXIT = 4
WINESERVER_TIMEOUT = 10   # seconds for each of wineserver -k and -w
FACTORY = re.compile(r"^\[rt\] d3d12_xs: D3D12XboxCreateDevice -> proxy (0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]* "
                     r"\(Vulkan device (0[xX])?0*[1-9A-Fa-f][0-9A-Fa-f]*\)$", re.M)


# Each stage's PASS line, whole, and what its numbers must satisfy. A line that only starts right ("Lanterns stage 1:
# PASS, nonsense") or whose counts do not fit together does not count.
N = r"(\d+)"
F = r"(\d+\.\d+)"   # a count printed with decimals
PASS_LINES = {
    1: (rf"Lanterns stage 1: PASS, {N} of 98304 pixels exact, {N} edge pixels plausible, depth-off control changed "
        rf"{N} pixels; shared edges through a wave64 VS/PS with the title's s8 table: {N} of 98304 pixels exact, {N} "
        rf"centres on shared edges each drawn once",
        lambda exact, edge, moved, fan, centres: exact + edge == 98304 and moved >= 1000 and fan == 98304
        and centres >= 500),
    3: (rf"Lanterns stage 3: PASS, {N} and {N} of 98304 pixels exact, 256 points exact after 24 and 48 compute steps, "
        rf"points moved {N} pixels", lambda a, b, moved: 0 < a <= 98304 and 0 < b <= 98304 and moved >= 50),
    5: (rf"Lanterns stage 5: PASS, fp16 scene {N} of {N} certain pixels exact, bloom composite {N} of {N} certain "
        rf"pixels exact and {N} bounded pixels within bounds, skipping the blur changed {N} pixels as predicted",
        lambda a, b, c, d, bounded, moved: a == b and c == d and d + bounded == 98304 and moved >= 1000),
    6: (rf"Lanterns stage 6: PASS, {N} of {N} certain pixels within the BC1/BC7 sampling bounds \(trilinear floor, "
        rf"image_sample_l solids\), control two mips down moved {N} of {N} predicted pixels",
        lambda a, b, moved, predicted: a == b and moved == predicted and predicted > 0),
    8: (rf"Lanterns stage 8: PASS, {N} of {N} single-colour pixels exact and {N} of {N} certain pixels within bounds "
        rf"\(D32 shadow map, PSO depth bias, 2x2 PCF with comparison samples\), the map cleared to far changed {N} "
        rf"pixels as predicted", lambda a, b, c, d, moved: a == b and c == d and moved >= 1000),
    9: (rf"Lanterns stage 9: PASS, G-buffer of 3 MRTs \(RGBA8 albedo {N} of {N} exact, RGB10A2 normal and R32F depth "
        rf"{N} of {N} probes\), {N} of {N} single-colour pixels exact through tiled culling with LDS atomics \({F} of "
        rf"256 lights per tile, the exact set in {N} of {N} tiles\), culling off gave the same image bit for bit "
        rf"from poisoned buffers, one light moved changed {N} of {N} predicted pixels",
        lambda a, b, p, q, s, t, per_tile, sets, tiles, moved, predicted: a == b and p == q and q >= 50000
        and s == t and per_tile < 128 and sets == tiles and moved == predicted and predicted >= 500),
    10: (rf"Lanterns stage 10: PASS, 4x MSAA resolved by ResolveSubresource: {N} of {N} certain pixels exact at the "
         rf"standard sample positions \({N} edge pixels blended exactly\), the 1x render differs on {N} of {N} predicted "
         rf"edge pixels and nowhere else certain; {N} of {N} uncertain pixels are one of their enumerated averages; "
         rf"SampleMask 1 << s isolated each sample \(at least {N} pixels only that sample's position explains\)",
         lambda a, b, blended, changed, predicted, e_ok, e, distinct: a == b and blended >= 1000
         and changed == predicted and predicted >= 1000 and e_ok == e and distinct > 0),
    11: (rf"Lanterns stage 11: PASS, one shadow ray per pixel through a TLAS of {N} instances over 2 BLASes built by "
         rf"BuildRaytracingAccelerationStructure, traversed with image_bvh_intersect_ray: {N} of {N} single-colour "
         rf"pixels exact, masking one instance changed {N} of {N} predicted pixels, swapping two BLAS references "
         rf"changed {N} of {N}; the step cap stopped {N} of {N} stalled rays and a full stack {N} of {N}",
         lambda inst, a, b, m, mp, s, sp, cap, rays, full, frays: inst >= 2 and a == b and b >= 50000
         and m == mp and mp > 0 and s == sp and sp > 0 and cap == rays and rays > 0 and full == frays and frays > 0),
}


def marker(stage):
    return re.compile(rf"^Lanterns stage {stage}: PASS\b.*$", re.M)


def check_transcript(stage, exit_code, stdout, stderr):
    """The reasons a finished run does not count as a pass (empty: it passes)."""
    stdout, stderr = stdout.replace("\r\n", "\n"), stderr.replace("\r\n", "\n")
    failures = []
    if exit_code != 0:
        failures.append(f"exit code {exit_code}, not 0")
    found = marker(stage).findall(stdout)
    if len(found) != 1:
        failures.append(f"stdout has the 'Lanterns stage {stage}: PASS' line {len(found)} times, not once")
    else:
        pattern, fits = PASS_LINES[stage]
        whole = re.fullmatch(pattern, found[0])
        if not whole or not fits(*(float(g) if "." in g else int(g) for g in whole.groups())):
            failures.append(f"the PASS line is not stage {stage}'s line with numbers that fit: {found[0]!r}")
    if not FACTORY.search(stderr) and not FACTORY.search(stdout):
        failures.append("no line shows D3D12XboxCreateDevice returning the shim's Vulkan device")
    return failures


def become_subreaper():
    """Linux: a process the run starts that leaves its session and then loses its parent is re-parented to this
    process instead of init, so descendants() still finds it. Elsewhere nothing happens."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0   # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        return False


def descendants(root):
    """The live (not zombie) processes below root, from /proc; empty where there is no /proc."""
    children = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read().decode("ascii", "replace")
        except OSError:
            continue
        state, parent = stat[stat.rfind(")") + 2:].split()[:2]
        if state != "Z":
            children.setdefault(int(parent), []).append(int(entry))
    found, todo = [], [root]
    while todo:
        for pid in children.get(todo.pop(), []):
            found.append(pid)
            todo.append(pid)
    return found


def kill_tree(proc):
    """Kill the launcher and everything it started and reap it. Returns what went wrong (empty: all of it is gone).
    Never waits unboundedly."""
    problems = []
    if os.name == "nt":
        try:
            # taskkill fails (128) for a launcher that exited after the timeout fired: that is not a survivor.
            r = subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30)
            if r.returncode != 0 and proc.poll() is None:
                problems.append(f"taskkill /T /F exited with {r.returncode}: "
                                f"{(r.stdout + r.stderr).decode('utf-8', 'replace').strip()}")
        except (OSError, subprocess.TimeoutExpired) as e:
            problems.append(f"taskkill /T /F did not run to the end: {e}")
            proc.kill()
    else:
        deadline = time.monotonic() + 10
        while True:
            left = descendants(os.getpid())
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            for pid in left:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            for pid in left:   # the orphans re-parented to this process
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            if not left or time.monotonic() > deadline:
                break
            time.sleep(0.1)
        if left:
            problems.append(f"still alive after SIGKILL: pids {left}")
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        problems.append(f"the launcher (pid {proc.pid}) did not exit")
    return problems


def stop_wineserver(env):
    """Wine mode, before each launch: stop the prefix's wineserver (and with it the explorer.exe it keeps), then wait
    for it to be gone, each under WINESERVER_TIMEOUT. A launch under a new virtual X server while the previous run's
    explorer is alive hangs in winevulkan's device creation (wine_vk_init), because that explorer keeps the old, closed
    DISPLAY: 20 of 20 such launches hung, 20 of 20 passed with this first. Returns what went wrong (empty: stopped, or
    none was running; wineserver exits nonzero when there is none)."""
    for flag in ("-k", "-w"):
        problem = run_bounded(["wineserver", flag], env, WINESERVER_TIMEOUT)
        if problem:
            return [problem]
    return []


def run_bounded(cmd, env, seconds):
    """Run cmd for at most seconds (plus 5 to reap it after SIGKILL); None when it exited in time, else what happened.
    Its output goes nowhere: a pipe a child of cmd keeps open would make even subprocess.run's own timeout wait."""
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=os.name != "nt")
    except OSError as e:
        return f"{' '.join(cmd)} did not start: {e}"
    try:
        proc.wait(timeout=seconds)
        return None
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return f"{' '.join(cmd)} did not return in {seconds:g} s and is still alive after SIGKILL (pid {proc.pid})"
    return f"{' '.join(cmd)} did not return in {seconds:g} s; killed"


def run(cmd, env, timeout, log_base):
    """(exit code, stdout, stderr, timed out, what the cleanup left). Output goes to <log_base>.stdout/.stderr files, not pipes: a process
    the launcher leaves behind (wineserver) may hold a pipe open long after gta6rt exits, so the wait is on the
    process. The child gets its own process group, so a timeout kills all of it."""
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with open(log_base + ".stdout", "wb") as out_file, open(log_base + ".stderr", "wb") as err_file:
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out_file, stderr=err_file, **kwargs)
        try:
            code, timed_out, left = proc.wait(timeout=timeout), False, []
        except subprocess.TimeoutExpired:
            left = kill_tree(proc)
            code, timed_out = None, True
    texts = []
    for suffix in (".stdout", ".stderr"):
        with open(log_base + suffix, "rb") as f:
            texts.append(f.read().decode("utf-8", "replace"))
    return code, texts[0], texts[1], timed_out, left


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
            if uncertain[i] == 1:
                row += b"\xe0\xc0\x20"
            elif uncertain[i] == 3 or (uncertain[i] == 0 and gpu[i * 4:i * 4 + 4] == ref[i * 4:i * 4 + 4]):
                row += b"\x50\x50\x50"
            else:
                row += b"\xff\x00\x00"
        rows.append(bytes(row))
    return rows


# The panels of each stage's PNGs: "<dump>.gpu" or "<dump>.ref" is a frame, "<dump>:mask" the uncertainty map of a dump.
PANELS = {
    1: {"main": [["stage1-depth.gpu"]],
        "sheet": [["stage1-depth.gpu", "stage1-depth.ref"], ["stage1-nodepth.gpu", "stage1-depth:mask"],
                  ["stage1-edges.gpu", "stage1-edges.ref"]]},
    3: {"main": [["stage3-frame1.gpu", "stage3-frame2.gpu"]],
        "sheet": [["stage3-frame2.gpu", "stage3-frame2.ref"], ["stage3-control.gpu", "stage3-frame2:mask"]]},
    5: {"main": [["stage5-scene.gpu", "stage5-final.gpu"]],
        "sheet": [["stage5-final.gpu", "stage5-final.ref"], ["stage5-skipped.gpu", "stage5-final:mask"]]},
    6: {"main": [["stage6-frame.gpu"]],
        "sheet": [["stage6-frame.gpu", "stage6-frame.ref"], ["stage6-control.gpu", "stage6-control:mask"]]},
    8: {"main": [["stage8-frame.gpu"]],
        "sheet": [["stage8-frame.gpu", "stage8-frame.ref"], ["stage8-control.gpu", "stage8-control:mask"]]},
    9: {"main": [["stage9-frame.gpu"]],
        "sheet": [["stage9-frame.gpu", "stage9-frame.ref"], ["stage9-albedo.gpu", "stage9-frame:mask"],
                  ["stage9-moved.gpu", "stage9-frame.gpu"]]},
    10: {"main": [["stage10-frame.gpu", "stage10-single.gpu"]],
         "sheet": [["stage10-frame.gpu", "stage10-frame.ref"], ["stage10-single.gpu", "stage10-frame:mask"]]},
    11: {"main": [["stage11-frame.gpu"]],
         "sheet": [["stage11-frame.gpu", "stage11-frame.ref"], ["stage11-masked.gpu", "stage11-frame:mask"],
                   ["stage11-swapped.gpu", "stage11-frame.gpu"]]},
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
    ap.add_argument("--out", required=True, help="each run makes a new folder in it")
    ap.add_argument("--stage", type=int, default=1, choices=sorted(PANELS))
    ap.add_argument("--wine", action="store_true", help="run gta6rt.exe under wine (paths become Z:\\...)")
    ap.add_argument("--timeout", type=float, default=240.0, help="seconds before the run is killed and fails")
    ap.add_argument("--png", help="where to write the main PNG (default: lanterns-stage<N>.png in the run's folder)")
    a = ap.parse_args(argv)
    for path in (a.gta6rt, a.guest, os.path.join(a.shaders, "spirv", "index.json")):
        if not os.path.exists(path):
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            return 2
    os.makedirs(a.out, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix=time.strftime(f"stage{a.stage}-%Y%m%d-%H%M%S-"), dir=a.out)
    print(f"lanterns: this run's files go to {run_dir}")
    if sys.platform.startswith("linux") and not become_subreaper():
        # Without it a process that left the launcher's session and lost its parent goes to init, where the cleanup
        # cannot find it; the runner would then claim a full cleanup it cannot check.
        print("lanterns: FAILED: prctl(PR_SET_CHILD_SUBREAPER) was refused, so the runner could not kill everything the "
              "run might start; not launching", file=sys.stderr)
        return CLEANUP_EXIT
    conv = wine_path if a.wine else os.path.abspath
    env = dict(os.environ, LEONIDA_D3D12="vulkan", LEONIDA_SHADERS=conv(os.path.join(a.shaders, "spirv")))
    cmd = (["wine"] if a.wine else []) + [a.gta6rt, "--run", conv(a.guest), "--", "--stage", str(a.stage),
                                          "--shaders", conv(a.shaders), "--out", conv(run_dir)]
    if a.wine:
        stuck = stop_wineserver(env)
        if stuck:
            print("lanterns: FAILED: the prefix's wineserver could not be stopped before the launch: " + "; ".join(stuck),
                  file=sys.stderr)
            return CLEANUP_EXIT
    started = time.monotonic()
    code, out, err, timed_out, left = run(cmd, env, a.timeout, os.path.join(run_dir, f"gta6rt-stage{a.stage}"))
    took = time.monotonic() - started
    print(f"--- gta6rt stdout ---\n{out}\n--- gta6rt stderr ---\n{err}\n--- exit: {code} ({took:.1f} s)")
    if timed_out:
        if left:
            print(f"lanterns: FAILED: no exit after {a.timeout:g} s, and the cleanup failed: " + "; ".join(left),
                  file=sys.stderr)
            return CLEANUP_EXIT
        print(f"lanterns: FAILED: no exit after {a.timeout:g} s; everything the run started was killed", file=sys.stderr)
        return TIMEOUT_EXIT
    failures = check_transcript(a.stage, code, out, err)
    try:
        main_png = a.png or os.path.join(run_dir, f"lanterns-stage{a.stage}.png")
        sheet = write_pngs(run_dir, a.stage, main_png)
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
