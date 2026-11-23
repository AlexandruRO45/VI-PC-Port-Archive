"""Draw stage 25's dumps (lanterns.exe --stage 25 --out DIR) as PNGs, and turn its profiler samples into hot spots.

    python3 -I runtime/test/lanterns/scale_report.py DIR [--crop X,Y] [--samples] [--gta6rt gta6rt.exe]
        [--symbolizer llvm-symbolizer]

PNGs written to DIR:
  stage25-main.png       the first frame's 3840x2160 main target, every second pixel (1920x1080)
  stage25-check.png      the same, grey, with the CPU reference check over it: red where a pixel in the 2x2 block is
                         wrong, yellow where one is uncertain (and one of its candidates)
  stage25-crop.png       256x256 pixels of the main target at full resolution around --crop (default: the first wrong
                         pixel), beside the same pixels of the check, both doubled
  stage25-aux.png        the 4x MSAA pass, resolved (1920x1080);  stage25-top.png  the RGBA8 pass (1920x1080)
  stage25-bad-N-diff.png for each saved bad frame N: the main target of frame N, with red where it differs from
                         frame 1 (2x2 blocks)
With --samples, stage25-samples.txt (module, RVA, thread, count) becomes stage25-hotspots.txt: samples per module,
and the functions of gta6rt.exe (the layer) by self samples with file:line, symbolized against its PDB.
"""
import argparse
import array
import collections
import importlib.util
import os
import struct
import subprocess
import sys

W, H = 3840, 2160
AUX_W, AUX_H, TOP_W, TOP_H = 1920, 1080, 1920, 1080
PARTICLES, INSTANCE = 256, 32
OFF_MAIN = 0
OFF_AUX = OFF_MAIN + W * 8 * H
OFF_TOP = OFF_AUX + AUX_W * 8 * AUX_H
OFF_PART_A = OFF_TOP + TOP_W * 4 * TOP_H
READBACK = OFF_PART_A + 2 * PARTICLES * INSTANCE


def load_run_lanterns():
    spec = importlib.util.spec_from_file_location("run_lanterns", os.path.join(os.path.dirname(__file__), "run_lanterns.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RL = load_run_lanterns()


def half_lut():
    lut = bytearray(65536)
    for h in range(65536):
        sign, exp, frac = h >> 15, (h >> 10) & 31, h & 1023
        if exp == 0:
            v = frac / 1024.0 * 2.0 ** -14
        elif exp == 31:
            v = 1.0 if frac == 0 else 0.0
        else:
            v = (1 + frac / 1024.0) * 2.0 ** (exp - 15)
        lut[h] = RL.srgb8(-v if sign else v)
    return lut


def read_frame(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != READBACK:
        raise ValueError(f"{path}: {len(data)} bytes, not {READBACK}")
    return data


def halves(data, offset, count):
    a = array.array("H")
    a.frombytes(data[offset:offset + count * 2])
    if sys.byteorder != "little":
        a.byteswap()
    return a


def hdr_rows(data, offset, width, height, step, lut, x0=0, y0=0, w=None, h=None):
    """RGB rows of an R16G16B16A16_FLOAT image, every step-th pixel, optionally a window."""
    w = w or width // step
    h = h or height // step
    rows = []
    for j in range(h):
        y = y0 + j * step
        line = halves(data, offset + (y * width + x0) * 8, (w * step) * 4)
        row = bytearray(w * 3)
        for i in range(w):
            k = i * step * 4
            row[i * 3] = lut[line[k]]
            row[i * 3 + 1] = lut[line[k + 1]]
            row[i * 3 + 2] = lut[line[k + 2]]
        rows.append(bytes(row))
    return rows


def ldr_rows(data, offset, width, height):
    rows = []
    for y in range(height):
        line = data[offset + y * width * 4: offset + (y + 1) * width * 4]
        rows.append(bytes(b for x in range(width) for b in line[x * 4:x * 4 + 3]))
    return rows


def check_rows(image_rows, marks, step, x0=0, y0=0, w=None, h=None):
    """Grey image with the check over it: red wrong, yellow uncertain, in step x step blocks."""
    w = w or W // step
    h = h or H // step
    rows = []
    for j in range(h):
        src = image_rows[j]
        row = bytearray(w * 3)
        for i in range(w):
            worst = 0
            for dy in range(step):
                base = (y0 + j * step + dy) * W + x0 + i * step
                for dx in range(step):
                    worst = max(worst, marks[base + dx])
            if worst == 2:
                row[i * 3:i * 3 + 3] = b"\xff\x00\x00"
            elif worst == 1:
                row[i * 3:i * 3 + 3] = b"\xe0\xc0\x20"
            else:
                g = (src[i * 3] + src[i * 3 + 1] + src[i * 3 + 2]) // 6
                row[i * 3:i * 3 + 3] = bytes((g, g, g))
        rows.append(bytes(row))
    return rows


def doubled(rows):
    out = []
    for r in rows:
        d = bytes(b for i in range(0, len(r), 3) for b in r[i:i + 3] * 2)
        out += [d, d]
    return out


def side_by_side(a, b):
    return [x + y for x, y in zip(a, b)]


def pngs(out, crop):
    lut = half_lut()
    frame1 = read_frame(os.path.join(out, "stage25-frame1.bin"))
    main = hdr_rows(frame1, OFF_MAIN, W, H, 2, lut)
    RL.png(os.path.join(out, "stage25-main.png"), W // 2, H // 2, main)
    marks_path = os.path.join(out, "stage25-frame1.marks.u8")
    marks = open(marks_path, "rb").read() if os.path.exists(marks_path) else bytes(W * H)
    RL.png(os.path.join(out, "stage25-check.png"), W // 2, H // 2, check_rows(main, marks, 2))
    if crop is None:
        bad = marks.find(b"\x02")
        crop = (bad % W, bad // W) if bad >= 0 else (W // 2, H // 2)
    cx = min(max(crop[0] - 128, 0), W - 256)
    cy = min(max(crop[1] - 128, 0), H - 256)
    window = hdr_rows(frame1, OFF_MAIN, W, H, 1, lut, cx, cy, 256, 256)
    checked = check_rows(window, marks, 1, cx, cy, 256, 256)
    RL.png(os.path.join(out, "stage25-crop.png"), 1024, 512, side_by_side(doubled(window), doubled(checked)))
    RL.png(os.path.join(out, "stage25-aux.png"), AUX_W, AUX_H, hdr_rows(frame1, OFF_AUX, AUX_W, AUX_H, 1, lut))
    RL.png(os.path.join(out, "stage25-top.png"), TOP_W, TOP_H, ldr_rows(frame1, OFF_TOP, TOP_W, TOP_H))
    print(f"scale_report: PNGs in {out} (crop at {cx},{cy})")
    for name in sorted(os.listdir(out)):
        if not (name.startswith("stage25-bad-") and name.endswith(".bin")):
            continue
        bad = read_frame(os.path.join(out, name))
        diff = bytearray(W * H)
        for y in range(H):
            a = frame1[OFF_MAIN + y * W * 8: OFF_MAIN + (y + 1) * W * 8]
            b = bad[OFF_MAIN + y * W * 8: OFF_MAIN + (y + 1) * W * 8]
            if a != b:
                for x in range(W):
                    if a[x * 8:x * 8 + 8] != b[x * 8:x * 8 + 8]:
                        diff[y * W + x] = 2
        rows = check_rows(hdr_rows(bad, OFF_MAIN, W, H, 2, lut), diff, 2)
        png = os.path.join(out, name[:-4] + "-diff.png")
        RL.png(png, W // 2, H // 2, rows)
        print(f"scale_report: {png}: {sum(1 for v in diff if v)} main pixels differ from frame 1")


def pe_image_base(path):
    with open(path, "rb") as f:
        data = f.read(4096)
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    magic = struct.unpack_from("<H", data, pe + 24)[0]
    return struct.unpack_from("<Q", data, pe + 24 + 24)[0] if magic == 0x20B else struct.unpack_from("<I", data, pe + 24 + 28)[0]


def module_file(name, gta6rt, wine_dlls):
    """The file on this machine behind a module name the guest recorded (a Wine path), or None."""
    low = name.lower()
    if low.endswith("gta6rt.exe"):
        return gta6rt
    if "\\windows\\system32\\" in low and wine_dlls:
        path = os.path.join(wine_dlls, low.rsplit("\\", 1)[1])
        return path if os.path.exists(path) else None
    if low.startswith("\\??\\z:"):
        return name[6:].replace("\\", "/")
    return None


def symbolize(symbolizer, path, rvas):
    """{rva: (function, file:line)} for the RVAs of one image, through llvm-symbolizer (PDB or export table)."""
    base = pe_image_base(path)
    addresses = sorted(rvas)
    proc = subprocess.run([symbolizer, f"--obj={path}", "--functions=linkage", "--demangle", "--inlining=false"],
                          input="\n".join(hex(base + a) for a in addresses) + "\n", capture_output=True, text=True,
                          timeout=600)
    blocks = proc.stdout.strip().split("\n\n")
    out = {}
    for a, block in zip(addresses, blocks):
        parts = block.split("\n")
        out[a] = (parts[0] if parts else "?", parts[1] if len(parts) > 1 else "?")
    return out


def short(name, width=150):
    return name if len(name) <= width else name[:width - 3] + "..."


def hotspots(out, gta6rt, symbolizer, top, wine_dlls):
    modules, samples, threads = {}, [], {}
    with open(os.path.join(out, "stage25-samples.txt")) as f:
        for line in f:
            parts = line.split()
            if parts[0] == "M":
                modules[int(parts[1])] = " ".join(parts[4:])
            elif parts[0] == "S":
                samples.append((int(parts[1]), int(parts[2], 16), int(parts[3]), int(parts[4])))
            elif parts[0] == "T":
                threads[int(parts[1])] = int(parts[2])
    total = sum(s[3] for s in samples)
    name_of = lambda m: (modules.get(m, "?") if m >= 0 else "(no image)").rsplit("\\", 1)[-1]
    # Symbolize every module we can find, once.
    symbols = {}
    for m in modules:
        path = module_file(modules[m], gta6rt, wine_dlls)
        if path and os.path.exists(path):
            symbols[m] = symbolize(symbolizer, path, {rva for mm, rva, _, _ in samples if mm == m})
    def function(m, rva):
        return symbols.get(m, {}).get(rva, ("?", "?"))[0] if m in symbols else f"{name_of(m)}+{rva:#x}"
    lines = [f"{total} samples, {len(threads)} threads (about one sample per thread per period)", ""]
    per_thread = collections.defaultdict(collections.Counter)
    for m, rva, t, n in samples:
        per_thread[t][(m, function(m, rva))] += n
    lines.append("by thread: samples, then the five functions it was most often in (module!function):")
    for t, counts in sorted(per_thread.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(counts.values())
        hot = ", ".join(f"{name_of(m)}!{short(fn, 60)} {100.0 * c / n:.0f}%" for (m, fn), c in counts.most_common(5))
        lines.append(f"  thread {t} (tid {threads.get(t, '?')}): {n} samples: {hot}")
    per_module = collections.Counter()
    for m, _, _, n in samples:
        per_module[name_of(m)] += n
    lines += ["", "samples by module:"]
    lines += [f"  {n:9d} {100.0 * n / total:5.1f}%  {name}" for name, n in per_module.most_common()]
    layer = [k for k, v in modules.items() if v.lower().endswith("gta6rt.exe")]
    if layer and layer[0] in symbols:
        sym = symbols[layer[0]]
        by_function, by_line = collections.Counter(), collections.Counter()
        for m, rva, _, n in samples:
            if m in layer:
                fn, loc = sym.get(rva, ("?", "?"))
                by_function[fn] += n
                by_line[(fn, loc)] += n
        where = {}
        for (fn, loc), n in by_line.most_common():
            where.setdefault(fn, loc)
        layer_total = sum(by_function.values())
        lines += ["", f"gta6rt.exe (the layer): {layer_total} samples; functions by self samples (the hottest line of each):"]
        for fn, n in by_function.most_common(top):
            lines.append(f"  {n:8d} {100.0 * n / layer_total:5.1f}% of the layer  {short(fn)}  [{where[fn]}]")
    for m in symbols:
        if m in layer:
            continue
        counts = collections.Counter()
        for mm, rva, _, n in samples:
            if mm == m:
                counts[symbols[m].get(rva, ("?", "?"))[0]] += n
        if sum(counts.values()) * 100 < total:
            continue
        lines += ["", f"{name_of(m)}: {sum(counts.values())} samples:"]
        lines += [f"  {n:8d} {short(fn)}" for fn, n in counts.most_common(8)]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(out, "stage25-hotspots.txt"), "w") as f:
        f.write(report)
    print(report)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PNGs and hot spots from Lanterns stage 25's dumps.")
    ap.add_argument("out")
    ap.add_argument("--crop", help="X,Y: the centre of the full-resolution crop")
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--gta6rt")
    ap.add_argument("--symbolizer", default="llvm-symbolizer")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--wine-dlls", default="/opt/wine-stable/lib/wine/x86_64-windows",
                    help="where the Wine builtin DLLs are (to name the ntdll/winevulkan samples)")
    a = ap.parse_args(argv)
    if not a.no_png:
        pngs(a.out, tuple(int(v) for v in a.crop.split(",")) if a.crop else None)
    if a.samples:
        hotspots(a.out, a.gta6rt, a.symbolizer, a.top, a.wine_dlls)
    return 0


if __name__ == "__main__":
    sys.exit(main())