"""Run a stratified sample of translated mesh shaders on one GPU two ways and compare the geometry they emit.

The worker is `runtime/test/vk_ms_sweep` (build the `vk_ms_sweep` target). It runs one mesh shader with the runtime's
bindless set 0 and a synthetic heap (zero, or small integers), and a generated pixel shader that outputs the sum of the
mesh outputs' varyings. It runs the shader twice on the same GPU: natively (`VK_EXT_mesh_shader`, mode `native`) and
as the `--mesh-compute` fallback that GPUs without mesh shaders get (mode `compute`: the shader as a compute shader
writing its vertices and primitives to a buffer, then `tools/mesh_compute_vs.py`'s vertex shader drawing them). They
should agree:

    python -I tools/ms_sweep.py sample    --spirv <translated spirv dir> --work <work>/shaders --out <dir> [--n 400] [--seed N]
    python -I tools/ms_sweep.py translate --out <dir>      # native and --mesh-compute, both from the sampled listings
    python -I tools/ms_sweep.py run       --out <dir> --exe <vk_ms_sweep> --glslang <glslangValidator> --mode native
    python -I tools/ms_sweep.py run       --out <dir> --exe <vk_ms_sweep> --glslang <glslangValidator> --mode compute
    python -I tools/ms_sweep.py compare   --out <dir>

The heap and the user SGPRs are zero. That is the limit of the synthetic input: the heap holds the shaders' own
descriptors (buffer bases, table pointers), so any data in it is read as a wild address and resets the device (tried:
floats, small integers, and the descriptor base SGPRs pointed at the heap), and most shaders then find a zero count and
emit one primitive per workgroup or nothing. `--lo/--hi/--only` select part of the sample.

What is compared, per shader: whether each mode ran (exit codes of the worker: 77 a missing device feature, 3 a
timeout, 4 device lost, 5 pipeline creation failed); the primitives the native workgroups generated
(`VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT`) against the primitives the fallback's slots hold; and the two
rendered 64x64 images (the pixel shader writes alpha 1, the target is cleared to -1). From the fallback's slots, the
only side that can be read back, it also counts vertices, NaN or infinite positions and primitive indices past the
vertex count.

Shaders are numbered by their place in `sample.json`; the summary lines and everything under `<dir>` except
`sample.json` and the staged inputs (which name the sampled shaders, so keep them out of reports) are counts only.
"""
import argparse
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W = H = 64
GROUPS = 4
HEAP_BYTES = 1 << 20
TOLERANCE = 1e-3
EDGE_PIXELS = 2          # a pixel or two on a triangle edge may differ between the two rasterised paths
VARIANTS = ("z",)


def load_ms(spirv):
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    return {k: v for k, v in index.items() if v.get("stage") == "ms"}


def stratum(meta, size_bin):
    """(size tercile, parameter count class, reads buffers, writes clip distances): what decides how a mesh shader runs."""
    p = len(meta.get("params", []))
    return (size_bin, "p0" if p == 0 else ("p1-6" if p <= 6 else "p7+"), bool(meta.get("bindings")),
            "ClipDistance" in meta.get("capabilities", []))


def size_bins(sizes):
    """Tercile (0, 1, 2) of each name by module size; sizes is {name: bytes}."""
    order = sorted(sizes, key=lambda k: (sizes[k], k))
    return {k: min(2, 3 * i // len(order)) for i, k in enumerate(order)}


def pick_sample(ms, sizes, n, seed, floor=6):
    """At least min(floor, size) shaders per stratum and the rest in proportion; `n` in all, or more when the floors
    alone add up to more."""
    bins = size_bins({k: sizes[k] for k in ms})
    groups = {}
    for name in sorted(ms):
        groups.setdefault(stratum(ms[name], bins[name]), []).append(name)
    total = sum(len(g) for g in groups.values())
    alloc = {s: min(len(g), max(floor, round(n * len(g) / total))) for s, g in groups.items()}
    while sum(alloc.values()) > n:
        big = max(sorted(alloc, key=str), key=lambda s: alloc[s])
        if alloc[big] <= min(floor, len(groups[big])):
            break
        alloc[big] -= 1
    # Rounding each group's share can leave the total short of n; give each missing slot to the group furthest below
    # its proportional share that still has candidates, the sorted stratum key breaking a tie, so n shaders come out
    # whenever the population holds n.
    order = sorted(groups, key=str)
    while sum(alloc.values()) < min(n, total):
        room = [s for s in order if alloc[s] < len(groups[s])]
        alloc[max(room, key=lambda s: n * len(groups[s]) / total - alloc[s])] += 1
    rng = random.Random(seed)
    picked = []
    for s in sorted(groups, key=str):
        picked += rng.sample(groups[s], alloc[s])
    return picked, {s: (len(groups[s]), alloc[s]) for s in groups}


def ps_glsl(locations):
    """A pixel shader for one mesh shader's outputs: the sum of its vec4 varyings, each weighted by its location, with
    alpha 1 so a covered pixel is recognisable even when a varying is NaN."""
    lines = ["#version 450"]
    for loc in locations:
        lines.append(f"layout(location={loc}) in vec4 i{loc};")
    lines.append("layout(location=0) out vec4 o;")
    lines.append("void main() {")
    lines.append("  vec4 s = vec4(0.0);")
    for loc in locations:
        lines.append(f"  s += i{loc} * {loc + 1}.0;")
    lines.append("  o = vec4(s.rgb, 1.0);")
    lines.append("}")
    return "\n".join(lines) + "\n"


def make_ps(glslang, locations, cache_dir):
    key = "ps_" + ("_".join(map(str, locations)) if locations else "none")
    spv = os.path.join(cache_dir, key + ".spv")
    if not os.path.exists(spv):
        src = spv[:-4] + ".frag"
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(ps_glsl(locations))
        r = subprocess.run([glslang, "-V", "-S", "frag", src, "-o", spv], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("glslangValidator failed: " + r.stdout[-300:])
    return spv


def cmd_sample(a):
    ms = load_ms(a.spirv)
    sizes = {k: os.path.getsize(os.path.join(a.spirv, k + ".spv")) for k in ms}
    picked, per = pick_sample(ms, sizes, a.n, a.seed)
    os.makedirs(a.out, exist_ok=True)
    bins = size_bins({k: sizes[k] for k in ms})
    with open(os.path.join(a.out, "sample.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump([{"name": k, "stratum": list(stratum(ms[k], bins[k]))} for k in picked], f)
    print(f"population {len(ms)} strata {len(per)} sampled {len(picked)}")
    for s in sorted(per, key=str):
        print(f"  stratum {s}: population {per[s][0]} sampled {per[s][1]}")
    if a.work:
        stage(a, picked)


def stage(a, names):
    """Copy the sampled shaders' listings, container headers and unpacked sidecars into <out>/stage, laid out the way
    rdna2spirv.py finds them (the listing folder's parent holds index.json, the .dxbc files and fssm/)."""
    dxbc = os.path.join(a.work, "dxbc")
    st = os.path.join(a.out, "stage")
    for d in ("disasm", "fssm"):
        os.makedirs(os.path.join(st, d), exist_ok=True)
    with open(os.path.join(dxbc, "index.json"), encoding="utf-8") as f:
        carve = json.load(f)
    want = set(names)
    keep = [e for e in carve if e["file"][:-5] in want]
    if len(keep) != len(want):
        raise SystemExit(f"carve index lists {len(keep)} of {len(want)} sampled shaders")
    with open(os.path.join(st, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(keep, f)
    for k in names:
        shutil.copyfile(os.path.join(dxbc, k + ".dxbc"), os.path.join(st, k + ".dxbc"))
        shutil.copyfile(os.path.join(dxbc, "disasm", k + ".s"), os.path.join(st, "disasm", k + ".s"))
        sidecar = os.path.join(dxbc, "fssm", k + ".fssm")
        if os.path.exists(sidecar):
            shutil.copyfile(sidecar, os.path.join(st, "fssm", k + ".fssm"))
    print("staged", len(names))


def cmd_translate(a):
    here = os.path.dirname(os.path.abspath(__file__))
    st = os.path.join(a.out, "stage", "disasm")
    for sub, extra in (("native", []), ("fb", ["--mesh-compute"])):
        out = os.path.join(a.out, sub)
        os.makedirs(out, exist_ok=True)
        r = subprocess.run([sys.executable, "-I", os.path.join(here, "rdna2spirv.py"), st, "--out", out,
                            "--wave64-emulate", "--no-asm"] + extra, capture_output=True, text=True)
        tail = [line for line in r.stdout.splitlines() if line.strip()][-3:]
        print(sub, "exit", r.returncode, *tail, sep="\n  " if tail else " ")
        if r.returncode:
            raise SystemExit(f"rdna2spirv.py failed for {sub}")
    import mesh_compute_vs
    built = mesh_compute_vs.build_for_index(os.path.join(a.out, "fb"))
    print("mesh vertex shaders", len(built))


def run_dir(a, tag):
    return os.path.join(a.out, f"{tag}_{a.variant}{a.suffix}")


def cmd_run(a):
    with open(os.path.join(a.out, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    with open(os.path.join(a.out, "fb", "index.json"), encoding="utf-8") as f:
        fb = json.load(f)
    out_dir = run_dir(a, a.mode)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(a.out, "ps")
    os.makedirs(cache, exist_ok=True)
    log = os.path.join(out_dir, f"results_{a.lo}.jsonl")
    done = set()
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            done = {json.loads(line)["i"] for line in f if line.strip()}
    hi = a.hi or len(sample)
    for i, s in enumerate(sample):
        if i < a.lo or i >= hi or i in done or (a.only and i not in a.only):
            continue
        name = s["name"]
        raw = os.path.join(out_dir, f"{i}.raw")
        if os.path.exists(raw):
            os.unlink(raw)
        env = dict(os.environ, VKS_TIMEOUT_S=str(a.timeout), VKS_HEAP_KB=str(HEAP_BYTES >> 10))
        entry = fb.get(name)
        if not entry or "mesh_compute" not in entry:
            rec = {"i": i, "code": 78, "reason": "not translated", "prims": None, "sec": 0}
        else:
            lay = entry["mesh_compute"]
            ps = make_ps(a.glslang, sorted(int(k) for k in lay.get("params", {})), cache)
            if a.mode == "native":
                cmd = [a.exe, "native", os.path.join(a.out, "native", name + ".spv"), ps, "-", raw, str(W), str(H),
                       str(GROUPS), "0", "0"]
            else:
                cmd = [a.exe, "compute", os.path.join(a.out, "fb", name + ".spv"), ps,
                       os.path.join(a.out, "fb", name + ".mesh_vs.spv"), raw, str(W), str(H), str(GROUPS),
                       str(lay["slot_bytes"]), str(lay["max_primitives"])]
            start = time.time()
            prims = None
            try:
                r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=a.timeout + 15,
                                   stdin=subprocess.DEVNULL)
                code, out, err = r.returncode, r.stdout, r.stderr
            except subprocess.TimeoutExpired:
                code, out, err = 3, "", ""
            reason = ""   # a coarse reason: a missing feature or the failing Vulkan call, never a shader name
            if code == 77:
                reason = out.strip().replace("unsupported: ", "")[:80]
            elif code not in (0, 3):
                for line in err.splitlines():
                    if line.startswith(("FAILED", "DEVICE_LOST", "PIPELINE")):
                        reason = line.split(" (line")[0][:60]
                        break
            for line in out.splitlines():
                if line.startswith("native_prims ") and line.split()[1].isdigit():
                    prims = int(line.split()[1])
            rec = {"i": i, "code": code, "reason": reason, "prims": prims, "sec": round(time.time() - start, 2)}
        with open(log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec) + "\n")
    print("done", a.mode, a.variant, a.lo)


def floats(data):
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data[: n * 4])


def analyse_slots(data, lay, groups=GROUPS):
    """What the fallback's slots hold: vertices and primitives emitted, NaN/infinite positions among the emitted
    vertices, primitives that index past the vertex count, culled primitives, and the largest count the shader asked
    for (the translator clamps to the maxima, so a count above them is a shader the clamp saved)."""
    slot = lay["slot_bytes"]
    res = {"verts": 0, "prims": 0, "culled": 0, "bad_pos": 0, "bad_index": 0, "over_max": 0, "groups": 0}
    for g in range(groups):
        base = g * slot
        if base + slot > len(data):
            return None
        nv, npr = struct.unpack_from("<II", data, base)
        if nv > lay["max_vertices"] or npr > lay["max_primitives"]:
            res["over_max"] += 1
            nv, npr = min(nv, lay["max_vertices"]), min(npr, lay["max_primitives"])
        res["groups"] += 1 if (nv or npr) else 0
        res["verts"] += nv
        res["prims"] += npr
        for v in range(nv):
            pos = struct.unpack_from("<4f", data, base + lay["vertices"] + v * lay["vertex_stride"])
            if not all(math.isfinite(x) for x in pos):
                res["bad_pos"] += 1
        for p in range(npr):
            i0, i1, i2, cull = struct.unpack_from("<4I", data, base + lay["primitives"] + p * lay["primitive_stride"])
            if cull:
                res["culled"] += 1
            if max(i0, i1, i2) >= nv:
                res["bad_index"] += 1
    return res


def compare_images(fa, fb, tol=TOLERANCE):
    """(covered pixels in a, covered in b, pixels that differ). A pixel is covered when the target was written (alpha 1,
    the clear is -1); it differs when the coverage differs or a colour channel differs beyond the tolerance, a NaN
    against a number or an infinity against anything else always counting."""
    pa = len(fa) // 4
    cov_a = cov_b = bad = 0
    for p in range(pa):
        a4, b4 = fa[4 * p: 4 * p + 4], fb[4 * p: 4 * p + 4]
        ca, cb = a4[3] == 1.0, b4[3] == 1.0
        cov_a += ca
        cov_b += cb
        if ca != cb:
            bad += 1
            continue
        if not ca:
            continue
        for x, y in zip(a4[:3], b4[:3]):
            if x == y or (math.isnan(x) and math.isnan(y)):
                continue
            if not (math.isfinite(x) and math.isfinite(y)) or abs(x - y) > tol + tol * max(abs(x), abs(y)):
                bad += 1
                break
    return cov_a, cov_b, bad


def classify(rec_n, rec_c, raw_n, raw_c, lay):
    """(class, sub-class, facts) of one shader from the native and fallback records and outputs."""
    cn, cc = rec_n["code"], rec_c["code"]
    if 3 in (cn, cc):
        return "timeout", "both" if cn == cc else ("native" if cn == 3 else "fallback"), {}
    if 4 in (cn, cc):
        return "device lost", "both" if cn == cc else ("native" if cn == 4 else "fallback"), {}
    if cn != 0 or cc != 0:
        if cn != 0 and cc != 0:
            return "fails in both modes", f"{cn}/{cc}", {}
        kind = lambda c: "missing device feature" if c == 77 else ("pipeline creation" if c == 5 else f"exit {c}")
        return "fails in one mode", ("native: " + kind(cn)) if cn else ("fallback: " + kind(cc)), {}
    with open(raw_n, "rb") as f:
        data_n = f.read()
    with open(raw_c, "rb") as f:
        data = f.read()
    # A capture shorter than the image would be compared only over the pixels it holds, so a missing or empty native
    # file read as "agree, empty"; either capture being short is a failure of that mode instead.
    if len(data_n) < W * H * 16:
        return "fails in one mode", "native: short output", {}
    if len(data) < W * H * 16:
        return "fails in one mode", "fallback: short output", {}
    img_n = floats(data_n[: W * H * 16])
    img_c = floats(data[: W * H * 16])
    slots = analyse_slots(data[W * H * 16:], lay)
    if slots is None:
        return "fails in one mode", "fallback: short output", {}
    cov_n, cov_c, bad = compare_images(img_n, img_c)
    facts = dict(slots, cov_native=cov_n, cov_fallback=cov_c, diff_px=bad, native_prims=rec_n.get("prims"))
    np_ = rec_n.get("prims")
    prim_ok = np_ is None or np_ in (slots["prims"], slots["prims"] - slots["culled"])
    if not prim_ok:
        return "primitive count differs", "", facts
    if bad > EDGE_PIXELS:
        return "image differs", "coverage" if cov_n != cov_c else "colour", facts
    return ("agree, empty" if not slots["prims"] else "agree, geometry"), ("NaN or Inf position" if slots["bad_pos"] else ""), facts


def load_results(a, tag):
    d = run_dir(a, tag)
    recs = {}
    for folder in (d, d + "_re"):   # re-runs override the first pass
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith("results_") and name.endswith(".jsonl"):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r = json.loads(line)
                            r["dir"] = folder
                            recs[r["i"]] = r
    return recs


def cmd_compare(a):
    with open(os.path.join(a.out, "sample.json"), encoding="utf-8") as f:
        sample = json.load(f)
    with open(os.path.join(a.out, "fb", "index.json"), encoding="utf-8") as f:
        fb = json.load(f)
    rn, rc = load_results(a, "native"), load_results(a, "compute")
    counts, sub, rows = {}, {}, []
    totals = {"verts": 0, "prims": 0, "bad_pos": 0, "bad_index": 0, "over_max": 0}
    for i, s in enumerate(sample):
        if i not in rn or i not in rc:
            raise SystemExit(f"shader {i} has no result in both modes")
        if rn[i]["code"] == 78 or rc[i]["code"] == 78:
            c, sc, facts = "not translated", "", {}
        else:
            lay = fb[s["name"]]["mesh_compute"]
            c, sc, facts = classify(rn[i], rc[i], os.path.join(rn[i]["dir"], f"{i}.raw"),
                                    os.path.join(rc[i]["dir"], f"{i}.raw"), lay)
        counts[c] = counts.get(c, 0) + 1
        sub[(c, sc)] = sub.get((c, sc), 0) + 1
        for k in totals:
            totals[k] += facts.get(k, 0)
        rows.append({"i": i, "class": c, "sub": sc, "stratum": s["stratum"], "native": rn[i]["code"],
                     "fallback": rc[i]["code"], "native_reason": rn[i]["reason"], "fallback_reason": rc[i]["reason"],
                     **{k: v for k, v in facts.items()}})
    with open(os.path.join(a.out, f"classes_{a.variant}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f)
    print("variant", a.variant, "total", len(sample))
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    for (c, sc), n in sorted(sub.items()):
        if sc:
            print(f"    {c} / {sc}: {n}")
    print("  emitted by the fallback, summed over the sample:", ", ".join(f"{k} {v}" for k, v in totals.items()))


WARNING = ("warning (run): a mesh shader that reads outside the synthetic heap can reset an NVIDIA GPU that drives a\n"
           "display (VK_ERROR_DEVICE_LOST, a driver reset). --timeout only bounds the wait on the host side. Run on a\n"
           "GPU with no display attached, or expect a freeze of a few seconds per hang.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=WARNING)
    ap.add_argument("cmd", choices=["sample", "translate", "run", "compare"])
    ap.add_argument("--spirv", help="the translated SPIR-V folder (index.json and the .spv files)")
    ap.add_argument("--work", help="<work>/shaders, to stage the sampled listings for `translate`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--mode", choices=["native", "compute"])
    ap.add_argument("--variant", choices=VARIANTS, default="z")
    ap.add_argument("--exe", help="vk_ms_sweep")
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("--timeout", type=int, default=20, help="seconds the GPU may take for one run")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=0)
    ap.add_argument("--only", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--suffix", default="", help="results folder suffix (a re-run uses _re)")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not (a.mode and a.exe):
        ap.error("run needs --mode and --exe")
    if a.cmd == "run" and a.only and not a.suffix:
        a.suffix = "_re"
    if a.cmd == "sample" and not a.spirv:
        ap.error("sample needs --spirv")
    {"sample": cmd_sample, "translate": cmd_translate, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
