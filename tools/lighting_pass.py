#!/usr/bin/env python3
"""Deferred lighting pass for the G-buffer sandbox: the game's own translated tiled-lighting PS over a G-buffer.

  python -I tools/lighting_pass.py --work <work> [--build runtime/build] [--keep <folder>]

Picks, by layout and never by name, a full-screen VS and two lighting PS from <work>/shaders (one that reads the
directional-light constants, one that does not; docs/lighting_pass.md §2), lays out the table the PS reads
(the normal, the specular F0 and the diffuse albedo of the G-buffer, the light direction, and the tint and mask that
multiply the whole lighting result), runs runtime/test/vk_lighting for each probe and prints counts only. Every input
is synthetic: a uniform normal, F0 and albedo per run, mid-grey in the other slots. Exit 0 PASS, 1 FAIL, 77 when the
tool or a shader is missing or vk_lighting reports that the device lacks a feature it needs (its own exit 77).
"""
import argparse
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

REG = r"s\[\d+:\d+\]"
# User SGPRs: s8 table A, s9 table B (the shader copies it into another pair, usually s[10:11]), s14 the resource heap of the material
# loop (the dynamic T# lookups at s[14:15] + sN). The VS reads s10 only: the depth of the full-screen triangle.
PS_USER = [("ptr32_lo", 8), ("ptr32_lo", 9), ("ptr32_lo", 14)]
VS_USER = [("const32", 10)]
# Table offsets measured on the standard-material variants (docs/lighting_pass.md §3).
NORMAL_T = 0x1240        # A: G-buffer mrt1, the 24-bit octahedral normal (12 bits per axis in g, b, a)
F0_T = 0xBC0             # A: rgb behaves as the specular F0 (docs/lighting_pass.md §3)
DIFFUSE_T = 0x920        # A: rgb is the diffuse albedo (a is read as well; the tool keeps it at 255)
MASK_T = 0x1400          # A: one value per pixel; with TINT it multiplies the whole lighting result by lerp(1, tint, mask)
SCALE_V = 0x2A0          # A: V# of the pass scales: 0x0 pre-scale, 0x4 output scale, 0x8 scales the lighting term
OCCL_V = 0x2C0           # A: V#; 0x244 attenuates the lighting term (0 = none)
LIGHT_V = 0x0            # B: V# of the light constants: light direction at 0x110 (x, y, z), the tint at 0x148
SUN_DIR, TINT, INTENSITY, OCCL = 0x110, 0x148, 0x8, 0x244
PS_LAYOUT = (rf"\bs_load_dwordx8 {REG}, s\[8:9\], {NORMAL_T:#x}\b", rf"\bs_load_dwordx8 {REG}, s\[8:9\], {F0_T:#x}\b",
             rf"\bs_load_dwordx8 {REG}, s\[8:9\], {MASK_T:#x}\b", rf"\bs_load_dwordx4 {REG}, s\[8:9\], {SCALE_V:#x}\b",
             r"\bv_and_b32_e32 v\d+, 0xfff, v\d+", r"\bv_bfe_u32 v\d+, v\d+, 12, 12\b")
DIR_LAYOUT = (rf"\bs_buffer_load_dwordx4 {REG}, {REG}, {TINT:#x}\b", rf"\bs_buffer_load_dword s\d+, {REG}, {INTENSITY:#x}\b")
# vk_lighting formats (VkFormat numbers).
RGBA8, RGBA16F = 37, 97
DIMS = {"2D": "2d", "2D_ARRAY": "2darray", "3D": "3d", "1D": "1d", "CUBE": "cube"}
TABLE_A, TABLE_B, RES_HEAP, CB0, CB_SIZE = 0x0, 0x4000, 0x8000, 0x10000, 0x4000
W, H = 32, 18
NORMALS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
SUN = (1.0, 0.0, 0.0)     # towards the light; the shader takes the negated vector (it lights where -dot(n, d) > 0)


class Skip(Exception):
    """vk_lighting exited 77: the device lacks a feature the runner needs."""


def _user(entry):
    return [(u.get("kind"), u.get("sgpr")) for u in entry.get("user_sgpr_reads") or []]


def _listing(disasm_dir, name):
    try:
        with open(os.path.join(disasm_dir, name + ".s"), encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _exports(text, kind):
    return sorted({int(n) for n in re.findall(rf"\bexp {kind}(\d+)\b", text)})


def ps_matches(entry, text):
    sig = [(e.get("semantic"), e.get("semantic_index")) for e in entry.get("input_signature") or []]
    return (entry.get("stage") == "ps" and sig == [("SV_Position", 0)] and entry.get("outputs") == [0, 1]
            and _user(entry) == PS_USER and _exports(text, "mrt") == [0, 1]
            and all(re.search(p, text) for p in PS_LAYOUT))


def is_directional(text):
    return all(re.search(p, text) for p in DIR_LAYOUT)


def vs_matches(entry, text):
    sig = [(e.get("semantic"), e.get("semantic_index")) for e in entry.get("input_signature") or []]
    return (entry.get("stage") == "vs" and not entry.get("bindings") and sig == [("SV_VertexID", 0)]
            and _user(entry) == VS_USER and _exports(text, "pos") == [0] and _exports(text, "param") == [0]
            and "image_" not in text and "s_load" not in text)


def pick(shaders_dir):
    """(vs, dir_ps, nodir_ps, counts): .spv paths of the shortest matching listings, or None in place of a missing one."""
    spirv, disasm = os.path.join(shaders_dir, "spirv"), os.path.join(shaders_dir, "dxbc", "disasm")
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return None, None, None, {}
    vs, dps, nps = [], [], []
    for n in sorted(index):
        e = index[n]
        if not os.path.isfile(os.path.join(spirv, n + ".spv")) or e.get("stage") not in ("vs", "ps"):
            continue
        text = _listing(disasm, n)
        if e["stage"] == "vs" and vs_matches(e, text):
            vs.append((text.count("\n"), n))
        elif e["stage"] == "ps" and ps_matches(e, text):
            (dps if is_directional(text) else nps).append((text.count("\n"), n))
    path = lambda lst: os.path.join(spirv, min(lst)[1] + ".spv") if lst else None
    return path(vs), path(dps), path(nps), {"vs": len(vs), "ps_dir": len(dps), "ps_nodir": len(nps)}


def _rng(s):
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.fullmatch(r"s(\d+)", s)
    return [int(m.group(1))] if m else []


def table_slots(text):
    """{(table, offset): ('V', None) | ('T', dimension)} for every V# (dwordx4) and T# (dwordx8) the listing loads at
    a constant offset of table A (s[8:9]) or B (the pair the listing copies s9 into, `s_mov_b32 sN, s9`: s[10:11] in
    most variants, s[12:13] or others in some); a T#'s dimension is the first one its image instructions name (2D
    when none does)."""
    slots, src = {}, {}
    m = re.search(r"\bs_mov_b32 s(\d+), s9\b", text)
    b_pair = f"s[{int(m.group(1))}:{int(m.group(1)) + 1}]" if m else "s[10:11]"
    bases = {"s[8:9]": "A", b_pair: "B"}
    for line in text.splitlines():
        code = line.split(";")[0].strip()
        if not code:
            continue
        op, _, rest = code.partition(" ")
        ops = [o.strip() for o in re.split(r",\s*(?![^\[]*\])", rest)]
        if op in ("s_load_dwordx4", "s_load_dwordx8") and len(ops) > 2 and ops[1] in bases:
            off = ops[2].split()[0]
            off = 0 if off == "null" else int(off, 16) if off.startswith("0x") else None
            if off is not None:
                key = (bases[ops[1]], off)
                slots.setdefault(key, ["V" if op.endswith("x4") else "T", None])
                for r in _rng(ops[0]):
                    src[r] = key
            continue
        if op.startswith("image_"):
            res = [o.split()[0] for o in ops if o.startswith("s[")]
            key = src.get(_rng(res[0])[0]) if res and _rng(res[0]) else None
            dim = re.search(r"dim:SQ_RSRC_IMG_(\S+)", code)
            if key in slots and slots[key][0] == "T" and slots[key][1] is None and dim:
                slots[key][1] = DIMS.get(dim.group(1))
            continue
        if op.startswith("s_") and ops and not op.startswith(("s_waitcnt", "s_cmp", "s_cbranch", "s_branch", "s_nop",
                                                                "s_inst", "s_endpgm", "s_bitcmp", "s_buffer")):
            for r in _rng(ops[0]):
                src.pop(r, None)
    return {k: (v[0], v[1] or ("2d" if v[0] == "T" else None)) for k, v in slots.items()}


def oct24(n):
    """The 24-bit octahedral normal the lighting PS decodes from mrt1 g, b, a (u in bits 0-11, v in 12-23)."""
    x, y, z = n
    l1 = abs(x) + abs(y) + abs(z)
    u, v = x / l1, y / l1
    if z < 0:
        u, v = (1 - abs(v)) * (1 if u >= 0 else -1), (1 - abs(u)) * (1 if v >= 0 else -1)
    q = lambda t: max(0, min(4095, round((t + 1) * 2047.5)))
    return q(u) | (q(v) << 12)


def build(folder, vs, ps, text, literal_samplers, normal, sun, tint, mask=255, f0=(200, 200, 200), diffuse=(128, 128, 128)):
    """Write heap.bin and plan.txt for one probe into folder; returns the plan path."""
    slots = table_slots(text)
    heap = bytearray(CB0 + CB_SIZE * sum(1 for v in slots.values() if v[0] == "V"))
    plan = [f"size {W} {H}", f"vs {vs}", f"ps {ps}", f"heap {os.path.join(folder, 'heap.bin')}",
            f"out {os.path.join(folder, 'o')}", f"push 8 {TABLE_A:x}", f"push 9 {TABLE_B:x}", f"push 14 {RES_HEAP:x}"]
    plan += ["sampler " + " ".join(f"{w:x}" for w in s) for s in literal_samplers]
    plan += [f"target {RGBA16F}", f"target {RGBA16F}"]
    cbs, at = {}, CB0
    for (tab, off), (kind, dim) in sorted(slots.items()):
        va = (TABLE_A if tab == "A" else TABLE_B) + off
        if kind == "V":
            struct.pack_into("<4I", heap, va, at, 0, CB_SIZE, 3 << 28)   # raw V#: stride 0, OOB_SELECT 3
            struct.pack_into(f"<{CB_SIZE // 4}f", heap, at, *([0.5] * (CB_SIZE // 4)))
            cbs[(tab, off)] = at
            at += CB_SIZE
            continue
        q = oct24(normal)
        texel = {("A", NORMAL_T): bytes([255, q & 255, (q >> 8) & 255, (q >> 16) & 255]),
                 ("A", F0_T): bytes([*f0, 255]), ("A", DIFFUSE_T): bytes([*diffuse, 255]),
                 ("A", MASK_T): bytes([mask] * 4)}.get((tab, off), bytes([128, 128, 128, 255]))
        w, h, d = {"2d": (W, H, 1), "2darray": (W, H, 4), "cube": (8, 8, 6), "3d": (8, 8, 8), "1d": (64, 1, 1)}[dim]
        name = os.path.join(folder, f"t_{tab}{off:x}.bin")
        with open(name, "wb") as f:
            f.write(texel * (w * h * d))
        plan.append(f"image {dim} {RGBA8} {w} {h} {d} {name} {va:#x}")
    def put(tab, off, at_off, values):
        if (tab, off) in cbs:
            struct.pack_into(f"<{len(values)}f", heap, cbs[(tab, off)] + at_off, *values)
    put("A", SCALE_V, 0, [1.0, 1.0, 1.0])          # pre-scale, output scale, lighting-term scale
    put("A", OCCL_V, OCCL, [0.0])
    put("B", LIGHT_V, SUN_DIR, [-c for c in sun])
    put("B", LIGHT_V, TINT, [tint] * 3)
    with open(os.path.join(folder, "heap.bin"), "wb") as f:
        f.write(heap)
    path = os.path.join(folder, "plan.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(plan) + "\n")
    return path


def read_rgb(folder):
    """(mean r, g, b over target 0, every pixel of both targets finite) from vk_lighting's readback."""
    finite, out = True, None
    for t in (0, 1):
        with open(os.path.join(folder, f"o_t{t}.bin"), "rb") as f:
            b = f.read()
        v = struct.unpack(f"<{len(b) // 2}e", b)
        finite = finite and all(math.isfinite(x) for x in v)
        if t == 0:
            out = [sum(v[c::4]) / (len(v) // 4) for c in range(3)]
    return out, finite


def lighting_terms(run, sun, **kw):
    """Per normal in NORMALS: the total lighting term, mean output r with the tint 1 minus with 0 (with the mask at 255
    that is the whole lit result; docs/lighting_pass.md §3), and whether every pixel was finite."""
    terms, finite = [], True
    for n in NORMALS:
        lit, f1 = run(normal=n, sun=sun, tint=1.0, **kw)
        dark, f0 = run(normal=n, sun=sun, tint=0.0, **kw)
        terms.append(lit[0] - dark[0])
        finite = finite and f1 and f0
    return terms, finite


def check(dir_terms, flip_terms, mask_terms, nodir_terms, finite):
    """(ok, message): the normal facing the light gets at least 3x the total lighting term of the opposite one, a
    flipped light swaps them, a zero mask removes at least 99 % of it (the tint is then no longer applied), the variant
    without directional light has none, and every output is finite. NORMALS[0] faces SUN, NORMALS[1] faces away."""
    lit, away = dir_terms[0], dir_terms[1]
    ok = (finite and lit > 0 and lit >= 3 * max(away, 0.0) and flip_terms[1] >= 3 * max(flip_terms[0], 0.0)
          and flip_terms[1] > 0 and abs(mask_terms[0]) <= 0.01 * lit and all(abs(t) <= 1e-3 * lit for t in nodir_terms))
    msg = (f"total lighting facing {lit:.4g} away {away:.4g}; flipped light facing {flip_terms[0]:.4g} away {flip_terms[1]:.4g}; "
           f"mask 0: {mask_terms[0]:.4g}; no-directional variant max |{max(abs(t) for t in nodir_terms):.3g}|; "
           f"finite {finite}")
    return ok, msg


def albedo_ok(outputs):
    """outputs[c]: mean output rgb with a one-hot diffuse albedo (DIFFUSE_T) in channel c (red, green, blue). Each one
    puts its own channel above the other two by at least 5 % and leaves those two equal (within 0.1 %): the diffuse
    albedo only scales its own channel."""
    for c, o in enumerate(outputs):
        others = [o[k] for k in range(3) if k != c]
        if not (o[c] > 0 and all(o[c] >= 1.05 * x for x in others) and abs(others[0] - others[1]) <= 1e-3 * o[c]):
            return False
    return True


def find_tool(build):
    for name in ("vk_lighting.exe", "vk_lighting"):
        for sub in ("", "Release", "RelWithDebInfo", "Debug"):
            path = os.path.join(build, sub, name)
            if os.path.isfile(path):
                return path
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", required=True, help="the make_game.py work folder (shaders/spirv and shaders/dxbc)")
    ap.add_argument("--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runtime", "build"))
    ap.add_argument("--keep", help="write the plans and readbacks here instead of a temporary folder")
    a = ap.parse_args(argv)
    tool = find_tool(a.build)
    if not tool:
        print(f"lighting_pass: vk_lighting not found in {a.build} (cmake --build ... --target vk_lighting)")
        return 77
    shaders = os.path.join(a.work, "shaders")
    vs, dps, nps, counts = pick(shaders)
    print(f"lighting_pass: {counts.get('vs', 0)} VS, {counts.get('ps_dir', 0)} directional and "
          f"{counts.get('ps_nodir', 0)} other lighting PS match the layout")
    if not (vs and dps and nps):
        print("lighting_pass: a VS, a directional and a non-directional lighting PS are needed")
        return 77
    with open(os.path.join(shaders, "spirv", "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    disasm = os.path.join(shaders, "dxbc", "disasm")
    root = a.keep or tempfile.mkdtemp(prefix="lighting_pass_")
    os.makedirs(root, exist_ok=True)
    runs = [0]

    def runner(ps):
        name = os.path.splitext(os.path.basename(ps))[0]
        text = _listing(disasm, name)
        samplers = index[name].get("literal_samplers") or []

        def run(**kw):
            runs[0] += 1
            folder = os.path.join(root, f"run{runs[0]:03d}")
            os.makedirs(folder, exist_ok=True)
            plan = build(folder, vs, ps, text, samplers, **kw)
            r = subprocess.run([tool, plan], capture_output=True, text=True)
            if r.returncode == 77:
                raise Skip((r.stderr or r.stdout).strip()[-300:])
            if r.returncode:
                raise RuntimeError(f"vk_lighting exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
            return read_rgb(folder)
        return run

    try:
        run_dir, run_nodir = runner(dps), runner(nps)
        d, f1 = lighting_terms(run_dir, SUN)
        flip, f2 = lighting_terms(run_dir, tuple(-c for c in SUN))
        shad, f3 = lighting_terms(run_dir, SUN, mask=0)
        nod, f4 = lighting_terms(run_nodir, SUN)
        hot = [run_dir(normal=NORMALS[0], sun=SUN, tint=1.0, diffuse=tuple(255 if k == c else 0 for k in range(3)))
               for c in range(3)]
    except Skip as e:
        print(f"lighting_pass: SKIP, vk_lighting exit 77: {e}")
        return 77
    except RuntimeError as e:
        print(f"lighting_pass: {e}")
        return 1
    finally:
        if not a.keep:
            shutil.rmtree(root, ignore_errors=True)
    ok, msg = check(d, flip, shad, nod, f1 and f2 and f3 and f4 and all(f for _, f in hot))
    outs = [o for o, _ in hot]
    ok = ok and albedo_ok(outs)
    print(f"lighting_pass: {runs[0]} runs of {W}x{H}; {msg}; one-hot diffuse albedo r/g/b: "
          + ", ".join("/".join(f"{x:.4g}" for x in o) for o in outs))
    print("lighting_pass: PASS" if ok else "lighting_pass: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
