#!/usr/bin/env python3
"""G-buffer sandbox: placed props through a translated deferred material pair, every target read back.

    python -I tools/gbuffer.py --work <work> --entities <entities.jsonl> --models <assets> --textures <ttpf out> \
        [--center X Y] [--radius M] [--max-props N] [--dump PREFIX]

It packs the props around --center with tools/sandbox_scene.py, picks a deferred VS/PS pair from
<work>/shaders/spirv/index.json and runs runtime/build/vk_gbuffer, which fills the tables the pair reads (see its
header). The pair is chosen by what the shaders read, not by name: a vertex shader whose only input is
SV_VertexID, with no bindings, that builds its vertex stream V# from s11/s12, reads its index through an index V#
table at s18 and loads its object, view and viewport constants from s[9] + 0x60/0x80/0xa0; and a pixel shader
that exports mrt0..mrt4, samples one T# picked through s14/s15 with the S# at s16 + 0x520, and whose inputs the
VS's outputs cover. The PS reads its inputs by attribute number and the VS writes them by param number; the game
links them by semantic, so the link is computed from the two DXBC signatures and passed as --link.

Prints counts and PASS/FAIL only: exit 0 when the PS shaded the covered pixels, the albedo shows texture detail,
the decoded normals face the camera and the motion vectors of the still camera are zero; 77 when the package, the
pair or a GPU is missing. Standard library only.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox_scene  # noqa: E402

REG = r"s\[\d+:\d+\]"
VS_USER = [("ptr32_lo", 8), ("ptr32_lo", 9), ("const32", 11), ("const32", 12), ("const32", 13), ("const32", 14),
           ("const32", 15), ("const32", 16), ("ptr32_lo", 18)]
VS_LAYOUT = (r"\bs_mov_b32 s4, s9\b", rf"\bs_load_dwordx4 {REG}, s\[4:5\], 0x60\b", rf"\bs_load_dwordx4 {REG}, s\[4:5\], 0x80\b",
             rf"\bs_load_dwordx4 {REG}, s\[4:5\], 0xa0\b", rf"\bs_load_dwordx4 {REG}, s\[8:9\], 0x6e0\b",
             r"\bs_or_b32 s\d+, s12, 0x10000\b", r"\bs_mov_b32 s\d+, s11\b", r"\bs_mov_b32 s\d+, 0x11014fac\b",
             rf"\bs_lshl_b32 s\d+, s14, 5\b", rf"\bbuffer_load_format_x v\d+, v0, {REG}, 0 idxen\b",
             rf"\bs_buffer_load_dwordx8 {REG}, {REG}, 0x60\b", rf"\bs_buffer_load_dwordx16 {REG}, {REG}, 0x160\b",
             rf"\bs_buffer_load_dwordx16 {REG}, {REG}, 0x1a0\b")
PS_USER = [("ptr32_lo", 8), ("ptr32_lo", 9), ("const32", 14), ("ptr32_lo", 15), ("ptr32_lo", 16), ("const32", 17)]
PS_LAYOUT = (rf"\bs_load_dwordx4 {REG}, s\[8:9\], 0x6e0\b", rf"\bs_load_dwordx4 {REG}, s\[8:9\], 0xac0\b",
             r"\bs_mov_b32 s20, s9\b", rf"\bs_load_dwordx4 {REG}, s\[20:21\], 0x20\b",
             rf"\bs_load_dwordx4 {REG}, s\[20:21\], 0x60\b", rf"\bs_buffer_load_dword s\d+, {REG}, s14\b",
             r"\bs_mov_b32 s18, s15\b", rf"\bs_load_dwordx8 {REG}, s\[18:19\], s\d+\b",
             rf"\bs_load_dwordx4 {REG}, s\[16:17\], 0x520\b")


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


def vs_matches(entry, text):
    sig = [(e.get("semantic"), e.get("semantic_index")) for e in entry.get("input_signature") or []]
    return (entry.get("stage") == "vs" and not entry.get("bindings") and _user(entry) == VS_USER
            and sig == [("SV_VertexID", 0)] and "image_" not in text and _exports(text, "param") == list(range(6))
            and all(re.search(p, text) for p in VS_LAYOUT))


def ps_matches(entry, text):
    return (entry.get("stage") == "ps" and entry.get("bindings") == [0, 1, 2] and _user(entry) == PS_USER
            and text.count("image_sample") == 1 and _exports(text, "mrt") == list(range(5))
            and all(re.search(p, text) for p in PS_LAYOUT))


def dxbc_signature(path, tags):
    """[(semantic, index, register, read/write mask, min precision)] of the first chunk in tags, or None."""
    try:
        with open(path, "rb") as f:
            d = f.read()
        count = struct.unpack_from("<I", d, 28)[0]
        for k in range(count):
            off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
            if d[off:off + 4] not in tags:
                continue
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            n, eo = struct.unpack_from("<II", p, 0)
            out = []
            for i in range(n):
                no, si, _, _, reg = struct.unpack_from("<5I", p, eo + 32 * i + 4)
                out.append((p[no:p.index(b"\0", no)].decode("ascii").upper(), si, reg, p[eo + 32 * i + 25],
                            struct.unpack_from("<I", p, eo + 32 * i + 28)[0]))
            return out
    except (OSError, struct.error, ValueError, UnicodeDecodeError):
        return None
    return None


def link(vs_out, vs_params, ps_in, interpolants):
    """Per PS attribute (the inputs it reads, by register), the VS param (outputs by register) of the same
    semantic and precision. None when one is missing, when the PS reads a different count than its interpolant
    count, when the VS exports a different count of params than it declares outputs (the order is unknown), or
    when two elements share a register."""
    vs_regs = [e[2] for e in vs_out if not e[0].startswith("SV_")]
    ps_regs = [e[2] for e in ps_in if not e[0].startswith("SV_") and e[3]]
    if len(set(vs_regs)) != len(vs_regs) or len(set(ps_regs)) != len(ps_regs):
        return None   # two elements packed in one register: params and attributes are per register
    params = [(s, i, mp) for s, i, _, _, mp in sorted((e for e in vs_out if not e[0].startswith("SV_")), key=lambda e: e[2])]
    attrs = [(s, i, mp) for s, i, _, rw, mp in sorted((e for e in ps_in if not e[0].startswith("SV_")), key=lambda e: e[2])
             if rw]
    if len(params) != vs_params or len(attrs) != interpolants or any(a not in params for a in attrs):
        return None
    return [params.index(a) for a in attrs]


def pick_pair(shaders_dir):
    """(vs.spv, ps.spv, link) of the first matching pair in name order, or None."""
    spirv = os.path.join(shaders_dir, "spirv")
    dxbc = os.path.join(shaders_dir, "dxbc")
    disasm = os.path.join(dxbc, "disasm")
    try:
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return None
    have = [n for n in sorted(index) if os.path.isfile(os.path.join(spirv, n + ".spv"))]
    vss = [n for n in have if index[n].get("stage") == "vs" and vs_matches(index[n], _listing(disasm, n))]
    pss = [n for n in have if index[n].get("stage") == "ps" and ps_matches(index[n], _listing(disasm, n))]
    for v in vss:
        out = dxbc_signature(os.path.join(dxbc, v + ".dxbc"), (b"OSG1",))
        if not out:
            continue
        nparams = len(_exports(_listing(disasm, v), "param"))
        for p in pss:
            inp = dxbc_signature(os.path.join(dxbc, p + ".dxbc"), (b"ISG1",))
            lk = inp and link(out, nparams, inp, index[p].get("ps_interpolants", -1))
            if lk:
                return os.path.join(spirv, v + ".spv"), os.path.join(spirv, p + ".spv"), lk
    return None


def find_tool(build):
    for name in ("vk_gbuffer.exe", "vk_gbuffer"):
        path = os.path.join(build, name)
        if os.path.isfile(path):
            return path
    return None


def parse_counts(line):
    return {k: float(v) for k, v in re.findall(r"(\w+)=([-\d.]+)", line)}


def check_counts(c, textured):
    """(ok, message) from vk_gbuffer's counts."""
    cov = c.get("covered", 0)
    ok = (c.get("pixels", 0) > 0 and cov >= c.get("pixels", 0) / 100 and c.get("shaded", 0) >= 0.99 * cov
          and (not textured or c.get("albedo_colours", 0) >= 256)
          and c.get("normals_facing", 0) >= 0.6 * cov and c.get("motion_still", 0) >= 0.99 * cov)
    return ok, (f"{int(cov)} of {int(c.get('pixels', 0))} pixels covered, {int(c.get('shaded', 0))} shaded, "
                f"{int(c.get('albedo_colours', 0))} albedo colours, {int(c.get('normals_facing', 0))} normals facing "
                f"the camera, {int(c.get('motion_still', 0))} still motion vectors")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="make_game.py work folder (holds shaders/spirv and shaders/dxbc)")
    ap.add_argument("--entities", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--textures", action="append", default=[])
    ap.add_argument("--center", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--radius", type=float, default=150.0)
    ap.add_argument("--max-props", type=int, default=400)
    ap.add_argument("--build", default=os.path.join(HERE, "..", "runtime", "build"))
    ap.add_argument("--scene", help="scene file to write (default: a temporary file)")
    ap.add_argument("--dump", help="also write the targets as PREFIX_*.ppm (keep them in the work folder)")
    a = ap.parse_args(argv)

    tool = find_tool(a.build)
    pair = pick_pair(os.path.join(a.work, "shaders"))
    if not tool or not pair or not os.path.isfile(a.entities):
        print("unsupported: " + ("vk_gbuffer is not built" if not tool else
                                 "no matching translated shader pair" if not pair else "no entities file"))
        return 77
    vs, ps, lk = pair
    tmp = None
    scene = a.scene
    if not scene:
        fd, scene = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        tmp = scene
    try:
        try:
            stats = sandbox_scene.build(argparse.Namespace(entities=a.entities, models=a.models, textures=a.textures,
                                                           center=a.center, radius=a.radius, max_props=a.max_props,
                                                           tex_size=256, out=scene))
        except SystemExit as e:   # build() exits when no entity resolves to an exported model
            print(f"unsupported: {e}")
            return 77
        print(f"scene: {stats['placed']} props, {stats['draws']} draws, {stats['textures']} textures, "
              f"{stats['textured_draws']} textured draws")
        if stats["draws"] == 0:
            print("unsupported: no prop near the centre resolves to an exported model")
            return 77
        x, y, z = stats["center"]
        cmd = [tool, scene, vs, ps, "--link", ",".join(map(str, lk)), "--size", "640", "360",
               "--cam", str(x - 40), str(y - 40), str(z + 25), "45", "-20"]
        if a.dump:
            cmd += ["--dump", a.dump]
        r = subprocess.run(cmd, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode:
            print(lines[-1] if lines else "vk_gbuffer failed")
            return r.returncode
        counts = next((parse_counts(s) for s in lines if s.startswith("vk_gbuffer:")), {})
        ok, msg = check_counts(counts, stats["textured_draws"] > 0)
        print(("PASS: " if ok else "FAIL: ") + msg)
        return 0 if ok else 1
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    sys.exit(main())
