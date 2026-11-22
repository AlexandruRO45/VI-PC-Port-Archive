"""Wave32 vertex shaders on a vertex subgroup wider than 32 lanes (the tested RX 7600 Windows driver runs the
vertex stage 64-wide and omits VERTEX from requiredSubgroupSizeStages).

    python3 -I runtime/test/shaders/w32_vs_halves.py <dir with vk_harness> <spirv dir> [name ...]

Each vs_w32_* fixture draws 960 zero-area vertices and stores two records of four words per vertex to the heap: A at
0x1000 + 16 * v and B at 0x5000 + 16 * v (v = vertex index). Which vertices share a wave is the driver's choice, so the
fixtures are checked against what every wave32 execution must satisfy, whatever the packing. The vertices that report
one key (v_readfirstlane of v0) are one wave. That wave must:
  - have at most 32 vertices, each on its own lane 0..31 (v_mbcnt_lo of -1);
  - have its key's vertex on its lowest lane;
  - give every member the same wave-wide values (count, first lane, readlane, vote result);
  - give each cross-lane value the member's value on the lane it names.

A translation that runs a wave32 VS as one 64-lane wave breaks these: 64 vertices share a key, lanes 32..63 report
lane 32, the ballot counts both halves, and v_readlane / ds_bpermute read the wrong half.

Exit 0 when every fixture meets its invariants, 1 when one does not, 77 when nothing could be drawn. With
LEONIDA_VK_VALIDATION=1 (or VKH_VALIDATION=1) the draws run under VK_LAYER_KHRONOS_validation, loaded through the
loader's VK_INSTANCE_LAYERS with synchronization validation on, and any "Validation Error" line fails the fixture
(an interface error, ps_flat behind vs_w32_lane, prints VUID-RuntimeSpirv-OpEntryPoint-08743 there). The fragment
shader is ps_clip_color, which reads no input; the triangles have zero area, so it never runs.
check.py and scripts/run_gpu_tests.py call check() for each fixture.
"""
import os
import struct
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

EXE = ".exe" if os.name == "nt" else ""
UNSUPPORTED = 77
VERTICES = 960
A_WORD, B_WORD = 0x1000 // 4, 0x5000 // 4
ENV = {"VKH_SGPRS": "0,0,65536,0x30000000", "VKH_HEAP": "0=0", "VKH_VERTICES": str(VERTICES)}
FIXTURES = ("vs_w32_lane", "vs_w32_ballot", "vs_w32_shuffle", "vs_w32_vote")
PS = "ps_clip_color"


def draw(tools, spirv, name, tmp):
    """(records A, records B, harness output) or (None, message, output)."""
    env = dict(os.environ, **ENV)
    heap, raw = Path(tmp) / f"{name}.heap", Path(tmp) / f"{name}.raw"
    env["VKH_HEAP_DUMP"] = str(heap)
    if "1" in (os.environ.get("LEONIDA_VK_VALIDATION"), os.environ.get("VKH_VALIDATION")):
        env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
        env["VK_KHRONOS_VALIDATION_VALIDATE_SYNC"] = "true"
    cmd = [os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, name + ".spv"),
           os.path.join(spirv, PS + ".spv"), str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    if r.returncode == UNSUPPORTED:
        return None, "unsupported: " + (out.strip().splitlines() or ["?"])[-1], out
    if r.returncode or not heap.is_file():
        return None, f"exit {r.returncode}: {' | '.join(out.strip().splitlines()[-3:])}", out
    w = struct.unpack("<%dI" % (65536 // 4), heap.read_bytes())
    rec = lambda base: [tuple(w[base + 4 * v:base + 4 * v + 4]) for v in range(VERTICES)]
    return rec(A_WORD), rec(B_WORD), out


def waves(a):
    """key -> list of (lane, v) of the vertices that name that key."""
    g = defaultdict(list)
    for v, (lane, _, key, vv) in enumerate(a):
        g[key].append((lane, v))
    return g


def shape_problems(a):
    """The wave32 shape every fixture's A record carries: (lane, *, key, v)."""
    p = []
    bad_v = [v for v, r in enumerate(a) if r[3] != v]
    if bad_v:
        p.append(f"{len(bad_v)} records were not written (first vertex {bad_v[0]})")
        return p
    for key, members in sorted(waves(a).items()):
        lanes = [lane for lane, _ in members]
        if len(members) > 32:
            p.append(f"wave {key}: {len(members)} vertices share it (a wave32 has at most 32)")
        if any(lane > 31 for lane in lanes):
            p.append(f"wave {key}: lane ids up to {max(lanes)} (a wave32 has lanes 0..31)")
        if len(set(lanes)) != len(lanes):
            p.append(f"wave {key}: {len(lanes) - len(set(lanes))} vertices repeat a lane")
        if min(members)[1] != key:
            p.append(f"wave {key}: its lowest lane holds vertex {min(members)[1]}, not the key's")
    return p


def check_lane(a, b):
    p = []
    for key, members in waves(a).items():
        counts = {a[v][1] for _, v in members}
        if counts != {len(members)}:
            p.append(f"wave {key}: s_bcnt1 of exec_lo gave {sorted(counts)}, the wave has {len(members)} vertices")
    return p


def check_ballot(a, b):
    p = []
    for key, members in waves(a).items():
        lane_of = {v: lane for lane, v in members}
        sel = sorted(lane for lane, v in members if v % 4 == 1)
        want_first = sel[0] if sel else 0xFFFFFFFF
        for lane, v in members:
            count, first, s, _ = b[v]
            rank = a[v][1]
            want_rank = sum(1 for l in sel if l < lane)
            if s != int(v % 4 == 1):
                p.append(f"vertex {v}: v_cndmask of its own VCC bit gave {s}")
            if count != len(sel):
                p.append(f"wave {key}: s_bcnt1 of vcc_lo gave {count}, the wave selects {len(sel)}")
                break
            if first != want_first:
                p.append(f"wave {key}: s_ff1 of vcc_lo gave {first}, the lowest selected lane is {want_first}")
                break
            if rank != want_rank:
                p.append(f"vertex {v} (lane {lane_of[v]}): v_mbcnt_lo of VCC gave {rank}, want {want_rank}")
                break
    return p


def check_shuffle(a, b):
    p = []
    for key, members in waves(a).items():
        on = {lane: v for lane, v in members}
        for lane, v in members:
            r5, r31 = a[v][1], b[v][2]
            xor1, nxt = b[v][0], b[v][1]
            if 5 in on and r5 != on[5]:
                p.append(f"wave {key}: v_readlane lane 5 gave {r5}, lane 5 holds vertex {on[5]}")
                break
            if 31 in on and r31 != on[31]:
                p.append(f"wave {key}: v_readlane lane 31 gave {r31}, lane 31 holds vertex {on[31]}")
                break
            if xor1 != on.get(lane ^ 1, 0):
                p.append(f"vertex {v} (lane {lane}): ds_bpermute from lane {lane ^ 1} gave {xor1}, "
                         f"want {on.get(lane ^ 1, 0)}")
                break
            if nxt != on.get((lane + 1) & 31, 0):
                p.append(f"vertex {v} (lane {lane}): ds_bpermute from lane {(lane + 1) & 31} gave {nxt}, "
                         f"want {on.get((lane + 1) & 31, 0)}")
                break
    return p


def check_vote(a, b):
    p = []
    for key, members in waves(a).items():
        has37 = any(v == 37 for _, v in members)
        for lane, v in members:
            if a[v][1] != (7 if has37 else 0):
                p.append(f"wave {key}: s5 = {a[v][1]} after s_cbranch_vccz, want {7 if has37 else 0} "
                         f"(the wave {'has' if has37 else 'does not have'} vertex 37)")
                break
        for lane, v in members:
            if b[v][0] != int(lane < 16):
                p.append(f"vertex {v} (lane {lane}): wrote {b[v][0]} under exec_lo = 0xffff, want {int(lane < 16)}")
                break
    return p


CHECKS = {"vs_w32_lane": check_lane, "vs_w32_ballot": check_ballot, "vs_w32_shuffle": check_shuffle,
          "vs_w32_vote": check_vote}

# What each problem line is about, for the per-category counts (the first pattern a line contains).
CATEGORIES = (("vertices share it", "wave > 32 vertices"), ("lane ids up to", "lane id > 31"),
              ("repeat a lane", "repeated lane"), ("lowest lane holds", "readfirstlane not the first lane"),
              ("of exec_lo", "s_bcnt1 exec_lo"), ("s_bcnt1 of vcc_lo", "s_bcnt1 vcc_lo"), ("s_ff1", "s_ff1 vcc_lo"),
              ("v_mbcnt_lo of VCC", "v_mbcnt_lo rank"), ("own VCC bit", "per-lane VCC bit"),
              ("readlane lane 5", "v_readlane 5"), ("readlane lane 31", "v_readlane 31"),
              ("ds_bpermute from lane", "ds_bpermute"), ("s_cbranch_vccz", "vccz vote"),
              ("exec_lo = 0xffff", "exec_lo write"), ("validation errors", "validation"),
              ("not written", "records missing"))


def category(line):
    return next((name for pat, name in CATEGORIES if pat in line), "other")


def evaluate(name, a, b, out):
    """Every problem of one draw: the wave shape, the fixture's own checks, validation errors."""
    problems = shape_problems(a)
    if not problems or not problems[0].endswith("not written (first vertex 0)"):
        problems += CHECKS[name](a, b)
    errors = [ln for ln in out.splitlines() if "Validation Error" in ln]
    if errors:
        problems.append(f"{len(errors)} validation errors, first: {errors[0][:200]}")
    return problems


def check(tools, spirv, name):
    """'' when `name` meets every wave32 invariant on this device, 'SKIP (...)' when it can't be drawn, else what is
    wrong (for check.py and scripts/run_gpu_tests.py)."""
    for n in (name, PS):
        if not os.path.isfile(os.path.join(spirv, n + ".spv")):
            return f"missing {n}.spv"
    with tempfile.TemporaryDirectory(prefix="leonida-w32vs-") as tmp:
        a, b, out = draw(tools, spirv, name, tmp)
    if a is None:
        return f"SKIP ({b})" if b.startswith("unsupported") else b
    problems = evaluate(name, a, b, out)
    return f"{len(problems)} problems, first: {problems[0]}" if problems else ""


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    tools, spirv = argv[1], argv[2]
    names = argv[3:] or list(FIXTURES)
    failed = drawn = 0
    with tempfile.TemporaryDirectory(prefix="leonida-w32vs-") as tmp:
        for name in names:
            a, b, out = draw(tools, spirv, name, tmp)
            if a is None:
                print(f"{name}: {'SKIP' if b.startswith('unsupported') else 'FAIL'} ({b})")
                failed += not b.startswith("unsupported")
                continue
            drawn += 1
            n_wave = len(waves(a))
            sizes = sorted({len(m) for m in waves(a).values()})
            problems = evaluate(name, a, b, out)
            status = "FAIL" if problems else "ok"
            print(f"{name}: {status}: {n_wave} waves, sizes {sizes[0]}..{sizes[-1]}, "
                  f"largest lane {max(r[0] for r in a)}, {len(problems)} problems")
            if problems:
                counts = defaultdict(int)
                for line in problems:
                    counts[category(line)] += 1
                print("    by kind: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
                firsts = {}
                for line in problems:
                    firsts.setdefault(category(line), line)
                for line in firsts.values():
                    print(f"    {line}")
            failed += bool(problems)
    if not drawn and not failed:
        return UNSUPPORTED
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
