"""SHADER-10 GPU checks: wave64 pixel and vertex shaders on 32-lane subgroups (docs/design.md, "Wave64 graphics:
partition invariance").

    python3 -I runtime/test/shaders/w64_graphics.py <build dir with spirv/> [tools dir] [name[@variant] ...]

A wave64 pixel shader is exact on 32-wide subgroups when its outputs do not change if the wave is cut into two
32-lane halves. The test draws each ps_w64_* fixture with the fragment stage's required subgroup size set to 64 (one
hardware wave, the reference) and to 32 (the halves) through VKH_SUBGROUP_SIZE of vk_harness, and compares:

  tier A, B   both images equal, and equal to the words the fixture's header derives (tier B's atomic compaction
              compares the set of stored slots, not their order)
  tier C      the words the header names differ between the two sizes, which shows the translator's detector is
              right to flag the shader (the other words stay equal)

A device that can require only one size (NVIDIA: 32 only) cannot run the comparison. There the default-size image is
checked against the derived words (tiers A, B), the tier C check is skipped, and two environment variables let a
second device stand in for the missing size: W64G_SAVE_REF=<dir> stores each 64-lane run of a device that can require
it (or that has 64 as its only size: AMD GCN), and W64G_REF=<dir> compares every run of the other device with those
files (equal for A and B, different for C).

A device whose subgroups can be narrower than 32 (Intel: 8..32) and that could not be held to 32 or 64 skips the tier
A and B pixel fixtures: its fragment stage may run fewer live lanes than 32 (Intel UHD 770 reports 32 but runs 16 live
invocations), and tiers A and B assume 32 or 64 live lanes (TEST-433).
A fixture
listed in NEEDS_LIVE_LANES (it reads lanes beyond the first 16) runs where ps_live_width measures that many live lanes
per fragment subgroup and skips, naming the width, where it is less. The measured width is not gl_SubgroupSize: the Intel UHD 770 reports 32 and
runs 16 live lanes (lane ids 0..15), so a read of lane L ^ 16 returns garbage there.

The vs_w64_* fixtures (TEST-251) are vertex shaders: Vulkan cannot require a vertex subgroup size, so the two sizes
come from RADV, which runs the VS stage as wave64 by default and as wave32 with RADV_PERFTEST=gewave32 (GFX10+).
Each draws 960 vertices with zero area (VKH_VERTICES) and stores four words per vertex to the heap; vs_w64_c_lane_id
doubles as the probe that the wave size really changed, and without RADV only the default-size checks run. RADV packs
62 or 64 vertices into a wave64 and 30 or 32 into a wave32, so the 32-lane run is a repacking, not the two halves of
the 64-lane waves: a stricter test of tiers A and B, which key every word on the vertex index.

The gs_w64_* fixtures are legacy geometry shaders, switched the same way (gewave32 covers the whole NGG stage). Each is
drawn after vs_ids with VKH_GS, finds its triangle from the vertex number vs_ids passes in TEXCOORD0, and stores four
words per triangle to the heap; gs_w64_c_lane_id is the geometry probe. RADV puts at most 42 GS invocations in a wave64.

TEST-363 runs the tier A and B pixel fixtures in three more ways, named <fixture>@<variant>:
  partial      drawn with vs_tri_partial, whose long edge cuts 2x2 quads, so they run with helper invocations. A
               covered pixel must give its derived words, an uncovered one keeps the clear value (-1.0 in all four
               channels). On RDNA2 a pixel wave's EXEC holds only the covered pixels until s_wqm_b64; ps_w64_a_quad_live
               reads that back as the covered count of each quad, and the ballot atomic counts covered pixels only.
  mr           the fragment shader gets the MaximallyReconvergesKHR execution mode (VK_KHR_shader_maximal_
               reconvergence, 77 without it), which the translator assumes for waterfall and election loops but does
               not request. With it a helper invocation must stay active in group operations; without it, it may not.
  partial+mr   both.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

EXE = ".exe" if os.name == "nt" else ""
UNSUPPORTED = 77
W = H = 64
CLEAR = 0xBF800000            # vk_harness clears the target to -1.0: a pixel no fragment wrote
PARTIAL_VS = "vs_tri_partial"
VARIANTS = ("partial", "mr", "partial+mr")
MAXIMALLY_RECONVERGES = 6023  # SPIR-V execution mode MaximallyReconvergesKHR
MR_EXTENSION = "SPV_KHR_maximal_reconvergence"


class Skip(str):
    """Neither a pass nor a failure."""


def _x(i):
    return i % W


def _y(i):
    return i // W


def _a_skip(i):
    x, y = _x(i), _y(i)
    return (7 * y + x if x % 4 == 1 else 1000 + x, int(x % 4 < 2), 2000 + y + (9 if y % 4 == 2 else 0), x % 4 + 1)


def _a_uniform_vote(i):
    return (_x(i) + 7, 11, 36, 27)

CLEAR = 0xBF800000     # the harness clears the target to -1.0f: the words of a pixel no shader wrote


def _a_discard(i):
    x, y = _x(i), _y(i)
    if x < 30 or y >= 50 or (x % 4 == 3 and y % 2 == 1):
        return (CLEAR,) * 4
    return (7 * y + x, 1, 2000 + y, x % 4 + 1)


def _b_waterfall(i):
    x, y = _x(i), _y(i)
    a = x % 4 + 4 * (y % 4)
    return (17 * a + x, a, 0, 0)


def _b_readfirstlane(i):
    return (_x(i) + 123, 277, 0, 0)


def _b_readlane_uniform(i):
    return (_x(i) + 123, 200, 0, 0)


def _b_ballot(i):
    return (int(_x(i) % 4 == 1), 0, 0, 0)


def _c_waterfall_count(i):
    x, y = _x(i), _y(i)
    return (17 * (x % 8 + 8 * (y % 8)) + x, None, 0, 0)       # None: the trip count, partition dependent


def _c_bcnt(i):
    return (None, int(_x(i) % 4 == 1), 0, 0)


def _c_readlane(i):
    return (None, _y(i) * W + _x(i), None, 0)                  # r2 = lane 7 of the wave: also partition dependent


def _c_lane_id(i):
    return (None, None, _x(i), 0)                              # r1 = lane id & 31 depends on the lane a pixel got


def _c_vote_stale(i):
    return (None, int(_x(i) == 0 and _y(i) == 0), 0, 0)


def _on(i):
    return int(_x(i) == 5 and _y(i) == 7)


def _a_one_lane(i):
    x, y, on = _x(i), _y(i), _on(i)
    return (7777 if on else 1000 + x, 3000 + 11 * on, 4000 + y + 13 * on, x % 4)


def _a_permlane(i):
    x, y = _x(i), _y(i)
    return ((x ^ 1) + 64 * y, x + 64 * (y ^ 1), 64 * y + x, 64 * y + x)


def _a_swizzle(i):
    x, y = _x(i), _y(i)
    return ((x ^ 1) + 64 * (y ^ 1), (x ^ 1) + 64 * y, 64 * y + x, (x & ~1) + 64 * (y & ~1))


def _a_bpermute(i):
    x, y = _x(i), _y(i)
    return ((x ^ 1) + 64 * y, (x ^ 1) + 64 * (y ^ 1), 64 * y + x, (x & ~1) + 64 * (y | 1))


def _c_vote_else(i):
    return (7 if _on(i) else None, _on(i), 0, 0)               # r0 of the other pixels: 0 or 5 by the partition


def _c_scc_vote(i):
    return (1 if _on(i) else None, _on(i), 0, 0)               # r0 = SCC of the vote: the wave's or the half's


def _c_execz_sgpr(i):
    return (7 if _on(i) else None, _on(i), 0, 0)               # r0 = s5: 7 only where the active side ran


def _a_quad_live(i):
    return (4, _x(i), _y(i), 0)                                # every quad covered when drawn with vs_ids


def covered(x, y):
    """Whether vs_tri_partial covers pixel (x, y): its centre (x + 0.5, y + 0.5) has 3x + 4y < 188.5."""
    return 3 * x + 4 * y <= 188


def _quad_covered(i):
    """How many pixels of pixel i's 2x2 quad vs_tri_partial covers."""
    qx, qy = _x(i) & ~1, _y(i) & ~1
    return sum(covered(qx + dx, qy + dy) for dx in (0, 1) for dy in (0, 1))


# vs_w64_* fixtures: i is the vertex index, x = i % 64 and y = i // 64 as for a pixel.
def _vs_a_skip(i):
    x, y = _x(i), _y(i)
    return (7 * y + x if x % 4 == 1 else 1000 + x, int(x % 4 == 1), 2000 + y + (9 if y % 4 == 2 else 0), x % 4 + 1)


def _vs_b_readlane_uniform(i):
    return (_x(i) + 123, 200, _x(i) + 123, 277)


def _vs_c_lane_id(i):
    return (None, None, i, 0)                                  # r0, r1 = the lane the vertex got (gs_: the triangle)


def _gs_a_ring(i):                                             # gs_w64_* fixtures: i is the triangle
    return (i, 3 * i + 1, 77, 0)


VS_ENV = {"VKH_SGPRS": "0,0,65536,0x30000000", "VKH_HEAP": "0=0"}
VS_VERTICES = 960        # 15 rows of 64 vertices: several waves at either size, and partial ones
VS_RECORDS = 0x1000 // 4  # heap word of vertex 0's record (4 words per vertex); below it, the atomic's counter and slots
GS_ENV = {"VKH_SGPRS": "0,0,0,0,0,0,65536,0x30000000", "VKH_HEAP": "0=0"}   # V# in s[4:7]: the GS entry sets s3


# name -> (tier, expected words per pixel (None = not fixed), environment of the draw)
FIXTURES = {
    "ps_w64_a_skip": ("A", _a_skip, {}),
    "ps_w64_a_one_lane": ("A", _a_one_lane, {}),
    "ps_w64_a_permlane": ("A", _a_permlane, {}),
    "ps_w64_a_swizzle": ("A", _a_swizzle, {}),
    "ps_w64_a_bpermute": ("A", _a_bpermute, {}),
    "ps_w64_a_uniform_vote": ("A", _a_uniform_vote, {"VKH_SGPRS": "0,0,0,0,9"}),

    "ps_w64_a_discard": ("A", _a_discard, {}),
    "ps_w64_a_quad_live": ("A", _a_quad_live, {}),
    "ps_w64_b_waterfall": ("B", _b_waterfall, {}),
    "ps_w64_b_readfirstlane": ("B", _b_readfirstlane, {"VKH_SGPRS": "0,0,0,0,123"}),
    "ps_w64_b_readlane_uniform": ("B", _b_readlane_uniform, {"VKH_SGPRS": "0,0,0,0,123"}),
    "ps_w64_b_readlane_uniform_lane": ("B", _b_readlane_uniform, {"VKH_SGPRS": "0,0,0,0,123"}),
    "ps_w64_b_ballot_atomic": ("B", _b_ballot, {"VKH_SGPRS": "0,0,65536,0x30000000", "VKH_HEAP": "0=0"}),
    "ps_w64_c_waterfall_count": ("C", _c_waterfall_count, {}),
    "ps_w64_c_bcnt_export": ("C", _c_bcnt, {}),
    "ps_w64_c_readlane": ("C", _c_readlane, {}),
    "ps_w64_c_lane_id": ("C", _c_lane_id, {}),
    "ps_w64_c_vote_stale": ("C", _c_vote_stale, {}),
    "ps_w64_c_vote_else": ("C", _c_vote_else, {}),
    "ps_w64_c_scc_vote": ("C", _c_scc_vote, {}),
    "ps_w64_c_execz_sgpr": ("C", _c_execz_sgpr, {}),
    "vs_w64_a_skip": ("A", _vs_a_skip, VS_ENV),
    "vs_w64_b_waterfall": ("B", _b_waterfall, VS_ENV),
    "vs_w64_b_readlane_uniform": ("B", _vs_b_readlane_uniform, dict(VS_ENV, VKH_SGPRS=VS_ENV["VKH_SGPRS"] + ",123")),
    "vs_w64_b_ballot_atomic": ("B", _b_ballot, VS_ENV),
    "vs_w64_c_bcnt_export": ("C", _c_bcnt, VS_ENV),
    "vs_w64_c_readlane": ("C", _c_readlane, VS_ENV),
    "vs_w64_c_lane_id": ("C", _vs_c_lane_id, VS_ENV),
    "gs_w64_a_ring": ("A", _gs_a_ring, GS_ENV),
    "gs_w64_c_lane_id": ("C", _vs_c_lane_id, GS_ENV),
}

PROBES = {"vs": "vs_w64_c_lane_id", "gs": "gs_w64_c_lane_id"}   # per stage: the largest lane id shows the wave size
_applied = {}
# TEST-363: the tier A and B pixel fixtures also run as <name>@<variant> (see the module docstring).
# Listed by hand: each one's words under partial coverage were derived for it (expectation()); a fixture added
# later joins only once its partial-coverage words are worked out.
VARIANT_FIXTURES = ["ps_w64_a_one_lane", "ps_w64_a_quad_live", "ps_w64_a_skip", "ps_w64_b_ballot_atomic",
                    "ps_w64_b_readfirstlane", "ps_w64_b_readlane_uniform", "ps_w64_b_waterfall"]
CHECKS = sorted(FIXTURES) + [f"{n}@{v}" for v in VARIANTS for n in VARIANT_FIXTURES]
ONLY_HERE = frozenset(FIXTURES) | {PARTIAL_VS}   # the shaders scripts/run_gpu_tests.py leaves to this runner
# Checks that fail on main for a known translator gap, reported as XFAIL. A pass is reported as XPASS but does not
# fail the run: whether a helper invocation joins a ballot without MaximallyReconvergesKHR is up to the driver, so
# some of these may pass on one GPU and not another. The translator starts a pixel shader with its helper
# invocations active (rdna2spirv.py's %active is true in every invocation); RDNA2's entry EXEC holds the covered
# pixels only.
_HELPERS_IN_EXEC = "helper lanes in the entry EXEC, issues/2078#issuecomment-7082188519"
KNOWN = {f"{n}@{v}": _HELPERS_IN_EXEC for n in ("ps_w64_a_quad_live", "ps_w64_a_skip", "ps_w64_b_ballot_atomic")
         for v in ("partial", "partial+mr")}


def split_check(check_name):
    """(fixture, variant) of a check name such as ps_w64_b_waterfall@partial (variant '' for the plain draw)."""
    name, _, variant = check_name.partition("@")
    return name, variant


def expectation(name, variant=""):
    """The expected words per pixel of `name` drawn in `variant` (None in a slot: not fixed by the fixture)."""
    base = FIXTURES[name][1]
    if "partial" not in variant:
        return base

    def expect(i):
        if not covered(_x(i), _y(i)):
            return (CLEAR,) * 4
        if name == "ps_w64_a_quad_live":
            return (_quad_covered(i), _x(i), _y(i), 0)
        if name == "ps_w64_a_skip":
            # r1 is s_wqm_b64 of a compare taken under the entry EXEC, where a helper's bit is 0 (a VOPC writes 0
            # for a lane off in EXEC): 1 only if a covered pixel of the quad has x % 4 == 1.
            qx, qy = _x(i) & ~1, _y(i) & ~1
            r1 = int(qx % 4 == 0 and (covered(qx + 1, qy) or covered(qx + 1, qy + 1)))
            r0, _, r2, r3 = base(i)
            return (r0, r1, r2, r3)
        return base(i)
    return expect


def _spirv_string(text):
    raw = text.encode() + b"\0"
    raw += b"\0" * (-len(raw) % 4)
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def with_maximal_reconvergence(src, dst):
    """Writes SPIR-V module `src` to `dst` with OpExtension SPV_KHR_maximal_reconvergence after the capabilities and
    OpExecutionMode MaximallyReconvergesKHR after the entry point."""
    data = Path(src).read_bytes()
    words = list(struct.unpack("<%dI" % (len(data) // 4), data))
    after_caps = after_entry = entry = None
    i = 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        if not count:
            raise ValueError(f"{src}: malformed instruction at word {i}")
        if op == 17:                                   # OpCapability
            after_caps = i + count
        elif op == 15 and entry is None:               # OpEntryPoint: model, id, name, interface
            entry, after_entry = words[i + 2], i + count
        elif op == 16 and words[i + 2] == MAXIMALLY_RECONVERGES:
            Path(dst).write_bytes(data)                # already there
            return
        i += count
    if after_caps is None or entry is None:
        raise ValueError(f"{src}: no OpCapability or OpEntryPoint")
    mode = [(3 << 16) | 16, entry, MAXIMALLY_RECONVERGES]
    ext = _spirv_string(MR_EXTENSION)
    words[after_entry:after_entry] = mode              # the later position first, so after_caps stays valid
    words[after_caps:after_caps] = [((1 + len(ext)) << 16) | 10] + ext
    Path(dst).write_bytes(struct.pack("<%dI" % len(words), *words))


def _radv_perftest(size):
    """RADV_PERFTEST for a vertex draw at `size`. Vulkan cannot require a vertex subgroup size (requiredSubgroupSizeStages
    has no VERTEX on RADV or on the AMD driver), but RADV runs the VS/GS (NGG) stage as wave32 with gewave32 and as
    wave64 without it on GFX10+. Other drivers ignore the variable; the probe then finds no change."""
    flags = [f for f in os.environ.get("RADV_PERFTEST", "").split(",") if f and f != "gewave32"]
    return ",".join(flags + ["gewave32"] * (size == 32))


def _wave_size(tools, spirv, stage, size, tmp):
    """Whether a vertex or geometry draw at `size` really ran at that wave size: the largest lane id of the stage's
    probe is 32..63 in a wave64 and below 32 in a wave32. Size 32 also needs size 64 to work, so the draws differ."""
    if (stage, size) not in _applied:
        def top(s):
            words, problem = _draw_heap(tools, spirv, PROBES[stage], s, tmp)
            if words is None and not problem.startswith("unsupported"):
                return None, f"{s}-wide {stage} subgroup probe: {problem}"
            return (None if words is None else max(w[0] for w in words)), None
        lane64, problem = top(64)
        if problem:
            return False, problem
        lane32, problem = top(32)
        if problem:
            return False, problem
        _applied[stage, 64] = lane64 is not None and 32 <= lane64 < 64
        _applied[stage, 32] = _applied[stage, 64] and lane32 is not None and lane32 < 32
    return _applied[stage, size], None


def _draw_heap(tools, spirv, name, size, tmp):
    """(records, heap) of one vs_/gs_ draw at `size` (None: as the environment says), or (None, message). A vertex
    fixture stores one record per vertex; a geometry fixture, drawn after vs_ids, one per triangle."""
    env = dict(os.environ, **FIXTURES[name][2])
    if size:
        env["RADV_PERFTEST"] = _radv_perftest(size)
    raw, heap = Path(tmp) / f"{name}.{size}.raw", Path(tmp) / f"{name}.{size}.heap"
    env.update(VKH_VERTICES=str(VS_VERTICES), VKH_HEAP_DUMP=str(heap))
    env.pop("VKH_SUBGROUP_SIZE", None)
    vs, records = os.path.join(spirv, name + ".spv"), VS_VERTICES
    if name.startswith("gs_"):
        env["VKH_GS"] = vs
        vs, records = os.path.join(spirv, "vs_ids.spv"), VS_VERTICES // 3
    # The triangles have zero area (a GS fixture emits no primitive), so no pixel is drawn and any fragment shader does.
    cmd = [os.path.join(tools, "vk_harness" + EXE), vs, os.path.join(spirv, "ps_w64_a_skip.spv"), str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    if r.returncode == UNSUPPORTED:
        return None, "unsupported: " + out.strip().splitlines()[-1]
    if r.returncode or not heap.is_file():
        return None, f"exit {r.returncode}: {' | '.join(out.strip().splitlines()[-3:])}"
    words = list(struct.unpack("<%dI" % (65536 // 4), heap.read_bytes()))
    return [tuple(words[VS_RECORDS + 4 * v:VS_RECORDS + 4 * v + 4]) for v in range(records)], words


_device_min = [None]   # the smallest subgroup size vk_harness reported for the device


# Pixel fixtures that read lanes beyond the first 16 (a half-wave exchange, lane L ^ 16), and how many live lanes per
# fragment subgroup they need. Empty on purpose: every fixture here passes at 16 live lanes (Intel UHD 770). The
# permute fixtures of TEST-319 (ps_w64_a_permlane, _a_swizzle, _a_bpermute) go in as {name: 32}.
NEEDS_LIVE_LANES = {}


def live_width(tools, spirv, tmp):
    """The most live invocations a fragment subgroup has on this device (ps_live_width.asm: popcount of EXEC over the
    frame), or None when the probe could not run. Not gl_SubgroupSize: an Intel iGPU reports 32 and runs 16."""
    raw = Path(tmp) / "live_width.raw"
    cmd = [os.path.join(tools, "vk_harness" + EXE), os.path.join(spirv, "vs_ids.spv"),
           os.path.join(spirv, "ps_live_width.spv"), str(raw)]
    env = dict(os.environ)
    env.pop("VKH_SUBGROUP_SIZE", None)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if r.returncode or not raw.is_file() or raw.stat().st_size != W * H * 16:
        return None
    return max(struct.unpack("<%dI" % (W * H * 4), raw.read_bytes()))


def draw(tools, spirv, name, size, tmp, variant=""):
    """(words per pixel, heap words or None, applied) of one draw. `applied`: the required size took effect (size
    None: no size asked, applied is None). The words are None when the draw failed; then `heap` is the message."""
    tier, expect, extra = FIXTURES[name]
    if name.startswith(("vs_", "gs_")):
        words, heap = _draw_heap(tools, spirv, name, size, tmp)
        if words is None:
            return None, heap, None
        if size:
            applied, problem = _wave_size(tools, spirv, name[:2], size, tmp)
            if problem:
                return None, problem, None
            return words, heap, applied
        return words, heap, None
    env = dict(os.environ, **extra)
    env.pop("VKH_SUBGROUP_SIZE", None)
    if size:
        env["VKH_SUBGROUP_SIZE"] = str(size)
    tag = f"{name}.{variant.replace('+', '_')}.{size}"
    raw, heap = Path(tmp) / f"{tag}.raw", Path(tmp) / f"{tag}.heap"
    if "VKH_HEAP" in extra:
        env["VKH_HEAP_DUMP"] = str(heap)
    fs = os.path.join(spirv, name + ".spv")
    if "mr" in variant:
        patched = Path(tmp) / f"{name}.mr.spv"
        if not patched.is_file():
            with_maximal_reconvergence(fs, patched)
        fs = str(patched)
    vs = os.path.join(spirv, (PARTIAL_VS if "partial" in variant else "vs_ids") + ".spv")
    cmd = [os.path.join(tools, "vk_harness" + EXE), vs, fs, str(raw)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    out = r.stdout + r.stderr
    if r.returncode == UNSUPPORTED:
        return None, "unsupported: " + out.strip().splitlines()[-1], None
    if r.returncode or not raw.is_file():
        return None, f"exit {r.returncode}: {' | '.join(out.strip().splitlines()[-3:])}", None
    data = raw.read_bytes()
    if len(data) != W * H * 16:
        return None, f"{len(data)} bytes of pixels, expected {W * H * 16}", None
    m = re.search(r"subgroup (\d+)\.\.(\d+)", out)
    if m:
        _device_min[0] = int(m.group(1))
    applied = None
    if size:
        applied = f"fragment subgroup size: {size}\n" in out.replace("\r", "")
    words = list(struct.iter_unpack("<4I", data))
    words_heap = None
    if "VKH_HEAP" in extra:
        words_heap = list(struct.unpack("<%dI" % (65536 // 4), heap.read_bytes()))
    return words, words_heap, applied


def expected_problem(name, words, heap, variant=""):
    """'' when the words (and heap) are what the fixture's header derives, else the first discrepancy."""
    expect = expectation(name, variant)
    wrong = []
    for i, w in enumerate(words):
        want = expect(i)
        if any(g is not None and g != v for g, v in zip(want, w)):
            wrong.append((i, w, want))
    if wrong:
        i, w, want = wrong[0]
        return f"{len(wrong)} of {len(words)} pixels differ from the derived words, first ({_x(i)},{_y(i)}): got {w}, want {want}"
    if name.endswith("_b_ballot_atomic"):
        partial = "partial" in variant           # only covered pixels are wanted lanes; a helper takes no slot
        want = sorted(i + 1 for i in range(len(words)) if _x(i) % 4 == 1 and (not partial or covered(_x(i), _y(i))))
        n = len(want)
        if heap[0] != n:
            return f"counter is {heap[0]}, want {n}"
        slots = heap[64:64 + n]
        if sorted(slots) != want:
            return f"slot words are not the {n} wanted pixel numbers once each ({len(set(slots))} distinct)"
        end = VS_RECORDS if name.startswith("vs_") else len(heap)   # the vertex records follow the slots
        if any(heap[64 + n:end]) or any(heap[1:64]):
            return "words outside the counter and the slots were written"
    return ""


def comparable(name, words, heap, variant=""):
    """What two draws of the same fixture must share when the shader is partition invariant."""
    expect = expectation(name, variant)
    stable = [tuple(0 if g is None else v for g, v in zip(expect(i), w)) for i, w in enumerate(words)]
    if name.endswith("_b_ballot_atomic"):
        return stable, sorted(heap[64:64 + heap[0]]), heap[0]
    return stable, None, None


def partition_dependent(name, a, b, variant=""):
    """Whether the words the fixture's header names as partition dependent differ between two draws."""
    expect = expectation(name, variant)
    return any(any(g is None and x != y for g, x, y in zip(expect(i), p, q)) for i, (p, q) in enumerate(zip(a, b)))


def check(tools, spirv, check_name):
    """"" when the check behaves as its tier says on this device, a Skip when the device cannot run the comparison."""
    name, variant = split_check(check_name)
    tier = FIXTURES[name][0]
    for shader in (name, PARTIAL_VS) if "partial" in variant else (name,):
        if not os.path.isfile(os.path.join(spirv, shader + ".spv")):
            return f"missing {shader}.spv"
    save, ref = os.environ.get("W64G_SAVE_REF"), os.environ.get("W64G_REF")
    with tempfile.TemporaryDirectory(prefix="leonida-w64g-") as tmp:
        runs = {}
        for size in (64, 32, None):
            words, heap, applied = draw(tools, spirv, name, size, tmp, variant)
            if words is None:
                if heap.startswith("unsupported"):
                    return Skip("SKIP (" + heap + ")")
                return f"{size or 'default'}: {heap}"
            if size is None or applied:
                runs[size] = (words, heap)
        compared = False
        # Tiers A and B mean exact on 32 or 64 live lanes. A device whose subgroups can be narrower than 32 (Intel:
        # 8..32) and that could not be held to 32 or 64 may run fewer live lanes than it reports (UHD 770: 16 live
        # behind a reported 32): reads across them return garbage.
        if tier in ("A", "B") and name.startswith("ps_") and 64 not in runs and 32 not in runs \
                and _device_min[0] is not None and _device_min[0] < 32:
            return Skip(f"SKIP (the fragment stage may run fewer live lanes than 32 here ({_device_min[0]}..) "
                        "and 32 or 64 could not be required: tiers A and B assume 32 or 64 live lanes)")
        # A fixture that reads a lane beyond the quad or the first 16 (NEEDS_LIVE_LANES) is judged only where the
        # measured live width is enough: the stage can report a subgroup size it does not fill (Intel: 32 reported,
        # 16 live, so a read of lane L ^ 16 is undefined). A required size (64 or 32 applied) is the device saying
        # the lanes are there, so then nothing is measured.
        need = NEEDS_LIVE_LANES.get(name)
        if need and 64 not in runs and 32 not in runs:
            live = live_width(tools, spirv, tmp)
            if live is not None and live < need:
                return Skip(f"SKIP (the fragment stage has {live} live lanes per subgroup here, measured with "
                            f"ps_live_width; this fixture needs {need})")
        if save and 64 in runs:
            Path(save).mkdir(parents=True, exist_ok=True)
            (Path(save) / (check_name + ".ref64.raw")).write_bytes(b"".join(struct.pack("<4I", *w) for w in runs[64][0]))
        if tier in ("A", "B"):
            for size, (words, heap) in runs.items():
                if size is None and len(runs) > 1:
                    continue                      # the default draw repeats one of the forced sizes
                p = expected_problem(name, words, heap, variant)
                if p:
                    return f"{size or 'default size'}: {p}"
            if 64 in runs and 32 in runs and comparable(name, *runs[64], variant) != comparable(name, *runs[32], variant):
                return "the images at required subgroup size 64 and 32 differ"
        elif 64 in runs and 32 in runs:
            if comparable(name, *runs[64], variant) != comparable(name, *runs[32], variant):
                return "a word that must not depend on the partition differs between 64 and 32"
            if not partition_dependent(name, runs[64][0], runs[32][0], variant):
                return "tier C but the images at required subgroup size 64 and 32 are equal: the detector is wrong"
            compared = True
        if ref and (Path(ref) / (check_name + ".ref64.raw")).is_file():
            data = (Path(ref) / (check_name + ".ref64.raw")).read_bytes()
            ref_words = list(struct.iter_unpack("<4I", data))
            for size, (words, heap) in runs.items():
                if size == 64:
                    continue
                if tier == "C" and comparable(name, ref_words, None) != comparable(name, words, None):
                    return "a partition-invariant word differs from the 64-lane reference"
                if tier == "C":
                    if comparable(name, ref_words, None, variant) != comparable(name, words, None, variant):
                        return "a word that must not depend on the partition differs from the 64-lane reference"
                    if not partition_dependent(name, ref_words, words, variant):
                        return "tier C but the run equals the 64-lane reference: the detector is wrong"
                    compared = True
                elif (not name.endswith("_b_ballot_atomic")
                      and comparable(name, ref_words, None, variant) != comparable(name, words, None, variant)):
                    return "the run differs from the 64-lane reference"
        if tier == "C" and not compared:
            return Skip("SKIP (tier C needs a device that can require both 64 and 32, or W64G_REF)")
    return ""


def main(argv):
    if not argv:
        sys.exit(__doc__)
    spirv = os.path.join(argv[0], "spirv")
    tools = argv[1] if len(argv) > 1 and os.path.isdir(argv[1]) else os.path.join(os.path.dirname(__file__), "..", "..", "build")
    start = 2 if len(argv) > 1 and os.path.isdir(argv[1]) else 1
    requested = argv[start:]
    unknown = [a for a in requested if a not in CHECKS]
    if unknown:
        print(f"Unknown fixture name{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}", file=sys.stderr)
        return 1
    names = requested or CHECKS
    failed = 0
    for name in names:
        problem = check(os.path.abspath(tools), spirv, name)
        if name in KNOWN and problem and not isinstance(problem, Skip):
            problem = Skip(f"XFAIL ({KNOWN[name]}): {problem}")
        elif name in KNOWN and not problem:
            problem = Skip(f"XPASS (expected to fail here: {KNOWN[name]}; not a failure, see KNOWN)")
        print(f"{name:39} {FIXTURES[split_check(name)[0]][0]}  {'ok' if not problem else problem}")
        failed += bool(problem) and not isinstance(problem, Skip)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
