"""Differential test of image_bvh_intersect_ray on real BVH nodes: the native RDNA2 instruction against a translator.

Takes runtime XDXR structures (for the game's BLASes: tools/bvh_rehydrate.py output, docs/bvh.md §7), samples nodes
and realistic rays for them, packs them into tools/hw_oracle kernels (one traversal step per case, the layout of the
measurement kernels in tools/bvh_tri_probe.py), runs the translator (tools/rdna2spirv.py of any checkout, executed by
that checkout's tests/semantic/spirv_mini_interp.py) on the same words, and prints counts of mismatches by class. It
prints counts only, and writes node words only to the --work folder the user names: a real structure is game data
and must not leave the user's own machine (docs/bvh.md §8).

    python -I tools/bvh_hw_diff.py sample <structure.bin> [...] --dehydrated <stream.bin> [...] --work DIR
    python -I tools/bvh_hw_diff.py translate --work DIR [--translator TREE] [--name main]
    python -I tools/bvh_hw_diff.py run --work DIR [--kernel NAME ...]      # the GPU: tools/hw_oracle, one dispatch each
    python -I tools/bvh_hw_diff.py score --work DIR [--against main]
    python -I tools/bvh_hw_diff.py --self-test                              # synthetic structures, no GPU

sample   Walks each structure from its runtime root (the first node of the last level-table entry of the dehydrated
         stream given with --dehydrated in the same order, or --root PTR,...), draws nodes (a BLAS is drawn with weight
         sqrt(its node count)) and rays, and writes DIR/cases.json and DIR/kernels/<k>.s. Triangle rays aim at
         interior points, the edge a paired leaf's two triangles share (slots 1-2), vertices and outer edges, from
         outside the parent's slot box or from nearby; each paired-leaf ray runs on TRI0 and TRI1. Box rays aim at a
         child, its centre or a corner, start inside a child, run axis-parallel, miss, or have the extent set exactly
         at a child's entry distance. Types 0, 1, 4 and 5 only; never 6/7.
run      Needs tools/bvh_tri_probe.py (PR #4202) for the kernel layout and its checks, and tools/hw_oracle's runner
         (LEONIDA_AMD_CLANG etc., as that probe documents). Box nodes run in the 32-bit form only, with their four
         child ids REPLACED by ids inside the oracle's buffer: the instruction returns child ids, and a real id would
         name memory outside it (the instruction is not known to fetch children, but that is not proven).
score    Mismatch classes (hardware as the reference, translator as the other side):
           T_EXACT, T_SCALE (quotients w0/w1, w2/w1, w3/w1 round alike), T_QUOT_ULP (t within 16 ulp, barycentrics
           within 2^-16), T_QUOT_FAR, T_MISS_WORDS (both miss), T_HITMISS_TIE (an edge function is exactly 0 in the
           float32 model), T_HITMISS_ROUND (an exact edge function, det or T within 16*2^-24 of its term magnitude),
           T_HITMISS_RULE; B_EXACT, B_LAYOUT, B_ORDER_TIE (inverted children have equal t0 exactly, in round-to-
           nearest or in round-toward-zero slab arithmetic), B_ORDER_ROUND (within 4 ulp), B_ORDER_RULE, B_SET_EQUAL
           (t0 == extent or t0 == tfar in RNE or RTZ), B_SET_ROUND (within 4 ulp), B_SET_RULE; *_UNDEFINED.
         Reference columns: the measured rules (docs/design.md Ray tracing row) in exact arithmetic, in float32 RNE
         and, for boxes, with slab distances rounded toward zero.
Standard library only. Exit status: 0 ok, 1 a stop condition or a failed self-test, 2 a usage or input error,
77 the hardware oracle or tools/bvh_tri_probe.py is unavailable.
"""
import argparse
import json
import math
import os
import random
import struct
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import bvh_rehydrate as R   # noqa: E402  (Header, levels, half helpers)

try:
    import bvh_tri_probe as P   # PR #4202: kernel layout, checks, results
except ImportError:             # pragma: no cover - main without #4202
    P = None

MISS = INVALID = 0xFFFFFFFF
INF = float("inf")
FMAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
NETWORK = ((0, 2), (1, 3), (0, 1), (2, 3), (1, 2))
K_BOX, K_TRI, N_T = 4, 16, 16
U24 = Fr(1, 1 << 24)
SURROGATE = [0x40, 0x48, 0x50, 0x58]       # in-buffer child ids: node region + 0x200..0x2c0 (tools/bvh_tri_probe layout)
VSLOTS = {0: (0, 1, 2), 1: (1, 3, 2), 2: (2, 3, 4), 3: (2, 4, 0)}
TRI_CLASSES = [("INT0", 15), ("INT1", 15), ("SHARED_EDGE", 30), ("SHARED_VERTEX", 10), ("OUTER_VERTEX", 10),
               ("OUTER_EDGE", 15), ("OFF_EDGE", 5)]
BOX_CLASSES = [("CHILD", 25), ("CENTER", 10), ("INSIDE", 15), ("AXIS", 10), ("EXTENT", 20), ("CORNER", 10),
               ("MISS", 10)]


# ---------------------------------------------------------------------------------------------------------------------
# float32 helpers and the measured rules
# ---------------------------------------------------------------------------------------------------------------------
def f2u(x):
    if x != x:
        return 0x7FC00000
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def u2f(w):
    return struct.unpack("<f", struct.pack("<I", w & 0xFFFFFFFF))[0]


def f32(x):
    return u2f(f2u(x)) if x == x else x


def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def ulp(x):
    a = abs(Fr(x))
    if a < Fr(1, 1 << 126):
        return Fr(1, 1 << 149)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fr(2) ** e > a:
        e -= 1
    return Fr(2) ** (e - 23)


def ulps_apart(a, b):
    a, b = Fr(a), Fr(b)
    return abs(a - b) / ulp(max(abs(a), abs(b)))


def rtz(q):
    if q == 0:
        return 0.0
    a = abs(q)
    if a >= Fr(FMAX):
        r = FMAX
    else:
        e = a.numerator.bit_length() - a.denominator.bit_length()
        if Fr(2) ** e > a:
            e -= 1
        step = Fr(2) ** (max(e, -126) - 23)
        r = float((a // step) * step)
    return r if q > 0 else -r


def boxes(node, ntype):
    out = []
    for i in range(4):
        if ntype == 5:
            b = [u2f(w) for w in node[4 + 6 * i:10 + 6 * i]]
        else:
            b = []
            for w in node[4 + 3 * i:7 + 3 * i]:
                b += [h2f(w), h2f(w >> 16)]
        out.append((b[:3], b[3:]))
    return out


def _isnan(x):
    return not isinstance(x, Fr) and x != x


def _op(mode, kind, a, b):
    """a - b or a * b: exact (Fractions while finite), float32 round-to-nearest, or float32 round-toward-zero."""
    fin = all(isinstance(v, Fr) or math.isfinite(v) for v in (a, b))
    if fin and mode in ("exact", "rtz"):
        q = Fr(a) - Fr(b) if kind == "-" else Fr(a) * Fr(b)
        return q if mode == "exact" else rtz(q)
    a, b = float(a), float(b)
    if kind == "*" and ((a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a))):
        return float("nan")
    r = (a - b) if kind == "-" else (a * b)
    return f32(r) if mode == "rne" and r == r else r


def box_eval(node, ntype, ray, mode="exact"):
    """The #3567 slab test with the measured extent rule max(tnear, 0) <= E (PR #4750), box sort on."""
    E = u2f(ray[0])
    o, inv = [u2f(w) for w in ray[1:4]], [u2f(w) for w in ray[7:10]]
    t0s, tfs, hits = [], [], []
    for lo, hi in boxes(node, ntype):
        near, far = [], []
        for c in range(3):
            b0 = _op(mode, "*", _op(mode, "-", lo[c], o[c]), inv[c])
            b1 = _op(mode, "*", _op(mode, "-", hi[c], o[c]), inv[c])
            n, f = (b0, b1) if inv[c] >= 0 else (b1, b0)
            near.append(n)
            far.append(f)
        nn, ff = [x for x in near if not _isnan(x)], [x for x in far if not _isnan(x)]
        if not nn or not ff:
            t0s.append(None), tfs.append(None), hits.append(False)
            continue
        t0, tf = max(max(nn), 0), min(ff)
        hits.append(E == E and t0 <= tf and t0 <= E)
        t0s.append(t0), tfs.append(tf)
    keys = [t0s[i] if hits[i] else INF for i in range(4)]
    ids = [node[i] if hits[i] else MISS for i in range(4)]
    for p, q in NETWORK:
        if keys[q] < keys[p]:
            keys[p], keys[q] = keys[q], keys[p]
            ids[p], ids[q] = ids[q], ids[p]
    return dict(ids=ids, t0=t0s, tf=tfs, hit=hits, E=E)


def _fin(x):
    return x is not None and (isinstance(x, Fr) or math.isfinite(x))


def box_margins(node, ntype, ray):
    ev = box_eval(node, ntype, ray, "exact")
    child = []
    for i in range(4):
        m = []
        if _fin(ev["t0"][i]) and _fin(ev["tf"][i]):
            m.append(ulps_apart(ev["t0"][i], ev["tf"][i]))
        if _fin(ev["t0"][i]) and math.isfinite(ev["E"]):
            m.append(ulps_apart(ev["t0"][i], ev["E"]))
        child.append(min(m) if m else None)
    pair = {(a, b): ulps_apart(ev["t0"][a], ev["t0"][b]) for a in range(4) for b in range(a + 1, 4)
            if ev["hit"][a] and ev["hit"][b] and _fin(ev["t0"][a]) and _fin(ev["t0"][b])}
    return ev, child, pair


def _kz(d):
    ad = [abs(c) for c in d]
    if ad[0] >= ad[1]:
        return 0 if ad[0] >= ad[2] else 2
    return 1 if ad[1] >= ad[2] else 2


def tri_eval(node, ntype, ray, mode="exact"):
    """The measured triangle step (docs/design.md Ray tracing row, PR #4202): the multiply-through shear, k = |d[kz]|^3.
    mode "exact": rationals; "rne": float32 after every op (the evaluation order of tools/bvh_tri_probe.rule_words)."""
    o, d = [u2f(w) for w in ray[1:4]], [u2f(w) for w in ray[4:7]]
    V = [[u2f(node[3 * s + c]) for c in range(3)] for s in VSLOTS[ntype]]
    if not all(math.isfinite(x) for x in o + d + [c for v in V for c in v]):
        return None
    kz = _kz(d)
    kx, ky = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[kz]
    if d[kz] < 0:
        kx, ky = ky, kx
    s = -1 if d[kz] < 0 else 1
    if d[kz] == 0:
        return None
    if mode == "exact":
        F = Fr
        sub = mul = None
        adz, px, py = abs(F(d[kz])), s * F(d[kx]), s * F(d[ky])
        rel = [[F(V[j][c]) - F(o[c]) for c in range(3)] for j in range(3)]
        sh = [(r[kx] * adz - px * r[kz], r[ky] * adz - py * r[kz], r[kz]) for r in rel]
        mag = [(abs(r[kx] * adz) + abs(px * r[kz]), abs(r[ky] * adz) + abs(py * r[kz])) for r in rel]
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = sh
        Ue, Ve, We = cx * by - cy * bx, ax * cy - ay * cx, bx * ay - by * ax
        (mx0, my0), (mx1, my1), (mx2, my2) = mag
        scale = (mx2 * my1 + my2 * mx1, mx0 * my2 + my0 * mx2, mx1 * my0 + my1 * mx0)
        det = Ue + Ve + We
        T = s * (Ue * az + Ve * bz + We * cz)
        Tsc = (scale[0] + abs(Ue)) * abs(az) + (scale[1] + abs(Ve)) * abs(bz) + (scale[2] + abs(We)) * abs(cz)
        return dict(U=Ue, V=Ve, W=We, det=det, T=T, t=T / (det * adz) if det else None, scale=scale, T_scale=Tsc)
    sub = lambda a, b: f32(a - b)     # noqa: E731
    mul = lambda a, b: f32(a * b)     # noqa: E731
    adz = abs(d[kz])
    px, py = mul(s, d[kx]), mul(s, d[ky])
    rel = [[sub(V[j][c], o[c]) for c in range(3)] for j in range(3)]
    sh = [(sub(mul(r[kx], adz), mul(px, r[kz])), sub(mul(r[ky], adz), mul(py, r[kz])), r[kz]) for r in rel]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = sh
    return dict(U=sub(mul(cx, by), mul(cy, bx)), V=sub(mul(ax, cy), mul(ay, cx)), W=sub(mul(bx, ay), mul(by, ax)))


def hw_model_words(node, ntype, ray):
    """The arithmetic the RX 6800 matched on real nodes (docs/bvh.md §7c): the measured rules with every float
    operation rounded toward zero. Boxes: box_eval(mode="rtz"). Triangles: V - o, the multiply-through shear, the edge
    functions, T = (U az + V bz) + W cz and word 1 = (k U + k V) + k W (k = |d[kz]|), each op RTZ; a zero edge
    function is decided by #4202's tie rule in the computation frame (it decided 202 of the 242 such real cases; the
    rest are open). None for inputs outside the model (non-finite vertices or direction)."""
    if ntype >= 4:
        return box_eval(node, ntype, ray, "rtz")["ids"]
    o, d, inv = ([u2f(w) for w in ray[a:a + 3]] for a in (1, 4, 7))
    if any(x != x for x in [u2f(ray[0])] + o + d + inv) or any(math.isinf(x) for x in o):
        return [0x7F800000, 0x3F800000, 0, 0]                          # invalid ray (docs/design.md)
    V = [[u2f(node[3 * s + c]) for c in range(3)] for s in VSLOTS[ntype]]
    if not all(math.isfinite(x) for x in d + [c for v in V for c in v]):
        return None
    r = lambda q: Fr(rtz(q))   # noqa: E731
    kz = _kz(d)
    kx, ky = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[kz]
    if d[kz] < 0:
        kx, ky = ky, kx
    s = -1 if d[kz] < 0 else 1
    adz, px, py = abs(Fr(d[kz])), s * Fr(d[kx]), s * Fr(d[ky])
    rel = [[r(Fr(V[j][c]) - Fr(o[c])) for c in range(3)] for j in range(3)]
    sh = [(r(r(q[kx] * adz) - r(px * q[kz])), r(r(q[ky] * adz) - r(py * q[kz])), q[kz]) for q in rel]
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = sh
    E = [r(r(cx * by) - r(cy * bx)), r(r(ax * cy) - r(ay * cx)), r(r(bx * ay) - r(by * ax))]
    kdet = r(r(r(adz * E[0]) + r(adz * E[1])) + r(adz * E[2]))
    T = r(s * r(r(r(E[0] * az) + r(E[1] * bz)) + r(E[2] * cz)))
    byte = (node[15] >> (8 * ntype)) & 0xFF
    pick = (E[0], E[1], E[2], E[0])
    out = [0 if x == 0 else f2u(float(x)) for x in (T, kdet, r(adz * pick[byte & 3]), r(adz * pick[(byte >> 2) & 3]))]
    hit = False
    if kdet != 0:
        sg = 1 if kdet > 0 else -1

        def passes(x, p, q):
            if x != 0:
                return x * sg > 0
            tx, ty = r(sh[q][0] - sh[p][0]), r(sh[q][1] - sh[p][1])
            return tx < 0 if ty == 0 else sg * ty > 0
        hit = passes(E[0], 1, 2) and passes(E[1], 2, 0) and passes(E[2], 0, 1) and sg * T >= 0
    if not hit:
        out[0] = 0x7F800000 | (out[1] & 0x80000000)
    return out


def tri_hit_exact(e):
    if e is None or e["det"] == 0:
        return False
    sg = 1 if e["det"] > 0 else -1
    return not any(x * sg < 0 for x in (e["U"], e["V"], e["W"])) and e["T"] * sg >= 0


def tri_sensitive(e):
    if e is None:
        return True
    if any(abs(x) <= K_TRI * U24 * sc for x, sc in zip((e["U"], e["V"], e["W"]), e["scale"])):
        return True
    return abs(e["det"]) <= K_TRI * U24 * sum(e["scale"]) or abs(e["T"]) <= K_TRI * U24 * e["T_scale"]


def words_hit(w):
    return (w[0] & 0x7F800000) != 0x7F800000


def quot(w, i):
    a, b = u2f(w[i]), u2f(w[1])
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return None
    return Fr(a) / Fr(b)


# ---------------------------------------------------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------------------------------------------------
def classify_tri(A, B, node, ntype, ray):
    if isinstance(B, str):
        return "T_UNDEFINED"
    if list(A) == list(B):
        return "T_EXACT"
    ha, hb = words_hit(A), words_hit(B)
    if ha and hb:
        qa, qb = [quot(A, i) for i in (0, 2, 3)], [quot(B, i) for i in (0, 2, 3)]
        if None in qa or None in qb:
            return "T_QUOT_FAR"
        if [f2u(float(x)) for x in qa] == [f2u(float(x)) for x in qb]:
            return "T_SCALE"
        if ulps_apart(qa[0], qb[0]) <= N_T and all(abs(x - y) <= Fr(1, 1 << 16) for x, y in zip(qa[1:], qb[1:])):
            return "T_QUOT_ULP"
        return "T_QUOT_FAR"
    if not ha and not hb:
        return "T_MISS_WORDS"
    rne = tri_eval(node, ntype, ray, "rne")
    if rne is not None and 0 in (rne["U"], rne["V"], rne["W"]):
        return "T_HITMISS_TIE"
    return "T_HITMISS_ROUND" if tri_sensitive(tri_eval(node, ntype, ray)) else "T_HITMISS_RULE"


def outside(A, node):
    hits = [w for w in A if w != MISS]
    return any(w not in node[:4] for w in hits) or len(set(hits)) != len(hits)


def classify_box(A, B, node, ntype, ray):
    if outside(A, node):
        return "B_OUTSIDE"
    if isinstance(B, str):
        return "B_UNDEFINED"
    if list(A) == list(B):
        return "B_EXACT"
    ids = node[:4]
    sa, sb = {w for w in A if w != MISS}, {w for w in B if w != MISS}
    ev, child, pair = box_margins(node, ntype, ray)
    rne, rz = box_eval(node, ntype, ray, "rne"), box_eval(node, ntype, ray, "rtz")
    if sa == sb:
        oa, ob = [ids.index(w) for w in A if w != MISS], [ids.index(w) for w in B if w != MISS]
        inv = [(x, y) for i, x in enumerate(oa) for y in oa[i + 1:] if ob.index(x) > ob.index(y)]
        if not inv:
            return "B_LAYOUT"
        if all(any(m["t0"][x] == m["t0"][y] for m in (ev, rne, rz)) for x, y in inv):
            return "B_ORDER_TIE"
        if all(pair.get((min(x, y), max(x, y)), K_BOX + 1) <= K_BOX for x, y in inv):
            return "B_ORDER_ROUND"
        return "B_ORDER_RULE"
    diff = [ids.index(w) for w in sa ^ sb if w in ids]

    def eq(m, i):
        return m["t0"][i] is not None and (m["t0"][i] == m["E"] or m["t0"][i] == m["tf"][i])
    if diff and all(eq(rne, i) or eq(rz, i) for i in diff):
        return "B_SET_EQUAL"
    if diff and all(child[i] is not None and child[i] <= K_BOX for i in diff):
        return "B_SET_ROUND"
    return "B_SET_RULE"


def classify(A, B, node, ntype, ray):
    return classify_tri(A, B, node, ntype, ray) if ntype <= 3 else classify_box(A, B, node, ntype, ray)


# ---------------------------------------------------------------------------------------------------------------------
# structures and sampling
# ---------------------------------------------------------------------------------------------------------------------
def runtime_root(dehydrated, header_offset=256):
    """The first node of the last level-table entry of a dehydrated stream (its header keeps the dehydrated root)."""
    src = dehydrated[header_offset:]
    hdr = R.Header(src, 0)
    b32s, b32n, b16s, _ = R.levels(src, hdr)[-1]
    return ((hdr.offset_to_nodes + (b32s if b32n else b16s)) >> 6) << 3 | (5 if b32n else 4)


def walk(buf, root):
    """Every node reachable from root: dict(ptr, type, off, parent, slot)."""
    seen, out, stack = set(), [], [(root, None, None)]
    while stack:
        p, parent, slot = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        t, off = p & 7, (p >> 3) << 6
        if t > 5 or off + (128 if t == 5 else 64) > len(buf):
            raise ValueError(f"node type {t} or offset out of range: only types 0..5 inside the structure are walked")
        out.append(dict(ptr=p, type=t, off=off, parent=parent, slot=slot))
        if t in (4, 5):
            for s, k in enumerate(struct.unpack_from("<4I", buf, off)):
                if k != INVALID:
                    stack.append((k, p, s))
    return out


def words(buf, off, n):
    return list(struct.unpack_from(f"<{n}I", buf, off))


def _unit(rng):
    while True:
        v = [rng.gauss(0, 1) for _ in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        if n > 1e-9:
            return [x / n for x in v]


def _exit(p, w, lo, hi):
    t = INF
    for c in range(3):
        if w[c] > 0:
            t = min(t, (hi[c] - p[c]) / w[c])
        elif w[c] < 0:
            t = min(t, (lo[c] - p[c]) / w[c])
    return max(0.0, t if t != INF else 0.0)


def _inv(d):
    return [(INF if math.copysign(1, c) > 0 else -INF) if c == 0 else f32(1.0 / c) for c in d]


def _ray(target, w, dist, exact, t32):
    """[tmax, o, d, inv] words of a ray from target + w*dist toward target; exact: d = T - O exactly in float32."""
    O = [f32(target[c] + w[c] * dist) for c in range(3)]
    if exact and t32 is not None:
        for _ in range(6):
            D = [f32(t32[c] - O[c]) for c in range(3)]
            if all(Fr(t32[c]) - Fr(O[c]) == Fr(D[c]) for c in range(3)) and any(D):
                return [f2u(f32(4 * dist))] + [f2u(x) for x in O + D + _inv(D)]
            O = [f32(t32[c] - D[c]) for c in range(3)]
    v = [target[c] - O[c] for c in range(3)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    d = [f32(x / n) for x in v]
    return [f2u(f32(4 * dist))] + [f2u(x) for x in O + d + _inv(d)]


def _pick(rng, classes):
    x = rng.uniform(0, sum(w for _, w in classes))
    for n, w in classes:
        x -= w
        if x <= 0:
            return n
    return classes[-1][0]


def _finite_box(lo, hi, pts):
    flo = [lo[c] if math.isfinite(lo[c]) else min(p[c] for p in pts) - 1.0 for c in range(3)]
    fhi = [hi[c] if math.isfinite(hi[c]) else max(p[c] for p in pts) + 1.0 for c in range(3)]
    return flo, fhi


def tri_case(rng, buf, e, cls):
    """(node words, ray words, aimed triangle) for a paired leaf and one ray class."""
    node = words(buf, e["off"], 16)
    V = [[u2f(node[3 * s + c]) for c in range(3)] for s in range(4)]
    pt = buf_slot_box(buf, e["parent"], e["slot"])
    plo, phi = _finite_box(pt[0], pt[1], V)
    tri = 0 if cls == "INT0" else 1 if cls == "INT1" else rng.randrange(2)
    t0, t1 = (V[0], V[1], V[2]), (V[1], V[3], V[2])
    t32 = None
    if cls in ("INT0", "INT1"):
        a, b = rng.uniform(0.05, 0.45), rng.uniform(0.05, 0.45)
        tv = t0 if tri == 0 else t1
        target = [tv[0][c] + a * (tv[1][c] - tv[0][c]) + b * (tv[2][c] - tv[0][c]) for c in range(3)]
    elif cls == "SHARED_EDGE":
        u = rng.uniform(0.02, 0.98)
        target = [V[1][c] + u * (V[2][c] - V[1][c]) for c in range(3)]
        t32 = [f32(x) for x in target]
    elif cls == "SHARED_VERTEX":
        target = t32 = list(V[rng.choice((1, 2))])
    elif cls == "OUTER_VERTEX":
        target = t32 = list(V[0] if tri == 0 else V[3])
    else:                                              # OUTER_EDGE, OFF_EDGE
        a, b, opp = ((0, 1, 2), (2, 0, 1))[rng.randrange(2)] if tri == 0 else ((1, 3, 2), (3, 2, 1))[rng.randrange(2)]
        u = rng.uniform(0.1, 0.9)
        target = [V[a][c] + u * (V[b][c] - V[a][c]) for c in range(3)]
        if cls == "OFF_EDGE":
            target = [target[c] + 1e-3 * (target[c] - V[opp][c]) for c in range(3)]
        else:
            t32 = [f32(x) for x in target]
    w = _unit(rng)
    diag = math.sqrt(sum((phi[c] - plo[c]) ** 2 for c in range(3))) or 1.0
    if rng.random() < 2 / 3:
        dist = _exit(target, w, plo, phi) + diag * rng.uniform(0.1, 2.0)
    else:
        dist = diag * math.exp(rng.uniform(math.log(1e-3), math.log(0.5)))
    return node, _ray(target, w, dist, t32 is not None and rng.random() < 0.5, t32), tri


def buf_slot_box(buf, parent, slot):
    t, off = parent & 7, (parent >> 3) << 6
    return boxes(words(buf, off, 32 if t == 5 else 16), t)[slot]


def box_case(rng, buf, e, cls):
    t = e["type"]
    node = words(buf, e["off"], 32 if t == 5 else 16)
    bx = boxes(node, t)
    valid = [i for i in range(4) if node[i] != INVALID and all(bx[i][0][c] <= bx[i][1][c] for c in range(3))
             and all(math.isfinite(x) for x in bx[i][0] + bx[i][1])]
    if not valid:
        return None
    blo = [min(bx[i][0][c] for i in valid) for c in range(3)]
    bhi = [max(bx[i][1][c] for i in valid) for c in range(3)]
    D = math.sqrt(sum((bhi[c] - blo[c]) ** 2 for c in range(3))) or 1.0
    lo, hi = bx[rng.choice(valid)]
    t32 = None
    if cls == "INSIDE":
        o = [rng.uniform(lo[c], hi[c]) for c in range(3)]
        d = [f32(x) for x in _unit(rng)]
        return node, [f2u(f32(4 * D))] + [f2u(f32(x)) for x in o] + [f2u(x) for x in d + _inv(d)]
    if cls == "CENTER":
        target = [(lo[c] + hi[c]) / 2 for c in range(3)]
    elif cls == "CORNER":
        target = t32 = [lo[c] if rng.random() < 0.5 else hi[c] for c in range(3)]
    elif cls == "MISS":
        target = [bhi[c] + 0.25 * (bhi[c] - blo[c]) + 0.1 for c in range(3)]
    else:
        target = [rng.uniform(lo[c], hi[c]) for c in range(3)]
    if cls == "AXIS":
        a, sg = rng.randrange(3), rng.choice((-1.0, 1.0))
        w = [0.0] * 3
        w[a] = sg
        dist = _exit(target, w, blo, bhi) + D * math.exp(rng.uniform(math.log(0.05), math.log(20)))
        d = [0.0] * 3
        d[a] = -sg
        O = [f32(target[c] + w[c] * dist) for c in range(3)]
        return node, [f2u(f32(4 * dist))] + [f2u(x) for x in O + d + _inv(d)]
    w = _unit(rng)
    dist = _exit(target, w, blo, bhi) + D * math.exp(rng.uniform(math.log(0.05), math.log(20)))
    ray = _ray(target, w, dist, t32 is not None, t32)
    if cls == "EXTENT":
        ev = box_eval(node, t, ray, rng.choice(("rne", "rtz")))
        ts = sorted({float(x) for x, h in zip(ev["t0"], ev["hit"]) if h})
        if ts:
            ray = [f2u(f32(rng.choice(ts)))] + ray[1:]
    return node, ray


def sample(structures, roots, n_tri_kernels, n_box_kernels, seed=1):
    """Kernel tables: [{"label", "kind" tri|b32|b16, "cases": [{node, type, ray, cls, pair}]}]."""
    rng = random.Random(seed)
    pools = {"pair": [], 5: [], 4: []}
    for si, (buf, root) in enumerate(zip(structures, roots)):
        for e in walk(buf, root):
            e["s"] = si
            if e["type"] == 0 and e["parent"] is not None:
                w = words(buf, e["off"], 16)
                if w[14] != INVALID and all(math.isfinite(u2f(x)) for x in w[:12]):
                    pools["pair"].append(e)
            elif e["type"] in (4, 5):
                pools[e["type"]].append(e)

    def draw(pool):
        by = {}
        for e in pool:
            by.setdefault(e["s"], []).append(e)
        keys = sorted(by)
        return by[rng.choices(keys, [math.sqrt(len(by[k])) for k in keys])[0]]

    out = []
    for k in range(n_tri_kernels if pools["pair"] else 0):
        cases = [None] * 32
        for p in range(16):
            e = rng.choice(draw(pools["pair"]))
            cls = _pick(rng, TRI_CLASSES)
            node, ray, _ = tri_case(rng, structures[e["s"]], e, cls)
            cases[p] = dict(node=node, type=0, ray=ray, cls=cls, pair=p)
            cases[16 + p] = dict(node=node, type=1, ray=ray, cls=cls, pair=p)
        out.append(dict(label=f"bhd_tri_k{k + 1:02d}", kind="tri", cases=cases))
    for t, kind in ((5, "b32"), (4, "b16")):
        for k in range(n_box_kernels if pools[t] else 0):
            cases = []
            while len(cases) < 14:
                e = rng.choice(draw(pools[t]))
                cls = _pick(rng, BOX_CLASSES)
                r = box_case(rng, structures[e["s"]], e, cls)
                if r:
                    node, ray = r
                    cases.append(dict(node=SURROGATE + node[4:], type=t, ray=ray, cls=cls))
            out.append(dict(label=f"bhd_{kind}_k{k + 1:02d}", kind=kind, cases=cases))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# kernels (tools/bvh_tri_probe.py layout) and the translator
# ---------------------------------------------------------------------------------------------------------------------
def kernel_of(tab):
    """A tools/bvh_tri_probe.Kernel (triangles) or its box variant: 14 two-slot node areas, 32-bit form only, the
    four child ids naming node-region slots 8..11 (inside the buffer)."""
    if P is None:
        raise RuntimeError("tools/bvh_tri_probe.py (PR #4202) is needed for kernels")
    if tab["kind"] == "tri":
        k = P.Kernel(tab["label"])
        for c in tab["cases"]:
            if c["type"] not in (0, 1):
                raise ValueError("triangle kernels run types 0 and 1 only")
            k.add_case(c["cls"], k.add_node(c["node"]), c["type"], c["ray"])
        return k
    return BoxKernel(tab)


if P is not None:
    class BoxKernel(P.Kernel):
        def __init__(self, tab):
            super().__init__(tab["label"])
            for i, c in enumerate(tab["cases"]):
                if c["type"] not in (4, 5) or c["node"][:4] != SURROGATE:
                    raise ValueError("box kernels run types 4 and 5 with the in-buffer child ids only")
                ws = list(c["node"]) + [0] * (32 - len(c["node"]))
                self.slots[2 * i] = [w & 0xFFFFFFFF for w in ws]
                self.cases.append(dict(label=c["cls"], slot=2 * i, type=c["type"], ray=list(c["ray"]), meta={}))
            for ch in SURROGATE:                       # every child id names memory inside the 8 KiB buffer
                if not self.L["node"] + (ch & ~7) * 8 + 128 <= 4 * self.L["words"]:
                    raise ValueError("child id outside the buffer")

        def _case_lines(self, i, c):
            out = P.Kernel._case_lines(self, i, c)
            j = next(n for n, ln in enumerate(out) if ln.startswith("s_add_u32 s12"))
            k = next(n for n, ln in enumerate(out) if ln.startswith("image_bvh64"))
            return out[:j] + out[k + 1:]               # the 32-bit form only

        def check(self):
            lines = self.lines()
            assert sum(ln.startswith("image_bvh_") for ln in lines) == len(self.cases)
            assert not any(ln.startswith("image_bvh64") for ln in lines)
            for ln in lines:
                if ln.startswith("v_mov_b32 v2,") and int(ln.split(", ")[1], 16) & 7 not in (4, 5):
                    raise ValueError("box node type")
            return True


def translate(tables, tree):
    """Translator words per kernel and case (32-bit form, descriptor base 0x1000), or 'undefined: ...'."""
    os.environ["LEONIDA_TOOLS_DIR"] = os.path.join(tree, "tools")
    sys.path.insert(0, os.path.join(tree, "tests", "semantic"))
    from support import mrt0_words, translate_lines
    out = {}
    for tab in tables:
        mem, slot_of = {}, []
        for i, c in enumerate(tab["cases"]):
            s = i if tab["kind"] == "tri" else 2 * i
            slot_of.append(s)
            for j, w in enumerate(c["node"]):
                mem[0x1000 + 64 * s + 4 * j] = w
        res = []
        for c, s in zip(tab["cases"], slot_of):
            lines = [f"v_mov_b32_e32 v2, 0x{(64 * s) >> 3 | c['type']:x}"]
            lines += [f"v_mov_b32_e32 v{3 + i}, 0x{w:x}" for i, w in enumerate(c["ray"])]
            lines += ["s_mov_b32 s4, 0x10", "s_mov_b32 s5, 0x80000000", "s_mov_b32 s6, 0xffffffff",
                      "s_mov_b32 s7, 0x81000000",
                      "image_bvh_intersect_ray v[20:23], [" + ", ".join(f"v{r}" for r in range(2, 13)) + "], s[4:7]",
                      "exp mrt0 v20, v21, v22, v23 done vm", "s_endpgm"]
            try:
                res.append(mrt0_words(translate_lines("ps_bvh_hw_diff", lines), memory=mem))
            except Exception as e:   # noqa: BLE001  an interpreter refusal (undefined SPIR-V) is a result too
                res.append(f"undefined: {type(e).__name__}")
        out[tab["label"]] = res
    return out


def score(tables, hw, other):
    """Counts {(kind, class): n} and stop conditions, hardware (hw) as the reference."""
    counts, stops = {}, []
    for tab in tables:
        if tab["label"] not in hw:
            continue
        for i, c in enumerate(tab["cases"]):
            r = hw[tab["label"]][i]
            if c["type"] <= 3 and r["hw32"] != r["hw64"]:
                stops.append(f"{tab['label']} case {i}: the two forms differ")
            k = classify(r["hw32"], other[tab["label"]][i], c["node"], c["type"], c["ray"])
            if k == "B_OUTSIDE":
                stops.append(f"{tab['label']} case {i}: a word outside the node's child ids")
            key = ("tri" if c["type"] <= 3 else "box16" if c["type"] == 4 else "box32", k)
            counts[key] = counts.get(key, 0) + 1
            m = hw_model_words(c["node"], c["type"], c["ray"])
            key = (key[0], "RTZ model " + ("equal" if m == r["hw32"] else "differs"))
            counts[key] = counts.get(key, 0) + 1
    return counts, stops


# ---------------------------------------------------------------------------------------------------------------------
def self_test():
    """Synthetic structures (tools/xdxr_bvh.py builds paired TRI0/TRI1 leaves under BOX32 or BOX16 nodes); no GPU."""
    import xdxr_bvh as X
    rng = random.Random(7)
    grid = {(x, y): (1000.3 + x + rng.uniform(-.1, .1), 2000.7 + y + rng.uniform(-.1, .1), 0.5 * math.sin(x + y))
            for x in range(7) for y in range(5)}                  # invented, non-dyadic, shared vertices
    tris = [[grid[p] for p in quad] for i in range(24)
            for quad in [[(i % 6, i // 6), (i % 6 + 1, i // 6), (i % 6, i // 6 + 1)],
                         [(i % 6 + 1, i // 6), (i % 6 + 1, i // 6 + 1), (i % 6, i // 6 + 1)]]]
    bufs, roots = [], []
    for box16 in (False, True):
        buf, info = X.build_blas([[list(map(f32, v)) for v in t] for t in tris], box16=box16, pair=True)
        bufs.append(buf)
        roots.append(info["root"])
    checks = []
    a, b = sample(bufs, roots, 1, 1, seed=3), sample(bufs, roots, 1, 1, seed=3)
    checks.append(("sampling is deterministic", a == b))
    kinds = sorted(t["kind"] for t in a)
    checks.append(("one kernel per kind (tri, b32, b16)", kinds == ["b16", "b32", "tri"]))
    checks.append(("box child ids are the in-buffer surrogates",
                   all(c["node"][:4] == SURROGATE for t in a if t["kind"] != "tri" for c in t["cases"])))
    checks.append(("types 0, 1, 4, 5 only", {c["type"] for t in a for c in t["cases"]} <= {0, 1, 4, 5}))
    tri = next(t for t in a if t["kind"] == "tri")
    checks.append(("each paired-leaf ray runs on TRI0 and TRI1",
                   all(tri["cases"][p]["ray"] == tri["cases"][16 + p]["ray"] and tri["cases"][p]["node"] ==
                       tri["cases"][16 + p]["node"] for p in range(16))))
    # classes on hand-made words
    hit = [f2u(-8.0), f2u(-1.0), f2u(-0.25), f2u(-0.125)]
    hit2 = [f2u(-16.0), f2u(-2.0), f2u(-0.5), f2u(-0.25)]
    node = [0] * 16
    checks.append(("T_EXACT / T_SCALE / T_MISS_WORDS / T_UNDEFINED",
                   [classify_tri(hit, hit, node, 0, [0] * 10), classify_tri(hit, hit2, node, 0, [0] * 10),
                    classify_tri([0x7F800000, 0x3F800000, 0, 0], [0xFF800000, 0xBF800000, 1, 2], node, 0, [0] * 10),
                    classify_tri(hit, "undefined: x", node, 0, [0] * 10)] ==
                   ["T_EXACT", "T_SCALE", "T_MISS_WORDS", "T_UNDEFINED"]))
    # a box node with two children sharing the entry plane z = 1 and one child entered at z = 3, ray +z
    bn = SURROGATE + [f2u(x) for x in (0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 1, 3, 0, 0, 3, 1, 1, 4)] + \
        [f2u(FMAX)] * 3 + [f2u(-FMAX)] * 3 + [0] * 4
    ray = [f2u(100.0)] + [f2u(x) for x in (0.5, 0.5, 0.0, 0.0, 0.0, 1.0)] + [f2u(INF), f2u(INF), f2u(1.0)]
    M = box_eval(bn, 5, ray, "rne")["ids"]
    checks.append(("box model: entry order 1, 1, 3", M == [0x40, 0x48, 0x50, MISS]))
    tie = [M[1], M[0], M[2], M[3]]
    far = [M[0], M[2], M[1], M[3]]
    drop = [M[0], M[1], MISS, MISS]
    ray_e = [f2u(3.0)] + ray[1:]
    checks.append(("B_ORDER_TIE / B_ORDER_RULE / B_SET_RULE / B_SET_EQUAL / B_OUTSIDE",
                   [classify_box(tie, M, bn, 5, ray), classify_box(far, M, bn, 5, ray),
                    classify_box(drop, M, bn, 5, ray), classify_box(drop, M, bn, 5, ray_e),
                    classify_box([0x1234, MISS, MISS, MISS], M, bn, 5, ray)] ==
                   ["B_ORDER_TIE", "B_ORDER_RULE", "B_SET_RULE", "B_SET_EQUAL", "B_OUTSIDE"]))
    checks.append(("round toward zero: 1/3 truncates", rtz(Fr(1, 3)) == u2f(0x3EAAAAAA) and f32(1 / 3) == u2f(0x3EAAAAAB)))
    if P is not None:
        ok = True
        for t in a:
            k = kernel_of(t)
            P.check_kernel(k) if t["kind"] == "tri" else k.check()
            ok &= len(k.lines()) > 100
        checks.append(("kernels build and pass the probe's checks (tools/bvh_tri_probe.py present)", ok))
    tr = translate(a[:1], ROOT)
    checks.append(("translator runs on every case", all(isinstance(w, list) for w in tr[a[0]["label"]])))
    fake = {a[0]["label"]: [dict(hw32=w, hw64=w) for w in tr[a[0]["label"]]]}
    counts, stops = score(a[:1], fake, tr)
    checks.append(("translator against itself: every row exact, no stop",
                   not stops and all(k[1] in ("T_EXACT", "B_EXACT") for k in counts if "model" not in k[1])))
    ok = all(v for _, v in checks)
    for n, v in checks:
        print(f"{'PASS' if v else 'FAIL'}  {n}")
    print(f"self-test {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _load(work):
    with open(os.path.join(work, "cases.json"), encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--self-test"]:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("sample", "translate", "run", "score"))
    ap.add_argument("structures", nargs="*")
    ap.add_argument("--work", required=True, help="the user's private folder; node words are written only here")
    ap.add_argument("--dehydrated", nargs="*", default=[], help="the dehydrated streams, to find the runtime roots")
    ap.add_argument("--header-offset", type=int, default=256)
    ap.add_argument("--root", default="", help="runtime root pointers, comma-separated, instead of --dehydrated")
    ap.add_argument("--tri-kernels", type=int, default=4)
    ap.add_argument("--box-kernels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--translator", default=ROOT, help="a checkout holding tools/ and tests/semantic/")
    ap.add_argument("--name", default="translator")
    ap.add_argument("--against", default="translator")
    ap.add_argument("--kernel", action="append", default=[])
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    try:
        if a.cmd == "sample":
            bufs = [open(p, "rb").read() for p in a.structures]
            if a.root:
                roots = [int(x, 0) for x in a.root.split(",")]
            else:
                roots = [runtime_root(open(p, "rb").read(), a.header_offset) for p in a.dehydrated]
            if len(roots) != len(bufs):
                print("error: one runtime root (or dehydrated stream) per structure", file=sys.stderr)
                return 2
            tabs = sample(bufs, roots, a.tri_kernels, a.box_kernels, a.seed)
            with open(os.path.join(a.work, "cases.json"), "w", encoding="utf-8") as f:
                json.dump(tabs, f)
            n = sum(len(t["cases"]) for t in tabs)
            print(f"{len(tabs)} kernels, {n} cases -> {a.work}")
            if P is not None:
                os.makedirs(os.path.join(a.work, "kernels"), exist_ok=True)
                for t in tabs:
                    with open(os.path.join(a.work, "kernels", t["label"] + ".s"), "w", newline="\n") as f:
                        f.write("\n".join(kernel_of(t).lines()) + "\n")
            return 0
        tabs = _load(a.work)
        if a.cmd == "translate":
            out = translate(tabs, os.path.abspath(a.translator))
            with open(os.path.join(a.work, f"tr_{a.name}.json"), "w", encoding="utf-8") as f:
                json.dump(out, f)
            print(f"{sum(len(v) for v in out.values())} cases translated -> tr_{a.name}.json")
            return 0
        if a.cmd == "run":
            if P is None:
                print("tools/bvh_tri_probe.py (PR #4202) is needed to run kernels", file=sys.stderr)
                return 77
            sys.path.insert(0, HERE)
            from hw_oracle import asm, runner
            try:
                oracle = runner.open_oracle()
            except runner.Unavailable as e:
                print(f"hardware oracle unavailable: {e}", file=sys.stderr)
                return 77
            p = os.path.join(a.work, "hw.json")
            hw = json.load(open(p)) if os.path.isfile(p) else {}
            for t in tabs:
                if (a.kernel and t["label"] not in a.kernel) or t["label"] in hw:
                    continue
                k = kernel_of(t)
                P.check_kernel(k) if t["kind"] == "tri" else k.check()
                output = oracle.run([k.program(asm)], words=k.L["words"])[0]   # one dispatch
                why = P.output_problem(k, output)
                if why:
                    print(f"STOP {t['label']}: {why}", file=sys.stderr)
                    return 1
                hw[t["label"]] = P.results(k, output)[0]
                with open(p, "w") as f:
                    json.dump(hw, f)
            return 0
        hw = json.load(open(os.path.join(a.work, "hw.json")))
        other = json.load(open(os.path.join(a.work, f"tr_{a.against}.json")))
        counts, stops = score(tabs, hw, other)
        for (kind, k), n in sorted(counts.items()):
            print(f"{kind:6} {k:16} {n}")
        for s in stops:
            print("STOP " + s)
        return 1 if stops else 0
    except (OSError, ValueError, KeyError, struct.error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
