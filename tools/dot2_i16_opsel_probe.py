r"""Measure RDNA2 VOP3P op_sel, op_sel_hi, and neg_lo/neg_hi on v_dot2_i32_i16 and v_dot2_u32_u16. Standard library only.

Probes all 16 combinations of ``op_sel[0..1]`` and ``op_sel_hi[0..1]`` with and without ``clamp`` across curated
edge vectors (including ``(-32768)^2 + (-32768)^2 + (-1) == 0x7FFFFFFF`` cancellation), source kinds (VGPR, SGPR
in S0/S1/S2, inline constants, 32-bit literal), omitted default modifier forms, and characterization controls on
asymmetric ``S2`` (non-default ``op_sel[2]``/``op_sel_hi[2]`` and set ``neg_lo``/``neg_hi`` bits) across wave32 and
wave64 on GFX10 (``gfx1036``).

Usage:
    python -I tools/dot2_i16_opsel_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/dot2_i16_opsel_probe.py --self-test

Exit 0 when all hardware results match the expected RDNA2 model with zero wave32/wave64 diffs; 1 on a mismatch or
wave parity failure; 2 on a harness/CLI error; 77 when the hardware oracle is unavailable or the GPU is not GFX10.
Nothing here reads game files.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OPS: tuple[str, ...] = ("v_dot2_i32_i16", "v_dot2_u32_u16")
MASK32 = 0xFFFFFFFF

GAME_MODE: dict[str, int] = {
    "ieee": 0,
    "dx10_clamp": 1,
    "denorm32": 0,
    "denorm16_64": 3,
    "round32": 0,
    "round16_64": 0,
}

# Curated edge triples (S0, S1, S2):
#   0: signed/unsigned boundary values across low/high halves
#   1: (-32768)^2 + (-32768)^2 + (-1) == 0x7FFFFFFF cancellation under op_sel:[0,1,0] op_sel_hi:[0,1,1] clamp
#   2: (-32768)^2 + (-32768)^2 == +2^31 overflow to 0x7FFFFFFF under clamp
#   3: unsigned overflow under clamp when matching 0xFFFF * 0xFFFF
#   4: signed negative underflow saturation to 0x80000000
#   5: asymmetric distinct halves
EDGE_VECS: tuple[tuple[int, int, int], ...] = (
    (0x80007FFF, 0xFFFF0002, 0x00000123),
    (0x00018000, 0x80000001, 0xFFFFFFFF),
    (0x00018000, 0x80000001, 0x00000000),
    (0xFFFF0001, 0x0001FFFF, 0x80000000),
    (0x80007FFF, 0x7FFF8000, 0x80000000),
    (0x1234ABCD, 0x5678EF01, 0x9ABCDEF0),
)


@dataclass(frozen=True)
class SrcCase:
    name: str
    s0_tok: str
    s1_tok: str
    s2_tok: str
    s0_val: int
    s1_val: int
    s2_val: int
    mod: str
    sel: tuple[int, int, int]
    selhi: tuple[int, int, int]
    setup: tuple[str, ...] = ()


SRC_CASES: tuple[SrcCase, ...] = (
    SrcCase(
        "omitted_opsel", "v1", "v2", "v3", 0x00030002, 0x00070005, 0x0000000A,
        "op_sel_hi:[0,1,1]", (0, 0, 0), (0, 1, 1),
    ),
    SrcCase(
        "omitted_opsel_hi", "v1", "v2", "v3", 0x00030002, 0x00070005, 0x0000000A,
        "op_sel:[1,0,0]", (1, 0, 0), (1, 1, 1),
    ),
    SrcCase(
        "explicit_default", "v1", "v2", "v3", 0x00030002, 0x00070005, 0x0000000A,
        "op_sel:[0,0,0] op_sel_hi:[1,1,1]", (0, 0, 0), (1, 1, 1),
    ),
    SrcCase(
        "neg_zero_noop", "v1", "v2", "v3", 0x00030002, 0x00070005, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[0,0,0] neg_hi:[0,0,0]", (1, 0, 0), (0, 1, 1),
    ),
    SrcCase(
        "inline_1_s0_hi", "1", "v2", "v3", 1, 0x00070005, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[1,1,1]", (1, 0, 0), (1, 1, 1),
    ),
    SrcCase(
        "inline_neg1_s0_hi", "-1", "v2", "v3", 0xFFFFFFFF, 0x00070005, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[1,1,1]", (1, 0, 0), (1, 1, 1),
    ),
    SrcCase(
        "inline_neg16_s0_hi", "-16", "v2", "v3", 0xFFFFFFF0, 0x00070005, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1),
    ),
    SrcCase(
        "inline_64_s1_hi", "v1", "64", "v3", 0x00030002, 64, 0x0000000A,
        "op_sel:[0,1,0] op_sel_hi:[1,1,1]", (0, 1, 0), (1, 1, 1),
    ),
    SrcCase(
        "literal_s0_sel", "0x00050003", "v2", "v3", 0x00050003, 0x00070002, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1),
    ),
    SrcCase(
        "sgpr_s0_sel", "s4", "v2", "v3", 0x00050003, 0x00070002, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1),
        setup=("s_mov_b32 s4, 0x00050003",),
    ),
    SrcCase(
        "sgpr_s1_sel", "v1", "s4", "v3", 0x00050003, 0x00070002, 0x0000000A,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1),
        setup=("s_mov_b32 s4, 0x00070002",),
    ),
    SrcCase(
        "sgpr_s2_sel", "v1", "v2", "s4", 0x00050003, 0x00070002, 0x12345678,
        "op_sel:[1,0,0] op_sel_hi:[0,1,1]", (1, 0, 0), (0, 1, 1),
        setup=("s_mov_b32 s4, 0x12345678",),
    ),
)


@dataclass(frozen=True)
class ControlCase:
    name: str
    mod: str
    s0: int = 0x00030002
    s1: int = 0x00070005
    s2: int = 0x87654321
    sel: tuple[int, int, int] = (0, 0, 0)
    selhi: tuple[int, int, int] = (1, 1, 1)
    neg_lo: tuple[int, int, int] = (0, 0, 0)
    neg_hi: tuple[int, int, int] = (0, 0, 0)


CONTROL_CASES: tuple[ControlCase, ...] = (
    ControlCase("s2_opsel_1", "op_sel:[0,0,1]", sel=(0, 0, 1)),
    ControlCase("s2_opsel_hi_0", "op_sel_hi:[1,1,0]", selhi=(1, 1, 0)),
    ControlCase("s2_opsel_both", "op_sel:[0,0,1] op_sel_hi:[1,1,0]", sel=(0, 0, 1), selhi=(1, 1, 0)),
    ControlCase("neg_lo_s0", "neg_lo:[1,0,0]", neg_lo=(1, 0, 0)),
    ControlCase("neg_hi_s0", "neg_hi:[1,0,0]", neg_hi=(1, 0, 0)),
    ControlCase("neg_lo_s1", "neg_lo:[0,1,0]", neg_lo=(0, 1, 0)),
    ControlCase("neg_hi_s1", "neg_hi:[0,1,0]", neg_hi=(0, 1, 0)),
    ControlCase("neg_lo_s2", "neg_lo:[0,0,1]", neg_lo=(0, 0, 1)),
    ControlCase("neg_hi_s2", "neg_hi:[0,0,1]", neg_hi=(0, 0, 1)),
    ControlCase("neg_lohi_s2", "neg_lo:[0,0,1] neg_hi:[0,0,1]", neg_lo=(0, 0, 1), neg_hi=(0, 0, 1)),
)


def _sext(val: int, bits: int) -> int:
    mask = (1 << bits) - 1
    v = val & mask
    return v - (1 << bits) if (v >> (bits - 1)) else v


def ref_dot2(
    op: str,
    s0: int,
    s1: int,
    s2: int,
    sel: tuple[int, int, int] = (0, 0, 0),
    selhi: tuple[int, int, int] = (1, 1, 1),
    clamp: bool = False,
) -> int:
    """Exact RDNA2 reference for v_dot2_i32_i16 / v_dot2_u32_u16 with VOP3P half-select."""
    if op not in OPS:
        raise ValueError(f"unknown op: {op}")
    signed = op == "v_dot2_i32_i16"

    def half(word: int, hi: int) -> int:
        raw = (word >> (16 if hi else 0)) & 0xFFFF
        return _sext(raw, 16) if signed else raw

    acc = _sext(s2, 32) if signed else (s2 & MASK32)
    total = half(s0, sel[0]) * half(s1, sel[1]) + half(s0, selhi[0]) * half(s1, selhi[1]) + acc
    if clamp:
        if signed:
            total = max(-0x80000000, min(0x7FFFFFFF, total))
        else:
            total = max(0, min(MASK32, total))
    return total & MASK32


def ref_dot2_neg(
    op: str,
    s0: int,
    s1: int,
    s2: int,
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    sel: tuple[int, int, int] = (0, 0, 0),
    selhi: tuple[int, int, int] = (1, 1, 1),
    clamp: bool = False,
) -> int:
    """Characterization model for RDNA2 neg_lo/neg_hi on v_dot2_i32_i16 / v_dot2_u32_u16.

    Measured on gfx1036: neg_lo[k] flips bit 15 and neg_hi[k] flips bit 31 of S0/S1; on S2, neg_hi[2] clears
    bit 31 before neg_lo[2] flips it, while op_sel[2]/op_sel_hi[2] are ignored.
    """
    s0 ^= (0x8000 if neg_lo[0] else 0) ^ (0x80000000 if neg_hi[0] else 0)
    s1 ^= (0x8000 if neg_lo[1] else 0) ^ (0x80000000 if neg_hi[1] else 0)
    s2 &= MASK32
    if neg_hi[2]:
        s2 &= 0x7FFFFFFF
    if neg_lo[2]:
        s2 ^= 0x80000000
    return ref_dot2(op, s0, s1, s2, sel=sel, selhi=selhi, clamp=clamp)


def grid_cases():
    """Yield (sel, selhi, clamp, vec_idx, s0, s1, s2) across all 16 S0/S1 half-select combinations."""
    for s0_lo, s1_lo, s0_hi, s1_hi in itertools.product((0, 1), repeat=4):
        sel = (s0_lo, s1_lo, 0)
        selhi = (s0_hi, s1_hi, 1)
        for clamp in (False, True):
            for vec_idx, (s0, s1, s2) in enumerate(EDGE_VECS):
                yield sel, selhi, clamp, vec_idx, s0, s1, s2


def _store_kernel(body_lines: list[str], words: int, wave: int):
    from hw_oracle import asm

    lines = list(body_lines) + ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return asm.Kernel(lines, wave=wave, threads=1, words=words, raw=True, mode=GAME_MODE)


def build_jobs(ops: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    kernels = []
    meta = []
    grid = list(grid_cases())

    for wave in waves:
        for op in ops:
            # 1. Exhaustive 16 (op_sel[0:2], op_sel_hi[0:2]) x clamp x EDGE_VECS
            g_lines = []
            g_cases = []
            for idx, (sel, selhi, clamp, vec_idx, s0, s1, s2) in enumerate(grid):
                mod = f"op_sel:[{sel[0]},{sel[1]},0] op_sel_hi:[{selhi[0]},{selhi[1]},1]" + (" clamp" if clamp else "")
                g_lines.extend([
                    f"v_mov_b32_e32 v1, {s0:#010x}",
                    f"v_mov_b32_e32 v2, {s1:#010x}",
                    f"v_mov_b32_e32 v3, {s2:#010x}",
                    f"{op} v8, v1, v2, v3 {mod}",
                    f"v_mov_b32_e32 v4, {idx * 4}",
                    "global_store_dword v4, v8, s[2:3]",
                ])
                g_cases.append((
                    f"vec{vec_idx} {mod}",
                    ref_dot2(op, s0, s1, s2, sel=sel, selhi=selhi, clamp=clamp),
                ))
            kernels.append(_store_kernel(g_lines, len(g_cases), wave))
            meta.append(("grid", wave, op, g_cases))

            # 2. Source kinds and omitted default modifiers
            s_lines = []
            s_cases = []
            for idx, sc in enumerate(SRC_CASES):
                s_lines.extend(sc.setup)
                if sc.s0_tok == "v1":
                    s_lines.append(f"v_mov_b32_e32 v1, {sc.s0_val:#010x}")
                if sc.s1_tok == "v2":
                    s_lines.append(f"v_mov_b32_e32 v2, {sc.s1_val:#010x}")
                if sc.s2_tok == "v3":
                    s_lines.append(f"v_mov_b32_e32 v3, {sc.s2_val:#010x}")
                s_lines.extend([
                    f"{op} v8, {sc.s0_tok}, {sc.s1_tok}, {sc.s2_tok} {sc.mod}",
                    f"v_mov_b32_e32 v4, {idx * 4}",
                    "global_store_dword v4, v8, s[2:3]",
                ])
                s_cases.append((
                    f"{sc.name} ({sc.mod})",
                    ref_dot2(op, sc.s0_val, sc.s1_val, sc.s2_val, sel=sc.sel, selhi=sc.selhi, clamp=False),
                ))
            kernels.append(_store_kernel(s_lines, len(s_cases), wave))
            meta.append(("sources", wave, op, s_cases))

            # 3. Characterization controls (S2 op_sel ignored; set neg_lo/neg_hi bit behavior)
            c_lines = []
            c_cases = []
            for idx, cc in enumerate(CONTROL_CASES):
                c_lines.extend([
                    f"v_mov_b32_e32 v1, {cc.s0:#010x}",
                    f"v_mov_b32_e32 v2, {cc.s1:#010x}",
                    f"v_mov_b32_e32 v3, {cc.s2:#010x}",
                    f"{op} v8, v1, v2, v3 {cc.mod}",
                    f"v_mov_b32_e32 v4, {idx * 4}",
                    "global_store_dword v4, v8, s[2:3]",
                ])
                c_cases.append((
                    f"{cc.name} ({cc.mod})",
                    ref_dot2_neg(
                        op, cc.s0, cc.s1, cc.s2,
                        neg_lo=cc.neg_lo, neg_hi=cc.neg_hi,
                        sel=cc.sel, selhi=cc.selhi, clamp=False,
                    ),
                ))
            kernels.append(_store_kernel(c_lines, len(c_cases), wave))
            meta.append(("controls", wave, op, c_cases))

    return kernels, meta


def evaluate_results(ops: list[str], waves: list[int], meta, raw_results) -> dict:
    """Compare hardware words against ref_dot2/ref_dot2_neg and check wave32/wave64 parity."""
    counts = {"grid": 0, "sources": 0, "controls": 0}
    mismatches = 0
    harness_errors = []
    first_mismatches = []
    per_wave = {}

    for (section, wave, op, cases), got in zip(meta, raw_results):
        if not isinstance(got, list):
            harness_errors.append({"section": section, "wave": wave, "op": op, "error": str(got)})
            continue
        if len(got) != len(cases):
            harness_errors.append({
                "section": section,
                "wave": wave,
                "op": op,
                "error": f"expected {len(cases)} words, got {len(got)}",
            })
            continue
        per_wave[(section, wave, op)] = got
        for (label, want), g in zip(cases, got):
            counts[section] += 1
            if g != want:
                mismatches += 1
                if len(first_mismatches) < 20:
                    first_mismatches.append({
                        "section": section,
                        "wave": wave,
                        "op": op,
                        "case": label,
                        "got": f"{g:#010x}",
                        "want": f"{want:#010x}",
                    })

    wave_diffs = 0
    first_wave_diffs = []
    if 32 in waves and 64 in waves:
        for section in ("grid", "sources", "controls"):
            for op in ops:
                w32 = per_wave.get((section, 32, op))
                w64 = per_wave.get((section, 64, op))
                if w32 is None or w64 is None:
                    continue
                for idx, (a, b) in enumerate(zip(w32, w64)):
                    if a != b:
                        wave_diffs += 1
                        if len(first_wave_diffs) < 20:
                            first_wave_diffs.append({
                                "section": section,
                                "op": op,
                                "index": idx,
                                "wave32": f"{a:#010x}",
                                "wave64": f"{b:#010x}",
                            })

    total_words = counts["grid"] + counts["sources"] + counts["controls"]
    return {
        "total_words": total_words,
        "counts": counts,
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "harness_errors": harness_errors,
        "first_mismatches": first_mismatches,
        "first_wave_diffs": first_wave_diffs,
    }


def write_report(path: str, doc: dict) -> bool:
    """Write JSON summary with LF line endings."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e}", file=sys.stderr)
        return False
    return True


def self_test() -> int:
    """Verify ref_dot2, ref_dot2_neg, grid/case shapes, and evaluation logic without hardware or clang."""
    # (-32768)^2 + (-32768)^2 + (-1) == 0x7FFFFFFF (no overflow); with s2=0 overflows to +2^31 -> clamp 0x7FFFFFFF
    assert ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0xFFFFFFFF,
                    sel=(0, 1, 0), selhi=(0, 1, 1), clamp=True) == 0x7FFFFFFF
    assert ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0,
                    sel=(0, 1, 0), selhi=(0, 1, 1), clamp=False) == 0x80000000
    assert ref_dot2("v_dot2_i32_i16", 0x00018000, 0x80000001, 0,
                    sel=(0, 1, 0), selhi=(0, 1, 1), clamp=True) == 0x7FFFFFFF
    # Unsigned vs signed extraction across swapped halves
    assert ref_dot2("v_dot2_i32_i16", 0x00030002, 0x00070005, 10,
                    sel=(1, 0, 0), selhi=(0, 1, 1), clamp=False) == (3 * 5 + 2 * 7 + 10)
    assert ref_dot2("v_dot2_u32_u16", 0x80000001, 0x00010002, 0,
                    sel=(1, 0, 0), selhi=(0, 1, 1), clamp=False) == (32768 * 2 + 1 * 1)
    assert ref_dot2("v_dot2_u32_u16", 0xFFFF0001, 0x0001FFFF, 0x80000000,
                    sel=(1, 0, 0), selhi=(0, 1, 1), clamp=True) == 0xFFFFFFFF
    # Characterization neg_lo / neg_hi behavior (plain: 2*5 + 3*7 = 31; neg_lo[0] flips bit 15 -> -32766 instead of +2)
    assert ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321, neg_lo=(1, 0, 0)) == 0x87654321 + 31 - 5 * 32768
    assert ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321, neg_hi=(0, 0, 1)) == 0x07654340
    assert ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321, neg_lo=(0, 0, 1)) == 0x07654340
    assert ref_dot2_neg("v_dot2_i32_i16", 0x00030002, 0x00070005, 0x87654321, neg_lo=(0, 0, 1), neg_hi=(0, 0, 1)) == 0x87654340

    grid = list(grid_cases())
    assert len(grid) == 16 * 2 * len(EDGE_VECS)

    # Synthetic evaluate_results round-trip without touching hw_oracle
    fake_meta = [("grid", 32, "v_dot2_i32_i16", [("c0", 11)]), ("sources", 32, "v_dot2_i32_i16", [("s0", 22)]),
                 ("controls", 32, "v_dot2_i32_i16", [("k0", 33)])]
    ev = evaluate_results(["v_dot2_i32_i16"], [32], fake_meta, [[11], [22], [33]])
    assert ev["total_words"] == 3 and ev["mismatches"] == 0 and ev["wave_diffs"] == 0 and not ev["harness_errors"]

    print(
        f"self-test ok ({len(grid)} grid cases/op/wave, {len(SRC_CASES)} source cases/op/wave, "
        f"{len(CONTROL_CASES)} control cases/op/wave)"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--waves", default="32,64", help="comma-separated wave sizes (32,64)")
    ap.add_argument("--ops", default=",".join(OPS), help="comma-separated opcodes")
    ap.add_argument("--hw-arch", help="target GFX10 GPU architecture (e.g. gfx1036)")
    ap.add_argument("--json", help="write measurement summary JSON here")
    ap.add_argument("--self-test", action="store_true", help="run CPU self-tests without hardware")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    wave_tokens = [w.strip() for w in a.waves.split(",") if w.strip()]
    bad_waves = [w for w in wave_tokens if w not in ("32", "64")]
    ops = [o.strip() for o in a.ops.split(",") if o.strip()]
    bad_ops = [o for o in ops if o not in OPS]
    if bad_waves or bad_ops or not wave_tokens or not ops:
        ap.error("unknown or empty: " + ", ".join(bad_waves + bad_ops or ["--waves/--ops"]))
    waves = [int(w) for w in wave_tokens]
    if a.json and os.path.isdir(a.json):
        ap.error(f"--json {a.json} is a folder, not a file")

    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10")
            return 77
        print(f"hardware oracle: {oracle.device} ({oracle.arch})")
        kernels, meta = build_jobs(ops, waves)
        raw_results = oracle.run(kernels)
        summary = evaluate_results(ops, waves, meta, raw_results)
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()

    report = {
        "device": str(oracle.device),
        "arch": oracle.arch,
        "waves": waves,
        "ops": ops,
        **summary,
    }
    if a.json and not write_report(a.json, report):
        return 2

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: {err['section']} wave{err['wave']} {err['op']}: {err['error']}")
        return 2

    c = summary["counts"]
    print(
        f"total_words={summary['total_words']} (grid={c['grid']}, sources={c['sources']}, "
        f"controls={c['controls']}) across {len(waves)} wave(s) x {len(ops)} op(s): "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}"
    )
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH {m['section']} wave{m['wave']} {m['op']} {m['case']}: got {m['got']}, want {m['want']}")
    for d in summary["first_wave_diffs"]:
        print(f"  WAVE DIFF {d['section']} {d['op']}[{d['index']}]: wave32={d['wave32']} wave64={d['wave64']}")

    return 1 if (summary["mismatches"] or summary["wave_diffs"]) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
