r"""Measure RDNA2 buffer_atomic_{and,or,xor,swap}_x2 straddling bounds on native hardware (SHADER-342). Standard library only.

docs/design.md records that buffer_atomic_{and,or,xor,swap}_x2 check bounds across the whole 8-byte element, all or
nothing, as for a 32_32 typed format, and marks how hardware treats a partly in-range 64-bit atomic as unmeasured.
This probe measures that behaviour directly on GFX10/RDNA2 through tools/hw_oracle/ in wave32 and wave64 across all
four ops, glc and non-glc forms, and all four V# OOB_SELECT modes (0..3).

One wrinkle measured alongside bounds: on gfx1036, buffer_atomic_*_x2 drops any access whose final byte address is not
8-byte aligned (just as 32-bit buffer_atomic_xor drops accesses not 4-byte aligned), leaving memory untouched and
returning (0, 0) with glc. Every straddling bounds case therefore keeps the final address 8-byte aligned so alignment
cannot mask bounds, while separate alignment-control cases confirm the alignment drop directly.

Each kernel launches one full wave (32 or 64 threads). Lane 0 seeds dwords 0..15 with distinct bit patterns, flushes L0
and GL1 caches, then only one active lane (lane 0 in wave32; lane 37 in the high half in wave64) runs the atomic with
vdata[0:1] = (DATA_LO, DATA_HI). Both the active lane's vdata and an inactive lane's vdata are written back after
the atomic alongside dwords 0..15.

Usage:
    python -I tools/buffer_atomic_x2_probe.py [--ops and,or,xor,swap] [--waves 32,64] [--json out.json]
    python -I tools/buffer_atomic_x2_probe.py --self-test

Exit 0 when all hardware results match the all-or-nothing 8-byte rule (with wave32 and wave64 agreeing bit-for-bit),
1 on a mismatch against that rule or a wave32/wave64 difference, 2 on a harness/CLI error, and 77 when the hardware
oracle is unavailable or the GPU is not GFX10. Nothing here reads game files.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OPS = ("and", "or", "xor", "swap")
OP_FN = {
    "and": lambda m, d: m & d,
    "or": lambda m, d: m | d,
    "xor": lambda m, d: m ^ d,
    "swap": lambda m, d: d,
}
DATA_LO = 0xF0F00F0F
DATA_HI = 0x0F0FF0F0
MEM_WORDS = 16
OUT_WORDS = 20


@dataclass(frozen=True)
class Case:
    name: str
    op: str
    glc: bool
    oob_select: int
    voff: int
    num_records: int
    stride: int = 0
    idx: int = 0
    idxen: bool = False
    base_add: int = 0
    soff: int = 0
    inst_off: int = 0
    group: str = "raw_straddle"


def mem_init(i):
    """Distinct non-zero 32-bit pattern per dword so and/or/xor/swap produce unique results."""
    return 0x12340000 | (i << 8) | 0x5A


def descriptor_word3(oob_select):
    """GFX10 buffer resource word3: DST_SEL=XYZW (0xFAC), FORMAT=22, RESOURCE_LEVEL=1, OOB_SELECT=(0..3)."""
    return 0x01016FAC | ((oob_select & 3) << 28)


def build_cases(ops=OPS):
    """Build straddling bounds, offset-term, OOB_SELECT 0..3, alignment, and 32-bit control cases."""
    cases = []
    rep_op = "xor" if "xor" in ops else ops[0]
    m12_ops = tuple(op for op in ("xor", "swap") if op in ops) or (ops[0],)

    # 1. Mode 3 (RAW), 8-byte aligned off=16 (bytes 16..23), byte-granular num_records across the boundary.
    for op in ops:
        for glc in (True, False):
            for nr in (16, 17, 19, 20, 21, 22, 23, 24):
                cases.append(Case(
                    name=f"raw_straddle:{op}:glc={int(glc)}:nr={nr}",
                    op=f"buffer_atomic_{op}_x2", glc=glc, oob_select=3,
                    voff=16, num_records=nr, group="raw_straddle",
                ))

    # 2. Mode 3 (RAW) with base+4 + off=12 (= base+16, 8-byte aligned), straddling num_records 16..20.
    for op in ops:
        for nr in (16, 19, 20):
            cases.append(Case(
                name=f"raw_base4:{op}:nr={nr}",
                op=f"buffer_atomic_{op}_x2", glc=True, oob_select=3,
                voff=12, num_records=nr, base_add=4, group="raw_base4",
            ))

    # 3. Mode 3 (RAW) confirming voff + soff + inst_offset all feed the bounds check.
    for voff, soff, inst_off in ((8, 8, 0), (8, 0, 8), (0, 8, 8), (12, 4, 0)):
        for nr in (23, 24):
            cases.append(Case(
                name=f"raw_terms:{rep_op}:v{voff}_s{soff}_i{inst_off}:nr={nr}",
                op=f"buffer_atomic_{rep_op}_x2", glc=True, oob_select=3,
                voff=voff, soff=soff, inst_off=inst_off, num_records=nr, group="raw_offset_terms",
            ))

    # 4. Mode 0 (STRUCTURED_WITH_OFFSET): idx < num_records and off + 8 <= stride (soff excluded from stride check).
    for op in ops:
        for st in (16, 20, 23, 24):
            cases.append(Case(
                name=f"mode0_straddle:{op}:idx0:st={st}",
                op=f"buffer_atomic_{op}_x2", glc=True, oob_select=0,
                voff=16, num_records=2, stride=st, idx=0, idxen=True, group="mode0_straddle",
            ))
        cases.append(Case(
            name=f"mode0_idx_oob:{op}:idx2",
            op=f"buffer_atomic_{op}_x2", glc=True, oob_select=0,
            voff=0, num_records=2, stride=24, idx=2, idxen=True, group="mode0_straddle",
        ))
    for st in (20, 24):
        cases.append(Case(
            name=f"mode0_soff:{rep_op}:st={st}",
            op=f"buffer_atomic_{rep_op}_x2", glc=True, oob_select=0,
            voff=16, soff=8, num_records=2, stride=st, idx=0, idxen=True, group="mode0_straddle",
        ))

    # 5. Mode 1 (STRUCTURED): idx < num_records only (stride ignored in bounds check).
    for op in m12_ops:
        for idx, voff in ((0, 16), (0, 24), (2, 16)):
            cases.append(Case(
                name=f"mode1:{op}:idx={idx}:off={voff}",
                op=f"buffer_atomic_{op}_x2", glc=True, oob_select=1,
                voff=voff, num_records=2, stride=24, idx=idx, idxen=True, group="mode1",
            ))

    # 6. Mode 2 (DISABLED): num_records != 0 only.
    for op in m12_ops:
        for nr, voff in ((1, 24), (0, 16)):
            cases.append(Case(
                name=f"mode2:{op}:nr={nr}:off={voff}",
                op=f"buffer_atomic_{op}_x2", glc=True, oob_select=2,
                voff=voff, num_records=nr, group="mode2",
            ))

    # 7. Alignment controls well inside num_records=64: addr & 7 != 0 drops _x2 on gfx1036.
    for voff in (16, 17, 18, 19, 20, 21, 22, 23, 24):
        cases.append(Case(
            name=f"align:{rep_op}:glc=1:b0:off={voff}",
            op=f"buffer_atomic_{rep_op}_x2", glc=True, oob_select=3,
            voff=voff, num_records=64, group="align_control",
        ))
    cases.append(Case(
        name=f"align:{rep_op}:glc=0:b0:off=20",
        op=f"buffer_atomic_{rep_op}_x2", glc=False, oob_select=3,
        voff=20, num_records=64, group="align_control",
    ))
    for voff in (0, 4):
        cases.append(Case(
            name=f"align:{rep_op}:glc=1:b4:off={voff}",
            op=f"buffer_atomic_{rep_op}_x2", glc=True, oob_select=3,
            voff=voff, num_records=64, base_add=4, group="align_control",
        ))

    # 8. 32-bit control: buffer_atomic_xor requires 4-byte alignment and 4-byte bounds.
    for nr in (16, 19, 20, 24):
        cases.append(Case(
            name=f"ctrl32:xor:off=16:nr={nr}",
            op="buffer_atomic_xor", glc=True, oob_select=3,
            voff=16, num_records=nr, group="ctrl32",
        ))
    for voff in (16, 17, 18, 19, 20):
        cases.append(Case(
            name=f"ctrl32:xor:nr=64:off={voff}",
            op="buffer_atomic_xor", glc=True, oob_select=3,
            voff=voff, num_records=64, group="ctrl32",
        ))

    return cases


def _in_bounds(case, offset_bytes, size_bytes):
    mode = case.oob_select
    if mode == 0:
        return (case.idx < case.num_records) and (offset_bytes + size_bytes <= case.stride)
    if mode == 1:
        return case.idx < case.num_records
    if mode == 2:
        return case.num_records != 0
    if mode == 3:
        return offset_bytes + size_bytes <= case.num_records
    raise ValueError(f"bad oob_select {mode}")


def expected_words(case, policy="all_or_nothing"):
    """Expected 20 output dwords ([mem[0..15], act_v2, act_v3, inact_v2, inact_v3]) under `policy`."""
    if policy not in ("all_or_nothing", "per_dword"):
        raise ValueError(f"unknown policy {policy!r}")
    is_x2 = case.op.endswith("_x2")
    kind = case.op[len("buffer_atomic_"):-3] if is_x2 else case.op[len("buffer_atomic_"):]
    fn = OP_FN[kind]
    off = case.inst_off + case.voff
    offs = off + case.soff
    chk_off = off if case.oob_select == 0 else offs
    eff_addr = case.base_add + offs + (case.idx * case.stride if case.idxen else 0)

    mem = [mem_init(i) for i in range(MEM_WORDS)]
    n_dwords = 2 if is_x2 else 1
    aligned = (eff_addr % (4 * n_dwords)) == 0
    dw = eff_addr // 4

    if policy == "all_or_nothing":
        ok = aligned and _in_bounds(case, chk_off, 4 * n_dwords)
        ok_dwords = [ok] * n_dwords
    else:
        ok_dwords = [aligned and _in_bounds(case, chk_off + 4 * w, 4) for w in range(n_dwords)]

    ret = [DATA_LO, DATA_HI]
    data = (DATA_LO, DATA_HI)
    for w in range(n_dwords):
        if ok_dwords[w]:
            old = mem[dw + w]
            mem[dw + w] = fn(old, data[w])
            if case.glc:
                ret[w] = old
        elif case.glc:
            ret[w] = 0
    return mem + ret + [DATA_LO, DATA_HI]


def kernel_lines(case, wave=32):
    """GFX10 raw kernel lines executing `case` on one active lane (0 in wave32, 37 in wave64)."""
    if wave not in (32, 64):
        raise ValueError(f"unsupported wave size {wave}")
    active_lane = 0 if wave == 32 else 37
    inactive_lane = active_lane ^ 1
    vcc = "vcc_lo" if wave == 32 else "vcc"
    exec_reg = "exec_lo" if wave == 32 else "exec"
    saveexec = "s_and_saveexec_b32 s12, vcc_lo" if wave == 32 else "s_and_saveexec_b64 s[12:13], vcc"
    saveexec_post = "s_and_saveexec_b32 s15, vcc_lo" if wave == 32 else "s_and_saveexec_b64 s[16:17], vcc"
    restore_pre = f"s_mov_b{'32' if wave == 32 else '64'} {exec_reg}, {'s12' if wave == 32 else 's[12:13]'}"
    restore_post = f"s_mov_b{'32' if wave == 32 else '64'} {exec_reg}, {'s15' if wave == 32 else 's[16:17]'}"

    lines = [f"v_cmp_eq_u32 {vcc}, 0, v0", saveexec]
    for i in range(MEM_WORDS):
        lines += [
            f"v_mov_b32 v4, {i * 4}",
            f"v_mov_b32 v5, 0x{mem_init(i):x}",
            "global_store_dword v4, v5, s[2:3] glc slc",
        ]
    lines += ["s_waitcnt vmcnt(0)", "buffer_gl0_inv", "buffer_gl1_inv", "s_waitcnt vmcnt(0)", restore_pre]
    lines += [f"v_mov_b32 v2, 0x{DATA_LO:x}", f"v_mov_b32 v3, 0x{DATA_HI:x}"]

    if case.base_add:
        lines += [f"s_add_u32 s8, s2, 0x{case.base_add:x}", "s_addc_u32 s9, s3, 0", "s_and_b32 s9, s9, 0xffff"]
    else:
        lines += ["s_mov_b32 s8, s2", "s_and_b32 s9, s3, 0xffff"]
    if case.stride:
        lines.append(f"s_or_b32 s9, s9, 0x{(case.stride & 0x3FFF) << 16:x}")
    lines += [
        f"s_mov_b32 s10, 0x{case.num_records:x}",
        f"s_mov_b32 s11, 0x{descriptor_word3(case.oob_select):x}",
        f"s_mov_b32 s14, 0x{case.soff:x}",
    ]

    if case.idxen:
        lines += [f"v_mov_b32 v6, 0x{case.idx:x}", f"v_mov_b32 v7, 0x{case.voff:x}"]
        vaddr, mods = "v[6:7]", "idxen offen"
    else:
        lines.append(f"v_mov_b32 v6, 0x{case.voff:x}")
        vaddr, mods = "v6", "offen"
    if case.inst_off:
        mods += f" offset:{case.inst_off}"
    if case.glc:
        mods += " glc"

    vdst = "v[2:3]" if case.op.endswith("_x2") else "v2"
    lines += [f"v_cmp_eq_u32 {vcc}, {active_lane}, v0", saveexec,
              f"{case.op} {vdst}, {vaddr}, s[8:11], s14 {mods}",
              "s_waitcnt vmcnt(0)", restore_pre]

    lines += [
        f"v_cmp_eq_u32 {vcc}, {active_lane}, v0", saveexec_post,
        "v_mov_b32 v4, 64", "global_store_dword v4, v2, s[2:3] glc",
        "v_mov_b32 v4, 68", "global_store_dword v4, v3, s[2:3] glc",
        restore_post,
        f"v_cmp_eq_u32 {vcc}, {inactive_lane}, v0", saveexec_post,
        "v_mov_b32 v4, 72", "global_store_dword v4, v2, s[2:3] glc",
        "v_mov_b32 v4, 76", "global_store_dword v4, v3, s[2:3] glc",
        "s_waitcnt vmcnt(0)", "s_endpgm",
    ]
    return lines


def _mem_diff_summary(got, exp):
    diffs = []
    for i in range(MEM_WORDS):
        if got[i] != exp[i]:
            diffs.append(f"mem[{i}]:0x{got[i]:08x}!=0x{exp[i]:08x}")
    if tuple(got[16:18]) != tuple(exp[16:18]):
        diffs.append(f"vdata:(0x{got[16]:08x},0x{got[17]:08x})!=(0x{exp[16]:08x},0x{exp[17]:08x})")
    return ", ".join(diffs)


def evaluate(cases, hw_by_wave):
    """Compare hardware outputs in `hw_by_wave` against `all_or_nothing` and `per_dword` models."""
    waves = sorted(hw_by_wave)
    harness_errors = []
    for w in waves:
        res = hw_by_wave[w]
        if len(res) != len(cases):
            harness_errors.append(f"wave{w}: expected {len(cases)} results, got {len(res)}")
            continue
        for c, words in zip(cases, res):
            if not isinstance(words, list) or len(words) != OUT_WORDS:
                harness_errors.append(f"wave{w} {c.name}: invalid output {words!r}")
            elif (words[18], words[19]) != (DATA_LO, DATA_HI):
                harness_errors.append(
                    f"wave{w} {c.name}: inactive lane vdata changed to (0x{words[18]:08x}, 0x{words[19]:08x})")

    mismatches = []
    wave_diffs = []
    straddle_distinguishing = 0
    straddle_matched_aon = 0
    straddle_matched_pdw = 0
    by_group = {}

    if not harness_errors:
        first_wave = waves[0]
        for idx, c in enumerate(cases):
            aon = expected_words(c, "all_or_nothing")
            pdw = expected_words(c, "per_dword")
            distinguishes = (aon != pdw)
            if distinguishes:
                straddle_distinguishing += 1

            g = by_group.setdefault(c.group, {"total": 0, "matched": 0, "distinguishing": 0})
            g["total"] += 1
            if distinguishes:
                g["distinguishing"] += 1

            all_ok = True
            for w in waves:
                got = hw_by_wave[w][idx]
                if got != aon:
                    all_ok = False
                    mismatches.append({
                        "wave": w, "case": c.name, "group": c.group,
                        "detail": _mem_diff_summary(got, aon),
                        "fits_per_dword": (got == pdw),
                    })
                if w != first_wave and got != hw_by_wave[first_wave][idx]:
                    wave_diffs.append({"wave": w, "case": c.name})

            if all_ok:
                g["matched"] += 1
                if distinguishes:
                    straddle_matched_aon += 1
            elif all(hw_by_wave[w][idx] == pdw for w in waves):
                straddle_matched_pdw += 1

    return {
        "cases": len(cases),
        "waves": waves,
        "harness_errors": harness_errors,
        "mismatches": len(mismatches),
        "wave_diffs": len(wave_diffs),
        "straddle_distinguishing": straddle_distinguishing,
        "straddle_matched_all_or_nothing": straddle_matched_aon,
        "straddle_matched_per_dword": straddle_matched_pdw,
        "by_group": by_group,
        "first_mismatches": mismatches[:10],
        "first_wave_diffs": wave_diffs[:10],
    }


def run_probe(oracle, cases, waves):
    from hw_oracle import asm
    hw_by_wave = {}
    for wave in waves:
        kernels = [asm.Kernel(kernel_lines(c, wave), wave, wave, OUT_WORDS, raw=True, vgprs=16, sgprs=24)
                   for c in cases]
        hw_by_wave[wave] = oracle.run(kernels, words=OUT_WORDS)
    return hw_by_wave


def self_test():
    assert descriptor_word3(0) == 0x01016FAC
    assert descriptor_word3(3) == 0x31016FAC
    cases = build_cases()
    assert len(cases) == 137, len(cases)
    distinguishing = [c for c in cases if expected_words(c, "all_or_nothing") != expected_words(c, "per_dword")]
    assert len(distinguishing) == 53, len(distinguishing)
    for op in OPS:
        op_dist = [c for c in distinguishing if c.op == f"buffer_atomic_{op}_x2"]
        assert len(op_dist) >= 10, (op, len(op_dist))
    for w in (32, 64):
        lines = kernel_lines(cases[0], wave=w)
        assert lines[-1] == "s_endpgm"
    print(f"self-test ok ({len(cases)} cases, {len(distinguishing)} straddling distinguishing)")
    return 0


def _write_json(path, payload):
    out_path = os.path.abspath(path)
    if os.path.isdir(out_path):
        print(f"ERROR: --json path is a directory: {out_path}", file=sys.stderr)
        return 2
    try:
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    except OSError as e:
        print(f"ERROR: cannot write --json output {out_path}: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ops", default=",".join(OPS))
    ap.add_argument("--waves", default="32,64")
    ap.add_argument("--hw-arch")
    ap.add_argument("--json", help="write case metadata, counts, and mismatches here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    ops = tuple(o.strip() for o in a.ops.split(",") if o.strip())
    wave_strs = [w.strip() for w in a.waves.split(",") if w.strip()]
    bad = [o for o in ops if o not in OPS] + [w for w in wave_strs if w not in ("32", "64")]
    if bad or not ops or not wave_strs:
        ap.error("unknown or empty: " + ", ".join(bad or ["--ops/--waves"]))
    waves = tuple(int(w) for w in wave_strs)

    from hw_oracle import asm, runner
    try:
        oracle = runner.open_oracle(arch=a.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    try:
        if asm.family(oracle.arch) != "gfx10":
            print(f"hardware oracle unavailable: {oracle.arch} is not GFX10; the descriptor here is GFX10's (RDNA2)")
            return 77
        print(f"hardware oracle: {oracle.device}")
        cases = build_cases(ops)
        hw_by_wave = run_probe(oracle, cases, waves)
        summary = evaluate(cases, hw_by_wave)
    except RuntimeError as e:
        print(f"harness error: {e}")
        return 2
    finally:
        oracle.close()

    if summary["harness_errors"]:
        for err in summary["harness_errors"][:5]:
            print(f"harness error: {err}")
        return 2

    print(f"buffer_atomic_*_x2 straddling bounds ({summary['cases']} cases x {len(waves)} waves): "
          f"all_or_nothing {summary['straddle_matched_all_or_nothing']}/{summary['straddle_distinguishing']}, "
          f"per_dword {summary['straddle_matched_per_dword']}/{summary['straddle_distinguishing']}, "
          f"mismatches {summary['mismatches']}, wave32/64 diffs {summary['wave_diffs']}")
    for grp, stats in summary["by_group"].items():
        print(f"  {grp:18s}: {stats['matched']}/{stats['total']} match all_or_nothing "
              f"({stats['distinguishing']} straddling)")
    for m in summary["first_mismatches"]:
        print(f"  MISMATCH wave{m['wave']} {m['case']}: {m['detail']} (per_dword={m['fits_per_dword']})")

    if a.json:
        payload = {
            "device": str(oracle.device),
            "arch": oracle.arch,
            "summary": summary,
            "cases": [asdict(c) for c in cases],
        }
        jrc = _write_json(a.json, payload)
        if jrc:
            return jrc

    return 1 if (summary["mismatches"] or summary["wave_diffs"]) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
