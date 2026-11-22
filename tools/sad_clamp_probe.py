r"""Measure RDNA2 VOP3 `clamp` across all eight SAD-family instructions. Standard library only.

`docs/design.md` previously recorded VOP3 `clamp` as `Unsupported` across the SAD family because the RDNA2 ISA
pseudocode does not spell out its effect on integer sum-of-absolute-differences opcodes. Running this probe natively
on RDNA2 (`gfx1036`, wave32 & wave64) via `tools/hw_oracle` measures:
- `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_sad_u32`, `v_msad_u8`: VOP3 `clamp` performs 32-bit unsigned
  saturating accumulation (`0xFFFFFFFF` on overflow). In `v_sad_hi_u8`, overflow of `(sad << 16) + S2 >= 2^32`
  saturates the entire 32-bit destination to `0xFFFFFFFF` (not only the high 16 bits).
- `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`: VOP3 `clamp` independently saturates each 16-bit lane `D.u16[k]` to
  `0xFFFF` on overflow with zero carry between adjacent halves.
- `v_mqsad_u32_u8`: VOP3 `clamp` independently saturates each 32-bit accumulator lane `D[k]` to `0xFFFFFFFF`.

Usage:
    python -I tools/sad_clamp_probe.py [--waves 32,64] [--ops ...] [--hw-arch gfx1036] [--json out.json]
    python -I tools/sad_clamp_probe.py --self-test

Exit codes:
    0   all hardware outputs match the reference model across requested waves (with wave32 == wave64)
    1   at least one word mismatch or wave32/wave64 difference
    2   bad CLI arguments, unwritable --json path, or harness error
    77  hardware oracle unavailable or GPU is not GFX10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

M16 = 0xFFFF
M32 = 0xFFFFFFFF

SAD32_OPS = ("v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8")
QSAD16_OPS = ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8")
MQSAD32_OPS = ("v_mqsad_u32_u8",)
OPS: tuple[str, ...] = SAD32_OPS + QSAD16_OPS + MQSAD32_OPS


def _parts(x: int, width: int) -> list[int]:
    mask = (1 << width) - 1
    return [(x >> off) & mask for off in range(0, 32, width)]


def _sad_sum(x: int, y: int, width: int, *, masked: bool = False) -> int:
    total = 0
    for a, b in zip(_parts(x, width), _parts(y, width)):
        if masked and b == 0:
            continue
        total += abs(a - b)
    return total


def expected_sad32(op: str, s0: int, s1: int, s2: int, *, clamp: bool) -> int:
    """Expected 32-bit result for v_sad_u8, v_sad_hi_u8, v_sad_u16, v_sad_u32, or v_msad_u8."""
    s0 &= M32
    s1 &= M32
    s2 &= M32
    if op == "v_sad_u8":
        total = _sad_sum(s0, s1, 8) + s2
    elif op == "v_sad_hi_u8":
        total = (_sad_sum(s0, s1, 8) << 16) + s2
    elif op == "v_sad_u16":
        total = _sad_sum(s0, s1, 16) + s2
    elif op == "v_sad_u32":
        total = abs(s0 - s1) + s2
    elif op == "v_msad_u8":
        total = _sad_sum(s0, s1, 8, masked=True) + s2
    else:
        raise ValueError(f"unknown 32-bit SAD op: {op}")
    if clamp and total > M32:
        return M32
    return total & M32


def expected_qsad_pk_u16(op: str, s0: int, s1: int, s2: int, *, clamp: bool) -> tuple[int, int]:
    """Expected (d0, d1) 32-bit words for v_qsad_pk_u16_u8 or v_mqsad_pk_u16_u8."""
    if op not in QSAD16_OPS:
        raise ValueError(f"unknown packed u16 QSAD op: {op}")
    masked = op == "v_mqsad_pk_u16_u8"
    s0 &= (1 << 64) - 1
    s1 &= M32
    s2 &= (1 << 64) - 1
    lanes: list[int] = []
    for k in range(4):
        win = (s0 >> (8 * k)) & M32
        val = ((s2 >> (16 * k)) & M16) + _sad_sum(win, s1, 8, masked=masked)
        lanes.append(min(val, M16) if clamp else (val & M16))
    return lanes[0] | (lanes[1] << 16), lanes[2] | (lanes[3] << 16)


def expected_mqsad_u32(s0: int, s1: int, accs: tuple[int, int, int, int], *, clamp: bool) -> tuple[int, int, int, int]:
    """Expected (d0, d1, d2, d3) 32-bit words for v_mqsad_u32_u8."""
    s0 &= (1 << 64) - 1
    s1 &= M32
    out: list[int] = []
    for k in range(4):
        win = (s0 >> (8 * k)) & M32
        val = (accs[k] & M32) + _sad_sum(win, s1, 8, masked=True)
        out.append(M32 if (clamp and val > M32) else (val & M32))
    return (out[0], out[1], out[2], out[3])


# Test vectors per op:
#   SAD32_OPS: (case_name, s0, s1, s2)
#   QSAD16_OPS: (case_name, s0_64, s1, s2_64)
#   MQSAD32_OPS: (case_name, s0_64, s1, (acc0, acc1, acc2, acc3))
CASES: dict[str, tuple[tuple, ...]] = {
    "v_sad_u8": (
        ("normal",       0x01020304, 0x04030201, 10),         # diff=8, sum=18
        ("exact_max",    0xFF00FF00, 0x00FF00FF, 0xFFFFFC03), # 0x3FC + 0xFFFFFC03 = 0xFFFFFFFF
        ("ovf_by_1",     0xFF00FF00, 0x00FF00FF, 0xFFFFFC04), # 0x3FC + 0xFFFFFC04 = 0x100000000
        ("max_wrap",     0xFF00FF00, 0x00FF00FF, 0xFFFFFFFF), # 0x3FC + 0xFFFFFFFF = 0x1000003FB
        ("zero_diff",    0x12345678, 0x12345678, 0xFFFFFFFF), # 0 + 0xFFFFFFFF = 0xFFFFFFFF
        ("small_ovf",    0x01000000, 0x00000000, 0xFFFFFFFF), # 1 + 0xFFFFFFFF
    ),
    "v_sad_hi_u8": (
        ("normal",       0x01020304, 0x04030201, 0x00000007), # (8 << 16) + 7
        ("hi_exact_ffff",0x01000000, 0x00000000, 0xFFFE1234), # (1 << 16) + 0xFFFE1234 = 0xFFFF1234
        ("hi_ovf_by_1",  0x01000000, 0x00000000, 0xFFFF1234), # (1 << 16) + 0xFFFF1234 = 0x100001234 -> 0xFFFFFFFF
        ("hi_ovf_lo0005",0xFF00FF00, 0x00FF00FF, 0xFFFF0005), # (0x3FC << 16) + 0xFFFF0005 -> 0xFFFFFFFF
        ("hi_ovf_lo1234",0xFF000000, 0x00000000, 0xFF801234), # (0xFF << 16) + 0xFF801234 -> 0xFFFFFFFF
        ("hi_zero_sad",  0x12345678, 0x12345678, 0xFFFF1234), # 0 + 0xFFFF1234 = 0xFFFF1234
        ("hi_ovf_lo0000",0x01000000, 0x00000000, 0xFFFF0000), # 0x100000000 -> 0xFFFFFFFF
        ("hi_max_no_ovf",0xFF00FF00, 0x00FF00FF, 0xFC03FFFF), # (0x3FC << 16) + 0xFC03FFFF = 0xFFFFFFFF
        ("hi_max_ovf1",  0xFF00FF00, 0x00FF00FF, 0xFC040000), # (0x3FC << 16) + 0xFC040000 = 0x100000000 -> 0xFFFFFFFF
    ),
    "v_sad_u16": (
        ("normal",       0x10000020, 0x00000010, 5),          # 0x1010 + 5
        ("exact_max",    0xFFFF0000, 0x0000FFFF, 0xFFFE0001), # 0x1FFFE + 0xFFFE0001 = 0xFFFFFFFF
        ("ovf_by_1",     0xFFFF0000, 0x0000FFFF, 0xFFFE0002), # 0x1FFFE + 0xFFFE0002 = 0x100000000
        ("max_wrap",     0xFFFF0000, 0x0000FFFF, 0xFFFFFFFF), # 0x1FFFE + 0xFFFFFFFF
        ("zero_diff",    0xABCD1234, 0xABCD1234, 0xFFFFFFFF), # 0 + 0xFFFFFFFF
    ),
    "v_sad_u32": (
        ("normal",       100,        20,         5),          # 85
        ("exact_max",    0xFFFFFFFF, 0x00000000, 0x00000000), # 0xFFFFFFFF
        ("ovf_by_1",     0xFFFFFFFF, 0x00000000, 0x00000001), # 0x100000000
        ("max_plus_max", 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF), # 0x1FFFFFFFE -> plain 0xFFFFFFFE, clamp 0xFFFFFFFF
        ("rev_diff_ovf", 0x00000000, 0xFFFFFFFF, 0x00000001), # |0 - 0xFFFFFFFF| + 1
        ("half_exact",   0x80000000, 0x00000000, 0x7FFFFFFF), # 0xFFFFFFFF exact
        ("half_ovf",     0x80000000, 0x00000000, 0x80000000), # 0x100000000
    ),
    "v_msad_u8": (
        ("masked_no_ovf",0xFF00FF00, 0x00FF00FF, 0x0000000A), # 2 masked bytes -> 0x1FE + 10
        ("all_masked",   0xFF00FF00, 0x00000000, 0xFFFFFFFF), # s1=0 -> 0 + 0xFFFFFFFF = 0xFFFFFFFF
        ("mask_spares",  0xFF00FF00, 0x00010001, 0xFFFFFEFE), # unmasked bytes diff=0+0 -> stays under; wait diff(0x00,0x01)*2=2 + FFFFFEFE = FFFFFF00
        ("mask_partial", 0xFF00FF00, 0x01000100, 0xFFFFFE04), # diff(0xFF,0x01)*2 = 0x1FC + 0xFFFFFE04 = 0x100000000
        ("all_active",   0xFF00FF00, 0x01FF01FF, 0xFFFFFF00), # all 4 non-zero -> overflows
        ("exact_max",    0xFF00FF00, 0x01FF01FF, 0xFFFFFC09), # diff = 0xFE+0xFF+0xFE+0xFF = 0x3F6; 0x3F6 + 0xFFFFFC09 = 0xFFFFFFFF
    ),
    "v_qsad_pk_u16_u8": (
        ("sliding_norm", 0x0807060504030201, 0x04030201, 0x0004000300020001),
        ("lo_ovf_hi_0",  0x0000000000000000, 0xFFFFFFFF, 0xFC03FC040000FFFF), # lane0/2 ovf (no carry!), lane1=0x03FC, lane3=0xFFFF
        ("hi_ovf_lo_ok", 0x00FF00FFFF00FF00, 0x00FF00FF, 0xFF00FFFFFFFFF800), # hi halves overflow cleanly
        ("all_ovf",      0x0000000000000000, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF), # all 4 lanes overflow
        ("all_exact",    0x0000000000000000, 0xFFFFFFFF, 0xFC03FC03FC03FC03), # all 4 lanes hit 0xFFFF exact
    ),
    "v_mqsad_pk_u16_u8": (
        ("sliding_norm", 0x0807060504030201, 0x04030201, 0x0004000300020001),
        ("all_masked",   0x123456789ABCDEF0, 0x00000000, 0xFFFF0000FFFF1234), # s1=0 leaves all 4 lanes untouched
        ("lo_ovf_hi_ok", 0x0000000000000000, 0xFF00FFFF, 0xFD02FFFF0000FD03), # sad=0x2FD: lane0/2 ovf (no carry!), lane1=0x02FD, lane3=0xFFFF
        ("hi_ovf_lo_ok", 0x00FF00FFFF00FF00, 0x01FF01FF, 0xFF00FFFFFFFFF800),
        ("all_ovf",      0x0000000000000000, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
    ),
    "v_mqsad_u32_u8": (
        ("sliding_norm", 0x0807060504030201, 0x04030201, (0, 0, 0, 0)),
        ("mixed_lanes",  0x00FF00FFFF00FF00, 0x01FF01FF, (0xFFFFFF00, 0xFFFFFF00, 0x00000001, 0xFFFFFFFF)),
        ("exact_vs_ovf", 0x0000000000000000, 0xFF00FFFF, (0xFFFFFD02, 0xFFFFFD03, 0xFFFFFFFF, 0x00000010)), # sad=0x2FD: lane0=0xFFFFFFFF, lane1=ovf
        ("all_masked",   0x123456789ABCDEF0, 0x00000000, (0xFFFFFFFF, 0x12345678, 0, 0xFFFFFFFF)),
    ),
}


def expected_case_words(op: str, case: tuple, *, clamp: bool) -> tuple[int, ...]:
    """Return the expected 32-bit destination words for one entry in CASES[op]."""
    if op in SAD32_OPS:
        _, s0, s1, s2 = case
        return (expected_sad32(op, s0, s1, s2, clamp=clamp),)
    if op in QSAD16_OPS:
        _, s0, s1, s2 = case
        return expected_qsad_pk_u16(op, s0, s1, s2, clamp=clamp)
    if op in MQSAD32_OPS:
        _, s0, s1, accs = case
        return expected_mqsad_u32(s0, s1, accs, clamp=clamp)
    raise ValueError(f"unknown SAD op: {op}")


def words_per_case(op: str) -> int:
    if op in SAD32_OPS:
        return 1
    if op in QSAD16_OPS:
        return 2
    if op in MQSAD32_OPS:
        return 4
    raise ValueError(f"unknown SAD op: {op}")


def expected_op_words(op: str, *, clamp: bool) -> list[int]:
    out: list[int] = []
    for case in CASES[op]:
        out.extend(expected_case_words(op, case, clamp=clamp))
    return out


def kernel_lines(op: str, clamp: bool) -> list[str]:
    """Raw GFX10 kernel lines running every case in CASES[op] sequentially on lane 0."""
    lines: list[str] = []
    suf = " clamp" if clamp else ""
    out_idx = 0
    for case in CASES[op]:
        if op in SAD32_OPS:
            _, s0, s1, s2 = case
            lines += [
                f"v_mov_b32_e32 v0, {s0:#010x}",
                f"v_mov_b32_e32 v2, {s1:#010x}",
                f"v_mov_b32_e32 v4, {s2:#010x}",
                f"{op} v8, v0, v2, v4{suf}",
                f"v_mov_b32_e32 v12, {out_idx * 4}",
                "global_store_dword v12, v8, s[2:3]",
            ]
            out_idx += 1
        elif op in QSAD16_OPS:
            _, s0, s1, s2 = case
            lines += [
                f"v_mov_b32_e32 v0, {s0 & M32:#010x}",
                f"v_mov_b32_e32 v1, {(s0 >> 32) & M32:#010x}",
                f"v_mov_b32_e32 v2, {s1:#010x}",
                f"v_mov_b32_e32 v4, {s2 & M32:#010x}",
                f"v_mov_b32_e32 v5, {(s2 >> 32) & M32:#010x}",
                f"{op} v[8:9], v[0:1], v2, v[4:5]{suf}",
                f"v_mov_b32_e32 v12, {out_idx * 4}",
                "global_store_dword v12, v8, s[2:3]",
                f"v_mov_b32_e32 v12, {(out_idx + 1) * 4}",
                "global_store_dword v12, v9, s[2:3]",
            ]
            out_idx += 2
        elif op in MQSAD32_OPS:
            _, s0, s1, accs = case
            lines += [
                f"v_mov_b32_e32 v0, {s0 & M32:#010x}",
                f"v_mov_b32_e32 v1, {(s0 >> 32) & M32:#010x}",
                f"v_mov_b32_e32 v2, {s1:#010x}",
                f"v_mov_b32_e32 v4, {accs[0]:#010x}",
                f"v_mov_b32_e32 v5, {accs[1]:#010x}",
                f"v_mov_b32_e32 v6, {accs[2]:#010x}",
                f"v_mov_b32_e32 v7, {accs[3]:#010x}",
                f"{op} v[8:11], v[0:1], v2, v[4:7]{suf}",
            ]
            for wi in range(4):
                lines += [
                    f"v_mov_b32_e32 v12, {(out_idx + wi) * 4}",
                    f"global_store_dword v12, v{8 + wi}, s[2:3]",
                ]
            out_idx += 4
        else:
            raise ValueError(f"unknown SAD op: {op}")
    lines += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return lines


def build_jobs(ops: list[str], waves: list[int]):
    """Build (kernels, metadata) pairs for oracle.run()."""
    from hw_oracle import asm

    kernels, meta = [], []
    for wave in waves:
        for op in ops:
            n_words = len(CASES[op]) * words_per_case(op)
            for clamp in (False, True):
                kernels.append(
                    asm.Kernel(kernel_lines(op, clamp), wave=wave, threads=1, words=n_words, raw=True)
                )
                meta.append((wave, op, clamp))
    return kernels, meta


def evaluate_results(ops: list[str], waves: list[int], meta, raw_results) -> dict:
    """Compare hardware outputs against expected_op_words and check wave32 vs wave64 parity."""
    by_key: dict[tuple[int, str, bool], list[int]] = {}
    harness_errors: list[dict] = []
    for (wave, op, clamp), res in zip(meta, raw_results):
        expected_len = len(CASES[op]) * words_per_case(op)
        if isinstance(res, str):
            harness_errors.append({"wave": wave, "op": op, "clamp": clamp, "error": res})
            continue
        if len(res) != expected_len:
            harness_errors.append({
                "wave": wave,
                "op": op,
                "clamp": clamp,
                "error": f"expected {expected_len} words, got {len(res)}",
            })
            continue
        by_key[(wave, op, clamp)] = [w & M32 for w in res]

    total_words = sum(len(CASES[op]) * words_per_case(op) for op in ops) * len(waves) * 2
    if harness_errors:
        return {
            "total_words": total_words,
            "mismatches": 0,
            "wave_diffs": 0,
            "harness_errors": harness_errors,
            "mismatch_details": [],
            "wave_diff_details": [],
        }

    mismatches = 0
    mismatch_details: list[dict] = []
    for (wave, op, clamp), words in by_key.items():
        want = expected_op_words(op, clamp=clamp)
        wpc = words_per_case(op)
        for ci, case in enumerate(CASES[op]):
            for wi in range(wpc):
                idx = ci * wpc + wi
                if words[idx] != want[idx]:
                    mismatches += 1
                    if len(mismatch_details) < 32:
                        mismatch_details.append({
                            "wave": wave,
                            "op": op,
                            "clamp": clamp,
                            "case": case[0],
                            "word_idx": wi,
                            "expected": f"0x{want[idx]:08x}",
                            "actual": f"0x{words[idx]:08x}",
                        })

    wave_diffs = 0
    wave_diff_details: list[dict] = []
    if 32 in waves and 64 in waves:
        for op in ops:
            wpc = words_per_case(op)
            for clamp in (False, True):
                w32 = by_key[(32, op, clamp)]
                w64 = by_key[(64, op, clamp)]
                for ci, case in enumerate(CASES[op]):
                    for wi in range(wpc):
                        idx = ci * wpc + wi
                        if w32[idx] != w64[idx]:
                            wave_diffs += 1
                            if len(wave_diff_details) < 32:
                                wave_diff_details.append({
                                    "op": op,
                                    "clamp": clamp,
                                    "case": case[0],
                                    "word_idx": wi,
                                    "wave32": f"0x{w32[idx]:08x}",
                                    "wave64": f"0x{w64[idx]:08x}",
                                })

    return {
        "total_words": total_words,
        "mismatches": mismatches,
        "wave_diffs": wave_diffs,
        "harness_errors": [],
        "mismatch_details": mismatch_details,
        "wave_diff_details": wave_diff_details,
    }


def self_test() -> int:
    """Verify reference tables, kernel generation, and evaluation logic without a GPU."""
    # 1. Verify v_sad_hi_u8 full-32-bit clamp vs plain wrap on boundary inputs.
    assert expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFE1234, clamp=True) == 0xFFFF1234
    assert expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFF1234, clamp=False) == 0x00001234
    assert expected_sad32("v_sad_hi_u8", 0x01000000, 0x00000000, 0xFFFF1234, clamp=True) == 0xFFFFFFFF

    # 2. Verify v_sad_u32 max+max wrap vs clamp.
    assert expected_sad32("v_sad_u32", 0xFFFFFFFF, 0, 0xFFFFFFFF, clamp=False) == 0xFFFFFFFE
    assert expected_sad32("v_sad_u32", 0xFFFFFFFF, 0, 0xFFFFFFFF, clamp=True) == 0xFFFFFFFF

    # 3. Verify packed u16 QSAD/MQSAD clamp stays lane-local with no carry into the upper half.
    assert expected_qsad_pk_u16("v_qsad_pk_u16_u8", 0, 0xFFFFFFFF, 0x0000FFFF, clamp=True) == (0x03FCFFFF, 0x03FC03FC)
    assert expected_qsad_pk_u16("v_mqsad_pk_u16_u8", 0, 0xFF00FFFF, 0x0000FD03, clamp=True) == (0x02FDFFFF, 0x02FD02FD)

    # 4. Verify v_mqsad_u32_u8 independent lane saturation and masked s1=0 passthrough.
    assert expected_mqsad_u32(0, 0xFF00FFFF, (0xFFFFFD02, 0xFFFFFD03, 0xFFFFFFFF, 0x00000010), clamp=True) == (
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000030D
    )
    assert expected_mqsad_u32(0x123456789ABCDEF0, 0, (0xFFFFFFFF, 5, 0, 0x80000000), clamp=True) == (
        0xFFFFFFFF, 5, 0, 0x80000000
    )

    # 5. Verify every op has distinct plain vs clamp results on at least one test case.
    for op in OPS:
        plain = expected_op_words(op, clamp=False)
        clamped = expected_op_words(op, clamp=True)
        assert plain != clamped, f"{op} test table does not exercise clamp saturation"
        lines_p = kernel_lines(op, False)
        lines_c = kernel_lines(op, True)
        assert lines_p[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]
        assert lines_c[-2:] == ["s_waitcnt vmcnt(0)", "s_endpgm"]

    # 6. Verify evaluate_results detects both exact passes and injected mismatches.
    meta = [(32, op, cl) for op in OPS for cl in (False, True)]
    raw_ok = [expected_op_words(op, clamp=cl) for _, op, cl in meta]
    rep_ok = evaluate_results(list(OPS), [32], meta, raw_ok)
    assert rep_ok["mismatches"] == 0 and not rep_ok["harness_errors"], rep_ok

    raw_bad = [list(w) for w in raw_ok]
    raw_bad[0][0] ^= 1
    rep_bad = evaluate_results(list(OPS), [32], meta, raw_bad)
    assert rep_bad["mismatches"] == 1, rep_bad

    total_cases = sum(len(CASES[op]) for op in OPS)
    print(f"self-test ok ({len(OPS)} ops, {total_cases} cases, {rep_ok['total_words']} words/wave-pair)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--waves", default="32,64", help="comma-separated wave sizes (default: 32,64)")
    ap.add_argument("--ops", default=",".join(OPS), help="comma-separated SAD opcodes to probe")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    ap.add_argument("--json", dest="json_out", default=None, help="write JSON report to this path")
    ap.add_argument("--self-test", action="store_true", help="verify reference model and kernel generator without GPU")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    try:
        waves = [int(w.strip()) for w in args.waves.split(",") if w.strip()]
    except ValueError:
        print(f"ERROR: invalid --waves value: {args.waves!r}", file=sys.stderr)
        return 2
    if not waves or any(w not in (32, 64) for w in waves):
        print(f"ERROR: --waves must list 32 and/or 64, got {args.waves!r}", file=sys.stderr)
        return 2
    waves = list(dict.fromkeys(waves))

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    unknown = [o for o in ops if o not in OPS]
    if not ops or unknown:
        print(f"ERROR: unknown or empty --ops {unknown or ops} (valid: {', '.join(OPS)})", file=sys.stderr)
        return 2
    ops = list(dict.fromkeys(ops))

    if args.json_out is not None:
        out_check = os.path.abspath(args.json_out)
        if os.path.isdir(out_check):
            print(f"ERROR: --json path is a directory: {out_check}", file=sys.stderr)
            return 2

    from hw_oracle import asm, runner

    try:
        oracle = runner.open_oracle(arch=args.hw_arch)
    except runner.Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77

    try:
        try:
            fam = asm.family(oracle.arch)
        except ValueError:
            fam = None
        if fam != "gfx10":
            print(f"skipping: requires GFX10 RDNA2 hardware, found {oracle.arch}")
            return 77

        kernels, meta = build_jobs(ops, waves)
        raw_results = oracle.run(kernels)
        summary = evaluate_results(ops, waves, meta, raw_results)
        device = oracle.device
        arch = oracle.arch
    finally:
        oracle.close()

    total_words = summary["total_words"]
    print(f"sad_clamp_probe on {device} ({arch}):")
    if summary["harness_errors"]:
        for err in summary["harness_errors"][:8]:
            print(f"  harness error (wave{err['wave']} {err['op']} clamp={err['clamp']}): {err['error']}")
        return 2

    print(
        f"  {total_words} word evaluations across {len(waves)} wave(s) x {len(ops)} op(s) x 2 (plain+clamp): "
        f"mismatches={summary['mismatches']}, wave_diffs={summary['wave_diffs']}"
    )
    for d in summary["mismatch_details"]:
        print(
            f"  MISMATCH wave{d['wave']} {d['op']} clamp={d['clamp']} case={d['case']} "
            f"w[{d['word_idx']}]: expected={d['expected']} actual={d['actual']}"
        )
    for d in summary["wave_diff_details"]:
        print(
            f"  WAVE DIFF {d['op']} clamp={d['clamp']} case={d['case']} w[{d['word_idx']}]: "
            f"wave32={d['wave32']} wave64={d['wave64']}"
        )

    if args.json_out is not None:
        out_path = os.path.abspath(args.json_out)
        if os.path.isdir(out_path):
            print(f"ERROR: cannot write {out_path}: path is a directory", file=sys.stderr)
            return 2
        payload = {
            "device": device,
            "arch": arch,
            "waves": waves,
            "ops": ops,
            "cases_per_op": {op: len(CASES[op]) for op in ops},
            **summary,
        }
        try:
            parent = os.path.dirname(out_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write {out_path}: {e}", file=sys.stderr)
            return 2

    return 1 if (summary["mismatches"] or summary["wave_diffs"]) else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
