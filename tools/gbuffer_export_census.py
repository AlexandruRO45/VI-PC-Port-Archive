"""Count how pixel shaders produce the `mrt0` alpha and how they export `mrt4` (docs/gbuffer.md section 2).

For every listing in a disassembly directory whose file name contains `--match` (default
`PS_DeferredTextured`), the tool reads the `OSG1` output signature from the `.dxbc` beside the
listings' parent directory (as `rdna2spirv.output_signature_full` does) and the listing itself, then
prints counts only, grouped by the number of declared `SV_Target` outputs:

- `mrt0 alpha`: the `exp mrt0 ... compr` instruction's last packed register is followed back to its
  `v_cvt_pkrtz_f16_f32`, and that instruction's second source (the alpha) to the instruction that
  last wrote it.  A `v_mul_f32`/`v_mul_legacy_f32` by `0x3b808081` (1/255, inline or moved into the
  register) is followed one step further to the conversion it scales: `byte N x 1/255` for
  `v_cvt_f32_ubyteN`, `u32 x 1/255` for `v_cvt_f32_u32`.  A multiply without 1/255 is `float product`;
  anything else is named by its opcode.
- `mrt4`: the `OSG1` mask of `SV_Target4` against the export's operand pattern (`v` or `off` per
  slot) and its `compr` flag.

Limits: "last wrote" is the nearest earlier writer in listing order, not a data-flow solution.
Where a label or branch lies between a writer and its use the result is counted under `crosses`,
since another path may write the register.  Shader names are never printed.

Usage:
    python -I gbuffer_export_census.py <work>/shaders/dxbc/disasm [--match TEXT]
"""
import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdna2spirv import VOP3_TWO_DESTS, base_op, output_signature_full  # noqa: E402

SV_TARGET = 64   # D3D_NAME_TARGET
INV255 = "0x3b808081"
REG = re.compile(r"^([sv])(\d+)$|^([sv])\[(\d+):(\d+)\]$")
LABEL = re.compile(r"^\S.*:\s*$")
BRANCH = re.compile(r"^s_(cbranch|branch|setpc|swappc)")
# Opcodes whose first operand is not a register they write: exports, branches and scalar housekeeping, stores,
# and the compares, which write SCC (s_cmp, s_cmpk, s_bitcmp), VCC or an SGPR mask (v_cmp, handled in `written`)
# or EXEC (v_cmpx, any form: GFX10 has no destination operand).  The list follows rdna2spirv.reg_reads_writes
# (`reads_all`) and feature_scan.NO_SDST.
NO_DEST = ("exp", "s_cbranch", "s_branch", "s_waitcnt", "s_endpgm", "s_nop", "s_setpc", "s_sendmsg", "s_barrier",
           "s_dcache", "s_icache", "s_gl1_inv", "s_setprio", "s_sleep", "s_ttrace", "s_setreg", "s_inst_prefetch",
           "s_round_mode", "s_denorm_mode", "s_clause", "s_trap", "s_wakeup", "s_code_end", "s_atc_probe",
           "s_setkill", "s_incperflevel", "s_decperflevel", "s_version", "s_set_gpr_idx",
           "s_cmp", "s_bitcmp", "v_cmpx", "v_nop",
           "buffer_store", "tbuffer_store", "image_store", "global_store", "flat_store", "scratch_store",
           "s_store", "s_buffer_store", "ds_write", "<label>")
ATOMIC = ("buffer_atomic", "global_atomic", "flat_atomic", "image_atomic")
DS_RETURNS = ("ds_read", "ds_load", "ds_swizzle", "ds_permute", "ds_bpermute", "ds_append", "ds_consume")
SUFFIX = re.compile(r"_e(32|64)$")


def regs(tok):
    """Registers named by one operand: `v3`, `s7` or a range such as `v[4:7]`."""
    m = REG.match(tok)
    if not m:
        return set()
    if m.group(1):
        return {(m.group(1), int(m.group(2)))}
    return {(m.group(3), i) for i in range(int(m.group(4)), int(m.group(5)) + 1)}


def parse(lines):
    """Listing lines -> [(opcode, [operands])]; labels become ('<label>', [])."""
    ins = []
    for line in lines:
        code = line.split(";")[0].split("//")[0].rstrip()
        if not code.strip():
            continue
        if LABEL.match(code):
            ins.append(("<label>", []))
            continue
        parts = code.strip().split(None, 1)
        ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
        ins.append((parts[0], ops))
    return ins


def written(op, ops):
    """The s/v registers an instruction writes (VCC, EXEC, SCC and M0 are not tracked).

    The first operand is the destination except for: the opcodes of NO_DEST; `v_cmp_*`, which names an SGPR mask
    destination only in the three-operand form (`_e64`, sdwa) and writes VCC in `_e32`, where the first printed
    operand is `vcc_lo` or a source; atomics and `ds_*` without a returned value (`glc`, `_rtn`); and the VOP3b
    forms with a second SGPR destination (rdna2spirv.VOP3_TWO_DESTS: the carry or `v_div_scale` flag)."""
    if not ops or op.startswith(NO_DEST):
        return set()
    if op.startswith("v_cmp"):
        return regs(ops[0]) if len(ops) >= 3 and not op.endswith("_e32") else set()
    if op.startswith(ATOMIC):
        return regs(ops[0]) if any("glc" in o.split() for o in ops) else set()
    if op.startswith("ds_") and "_rtn" not in op and not op.startswith(DS_RETURNS):
        return set()
    dest = regs(ops[0])
    if base_op(op) in VOP3_TWO_DESTS and len(ops) >= 4:
        dest |= regs(ops[1])
    return dest


def last_writer(ins, idx, reg):
    """Index of the nearest earlier instruction that writes reg, and whether a label or branch lies between."""
    crossed = False
    for j in range(idx - 1, -1, -1):
        op, ops = ins[j]
        if op == "<label>" or BRANCH.match(op):
            crossed = True
        if reg in written(op, ops):
            return j, crossed
    return None, crossed


def literal(ins, idx, tok):
    """An operand's literal: inline, or the literal last moved into that register."""
    if tok.startswith("0x"):
        return tok
    named = regs(tok)
    if len(named) != 1:
        return None
    j, _ = last_writer(ins, idx, next(iter(named)))
    if j is None:
        return None
    op, ops = ins[j]
    if op.startswith(("s_mov_b32", "v_mov_b32")) and len(ops) == 2 and ops[1].startswith("0x"):
        return ops[1]
    return None


def single(tok):
    named = regs(tok)
    return next(iter(named)) if len(named) == 1 else None


def mrt0_alpha(ins, ei):
    """(kind, crosses) for the alpha of the export at ins[ei]."""
    ops = ins[ei][1]
    if not any("compr" in o for o in ops):
        return "not compr", False
    packed = single(ops[3]) if len(ops) > 3 else None
    if packed is None:
        return "alpha off", False
    j, crossed = last_writer(ins, ei, packed)
    if j is None:
        return "no writer", crossed
    op, ops = ins[j]
    if not op.startswith("v_cvt_pkrtz_f16_f32") or len(ops) < 3 or single(ops[2]) is None:
        return "packed by " + SUFFIX.sub("", op), crossed
    k, more = last_writer(ins, j, single(ops[2]))
    crossed |= more
    if k is None:
        return "no writer", crossed
    op, ops = ins[k]
    base = SUFFIX.sub("", op)
    if base not in ("v_mul_f32", "v_mul_legacy_f32") or len(ops) < 3:
        return base, crossed
    for scale, other in ((1, 2), (2, 1)):
        if literal(ins, k, ops[scale]) == INV255:
            src = single(ops[other])
            m, more = last_writer(ins, k, src) if src else (None, False)
            crossed |= more
            conv = SUFFIX.sub("", ins[m][0]) if m is not None else "?"
            if conv.startswith("v_cvt_f32_ubyte"):
                return f"byte {conv[-1]} x 1/255", crossed
            if conv == "v_cvt_f32_u32":
                return "u32 x 1/255", crossed
            return conv + " x 1/255", crossed
    return "float product", crossed


def mrt4_form(ins, ei):
    ops = ins[ei][1]
    pattern = "".join("-" if o.split()[0] == "off" else "v" for o in ops[1:5])
    return pattern + (" compr" if any("compr" in o for o in ops) else "")


def census(disasm_dir, match):
    alpha, mrt4, shaders = Counter(), Counter(), Counter()
    for name in sorted(os.listdir(disasm_dir)):
        if not name.endswith(".s") or match not in name:
            continue
        path = os.path.join(disasm_dir, name)
        osg1 = output_signature_full(path, name[:-2])
        targets = {reg: mask for reg, _idx, sysv, mask, _rw in osg1 if sysv == SV_TARGET}
        n = len(targets)
        shaders[n] += 1
        with open(path, encoding="utf-8", errors="replace") as f:
            ins = parse(f)
        for ei, (op, ops) in enumerate(ins):
            if op != "exp" or not ops:
                continue
            if ops[0] == "mrt0":
                kind, crossed = mrt0_alpha(ins, ei)
                alpha[(n, kind, "crosses" if crossed else "straight")] += 1
            elif ops[0] == "mrt4":
                mrt4[(n, targets.get(4), mrt4_form(ins, ei))] += 1
    return shaders, alpha, mrt4


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count mrt0 alpha producers and mrt4 export forms (counts only).")
    ap.add_argument("disasm", help="the disassembly directory (<work>/shaders/dxbc/disasm)")
    ap.add_argument("--match", default="PS_DeferredTextured", help="substring a listing name must contain")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.disasm):
        print(f"ERROR: {args.disasm} is not a directory", file=sys.stderr)
        return 2
    shaders, alpha, mrt4 = census(args.disasm, args.match)
    print(f"listings {sum(shaders.values())}: " + ", ".join(f"{n} targets:{c}" for n, c in sorted(shaders.items())))
    for (n, kind, path), c in sorted(alpha.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1:])):
        print(f"mrt0 alpha, {n} targets: {kind} ({path}) {c}")
    for (n, mask, form), c in sorted(mrt4.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        mask_text = "none" if mask is None else f"0x{mask:x}"
        print(f"mrt4, {n} targets, OSG1 mask {mask_text}: {form} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
