"""How many shaders of a disassembled corpus use each instruction, all encodings of it counted as one.

    python -I tools/op_reach.py <work>/shaders/dxbc/disasm [OP ...] [--index <work>/shaders/dxbc/index.json]
                                [--json OUT]

Reads every `.s` listing in the folder (shader_disasm.py output: an instruction line starts with a tab). An
instruction's base name drops the encoding suffix LLVM prints (`_e32`, `_e64`, `_sdwa`, `_dpp`, the same ones
rdna2spirv's `base_op()` strips), so `v_cndmask_b32_e32` and `v_cndmask_b32_e64` count as `v_cndmask_b32`, and a
shader using both counts once. tools/shader_profile.py counts the printed names instead.

Only the program counts: lines after a listing's last `s_endpgm` are data words LLVM decoded anyway (some
listings run past the program; on 1.0.2.1 every `s_cmov_b32` in a listing is there), so they are left out and
their total is reported separately as `after_last_endpgm`. A listing with no `s_endpgm` counts whole.

With OP names, prints those only, a name no listing uses with 0 (that is the answer to "does the game reach this
rule"). Without, prints every base name, most-used first. `--index` (the carve index) adds the count per stage.

Standard library only. Prints and writes counts and instruction names only, nothing from the listings.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

INSTRUCTION = re.compile(r"^\t([a-z][a-z0-9_]*)", re.M)
ENCODING = re.compile(r"_(e32|e64|sdwa|dpp)$")


def base_name(op):
    """The instruction without the encoding suffix LLVM prints."""
    return ENCODING.sub("", op)


def listing_ops(text):
    """(base name -> instructions up to the last s_endpgm, instructions after it) for one listing."""
    ops = [base_name(op) for op in INSTRUCTION.findall(text)]
    end = max((i for i, op in enumerate(ops) if op == "s_endpgm"), default=len(ops) - 1)
    return Counter(ops[:end + 1]), len(ops) - end - 1


def stages_of(index_path):
    """Listing stem -> stage from the carve index."""
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, list) or not all(isinstance(e, dict) for e in index):
        raise ValueError("index.json is not a list of shader entries")
    return {os.path.splitext(e["file"])[0]: e.get("stage") for e in index if isinstance(e.get("file"), str)}


def census(disasm_dir, stages=None):
    """Counts over every .s file: shaders and instructions per base name, and shaders per base name per stage."""
    names = sorted(n for n in os.listdir(disasm_dir) if n.endswith(".s"))
    shaders, instructions = Counter(), Counter()
    by_stage = defaultdict(Counter)
    listings_by_stage = Counter()
    after = 0
    for name in names:
        with open(os.path.join(disasm_dir, name), encoding="utf-8", errors="replace") as f:
            ops, tail = listing_ops(f.read())
        after += tail
        shaders.update(ops.keys())
        instructions.update(ops)
        if stages is not None:
            stage = stages.get(name[:-2]) or "unindexed"
            listings_by_stage[stage] += 1
            by_stage[stage].update(ops.keys())
    return {"listings": len(names), "listings_by_stage": dict(sorted(listings_by_stage.items())),
            "after_last_endpgm": after, "shaders": shaders, "instructions": instructions, "by_stage": by_stage}


def report(result, ops=None):
    """Rows of (base name, shaders, instructions, {stage: shaders}) for the selected names, or all of them."""
    chosen = [base_name(op) for op in ops] if ops else sorted(result["shaders"],
                                                             key=lambda o: (-result["shaders"][o], o))
    return [{"op": op, "shaders": result["shaders"].get(op, 0), "instructions": result["instructions"].get(op, 0),
             "by_stage": {stage: c[op] for stage, c in sorted(result["by_stage"].items()) if c.get(op)}}
            for op in chosen]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disasm", help="folder of .s listings (<work>/shaders/dxbc/disasm)")
    ap.add_argument("ops", nargs="*", help="base instruction names to report (default: all)")
    ap.add_argument("--index", help="carve index.json: also count per stage")
    ap.add_argument("--json", help="also write the rows as JSON")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.disasm):
        print(f"ERROR: {a.disasm} is not a folder", file=sys.stderr)
        return 2
    try:
        stages = stages_of(a.index) if a.index else None
        result = census(a.disasm, stages)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if not result["listings"]:
        print(f"ERROR: no .s listing in {a.disasm}", file=sys.stderr)
        return 1
    rows = report(result, a.ops)
    print(f"{result['listings']} listings" + (f" {result['listings_by_stage']}" if stages is not None else "") +
          f", {result['after_last_endpgm']} decoded instructions after the last s_endpgm left out")
    for row in rows:
        stage = f"  {row['by_stage']}" if stages is not None and row["by_stage"] else ""
        print(f"{row['op']:32s} {row['shaders']:7d} shaders {row['instructions']:9d} instructions{stage}")
    if a.json:
        try:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"listings": result["listings"], "listings_by_stage": result["listings_by_stage"],
                           "after_last_endpgm": result["after_last_endpgm"], "rows": rows}, f, indent=1)
        except OSError as e:
            print(f"ERROR: cannot write {a.json}: {e.strerror or e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
