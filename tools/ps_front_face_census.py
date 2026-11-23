"""Count first uses of the entry FRONT_FACE VGPR by indexed PS stem.

Run python -I tools/ps_front_face_census.py <dxbc> [--disasm <listing directory>].
Both direct successors are considered regardless of branch feasibility. A first
syntactic read ends that path. The tracked lanes are those active at entry.
After EXEC may have changed, a destination write ends analysis of that path
with uncertainty. Read kinds and uncertainty reasons each count distinct stems
and can overlap if different paths have different outcomes.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fssm_header
import rdna2spirv as isa
from ps_interp_modes import chunk_table, isg1_elements
from shader_ir import parse_file


def input_register(data):
    chunks = chunk_table(data)
    if b"ISG1" not in chunks:
        raise ValueError("missing signature")
    offset, size = chunks[b"ISG1"]
    if size < 8 or offset + 8 + size > len(data):
        raise ValueError("invalid signature")
    count, first = struct.unpack_from("<II", data, offset + 8)
    if first < 8 or first + count * 32 > size:
        raise ValueError("invalid signature")
    rows = isg1_elements(data)
    if not any(row["system_value"] == 9 for row in rows):
        return None
    positions = [row["rw"] for row in rows if row["system_value"] == 1]
    if len(positions) > 1 or any(mask > 15 for mask in positions):
        raise ValueError("ambiguous position layout")
    return 2 + sum(mask.bit_count() for mask in positions)


def first_uses(info, native, reg):
    pc = start = info["code_start"]
    end = info["code_end"]
    if end is None or len(native) < 0xc4 or start != struct.unpack_from("<I", native, 0xc0)[0] + 0x28:
        return set(), {"unknown entry"}
    instructions = {}
    for ins in info["insns"]:
        enc = bytes.fromhex(ins["enc"] or "")
        nxt = pc + len(enc)
        if len(enc) == 0 or len(enc) % 4 or nxt > end or native[pc:nxt] != enc:
            break  # Later addresses cannot be established from this listing.
        instructions[pc] = ins, nxt
        pc = nxt

    reads, unresolved, seen = set(), set(), set()
    pending = [(start, False)]
    while pending:
        pc, masked = pending.pop()
        if (pc, masked) in seen:
            continue
        seen.add((pc, masked))
        if pc not in instructions:
            unresolved.add("unverified instruction address")
            continue
        ins, nxt = instructions[pc]
        op, args = ins["op"], ins["args"]
        if op in isa.BRANCHES:
            if len(args) == 1 and args[0]["k"] == "imm":
                displacement = (args[0]["v"] + 32768) % 65536 - 32768
                pending.append((nxt + 4 * displacement, masked))
            else:
                unresolved.add("unknown branch target")
            if op != "s_branch":
                pending.append((nxt, masked))
            continue
        if op == "s_endpgm":
            continue
        if not known_fallthrough(op):
            unresolved.add("unknown flow or addressing")
            continue
        source, dest = isa.reg_reads_writes([ins])
        if reg in source:
            kind = op
            if op.startswith("v_cmp"):
                zero = any(a["k"] == "imm" and a["v"] == 0 and not a.get("neg") for a in args[-2:])
                kind += " immediate zero" if zero else " other operands"
            reads.add(kind)
            continue
        if reg in dest:
            if masked or op.startswith("v_writelane") or op.endswith(("_sdwa", "_dpp")):
                unresolved.add("possible lane-partial write")
            elif ins["mods"] or isa.base_op(op) not in FULL_WRITES:
                unresolved.add("unclassified destination")
            continue
        pending.append((nxt, masked or isa.writes_exec(ins)))
    return reads, unresolved


def listing_first_read(insns, reg):
    """Historical selection rule: first printed register access, ignoring control flow and EXEC."""
    for ins in insns:
        source, dest = isa.reg_reads_writes([ins])
        if reg in source:
            return True
        if reg in dest:
            break
    return False


# Only these ordinary full-dword destinations end a path with an established kill.
FULL_WRITES = {"v_mov_b32", "v_interp_p1_f32", "v_mul_f32", "v_add_f32", "v_fma_f32",
               "v_cndmask_b32", "v_sub_f32", "v_rcp_f32", "v_and_b32", "v_lshlrev_b32"}


def known_fallthrough(op):
    if "movrel" in op or "gpr_idx" in op or op == "s_code_end":
        return False
    return (op in isa.NOOPS or op == "exp" or op.startswith((
        "v_", "ds_", "buffer_", "tbuffer_", "image_", "global_", "flat_", "scratch_",
        "s_mov_", "s_load_", "s_buffer_load_", "s_cmp", "s_bitcmp", "s_add", "s_sub",
        "s_mul", "s_min", "s_max", "s_cselect", "s_and", "s_or", "s_xor", "s_not",
        "s_nand", "s_nor", "s_xnor", "s_lsh", "s_ashr", "s_bfe", "s_bfm", "s_brev",
        "s_bcnt", "s_ff", "s_getpc", "s_wqm", "s_quadmask", "s_pack", "s_getreg")))


def census(base, listings):
    stages = fssm_header.stage_index(str(base))
    if not stages:
        raise ValueError("missing usable stage index")
    totals = Counter(indexed_stems=len(stages), indexed_ps=0, declarations=0, skipped=0,
                     listing_first_access_is_read=0, listings_examined=0,
                     resolved_reader=0, resolved_without_read=0, unresolved=0)
    kinds, reasons = Counter(), Counter()
    for stem, stage in stages.items():
        if stage != "ps":
            continue
        totals["indexed_ps"] += 1
        try:
            reg = input_register((base / (stem + ".dxbc")).read_bytes())
        except (OSError, ValueError, struct.error):
            totals["skipped"] += 1
            continue
        if reg is None:
            continue
        totals["declarations"] += 1
        try:
            blob = (base / "fssm" / (stem + ".fssm")).read_bytes()
            info = parse_file(listings / (stem + ".s"))
            reads, unknown = first_uses(info, blob, reg)
            totals["listings_examined"] += 1
            totals["listing_first_access_is_read"] += listing_first_read(info["insns"], reg)
        except (OSError, ValueError, struct.error):
            reads, unknown = set(), {"missing or malformed sidecar"}
        kinds.update(reads)
        reasons.update(unknown)
        totals["unresolved" if unknown else "resolved_reader" if reads else "resolved_without_read"] += 1
    return dict(totals, shaders_per_read_kind=kinds, shaders_per_unresolved_reason=reasons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxbc", type=Path)
    parser.add_argument("--disasm", type=Path)
    args = parser.parse_args()
    print(json.dumps(census(args.dxbc, args.disasm or args.dxbc / "disasm"), indent=2, sort_keys=True))
