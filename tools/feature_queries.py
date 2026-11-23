"""List the D3D12 feature queries a PE image makes: CheckFeatureSupport calls whose feature id and data size are
constants, by feature.

Usage:

    python -I tools/feature_queries.py <image> [--slot 13] [--feature 27] [--any-call] [--json] [--sites]

`ID3D12Device::CheckFeatureSupport(Feature, pFeatureSupportData, FeatureSupportDataSize)` is device vtable slot 13
(`call [reg+68h]`), with the feature id in edx and the data size in r9d. The console device keeps the PC positions
around it: slot 14 is `CreateDescriptorHeap` and slots 14-19 were measured at their PC shapes
(docs/d3d12_vulkan.md §8.2, row 2). docs/boot.md Q11 and Q22 ask which features the game queries; a trace answers
that only for the calls a run reaches, while this reads every call site in the file.

How it finds them (x86-64 images only, nothing is run):
1. Every `call qword ptr [reg+disp]` encoding with disp = 8 * slot in the executable sections, by byte pattern.
2. The .pdata function holding each one is decoded from its start with the decoder in tools/gdk_runtime_calls.py.
   Constants are followed through `mov reg,imm`, `xor reg,reg` / `sub reg,reg`, `lea reg,[reg+disp]` from a known
   register and register-to-register `mov`; any other write to a register forgets its value. A call, an unconditional
   jump, a return or a branch target inside the function forgets every value (another path can arrive there); a
   conditional jump does not (its fall-through path keeps them). --across-labels keeps the values at branch targets
   too: more of them are known, and some of those come from another path, so it is the upper bound of what a call
   can be shown to pass.
3. At the call, edx is the feature id and r9d the data size, each when it is known.
A pattern match the decoding does not land on (bytes inside another instruction, or data) is not counted as a call.

--any-call decodes every .pdata function and adds every call (direct or indirect) with a known edx and r9d: a
wrapper that takes the device, the feature and the size passes them this way. With --feature N only that feature
id is listed there. It takes longer (every function of the image is decoded).

What it does not rule out: a feature id or size that comes from memory, a parameter or a table at run time shows as
"not known", and the call is still counted. The PC data sizes are sizeof(D3D12_FEATURE_DATA_*) from the Windows SDK
10.0.26100 d3d12.h; a console structure may be larger (docs/d3d12_vulkan.md §4.1).

Output: per image, the slot calls by what is known at each (feature and size, feature only, size only, neither), the
features named, and whether each size is the PC size. --json prints the same as JSON. --sites adds each call's RVA:
an executable's RVAs are for reading the code on your own machine; don't post them.

Exit status 1 when an image can't be read, 2 for an image with no .pdata (nothing to decode from); 0 otherwise.
Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import struct
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdk_runtime_calls import _PREFIXES, RIP, Code, Insn, writes  # noqa: E402
from pe_inspect import PeFormatError, PeImage  # noqa: E402

CHECK_FEATURE_SUPPORT_SLOT = 13
RDX, R9 = 2, 9
OPTIONS5 = 27

# D3D12_FEATURE (Windows SDK 10.0.26100 d3d12.h): id -> (name, sizeof the PC D3D12_FEATURE_DATA_* structure).
# The sizes were printed by a program built against that header; None where the SDK has no structure of that name.
FEATURES = {
    0: ("D3D12_OPTIONS", 60), 1: ("ARCHITECTURE", 16), 2: ("FEATURE_LEVELS", 24), 3: ("FORMAT_SUPPORT", 12),
    4: ("MULTISAMPLE_QUALITY_LEVELS", 16), 5: ("FORMAT_INFO", 8), 6: ("GPU_VIRTUAL_ADDRESS_SUPPORT", 8),
    7: ("SHADER_MODEL", 4), 8: ("D3D12_OPTIONS1", 24), 10: ("PROTECTED_RESOURCE_SESSION_SUPPORT", 8),
    12: ("ROOT_SIGNATURE", 4), 16: ("ARCHITECTURE1", 20), 18: ("D3D12_OPTIONS2", 8), 19: ("SHADER_CACHE", 4),
    20: ("COMMAND_QUEUE_PRIORITY", 12), 21: ("D3D12_OPTIONS3", 20), 22: ("EXISTING_HEAPS", 4),
    23: ("D3D12_OPTIONS4", 12), 24: ("SERIALIZATION", 8), 25: ("CROSS_NODE", 8), 27: ("D3D12_OPTIONS5", 12),
    28: ("DISPLAYABLE", 8), 30: ("D3D12_OPTIONS6", 20), 31: ("QUERY_META_COMMAND", 56), 32: ("D3D12_OPTIONS7", 8),
    33: ("PROTECTED_RESOURCE_SESSION_TYPE_COUNT", 8), 34: ("PROTECTED_RESOURCE_SESSION_TYPES", 16),
    36: ("D3D12_OPTIONS8", 4), 37: ("D3D12_OPTIONS9", 24), 39: ("D3D12_OPTIONS10", 8), 40: ("D3D12_OPTIONS11", 4),
    41: ("D3D12_OPTIONS12", 12), 42: ("D3D12_OPTIONS13", 24), 43: ("D3D12_OPTIONS14", 12),
    44: ("D3D12_OPTIONS15", 8), 45: ("D3D12_OPTIONS16", 8), 46: ("D3D12_OPTIONS17", 8), 47: ("D3D12_OPTIONS18", 4),
    48: ("D3D12_OPTIONS19", 40), 49: ("D3D12_OPTIONS20", 8), 50: ("PREDICATION", 4),
    51: ("PLACED_RESOURCE_SUPPORT_INFO", 32), 52: ("HARDWARE_COPY", 4), 53: ("D3D12_OPTIONS21", 16),
    56: ("APPLICATION_SPECIFIC_DRIVER_STATE", 4), 57: ("BYTECODE_BYPASS_HASH_SUPPORTED", 4),
    61: ("SHADER_CACHE_ABI_SUPPORT", None),
}

_JCC = frozenset((*range(0x70, 0x80), *range(0x180, 0x190), 0xE0, 0xE1, 0xE2, 0xE3))
# VEX forms that write a general register (ModRM reg): andn, blsr/blsmsk/blsi, bzhi/pdep/pext, mulx, bextr/shlx, rorx
_VEX_GPR = frozenset((0x12F2, 0x12F3, 0x12F5, 0x12F6, 0x12F7, 0x13F0))


def feature_name(feature: int | None) -> str:
    if feature is None:
        return "not known"
    name = FEATURES.get(feature, (None, None))[0]
    return f"{name} ({feature})" if name else f"unknown feature {feature}"


def slot_pattern(slot: int) -> re.Pattern:
    """`FF /2` with a [base+disp] operand of disp 8*slot: mod 01 (disp8, when it fits) or mod 10 (disp32), with or
    without a SIB byte. A REX prefix before it, if any, is left to the decoder."""
    disp = 8 * slot
    forms = []
    if disp < 0x80:
        d8 = re.escape(bytes((disp,)))
        forms += [rb"[\x50-\x53\x55-\x57]" + d8, rb"\x54[\x00-\xff]" + d8]
    d32 = re.escape(struct.pack("<i", disp))
    forms += [rb"[\x90-\x93\x95-\x97]" + d32, rb"\x94[\x00-\xff]" + d32]
    return re.compile(rb"\xff(?:" + rb"|".join(forms) + rb")")


def is_slot_call(i: Insn, slot: int) -> bool:
    return (i.op == 0xFF and i.mod in (1, 2) and i.reg & 7 == 2 and i.base is not None and i.base != RIP
            and i.index is None and i.disp == 8 * slot)


def ends_path(i: Insn) -> bool:
    """A call, an unconditional jump or a return: what follows it is reached some other way."""
    return i.is_call() or i.ends_flow() or i.op in (0xCA, 0xCB, 0xCC, 0xCD, 0x10B)


def branch_targets(insns: list[Insn], begin: int, end: int) -> set[int]:
    out = set()
    for i in insns:
        if (i.op in _JCC or i.op in (0xE9, 0xEB)) and i.target is not None and begin <= i.target < end:
            out.add(i.target)
    return out


class Tracker:
    """The constants the registers hold along one path (32-bit values, as edx and r9d are read)."""

    def __init__(self, code: Code):
        self.code = code
        self.known: dict[int, int] = {}

    def imm(self, i: Insn, size: int) -> int:
        data, base = self.code.view(i.rva)
        return int.from_bytes(data[i.end - base - size:i.end - base], "little")

    def word_sized(self, i: Insn) -> bool:
        """An operand-size prefix (66) without REX.W: the instruction writes 16 bits and keeps the upper half."""
        if i.wide:
            return False
        data, base = self.code.view(i.rva)
        at = i.rva - base
        while at < i.end - base and data[at] in _PREFIXES:
            if data[at] == 0x66:
                return True
            at += 1
        return False

    def step(self, i: Insn) -> None:
        known = self.known
        op = i.op
        if op < 0x100 and self.word_sized(i):
            # mov dx, imm16 or xor dx, dx changes only the low half; the tracker keeps 32-bit values, so the
            # register is no longer known rather than wrongly known
            for r in writes(i):
                known.pop(r, None)
            if op in (0x29, 0x2B, 0x31, 0x33, 0x89, 0x8B, 0x8D) and i.reg is not None:
                known.pop(i.reg if op in (0x2B, 0x33, 0x8B, 0x8D) else i.rm, None)
            return
        if i.mod is None and 0xB8 <= op <= 0xBF:                          # mov r32/r64, imm
            reg = (op & 7) | ((i.rex & 1) << 3)
            known[reg] = self.imm(i, 8 if i.wide else 4) & 0xFFFFFFFF
            return
        if op == 0xC7 and i.mod == 3 and i.reg & 7 == 0:                 # mov r/m, imm32
            known[i.rm] = self.imm(i, 4)
            return
        if op in (0x29, 0x2B, 0x31, 0x33) and i.mod == 3 and i.reg == i.rm:  # sub/xor reg,reg
            known[i.reg] = 0
            return
        if op == 0x8D and i.mod != 3:                                    # lea reg, [base+disp]
            if i.base is not None and i.base != RIP and i.index is None and i.base in known:
                known[i.reg] = (known[i.base] + i.disp) & 0xFFFFFFFF
            else:
                known.pop(i.reg, None)
            return
        if op in (0x89, 0x8B) and i.mod == 3:                            # mov reg, reg
            dst, src = (i.rm, i.reg) if op == 0x89 else (i.reg, i.rm)
            if src in known:
                known[dst] = known[src]
            else:
                known.pop(dst, None)
            return
        for r in writes(i):
            known.pop(r, None)
        if op in _VEX_GPR and i.reg is not None:
            known.pop(i.reg, None)
        if op in (0x17E, 0x117E) and i.mod == 3:                         # movd/movq r/m, xmm
            known.pop(i.rm, None)


def decode_function(code: Code, begin: int, end: int) -> list[Insn] | None:
    """The function's instructions in order, up to where the decoder stops (data, or an encoding it doesn't know);
    None when not even the first one decodes."""
    out = []
    rva = begin
    while rva < end:
        i = code.decode(rva)
        if i is None:
            return out or None
        out.append(i)
        rva = i.end
    return out


def scan_function(code: Code, begin: int, end: int, slot: int, any_call: bool, sink,
                  across_labels: bool = False) -> bool:
    """Feed sink(kind, rva, edx, r9d) for the function's slot calls (kind 'slot') and, with any_call, every other
    call with both values known (kind 'direct' or 'indirect'). False when nothing could be decoded."""
    insns = decode_function(code, begin, end)
    if not insns:
        return False
    targets = set() if across_labels else branch_targets(insns, begin, end)
    t = Tracker(code)
    for i in insns:
        if i.rva in targets:
            t.known = {}
        if i.is_call():
            edx, r9 = t.known.get(RDX), t.known.get(R9)
            if is_slot_call(i, slot):
                sink("slot", i.rva, edx, r9)
            elif any_call and edx is not None and r9 is not None:
                sink("direct" if i.op == 0xE8 or i.base == RIP else "indirect", i.rva, edx, r9)
        if ends_path(i):
            t.known = {}
        else:
            t.step(i)
    return True


def scan(img: PeImage, slot: int = CHECK_FEATURE_SUPPORT_SLOT, any_call: bool = False,
         across_labels: bool = False) -> dict:
    code = Code(img)
    if not code.begins:
        return {"pdata": False}
    pattern = slot_pattern(slot)
    candidates = set()
    for base, data in code.sections:
        for m in pattern.finditer(data):
            candidates.add(base + m.start())
    functions = set()
    outside = 0
    for rva in candidates:
        k = bisect.bisect_right(code.begins, rva) - 1
        if k >= 0 and code.begins[k] <= rva < code.ends[k]:
            functions.add(k)
        else:
            outside += 1
    if any_call:
        functions = set(range(len(code.begins)))

    calls, others = [], []
    undecoded = 0

    def sink(kind, rva, edx, r9):
        (calls if kind == "slot" else others).append({"kind": kind, "rva": rva, "feature": edx, "size": r9})

    for k in sorted(functions):
        if not scan_function(code, code.begins[k], code.ends[k], slot, any_call, sink, across_labels):
            undecoded += 1
    return {"pdata": True, "slot": slot, "outside_pdata": outside, "functions": len(functions),
            "undecoded_functions": undecoded, "calls": calls, "other_calls": others}


def size_note(feature: int | None, size: int | None) -> str:
    if feature is None or size is None:
        return ""
    pc = FEATURES.get(feature, (None, None))[1]
    if pc is None:
        return "no PC size to compare"
    return "the PC size" if size == pc else f"PC size {pc}"


def summarize(path: str, found: dict, feature_filter: int | None) -> dict:
    calls = found["calls"]
    both = Counter((c["feature"], c["size"]) for c in calls if c["feature"] is not None and c["size"] is not None)
    feature_only = Counter(c["feature"] for c in calls if c["feature"] is not None and c["size"] is None)
    size_only = Counter(c["size"] for c in calls if c["feature"] is None and c["size"] is not None)
    neither = sum(1 for c in calls if c["feature"] is None and c["size"] is None)
    others = Counter((c["kind"], c["feature"], c["size"]) for c in found["other_calls"]
                     if feature_filter is None or c["feature"] == feature_filter)
    r = {
        "file": os.path.basename(path), "slot": found["slot"], "across_labels": found.get("across_labels", False),
        "slot_calls": len(calls),
        "functions_decoded": found["functions"] - found["undecoded_functions"],
        "functions_undecoded": found["undecoded_functions"], "matches_outside_pdata": found["outside_pdata"],
        "feature_and_size": [{"feature": f, "name": feature_name(f), "size": s, "note": size_note(f, s), "calls": n}
                             for (f, s), n in sorted(both.items())],
        "feature_only": [{"feature": f, "name": feature_name(f), "calls": n} for f, n in sorted(feature_only.items())],
        "size_only": [{"size": s, "calls": n} for s, n in sorted(size_only.items())],
        "neither": neither,
        "options5_calls": sum(1 for c in calls if c["feature"] == OPTIONS5),
    }
    if found.get("any_call"):
        # Outside the slot, edx and r9d are only numbers: a call is a feature query only if r9d is the PC data size
        # of the feature id in edx (or a console size nobody has measured, which this cannot tell).
        r["other_calls"] = [{"kind": k, "edx": f, "r9d": s, "pc_size_of_feature": pc_size_matches(f, s), "calls": n}
                            for (k, f, s), n in sorted(others.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0]))]
    return r


def pc_size_matches(feature: int, size: int) -> bool:
    return FEATURES.get(feature, (None, None))[1] == size


def to_text(r: dict, sites: list | None) -> str:
    lines = [f"{r['file']}: {r['slot_calls']} call(s) through slot {r['slot']} (+0x{8 * r['slot']:x}) in "
             f"{r['functions_decoded']} decoded function(s)"
             + (", values kept across branch targets" if r["across_labels"] else "")
             + (f"; {r['functions_undecoded']} function(s) not decoded" if r["functions_undecoded"] else "")
             + (f"; {r['matches_outside_pdata']} match(es) outside .pdata, not read"
                if r["matches_outside_pdata"] else "")]
    lines.append(f"  feature id and size known: {sum(x['calls'] for x in r['feature_and_size'])}")
    for x in r["feature_and_size"]:
        lines.append(f"    {x['name']}, {x['size']} bytes ({x['note']}): {x['calls']}")
    lines.append(f"  feature id known, size not: {sum(x['calls'] for x in r['feature_only'])}")
    for x in r["feature_only"]:
        lines.append(f"    {x['name']}: {x['calls']}")
    lines.append(f"  size known, feature id not: {sum(x['calls'] for x in r['size_only'])}")
    for x in r["size_only"]:
        lines.append(f"    {x['size']} bytes: {x['calls']}")
    lines.append(f"  neither known: {r['neither']}")
    lines.append(f"  D3D12_OPTIONS5 ({OPTIONS5}) in edx: {r['options5_calls']}")
    if "other_calls" in r:
        lines.append(f"  other calls with edx and r9d known (--any-call): {sum(x['calls'] for x in r['other_calls'])}, "
                     f"with r9d the PC data size of the feature id in edx: "
                     f"{sum(x['calls'] for x in r['other_calls'] if x['pc_size_of_feature'])}")
        for x in r["other_calls"]:
            mark = f" (the PC size of {feature_name(x['edx'])})" if x["pc_size_of_feature"] else ""
            lines.append(f"    {x['kind']}: edx {x['edx']}, r9d {x['r9d']}{mark}: {x['calls']}")
    if sites is not None:
        lines.append("  sites (RVAs stay on your machine):")
        for c in sorted(sites, key=lambda c: c["rva"]):
            lines.append(f"    0x{c['rva']:x} {c['kind']} edx={c['feature']} r9d={c['size']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("image", nargs="+")
    ap.add_argument("--slot", type=int, default=CHECK_FEATURE_SUPPORT_SLOT,
                    help="the vtable slot to read (default: %(default)s, CheckFeatureSupport)")
    ap.add_argument("--any-call", action="store_true",
                    help="also list every other call with edx and r9d known (decodes every function)")
    ap.add_argument("--feature", type=lambda s: int(s, 0), help="with --any-call, list only this feature id")
    ap.add_argument("--across-labels", action="store_true",
                    help="keep known values at branch targets (an upper bound: some come from another path)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--sites", action="store_true", help="print each call's RVA (keep them on your machine)")
    a = ap.parse_args(argv)
    if a.slot < 0:
        ap.error("--slot must be 0 or more")

    rc = 0
    out = []
    for path in a.image:
        try:
            with open(path, "rb") as f:
                img = PeImage(f.read())
            found = scan(img, a.slot, a.any_call, a.across_labels)
        except (PeFormatError, OSError, struct.error) as e:
            print(f"{path}: not a readable PE image ({e})", file=sys.stderr)
            rc = 1
            continue
        if not found["pdata"]:
            print(f"{path}: no .pdata function table, nothing to decode from", file=sys.stderr)
            rc = max(rc, 2)
            continue
        found["any_call"] = a.any_call
        found["across_labels"] = a.across_labels
        r = summarize(path, found, a.feature)
        sites = None
        if a.sites:
            sites = found["calls"] + [c for c in found["other_calls"]
                                      if a.feature is None or c["feature"] == a.feature]
            r["sites"] = [dict(c, rva=f"0x{c['rva']:x}") for c in sites]
        out.append((r, sites))
    if a.json:
        print(json.dumps([r for r, _ in out], indent=1))
    else:
        for r, sites in out:
            print(to_text(r, sites))
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
