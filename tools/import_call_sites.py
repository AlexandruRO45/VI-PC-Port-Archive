"""Count and locate the direct calls to chosen imports of an x86-64 PE image, from the file alone.

Usage:
    python -I tools/import_call_sites.py <image> [--import DLL!NAME ...] [--dll DLL ...] [--json]

An MSVC image calls an imported function through a thunk, `jmp qword [rip+disp32]` through the import's address-table
slot, and reaches the thunk with `call rel32` (or `jmp rel32` as a tail call). This finds each wanted import's thunk,
then every `E8`/`E9 rel32` in an executable section whose target is that thunk. Nothing is run and nothing is
disassembled, so this is a byte pattern: a stray `E8` or `E9` inside another instruction that happens to land on a
thunk would be counted too, which is rare, and a call that goes through a register (`call rax` after a load of the
slot) is not seen. Read the counts as a lower bound for indirect uses and an upper bound for direct ones.

A function that ends in a tail call to a `dllimport` function jumps through the slot itself, and MSVC writes that jump
with a REX.W byte, `48 FF 25 disp32` (Wine's x64 unwinder accepts no other REX byte on the jump that ends an epilog).
So each `FF 25` through a wanted slot is classified from what the bytes show:

- an `E8`/`E9` targets the `FF`: a thunk (the byte before it does not matter);
- the byte before is `48` and an `E8`/`E9` targets that byte: a thunk that starts at the `48`;
- the byte before is `48` and nothing targets either: a tail jump, listed in `iat_jmp` at the RVA of the `48` and
  counted as a site;
- no `48` before it and nothing targets it: a thunk nothing calls, as before.

The `48` rule is a heuristic, not a decoding. An uncalled thunk that happens to follow a `48` byte is reported as a tail
jump, and a tail jump written without the prefix is reported as an uncalled thunk; the output does not mark either as
uncertain, so `iat_jmp` is a bound like the other counts.

With no `--import` and no `--dll`, every import is listed. An import named `#N` is an ordinal import. Output is counts
and RVAs only, no instruction bytes (docs/runtime_imports.md, "The retail executable (RT-5)").
"""
import argparse
import json
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import pe_inspect  # noqa: E402

JMP_INDIRECT = re.compile(b"\xff\x25")
REL_BRANCH = re.compile(b"[\xe8\xe9]")
REX_W = 0x48  # the prefix of `rex.w jmp [rip+disp32]`, a tail call through an address-table slot


def import_slots(img):
    """{IAT slot RVA: 'dll!name'} for every import of a PE32+ image."""
    rva, size = img.directory(1)
    slots = {}
    if not rva or not size:
        return slots
    for i in range(size // 20):
        at = img.offset(rva + i * 20, 20)
        if at is None:
            break
        oft, _stamp, _forwarder, dll_rva, iat_rva = struct.unpack_from("<IIIII", img.data, at)
        if not dll_rva:
            break
        dll = img.string_at(dll_rva)
        lookup = oft or iat_rva
        if dll is None or not lookup:
            continue
        for j, name in enumerate(img.thunk_names(lookup)):
            slots[iat_rva + j * 8] = f"{dll}!{name}"
    return slots


def call_sites(data, wanted=None, dlls=None):
    """Per import: its thunk RVAs, the RVAs of the `call` and `jmp` instructions that target them, and in `iat_jmp`
    the RVAs of tail jumps through its slot (`48 FF 25`, reported at the `48`; the module docstring has the rule).

    `wanted` is a set of 'dll!name' (compared case-insensitively on the dll), `dlls` a set of dll names; with both
    None every import is scanned.
    """
    img = pe_inspect.PeImage(data)
    if not img.is64:
        raise pe_inspect.PeFormatError("only PE32+ (x86-64) images are scanned")
    lower_wanted = {w.lower() for w in wanted} if wanted else None
    lower_dlls = {d.lower() for d in dlls} if dlls else None

    def selected(key):
        dll, _, name = key.partition("!")
        if lower_wanted is None and lower_dlls is None:
            return True
        return (lower_wanted is not None and f"{dll.lower()}!{name}".lower() in lower_wanted) or \
               (lower_dlls is not None and dll.lower() in lower_dlls)

    slots = {rva: key for rva, key in import_slots(img).items() if selected(key)}
    code = [(s["rva"], img.section_data(s)) for s in img.sections if s["exec"]]

    jumps = {}  # RVA of each `FF 25` through a slot: (import, whether a `48` byte comes right before it)
    for rva, blob in code:
        for m in JMP_INDIRECT.finditer(blob):
            s = m.start()
            if s + 6 > len(blob):
                continue
            target = rva + s + 6 + struct.unpack_from("<i", blob, s + 2)[0]
            if target in slots:
                jumps[rva + s] = (slots[target], s > 0 and blob[s - 1] == REX_W)

    # Where an `FF 25`, or the `48` before one, could start an instruction that an `E8`/`E9` reaches.
    starts = set(jumps) | {at - 1 for at, (_key, prefixed) in jumps.items() if prefixed}
    branches = {}  # branch target: [(branch RVA, opcode)]
    for rva, blob in code:
        for m in REL_BRANCH.finditer(blob):
            s = m.start()
            if s + 5 > len(blob):
                continue
            target = rva + s + 5 + struct.unpack_from("<i", blob, s + 1)[0]
            if target in starts:
                branches.setdefault(target, []).append((rva + s, blob[s]))

    out = {key: {"thunks": [], "call": [], "jmp": [], "iat_jmp": []} for key in slots.values()}
    for at, (key, prefixed) in jumps.items():
        if at in branches or not prefixed:
            start = at  # a thunk: something calls it, or nothing marks it as a tail jump
        elif at - 1 in branches:
            start = at - 1  # a thunk that starts with the `48`
        else:
            out[key]["iat_jmp"].append(at - 1)
            continue
        out[key]["thunks"].append(start)
        for site, opcode in branches.get(start, ()):
            out[key]["call" if opcode == 0xE8 else "jmp"].append(site)
    for entry in out.values():
        for field in entry.values():
            field.sort()
    return {"image_base": img.image_base, "imports": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count the direct calls to chosen imports of an x86-64 PE image.")
    ap.add_argument("image")
    ap.add_argument("--import", dest="imports", action="append", default=[], metavar="DLL!NAME",
                    help="an import to look for (repeatable); '#N' names an ordinal import")
    ap.add_argument("--dll", dest="dlls", action="append", default=[], metavar="DLL",
                    help="every import of this DLL (repeatable)")
    ap.add_argument("--json", action="store_true", help="print JSON (RVAs in hex) instead of the table")
    args = ap.parse_args(argv)
    try:
        with open(args.image, "rb") as f:
            data = f.read()
        result = call_sites(data, set(args.imports) or None, set(args.dlls) or None)
    except (OSError, pe_inspect.PeFormatError) as e:
        print(f"{os.path.basename(sys.argv[0])}: {e}", file=sys.stderr)
        return 2
    imports = result["imports"]
    if args.json:
        printable = {k: {f: [hex(x) for x in v] for f, v in e.items()} for k, e in imports.items()}
        print(json.dumps({"image_base": hex(result["image_base"]), "imports": printable}, indent=1))
        return 0

    def sites(e):
        return len(e["call"]) + len(e["jmp"]) + len(e["iat_jmp"])

    print(f"{'import':56s} {'thunks':>6s} {'call':>5s} {'jmp':>4s} {'iat_jmp':>7s}")
    for key, e in sorted(imports.items(), key=lambda kv: (-sites(kv[1]), kv[0])):
        print(f"{key:56s} {len(e['thunks']):6d} {len(e['call']):5d} {len(e['jmp']):4d} {len(e['iat_jmp']):7d}")
    total = sum(sites(e) for e in imports.values())
    print(f"{len(imports)} import(s), {total} call site(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
