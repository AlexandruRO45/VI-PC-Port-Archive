#!/usr/bin/env python3
"""load_config: what a PE32+ image's load-config directory expects from the Windows loader that gta6rt doesn't do.

`runtime/src/pe_loader.cpp` maps an image, relocates it, binds its imports, registers `.pdata` and runs TLS, but never
reads the load-config directory (`IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG`, index 10). Windows reads it in three ways that a
mapped image therefore does without:

- **Process settings.** For the process's own executable only, Windows applies `GlobalFlagsSet`/`Clear`,
  `CriticalSectionDefaultTimeout`, the heap thresholds, `ProcessHeapFlags`, `ProcessAffinityMask` and `CSDVersion`.
  Under gta6rt the process image is gta6rt.exe, so the game's values are ignored. `DependentLoadFlags` steers how
  Windows loads an image's own imports, which gta6rt binds itself.
- **Pointer slots the OS may overwrite.** The Control Flow Guard / XFG / RFG / guarded-memcpy check and dispatch
  pointers. The linker fills each slot with a default in the image's own code (a no-op check, a `jmp rax`
  dispatch), so an image runs with the defaults; a slot that holds 0 crashes on its first call unless a loader
  fills it.
- **Tables the OS registers or applies.** The `/GS` security cookie (randomised by Windows, else by the CRT when it
  still holds the default), EH-continuation targets (checked when the process enforces CET shadow stacks), dynamic
  value relocations (rewritten by the Windows loader) and enclave configuration.

Standard library only; reuses `pe_inspect.PeImage`. Run with `python -I tools/load_config.py <image>... [--json]`.
Prints one block per image: the values above, then `attention:` lines for what a mapped image needs that gta6rt does
not provide (a zero pointer slot, dynamic relocations, an enclave, non-zero process settings). Exit status: 0 with no
`attention:` line, 1 with one, 2 when an input can't be read (one ERROR line). It prints field values and section
names only, no code or data of the image.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_inspect  # noqa: E402

LOAD_CONFIG = 10
DEFAULT_COOKIE_64 = 0x00002B992DDFA232  # the value the MSVC linker stores; Windows or the CRT replaces it at start

# IMAGE_LOAD_CONFIG_DIRECTORY64 offsets (winnt.h layout; a field exists only when it ends within `Size`).
PROCESS_FIELDS = (  # (offset, size, name): applied by Windows to the process executable only
    (12, 4, "GlobalFlagsClear"), (16, 4, "GlobalFlagsSet"), (20, 4, "CriticalSectionDefaultTimeout"),
    (24, 8, "DeCommitFreeBlockThreshold"), (32, 8, "DeCommitTotalFreeThreshold"), (48, 8, "MaximumAllocationSize"),
    (56, 8, "VirtualMemoryThreshold"), (64, 8, "ProcessAffinityMask"), (72, 4, "ProcessHeapFlags"),
    (76, 2, "CSDVersion"),
)
DEPENDENT_LOAD_FLAGS = 78
SECURITY_COOKIE = 88
POINTER_SLOTS = (  # (offset, name): VA of a slot holding a function pointer the OS may replace
    (112, "GuardCFCheckFunctionPointer"), (120, "GuardCFDispatchFunctionPointer"),
    (216, "GuardRFFailureRoutineFunctionPointer"), (232, "GuardRFVerifyStackPointerFunctionPointer"),
    (280, "GuardXFGCheckFunctionPointer"), (288, "GuardXFGDispatchFunctionPointer"),
    (296, "GuardXFGTableDispatchFunctionPointer"), (312, "GuardMemcpyFunctionPointer"),
)
GUARD_FLAGS = 144
CHPE_METADATA = 200
DYN_RELOC_OFFSET, DYN_RELOC_SECTION = 224, 228
ENCLAVE_CONFIG = 248
EH_CONT_TABLE, EH_CONT_COUNT = 264, 272

GUARD_FLAG_NAMES = (  # winnt.h IMAGE_GUARD_* (the top nibble is the function-table stride)
    (0x100, "CF_INSTRUMENTED"), (0x200, "CFW_INSTRUMENTED"), (0x400, "CF_FUNCTION_TABLE_PRESENT"),
    (0x800, "SECURITY_COOKIE_UNUSED"), (0x1000, "PROTECT_DELAYLOAD_IAT"),
    (0x2000, "DELAYLOAD_IAT_IN_ITS_OWN_SECTION"), (0x4000, "CF_EXPORT_SUPPRESSION_INFO_PRESENT"),
    (0x8000, "CF_ENABLE_EXPORT_SUPPRESSION"), (0x10000, "CF_LONGJUMP_TABLE_PRESENT"),
    (0x20000, "RF_INSTRUMENTED"), (0x40000, "RF_ENABLE"), (0x80000, "RF_STRICT"), (0x100000, "RETPOLINE_PRESENT"),
    (0x400000, "EH_CONTINUATION_TABLE_PRESENT"), (0x800000, "XFG_ENABLED"), (0x1000000, "CASTGUARD_PRESENT"),
    (0x2000000, "MEMCPY_PRESENT"),
)


class LoadConfigError(ValueError):
    """The image can't be read as a PE32+ with a load-config directory this tool understands."""


def guard_flag_names(flags: int) -> list[str]:
    names = [n for bit, n in GUARD_FLAG_NAMES if flags & bit]
    stride = flags >> 28
    if stride:
        names.append("FUNCTION_TABLE_STRIDE_%d" % stride)
    unknown = flags & 0x0FFFFFFF & ~sum(bit for bit, _ in GUARD_FLAG_NAMES)
    if unknown:
        names.append("unknown 0x%x" % unknown)
    return names


def section_of(img, rva: int) -> dict | None:
    for s in img.sections:
        if s["rva"] <= rva < s["rva"] + max(s["vsize"], s["rawsize"]):
            return s
    return None


def read_slot(img, va: int) -> tuple[str, int | None, str | None]:
    """(kind, value, section name) for the 8-byte slot at `va`: kind is 'file' (value read from the file), 'virtual'
    (inside a section but past its file bytes, so 0 at load), or 'outside' (not in the image)."""
    rva = va - img.image_base
    if not 0 <= rva < img.size_of_image or section_of(img, rva) is None:
        return "outside", None, None
    at = img.offset(rva, 8)
    if at is None:
        return "virtual", 0, section_of(img, rva)["name"]
    return "file", struct.unpack_from("<Q", img.data, at)[0], section_of(img, rva)["name"]


def inspect(data: bytes) -> dict:
    try:
        img = pe_inspect.PeImage(data)
    except pe_inspect.PeFormatError as e:
        raise LoadConfigError("not a PE image: %s" % e) from None
    if not img.is64:
        raise LoadConfigError("a 32-bit image; this tool reads the PE32+ layout only")
    rva, size = img.directory(LOAD_CONFIG)
    out = {"present": bool(rva), "attention": []}
    if not rva:
        return out
    at = img.offset(rva, 4)
    if at is None:
        raise LoadConfigError("the load-config directory at RVA 0x%x has no file bytes" % rva)
    declared = struct.unpack_from("<I", data, at)[0]
    avail = len(data) - at
    if declared < 4 or declared > avail:
        raise LoadConfigError("load-config Size %d does not fit the file" % declared)
    out["size"] = declared
    lc = data[at:at + declared]

    def field(off, width):
        if off + width > declared:
            return None
        return int.from_bytes(lc[off:off + width], "little")

    process = {name: v for off, w, name in PROCESS_FIELDS if (v := field(off, w))}
    out["process_settings"] = process
    for name, v in process.items():
        out["attention"].append("%s = 0x%x is a process setting; under gta6rt the process image is gta6rt.exe, so "
                                "it is not applied" % (name, v))
    dlf = field(DEPENDENT_LOAD_FLAGS, 2)
    if dlf:
        out["dependent_load_flags"] = dlf

    cookie_va = field(SECURITY_COOKIE, 8)
    if cookie_va:
        kind, value, sect = read_slot(img, cookie_va)
        out["security_cookie"] = {"slot": kind, "section": sect,
                                  "default": value == DEFAULT_COOKIE_64 if kind == "file" else None}

    slots = {}
    for off, name in POINTER_SLOTS:
        va = field(off, 8)
        if not va:
            continue
        kind, value, sect = read_slot(img, va)
        if kind == "outside":
            state = "slot outside the image"
        elif not value:
            state = "zero"
        else:
            target = section_of(img, value - img.image_base) if value >= img.image_base else None
            if target is None:
                state = "points outside the image"
            else:
                state = "own code (%s)" % target["name"] if target["exec"] else "own data (%s)" % target["name"]
        slots[name] = state
        if state == "zero" or state.startswith(("points outside", "slot outside")):
            out["attention"].append("%s: %s; a call through it needs a loader that fills the slot" % (name, state))
    out["pointer_slots"] = slots

    flags = field(GUARD_FLAGS, 4)
    if flags is not None:
        out["guard_flags"] = {"value": flags, "names": guard_flag_names(flags)}
    eh = field(EH_CONT_TABLE, 8)
    if eh:
        out["eh_continuation_targets"] = field(EH_CONT_COUNT, 8) or 0
    if field(CHPE_METADATA, 8):
        out["attention"].append("CHPE metadata present (a hybrid image); gta6rt maps x64 images only")
    dyn_off, dyn_sec = field(DYN_RELOC_OFFSET, 4), field(DYN_RELOC_SECTION, 2)
    if dyn_sec:
        out["dynamic_relocations"] = {"section": dyn_sec, "offset": dyn_off}
        out["attention"].append("dynamic value relocations (section %d, offset 0x%x): Windows rewrites these at "
                                "load, gta6rt does not" % (dyn_sec, dyn_off or 0))
    if field(ENCLAVE_CONFIG, 8):
        out["attention"].append("an enclave configuration is present; gta6rt has no enclave support")
    return out


def describe(name: str, r: dict) -> list[str]:
    if not r["present"]:
        return ["%s: no load-config directory" % name]
    lines = ["%s: load config, %d bytes" % (name, r["size"])]
    c = r.get("security_cookie")
    if c:
        state = {True: "the linker's default (the CRT or Windows sets it at start)", False: "not the default",
                 None: "not in the file"}[c["default"]]
        lines.append("  security cookie: %s, in %s" % (state, c["section"] or "no section"))
    for n, s in r["pointer_slots"].items():
        lines.append("  %s: %s" % (n, s))
    if "guard_flags" in r:
        g = r["guard_flags"]
        lines.append("  GuardFlags 0x%x: %s" % (g["value"], ", ".join(g["names"]) or "none"))
    if "eh_continuation_targets" in r:
        lines.append("  EH-continuation table: %d targets (checked only when the process enforces CET shadow stacks)"
                     % r["eh_continuation_targets"])
    if "dependent_load_flags" in r:
        lines.append("  DependentLoadFlags 0x%x (gta6rt binds imports itself)" % r["dependent_load_flags"])
    lines += ["  attention: " + a for a in r["attention"]]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--json", action="store_true", help="one JSON document")
    args = ap.parse_args(argv)
    results = {}
    try:
        for path in args.images:
            with open(path, "rb") as f:
                try:
                    results[os.path.basename(path)] = inspect(f.read())
                except LoadConfigError as e:
                    raise LoadConfigError("%s: %s" % (path, e)) from None
    except (OSError, LoadConfigError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if args.json:
        json.dump(results, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        for name, r in results.items():
            print("\n".join(describe(name, r)))
        need = sum(1 for r in results.values() if r["attention"])
        print("summary: %d image(s), %d with an attention line" % (len(results), need))
    return 1 if any(r["attention"] for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
