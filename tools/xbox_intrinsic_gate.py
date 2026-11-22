"""Reject unresolved xbox.* intrinsic calls before a PSO submission.

The gate reads the read-only inventory from `probe_roots.xbox_intrinsic_inventory()` and keeps
the three groups apart on purpose: a symbol a module only declares, or one that only metadata
mentions, is not a call and does not block submission. What blocks is a call the module still
executes. A refused module never reaches the submission callable.

A lowering counts only when it is visible in the transformed module: the call is gone from the
instructions, so the inventory no longer reports it. An adapter status table is advisory and
never clears a call the module still emits.
"""

import argparse
import json
from pathlib import Path
import sys

from probe_roots import XBOX_PREFIX, xbox_intrinsic_inventory

LOWERED = "LOWERED"
UNRESOLVED = "UNRESOLVED"
ALLOW = "ALLOW"
REJECT = "REJECT"
NOT_CHECKED = "NOT_CHECKED"


def default_status_of(name):
    """No builtin is lowered in this repository, so every called name stays unresolved."""
    return UNRESOLVED, "no-verified-lowering-in-this-repository"


def status_table(lowered=()):
    """Build an advisory status table: it reports what an adapter claims, nothing more.

    A lowering counts only when it is visible in the transformed module, as a call that is
    gone. What an adapter says about a call it still emits cannot clear that call, so this
    table never reaches the verdict; it only annotates the report.
    """
    lowered = frozenset(lowered)

    def status_of(name):
        if name in lowered:
            return LOWERED, "reported-by-adapter"
        return default_status_of(name)

    return status_of


def decide(inventory, status_of=None):
    """Refuse a module while it still calls xbox.*, whatever any adapter reports.

    Every call blocks. `status_of` is advisory: it fills `advisory` on each blocking entry so a
    report can show what an adapter claims, and the verdict ignores it.
    """
    status_of = status_of or default_status_of
    blocking = []
    for symbol in sorted(inventory["calls"]):
        builtin = symbol[len(XBOX_PREFIX):] if symbol.startswith(XBOX_PREFIX) else symbol
        advisory, unblocks = status_of(builtin)
        blocking.append({
            "name": symbol,
            "builtin": builtin,
            "sites": inventory["calls"][symbol],
            "status": UNRESOLVED,
            "unblocks": unblocks,
            "advisory": advisory,
        })
    verdict = REJECT if blocking else ALLOW
    names = [entry["name"] for entry in blocking]
    if verdict == REJECT:
        message = f"REJECT: {len(names)} executable xbox.* call(s): {', '.join(names)}"
        claimed = [entry["builtin"] for entry in blocking if entry["advisory"] == LOWERED]
        if claimed:
            message += f" (advisory reports LOWERED for {', '.join(claimed)}; the module still calls them)"
    else:
        message = (
            f"ALLOW: no executable xbox.* calls "
            f"(declarations={len(inventory['declarations'])}, metadata_only={len(inventory['metadata_only'])})"
        )
    return {
        "verdict": verdict,
        "reason": "executable-xbox-calls" if blocking else "no-executable-xbox-calls",
        "message": message,
        "calls": inventory["calls"],
        "declarations": inventory["declarations"],
        "metadata_only": inventory["metadata_only"],
        "blocking": blocking,
        "reported_names": names,
        "checked": True,
    }


def not_checked(reason):
    """Record that no module was examined, so a caller cannot read it as a cleared module.

    A submission path that has no disassembly of the final module reports this instead of
    gating nothing: the verdict stays distinct from ALLOW and `checked` is False, so a
    caller cannot treat it as cleared.
    """
    return {
        "verdict": NOT_CHECKED,
        "reason": reason,
        "message": f"NOT_CHECKED: the final module was not disassembled ({reason})",
        "checked": False,
        "calls": {},
        "declarations": [],
        "metadata_only": [],
        "blocking": [],
        "reported_names": [],
    }


def check_text(text, status_of=None):
    """Gate a module given as disassembly text."""
    return decide(xbox_intrinsic_inventory(text), status_of)


def check_container(probe, blob, program, status_of=None, optional_modes=()):
    """Gate a DXBC container by disassembling its DXIL part through the validator probe."""
    from probe_roots import validate_shader

    checks = validate_shader(probe, blob, program, optional_modes=optional_modes)
    decision = decide(checks["ir"]["xbox_intrinsics"], status_of)
    decision["validation"] = checks
    return decision


def gate_submission(decision, module_text, submit):
    """Submit only for an ALLOW verdict; REJECT and NOT_CHECKED reach `submit` zero times."""
    decision["submitted"] = decision["verdict"] == ALLOW
    if decision["submitted"]:
        submit(module_text)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    text = sub.add_parser("check-text", help="Gate a module given as disassembly text")
    text.add_argument("path", type=Path, help="Disassembly text file ('-' for standard input)")
    shader = sub.add_parser("check", help="Gate a DXBC container through a validator probe")
    shader.add_argument("input", type=Path, help="DXBC container")
    shader.add_argument("--probe", type=Path, help="root_probe.exe path")
    args = parser.parse_args()

    if args.command not in ("check-text", "check"):
        parser.print_help(sys.stderr)
        return 2

    try:
        if args.command == "check-text":
            body = (sys.stdin.read() if str(args.path) == "-"
                    else args.path.read_text(encoding="utf-8", errors="replace"))
            decision = check_text(body)
        else:
            if not args.probe:
                parser.error("check requires --probe")
            from resource_inventory import parts

            raw = args.input.read_bytes()
            sections = parts(raw)
            if b"DXIL" not in sections:
                raise ValueError("DXBC container has no DXIL part")
            decision = check_container(args.probe.resolve(strict=True), raw, sections[b"DXIL"])
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(decision["message"])
    print(json.dumps(decision, indent=2))
    return 0 if decision["verdict"] == ALLOW else 1


if __name__ == "__main__":
    sys.exit(main())
