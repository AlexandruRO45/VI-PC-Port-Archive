"""Summarize `gta6rt`'s two first-boot traces: the startup JSONL and the `[d3d12-trace]` lines.

The loader writes the startup trace with `gta6rt --trace FILE` (one JSON object per line, flushed per event so it
survives a crash; `runtime/README.md`, "Startup trace"), and the D3D12 proxy writes one `[d3d12-trace] #<seq>
<method> <outcome>` line to stderr with `--trace-d3d12`. This tool turns both into the small report a first-boot
triage needs: modules, unimplemented calls and their callers, DllMain/TLS results, exceptions and the exit, then
the D3D12 calls by outcome and the first call per method.

The method field can contain spaces (`SetMarker/BeginEvent with unknown metadata`), so the parser takes the
outcome as the last field instead of assuming a single-word method. PowerShell 5.1 wraps the first stderr line of
a native program in an error record (`program.exe : [d3d12-trace] #1 ...`), so a `<name> : ` prefix before the
tag is accepted too. The reader accepts UTF-8 and BOM-marked UTF-16, including PowerShell 5.1 stderr redirection.

Usage:
    python -I boot_trace.py boot.jsonl [--d3d12 d3d12.log] [--json] [--limit N]
                                  [--symbols BUILD_DIR [--symbolizer PATH]]

`--symbols` resolves the stack frames of host modules that the build produced (`host:gta6rt.exe+0x…`,
`host:gta6rt_tls_host.dll+0x…`) against that build's PDB with `llvm-symbolizer`, printing the function and
file:line under each frame. Mapped images (the title and its own DLLs) and host modules the build did not produce
keep the module+RVA text the runtime wrote; frames are never rewritten, only annotated. The frame's RVA is
converted with the image's preferred base from its PE header, so a relocated module resolves the same way. The
symbolizer is looked up in `--symbolizer`, then `LLVM_SYMBOLIZER`, then PATH. When no symbolizer can be run the
report is still printed with the frames as they were, the `symbols` block records why, and the exit code is 2.

The traces can quote what the game prints, so they stay on the player's machine: the text report counts debug
strings instead of printing them, and only the tests' synthetic fixtures go into the repository. Share counts and
error classes from a real run, not its trace (AGENTS.md, "No game data").

Standard library only apart from the optional external symbolizer; reads files, never runs them. A truncated last
line (the expected shape after a crash) is counted, not fatal; a file with no valid event is an error (exit 2).
When one of two requested traces fails to parse, the report still includes the other trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

D3D12_LINE = re.compile(r"^(?:.+ : )?\[d3d12-trace\] #(\d+) (.+) (\S+)\s*$")

# A host-module frame, as runtime/shims/registry.cpp writes it: `host:name.dll+0xRVA`. Mapped images (the title and
# its own DLLs) have no `host:` prefix, so they are left alone.
HOST_FRAME = re.compile(r"^host:(?P<module>[^+]+)\+(?P<rva>0x[0-9a-fA-F]+)$")

# llvm-symbolizer's location line: `file:line:column`. A PDB without line information prints `file:line:0`.
SYMBOL_LOCATION = re.compile(r"^(?P<file>.+):(?P<line>\d+):(?P<column>\d+)$")

# One frame per address keeps the pairing in parse_symbolizer_output to two states. `--output-style=LLVM` is
# explicit because llvm-addr2line (the same binary under another name) defaults to `file:line` without a column,
# which the parser would misread as a function name.
SYMBOLIZER_ARGUMENTS = ("--functions=short", "--no-inlines", "--output-style=LLVM")
SYMBOLIZER_TIMEOUT = 120


class TraceError(ValueError):
    """The file is unreadable or holds no valid trace event."""


def _open_text(path: str):
    """Read UTF-8 or use the byte-order mark to decode a UTF-16 trace."""
    try:
        with open(path, "rb") as raw:
            head = raw.read(2)
        encoding = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        return open(path, encoding=encoding, errors="replace")
    except OSError as e:
        raise TraceError(f"cannot read {path}: {e.strerror or e}") from e


def parse_startup(path: str, limit: int = 10) -> dict:
    """Read the `--trace` JSONL into the units the first-boot triage uses."""
    trace = {
        "path": path,
        "pid": None,
        "events": 0,
        "bad_lines": 0,
        "bad_line_numbers": [],
        "kinds": {},
        "modules": [],
        "tls_callbacks": 0,
        "dllmain": {"ok": 0, "failed": 0, "failed_modules": []},
        "unimplemented": [],
        "unimplemented_by_dll": {},
        "debug_strings": 0,
        "file_open_failed": [],
        "file_open_failed_count": 0,
        "int3_skipped": [],
        "exceptions": [],
        "exception_count": 0,
        "exit": None,
        "error": None,
    }
    seen_imports = set()
    with _open_text(path) as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict) and event.get("kind") == "exit" and type(event.get("code")) is not int:
                    event = None
            except json.JSONDecodeError:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            if not isinstance(event, dict) or "kind" not in event:
                trace["bad_lines"] += 1
                if len(trace["bad_line_numbers"]) < limit:
                    trace["bad_line_numbers"].append(number)
                continue
            trace["events"] += 1
            kind = str(event.get("kind"))
            trace["kinds"][kind] = trace["kinds"].get(kind, 0) + 1
            if kind == "trace_open":
                trace["pid"] = event.get("pid")
            elif kind == "map":
                trace["modules"].append(
                    {
                        "module": event.get("module"),
                        "base": event.get("base"),
                        "size": event.get("size"),
                        "relocated": event.get("relocated"),
                    }
                )
            elif kind == "tls_callback":
                trace["tls_callbacks"] += 1
                trace["last_tls_callback"] = {
                    "module": event.get("module"),
                    "at": event.get("at"),
                    "reason": event.get("reason"),
                }
            elif kind == "dllmain":
                if event.get("result"):
                    trace["dllmain"]["ok"] += 1
                else:
                    trace["dllmain"]["failed"] += 1
                    if len(trace["dllmain"]["failed_modules"]) < limit:
                        trace["dllmain"]["failed_modules"].append(event.get("module"))
            elif kind == "unimplemented_call":
                name = str(event.get("import", "?"))
                if name not in seen_imports:
                    seen_imports.add(name)
                    trace["unimplemented"].append(
                        {"import": name, "caller": event.get("caller")}
                    )
                    dll = name.split("!", 1)[0]
                    trace["unimplemented_by_dll"][dll] = (
                        trace["unimplemented_by_dll"].get(dll, 0) + 1
                    )
            elif kind == "debug_string":
                trace["debug_strings"] += 1
            elif kind == "file_open_failed":
                trace["file_open_failed_count"] += 1
                if len(trace["file_open_failed"]) < limit:
                    trace["file_open_failed"].append(
                        {
                            "path": event.get("path"),
                            "error": event.get("error"),
                            "caller": event.get("caller"),
                        }
                    )
            elif kind == "int3_skipped":
                trace["int3_skipped"].append(event.get("at"))
            elif kind == "exception":
                trace["exception_count"] += 1
                if limit <= 0:
                    continue
                stack = event.get("stack")
                detail = {
                    "code": event.get("code"),
                    "at": event.get("at"),
                    "cpp": event.get("cpp"),
                    "access": event.get("access"),
                    "address": event.get("address"),
                    "stack": stack[:8] if isinstance(stack, list) else [],
                }
                if len(trace["exceptions"]) < limit:
                    trace["exceptions"].append(detail)
                else:
                    trace["exceptions"][-1] = detail
            elif kind == "exit":
                trace["exit"] = event.get("code")
            elif kind == "error":
                trace["error"] = event.get("what")
    if trace["events"] == 0:
        raise TraceError(f"{path} has no valid startup event")
    return trace


def parse_d3d12(path: str, limit: int = 10) -> dict:
    """Read the `--trace-d3d12` stderr lines, ignoring the runtime's ordinary diagnostics."""
    trace = {
        "path": path,
        "calls": 0,
        "ignored_lines": 0,
        "outcomes": {},
        "methods": [],
        "first": {},
        "failed": [],
        "unsupported": [],
        "refused": [],
        "partial": [],
        "last_seq": 0,
    }
    with _open_text(path) as fh:
        for line in fh:
            match = D3D12_LINE.match(line.rstrip("\n"))
            if not match:
                if line.strip():
                    trace["ignored_lines"] += 1
                continue
            seq, method, outcome = int(match.group(1)), match.group(2), match.group(3)
            trace["calls"] += 1
            trace["last_seq"] = seq
            trace["outcomes"][outcome] = trace["outcomes"].get(outcome, 0) + 1
            if method not in trace["first"]:
                trace["methods"].append(method)
                trace["first"][method] = {"seq": seq, "outcome": outcome}
            if outcome in ("failed", "unsupported", "refused", "partial"):
                bucket = trace[outcome]
                if len(bucket) < limit:
                    bucket.append({"seq": seq, "method": method})
    if trace["calls"] == 0:
        raise TraceError(
            f"{path} has no [d3d12-trace] line among {trace['ignored_lines']} line(s); "
            "is it the stderr log of a --trace-d3d12 run, with the Vulkan device enabled?"
        )
    return trace


def image_base(path: str) -> int:
    """The preferred image base from a PE's optional header.

    A PDB stores its symbols against this base, not against wherever the loader mapped the image, so a relocated
    module resolves from the same RVA.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(0x40)
            if len(head) < 0x40 or head[:2] != b"MZ":
                raise TraceError(f"{path} is not a PE image")
            offset = int.from_bytes(head[0x3C:0x40], "little")
            fh.seek(offset)
            if fh.read(4) != b"PE\0\0":
                raise TraceError(f"{path} has no PE header")
            fh.seek(offset + 4 + 20)
            optional = fh.read(32)
    except OSError as e:
        raise TraceError(f"cannot read {path}: {e.strerror or e}") from e
    magic = int.from_bytes(optional[0:2], "little")
    if len(optional) < 32:
        raise TraceError(f"{path} has a truncated optional header")
    if magic == 0x20B:  # PE32+
        return int.from_bytes(optional[24:32], "little")
    if magic == 0x10B:  # PE32
        return int.from_bytes(optional[28:32], "little")
    raise TraceError(f"{path} has no PE optional header")


FrameKey = tuple[str, int]  # (module name, RVA): one host-module frame, deduplicated across exceptions


@dataclass
class SymbolizerRun:
    """One llvm-symbolizer invocation: the module's image, its preferred base, and the RVAs to resolve."""

    module: str
    image: str
    base: int
    rvas: list[int]

    @property
    def addresses(self) -> list[int]:
        return [self.base + rva for rva in self.rvas]


@dataclass
class SymbolPlan:
    """The work `--symbols` has to do, decided before any symbolizer runs.

    `skipped` counts the frames of modules the build folder has no symbolizable image for, so the report can tell
    "no debug information at this address" apart from "not a module this build produced".
    """

    runs: list[SymbolizerRun]
    skipped: dict[str, int]
    errors: dict[str, str]  # a module that has a PDB but whose image header could not be read
    frames: int


def frame_key(frame: object) -> FrameKey | None:
    """The (module, RVA) a `host:` frame names, or None for a mapped-image or bare-address frame."""
    match = HOST_FRAME.match(str(frame))
    if not match:
        return None
    return match.group("module"), int(match.group("rva"), 16)


def host_frames(startup: dict) -> list[FrameKey]:
    """Every (module, RVA) in an exception stack that belongs to a host module, in first-seen order."""
    frames, seen = [], set()
    for exception in startup.get("exceptions", []):
        for frame in exception.get("stack") or []:
            key = frame_key(frame)
            if key is not None and key not in seen:
                seen.add(key)
                frames.append(key)
    return frames


def find_symbolizer(explicit: str | None = None) -> str | None:
    """The llvm-symbolizer to run: the option, then LLVM_SYMBOLIZER, then PATH.

    An unusable `LLVM_SYMBOLIZER` falls through to PATH (a stale variable must not hide a working copy); an explicit
    `--symbolizer` that is not a file is a usage error and is rejected before this runs.
    """
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    from_env = os.environ.get("LLVM_SYMBOLIZER")
    if from_env and os.path.isfile(from_env):
        return from_env
    return shutil.which("llvm-symbolizer")


def _run_symbolizer(command: list[str], addresses: list[int]) -> str:
    """Run llvm-symbolizer once for a batch of addresses. The tests replace this with a fake."""
    request = "\n".join(f"0x{address:x}" for address in addresses) + "\n"
    try:
        done = subprocess.run(
            command,
            input=request,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SYMBOLIZER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise TraceError(f"cannot run {command[0]}: {e}") from e
    if done.returncode != 0:
        detail = (done.stderr or "").strip().splitlines()
        raise TraceError(
            f"{command[0]} exited {done.returncode}" + (f": {detail[-1]}" if detail else "")
        )
    return done.stdout


def parse_symbolizer_output(text: str, addresses: list[int]) -> dict[int, dict]:
    """Pair llvm-symbolizer's output with the addresses it was asked about.

    With `--no-inlines` one address prints at most one frame, in two steps: a function line, then an optional
    `file:line:column` line. The parser walks those two states per address. A function without line information
    prints no location, so the next line already belongs to the next address. An address the symbolizer cannot
    resolve prints `??`, which is dropped instead of being reported as a function name.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results, position = {}, 0
    for address in addresses:
        if position >= len(lines):
            break  # fewer results than addresses: keep what was printed
        # State 1: the function line for this address.
        function = lines[position]
        position += 1
        if function == "??":
            # An unresolved address may still print its `??:0:0` line; consume it so the next
            # address stays aligned.
            if position < len(lines) and SYMBOL_LOCATION.match(lines[position]):
                position += 1
            continue
        # State 2: a location line, printed only when the address has line information.
        location = None
        if position < len(lines):
            match = SYMBOL_LOCATION.match(lines[position])
            if match:
                position += 1
                if match.group("file") != "??":
                    location = {"file": match.group("file"), "line": int(match.group("line"))}
        results[address] = {"function": function, **(location or {"file": None, "line": None})}
    return results


def has_symbols(image: str) -> bool:
    """A module can be resolved only when both its image and a `.pdb` beside it are in the build folder."""
    return os.path.isfile(image) and os.path.isfile(os.path.splitext(image)[0] + ".pdb")


def plan_symbolizer_runs(startup: dict, build_dir: str) -> SymbolPlan:
    """Decide which frames `--symbols` can resolve, without running anything.

    A frame is planned only when the build folder holds both the module and a `.pdb` beside it. Everything else is
    counted as skipped, which is what leaves the title and the host's own modules alone.
    """
    all_frames = host_frames(startup)
    rvas_by_module: dict[str, list[int]] = {}
    image_by_module: dict[str, str] = {}
    skipped: dict[str, int] = {}
    for module, rva in all_frames:
        image = os.path.join(build_dir, module)
        if not has_symbols(image):
            skipped[module] = skipped.get(module, 0) + 1
            continue
        image_by_module[module] = image
        rvas_by_module.setdefault(module, []).append(rva)

    runs = []
    errors: dict[str, str] = {}
    for module, image in image_by_module.items():
        try:
            base = image_base(image)
        except TraceError as e:
            # A PDB without a readable image is a build problem worth naming, not a silent skip.
            errors[module] = str(e)
            skipped[module] = skipped.get(module, 0) + len(rvas_by_module[module])
            continue
        runs.append(SymbolizerRun(module, image, base, rvas_by_module[module]))
    return SymbolPlan(runs=runs, skipped=skipped, errors=errors, frames=len(all_frames))


def resolve_symbols(
    plan: SymbolPlan, symbolizer: str, run=None
) -> tuple[dict[FrameKey, dict], dict[str, str]]:
    """Run the planned symbolizer invocations; return the resolved frames and the per-module errors."""
    run = run or _run_symbolizer
    resolved: dict[FrameKey, dict] = {}
    errors: dict[str, str] = {}
    for item in plan.runs:
        addresses = item.addresses
        try:
            found = run([symbolizer, f"--obj={item.image}", *SYMBOLIZER_ARGUMENTS], addresses)
            symbols = parse_symbolizer_output(found, addresses)
        except TraceError as e:
            errors[item.module] = str(e)
            continue
        for rva, address in zip(item.rvas, addresses):
            symbol = symbols.get(address)
            if symbol:
                resolved[(item.module, rva)] = symbol
    return resolved, errors


def annotate_exceptions(startup: dict, resolved: dict[FrameKey, dict]) -> None:
    """Attach a `stack_symbols` map to each exception that has at least one resolved frame."""
    for exception in startup.get("exceptions", []):
        stack_symbols = {}
        for frame in exception.get("stack") or []:
            key = frame_key(frame)
            symbol = resolved.get(key) if key is not None else None
            if symbol:
                stack_symbols[frame] = symbol
        if stack_symbols:
            exception["stack_symbols"] = stack_symbols


def symbols_block(
    build_dir: str,
    symbolizer: str | None,
    plan: SymbolPlan,
    resolved: dict[FrameKey, dict],
    errors: dict[str, str],
    tool_error: str | None = None,
) -> dict:
    """The `symbols` block the report prints; `symbolizer` is None when no tool could be run.

    `errors` is keyed by module; `error` is the one tool-level failure (no symbolizer found), so a JSON consumer
    never has to string-match a reserved module name.
    """
    skipped_frames = sum(plan.skipped.values())
    return {
        "dir": os.path.abspath(build_dir),
        "symbolizer": symbolizer,
        "host_frames": plan.frames,
        "resolved": len(resolved),
        "unresolved": plan.frames - skipped_frames - len(resolved),
        "skipped_frames": skipped_frames,
        "modules": sorted(item.module for item in plan.runs),
        "skipped": sorted(plan.skipped),
        "errors": {**plan.errors, **errors},
        "error": tool_error,
    }


def symbolize(startup: dict, build_dir: str, symbolizer: str, run=None) -> dict:
    """Resolve the planned frames and annotate the trace. Returns the `symbols` block."""
    plan = plan_symbolizer_runs(startup, build_dir)
    resolved, errors = resolve_symbols(plan, symbolizer, run)
    annotate_exceptions(startup, resolved)
    return symbols_block(build_dir, symbolizer, plan, resolved, errors)


def symbolizer_error_block(startup: dict, build_dir: str, message: str) -> dict:
    """The `symbols` block when no symbolizer could be run: the plan is still reported, nothing is resolved."""
    plan = plan_symbolizer_runs(startup, build_dir)
    return symbols_block(build_dir, None, plan, {}, {}, tool_error=message)


def summarize(startup: dict | None, d3d12: dict | None) -> dict:
    return {"startup": startup, "d3d12": d3d12}


def _tail(items, limit: int) -> str:
    shown = ", ".join(str(x) for x in items[:limit])
    return shown + (f", ... {len(items) - limit} more" if len(items) > limit else "")


def format_text(summary: dict, limit: int) -> str:
    out = []
    startup = summary.get("startup")
    if startup:
        out.append(f"startup trace: {startup['path']}")
        if startup["events"]:
            out.append(
                f"  pid {startup['pid']}, {startup['events']} event(s), {startup['bad_lines']} bad line(s)"
            )
            kinds = ", ".join(f"{k} {v}" for k, v in sorted(startup["kinds"].items()))
            out.append(f"  kinds: {kinds}")
        if startup["modules"]:
            names = [m["module"] for m in startup["modules"]]
            out.append(f"  modules ({len(names)}): {_tail(names, limit)}")
        out.append(f"  tls callbacks: {startup['tls_callbacks']}")
        if callback := startup.get("last_tls_callback"):
            reason = callback["reason"]
            phase = "phase unknown"
            if type(reason) is int:
                phase = {0: "process detach", 1: "process attach"}.get(reason, phase)
            at = callback["at"] or "address unknown"
            out.append(f"    last observed TLS callback: {at} ({phase})")
        dm = startup["dllmain"]
        line = f"  dllmain: {dm['ok']} ok, {dm['failed']} failed"
        if dm["failed_modules"]:
            line += f" ({_tail(dm['failed_modules'], limit)})"
        out.append(line)
        if startup["unimplemented"]:
            out.append(f"  unimplemented: {len(startup['unimplemented'])} unique")
            for item in startup["unimplemented"][:limit]:
                out.append(f"    {item['import']} from {item['caller']}")
            if len(startup["unimplemented"]) > limit:
                out.append(f"    ... {len(startup['unimplemented']) - limit} more")
            by_dll = ", ".join(
                f"{dll} {count}"
                for dll, count in sorted(startup["unimplemented_by_dll"].items())
            )
            out.append(f"  by dll (unique imports): {by_dll}")
        else:
            out.append("  unimplemented: none")
        if startup["debug_strings"]:
            out.append(f"  debug strings: {startup['debug_strings']} (text not shown)")
        if startup["file_open_failed_count"]:
            out.append(f"  failed file opens: {startup['file_open_failed_count']}")
            for item in startup["file_open_failed"][:limit]:
                out.append(f"    {item['path']} error {item['error']} from {item['caller']}")
            if startup["file_open_failed_count"] > limit:
                out.append(f"    ... {startup['file_open_failed_count'] - limit} more")
        if startup["int3_skipped"]:
            out.append(f"  int3 skipped (LEONIDA_SKIP_INT3, one per address): {_tail(startup['int3_skipped'], limit)}")
        if startup["exception_count"]:
            out.append(f"  exceptions: {startup['exception_count']}")
            for exc in startup["exceptions"][:limit]:
                cpp = exc.get("cpp")
                detail = " (C++ throw)" if cpp is True else f" ({cpp})" if cpp else ""
                if exc.get("access"):
                    detail += f" ({exc['access']}"
                    if exc.get("address"):
                        detail += f" {exc['address']}"
                    detail += ")"
                out.append(f"    code {exc['code']} at {exc['at']}{detail}")
                if exc.get("stack"):
                    out.append(f"      stack: {_tail(exc['stack'], limit)}")
                    for frame in exc["stack"][:limit]:
                        symbol = (exc.get("stack_symbols") or {}).get(frame)
                        if not symbol:
                            continue
                        location = f" ({symbol['file']}:{symbol['line']})" if symbol.get("file") else ""
                        out.append(f"        {frame} -> {symbol['function']}{location}")
        symbols = startup.get("symbols")
        if symbols:
            total = symbols["host_frames"]
            if symbols["symbolizer"]:
                line = (
                    f"  symbols: {symbols['resolved']} of {total} host frame(s) "
                    f"resolved from {symbols['dir']}"
                )
                if symbols["skipped_frames"]:
                    line += f" ({symbols['skipped_frames']} skipped)"
                out.append(line)
            else:
                out.append(
                    f"  symbols: none of {total} host frame(s) resolved from {symbols['dir']}"
                )
            if symbols["error"]:
                out.append(f"    {symbols['error']}")
            if symbols["skipped"]:
                out.append(f"    not resolvable from that folder: {_tail(symbols['skipped'], limit)}")
            for module, message in symbols["errors"].items():
                out.append(f"    {module}: {message}")
        if startup["error"]:
            out.append(f"  error: {startup['error']}")
        if startup["exit"] is not None:
            out.append(f"  exit: {startup['exit']}")
    d3d12 = summary.get("d3d12")
    if d3d12:
        out.append(f"d3d12 trace: {d3d12['path']}")
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(d3d12["outcomes"].items()))
        out.append(
            f"  {d3d12['calls']} call(s), {d3d12['ignored_lines']} other line(s)"
            " (the [rt] diagnostics are not summarized)"
            + (f": {outcomes}" if outcomes else "")
        )
        if d3d12["methods"]:
            out.append(f"  first per method: {_tail(d3d12['methods'], limit)}")
        if d3d12["unsupported"]:
            unsupported = [f"#{c['seq']} {c['method']}" for c in d3d12["unsupported"]]
            out.append(f"  unsupported: {_tail(unsupported, limit)}")
        if d3d12["failed"]:
            failed = [f"#{c['seq']} {c['method']}" for c in d3d12["failed"]]
            out.append(f"  failed: {_tail(failed, limit)}")
        if d3d12["refused"]:
            refused = [f"#{c['seq']} {c['method']}" for c in d3d12["refused"]]
            out.append(f"  refused: {_tail(refused, limit)}")
        if d3d12.get("partial"):
            partial = [f"#{c['seq']} {c['method']}" for c in d3d12["partial"]]
            out.append(f"  partial: {_tail(partial, limit)}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize gta6rt's startup and D3D12 first-boot traces."
    )
    ap.add_argument("startup", nargs="?", help="the JSONL written by gta6rt --trace")
    ap.add_argument("--d3d12", help="the stderr log written with --trace-d3d12")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument(
        "--limit", type=int, default=10, help="items listed per section (default 10)"
    )
    ap.add_argument(
        "--symbols",
        metavar="BUILD_DIR",
        help="resolve exception stack frames of the modules this build produced against their PDB",
    )
    ap.add_argument(
        "--symbolizer",
        metavar="PATH",
        help="llvm-symbolizer to run (default: LLVM_SYMBOLIZER, then PATH)",
    )
    a = ap.parse_args(argv)
    if not a.startup and not a.d3d12:
        print("error: give the startup trace, --d3d12, or both", file=sys.stderr)
        return 2
    if a.limit < 1:
        print("error: --limit must be at least 1", file=sys.stderr)
        return 2
    if a.symbolizer and not a.symbols:
        print("error: --symbolizer needs --symbols", file=sys.stderr)
        return 2
    if a.symbolizer and not (os.path.isfile(a.symbolizer) and os.access(a.symbolizer, os.X_OK)):
        print(f"error: --symbolizer is not an executable file: {a.symbolizer}", file=sys.stderr)
        return 2
    if a.symbols and not os.path.isdir(a.symbols):
        print(f"error: --symbols is not a directory: {a.symbols}", file=sys.stderr)
        return 2
    summary = summarize(None, None)
    exit_code = 0
    for section, path, parse in (("startup", a.startup, parse_startup), ("d3d12", a.d3d12, parse_d3d12)):
        if path:
            try:
                summary[section] = parse(path, a.limit)
            except TraceError as e:
                print(f"error: {e}", file=sys.stderr)
                exit_code = 2
    if a.symbols:
        if summary.get("startup"):
            symbolizer = find_symbolizer(a.symbolizer)
            if symbolizer:
                summary["startup"]["symbols"] = symbolize(summary["startup"], a.symbols, symbolizer)
            else:
                # The report is still worth printing: the stack and the exception address are the caller's first
                # triage facts, and the symbols block says why they are not annotated. The exit code still signals
                # the failure.
                message = "--symbols needs llvm-symbolizer (--symbolizer PATH, LLVM_SYMBOLIZER, or PATH)"
                print(f"error: {message}", file=sys.stderr)
                summary["startup"]["symbols"] = symbolizer_error_block(
                    summary["startup"], a.symbols, message
                )
                exit_code = 2
        else:
            # The startup trace failed to parse; keep the other half of the report, as without --symbols.
            print("error: --symbols needs the startup trace", file=sys.stderr)
            exit_code = 2
    if not any(summary.values()):
        return 2
    if a.json:
        print(json.dumps(summary, indent=1))
    else:
        print(format_text(summary, a.limit))
    return exit_code


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
