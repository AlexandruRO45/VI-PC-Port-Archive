"""Summarize a Windows minidump and attribute its addresses to modules, including the ones `gta6rt` maps itself.

The title has its own crash reporter. Under `gta6rt` it writes a minidump through `CreateFileW` to `T:\\<guid>.dmp`,
which lands in the overlay's `.gta6rt-temp` folder (`docs/boot.md`, "The title's own crash dumps"). The dump's
ModuleList comes from the host loader, so it names `gta6rt.exe` and the host DLLs but not the executable and DLLs
that `gta6rt`'s PE loader mapped by hand: a debugger shows the fault and most of the stack as bare addresses. The
startup trace (`gta6rt --trace FILE`) has a `map` event with `module`, `base` and `size` for each of those images,
so this tool reads both files and prints every address as `module+0xRVA`.

It prints:
- the exception: code, thread, the faulting address and, for an access violation or in-page error, whether it was a
  read, write or execute and the target address;
- each thread's instruction pointer and stack pointer, the faulting thread from the exception's own context;
- for the faulting thread (or every thread with `--all-threads`), the stack slots that hold an address inside a
  known module past its header page, from the stack memory the dump saved (64 by default, `--frames N`). This is
  a scan, not an unwind: a slot can be a stale return address or a pointer into a module's data, so read the list
  as candidates, top of the stack first.

Usage:
    python -I tools/minidump_summary.py DUMP [--trace TRACE] [--all-threads] [--frames N] [--json]

Without `--trace` only the dump's own modules are used, and the mapped images stay unattributed. Use the trace of the
same run: the tool compares the dump's process id (MiscInfo stream) with the trace's `trace_open` pid and prints a
warning when they differ.

A dump holds the title's memory and the report names addresses inside the title's code, so both stay on the player's
machine; share the exception code, access type and module names, not the dump or the offsets (AGENTS.md, "No game
data"). Thread names are counted, not printed.

Standard library only; reads files, never runs them. A missing or unreadable input, a file that is not a minidump
or a trace with no `map` event prints one `ERROR:` line and exits with 2.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

SIGNATURE = 0x504D444D  # "MDMP"

THREAD_LIST = 3
MODULE_LIST = 4
MEMORY_LIST = 5
EXCEPTION = 6
SYSTEM_INFO = 7
MEMORY64_LIST = 9
MISC_INFO = 15
THREAD_NAMES = 24

AMD64 = 9
CONTEXT_SIZE = 0x4D0
RSP_OFFSET = 0x98
RIP_OFFSET = 0xF8
HEADER_PAGE = 0x1000  # a value in a module's first page points at its PE headers, not at code

EXCEPTION_NAMES = {
    0x80000003: "breakpoint",
    0xC0000005: "access violation",
    0xC0000006: "in-page error",
    0xC000001D: "illegal instruction",
    0xC0000094: "integer divide by zero",
    0xC00000FD: "stack overflow",
    0xC0000409: "stack buffer overrun (fail fast)",
    0xE06D7363: "C++ exception",
}
ACCESS = {0: "read", 1: "write", 8: "execute"}


class DumpError(ValueError):
    """The input is unreadable or not in the expected format."""


class Minidump:
    """The parts of a minidump this report reads. Offsets follow `minidumpapiset.h`."""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 32:
            raise DumpError("file is shorter than a minidump header")
        signature, _version, count, directory = struct.unpack_from("<IIII", data, 0)
        if signature != SIGNATURE:
            raise DumpError("not a minidump (no MDMP signature)")
        self.streams = {}
        for i in range(count):
            stream_type, size, rva = self._unpack("<III", directory + 12 * i)
            self.streams.setdefault(stream_type, (rva, size))

    def _unpack(self, fmt: str, offset: int):
        try:
            return struct.unpack_from(fmt, self.data, offset)
        except struct.error as e:
            raise DumpError(f"truncated minidump at offset {offset:#x}") from e

    def _string(self, rva: int) -> str:
        (length,) = self._unpack("<I", rva)
        return self.data[rva + 4:rva + 4 + length].decode("utf-16-le", "replace")

    def architecture(self):
        if SYSTEM_INFO not in self.streams:
            return None
        return self._unpack("<H", self.streams[SYSTEM_INFO][0])[0]

    def process_id(self):
        """MINIDUMP_MISC_INFO.ProcessId when MINIDUMP_MISC1_PROCESS_ID is set."""
        if MISC_INFO not in self.streams:
            return None
        _size, flags, pid = self._unpack("<III", self.streams[MISC_INFO][0])
        return pid if flags & 1 else None

    def modules(self) -> list:
        if MODULE_LIST not in self.streams:
            return []
        rva = self.streams[MODULE_LIST][0]
        (count,) = self._unpack("<I", rva)
        out = []
        for i in range(count):
            base, size, _checksum, _stamp, name_rva = self._unpack("<QIIII", rva + 4 + 108 * i)
            name = self._string(name_rva).replace("/", "\\").rsplit("\\", 1)[-1]
            out.append({"module": name, "base": base, "size": size, "source": "dump"})
        return out

    def context(self, size: int, rva: int):
        """(rip, rsp) of an AMD64 CONTEXT, or None for a short one."""
        if size < CONTEXT_SIZE:
            return None
        (rsp,) = self._unpack("<Q", rva + RSP_OFFSET)
        (rip,) = self._unpack("<Q", rva + RIP_OFFSET)
        return rip, rsp

    def threads(self) -> list:
        if THREAD_LIST not in self.streams:
            return []
        rva = self.streams[THREAD_LIST][0]
        (count,) = self._unpack("<I", rva)
        out = []
        for i in range(count):
            tid, _suspend, _cls, _prio, _teb, stack_start, stack_size, stack_rva, ctx_size, ctx_rva = self._unpack(
                "<IIIIQQIIII", rva + 4 + 48 * i)
            out.append({"id": tid, "context": self.context(ctx_size, ctx_rva),
                        "stack": (stack_start, stack_size, stack_rva)})
        return out

    def thread_name_count(self) -> int:
        if THREAD_NAMES not in self.streams:
            return 0
        return self._unpack("<I", self.streams[THREAD_NAMES][0])[0]

    def exception(self):
        if EXCEPTION not in self.streams:
            return None
        rva = self.streams[EXCEPTION][0]
        tid, _align, code, flags, _record, address, count = self._unpack("<IIIIQQI", rva)
        params = self._unpack("<15Q", rva + 40)[:min(count, 15)]
        ctx_size, ctx_rva = self._unpack("<II", rva + 160)
        return {"thread": tid, "code": code, "flags": flags, "address": address, "params": list(params),
                "context": self.context(ctx_size, ctx_rva)}

    def memory(self) -> list:
        """(start, size, file offset) of every saved memory range."""
        ranges = []
        if MEMORY_LIST in self.streams:
            rva = self.streams[MEMORY_LIST][0]
            (count,) = self._unpack("<I", rva)
            for i in range(count):
                ranges.append(self._unpack("<QII", rva + 4 + 16 * i))
        if MEMORY64_LIST in self.streams:
            rva = self.streams[MEMORY64_LIST][0]
            count, offset = self._unpack("<QQ", rva)
            for i in range(count):
                start, size = self._unpack("<QQ", rva + 16 + 16 * i)
                ranges.append((start, size, offset))
                offset += size
        return ranges

    def read(self, address: int, size: int, ranges: list):
        for start, length, offset in ranges:
            if start <= address and address + size <= start + length:
                begin = offset + (address - start)
                if begin + size <= len(self.data):
                    return self.data[begin:begin + size]
        return None


def read_trace(path: str):
    """(pid, mapped modules) from a `gta6rt --trace` JSONL file."""
    pid = None
    mapped = []
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # a crash can cut the last line short
                if not isinstance(event, dict):
                    continue
                if event.get("kind") == "trace_open" and isinstance(event.get("pid"), int):
                    pid = event["pid"]
                elif event.get("kind") == "map":
                    try:
                        mapped.append({"module": str(event["module"]), "base": int(str(event["base"]), 0),
                                       "size": int(str(event["size"]), 0), "source": "trace"})
                    except (KeyError, ValueError):
                        continue
    except OSError as e:
        raise DumpError(f"cannot read {path}: {e.strerror or e}") from e
    if not mapped:
        raise DumpError(f"{path} has no map event (is it a gta6rt --trace file?)")
    return pid, mapped


class Symbolizer:
    def __init__(self, modules: list):
        # A mapped image is attributed first: the trace's ranges are the ones the dump leaves out.
        self.modules = sorted(modules, key=lambda m: m["source"] != "trace")

    def find(self, address: int):
        for m in self.modules:
            if m["base"] <= address < m["base"] + m["size"]:
                return m
        return None

    def __call__(self, address: int) -> str:
        m = self.find(address)
        if m is None:
            return f"{address:#x}"
        return f"{m['module']}+{address - m['base']:#x}"


def overlaps(mapped: list, own: list) -> list:
    return [(a["module"], b["module"]) for a in mapped for b in own
            if a["base"] < b["base"] + b["size"] and b["base"] < a["base"] + a["size"]]


def stack_scan(dump: Minidump, thread: dict, rsp: int, ranges: list, symbolize: Symbolizer, limit: int) -> list:
    """Stack slots at or above `rsp` that hold an address inside a known module."""
    start, size, _rva = thread["stack"]
    # The thread's own stack descriptor first: a dump need not repeat the saved stack in MemoryList/Memory64List.
    ranges = [thread["stack"]] + list(ranges)
    lo = max(start, rsp) if start <= rsp < start + size else start
    frames = []
    address = lo - (lo % 8)
    while address + 8 <= start + size and len(frames) < limit:
        raw = dump.read(address, 8, ranges)
        if raw is None:
            break
        (value,) = struct.unpack("<Q", raw)
        module = symbolize.find(value)
        if module is not None and value - module["base"] >= HEADER_PAGE:
            frames.append({"slot": f"rsp+{address - rsp:#x}" if address >= rsp else f"{address:#x}",
                           "address": symbolize(value)})
        address += 8
    return frames


def summarize(dump: Minidump, mapped: list, trace_pid, all_threads: bool, limit: int) -> dict:
    own = dump.modules()
    symbolize = Symbolizer(own + mapped)
    ranges = dump.memory()
    report = {
        "architecture": dump.architecture(),
        "dump_pid": dump.process_id(),
        "trace_pid": trace_pid,
        "dump_modules": len(own),
        "mapped_modules": [{"module": m["module"], "size": f"{m['size']:#x}"} for m in mapped],
        "overlaps": overlaps(mapped, own),
        "thread_names": dump.thread_name_count(),
        "warnings": [],
        "exception": None,
        "threads": [],
    }
    if report["architecture"] not in (None, AMD64):
        report["warnings"].append(f"processor architecture {report['architecture']} is not AMD64; "
                                  "thread contexts are not read")
    if mapped and trace_pid is not None and report["dump_pid"] is not None and trace_pid != report["dump_pid"]:
        report["warnings"].append(f"the dump is from pid {report['dump_pid']} and the trace from pid {trace_pid}: "
                                  "the mapped modules may not match this dump")
    for a, b in report["overlaps"]:
        report["warnings"].append(f"mapped {a} overlaps the dump's {b}: the trace may be from another run")
    amd64 = report["architecture"] in (None, AMD64)

    exc = dump.exception()
    if exc is not None:
        entry = {"thread": exc["thread"], "code": f"{exc['code']:#010x}",
                 "name": EXCEPTION_NAMES.get(exc["code"], "unknown"), "at": symbolize(exc["address"])}
        if exc["code"] in (0xC0000005, 0xC0000006) and len(exc["params"]) >= 2:
            entry["access"] = ACCESS.get(exc["params"][0], f"type {exc['params'][0]}")
            entry["target"] = symbolize(exc["params"][1])
        report["exception"] = entry

    for thread in dump.threads():
        context = thread["context"] if amd64 else None
        faulting = exc is not None and thread["id"] == exc["thread"]
        if faulting and amd64 and exc["context"] is not None:
            context = exc["context"]  # the thread's own context is the crash reporter's, not the fault's
        entry = {"id": thread["id"], "faulting": faulting}
        if context is not None:
            rip, rsp = context
            entry["rip"] = symbolize(rip)
            entry["rsp"] = f"{rsp:#x}"
            if faulting or all_threads:
                entry["stack"] = stack_scan(dump, thread, rsp, ranges, symbolize, limit)
        report["threads"].append(entry)
    return report


def print_text(report: dict, out) -> None:
    w = out.write
    w(f"dump: {report['dump_modules']} module(s) in its ModuleList, {len(report['threads'])} thread(s)"
      f" ({report['thread_names']} named), pid {report['dump_pid']}\n")
    if report["mapped_modules"]:
        names = ", ".join(m["module"] for m in report["mapped_modules"])
        w(f"trace: {len(report['mapped_modules'])} mapped module(s) ({names}), pid {report['trace_pid']}\n")
    else:
        w("trace: none (images gta6rt mapped itself are not attributed; pass --trace)\n")
    for warning in report["warnings"]:
        w(f"WARNING: {warning}\n")
    exc = report["exception"]
    if exc is None:
        w("exception: none in this dump\n")
    else:
        w(f"exception: {exc['code']} ({exc['name']}) on thread {exc['thread']} at {exc['at']}\n")
        if "access" in exc:
            w(f"  {exc['access']} of {exc['target']}\n")
    for thread in report["threads"]:
        mark = " (faulting)" if thread["faulting"] else ""
        if "rip" not in thread:
            w(f"thread {thread['id']}{mark}: no context\n")
            continue
        w(f"thread {thread['id']}{mark}: rip {thread['rip']}  rsp {thread['rsp']}\n")
        if "stack" in thread:
            if not thread["stack"]:
                w("  stack scan: no module address in the saved stack\n")
            for frame in thread["stack"]:
                w(f"  {frame['slot']:>12}  {frame['address']}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("dump", help="the minidump (.dmp)")
    parser.add_argument("--trace", help="the gta6rt --trace file of the same run, for the mapped modules")
    parser.add_argument("--all-threads", action="store_true", help="scan every thread's stack, not just the faulting one")
    parser.add_argument("--frames", type=int, default=32, help="stack slots to list per thread (default 32)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    try:
        try:
            with open(args.dump, "rb") as fh:
                data = fh.read()
        except OSError as e:
            raise DumpError(f"cannot read {args.dump}: {e.strerror or e}") from e
        dump = Minidump(data)
        trace_pid, mapped = read_trace(args.trace) if args.trace else (None, [])
        report = summarize(dump, mapped, trace_pid, args.all_threads, max(0, args.frames))
    except DumpError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print_text(report, sys.stdout)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
