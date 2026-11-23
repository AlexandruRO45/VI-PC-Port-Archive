"""tools/minidump_summary.py attributes a minidump's addresses to the dump's modules and the trace's mapped ones.

    python -I tests/test_minidump_summary.py

Builds a small synthetic AMD64 minidump in memory (header, stream directory, SystemInfo, MiscInfo, ModuleList with
one host module, ThreadList with one thread whose CONTEXT and stack memory are saved, Exception with its own
CONTEXT) and a `gta6rt --trace` file with a `map` event for an image the dump does not list. No real dump bytes.
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import minidump_summary as ms  # noqa: E402

HOST_BASE, HOST_SIZE = 0x7FF000000000, 0x10000
GAME_BASE, GAME_SIZE = 0x140000000, 0x200000
STACK_BASE, STACK_SLOTS = 0x5000, 8
FAULT_RSP = STACK_BASE + 0x10
TID = 7
PID = 32


def context(rip, rsp):
    ctx = bytearray(ms.CONTEXT_SIZE)
    struct.pack_into("<Q", ctx, ms.RSP_OFFSET, rsp)
    struct.pack_into("<Q", ctx, ms.RIP_OFFSET, rip)
    return bytes(ctx)


def build_dump(code=0xC0000005, params=(1, 0x10), pid=PID):
    """A minidump laid out as: header, directory, stream bodies, then the blobs they point at."""
    blobs = bytearray()
    base = 0x1000  # blobs start here, well past the header and the directory

    def blob(data):
        offset = base + len(blobs)
        blobs.extend(data)
        while len(blobs) % 8:
            blobs.append(0)
        return offset

    name = "C:\\windows\\system32\\host.dll".encode("utf-16-le")
    name_rva = blob(struct.pack("<I", len(name)) + name)
    # The thread's own context is the crash reporter's (inside the host DLL); the exception's is the fault.
    thread_ctx = context(HOST_BASE + 0x100, STACK_BASE)
    fault_ctx = context(GAME_BASE + 0x1234, FAULT_RSP)
    thread_ctx_rva = blob(thread_ctx)
    fault_ctx_rva = blob(fault_ctx)
    slots = [0, 0, GAME_BASE + 0x5678, GAME_BASE + 0x10, HOST_BASE + 0x1200, 0, GAME_BASE + 0x9ABC, 0]
    stack_rva = blob(struct.pack(f"<{STACK_SLOTS}Q", *slots))

    streams = []
    streams.append((ms.SYSTEM_INFO, struct.pack("<HHH", ms.AMD64, 6, 0) + bytes(50)))
    streams.append((ms.MISC_INFO, struct.pack("<III", 24, 1, pid) + bytes(12)))
    module = struct.pack("<QIIII", HOST_BASE, HOST_SIZE, 0, 0, name_rva) + bytes(108 - 24)
    streams.append((ms.MODULE_LIST, struct.pack("<I", 1) + module))
    thread = struct.pack("<IIIIQQIIII", TID, 0, 0, 0, 0, STACK_BASE, STACK_SLOTS * 8, stack_rva,
                         ms.CONTEXT_SIZE, thread_ctx_rva)
    streams.append((ms.THREAD_LIST, struct.pack("<I", 1) + thread))
    streams.append((ms.MEMORY_LIST, struct.pack("<IQII", 1, STACK_BASE, STACK_SLOTS * 8, stack_rva)))
    info = list(params) + [0] * (15 - len(params))
    exc = struct.pack("<IIIIQQII", TID, 0, code, 0, 0, GAME_BASE + 0x1234, len(params), 0)
    exc += struct.pack("<15Q", *info) + struct.pack("<II", ms.CONTEXT_SIZE, fault_ctx_rva)
    streams.append((ms.EXCEPTION, exc))

    directory_rva = 32
    body = bytearray()
    directory = bytearray()
    body_rva = directory_rva + 12 * len(streams)
    for stream_type, data in streams:
        directory += struct.pack("<III", stream_type, len(data), body_rva + len(body))
        body += data
    assert body_rva + len(body) <= base
    header = struct.pack("<IIIIIIQ", ms.SIGNATURE, 0xA793, len(streams), directory_rva, 0, 0, 0)
    out = bytearray(header + directory + body)
    out += bytes(base - len(out))
    return bytes(out + blobs)


def trace_lines(pid=PID):
    return [
        json.dumps({"seq": 1, "t": 0.0, "kind": "trace_open", "pid": pid}),
        json.dumps({"seq": 2, "t": 0.1, "kind": "map", "module": "title.exe", "base": hex(GAME_BASE),
                    "size": hex(GAME_SIZE), "relocated": False}),
        '{"seq":3,"t":0.2,"kind":"exce',  # cut short by the crash
    ]


class MinidumpSummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        if isinstance(data, bytes):
            with open(path, "wb") as fh:
                fh.write(data)
        else:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("\n".join(data) + "\n")
        return path

    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ms.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def report(self, dump=None, trace=None):
        args = [self.write("a.dmp", dump or build_dump())]
        if trace is not False:
            args += ["--trace", self.write("a.trace", trace or trace_lines())]
        code, out, err = self.run_tool(*args, "--json")
        self.assertEqual(code, 0, err)
        return json.loads(out)

    def test_exception_and_access_attributed_to_mapped_module(self):
        report = self.report()
        exc = report["exception"]
        self.assertEqual(exc["code"], "0xc0000005")
        self.assertEqual(exc["name"], "access violation")
        self.assertEqual(exc["at"], "title.exe+0x1234")
        self.assertEqual(exc["access"], "write")
        self.assertEqual(exc["target"], "0x10")
        self.assertEqual(report["warnings"], [])

    def test_faulting_thread_uses_exception_context_and_scans_stack(self):
        (thread,) = self.report()["threads"]
        self.assertTrue(thread["faulting"])
        self.assertEqual(thread["rip"], "title.exe+0x1234")
        self.assertEqual(thread["rsp"], hex(FAULT_RSP))
        # Slots below the fault's rsp belong to the crash reporter's frames, and a value in a header page is no
        # return address: both are skipped.
        self.assertEqual([f["address"] for f in thread["stack"]],
                         ["title.exe+0x5678", "host.dll+0x1200", "title.exe+0x9abc"])
        self.assertEqual(thread["stack"][0]["slot"], "rsp+0x0")

    def test_without_trace_mapped_addresses_stay_bare(self):
        report = self.report(trace=False)
        self.assertEqual(report["exception"]["at"], hex(GAME_BASE + 0x1234))
        self.assertEqual([f["address"] for f in report["threads"][0]["stack"]], ["host.dll+0x1200"])

    def test_pid_mismatch_warns(self):
        report = self.report(trace=trace_lines(pid=PID + 1))
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("pid", report["warnings"][0])

    def test_read_access_and_other_codes(self):
        self.assertEqual(self.report(build_dump(params=(0, 0x20)))["exception"]["access"], "read")
        exc = self.report(build_dump(code=0xC0000409, params=()))["exception"]
        self.assertEqual(exc["name"], "stack buffer overrun (fail fast)")
        self.assertNotIn("access", exc)

    def test_text_report(self):
        dump = self.write("a.dmp", build_dump())
        trace = self.write("a.trace", trace_lines())
        code, out, _ = self.run_tool(dump, "--trace", trace)
        self.assertEqual(code, 0)
        self.assertIn("exception: 0xc0000005 (access violation) on thread 7 at title.exe+0x1234", out)
        self.assertIn("write of 0x10", out)
        self.assertIn("thread 7 (faulting): rip title.exe+0x1234", out)

    def test_bad_inputs_exit_2_with_one_error_line(self):
        cases = [
            [os.path.join(self.tmp.name, "missing.dmp")],
            [self.write("bad.dmp", b"NOTADUMP" * 8)],
            [self.write("ok.dmp", build_dump()), "--trace", os.path.join(self.tmp.name, "missing.trace")],
            [self.write("ok2.dmp", build_dump()), "--trace", self.write("empty.trace", ['{"kind":"trace_open"}'])],
        ]
        for args in cases:
            code, out, err = self.run_tool(*args)
            self.assertEqual(code, 2, args)
            self.assertEqual(out, "")
            self.assertEqual(len(err.splitlines()), 1, err)
            self.assertTrue(err.startswith("ERROR: "), err)


if __name__ == "__main__":
    unittest.main()
