"""Unit tests for tools/boot_trace.py on synthetic traces (no game data, no Windows, no GPU)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import boot_trace as bt

STARTUP = [
    {"seq": 1, "t": 0.0, "kind": "trace_open", "pid": 123},
    {
        "seq": 2,
        "kind": "map",
        "module": "game.exe",
        "base": "0x140000000",
        "size": "0x1000",
        "relocated": False,
    },
    {
        "seq": 3,
        "kind": "map",
        "module": "x.dll",
        "base": "0x180000000",
        "size": "0x2000",
        "relocated": True,
    },
    {
        "seq": 4,
        "kind": "tls_callback",
        "module": "x.dll",
        "at": "x.dll+0x10",
        "reason": 1,
    },
    {"seq": 5, "kind": "dllmain", "module": "x.dll", "result": 1},
    {"seq": 6, "kind": "dllmain", "module": "y.dll", "result": 0},
    {
        "seq": 7,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x10",
    },
    {
        "seq": 8,
        "kind": "unimplemented_call",
        "import": "x.dll!Foo",
        "caller": "game.exe+0x20",
    },
    {
        "seq": 9,
        "kind": "unimplemented_call",
        "import": "y.dll!Bar",
        "caller": "game.exe+0x30",
    },
    {"seq": 10, "kind": "debug_string", "text": "secret game text"},
    {
        "seq": 11,
        "kind": "exception",
        "code": "0xc0000005",
        "at": "game.exe+0x40",
        "access": "read",
        "address": "0x0",
        "stack": ["game.exe+0x40", "game.exe+0x120"],
    },
    {
        "seq": 12,
        "kind": "exception",
        "code": "0xe06d7363",
        "at": "game.exe+0x30",
        "cpp": True,
    },
    {"seq": 13, "kind": "future_event", "detail": "a kind this version does not know"},
    {"seq": 14, "kind": "exit", "code": 0},
]

D3D12 = [
    "[rt] ordinary diagnostic",
    "[d3d12-trace] #1 D3D12XboxCreateDevice ok",
    "[d3d12-trace] #2 CreateCommandQueue failed",
    "[d3d12-trace] #3 Foo unsupported",
    "[d3d12-trace] #4 D3D12XboxCreateDevice ok",
    "[d3d12-trace] #5 SetMarker/BeginEvent with unknown metadata unsupported",
    "[d3d12-trace] #6 MapTiles refused",
    "[d3d12-trace] #7 CreateRootSignature called",
    "[d3d12-trace] #8 SetMarker ignored",
    "d3d12_test.exe : [d3d12-trace] #9 ID3D12CommandQueue::ExecuteCommandLists called",
]


class BootTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.startup = os.path.join(self.tmp.name, "boot.jsonl")
        self.d3d12 = os.path.join(self.tmp.name, "d3d12.log")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, events, newline="\n"):
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.writelines(json.dumps(event) + newline for event in events)

    def write_lines(self, path, lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def test_startup_summary(self):
        self.write(self.startup, STARTUP)
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["pid"], 123)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 0)
        self.assertEqual(trace["kinds"]["map"], 2)
        self.assertEqual(trace["kinds"]["future_event"], 1)
        self.assertEqual([m["module"] for m in trace["modules"]], ["game.exe", "x.dll"])
        self.assertEqual(trace["tls_callbacks"], 1)
        self.assertEqual(
            trace["dllmain"], {"ok": 1, "failed": 1, "failed_modules": ["y.dll"]}
        )
        self.assertEqual(
            [u["import"] for u in trace["unimplemented"]], ["x.dll!Foo", "y.dll!Bar"]
        )
        self.assertEqual(trace["unimplemented"][0]["caller"], "game.exe+0x10")
        self.assertEqual(trace["unimplemented_by_dll"], {"x.dll": 1, "y.dll": 1})
        self.assertEqual(trace["debug_strings"], 1)
        self.assertEqual(trace["exception_count"], 2)
        self.assertEqual(
            trace["exceptions"][0]["stack"], ["game.exe+0x40", "game.exe+0x120"]
        )
        self.assertIs(trace["exceptions"][1]["cpp"], True)
        self.assertEqual(trace["exit"], 0)

    def test_exception_limit_keeps_final_observation(self):
        early = [
            {"kind": "exception", "code": "0xe06d7363", "at": f"synthetic.exe+0x{0x1100 + i:x}", "cpp": True}
            for i in range(10)
        ]
        final = {
            "kind": "exception",
            "code": "0xc0000005",
            "at": "synthetic.exe+0x2010",
            "access": "read",
            "address": "0x0",
            "stack": ["synthetic.exe+0x2010", "synthetic.exe+0x1000"],
        }
        for limit, preceding in ((1, early), (10, early), (10, early[:-1])):
            with self.subTest(limit=limit, count=len(preceding) + 1):
                self.write(self.startup, [STARTUP[0], *preceding, final])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(bt.main([self.startup, "--limit", str(limit), "--json"]), 0)
                summary = json.loads(output.getvalue())
                startup = summary["startup"]
                self.assertEqual(startup["exception_count"], len(preceding) + 1)
                self.assertEqual(
                    [exc["at"] for exc in startup["exceptions"]],
                    [exc["at"] for exc in preceding[:limit - 1]] + [final["at"]],
                )
                self.assertEqual(startup["exceptions"][-1]["stack"], final["stack"])
                self.assertIn("synthetic.exe+0x2010 (read 0x0)", bt.format_text(summary, limit))

    def test_debug_text_is_not_printed(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        self.assertIn("debug strings: 1", text)
        self.assertNotIn("secret game text", text)

    def test_truncated_line_is_counted_not_fatal(self):
        self.write(self.startup, STARTUP)
        with open(self.startup, "a", encoding="utf-8") as f:
            f.write('{"seq": 15, "kind":')
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["events"], len(STARTUP))
        self.assertEqual(trace["bad_lines"], 1)
        self.assertEqual(trace["bad_line_numbers"], [len(STARTUP) + 1])

    def test_crlf_and_non_object_lines_are_counted(self):
        self.write(
            self.startup,
            [
                {"seq": 1, "kind": "trace_open", "pid": 7},
                {"seq": 2, "kind": "exit", "code": 0},
            ],
            newline="\r\n",
        )
        with open(self.startup, "a", encoding="utf-8") as f:
            f.write("null\n")
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["events"], 2)
        self.assertEqual(trace["bad_lines"], 1)

    def test_malformed_exit_records_preserve_the_last_valid_code(self):
        for code in (0, -7):
            with self.subTest(code=code):
                self.write(self.startup, [
                    {"kind": "trace_open", "pid": 123},
                    {"kind": "exit", "code": code},
                    {"kind": "exit"},
                    {"kind": "exit", "code": "0"},
                    {"kind": "exit", "code": True},
                    {"kind": "exit", "code": None},
                    {"kind": "exit", "code": 0.0},
                    {"kind": "debug_string", "text": "synthetic"},
                ])
                trace = bt.parse_startup(self.startup)
                self.assertEqual(trace["bad_lines"], 5)
                self.assertEqual(trace["bad_line_numbers"], [3, 4, 5, 6, 7])
                self.assertEqual(trace["events"], 3)
                self.assertEqual(trace["kinds"]["exit"], 1)
                self.assertIs(type(trace["exit"]), int)
                self.assertEqual(trace["exit"], code)
                self.assertEqual(trace["debug_strings"], 1)

    def test_d3d12_summary(self):
        self.write_lines(self.d3d12, D3D12)
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 9)
        self.assertEqual(trace["ignored_lines"], 1)
        self.assertEqual(
            trace["outcomes"],
            {
                "ok": 2,
                "failed": 1,
                "unsupported": 2,
                "refused": 1,
                "called": 2,
                "ignored": 1,
            },
        )
        self.assertEqual(
            trace["methods"],
            [
                "D3D12XboxCreateDevice",
                "CreateCommandQueue",
                "Foo",
                "SetMarker/BeginEvent with unknown metadata",
                "MapTiles",
                "CreateRootSignature",
                "SetMarker",
                "ID3D12CommandQueue::ExecuteCommandLists",
            ],
        )
        self.assertEqual(
            trace["first"]["D3D12XboxCreateDevice"], {"seq": 1, "outcome": "ok"}
        )
        self.assertEqual(
            trace["first"]["ID3D12CommandQueue::ExecuteCommandLists"],
            {"seq": 9, "outcome": "called"},
        )
        self.assertEqual(trace["failed"], [{"seq": 2, "method": "CreateCommandQueue"}])
        self.assertEqual(
            trace["unsupported"],
            [
                {"seq": 3, "method": "Foo"},
                {"seq": 5, "method": "SetMarker/BeginEvent with unknown metadata"},
            ],
        )
        self.assertEqual(trace["refused"], [{"seq": 6, "method": "MapTiles"}])
        self.assertEqual(trace["last_seq"], 9)

    def test_powershell_error_record_prefix_keeps_the_first_call(self):
        self.write_lines(
            self.d3d12,
            [
                "d3d12_test.exe : [d3d12-trace] #1 ID3D12GraphicsCommandList::SetPipelineState called",
                "At line:1 char:1",
                "+ & '.\\d3d12_test.exe' --trace-d3d12 2> log",
            ],
        )
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(
            trace["first"]["ID3D12GraphicsCommandList::SetPipelineState"],
            {"seq": 1, "outcome": "called"},
        )

    def test_d3d12_text_report_names_refused_and_spaced_methods(self):
        self.write_lines(self.d3d12, D3D12)
        text = bt.format_text(bt.summarize(None, bt.parse_d3d12(self.d3d12)), 10)
        self.assertIn("refused: #6 MapTiles", text)
        self.assertIn("#5 SetMarker/BeginEvent with unknown metadata", text)
        self.assertIn("ID3D12CommandQueue::ExecuteCommandLists", text)
        self.assertIn("[rt] diagnostics are not summarized", text)

    def test_partial_submissions_are_named_in_text_and_json(self):
        method = "ID3D12CommandQueue::ExecuteCommandLists"
        self.write_lines(self.d3d12, [
            f"[d3d12-trace] #1 {method} ok",
            f"[d3d12-trace] #2 {method} partial",
            f"[d3d12-trace] #3 {method} partial",
        ])
        trace = bt.parse_d3d12(self.d3d12, limit=1)
        self.assertEqual(trace["first"][method], {"seq": 1, "outcome": "ok"})
        self.assertEqual(trace["outcomes"]["partial"], 2)
        text = bt.format_text(bt.summarize(None, trace), 1)
        self.assertIn(f"  partial: #2 {method}", text)
        self.assertNotIn(f"#3 {method}", text)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(bt.main(["--d3d12", self.d3d12, "--limit", "1", "--json"]), 0)
        parsed = json.loads(stdout.getvalue())["d3d12"]
        self.assertEqual(parsed["partial"], [{"seq": 2, "method": method}])
        self.assertEqual(parsed["outcomes"]["partial"], 2)

    def test_json_cli(self):
        self.write(self.startup, STARTUP)
        self.write_lines(self.d3d12, D3D12)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(
                bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 0
            )
        data = json.loads(out.getvalue())
        self.assertEqual(data["startup"]["pid"], 123)
        self.assertEqual(data["d3d12"]["calls"], 9)

    def test_missing_and_empty_files_are_errors(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([os.path.join(self.tmp.name, "missing.jsonl")]), 2)
            open(self.startup, "w").close()
            self.assertEqual(bt.main([self.startup]), 2)
            self.assertEqual(bt.main(["--limit", "0", self.startup]), 2)
        self.assertIn("cannot read", err.getvalue())

    def test_attach_failure_is_reported_before_any_d3d12_call(self):
        self.write(self.startup, [{"kind": "error", "what": "guest.dll: DllMain returned FALSE"}])
        self.write_lines(self.d3d12, ["error: guest.dll: DllMain returned FALSE"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12]), 2)
        self.assertIn("error: guest.dll: DllMain returned FALSE", out.getvalue())
        self.assertIn("no [d3d12-trace] line", err.getvalue())

    def test_missing_startup_preserves_d3d12_json(self):
        self.write_lines(self.d3d12, ["[d3d12-trace] #1 CreateCommandQueue failed"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 2)
        self.assertIn("cannot read", err.getvalue())
        self.assertEqual(json.loads(out.getvalue()), bt.summarize(None, bt.parse_d3d12(self.d3d12)))

    def test_no_report_when_neither_input_can_be_read(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(bt.main([self.startup, "--d3d12", self.d3d12, "--json"]), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertIn(self.startup, err.getvalue())
        self.assertIn(self.d3d12, err.getvalue())

    def test_d3d12_log_without_trace_lines_is_an_error(self):
        self.write_lines(self.d3d12, ["[rt] one", "[rt] two"])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(bt.main(["--d3d12", self.d3d12]), 2)
        self.assertIn("no [d3d12-trace] line", err.getvalue())

    def test_utf16_logs_keep_both_wrapped_and_unwrapped_calls(self):
        lines = ("synthetic.exe : [d3d12-trace] #1 First called\r\n"
                 "[d3d12-trace] #2 Second ok\r\n")
        for encoding, bom in (("utf-16-le", b"\xff\xfe"), ("utf-16-be", b"\xfe\xff")):
            with self.subTest(encoding=encoding):
                with open(self.d3d12, "wb") as f:
                    f.write(bom + lines.encode(encoding))
                output = io.StringIO()
                errors = io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    result = bt.main(["--d3d12", self.d3d12, "--json"])
                self.assertEqual(result, 0, errors.getvalue())
                trace = json.loads(output.getvalue())["d3d12"]
                self.assertEqual(trace["calls"], 2)
                self.assertEqual(trace["methods"], ["First", "Second"])

    def test_utf8_bom_is_stripped(self):
        with open(self.d3d12, "wb") as f:
            f.write(b"\xef\xbb\xbf" + b"[d3d12-trace] #1 Foo ok\n")
        trace = bt.parse_d3d12(self.d3d12)
        self.assertEqual(trace["calls"], 1)
        self.assertEqual(trace["methods"], ["Foo"])

    def test_text_report_has_the_triage_units(self):
        self.write(self.startup, STARTUP)
        text = bt.format_text(bt.summarize(bt.parse_startup(self.startup), None), 10)
        for needle in (
            "pid 123",
            "modules (2)",
            "dllmain: 1 ok, 1 failed",
            "unimplemented: 2 unique",
            "x.dll!Foo from game.exe+0x10",
            "by dll (unique imports): x.dll 1, y.dll 1",
            "exceptions: 2",
            "code 0xc0000005 at game.exe+0x40 (read 0x0)",
            "code 0xe06d7363 at game.exe+0x30 (C++ throw)",
            "stack: game.exe+0x40, game.exe+0x120",
            "exit: 0",
        ):
            self.assertIn(needle, text)

    def test_failed_file_opens_and_skipped_int3_are_listed(self):
        self.write(
            self.startup,
            [
                {"seq": 1, "kind": "trace_open", "pid": 1},
                {"seq": 2, "kind": "file_open_failed", "api": "CreateFileW", "path": "GAME:/probe/missing.bin", "error": 2, "caller": "game.exe+0x20"},
                {"seq": 3, "kind": "int3_skipped", "at": "game.exe+0x1330"},
            ],
        )
        trace = bt.parse_startup(self.startup)
        self.assertEqual(trace["file_open_failed"], [{"path": "GAME:/probe/missing.bin", "error": 2, "caller": "game.exe+0x20"}])
        self.assertEqual(trace["int3_skipped"], ["game.exe+0x1330"])
        text = bt.format_text(bt.summarize(trace, None), 10)
        self.assertIn("failed file opens: 1", text)
        self.assertIn("GAME:/probe/missing.bin error 2 from game.exe+0x20", text)
        self.assertIn("game.exe+0x1330", text)

    def test_last_observed_tls_callback_in_incomplete_startup(self):
        first = {"kind": "tls_callback", "module": "first.dll", "at": "first.dll+0x1100", "reason": 1}
        for fields, reason, phase in (
            ({"reason": 1}, 1, "process attach"),
            ({"reason": 0}, 0, "process detach"),
            ({}, None, "phase unknown"),
        ):
            with self.subTest(phase=phase):
                last = {"kind": "tls_callback", "module": "fixture.exe", "at": "fixture.exe+0x2200", **fields}
                self.write(self.startup, [STARTUP[0], first, last])
                with open(self.startup, "a", encoding="utf-8", newline="\n") as f:
                    f.write('{"kind":"tls_callback","at":"truncated')

                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(bt.main([self.startup, "--json"]), 0)
                summary = json.loads(output.getvalue())
                startup = summary["startup"]
                self.assertEqual(startup["tls_callbacks"], 2)
                self.assertEqual(startup["bad_lines"], 1)
                self.assertIsNone(startup["exit"])
                self.assertEqual(
                    startup["last_tls_callback"],
                    {"module": "fixture.exe", "at": "fixture.exe+0x2200", "reason": reason},
                )
                text = bt.format_text(summary, 10)
                self.assertIn("tls callbacks: 2", text)
                self.assertIn(f"last observed TLS callback: fixture.exe+0x2200 ({phase})", text)


def make_pe(path, base, pe32plus=True):
    """A minimal PE header that only carries the optional header's ImageBase, for the symbol tests."""
    dos = 0x80
    optional = dos + 4 + 20
    image = bytearray(optional + 32)
    image[0:2] = b"MZ"
    image[0x3C:0x40] = dos.to_bytes(4, "little")
    image[dos:dos + 4] = b"PE\0\0"
    if pe32plus:
        image[optional:optional + 2] = (0x20B).to_bytes(2, "little")
        image[optional + 24:optional + 32] = base.to_bytes(8, "little")
    else:
        image[optional:optional + 2] = (0x10B).to_bytes(2, "little")
        image[optional + 28:optional + 32] = base.to_bytes(4, "little")
    with open(path, "wb") as f:
        f.write(image)


class SymbolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.startup = os.path.join(self.dir, "boot.jsonl")

    def tearDown(self):
        self.tmp.cleanup()

    def module(self, name, base, pe32plus=True, pdb=True):
        path = os.path.join(self.dir, name)
        make_pe(path, base, pe32plus)
        if pdb:
            with open(os.path.splitext(path)[0] + ".pdb", "wb") as f:
                f.write(b"synthetic debug directory, never parsed by these tests")
        return path

    def write_trace(self, stack, at=None):
        events = [
            {"seq": 1, "kind": "trace_open", "pid": 1},
            {
                "seq": 2,
                "kind": "exception",
                "code": "0xc0000005",
                "at": at or stack[0],
                "access": "write",
                "stack": stack,
            },
        ]
        with open(self.startup, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(event) + "\n" for event in events)

    def test_image_base_reads_both_optional_headers(self):
        plus = self.module("a.exe", 0x140000000)
        flat = self.module("b.exe", 0x400000, pe32plus=False)
        self.assertEqual(bt.image_base(plus), 0x140000000)
        self.assertEqual(bt.image_base(flat), 0x400000)

    def test_image_base_rejects_a_non_pe_file(self):
        path = os.path.join(self.dir, "not-a-pe.bin")
        with open(path, "wb") as f:
            f.write(b"not a PE image")
        with self.assertRaises(bt.TraceError):
            bt.image_base(path)

    def test_host_frames_keeps_host_modules_in_first_seen_order(self):
        trace = {"exceptions": [
            {"stack": ["game.exe+0x10", "host:gta6rt.exe+0x20", "host:gta6rt.exe+0x20"]},
            {"stack": ["host:gta6rt_tls_host.dll+0x5", "0x7ff61234", "host:bad.dll+0xzz"]},
        ]}
        self.assertEqual(
            bt.host_frames(trace),
            [("gta6rt.exe", 0x20), ("gta6rt_tls_host.dll", 0x5)],
        )

    def test_symbolize_resolves_only_modules_the_build_produced(self):
        self.module("gta6rt.exe", 0x140000000)
        self.module("gta6rt_tls_host.dll", 0x180000000)
        self.write_trace(
            [
                "game.exe+0x10",
                "host:gta6rt.exe+0x20",
                "host:kernel32.dll+0x30",
                "host:gta6rt_tls_host.dll+0x40",
            ]
        )
        trace = bt.parse_startup(self.startup)
        calls = []

        def fake(command, addresses):
            calls.append((command, addresses))
            return "\n".join(
                f"frame_{address:x}\nC:\\src\\f{address:x}.cpp:{address & 0xfff}:0"
                for address in addresses
            ) + "\n"

        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=fake)
        self.assertEqual(symbols["host_frames"], 3)
        self.assertEqual(symbols["resolved"], 2)
        self.assertEqual(symbols["unresolved"], 0)
        self.assertEqual(symbols["skipped_frames"], 1)
        self.assertEqual(symbols["skipped"], ["kernel32.dll"])
        self.assertEqual(symbols["modules"], ["gta6rt.exe", "gta6rt_tls_host.dll"])
        self.assertEqual(
            [addresses for _, addresses in calls],
            [[0x140000020], [0x180000040]],
        )
        self.assertIn("--no-inlines", calls[0][0])
        self.assertIn("--output-style=LLVM", calls[0][0])
        self.assertIn("--obj=" + os.path.join(self.dir, "gta6rt.exe"), calls[0][0])
        stack_symbols = trace["exceptions"][0]["stack_symbols"]
        self.assertEqual(
            stack_symbols["host:gta6rt.exe+0x20"],
            {"function": "frame_140000020", "file": "C:\\src\\f140000020.cpp", "line": 32},
        )
        self.assertNotIn("game.exe+0x10", stack_symbols)
        self.assertNotIn("host:kernel32.dll+0x30", stack_symbols)

    def test_a_module_without_a_pdb_is_left_alone(self):
        self.module("gta6rt.exe", 0x140000000, pdb=False)
        self.write_trace(["host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)

        def fake(command, addresses):
            raise AssertionError("no symbolizer run expected")

        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=fake)
        self.assertEqual(symbols["host_frames"], 1)
        self.assertEqual(symbols["resolved"], 0)
        self.assertEqual(symbols["skipped"], ["gta6rt.exe"])
        self.assertNotIn("stack_symbols", trace["exceptions"][0])

    def test_unresolved_addresses_keep_the_frame_text(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)
        symbols = bt.symbolize(
            trace, self.dir, "fake-symbolizer", run=lambda command, addresses: "??\n??:0:0\n"
        )
        self.assertEqual((symbols["resolved"], symbols["unresolved"]), (0, 1))
        self.assertNotIn("stack_symbols", trace["exceptions"][0])

    def test_a_symbolizer_failure_is_reported_not_fatal(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)

        def fake(command, addresses):
            raise bt.TraceError("llvm-symbolizer exited 1: bad object file")

        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=fake)
        self.assertEqual(symbols["errors"], {"gta6rt.exe": "llvm-symbolizer exited 1: bad object file"})
        self.assertEqual(symbols["resolved"], 0)

    def test_parse_output_pairs_functions_and_locations(self):
        text = "frame_a\nC:\\src\\a.cpp:12:0\n\nframe_b\nC:\\src\\b.cpp:34:0\n\n??\n??:0:0\n"
        self.assertEqual(
            bt.parse_symbolizer_output(text, [1, 2, 3]),
            {
                1: {"function": "frame_a", "file": "C:\\src\\a.cpp", "line": 12},
                2: {"function": "frame_b", "file": "C:\\src\\b.cpp", "line": 34},
            },
        )

    def test_parse_output_handles_a_function_without_a_location(self):
        # The first address prints no location, so its second line is already the next address's function: the
        # parser has to stay in the function state rather than consume it as a location.
        text = "no_line_function\nframe_b\nC:\\src\\b.cpp:34:0\n"
        self.assertEqual(
            bt.parse_symbolizer_output(text, [1, 2]),
            {
                1: {"function": "no_line_function", "file": None, "line": None},
                2: {"function": "frame_b", "file": "C:\\src\\b.cpp", "line": 34},
            },
        )

    def test_parse_output_keeps_the_addresses_it_was_given_a_result_for(self):
        self.assertEqual(
            bt.parse_symbolizer_output("only\nC:\\src\\a.cpp:1:0\n", [1, 2]),
            {1: {"function": "only", "file": "C:\\src\\a.cpp", "line": 1}},
        )

    def test_report_prints_the_function_and_line_under_the_frame(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["game.exe+0x10", "host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)
        trace["symbols"] = bt.symbolize(
            trace,
            self.dir,
            "fake-symbolizer",
            run=lambda command, addresses: "rt::shims::rt_unimplemented\nC:\\src\\registry.cpp:184:0\n",
        )
        text = bt.format_text(bt.summarize(trace, None), 10)
        self.assertIn("host:gta6rt.exe+0x20 -> rt::shims::rt_unimplemented (C:\\src\\registry.cpp:184)", text)
        self.assertIn("symbols: 1 of 1 host frame(s) resolved from " + os.path.abspath(self.dir), text)
        self.assertNotIn("game.exe+0x10 ->", text)

    def test_json_carries_the_stack_symbols(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["host:gta6rt.exe+0x20"])
        with mock.patch.object(
            bt,
            "_run_symbolizer",
            lambda command, addresses: "f\nC:\\src\\f.cpp:7:0\n",
        ):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    bt.main([self.startup, "--symbols", self.dir, "--symbolizer", sys.executable, "--json"]),
                    0,
                )
        startup = json.loads(output.getvalue())["startup"]
        self.assertEqual(startup["symbols"]["resolved"], 1)
        self.assertEqual(
            startup["exceptions"][0]["stack_symbols"]["host:gta6rt.exe+0x20"],
            {"function": "f", "file": "C:\\src\\f.cpp", "line": 7},
        )

    def test_a_missing_symbolizer_keeps_the_exception_report(self):
        """A missing tool must never suppress the parsed trace: the stack and the error both stay in the JSON."""
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["host:gta6rt.exe+0x20"], at="host:gta6rt.exe+0x20")
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"PATH": "", "LLVM_SYMBOLIZER": ""}):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                code = bt.main([self.startup, "--symbols", self.dir, "--json"])
        self.assertEqual(code, 2)
        startup = json.loads(output.getvalue())["startup"]
        self.assertEqual(startup["exception_count"], 1)
        self.assertEqual(startup["exceptions"][0]["stack"], ["host:gta6rt.exe+0x20"])
        self.assertEqual(
            startup["symbols"],
            {
                "dir": os.path.abspath(self.dir),
                "symbolizer": None,
                "host_frames": 1,
                "resolved": 0,
                "unresolved": 1,
                "skipped_frames": 0,
                "modules": ["gta6rt.exe"],
                "skipped": [],
                "errors": {},
                "error": "--symbols needs llvm-symbolizer "
                "(--symbolizer PATH, LLVM_SYMBOLIZER, or PATH)",
            },
        )
        self.assertIn("--symbols needs llvm-symbolizer", error.getvalue())

    def test_symbolize_keeps_each_frame_with_its_own_address(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(
            ["host:gta6rt.exe+0x20", "host:gta6rt.exe+0x30", "host:gta6rt.exe+0x40"]
        )
        trace = bt.parse_startup(self.startup)
        calls = []

        def fake(command, addresses):
            calls.append((command, addresses))
            return "".join(f"fn_{address:x}\nC:\\src\\f{address:x}.cpp:1:0\n" for address in addresses)

        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=fake)
        stack = trace["exceptions"][0]["stack"]
        self.assertEqual(
            [trace["exceptions"][0]["stack_symbols"][frame]["function"] for frame in stack],
            ["fn_140000020", "fn_140000030", "fn_140000040"],
        )
        self.assertEqual(symbols["host_frames"], 3)
        self.assertEqual(symbols["resolved"], 3)
        self.assertEqual(len(calls), 1)  # one batch per module, not one call per frame
        self.assertEqual(calls[0][1], [0x140000020, 0x140000030, 0x140000040])

    def test_a_module_with_two_frames_and_no_pdb_counts_both_as_skipped(self):
        self.write_trace(["host:gta6rt.exe+0x20", "host:gta6rt.exe+0x30"])
        trace = bt.parse_startup(self.startup)

        def fake(command, addresses):
            raise AssertionError("no symbolizer run expected")

        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=fake)
        self.assertEqual(symbols["host_frames"], 2)
        self.assertEqual(symbols["skipped_frames"], 2)
        self.assertEqual(symbols["skipped"], ["gta6rt.exe"])
        self.assertEqual(symbols["unresolved"], 0)

    def test_an_unreadable_image_is_reported_not_silently_skipped(self):
        path = os.path.join(self.dir, "gta6rt.exe")
        with open(path, "wb") as f:
            f.write(b"MZ" + b"\0" * 0x3E)  # a DOS header whose PE offset points at nothing
        with open(os.path.join(self.dir, "gta6rt.pdb"), "wb") as f:
            f.write(b"synthetic")
        self.write_trace(["host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)
        symbols = bt.symbolize(trace, self.dir, "fake-symbolizer", run=lambda c, a: "")
        self.assertEqual(symbols["skipped"], ["gta6rt.exe"])
        self.assertIn("gta6rt.exe", symbols["errors"])
        self.assertIn("PE header", symbols["errors"]["gta6rt.exe"])

    def test_parse_output_handles_unresolved_addresses_in_the_middle(self):
        # `??` is followed by its own location line; consuming it keeps the next address aligned.
        text = "??\n??:0:0\nframe_b\nC:\\src\\b.cpp:1:0\n"
        self.assertEqual(
            bt.parse_symbolizer_output(text, [1, 2]),
            {2: {"function": "frame_b", "file": "C:\\src\\b.cpp", "line": 1}},
        )

    def test_parse_output_keeps_a_public_symbol_without_line_info(self):
        text = "pub_only\n??:0:0\nframe_b\nC:\\src\\b.cpp:1:0\n"
        self.assertEqual(
            bt.parse_symbolizer_output(text, [1, 2]),
            {
                1: {"function": "pub_only", "file": None, "line": None},
                2: {"function": "frame_b", "file": "C:\\src\\b.cpp", "line": 1},
            },
        )

    def test_report_prints_a_function_without_a_location_alone(self):
        self.module("gta6rt.exe", 0x140000000)
        self.write_trace(["host:gta6rt.exe+0x20"])
        trace = bt.parse_startup(self.startup)
        trace["symbols"] = bt.symbolize(
            trace, self.dir, "fake-symbolizer", run=lambda command, addresses: "pub_only\n??:0:0\n"
        )
        text = bt.format_text(bt.summarize(trace, None), 10)
        self.assertIn("host:gta6rt.exe+0x20 -> pub_only", text)
        self.assertNotIn("(??:0)", text)

    def test_run_symbolizer_sends_the_addresses_and_reads_the_output(self):
        script = "import sys\nfor line in sys.stdin:\n    print('f'); print('a.cpp:1:0')\n"
        output = bt._run_symbolizer([sys.executable, "-c", script], [0x140001000])
        self.assertIn("f", output)
        self.assertIn("a.cpp:1:0", output)

    def test_run_symbolizer_reports_a_nonzero_exit(self):
        with self.assertRaises(bt.TraceError) as caught:
            bt._run_symbolizer([sys.executable, "--obj=x", "--no-inlines"], [0x140001000])
        self.assertIn("exited", str(caught.exception))

    def test_an_unusable_env_symbolizer_falls_through_to_path(self):
        with mock.patch.dict(os.environ, {"LLVM_SYMBOLIZER": os.path.join(self.dir, "missing")}):
            with mock.patch.object(bt.shutil, "which", lambda name: "/path/llvm-symbolizer"):
                self.assertEqual(bt.find_symbolizer(), "/path/llvm-symbolizer")

    def test_a_valid_env_symbolizer_is_used(self):
        path = os.path.join(self.dir, "symbolizer")
        with open(path, "wb"):
            pass
        with mock.patch.dict(os.environ, {"LLVM_SYMBOLIZER": path}):
            self.assertEqual(bt.find_symbolizer(), path)

    def test_symbols_with_a_missing_startup_keeps_the_d3d12_report(self):
        d3d12 = os.path.join(self.dir, "d3d12.log")
        with open(d3d12, "w", encoding="utf-8") as f:
            f.write("[d3d12-trace] #1 D3D12XboxCreateDevice ok\n")
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = bt.main(
                [os.path.join(self.dir, "missing.jsonl"), "--d3d12", d3d12, "--symbols", self.dir, "--json"]
            )
        self.assertEqual(code, 2)
        summary = json.loads(output.getvalue())
        self.assertIsNone(summary["startup"])
        self.assertEqual(summary["d3d12"]["calls"], 1)
        self.assertIn("--symbols needs the startup trace", error.getvalue())

    @unittest.skipIf(os.name == "nt", "os.access(X_OK) is always true on Windows")
    def test_symbolizer_must_be_executable(self):
        self.write_trace(["host:gta6rt.exe+0x20"])
        path = os.path.join(self.dir, "not-executable")
        with open(path, "w", encoding="utf-8") as f:
            f.write("#!/bin/sh\n")
        os.chmod(path, 0o644)
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(bt.main([self.startup, "--symbols", self.dir, "--symbolizer", path]), 2)
        self.assertIn("not an executable file", error.getvalue())

    def test_option_errors(self):
        self.write_trace(["host:gta6rt.exe+0x20"])
        cases = [
            (["--symbolizer", sys.executable], "--symbolizer needs --symbols"),
            (["--symbols", os.path.join(self.dir, "missing")], "--symbols is not a directory"),
            (
                ["--symbols", self.dir, "--symbolizer", os.path.join(self.dir, "no-such-symbolizer")],
                "--symbolizer is not an executable file",
            ),
        ]
        for extra, message in cases:
            with self.subTest(message=message):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    self.assertEqual(bt.main([self.startup, *extra]), 2)
                self.assertIn(message, error.getvalue())

        error = io.StringIO()
        output = io.StringIO()
        with mock.patch.object(bt, "find_symbolizer", lambda explicit=None: None):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
                self.assertEqual(bt.main([self.startup, "--symbols", self.dir, "--json"]), 2)
        self.assertIn("--symbols needs llvm-symbolizer", error.getvalue())
        startup = json.loads(output.getvalue())["startup"]
        self.assertEqual(startup["exceptions"][0]["stack"], ["host:gta6rt.exe+0x20"])
        self.assertIsNone(startup["symbols"]["symbolizer"])
        self.assertIn("--symbols needs llvm-symbolizer", startup["symbols"]["error"])

        error = io.StringIO()
        with mock.patch.object(bt, "find_symbolizer", lambda explicit=None: "llvm-symbolizer"):
            with contextlib.redirect_stderr(error):
                self.assertEqual(bt.main(["--d3d12", os.path.join(self.dir, "x.log"), "--symbols", self.dir]), 2)
        self.assertIn("--symbols needs the startup trace", error.getvalue())


if __name__ == "__main__":
    unittest.main()
