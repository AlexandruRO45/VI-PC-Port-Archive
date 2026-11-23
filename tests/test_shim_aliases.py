"""Tests for tools/shim_aliases.py: reading SHIM_TABLE rows (literal rows, dll-taking macros, nested and stringizing
macros, macros from a header, commented-out rows), the contract rule (last version segment dropped), and the bypass
report on a PE32+ image built here byte by byte with import and delay-import tables. A last test reads this
checkout's own runtime/shims and checks rows known to be there. No game files, no third-party packages.

    python -I tests/test_shim_aliases.py
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "shim_aliases.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import shim_aliases as sa  # noqa: E402

HEADER = r'''
#pragma once
#define SHARED_ROWS(dll) {dll, "SharedFn", reinterpret_cast<void*>(&shared_fn)}
'''

SOURCE = r'''
#include "shims.h"
namespace {
#define EXIT_ROWS(dll) {dll, "ExitProcess", reinterpret_cast<void*>(&game_exit)}
#define FILE_ROWS(dll)                                          \
    {dll, "CreateFileW", reinterpret_cast<void*>(&create_file)}, \
    {dll, "ReadFile", reinterpret_cast<void*>(&read_file)},     \
    EXIT_ROWS(dll)
#define NAMED(name) {"xnamed.dll", #name, reinterpret_cast<void*>(&name)}
}  // namespace

// {"commented.dll", "NotARow", nullptr},
SHIM_TABLE(sample,
    FILE_ROWS("kernel32.dll"),
    FILE_ROWS("api-ms-win-core-file-l1-1-0.dll"),
    EXIT_ROWS("API-MS-WIN-CORE-PROCESSTHREADS-L1-1-0.DLL"),
    SHARED_ROWS("kernelbase.dll"),
    NAMED(XNamedThing),
    /* {"blocked.dll", "AlsoNotARow", nullptr}, */
    {"xmem.dll", "XMemAlloc", reinterpret_cast<void*>(&x_mem_alloc)},
    {"weird//name.dll", "Slashes", nullptr},
);
'''


def build_pe(imports, delay_imports=()):
    """A PE32+ image with one section holding the import and delay-import tables.

    imports / delay_imports: [(dll, [function name or ordinal int])]."""
    sect_rva, sect_raw = 0x1000, 0x200
    blob = bytearray()

    def put(data, align=2):
        while len(blob) % align:
            blob.append(0)
        at = len(blob)
        blob.extend(data)
        return sect_rva + at

    def thunks(funcs):
        entries = []
        for f in funcs:
            if isinstance(f, int):
                entries.append((1 << 63) | f)
            else:
                entries.append(put(struct.pack("<H", 0) + f.encode() + b"\0"))
        return put(b"".join(struct.pack("<Q", e) for e in entries) + bytes(8), 8)

    norm = []
    for dll, funcs in imports:
        name = put(dll.encode() + b"\0")
        lookup = thunks(funcs)
        iat = thunks(funcs)
        norm.append(struct.pack("<IIIII", lookup, 0, 0, name, iat))
    import_dir = put(b"".join(norm) + bytes(20), 4)
    delays = []
    for dll, funcs in delay_imports:
        name = put(dll.encode() + b"\0")
        names = thunks(funcs)
        iat = thunks(funcs)
        delays.append(struct.pack("<IIIIIIII", 1, name, 0, iat, names, 0, 0, 0))
    delay_dir = put(b"".join(delays) + bytes(32), 4) if delays else 0

    vsize = len(blob)
    raw = bytes(blob) + bytes((-len(blob)) % 0x200)
    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0xF0, 0x2022)
    opt = bytearray(0xF0)
    struct.pack_into("<HBBIIIII", opt, 0, 0x20B, 14, 0, 0, 0, 0, 0, 0)
    struct.pack_into("<Q", opt, 24, 0x180000000)
    struct.pack_into("<II", opt, 32, 0x1000, 0x200)
    struct.pack_into("<II", opt, 56, sect_rva + ((vsize + 0xFFF) & ~0xFFF), sect_raw)
    struct.pack_into("<HH", opt, 68, 3, 0)
    struct.pack_into("<I", opt, 108, 16)
    struct.pack_into("<II", opt, 112 + 8 * 1, import_dir, 20 * (len(norm) + 1))
    if delays:
        struct.pack_into("<II", opt, 112 + 8 * 13, delay_dir, 32 * (len(delays) + 1))
    section = struct.pack("<8sIIIIIIHHI", b".idata", vsize, sect_rva, len(raw), sect_raw, 0, 0, 0, 0, 0xC0000040)
    headers = bytes(dos) + coff + bytes(opt) + section
    return headers + bytes(sect_raw - len(headers)) + raw


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sa.main(argv)
    return code, out.getvalue(), err.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.runtime = os.path.join(self.dir, "shims")
        os.mkdir(self.runtime)
        for name, text in (("shims_extra.h", HEADER), ("sample.cpp", SOURCE)):
            with open(os.path.join(self.runtime, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)

    def image(self, name, imports, delay=()):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(build_pe(imports, delay))
        return path


class Table(Fixture):
    def test_rows_from_literals_and_macros(self):
        table = sa.read_table(self.runtime)
        self.assertEqual(table["CreateFileW"], {"kernel32.dll", "api-ms-win-core-file-l1-1-0.dll"})
        self.assertEqual(table["ExitProcess"], {"kernel32.dll", "api-ms-win-core-file-l1-1-0.dll",
                                                "api-ms-win-core-processthreads-l1-1-0.dll"})  # nested, lower-cased
        self.assertEqual(table["SharedFn"], {"kernelbase.dll"})        # a macro the header defines
        self.assertEqual(table["XNamedThing"], {"xnamed.dll"})         # #name stringized
        self.assertEqual(table["XMemAlloc"], {"xmem.dll"})
        self.assertEqual(table["Slashes"], {"weird//name.dll"})        # "//" inside a string is not a comment
        self.assertNotIn("NotARow", table)
        self.assertNotIn("AlsoNotARow", table)

    def test_contract_drops_the_last_version_segment_only(self):
        self.assertEqual(sa.contract("api-ms-win-core-file-l1-2-0.dll"), "api-ms-win-core-file-l1-2")
        self.assertEqual(sa.contract("API-MS-Win-Core-ProcessThreads-L1-1-3.dll"), "api-ms-win-core-processthreads-l1-1")
        self.assertEqual(sa.contract("ext-ms-win-ntuser-window-l1-1-4.dll"), "ext-ms-win-ntuser-window-l1-1")
        self.assertIsNone(sa.contract("kernel32.dll"))
        self.assertNotEqual(sa.contract("api-ms-win-core-file-l1-1-0.dll"), sa.contract("api-ms-win-core-file-l1-2-0.dll"))

    def test_contracts_and_table_commands(self):
        code, out, _ = run(["--runtime", self.runtime, "contracts"])
        self.assertEqual(code, 0)
        self.assertIn("api-ms-win-core-file-l1-1: -0 (3)", out)
        self.assertIn("api-ms-win-core-processthreads-l1-1: -0 (1)", out)
        code, out, _ = run(["--runtime", self.runtime, "table", "--names"])
        self.assertEqual(code, 0)
        self.assertIn("ExitProcess: api-ms-win-core-file-l1-1-0.dll, api-ms-win-core-processthreads-l1-1-0.dll, "
                      "kernel32.dll", out)
        self.assertIn("summary: 11 rows, 7 functions, 7 DLL names", out)

    def test_a_folder_without_tables_is_an_error(self):
        empty = os.path.join(self.dir, "empty")
        os.mkdir(empty)
        code, out, err = run(["--runtime", empty, "table"])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("no SHIM_TABLE rows", err)


class Check(Fixture):
    def test_bypass_kinds(self):
        img = self.image("game.exe", [
            ("KERNEL32.dll", ["CreateFileW", "GetTickCount", 7]),               # shimmed, unshimmed, ordinal (not counted)
            ("api-ms-win-core-file-l1-1-1.dll", ["ReadFile"]),                  # version alias of a registered contract
            ("api-ms-win-core-processthreads-l1-1-2.dll", ["ExitProcess"]),    # version alias
            ("kernel32.dll", ["SharedFn"]),                                     # registered under kernelbase only
        ], delay=[("api-ms-win-core-file-l1-2-0.dll", ["CreateFileW"])])       # another contract, delay-loaded
        code, out, err = run(["--runtime", self.runtime, "check", img])
        self.assertEqual((code, err), (1, ""))
        lines = out.splitlines()
        self.assertIn("bypass  game.exe  api-ms-win-core-file-l1-1-1.dll!ReadFile  (shim under: "
                      "api-ms-win-core-file-l1-1-0.dll, kernel32.dll; version alias)", lines)
        self.assertIn("bypass  game.exe  api-ms-win-core-processthreads-l1-1-2.dll!ExitProcess  (shim under: "
                      "api-ms-win-core-file-l1-1-0.dll, api-ms-win-core-processthreads-l1-1-0.dll, kernel32.dll; "
                      "version alias)", lines)
        self.assertIn("bypass  game.exe  kernel32.dll!SharedFn  (shim under: kernelbase.dll)", lines)
        self.assertIn("bypass  game.exe  api-ms-win-core-file-l1-2-0.dll!CreateFileW [delay]  (shim under: "
                      "api-ms-win-core-file-l1-1-0.dll, kernel32.dll)", lines)
        self.assertIn("summary: game.exe: 6 imports, 1 through a shim, 4 bypass one", lines)

    def test_no_bypass_exits_0_and_json(self):
        img = self.image("clean.dll", [("kernel32.dll", ["CreateFileW", "ExitProcess"]), ("xmem.dll", ["XMemAlloc"])])
        code, out, _ = run(["--runtime", self.runtime, "check", img])
        self.assertEqual(code, 0)
        self.assertIn("summary: clean.dll: 3 imports, 3 through a shim, 0 bypass one", out)
        p = subprocess.run([sys.executable, "-I", TOOL, "--runtime", self.runtime, "check", img, "--json"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout), [{"image": "clean.dll", "imports": 3, "shimmed": 3, "bypass": []}])

    def test_unreadable_inputs_exit_2(self):
        text = os.path.join(self.dir, "notes.txt")
        with open(text, "w", encoding="utf-8", newline="\n") as f:
            f.write("not a PE")
        for path, why in ((text, "not a PE image"), (os.path.join(self.dir, "missing.exe"), "missing.exe")):
            code, out, err = run(["--runtime", self.runtime, "check", path])
            self.assertEqual((code, out), (2, ""))
            self.assertEqual(len(err.splitlines()), 1)
            self.assertIn(why, err)


class ThisCheckout(unittest.TestCase):
    def test_reads_the_repository_shims(self):
        table = sa.read_table(sa.DEFAULT_RUNTIME)
        # Rows written as macro calls in runtime/shims/process.cpp and xmem.cpp.
        self.assertIn("api-ms-win-core-processthreads-l1-1-0.dll", table.get("ExitProcess", set()))
        self.assertIn("kernel32.dll", table.get("ExitProcess", set()))
        self.assertIn("xmem.dll", {d for dlls in table.values() for d in dlls})
        self.assertGreater(sum(len(d) for d in table.values()), 100)


if __name__ == "__main__":
    unittest.main()
