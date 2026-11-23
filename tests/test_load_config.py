"""Tests for tools/load_config.py on PE32+ images built here byte by byte: a clean image whose pointer slots hold the
linker's defaults, an image with every condition that needs a loader (zero and virtual slots, process settings,
dynamic relocations, CHPE, an enclave), an older short load-config layout, no load config, and inputs the tool
refuses. No game files, no third-party packages.

    python -I tests/test_load_config.py
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
TOOL = os.path.join(ROOT, "tools", "load_config.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import load_config as lc  # noqa: E402

BASE = 0x180000000
TEXT, RDATA, DATA = 0x1000, 0x2000, 0x3000      # section RVAs
SLOT = RDATA + 0x180                             # first function-pointer slot
VIRTUAL_SLOT = RDATA + 0x300                     # inside .rdata's virtual size, past its file bytes
COOKIE = DATA + 0x10


def build(lc_size=320, fields=None, slots=None, cookie=lc.DEFAULT_COOKIE_64, with_lc=True, magic=0x20B):
    """fields: {offset: (width, value)} written into the load config; slots: {index: value} for 8-byte slots at SLOT."""
    text = b"\xc3" * 0x200                                     # ret
    rdata = bytearray(0x200)
    if with_lc:
        struct.pack_into("<I", rdata, 0, lc_size)
        defaults = {lc.SECURITY_COOKIE: (8, BASE + COOKIE)}
        for i, (off, _name) in enumerate(lc.POINTER_SLOTS[:5]):
            defaults[off] = (8, BASE + SLOT + 8 * i)
        defaults[lc.GUARD_FLAGS] = (4, 0x10417500)
        defaults[lc.EH_CONT_TABLE] = (8, BASE + RDATA + 0x1C0)
        defaults[lc.EH_CONT_COUNT] = (8, 3)
        defaults.update(fields or {})
        for off, (width, value) in defaults.items():
            if off + width <= lc_size:
                rdata[off:off + width] = value.to_bytes(width, "little")
        values = {i: BASE + TEXT + 0x10 * i for i in range(5)}  # the linker's defaults: code in .text
        values.update(slots or {})
        for i, v in values.items():
            struct.pack_into("<Q", rdata, SLOT - RDATA + 8 * i, v)
    data = bytearray(0x200)
    struct.pack_into("<Q", data, COOKIE - DATA, cookie)

    sections = [(b".text", TEXT, 0x200, 0x400, 0x200, 0x60000020, text),
                (b".rdata", RDATA, 0x400, 0x600, 0x200, 0x40000040, bytes(rdata)),
                (b".data", DATA, 0x200, 0x800, 0x200, 0xC0000040, bytes(data))]
    dos = bytearray(0x40)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    opt_size = 0xF0 if magic == 0x20B else 0xE0
    coff = b"PE\0\0" + struct.pack("<HHIIIHH", 0x8664 if magic == 0x20B else 0x14C, len(sections), 0, 0, 0, opt_size,
                                   0x2022)
    opt = bytearray(opt_size)
    struct.pack_into("<H", opt, 0, magic)
    if magic == 0x20B:
        struct.pack_into("<Q", opt, 24, BASE)
        dirs = 112
    else:
        struct.pack_into("<I", opt, 28, 0x10000000)
        dirs = 96
    struct.pack_into("<II", opt, 32, 0x1000, 0x200)
    struct.pack_into("<II", opt, 56, 0x4000, 0x400)
    struct.pack_into("<I", opt, dirs - 4, 16)
    if with_lc:
        struct.pack_into("<II", opt, dirs + 8 * lc.LOAD_CONFIG, RDATA, lc_size)
    table = b"".join(struct.pack("<8sIIIIIIHHI", n, vs, rva, rs, rp, 0, 0, 0, 0, ch)
                     for n, rva, vs, rp, rs, ch, _ in sections)
    head = bytes(dos) + coff + bytes(opt) + table
    image = bytearray(head + bytes(0x400 - len(head)))
    for _n, _rva, _vs, rp, rs, _ch, body in sections:
        image[rp:rp + rs] = body
    return bytes(image)


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = lc.main(argv)
    return code, out.getvalue(), err.getvalue()


class Inspect(unittest.TestCase):
    def test_clean_image_needs_nothing(self):
        r = lc.inspect(build())
        self.assertEqual(r["attention"], [])
        self.assertEqual(r["security_cookie"], {"slot": "file", "section": ".data", "default": True})
        self.assertEqual(set(r["pointer_slots"].values()), {"own code (.text)"})
        self.assertEqual(len(r["pointer_slots"]), 5)
        self.assertEqual(r["eh_continuation_targets"], 3)
        self.assertIn("EH_CONTINUATION_TABLE_PRESENT", r["guard_flags"]["names"])
        self.assertIn("FUNCTION_TABLE_STRIDE_1", r["guard_flags"]["names"])

    def test_everything_that_needs_a_loader(self):
        fields = {lc.POINTER_SLOTS[5][0]: (8, BASE + VIRTUAL_SLOT),    # a slot in the virtual tail: 0 at load
                  lc.POINTER_SLOTS[6][0]: (8, BASE + 0x9000),          # a slot outside the image
                  72: (4, 0x40000), 64: (8, 0xFF),                     # ProcessHeapFlags, ProcessAffinityMask
                  lc.DEPENDENT_LOAD_FLAGS: (2, 0x800),
                  lc.CHPE_METADATA: (8, BASE + RDATA),
                  lc.DYN_RELOC_OFFSET: (4, 0x40), lc.DYN_RELOC_SECTION: (2, 2),
                  lc.ENCLAVE_CONFIG: (8, BASE + RDATA)}
        r = lc.inspect(build(fields=fields, slots={1: 0, 2: BASE + DATA}, cookie=0x1234))
        slots = r["pointer_slots"]
        self.assertEqual(slots["GuardCFDispatchFunctionPointer"], "zero")
        self.assertEqual(slots["GuardRFFailureRoutineFunctionPointer"], "own data (.data)")
        self.assertEqual(slots["GuardXFGDispatchFunctionPointer"], "zero")            # the virtual-tail slot
        self.assertEqual(slots["GuardXFGTableDispatchFunctionPointer"], "slot outside the image")
        self.assertFalse(r["security_cookie"]["default"])
        self.assertEqual(r["dependent_load_flags"], 0x800)
        text = "\n".join(r["attention"])
        for needle in ("ProcessAffinityMask = 0xff", "ProcessHeapFlags = 0x40000", "GuardCFDispatchFunctionPointer: zero",
                       "GuardXFGTableDispatchFunctionPointer: slot outside", "CHPE metadata",
                       "dynamic value relocations (section 2, offset 0x40)", "enclave"):
            self.assertIn(needle, text)
        self.assertEqual(len(r["attention"]), 8)

    def test_a_short_older_layout_reads_only_its_fields(self):
        r = lc.inspect(build(lc_size=0x70))  # ends before the CFG pointers (offset 112)
        self.assertEqual(r["pointer_slots"], {})
        self.assertNotIn("guard_flags", r)
        self.assertIn("security_cookie", r)
        self.assertEqual(r["attention"], [])

    def test_guard_flag_names(self):
        self.assertEqual(lc.guard_flag_names(0), [])
        self.assertEqual(lc.guard_flag_names(0x20000100), ["CF_INSTRUMENTED", "FUNCTION_TABLE_STRIDE_2"])
        self.assertEqual(lc.guard_flag_names(0x4000000), ["unknown 0x4000000"])


class CommandLine(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, name, data):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_exit_codes_and_output(self):
        clean, none = self.write("clean.dll", build()), self.write("none.dll", build(with_lc=False))
        code, out, err = run([clean, none])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("clean.dll: load config, 320 bytes", out)
        self.assertIn("  GuardCFCheckFunctionPointer: own code (.text)", out)
        self.assertIn("none.dll: no load-config directory", out)
        self.assertIn("summary: 2 image(s), 0 with an attention line", out)
        bad = self.write("game.exe", build(slots={0: 0}))
        code, out, _ = run([bad])
        self.assertEqual(code, 1)
        self.assertIn("  attention: GuardCFCheckFunctionPointer: zero;", out)
        p = subprocess.run([sys.executable, "-I", TOOL, bad, "--json"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertEqual(json.loads(p.stdout)["game.exe"]["pointer_slots"]["GuardCFCheckFunctionPointer"], "zero")

    def test_refusals(self):
        cases = ((self.write("x86.dll", build(magic=0x10B)), "32-bit"),
                 (self.write("notes.txt", b"hello"), "not a PE image"),
                 (self.write("huge.dll", build(lc_size=0x10000)), "does not fit"),
                 (os.path.join(self.dir, "missing.dll"), "missing.dll"))
        for path, why in cases:
            with self.subTest(why=why):
                code, out, err = run([path])
                self.assertEqual((code, out), (2, ""))
                self.assertEqual(len(err.splitlines()), 1)
                self.assertIn(why, err)


if __name__ == "__main__":
    unittest.main()
