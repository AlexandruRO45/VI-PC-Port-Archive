"""Tests for tools/gameconfig_hive.py: the registry hive reader (key nodes, values, every subkey-list kind, inline and
out-of-line data, compressed and UTF-16 names), its refusals of corrupt files, and the check against a config.
Every hive and config here is built by hand; no game data is used.

    python -I tests/test_gameconfig_hive.py
"""
import contextlib
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gameconfig_hive.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gameconfig_hive as gh  # noqa: E402

REG_SZ, REG_BINARY, REG_DWORD, REG_MULTI_SZ, REG_QWORD = 1, 3, 4, 7, 11


def sz(s):
    return (s + "\0").encode("utf-16-le")


class HiveBuilder:
    """Writes a minimal regf 1.3 file: base block, one hbin, allocated cells, the rest one free cell."""

    def __init__(self):
        self.cells = bytearray()

    def alloc(self, payload):
        size = (4 + len(payload) + 7) & ~7
        off = 0x20 + len(self.cells)  # cell offsets count from the start of the hive bins; the hbin header is 0x20
        self.cells += struct.pack("<i", -size) + payload + bytes(size - 4 - len(payload))
        return off

    def value(self, name, vtype, data, ascii_name=True):
        raw = name.encode("latin-1") if ascii_name else name.encode("utf-16-le")
        if len(data) <= 4:
            size, off = 0x80000000 | len(data), struct.unpack("<I", data.ljust(4, b"\0"))[0]
        else:
            size, off = len(data), self.alloc(data)
        return self.alloc(struct.pack("<2sHIIIHH", b"vk", len(raw), size, off, vtype, 1 if ascii_name else 0, 0) + raw)

    def key(self, name, values=(), subkeys=(), list_kind=b"lf", ascii_name=True):
        """values: offsets from value(); subkeys: offsets from key(). list_kind: lf, lh, li or ri (ri -> two li)."""
        val_list = self.alloc(b"".join(struct.pack("<I", v) for v in values)) if values else 0xFFFFFFFF
        sub_list = 0xFFFFFFFF
        if subkeys:
            if list_kind in (b"lf", b"lh"):
                body = b"".join(struct.pack("<II", s, 0) for s in subkeys)
                sub_list = self.alloc(list_kind + struct.pack("<H", len(subkeys)) + body)
            elif list_kind == b"li":
                sub_list = self.alloc(b"li" + struct.pack("<H", len(subkeys)) + b"".join(struct.pack("<I", s) for s in subkeys))
            else:
                half = (len(subkeys) + 1) // 2
                parts = [subkeys[:half], subkeys[half:]]
                lis = [self.alloc(b"li" + struct.pack("<H", len(p)) + b"".join(struct.pack("<I", s) for s in p))
                       for p in parts if p]
                sub_list = self.alloc(b"ri" + struct.pack("<H", len(lis)) + b"".join(struct.pack("<I", x) for x in lis))
        raw = name.encode("latin-1") if ascii_name else name.encode("utf-16-le")
        head = struct.pack("<2sHQ15I", b"nk", 0x20 if ascii_name else 0, 0, 0, 0, len(subkeys), 0, sub_list,
                           0xFFFFFFFF, len(values), val_list, 0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0)
        return self.alloc(head + struct.pack("<HH", len(raw), 0) + raw)

    def build(self, root, seq=(4, 4), file_name="gc.bin"):
        free = 0x1000 - ((0x20 + len(self.cells)) % 0x1000 or 0x1000)
        cells = bytes(self.cells) + (struct.pack("<i", free) + bytes(free - 4) if free else b"")
        bins = b"hbin" + struct.pack("<III", 0, 0x20 + len(cells), 0) + bytes(0x10) + cells
        base = bytearray(0x1000)
        struct.pack_into("<4sIIQIIIIIII", base, 0, b"regf", seq[0], seq[1], 0, 1, 3, 0, 1, root, len(bins), 1)
        name = file_name.encode("utf-16-le")[:64]
        base[48:48 + len(name)] = name
        x = 0
        for (d,) in struct.iter_unpack("<I", bytes(base[:508])):
            x ^= d
        struct.pack_into("<I", base, 508, 0xFFFFFFFE if x == 0xFFFFFFFF else (x or 1))
        return bytes(base) + bins


def sample_hive(title_id=0x1234ABCD, extra_value=False, seq=(4, 4)):
    """A hive shaped like the package's: identity and ids on the root, one numbered executable, three sections."""
    b = HiveBuilder()
    exe = b.key("00000000", [b.value("Name", REG_SZ, sz("game.exe")), b.value("IsDevOnly", REG_DWORD, struct.pack("<I", 0))])
    executable = b.key("Executable", subkeys=[exe], list_kind=b"lh")
    vm = b.key("VirtualMachine", [b.value("SMTEnabled", REG_DWORD, struct.pack("<I", 1)),
                                  b.value("XboxAnacondaTitleMemory", REG_SZ, sz("Advanced"))])
    pls = b.key("PersistentLocalStorage", [b.value("SizeMB", REG_DWORD, struct.pack("<I", 512))])
    shell = b.key("ShellVisuals", [b.value("DefaultDisplayName", REG_SZ, sz("Test Title é"))])
    protocol = b.key("Protocol")
    values = [b.value("GameConfigRegRWVersion", REG_DWORD, struct.pack("<I", 1)),
              b.value("configVersion", REG_DWORD, struct.pack("<I", 1)),
              b.value("Name", REG_SZ, sz("Example.Title")),
              b.value("Languages", REG_SZ, sz("en-US;fr-FR;")),
              b.value("MSAAppId", REG_QWORD, struct.pack("<Q", 0x00000000A1B2C3D4)),
              b.value("TitleId", REG_DWORD, struct.pack("<I", title_id))]
    if extra_value:
        values.append(b.value("AdvancedUserModel", REG_DWORD, struct.pack("<I", 1)))
    root = b.key("{00000000-1111-2222-3333-444444444444}", values,
                 [executable, protocol, pls, shell, vm], list_kind=b"lf")
    return b.build(root, seq=seq)


CONFIG = """﻿<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <MSAAppId>00000000A1B2C3D4</MSAAppId>
  <TitleId>{title}</TitleId>
  <Identity Name="Example.Title" />
  <ShellVisuals DefaultDisplayName="Test Title é" />
  <ExecutableList>
    <Executable Name="game.exe" IsDevOnly="false" />
  </ExecutableList>
  <PersistentLocalStorage>
    <SizeMB>512</SizeMB>
  </PersistentLocalStorage>
  <VirtualMachine>
    <SMTEnabled>true</SMTEnabled>
    <XboxAnacondaTitleMemory>Advanced</XboxAnacondaTitleMemory>
  </VirtualMachine>
  <Resources>
    <Resource Language="en-US" />
    <Resource Language="fr-FR" />
  </Resources>{extra}
</Game>
"""


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gh.main(argv)
    return code, out.getvalue(), err.getvalue()


class Reader(unittest.TestCase):
    def test_tree_and_value_types(self):
        hive = gh.Hive(sample_hive(extra_value=True))
        self.assertTrue(hive.clean)
        self.assertEqual((hive.major, hive.minor, hive.file_name), (1, 3, "gc.bin"))
        root = hive.root()
        self.assertEqual(root.name, "{00000000-1111-2222-3333-444444444444}")
        self.assertEqual([k.name for k in root.subkeys],
                         ["Executable", "Protocol", "PersistentLocalStorage", "ShellVisuals", "VirtualMachine"])
        flat = gh.flatten(root)
        self.assertEqual(flat["TitleId"], (REG_DWORD, 0x1234ABCD))
        self.assertEqual(flat["MSAAppId"], (REG_QWORD, 0xA1B2C3D4))         # out-of-line 8-byte data
        self.assertEqual(flat["Languages"], (REG_SZ, "en-US;fr-FR;"))
        self.assertEqual(flat["Executable/00000000/Name"], (REG_SZ, "game.exe"))
        self.assertEqual(flat["ShellVisuals/DefaultDisplayName"], (REG_SZ, "Test Title é"))
        self.assertNotIn("Protocol", "".join(flat))                          # an empty key has no values

    def test_every_subkey_list_kind_and_utf16_names(self):
        for kind in (b"lf", b"lh", b"li", b"ri"):
            with self.subTest(kind=kind):
                b = HiveBuilder()
                subs = [b.key("Sub%d" % i, [b.value("V", REG_DWORD, struct.pack("<I", i))]) for i in range(5)]
                subs.append(b.key("Wideß", [b.value("Näme", REG_BINARY, b"\x01\x02\x03\x04\x05", ascii_name=False)],
                                  ascii_name=False))
                root = gh.Hive(b.build(b.key("R", subkeys=subs, list_kind=kind))).root()
                self.assertEqual([k.name for k in root.subkeys], ["Sub0", "Sub1", "Sub2", "Sub3", "Sub4", "Wideß"])
                flat = gh.flatten(root)
                self.assertEqual(flat["Sub3/V"], (REG_DWORD, 3))
                self.assertEqual(flat["Wideß/Näme"], (REG_BINARY, "0102030405"))

    def test_decode(self):
        self.assertEqual(gh.decode(REG_MULTI_SZ, sz("a") + sz("bc") + b"\0\0"), ["a", "bc"])
        self.assertEqual(gh.decode(5, b"\x00\x00\x01\x00"), 256)              # REG_DWORD_BIG_ENDIAN
        self.assertEqual(gh.decode(REG_DWORD, b"\x01\x00"), "0100")           # a short DWORD stays raw


class Refusals(unittest.TestCase):
    def test_not_a_hive(self):
        with self.assertRaisesRegex(gh.HiveError, "regf"):
            gh.Hive(b"<Game/>" + bytes(0x2000))

    def test_checksum(self):
        data = bytearray(sample_hive())
        data[0x30] ^= 0xFF  # inside the file name, covered by the checksum
        with self.assertRaisesRegex(gh.HiveError, "checksum"):
            gh.Hive(bytes(data))

    def test_offsets_outside_the_bins(self):
        b = HiveBuilder()
        b.key("R")
        with self.assertRaisesRegex(gh.HiveError, "outside"):
            gh.Hive(b.build(0x7FFF0000)).root()  # a root offset past the end of the bins

    def test_cycle(self):
        b = HiveBuilder()
        # A key whose subkey list names the key itself: the list is written before the key, so point it at the key's
        # future offset (the next allocation after the list).
        list_off = 0x20 + len(b.cells)
        key_off = list_off + 16
        self.assertEqual(b.alloc(b"lf" + struct.pack("<HII", 1, key_off, 0)), list_off)
        head = struct.pack("<2sHQ15I", b"nk", 0x20, 0, 0, 0, 1, 0, list_off, 0xFFFFFFFF, 0, 0xFFFFFFFF,
                           0xFFFFFFFF, 0xFFFFFFFF, 0, 0, 0, 0, 0)
        self.assertEqual(b.alloc(head + struct.pack("<HH", 1, 0) + b"R"), key_off)
        with self.assertRaisesRegex(gh.HiveError, "cycle"):
            gh.Hive(b.build(key_off)).root()

    def test_inline_and_big_data(self):
        b = HiveBuilder()
        bad = b.alloc(struct.pack("<2sHIIIHH", b"vk", 1, 0x80000008, 0, REG_QWORD, 1, 0) + b"X")
        with self.assertRaisesRegex(gh.HiveError, "inline"):
            gh.Hive(b.build(b.key("R", [bad]))).root()
        b = HiveBuilder()
        db = b.alloc(b"db" + struct.pack("<HI", 2, 0))
        big = b.alloc(struct.pack("<2sHIIIHH", b"vk", 1, 20000, db, REG_BINARY, 1, 0) + b"Y")
        with self.assertRaisesRegex(gh.HiveError, "big-data"):
            gh.Hive(b.build(b.key("R", [big]))).root()


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def folder(self, hive, title="1234ABCD", extra=""):
        with open(os.path.join(self.tmp.name, gh.HIVE_NAME), "wb") as f:
            f.write(hive)
        with open(os.path.join(self.tmp.name, gh.CONFIG_NAME), "w", encoding="utf-8", newline="\n") as f:
            f.write(CONFIG.format(title=title, extra=extra))
        return self.tmp.name

    def test_match(self):
        code, out, err = run(["check", self.folder(sample_hive(extra_value=True))])
        self.assertEqual((code, err), (0, ""))
        self.assertIn("match       Executable/00000000/IsDevOnly", out)   # "false" -> DWORD 0
        self.assertIn("match       VirtualMachine/SMTEnabled", out)       # "true" -> DWORD 1
        self.assertIn("match       Languages", out)                        # list -> "a;b;"
        self.assertIn("hive only   AdvancedUserModel", out)               # informational, not a failure
        self.assertIn("summary: 11 match, 0 differs, 2 hive only, 0 config only", out)

    def test_hex_ids_are_hex_even_when_all_digits(self):
        code, out, _ = run(["check", self.folder(sample_hive(title_id=0x12345678), title="12345678")])
        self.assertEqual(code, 0)
        self.assertIn("match       TitleId", out)

    def test_differs(self):
        folder = self.folder(sample_hive(), title="1234ABCE")
        code, out, _ = run(["check", folder])
        self.assertEqual(code, 1)
        self.assertIn("differs     TitleId\n", out)
        _, out, _ = run(["check", folder, "--values"])
        self.assertIn("differs     TitleId  hive=305441741 config=305441742", out)

    def test_config_only(self):
        code, out, _ = run(["check", self.folder(sample_hive(), extra="\n  <StoreId>9ABC</StoreId>")])
        self.assertEqual(code, 1)
        self.assertIn("config only StoreId", out)

    def test_namespaced_config(self):
        folder = self.folder(sample_hive())
        path = os.path.join(folder, gh.CONFIG_NAME)
        with open(path, encoding="utf-8-sig") as f:
            text = f.read().replace('<Game configVersion="1">', '<Game xmlns="urn:example" configVersion="1">')
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        code, out, _ = run(["check", folder])
        self.assertEqual(code, 0, out)

    def test_dirty_hive_is_noted(self):
        code, out, _ = run(["check", self.folder(sample_hive(seq=(5, 4)))])
        self.assertEqual(code, 0)
        self.assertIn("note: the hive's sequence numbers differ", out)

    def test_inputs_that_cannot_be_read(self):
        code, out, err = run(["check", self.tmp.name])
        self.assertEqual((code, out), (2, ""))
        self.assertEqual(len(err.splitlines()), 1)
        self.assertTrue(err.startswith("ERROR: ") and gh.HIVE_NAME in err)
        folder = self.folder(sample_hive())
        with open(os.path.join(folder, gh.CONFIG_NAME), "w", encoding="utf-8", newline="\n") as f:
            f.write("<Game><TitleId>")
        code, _, err = run(["check", folder])
        self.assertEqual(code, 2)
        self.assertIn("not well-formed XML", err)
        code, _, err = run(["dump", os.path.join(folder, gh.CONFIG_NAME)])
        self.assertEqual(code, 2)
        self.assertIn("not a registry hive", err)

    def test_dump_and_script(self):
        folder = self.folder(sample_hive())
        code, out, _ = run(["dump", os.path.join(folder, gh.HIVE_NAME)])
        self.assertEqual(code, 0)
        self.assertIn("regf 1.3, clean, root key {00000000-1111-2222-3333-444444444444}", out)
        self.assertIn("  [VirtualMachine]\n    SMTEnabled (REG_DWORD) = 1", out)
        p = subprocess.run([sys.executable, "-I", TOOL, "dump", os.path.join(folder, gh.HIVE_NAME), "--json"],
                           capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('"SMTEnabled"', p.stdout)
        self.assertIn('"Test Title \\u00e9"', p.stdout)


if __name__ == "__main__":
    unittest.main()
