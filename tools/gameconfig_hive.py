#!/usr/bin/env python3
"""gameconfig_hive: read the package's `microsoft_gameconfig.bin` and check it against `MicrosoftGame.Config`.

`microsoft_gameconfig.bin` is a Windows registry hive file (`regf`, format 1.3): the packaging tool compiles
`MicrosoftGame.Config` into it. The root key is named by a GUID in braces; the config's identity and ids are values
of the root key, and each section (`ShellVisuals`, `PersistentLocalStorage`, `VirtualMachine`, one numbered subkey
per `<Executable>`) is a subkey. See docs/package_metadata.md for the measured layout and the mapping.

Standard library only. Run with `python -I tools/gameconfig_hive.py <command> ...`.

Commands
  dump <hive> [--json]       print every key and value (with --json: one JSON document)
  check <game folder>        compare each hive value with the config element or attribute it is compiled from;
                             prints one line per value (`match`, `differs`, `hive only`, `config only`) and a
                             summary. --values adds the two values to `differs` lines.
                             --hive / --config read other files than the folder's two.

Exit status: 0 when every config value is in the hive with the same value, 1 when one differs or is missing from
the hive, 2 when an input is missing or is not a hive / config this tool can read (one ERROR line).

The values are the package's own metadata (identity, ids, display names). Print them locally; share counts and
statuses, not values.
"""
import argparse
import json
import os
import struct
import sys
import xml.etree.ElementTree as ET

HIVE_NAME = "microsoft_gameconfig.bin"
CONFIG_NAME = "MicrosoftGame.Config"

BASE_BLOCK = 0x1000          # the hive bins start after the 4 KiB base block; cell offsets count from here
MAX_DEPTH = 64               # deeper than any config; a deeper tree is a corrupt or hostile file
REG_TYPES = {0: "REG_NONE", 1: "REG_SZ", 2: "REG_EXPAND_SZ", 3: "REG_BINARY", 4: "REG_DWORD",
             5: "REG_DWORD_BIG_ENDIAN", 6: "REG_LINK", 7: "REG_MULTI_SZ", 11: "REG_QWORD"}
# Config fields written as hexadecimal digits that the hive stores as numbers (TitleId as a DWORD, MSAAppId as a
# QWORD). Every other numeric field (SizeMB, configVersion) is decimal; booleans are "true"/"false" -> 1/0.
HEX_FIELDS = {"TitleId", "MSAAppId"}


class HiveError(ValueError):
    """The file is not a registry hive this reader handles, or a cell points outside it."""


class Key:
    def __init__(self, name):
        self.name = name
        self.values = []    # [(name, type, bytes)]
        self.subkeys = []   # [Key]


# ---------------------------------------------------------------------------------------------- hive reader
def _checksum(data):
    x = 0
    for (d,) in struct.iter_unpack("<I", data[:508]):
        x ^= d
    if x == 0xFFFFFFFF:
        return 0xFFFFFFFE
    return x or 1


class Hive:
    def __init__(self, data):
        if len(data) < BASE_BLOCK or data[:4] != b"regf":
            raise HiveError("not a registry hive (no 'regf' base block)")
        seq1, seq2 = struct.unpack_from("<II", data, 4)
        self.major, self.minor = struct.unpack_from("<II", data, 20)
        self.root_offset, self.bins_size = struct.unpack_from("<II", data, 36)
        self.file_name = data[48:48 + 64].decode("utf-16-le", "replace").split("\0")[0]
        stored = struct.unpack_from("<I", data, 508)[0]
        if stored != _checksum(data):
            raise HiveError("base block checksum %#010x does not match its contents (%#010x)" % (stored, _checksum(data)))
        if self.major != 1:
            raise HiveError("hive format %d.%d is not 1.x" % (self.major, self.minor))
        self.clean = seq1 == seq2  # unequal: a write was interrupted and the log files hold the rest
        self.data = data
        self.end = min(len(data), BASE_BLOCK + self.bins_size)
        if data[BASE_BLOCK:BASE_BLOCK + 4] != b"hbin":
            raise HiveError("no 'hbin' at the start of the hive bins")

    def cell(self, offset):
        """The payload of the allocated cell at `offset` (relative to the hive bins)."""
        start = BASE_BLOCK + offset
        if offset < 0 or start + 4 > self.end:
            raise HiveError("cell offset %#x is outside the hive bins" % offset)
        size = struct.unpack_from("<i", self.data, start)[0]
        if size >= 0:
            raise HiveError("cell %#x is free, not allocated" % offset)
        if start - size > self.end or -size < 8:
            raise HiveError("cell %#x has a size (%d) that runs outside the hive bins" % (offset, -size))
        return self.data[start + 4:start - size]

    def root(self):
        return self._key(self.root_offset, 0, set())

    def _name(self, raw, compressed):
        return raw.decode("latin-1") if compressed else raw.decode("utf-16-le", "replace")

    def _key(self, offset, depth, seen):
        if depth > MAX_DEPTH:
            raise HiveError("key tree deeper than %d levels" % MAX_DEPTH)
        if offset in seen:
            raise HiveError("key %#x is reached twice (a cycle)" % offset)
        seen.add(offset)
        c = self.cell(offset)
        if c[:2] != b"nk" or len(c) < 76:
            raise HiveError("cell %#x is not a key node" % offset)
        flags = struct.unpack_from("<H", c, 2)[0]
        n_sub, = struct.unpack_from("<I", c, 20)
        sub_list, = struct.unpack_from("<I", c, 28)
        n_val, val_list = struct.unpack_from("<II", c, 36)
        name_len = struct.unpack_from("<H", c, 72)[0]
        if 76 + name_len > len(c):
            raise HiveError("key %#x has a name longer than its cell" % offset)
        key = Key(self._name(c[76:76 + name_len], flags & 0x20))
        if n_val:
            vl = self.cell(val_list)
            if 4 * n_val > len(vl):
                raise HiveError("key %#x lists %d values in a %d-byte cell" % (offset, n_val, len(vl)))
            for i in range(n_val):
                key.values.append(self._value(struct.unpack_from("<I", vl, 4 * i)[0]))
        if n_sub:
            offsets = self._subkey_offsets(sub_list, 0)
            if len(offsets) != n_sub:
                raise HiveError("key %#x counts %d subkeys, its list holds %d" % (offset, n_sub, len(offsets)))
            key.subkeys = [self._key(o, depth + 1, seen) for o in offsets]
        return key

    def _subkey_offsets(self, offset, depth):
        c = self.cell(offset)
        sig, count = c[:2], struct.unpack_from("<H", c, 2)[0]
        step = {b"li": 4, b"ri": 4, b"lf": 8, b"lh": 8}.get(sig)
        if step is None:
            raise HiveError("cell %#x is not a subkey list (%r)" % (offset, sig))
        if 4 + step * count > len(c):
            raise HiveError("subkey list %#x holds %d entries in a %d-byte cell" % (offset, count, len(c)))
        entries = [struct.unpack_from("<I", c, 4 + step * i)[0] for i in range(count)]
        if sig != b"ri":
            return entries
        if depth:
            raise HiveError("index root %#x inside another index root" % offset)
        out = []
        for e in entries:
            out += self._subkey_offsets(e, depth + 1)
        return out

    def _value(self, offset):
        c = self.cell(offset)
        if c[:2] != b"vk" or len(c) < 20:
            raise HiveError("cell %#x is not a value" % offset)
        name_len, size, data_off, vtype, flags = struct.unpack_from("<HIIIH", c, 2)
        name = self._name(c[20:20 + name_len], flags & 1)
        if size & 0x80000000:                    # up to four bytes stored in the offset field itself
            size &= 0x7FFFFFFF
            if size > 4:
                raise HiveError("value %#x claims %d inline bytes" % (offset, size))
            data = struct.pack("<I", data_off)[:size]
        elif size == 0:
            data = b""
        else:
            d = self.cell(data_off)
            if d[:2] == b"db" and size > len(d):
                raise HiveError("value %#x uses a big-data record, which this reader does not handle" % offset)
            if size > len(d):
                raise HiveError("value %#x has %d bytes of data in a %d-byte cell" % (offset, size, len(d)))
            data = d[:size]
        return name, vtype, data


def decode(vtype, data):
    """A value's data as Python: str for strings, int for numbers, list for REG_MULTI_SZ, hex for the rest."""
    if vtype in (1, 2, 6):
        return data.decode("utf-16-le", "replace").split("\0")[0]
    if vtype == 7:
        return [s for s in data.decode("utf-16-le", "replace").split("\0") if s]
    if vtype == 4 and len(data) == 4:
        return struct.unpack("<I", data)[0]
    if vtype == 5 and len(data) == 4:
        return struct.unpack(">I", data)[0]
    if vtype == 11 and len(data) == 8:
        return struct.unpack("<Q", data)[0]
    return data.hex()


def flatten(root):
    """{path: (type, decoded)} for every value; paths leave out the root key's name ("Executable/00000000/Name")."""
    out = {}

    def walk(key, prefix):
        for name, vtype, data in key.values:
            out[prefix + (name or "@")] = (vtype, decode(vtype, data))
        for sub in key.subkeys:
            walk(sub, prefix + sub.name + "/")

    walk(root, "")
    return out


def key_tree(key):
    return {"name": key.name,
            "values": [{"name": n, "type": REG_TYPES.get(t, t), "data": decode(t, d)} for n, t, d in key.values],
            "subkeys": [key_tree(s) for s in key.subkeys]}


# ---------------------------------------------------------------------------------------------- config side
def _local(tag):
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _leaves(elem, prefix, out):
    for k, v in elem.attrib.items():
        out[prefix + _local(k)] = v
    for child in elem:
        tag = _local(child.tag)
        if not tag:
            continue
        if len(child) == 0 and not child.attrib:
            out[prefix + tag] = (child.text or "").strip()
        else:
            _leaves(child, prefix + tag + "/", out)


def config_fields(root):
    """{hive path: text} for every value MicrosoftGame.Config declares, named the way the hive names it.

    `<Game>`'s attributes, its text-only children and `<Identity>`'s attributes are root values; `<Resources>` is the
    root value `Languages`; an `<XxxList>` is the key `Xxx` with one numbered subkey (`%08d`) per entry; any other
    section is a key of its own name holding its attributes and text-only children. Package 1.0.2.1 has one list
    with entries (`ExecutableList`); other lists are mapped the same way without a package to confirm it.
    """
    out = {}
    for k, v in root.attrib.items():
        out[_local(k)] = v
    for child in root:
        tag = _local(child.tag)
        if not tag:
            continue
        if tag == "Identity":
            for k, v in child.attrib.items():
                out[_local(k)] = v
        elif tag == "Resources":
            out["Languages"] = [r.get("Language", "") for r in child if _local(r.tag) == "Resource"]
        elif tag.endswith("List") and len(tag) > 4:
            items = [c for c in child if _local(c.tag)]
            for i, item in enumerate(items):
                _leaves(item, "%s/%08d/" % (tag[:-4], i), out)
        elif len(child) == 0 and not child.attrib:
            out[tag] = (child.text or "").strip()
        else:
            _leaves(child, tag + "/", out)
    return out


def _same(path, vtype, hive_value, text):
    """Whether the config text compiles to the hive value. Returns (bool, how the config text reads)."""
    field = path.rsplit("/", 1)[-1]
    if isinstance(text, list):          # Languages: one REG_SZ, ';'-separated (the package ends it with ';')
        if not isinstance(hive_value, str):
            return False, text
        return [s for s in hive_value.split(";") if s] == text, text
    if vtype in (4, 5, 11):
        t = text.strip().lower()
        try:
            want = {"true": 1, "false": 0}[t] if t in ("true", "false") else int(t, 16 if field in HEX_FIELDS else 10)
        except ValueError:
            return False, text
        return want == hive_value, want
    return hive_value == text, text


def compare(hive_fields, cfg_fields):
    """[(status, path, hive value, config value)] in hive order, then the config-only fields."""
    rows = []
    for path, (vtype, value) in hive_fields.items():
        if path not in cfg_fields:
            rows.append(("hive only", path, value, None))
            continue
        ok, want = _same(path, vtype, value, cfg_fields[path])
        rows.append(("match" if ok else "differs", path, value, want))
    for path, text in cfg_fields.items():
        if path not in hive_fields:
            rows.append(("config only", path, None, text))
    return rows


# ---------------------------------------------------------------------------------------------- command line
def read_hive(path):
    with open(path, "rb") as f:
        return Hive(f.read())


def read_config(path):
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as e:
        raise HiveError("%s is not well-formed XML: %s" % (path, e)) from None


def cmd_dump(args):
    hive = read_hive(args.hive)
    root = hive.root()
    if args.json:
        doc = {"format": "%d.%d" % (hive.major, hive.minor), "clean": hive.clean, "file_name": hive.file_name,
               "root": key_tree(root)}
        json.dump(doc, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    print("regf %d.%d, %s, root key %s" % (hive.major, hive.minor, "clean" if hive.clean else "DIRTY (log pending)",
                                          root.name))

    def show(key, depth):
        pad = "  " * depth
        for name, vtype, data in key.values:
            print("%s  %s (%s) = %r" % (pad, name or "@", REG_TYPES.get(vtype, vtype), decode(vtype, data)))
        for sub in key.subkeys:
            print("%s  [%s]" % (pad, sub.name))
            show(sub, depth + 1)

    show(root, 0)
    return 0


def cmd_check(args):
    hive_path = args.hive or os.path.join(args.folder, HIVE_NAME)
    cfg_path = args.config or os.path.join(args.folder, CONFIG_NAME)
    for p in (hive_path, cfg_path):
        if not os.path.isfile(p):
            raise HiveError("%s is not a file" % p)
    hive = read_hive(hive_path)
    rows = compare(flatten(hive.root()), config_fields(read_config(cfg_path)))
    counts = {}
    for status, path, have, want in rows:
        counts[status] = counts.get(status, 0) + 1
        line = "%-11s %s" % (status, path)
        if args.values and status == "differs":
            line += "  hive=%r config=%r" % (have, want)
        print(line)
    if not hive.clean:
        print("note: the hive's sequence numbers differ, so it may be missing a logged write")
    print("summary: " + ", ".join("%d %s" % (counts.get(s, 0), s) for s in ("match", "differs", "hive only",
                                                                            "config only")))
    return 1 if counts.get("differs") or counts.get("config only") else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump", help="print the hive's keys and values")
    d.add_argument("hive")
    d.add_argument("--json", action="store_true", help="one JSON document instead of the indented listing")
    c = sub.add_parser("check", help="compare the hive with MicrosoftGame.Config")
    c.add_argument("folder", nargs="?", default=".", help="the game folder that holds both files")
    c.add_argument("--hive", help="read this hive instead of <folder>/" + HIVE_NAME)
    c.add_argument("--config", help="read this config instead of <folder>/" + CONFIG_NAME)
    c.add_argument("--values", action="store_true", help="print both values on `differs` lines")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # a display name the console code page lacks
    try:
        return cmd_dump(args) if args.command == "dump" else cmd_check(args)
    except (OSError, HiveError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
