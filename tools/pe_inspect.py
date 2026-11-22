"""Inspect a PE image (exe or dll) for porting work.

Usage:
    python -I pe_inspect.py <image> [--game-dir DIR] [--out DIR]
    python -I pe_inspect.py <image> --pdb <file.pdb>

Writes <out>/<file>.json and <out>/<file>.md, where <file> is the image's full file name (game.exe.json,
game.dll.md), so an exe and a dll with the same stem get separate reports. Images with the same file name from
different directories get a numbered suffix (x.exe-2.json), and the name used is printed to stderr. The same file
given twice with the same import search folder is inspected once. Reads the file only; never runs it.

The debug directory (data directory 6) is listed entry by entry. A CodeView `RSDS` entry gives the GUID, age and
file name of the PDB the linker wrote for this image, which tells which build a PDB belongs to. `--pdb` reads that
PDB (MSF 7.0: the PDB info stream for the GUID, the DBI stream for the age the debuggers match on) and says whether
it is the one the image names; the exit code is 0 for a match, 1 for no match or a file that can't be read, and 2
for a usage error. Only the blocks that hold the stream directory and the start of those two streams are read, so a
PDB of several GB is not loaded into memory.

Standard library only: the headers, the section table, the import, delay-import, export, TLS, relocation and debug
directories are parsed with struct, so the tool runs under `python -I` without third-party packages and is covered
by the same standard-library-only CI job as the other tools.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import struct
import sys
import uuid
from collections import Counter

SYSTEM32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")

MACHINES = {0x8664: "x86-64", 0x14C: "x86", 0xAA64: "ARM64"}
SUBSYSTEMS = {2: "GUI", 3: "Console", 14: "Xbox"}

DIR_EXPORT = 0
DIR_IMPORT = 1
DIR_EXCEPTION = 3
DIR_BASERELOC = 5
DIR_DEBUG = 6
DIR_TLS = 9
DIR_DELAY_IMPORT = 13

MAX_ENTRIES = 65536  # a table this long is not a table; stops a malformed image from looping forever


class PeFormatError(ValueError):
    """The file is not a PE image this parser can read."""


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in Counter(data).values())


# An MSVC decorated name can be 4,096 characters (the documented limit of a decorated name); with the 512 this
# was, a long template import became "?" and a DLL with a long name lost its whole entry.
NAME_LIMIT = 4097


def _cstring(data: bytes, offset: int, limit: int = NAME_LIMIT) -> str | None:
    end = data.find(b"\0", offset, min(len(data), offset + limit))
    if end < 0:
        return None
    return data[offset:end].decode(errors="replace")


class PeImage:
    """The parts of a PE file this tool reports on, read in place from the file bytes."""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 0x40 or data[:2] != b"MZ":
            raise PeFormatError("no MZ header")
        self.nt = struct.unpack_from("<I", data, 0x3C)[0]
        if self.nt + 24 > len(data) or data[self.nt:self.nt + 4] != b"PE\0\0":
            raise PeFormatError("no PE signature")
        (self.machine, self.section_count, self.timestamp, _symbol_table, _symbols, opt_size,
         self.characteristics) = struct.unpack_from("<HHIIIHH", data, self.nt + 4)
        opt = self.nt + 24
        if opt + 2 > len(data):
            raise PeFormatError("truncated optional header")
        magic = struct.unpack_from("<H", data, opt)[0]
        if magic not in (0x10B, 0x20B):
            raise PeFormatError(f"unknown optional header magic 0x{magic:x}")
        # The declared optional header must contain every fixed field we read,
        # and the complete declared header must fit inside the file.
        if opt_size < 72 or opt + opt_size > len(data):
            raise PeFormatError("truncated optional header")
        self.is64 = magic == 0x20B
        self.entry_rva = struct.unpack_from("<I", data, opt + 16)[0]
        self.image_base = struct.unpack_from("<Q" if self.is64 else "<I", data, opt + (24 if self.is64 else 28))[0]
        self.size_of_image = struct.unpack_from("<I", data, opt + 56)[0]
        self.size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
        self.subsystem = struct.unpack_from("<H", data, opt + 68)[0]
        self.dll_characteristics = struct.unpack_from("<H", data, opt + 70)[0]

        # NumberOfRvaAndSizes counts the data directories: a slot past it is not a directory, whatever bytes it
        # holds (runtime/src/pe_loader.cpp reads them the same way). The count is the dword before the table.
        opt_end = min(len(data), opt + opt_size)
        dirs = opt + (112 if self.is64 else 96)
        count = struct.unpack_from("<I", data, dirs - 4)[0] if dirs <= opt_end else 0
        self.directories = []
        for i in range(16):
            at = dirs + i * 8
            present = i < count and at + 8 <= opt_end
            self.directories.append(struct.unpack_from("<II", data, at) if present else (0, 0))

        # A section's data can be virtual only, so lookups past SizeOfRawData have no file bytes to read.
        self.sections = []
        table = opt + opt_size
        for i in range(self.section_count):
            at = table + i * 40
            if at + 40 > len(data):
                raise PeFormatError("truncated section table")
            vsize, rva, rawsize, rawptr = struct.unpack_from("<IIII", data, at + 8)
            chars = struct.unpack_from("<I", data, at + 36)[0]
            self.sections.append({
                "name": data[at:at + 8].rstrip(b"\0").decode(errors="replace"),
                "rva": rva, "vsize": vsize, "rawsize": rawsize, "rawptr": rawptr,
                "exec": bool(chars & 0x20000000), "write": bool(chars & 0x80000000),
            })

    def directory(self, index: int) -> tuple[int, int]:
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def offset(self, rva: int) -> int | None:
        """File offset of an RVA, or None when no file bytes back it."""
        if rva < self.size_of_headers:
            return rva if rva < len(self.data) else None
        for s in self.sections:
            # A section is VirtualSize bytes in memory (SizeOfRawData when that is 0). File bytes past it are not
            # mapped, as in runtime/src/pe_loader.cpp: with SizeOfRawData larger than the span to the next
            # section, max(vsize, rawsize) answered the next section's RVAs from this one's raw tail.
            span = s["vsize"] or s["rawsize"]
            if s["rva"] <= rva < s["rva"] + span:
                delta = rva - s["rva"]
                if delta >= min(s["rawsize"], span):
                    return None
                at = s["rawptr"] + delta
                return at if at < len(self.data) else None
        return None

    def section_data(self, section: dict) -> bytes:
        at = section["rawptr"]
        if at >= len(self.data):
            return b""
        return self.data[at:at + min(section["rawsize"], len(self.data) - at)]

    def string_at(self, rva: int) -> str | None:
        at = self.offset(rva)
        return None if at is None else _cstring(self.data, at)

    def word(self, at: int) -> int:
        return struct.unpack_from("<Q" if self.is64 else "<I", self.data, at)[0]

    def thunk_names(self, lookup_rva: int) -> list[str]:
        """Names (or '#ordinal') of one lookup table, which is terminated by a zero entry."""
        ordinal_flag = 1 << (63 if self.is64 else 31)
        out = []
        width = 8 if self.is64 else 4
        for i in range(MAX_ENTRIES):
            at = self.offset(lookup_rva + i * width)
            if at is None:
                break
            value = self.word(at)
            if not value:
                break
            if value & ordinal_flag:
                out.append(f"#{value & 0xFFFF}")
                continue
            name = self.string_at((value & 0x7FFFFFFF) + 2)  # skip the 2-byte hint
            out.append(name if name is not None else "?")
        return out


def _imports(img: PeImage, kind: str, index: int, delay: bool = False) -> list[dict]:
    rva, size = img.directory(index)
    if not rva or not size:
        return []
    at = img.offset(rva)
    if at is None:
        return []

    entries = []
    step = 32 if delay else 20
    for i in range(MAX_ENTRIES):
        base = at + i * step
        if base + step > len(img.data):
            break
        if delay:
            (attributes, dll_rva, _handle, iat_rva, int_rva, _bound, _unload,
             _stamp) = struct.unpack_from("<IIIIIIII", img.data, base)
            if not dll_rva:
                break
            # dlattrRva (0x1) is set on every descriptor Windows writes; without it the fields are virtual
            # addresses, which the linker uses for 32-bit images. Convert them so both kinds read the same here.
            if not attributes & 1:
                dll_rva = dll_rva - img.image_base
                iat_rva = iat_rva - img.image_base
                int_rva = int_rva - img.image_base
        else:
            oft, _stamp, _forwarder, dll_rva, iat_rva = struct.unpack_from("<IIIII", img.data, base)
            int_rva = oft
            if not dll_rva:
                break
        if dll_rva < 0:
            continue
        dll = img.string_at(dll_rva)
        if dll is None:
            continue
        lookup = int_rva if int_rva > 0 else iat_rva
        entries.append({"dll": dll, "kind": kind, "functions": img.thunk_names(lookup) if lookup > 0 else []})
    return entries


def _exports(img: PeImage) -> list[str]:
    rva, size = img.directory(DIR_EXPORT)
    if not rva or size < 40:
        return []
    at = img.offset(rva)
    if at is None:
        return []
    (_chars, _stamp, _major, _minor, _name, base, functions, name_count, addr_functions,
     addr_names, addr_ordinals) = struct.unpack_from("<IIHHIIIIIII", img.data, at)

    def has_export_at(index: int) -> bool:
        if index >= functions:
            return False
        # A zero RVA marks an unused slot. An RVA inside the export directory is a valid forwarder.
        slot = img.offset(addr_functions + index * 4)
        return (slot is not None and slot + 4 <= len(img.data)
                and struct.unpack_from("<I", img.data, slot)[0] != 0)

    named, names = set(), []
    for i in range(min(name_count, MAX_ENTRIES)):
        name_at = img.offset(addr_names + i * 4)
        ord_at = img.offset(addr_ordinals + i * 2)
        if name_at is None or ord_at is None:
            break
        index = struct.unpack_from("<H", img.data, ord_at)[0]
        if not has_export_at(index):
            continue
        symbol = img.string_at(struct.unpack_from("<I", img.data, name_at)[0])
        names.append(symbol if symbol is not None else "?")
        named.add(index)
    for i in range(min(functions, MAX_ENTRIES)):
        if i not in named and has_export_at(i):
            names.append(f"#{base + i}")
    return names


def _tls_callbacks(img: PeImage) -> int:
    rva, size = img.directory(DIR_TLS)
    if not rva or size < (40 if img.is64 else 24):
        return 0
    at = img.offset(rva)
    if at is None:
        return 0
    callbacks_va = struct.unpack_from("<Q" if img.is64 else "<I", img.data, at + (24 if img.is64 else 12))[0]
    if not callbacks_va or callbacks_va < img.image_base:
        return 0
    callbacks = callbacks_va - img.image_base
    count = 0
    width = 8 if img.is64 else 4
    for i in range(MAX_ENTRIES):
        entry = img.offset(callbacks + i * width)
        if entry is None or not img.word(entry):
            break
        count += 1
    return count


# IMAGE_DEBUG_TYPE_* (winnt.h and the PE format specification), without the prefix.
DEBUG_TYPES = {
    0: "UNKNOWN", 1: "COFF", 2: "CODEVIEW", 3: "FPO", 4: "MISC", 5: "EXCEPTION", 6: "FIXUP",
    7: "OMAP_TO_SRC", 8: "OMAP_FROM_SRC", 9: "BORLAND", 10: "RESERVED10", 11: "CLSID", 12: "VC_FEATURE",
    13: "POGO", 14: "ILTCG", 15: "MPX", 16: "REPRO", 17: "EMBEDDED_PORTABLE_PDB", 18: "SPGO",
    19: "PDBCHECKSUM", 20: "EX_DLLCHARACTERISTICS",
}
DEBUG_TYPE_CODEVIEW = 2
DEBUG_ENTRY_SIZE = 28   # IMAGE_DEBUG_DIRECTORY
PDB_NAME_LIMIT = 1024   # longer than any path the linker writes; bounds a missing terminator


def _codeview(data: bytes, at: int, size: int) -> dict:
    """A CodeView record: RSDS (PDB 7.0: GUID, age, name) or NB10 (PDB 2.0: signature, age, name). Only the bytes
    the entry declares are read; a record too short for its fixed fields is marked truncated and has no others."""
    end = min(len(data), at + size)
    sig = data[at:at + 4] if at + 4 <= end else b""
    if sig == b"RSDS" and at + 24 <= end:
        guid = uuid.UUID(bytes_le=bytes(data[at + 4:at + 20]))
        age = struct.unpack_from("<I", data, at + 20)[0]
        name = _cstring(data, at + 24, min(PDB_NAME_LIMIT, end - at - 24))
        return {"format": "RSDS", "guid": str(guid), "age": age, "pdb": name}
    if sig == b"NB10" and at + 16 <= end:
        signature, age = struct.unpack_from("<II", data, at + 8)
        return {"format": "NB10", "signature": f"{signature:08x}", "age": age,
                "pdb": _cstring(data, at + 16, min(PDB_NAME_LIMIT, end - at - 16))}
    if sig in (b"RSDS", b"NB10"):
        return {"format": sig.decode("ascii"), "truncated": True}
    return {"format": sig.decode("ascii", "replace") if sig else "truncated"}


def _debug_entries(img: PeImage) -> list[dict]:
    """Every IMAGE_DEBUG_DIRECTORY entry. The directory's size is a whole number of 28-byte entries; a CodeView
    entry's data is read from its file pointer (the linker always writes one), else from its RVA."""
    rva, size = img.directory(DIR_DEBUG)
    if not rva or size < DEBUG_ENTRY_SIZE:
        return []
    at = img.offset(rva)
    if at is None:
        return []
    entries = []
    for i in range(min(size // DEBUG_ENTRY_SIZE, MAX_ENTRIES)):
        base = at + i * DEBUG_ENTRY_SIZE
        if base + DEBUG_ENTRY_SIZE > len(img.data):
            break
        (_chars, stamp, major, minor, kind, data_size, data_rva,
         data_ptr) = struct.unpack_from("<IIHHIIII", img.data, base)
        entry = {"type": DEBUG_TYPES.get(kind, kind), "timestamp": stamp, "version": f"{major}.{minor}",
                 "size": data_size}
        if kind == DEBUG_TYPE_CODEVIEW:
            where = data_ptr if data_ptr and data_ptr < len(img.data) else (img.offset(data_rva) if data_rva else None)
            entry["codeview"] = _codeview(img.data, where, data_size) if where is not None else {"format": "missing"}
        entries.append(entry)
    return entries


def rsds(report: dict) -> dict | None:
    """The image's first CodeView RSDS record, the one a debugger matches a PDB against."""
    for entry in report.get("debug", []):
        cv = entry.get("codeview")
        if cv and cv.get("format") == "RSDS":
            return cv
    return None


class PdbFormatError(ValueError):
    """The file is not an MSF 7.0 PDB this reader can read."""


MSF7_MAGIC = b"Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0"
MSF_BLOCK_SIZES = (512, 1024, 2048, 4096)
PDB_INFO_STREAM = 1
PDB_DBI_STREAM = 3
NIL_STREAM = 0xFFFFFFFF


def read_pdb_identity(source) -> dict:
    """GUID and ages of an MSF 7.0 PDB, given as bytes or as a seekable binary file. The superblock gives the block
    size and the block that lists the stream directory's blocks; the directory gives each stream's size and blocks.
    The PDB info stream (1) holds version, signature, age and GUID; the DBI stream (3) holds the age DbgHelp matches
    against an image's RSDS age, which can differ from the info stream's after an incremental link. From a file only
    the superblock, the directory and the first block of those two streams are read; every block the two streams
    list must still lie inside the file."""
    f = io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source
    file_size = f.seek(0, os.SEEK_END)
    f.seek(0)
    head = f.read(0x38)
    if len(head) < 0x38 or head[:len(MSF7_MAGIC)] != MSF7_MAGIC:
        raise PdbFormatError("no MSF 7.0 superblock (an older PDB 2.0 file, or not a PDB)")
    block, _free_map, blocks, dir_bytes, _unknown, map_block = struct.unpack_from("<IIIIII", head, 0x20)
    if block not in MSF_BLOCK_SIZES:
        raise PdbFormatError(f"block size {block} is not one of {MSF_BLOCK_SIZES}")

    def check_block(index: int) -> None:
        if index >= blocks or index * block + block > file_size:
            raise PdbFormatError(f"block {index} is past the end of the file")

    def block_bytes(index: int) -> bytes:
        check_block(index)
        f.seek(index * block)
        data = f.read(block)
        if len(data) != block:
            raise PdbFormatError(f"block {index} is past the end of the file")
        return data

    def gather(indices, length: int) -> bytes:
        return b"".join(block_bytes(i) for i in indices)[:length]

    dir_blocks = -(-dir_bytes // block)
    if dir_blocks * 4 > block:
        raise PdbFormatError("the stream directory needs more blocks than one block map holds")
    directory = gather(struct.unpack_from(f"<{dir_blocks}I", block_bytes(map_block)), dir_bytes)
    if len(directory) < 4:
        raise PdbFormatError("stream directory is empty")
    count = struct.unpack_from("<I", directory, 0)[0]
    if 4 + 4 * count > len(directory):
        raise PdbFormatError("stream directory is cut short")
    sizes = struct.unpack_from(f"<{count}I", directory, 4)
    streams, pos = [], 4 + 4 * count
    for size in sizes:
        n = 0 if size == NIL_STREAM else -(-size // block)
        if pos + 4 * n > len(directory):
            raise PdbFormatError("stream directory is cut short")
        streams.append((0 if size == NIL_STREAM else size, struct.unpack_from(f"<{n}I", directory, pos)))
        pos += 4 * n

    def stream_head(i: int, length: int) -> bytes:
        """The first length bytes of stream i; its other blocks are only checked to be inside the file."""
        size, indices = streams[i]
        for index in indices:
            check_block(index)
        return gather(indices[:-(-min(length, size) // block)], min(length, size))

    if count <= PDB_INFO_STREAM:
        raise PdbFormatError("no PDB info stream")
    info = stream_head(PDB_INFO_STREAM, 28)
    if len(info) < 28:
        raise PdbFormatError("PDB info stream is shorter than its 28-byte header")
    version, signature, age = struct.unpack_from("<III", info, 0)
    out = {"version": version, "signature": f"{signature:08x}", "info_age": age,
           "guid": str(uuid.UUID(bytes_le=bytes(info[12:28]))), "dbi_age": None}
    if count > PDB_DBI_STREAM:
        dbi = stream_head(PDB_DBI_STREAM, 12)
        if len(dbi) >= 12:
            out["dbi_age"] = struct.unpack_from("<I", dbi, 8)[0]
    return out


def match_pdb(record: dict | None, pdb: dict) -> dict:
    """Whether a PDB is the one an RSDS record names: the same GUID and the same age (the DBI stream's age when the
    PDB has one, as DbgHelp compares it; the info stream's otherwise)."""
    age = pdb["dbi_age"] if pdb["dbi_age"] is not None else pdb["info_age"]
    if record is None:
        return {"match": False, "reason": "the image has no CodeView RSDS record", "pdb": pdb}
    if record.get("truncated"):
        return {"match": False, "reason": "the image's CodeView RSDS record is cut short (no GUID or age)", "pdb": pdb}
    guid_match = record["guid"] == pdb["guid"]
    age_match = record["age"] == age
    reason = "match" if guid_match and age_match else (
        "GUID differs: another link" if not guid_match else f"age differs: image {record['age']}, PDB {age}")
    return {"match": guid_match and age_match, "guid_match": guid_match, "age_match": age_match,
            "age_from": "dbi" if pdb["dbi_age"] is not None else "info", "reason": reason,
            "image": {"guid": record["guid"], "age": record["age"]}, "pdb": pdb}


_apiset_cache: dict[str, bool] = {}


def _apiset_resolves(name: str) -> bool:
    """True if this Windows' API-set schema maps the contract to a host DLL (ntdll query; nothing is loaded)."""
    if name not in _apiset_cache:
        _apiset_cache[name] = _query_apiset(name)
    return _apiset_cache[name]


def _query_apiset(name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", wintypes.LPWSTR)]

        query = ctypes.WinDLL("ntdll").ApiSetQueryApiSetPresence
        query.argtypes = [ctypes.POINTER(UNICODE_STRING), ctypes.POINTER(ctypes.c_ubyte)]
        query.restype = ctypes.c_long
        base = name[:-4] if name.lower().endswith(".dll") else name
        buf = ctypes.create_unicode_buffer(base)
        text = ctypes.cast(buf, wintypes.LPWSTR)
        us = UNICODE_STRING(len(base) * 2, (len(base) + 1) * 2, text)
        present = ctypes.c_ubyte(0)
        status = query(ctypes.byref(us), ctypes.byref(present))
        return status == 0 and bool(present.value)
    except (OSError, AttributeError, ValueError):
        return False


def classify_dll(name: str, game_dir: str | None) -> str:
    """bundled = ships beside the game, system = present on this Windows, shim = must be reimplemented."""
    if game_dir and os.path.isfile(os.path.join(game_dir, name)):
        return "bundled"
    low = name.lower()
    if low.startswith("api-ms-win-") or low.startswith("ext-ms-win-"):
        # API sets resolve through the OS schema: ask this Windows whether it knows the contract.
        return "system" if _apiset_resolves(name) else "shim"
    if os.path.isfile(os.path.join(SYSTEM32, name)):
        return "system"
    return "shim"


def inspect(path: str, game_dir: str | None) -> dict:
    """The report for one image. A file cut off inside a table raises PeFormatError like any other bad image;
    the struct.error it used to raise escaped main() and stopped a multi-file run at that file."""
    try:
        return _inspect(path, game_dir)
    except struct.error as e:
        raise PeFormatError(f"truncated or corrupt table: {e}") from None


def _inspect(path: str, game_dir: str | None) -> dict:
    with open(path, "rb") as f:
        img = PeImage(f.read())
    sections = []
    for s in img.sections:
        data = img.section_data(s)
        e = entropy(data)
        sections.append({
            "name": s["name"],
            "rva": s["rva"], "vsize": s["vsize"], "rawsize": s["rawsize"],
            "exec": s["exec"],
            "write": s["write"],
            "entropy": round(e, 3),
            "looks_encrypted": e > 7.9 and len(data) > 4096,
        })

    imports = _imports(img, "normal", DIR_IMPORT) + _imports(img, "delay", DIR_DELAY_IMPORT, delay=True)
    for entry in imports:
        entry["where"] = classify_dll(entry["dll"], game_dir)

    exports = _exports(img)
    pdata_rva, pdata_size = img.directory(DIR_EXCEPTION)
    reloc_rva, reloc_size = img.directory(DIR_BASERELOC)

    return {
        "file": os.path.basename(path),
        "size": os.path.getsize(path),
        "machine": MACHINES.get(img.machine, hex(img.machine)),
        "timestamp": img.timestamp,
        "is_dll": bool(img.characteristics & 0x2000),
        "subsystem": SUBSYSTEMS.get(img.subsystem, img.subsystem),
        "image_base": hex(img.image_base),
        "size_of_image": img.size_of_image,
        "entry_rva": hex(img.entry_rva),
        "has_relocs": bool(reloc_rva and reloc_size),
        "dynamic_base": bool(img.dll_characteristics & 0x40),
        "sections": sections,
        "imports": imports,
        "export_count": len(exports),
        "exports": exports,
        "pdata_functions": pdata_size // 12 if pdata_rva and pdata_size >= 12 else 0,
        "tls_callbacks": _tls_callbacks(img),
        "debug": _debug_entries(img),
    }


def to_markdown(r: dict) -> str:
    out = [f"# {r['file']}", "",
           f"- {r['machine']}, {'DLL' if r['is_dll'] else 'EXE'}, subsystem {r['subsystem']}, {r['size']:,} bytes",
           f"- image base {r['image_base']}, size {r['size_of_image']:,}, entry RVA {r['entry_rva']}",
           f"- relocations: {'yes' if r['has_relocs'] else 'NO (must load at preferred base)'}; "
           f"ASLR flag: {r['dynamic_base']}",
           f"- .pdata functions: {r['pdata_functions']:,}; TLS callbacks: {r['tls_callbacks']}; exports: {r['export_count']}",
           "", "## Sections", "", "| name | rva | vsize | exec | write | entropy | |", "|---|---|---|---|---|---|---|"]
    for s in r["sections"]:
        flag = "**encrypted?**" if s["looks_encrypted"] else ""
        out.append(f"| {s['name']} | {s['rva']:#x} | {s['vsize']:,} | {s['exec']} | {s['write']} | {s['entropy']} | {flag} |")
    out += ["", "## Debug directory", ""]
    if not r.get("debug"):
        out.append("none")
    else:
        out += ["| type | version | size | details |", "|---|---|---|---|"]
        for d in r["debug"]:
            cv = d.get("codeview")
            if cv and cv.get("truncated"):
                detail = f"{cv['format']} record cut short (not read)"
            elif cv and cv.get("format") == "RSDS":
                detail = f"RSDS {cv['guid']} age {cv['age']} `{cv['pdb']}`"
            elif cv and cv.get("format") == "NB10":
                detail = f"NB10 {cv['signature']} age {cv['age']} `{cv['pdb']}`"
            elif cv:
                detail = f"CodeView `{cv['format']}` (not read)"
            else:
                detail = ""
            out.append(f"| {d['type']} | {d['version']} | {d['size']:,} | {detail} |")
    out += ["", "## Imports", ""]
    by_where = Counter()
    for imp in r["imports"]:
        by_where[imp["where"]] += len(imp["functions"])
    out.append(", ".join(f"{k}: {v}" for k, v in sorted(by_where.items())) or "none")
    out += ["", "| dll | kind | resolves | functions |", "|---|---|---|---|"]
    for imp in sorted(r["imports"], key=lambda i: (i["where"] != "shim", i["dll"].lower())):
        out.append(f"| {imp['dll']} | {imp['kind']} | {imp['where']} | {len(imp['functions'])} |")
    shims = [i for i in r["imports"] if i["where"] == "shim"]
    if shims:
        out += ["", "## Needs checking or a shim", ""]
        for imp in shims:
            out.append(f"### {imp['dll']} ({imp['where']})")
            out.append("")
            out.append(", ".join(f"`{f}`" for f in imp["functions"]))
            out.append("")
    return "\n".join(out) + "\n"


def same_output_path(first: str, second: str) -> bool:
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def check_pdb(images: list[str], pdb_path: str) -> int:
    """--pdb: compare one image's RSDS record with a PDB. Writes no report files."""
    if len(images) != 1:
        print("--pdb takes exactly one image", file=sys.stderr)
        return 2
    try:
        report = inspect(images[0], None)
    except (PeFormatError, OSError) as e:
        print(f"{images[0]}: not a PE image ({e})", file=sys.stderr)
        return 1
    try:
        with open(pdb_path, "rb") as f:
            pdb = read_pdb_identity(f)
    except struct.error as e:
        print(f"{pdb_path}: not a readable PDB (truncated or corrupt table: {e})", file=sys.stderr)
        return 1
    except (PdbFormatError, OSError) as e:
        print(f"{pdb_path}: not a readable PDB ({e})", file=sys.stderr)
        return 1
    result = match_pdb(rsds(report), pdb)
    print(json.dumps({"image_file": report["file"], "pdb_file": os.path.basename(pdb_path), **result}, indent=1))
    return 0 if result["match"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="+")
    ap.add_argument("--game-dir")
    ap.add_argument("--out", help="report folder (default: reports/pe)")
    ap.add_argument("--pdb", help="check that this PDB is the one the image's RSDS record names (one image only); "
                                  "prints the result as JSON and writes no report. Exit code 0 on a match, 1 for "
                                  "no match or an unreadable file, 2 for a usage error")
    a = ap.parse_args(argv)
    if a.pdb:
        if a.out is not None or a.game_dir is not None:
            print("--out and --game-dir are not used with --pdb, which writes no report", file=sys.stderr)
            return 2
        return check_pdb(a.image, a.pdb)
    if a.out is None:
        a.out = os.path.join(os.path.dirname(__file__), "..", "reports", "pe")
    if os.path.isfile(a.out):
        print(f"cannot create output directory {a.out}: path is a file, not a directory", file=sys.stderr)
        return 1
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        print(f"cannot create output directory {a.out}: {e.strerror or e}", file=sys.stderr)
        return 1
    rc = 0
    seen_paths: set[tuple[str, str]] = set()
    used_names: set[str] = set()
    for img in a.image:
        search_dir = a.game_dir or os.path.dirname(os.path.abspath(img))
        key = (os.path.normcase(os.path.realpath(img)), os.path.normcase(os.path.abspath(search_dir)))
        if key in seen_paths:
            print(f"{img}: given more than once; inspected once", file=sys.stderr)
            continue
        seen_paths.add(key)
        try:
            r = inspect(img, search_dir)
        except (PeFormatError, OSError) as e:
            print(f"{img}: not a PE image ({e})", file=sys.stderr)
            rc = 1
            continue
        # One report per image: the full file name keeps game.exe and game.dll apart, and a numbered suffix keeps
        # a/x.exe and b/x.exe apart. Compared case-insensitively, as Windows file names are.
        name = r["file"]
        n = 1
        while name.lower() in used_names:
            n += 1
            name = f"{r['file']}-{n}"
        used_names.add(name.lower())
        if n > 1:
            print(f"{img}: another input is also named {r['file']}; report written as {name}.json/.md",
                  file=sys.stderr)
        stem = os.path.join(a.out, name)
        json_path = stem + ".json"
        md_path = stem + ".md"
        collision = None
        for other in a.image:
            if same_output_path(other, json_path):
                collision = json_path
                break
            if same_output_path(other, md_path):
                collision = md_path
                break
        if collision:
            print(f"{img}: report {collision} would overwrite input file", file=sys.stderr)
            rc = 1
            continue
        try:
            with open(json_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(r, f, indent=1)
            with open(md_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(to_markdown(r))
        except OSError as e:
            print(f"{img}: cannot write report ({e.strerror or e})", file=sys.stderr)
            rc = 1
            continue
        enc = [s["name"] for s in r["sections"] if s["looks_encrypted"]]
        shim = sum(len(i["functions"]) for i in r["imports"] if i["where"] == "shim")
        print(f"{r['file']:<40} {r['machine']:<7} subsys={r['subsystem']!s:<7} pdata={r['pdata_functions']:<7} "
              f"shim-imports={shim:<5} encrypted={','.join(enc) or '-'}")
    return rc


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
