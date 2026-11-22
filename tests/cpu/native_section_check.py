"""Compare every mapped section of the register probe with the Windows loader, on copies with patched section file fields.

Windows reads raw section data in whole 0x200-byte file sectors, and leaves a section with PointerToRawData 0 zero-filled.
"""

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, STATUS

FIELDS = {"VirtualSize": 8, "SizeOfRawData": 16, "PointerToRawData": 20}  # offsets in IMAGE_SECTION_HEADER
SECTOR = 0x200
kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
kernel.LoadLibraryExW.restype = ctypes.c_void_p
kernel.FreeLibrary.argtypes = [ctypes.c_void_p]

# (case, section, header edits from the original section, bytes cut from the end of the file, must differ
# from an exact read). .rsrc edits keep its resource directory intact, which LoadLibraryExW validates.
CASES = [
    ("no file offset", ".rsrc", lambda s: {"PointerToRawData": 0}, 0, True),
    ("start inside a sector", ".rdata", lambda s: {"PointerToRawData": s.PointerToRawData + 0x10}, 0, True),
    ("start in the first sector", ".rdata", lambda s: {"PointerToRawData": SECTOR // 2}, 0, True),
    ("raw size below a sector", ".rdata", lambda s: {"SizeOfRawData": 0x10}, 0, True),
    ("raw data crossing a sector", ".rdata",
     lambda s: {"PointerToRawData": s.PointerToRawData + SECTOR - 0x10, "SizeOfRawData": 0x20}, 0, True),
    ("rounded end past the file", ".rsrc",
     lambda s: {"VirtualSize": SECTOR, "SizeOfRawData": SECTOR - 0x10}, 0x10, False),
]


def layout(image):
    """Return (name, rva, span) for each section; refuse images whose load would run code."""
    with pefile.PE(str(image), fast_load=True) as pe:
        if pe.OPTIONAL_HEADER.AddressOfEntryPoint or pe.OPTIONAL_HEADER.DATA_DIRECTORY[1].Size:
            raise AssertionError(f"{image.name}: needs no entry point and no imports to be loaded safely")
        return [(s.Name.rstrip(b"\0").decode(), s.VirtualAddress, s.Misc_VirtualSize or s.SizeOfRawData)
                for s in pe.sections]


def patch(source, target, name, edits, cut):
    """Copy source with one section header edited and the file shortened; return the exact-read bytes."""
    data = bytearray(source.read_bytes())
    with pefile.PE(data=bytes(data), fast_load=True) as pe:
        section = next(s for s in pe.sections if s.Name.rstrip(b"\0") == name.encode())
        fields = edits(section)
        if cut and (section.PointerToRawData + section.SizeOfRawData != len(data) or section.SizeOfRawData != SECTOR):
            raise AssertionError(f"{name}: needs to be the last {SECTOR:#x} bytes of the file to shorten it")
        for field, value in fields.items():
            struct.pack_into("<I", data, section.get_file_offset() + FIELDS[field], value)
    if cut:
        del data[-cut:]
    target.write_bytes(data)
    span = fields.get("VirtualSize", section.Misc_VirtualSize)
    offset = fields.get("PointerToRawData", section.PointerToRawData)
    size = fields.get("SizeOfRawData", section.SizeOfRawData)
    exact = bytes(data[offset:offset + min(size, span)])
    return (exact + bytes(span))[:span]


def read(base, sections):
    return {name: bytes((ctypes.c_ubyte * span).from_address(base + rva)) for name, rva, span in sections}


def windows_sections(image, sections):
    module = kernel.LoadLibraryExW(str(image), None, 0)
    if not module:
        raise AssertionError(f"{image.name}: LoadLibraryExW failed with {ctypes.get_last_error()}")
    try:
        return read(module, sections)
    finally:
        kernel.FreeLibrary(module)


def mapped_sections(mapper, image, sections):
    base, status, detail, requests = mapper(image, {})
    if status != STATUS["ok"] or not base or requests:
        raise AssertionError(f"{image.name}: map returned {status} ({detail}), requests {requests}")
    try:
        return read(base, sections)
    finally:
        mapper.release(base)


def compare(mapper, case, image, exact=None, target=None):
    sections = layout(image)
    native = windows_sections(image, sections)
    mapped = mapped_sections(mapper, image, sections)
    for name, _, _ in sections:
        if mapped[name] != native[name]:
            first = next(i for i, (a, b) in enumerate(zip(mapped[name], native[name])) if a != b)
            hint = ", as an exact read of the header fields gives" if name == target and mapped[name] == exact else ""
            raise AssertionError(f"{case}: pe_map {name} differs from Windows at byte {first:#x}{hint}")
    print(f"{case}: {len(sections)} sections match the Windows loader byte for byte")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("probe", type=Path, help="source-built image with no entry point and no imports")
    args = parser.parse_args()
    mapper = Mapper(args.mapper)

    with TemporaryDirectory() as folder:
        original = Path(folder) / "section_original.dll"
        original.write_bytes(args.probe.read_bytes())
        before = compare(mapper, "unpatched", original)

        failures = []
        for index, (case, name, edits, cut, discriminating) in enumerate(CASES):
            image = Path(folder) / f"section_case{index}.dll"
            exact = patch(args.probe, image, name, edits, cut)
            try:
                after = compare(mapper, f"{name} {case}", image, exact, name)
            except AssertionError as error:
                print(f"FAIL {error}")
                failures.append(case)
                continue
            if discriminating and after[name] == exact:
                raise AssertionError(f"{name} {case}: Windows matches an exact read, so the case checks nothing")
            changed = [other for other in before if other != name and after[other] != before[other]]
            if changed:
                raise AssertionError(f"{name} {case}: patching {name} changed other sections: {changed}")
            if case == "no file offset" and any(after[name]):
                raise AssertionError(f"Windows no longer zero-fills {name} without a file offset")
        if failures:
            raise AssertionError(f"{len(failures)} of {len(CASES)} cases differ from the Windows loader: {failures}")


if __name__ == "__main__":
    main()
