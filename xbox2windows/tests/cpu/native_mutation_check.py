"""Map mutated copies of the source-built fixtures in child processes and require named results.

Each case runs in its own process, so an access violation or a hang is reported by the parent as a
failed case instead of ending the suite. A mutation is accepted when ``pe_map_image`` either returns
a mapped base or a named ``MapStatus`` with a detail string. A crash, a timeout, an unknown status
or a status that disagrees with the returned base fails the case.
"""

import argparse
import ctypes
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_map_check import Mapper  # noqa: E402

STATUS_NAMES = {
    0: "ok", 1: "io", 2: "format", 3: "memory", 4: "relocation",
    5: "unresolved", 6: "unsupported", 7: "protect", 8: "function_table", 9: "init_failed",
}


def child(mapper_path, image, provider):
    """Map one copy and print the status as JSON. Runs in its own process."""
    mapper = Mapper(mapper_path)
    bindings = {}
    if provider:
        bindings[Path(provider).name.casefold()] = ctypes.CDLL(provider)
    base, status, detail, _ = mapper(image, bindings)
    print(json.dumps({"mapped": bool(base), "status": status, "detail": detail}))
    if base:
        mapper.release(base)
    return 0


def header_offsets(pe):
    return {
        "e_lfanew": pe.DOS_HEADER.get_field_absolute_offset("e_lfanew"),
        "NumberOfSections": pe.FILE_HEADER.get_field_absolute_offset("NumberOfSections"),
        "SizeOfOptionalHeader": pe.FILE_HEADER.get_field_absolute_offset("SizeOfOptionalHeader"),
        "SizeOfImage": pe.OPTIONAL_HEADER.get_field_absolute_offset("SizeOfImage"),
        "SizeOfHeaders": pe.OPTIONAL_HEADER.get_field_absolute_offset("SizeOfHeaders"),
        "AddressOfEntryPoint": pe.OPTIONAL_HEADER.get_field_absolute_offset("AddressOfEntryPoint"),
    }


def first_section_offsets(pe):
    table = pe.OPTIONAL_HEADER.get_file_offset() + pe.FILE_HEADER.SizeOfOptionalHeader
    return {
        "VirtualAddress": table + 12,
        "SizeOfRawData": table + 16,
        "PointerToRawData": table + 20,
        "Characteristics": table + 36,
    }


def directory_offsets(pe, index):
    directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[index]
    return (directory.get_field_absolute_offset("VirtualAddress"),
            directory.get_field_absolute_offset("Size"))


def edits(image):
    """A fixed list of single-field edits. Offsets are stable, so every run is reproducible."""
    with pefile.PE(str(image)) as pe:
        header = header_offsets(pe)
        section = first_section_offsets(pe)
        export = directory_offsets(pe, 0)
        imports = directory_offsets(pe, 1)
        relocs = directory_offsets(pe, 5)
        size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
    past = size_of_image + 0x1000
    return [
        ("control", None, None),
        ("e_lfanew 0", header["e_lfanew"], ("<I", 0)),
        ("e_lfanew past file", header["e_lfanew"], ("<I", image.stat().st_size + 0x100)),
        ("no sections", header["NumberOfSections"], ("<H", 0)),
        ("many sections", header["NumberOfSections"], ("<H", 0xffff)),
        ("no optional header", header["SizeOfOptionalHeader"], ("<H", 0)),
        ("image size 0", header["SizeOfImage"], ("<I", 0)),
        ("headers size 0", header["SizeOfHeaders"], ("<I", 0)),
        ("entry past image", header["AddressOfEntryPoint"], ("<I", 0xffffffff)),
        ("section VA 0", section["VirtualAddress"], ("<I", 0)),
        ("section raw size", section["SizeOfRawData"], ("<I", 0xffffffff)),
        ("section raw offset", section["PointerToRawData"], ("<I", 0xffffffff)),
        ("section no access", section["Characteristics"], ("<I", 0)),
        ("section execute only", section["Characteristics"], ("<I", 0x20000000)),
        ("section read write execute", section["Characteristics"], ("<I", 0xe0000000)),
        ("export VA past image", export[0], ("<I", past)),
        ("export size", export[1], ("<I", 0xffffffff)),
        ("import VA past image", imports[0], ("<I", past)),
        ("reloc VA past image", relocs[0], ("<I", past)),
        ("reloc size", relocs[1], ("<I", 0xffffffff)),
    ]


def mutated(data, edit):
    if edit is None:
        return bytes(data)
    offset, (format, value) = edit
    copy = bytearray(data)
    struct.pack_into(format, copy, offset, value)
    return bytes(copy)


def run_case(mapper, fixture, provider, name, data, temp, control=False):
    copy = Path(temp) / f"{fixture.stem}-{name.replace(' ', '-')}.dll"
    copy.write_bytes(data)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(mapper), str(copy),
               str(provider) if provider else ""]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"{fixture.name}: {name}: timed out") from None
    if result.returncode != 0:
        raise AssertionError(f"{fixture.name}: {name}: child exited {result.returncode:#x}\n"
                             f"{result.stdout}{result.stderr}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError(f"{fixture.name}: {name}: no result\n{result.stderr}")
    outcome = json.loads(lines[-1])
    status, detail, mapped = outcome["status"], outcome["detail"], outcome["mapped"]
    if status not in STATUS_NAMES:
        raise AssertionError(f"{fixture.name}: {name}: unknown status {status}")
    if mapped != (status == 0):
        raise AssertionError(f"{fixture.name}: {name}: status {status} with mapped={mapped}")
    if status and not detail:
        raise AssertionError(f"{fixture.name}: {name}: status {status} without a detail")
    if control and status != 0:
        raise AssertionError(f"{fixture.name}: control did not map: {status}: {detail}")
    return STATUS_NAMES[status]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=3, metavar=("MAPPER", "IMAGE", "PROVIDER"))
    parser.add_argument("mapper", nargs="?", type=Path)
    parser.add_argument("fixtures", nargs="*", type=Path)
    parser.add_argument("--provider", type=Path, help="DLL that serves the fixtures' imports")
    args = parser.parse_args()
    if args.child:
        mapper, image, provider = args.child
        return child(mapper, image, provider or None)
    if not args.mapper or not args.fixtures:
        parser.error("expected a mapper and at least one fixture")
    outcomes = {}
    with TemporaryDirectory() as temp:
        for fixture in args.fixtures:
            data = fixture.read_bytes()
            for name, offset, pack in edits(fixture):
                edit = None if offset is None else (offset, pack)
                outcome = run_case(args.mapper, fixture, args.provider, name, mutated(data, edit),
                                   temp, control=offset is None)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
            for length in (0x40, 0x100, 0x200):
                outcome = run_case(args.mapper, fixture, args.provider, f"truncated {length:#x}",
                                   data[:length], temp)
                outcomes[outcome] = outcomes.get(outcome, 0) + 1
    print(f"{len(args.fixtures)} fixtures survived; outcomes: " +
          ", ".join(f"{name}={count}" for name, count in sorted(outcomes.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
