"""Map a source-built image with delay imports and compare eager binding with the Windows delay helper."""

import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, STATUS

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
kernel.GetProcAddress.restype = ctypes.c_void_p

def delay_import(image, symbol=None, ordinal=None):
    """Return the descriptor DLL name and the IAT slot of a delay import by name or ordinal."""
    with pefile.PE(str(image)) as pe:
        desc, = pe.DIRECTORY_ENTRY_DELAY_IMPORT
        for index, imported in enumerate(desc.imports):
            named = imported.name is not None and imported.name.decode() == symbol
            numbered = imported.name is None and imported.ordinal == ordinal
            if named or numbered:
                return desc.dll.decode(), desc.struct.pIAT + 8 * index
        raise AssertionError(f"{symbol or f'#{ordinal}'} is not a delay import of {image.name}")


def delay_thunk_offset(image, symbol):
    """File offset of the INT thunk for a named delay import."""
    with pefile.PE(str(image)) as pe:
        desc, = pe.DIRECTORY_ENTRY_DELAY_IMPORT
        for index, imported in enumerate(desc.imports):
            if imported.name and imported.name.decode() == symbol:
                return pe.get_offset_from_rva(desc.struct.pINT + 8 * index)
    raise AssertionError(f"{symbol} is not a named delay import of {image.name}")


def exports(image):
    with pefile.PE(str(image)) as pe:
        return {entry.name.decode(): entry.address for entry in pe.DIRECTORY_ENTRY_EXPORT.symbols if entry.name}


def system_bindings(image):
    """Load the ordinary imports through Windows, as the host would before mapping."""
    with pefile.PE(str(image)) as pe:
        names = {desc.dll.decode() for desc in pe.DIRECTORY_ENTRY_IMPORT}
    return {name.casefold(): ctypes.WinDLL(name) for name in names}


def check_call(mapper, image, provider, altered):
    dll, slot = delay_import(image, symbol="delayed_value")
    _, ordinal_slot = delay_import(image, ordinal=2)
    exports_by_name = exports(image)
    export, ordinal_export = exports_by_name["call_delayed"], exports_by_name["call_delayed_ordinal"]
    # Load the provider by path first: the native helper asks for it by name, and the Windows
    # default search order does not include the route's directory.
    module = ctypes.CDLL(str(provider))
    altered_module = ctypes.CDLL(str(altered))
    native = ctypes.CDLL(str(image))
    native.call_delayed.restype = ctypes.c_int
    native.call_delayed_ordinal.restype = ctypes.c_int
    expected = native.call_delayed()
    expected_ordinal = native.call_delayed_ordinal()
    if (expected, expected_ordinal) != (31, 59):
        raise AssertionError(f"native delay calls returned {expected}, {expected_ordinal}")

    system = system_bindings(image)
    base, status, detail, requests = mapper(image, {**system, dll.casefold(): module})
    if not base or status != STATUS["ok"]:
        raise AssertionError(f"mapped delay image: {base=} {status=} {detail=}")
    try:
        bound = kernel.GetProcAddress(module._handle, b"delayed_value")
        bound_ordinal = kernel.GetProcAddress(module._handle, 2)
        if ctypes.c_uint64.from_address(base + slot).value != bound:
            raise AssertionError("delay IAT slot does not hold the named provider export")
        if ctypes.c_uint64.from_address(base + ordinal_slot).value != bound_ordinal:
            raise AssertionError("delay IAT slot does not hold the ordinal provider export")
        pairs = [(name.casefold(), item) for name, item in requests]
        if (dll.casefold(), "delayed_value") not in pairs or (dll.casefold(), 2) not in pairs:
            raise AssertionError(f"a delay import was not resolved at map time: {requests}")
        actual = ctypes.CFUNCTYPE(ctypes.c_int)(base + export)()
        actual_ordinal = ctypes.CFUNCTYPE(ctypes.c_int)(base + ordinal_export)()
        if (actual, actual_ordinal) != (expected, expected_ordinal):
            raise AssertionError(f"mapped delay calls returned {actual}, {actual_ordinal}; "
                                 f"native returned {expected}, {expected_ordinal}")
    finally:
        mapper.release(base)

    base, status, detail, _ = mapper(image, {**system, dll.casefold(): altered_module})
    if not base or status != STATUS["ok"]:
        raise AssertionError(f"mapped altered delay image: {base=} {status=} {detail=}")
    try:
        actual = ctypes.CFUNCTYPE(ctypes.c_int)(base + export)()
        actual_ordinal = ctypes.CFUNCTYPE(ctypes.c_int)(base + ordinal_export)()
        if (actual, actual_ordinal) != (47, 61):
            raise AssertionError("the altered delay provider was not used")
    finally:
        mapper.release(base)

    # The descriptor order is linker-defined, so the first unresolved identity comes from the file.
    with pefile.PE(str(image)) as pe:
        first = pe.DIRECTORY_ENTRY_DELAY_IMPORT[0].imports[0]
        identity = first.name.decode() if first.name else f"#{first.ordinal}"
    base, status, detail, _ = mapper(image, system)
    if base or status != STATUS["unresolved"] or detail != f"unresolved delay import {dll}!{identity}":
        raise AssertionError(f"unbound delay import: {base=} {status=} {detail=}")

    # The directory is found by its address, like Windows; Size is not used. A zero Size must still bind.
    with pefile.PE(str(image)) as pe:
        size_offset = pe.OPTIONAL_HEADER.DATA_DIRECTORY[13].get_field_absolute_offset("Size")
    copy = bytearray(Path(image).read_bytes())
    struct.pack_into("<I", copy, size_offset, 0)
    with TemporaryDirectory() as temp:
        zero_size = Path(temp) / image.name
        zero_size.write_bytes(copy)
        base, status, detail, _ = mapper(zero_size, {**system, dll.casefold(): module})
    if not base or status != STATUS["ok"]:
        raise AssertionError(f"zero-size delay directory: {base=} {status=} {detail=}")
    try:
        if ctypes.CFUNCTYPE(ctypes.c_int)(base + export)() != expected:
            raise AssertionError("zero-size delay directory did not bind the named import")
    finally:
        mapper.release(base)

    # A named thunk with a bit above the 31-bit RVA must be rejected, not masked into a valid RVA.
    thunk = delay_thunk_offset(image, "delayed_value")
    copy = bytearray(Path(image).read_bytes())
    original = struct.unpack_from("<Q", copy, thunk)[0]
    struct.pack_into("<Q", copy, thunk, original | (1 << 40))
    with TemporaryDirectory() as temp:
        high_bits = Path(temp) / image.name
        high_bits.write_bytes(copy)
        base, status, detail, _ = mapper(high_bits, {**system, dll.casefold(): module})
    if base or status != STATUS["format"] or detail != f"delay import name for {dll} outside image":
        raise AssertionError(f"high-bit delay thunk: {base=} {status=} {detail=}")
    print(f"{image.name}: {dll}!delayed_value and {dll}!#2 bound at map time, the mapped calls match Windows, "
          "the altered provider changes both results, a zero directory Size still binds, a named thunk with "
          "bits above the 31-bit RVA is rejected, and without a binding the map is rejected")


def check_rejections(mapper_path, image):
    with pefile.PE(str(image)) as pe:
        directory = pe.OPTIONAL_HEADER.DATA_DIRECTORY[13]
        address = directory.get_field_absolute_offset("VirtualAddress")
        size = directory.get_field_absolute_offset("Size")
        descriptor = pe.get_offset_from_rva(directory.VirtualAddress)
        image_size = pe.OPTIONAL_HEADER.SizeOfImage
    system = system_bindings(image)
    mapper = Mapper(mapper_path)
    controls = (
        (descriptor + 0, 0, "unsupported", "delay imports without RVA attributes"),
        (descriptor + 4, image_size + 0x1000, "format", "delay import DLL name outside image"),
        (descriptor + 16, 0, "format", "delay import tables for cpu_delay_provider.dll outside image"),
        (descriptor + 12, image_size - 4, "format", "delay import IAT for cpu_delay_provider.dll outside image"),
        (address, image_size + 0x1000, "format", "delay import directory outside image"),
    )
    for offset, value, expected, message in controls:
        copy = bytearray(Path(image).read_bytes())
        struct.pack_into("<I", copy, offset, value)
        with TemporaryDirectory() as temp:
            damaged = Path(temp) / image.name
            damaged.write_bytes(copy)
            base, status, detail, requests = mapper(damaged, system)
        if base or status != STATUS[expected] or detail != message:
            raise AssertionError(f"{message} control: {base=} {status=} {detail=}")
        if any(name.casefold() == "cpu_delay_provider.dll" for name, _ in requests):
            raise AssertionError(f"{message} control requested the provider")

    # A directory address of zero is ignored whatever Size says, as the Windows loader does.
    copy = bytearray(Path(image).read_bytes())
    struct.pack_into("<I", copy, address, 0)
    struct.pack_into("<I", copy, size, 32)
    with TemporaryDirectory() as temp:
        ignored = Path(temp) / image.name
        ignored.write_bytes(copy)
        base, status, detail, requests = mapper(ignored, system)
    if not base or status != STATUS["ok"]:
        raise AssertionError(f"address-zero delay directory: {base=} {status=} {detail=}")
    mapper.release(base)
    if any(name.casefold() == "cpu_delay_provider.dll" for name, _ in requests):
        raise AssertionError("address-zero delay directory requested the provider")
    print("rejected before any provider request: non-RVA attributes, a name outside the image, a zero name "
          "table, an IAT outside the image and a directory address outside the image; a zero Size still binds "
          "and address zero ignores a nonzero Size")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapper", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("provider", type=Path)
    parser.add_argument("altered", type=Path)
    args = parser.parse_args()
    mapper_path, image = args.mapper.resolve(), args.image.resolve()
    provider, altered = args.provider.resolve(), args.altered.resolve()
    # The native delay helper loads the provider by name, so the route's directory must be current.
    os.chdir(image.parent)
    check_call(Mapper(mapper_path), image, provider, altered)
    check_rejections(mapper_path, image)
