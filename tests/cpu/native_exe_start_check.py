"""Start a source-built EXE with pe_map and compare its exit code with a normal Windows launch."""

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, STATUS, kernel

IMAGE_FILE_DLL = 0x2000
ACCESS_VIOLATION = 0xC0000005
IMPORTED_DLL = "cpu_import_provider.dll"  # what exe_start.asm imports; the altered provider is bound under this name
# exe_start.asm: base-4 log digits, then mix_args(5, 3, ...) = 13 (14 from the altered provider).
TLS_ATTACH, ENTRY = 1, 2  # a reserved argument other than null would log 3 instead of 1


def exit_code(log, result):
    code = 0
    for digit in log:
        code = code * 4 + digit
    return (code << 8) | result


def native(image):
    """Launch the EXE as a process; its entry point's return value is the exit code."""
    done = subprocess.run([str(image)], capture_output=True, timeout=60)
    if done.stdout or done.stderr:  # the fixture writes nothing; anything captured is a diagnostic
        raise AssertionError(f"{Path(image).name} wrote output: {done.stdout!r} {done.stderr!r}")
    return done.returncode & 0xffffffff


def mapped(mapper_path, image, bindings):
    """Map and start the EXE in a child process, so a crash in the guest cannot hide the result."""
    result = subprocess.run([sys.executable, "-B", __file__, "--run", str(mapper_path), str(image), *bindings],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise AssertionError(f"mapped {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    if result.stderr:
        raise AssertionError(f"mapped {Path(image).name}: child wrote to stderr: {result.stderr}")
    return json.loads(result.stdout)


def run(mapper_path, image, provider):
    mapper = Mapper(mapper_path)
    run_entry = ctypes.CDLL(str(mapper_path)).pe_map_run
    run_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    run_entry.restype = ctypes.c_int
    base, status, detail, requests = mapper(Path(image), {IMPORTED_DLL: ctypes.CDLL(provider)})
    if not base:
        print(json.dumps({"status": status, "detail": detail}))
        return
    code = ctypes.c_uint32(0xdeadbeef)
    started = run_entry(base, ctypes.byref(code))
    mapper.release(base)
    print(json.dumps({"status": status, "started": started, "exit": code.value, "requests": requests}))


def check(mapper, original, no_tls, provider, altered):
    # The EXE imports cpu_import_provider.dll by name; Windows finds it next to the EXE.
    reference, reference_no_tls = native(original), native(no_tls)
    print(f"native launch: original {reference:#x}, without TLS {reference_no_tls:#x}")
    if reference != exit_code([TLS_ATTACH, ENTRY], 13) or reference_no_tls != exit_code([ENTRY], 13):
        raise AssertionError("native launch does not run the TLS callback (null reserved) before the entry point")
    results = {}
    for name, image, bound in (("original", original, provider), ("without TLS", no_tls, provider),
                               ("altered provider", original, altered)):
        results[name] = mapped(mapper, image, [bound])
        if results[name].get("status") != STATUS["ok"] or results[name].get("started") != 1:
            raise AssertionError(f"mapped {name}: {results[name]}")
    if results["original"]["exit"] != reference:
        raise AssertionError(f"mapped original {results['original']['exit']:#x} != native {reference:#x}")
    if results["without TLS"]["exit"] != reference_no_tls:
        raise AssertionError(f"mapped without TLS {results['without TLS']['exit']:#x} != native {reference_no_tls:#x}")
    if results["altered provider"]["exit"] != reference + 1:
        raise AssertionError(f"altered provider: {results['altered provider']['exit']:#x}")
    if results["original"]["requests"] != [[IMPORTED_DLL, "mix_args"]]:
        raise AssertionError(f"import requests: {results['original']['requests']}")
    print(f"mapped: original {results['original']['exit']:#x} = native, without TLS "
          f"{results['without TLS']['exit']:#x} = native, altered provider {results['altered provider']['exit']:#x}")


def check_dll_refused(mapper, original, provider):
    """pe_map_run starts a process image only; with IMAGE_FILE_DLL set it must not call the entry point."""
    with pefile.PE(str(original)) as pe:
        field = pe.FILE_HEADER.get_field_absolute_offset("Characteristics")
        flags = pe.FILE_HEADER.Characteristics
    with TemporaryDirectory() as temp:
        copy = bytearray(Path(original).read_bytes())
        struct.pack_into("<H", copy, field, flags | IMAGE_FILE_DLL)
        image = Path(temp) / Path(original).name
        image.write_bytes(copy)
        result = mapped(mapper, image, [provider])
    if result.get("started") != 0 or result["exit"] != 0xdeadbeef:
        raise AssertionError(f"DLL-flagged image: {result}")
    print("pe_map_run refuses an image with IMAGE_FILE_DLL set and leaves the exit code untouched")


def check_refusals(mapper_path, original, provider):
    """Each refused startup leaves the fixture's writable log and output sentinel intact."""
    mapper = Mapper(mapper_path)
    start = ctypes.CDLL(str(mapper_path)).pe_map_run
    start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    start.restype = ctypes.c_int
    kernel.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_uint32)]
    with TemporaryDirectory() as temp, pefile.PE(str(original)) as pe:
        data = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".data")
        table = pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks - pe.OPTIONAL_HEADER.ImageBase
        no_entry = Path(temp) / "no-entry.exe"
        pe.OPTIONAL_HEADER.AddressOfEntryPoint = 0
        pe.write(str(no_entry))
        for case in ("no entry", "null output", "unreadable callbacks"):
            image = no_entry if case == "no entry" else original
            modules = {IMPORTED_DLL: ctypes.CDLL(str(provider))}
            base, status, detail, _ = mapper(image, modules)
            if not base:
                raise AssertionError(f"{case} map failed: {status} {detail}")
            old = ctypes.c_uint32()
            try:
                before = ctypes.string_at(base + data.VirtualAddress, data.Misc_VirtualSize)
                if case == "unreadable callbacks":
                    if not kernel.VirtualProtect(base + table, 8, 1, ctypes.byref(old)):
                        raise ctypes.WinError(ctypes.get_last_error())
                output = ctypes.c_uint32(0xdeadbeef)
                argument = None if case == "null output" else ctypes.byref(output)
                result = start(base, argument)
                after = ctypes.string_at(base + data.VirtualAddress, data.Misc_VirtualSize)
                if result != 0 or output.value != 0xdeadbeef or before != after:
                    raise AssertionError(f"{case}: result={result} output={output.value:#x}")
            finally:
                if old.value:
                    kernel.VirtualProtect(base + table, 8, old.value, ctypes.byref(old))
                mapper.release(base)
            print(f"refused {case} before guest startup")


def check_entry_outside_code(mapper, original, provider):
    """An entry point in .data: pe_map rejects the image before running anything; Windows runs it and faults."""
    with pefile.PE(str(original)) as pe:
        field = pe.OPTIONAL_HEADER.get_field_absolute_offset("AddressOfEntryPoint")
        data = next(sec.VirtualAddress for sec in pe.sections if sec.Name.rstrip(b"\x00") == b".data")
    with TemporaryDirectory() as temp:
        copy = bytearray(Path(original).read_bytes())
        struct.pack_into("<I", copy, field, data)
        image = Path(temp) / Path(original).name
        image.write_bytes(copy)
        shutil.copy(provider, Path(temp) / provider.name)
        launched = subprocess.run([str(image)], capture_output=True, timeout=60)
        refused = mapped(mapper, image, [provider])
    if launched.returncode & 0xffffffff != ACCESS_VIOLATION:
        raise AssertionError(f"entry point in .data: launch exited {launched.returncode & 0xffffffff:#x}")
    if refused.get("status") != STATUS["format"] or refused.get("detail") != "entry point outside executable section":
        raise AssertionError(f"entry point in .data: {refused}")
    print(f"entry point in .data: Windows faults ({ACCESS_VIOLATION:#x}), pe_map refuses "
          f"'{refused['detail']}' and runs nothing")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run(*sys.argv[2:5])
    else:
        parser = argparse.ArgumentParser()
        for name in ("mapper", "original", "no_tls", "provider", "altered"):
            parser.add_argument(name, type=Path)
        args = parser.parse_args()
        check(args.mapper, args.original, args.no_tls, args.provider, args.altered)
        check_dll_refused(args.mapper, args.original, args.provider)
        check_refusals(args.mapper, args.original, args.provider)
        check_entry_outside_code(args.mapper, args.original, args.provider)
