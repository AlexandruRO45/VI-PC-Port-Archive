"""Compare the source-built physical-page EXE under Windows and pe_map."""

import ctypes
from pathlib import Path
import subprocess
import sys

import pefile

from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): sorted(imp.name.decode() for imp in module.imports if imp.name)
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def compare(mapper_path, exe_path, xmem_path):
    mapper_path, exe, xmem = map(Path, (mapper_path, exe_path, xmem_path))
    imports = imports_by_dll(exe)
    check("xmem imports", imports.get("xmem.dll"), ["XMemAllocatePhysicalPages", "XMemFreePhysicalPages"])

    native = subprocess.run([str(exe)], cwd=str(exe.parent), timeout=20)
    check("native physical pages", native.returncode, 0)
    print(f"native: exit {native.returncode}; modules {sorted(imports)}")

    providers = {}
    for dll in imports:
        providers[dll] = ctypes.WinDLL(str(xmem)) if dll == "xmem.dll" else ctypes.WinDLL(dll)

    mapper = Mapper(mapper_path)
    base, status, detail, _ = mapper(exe, {name: module for name, module in providers.items() if name != "xmem.dll"})
    if base:
        mapper.release(base)
        raise AssertionError("physical-page EXE mapped without xmem.dll")
    check("missing xmem status", status, STATUS["unresolved"])
    if not detail.startswith("unresolved import xmem.dll!XMem"):
        raise AssertionError(f"missing xmem diagnostic: {detail}")
    print(f"before entry: {detail}")

    mapped = run_mapped(mapper_path, exe, providers)
    check("mapped physical pages", mapped, native.returncode)
    print(f"mapped: matching exit {mapped}")


if __name__ == "__main__":
    compare(*sys.argv[1:])