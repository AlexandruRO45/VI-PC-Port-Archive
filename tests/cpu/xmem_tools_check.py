"""Compare a source-built XMemIsToolsMemAvailable caller with recording providers."""

import ctypes
from pathlib import Path
import subprocess
import sys

import pefile

from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS

EXPECTED = {
    "tools partition": 0x7B11,
    "desktop without a tools partition": 0x7A21,
}


def check(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
        return {module.dll.decode().casefold(): [imp.name.decode() for imp in module.imports if imp.name]
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def compare(mapper_path, true_caller, true_provider, false_caller, false_provider):
    cases = (
        ("tools partition", Path(true_caller), Path(true_provider)),
        ("desktop without a tools partition", Path(false_caller), Path(false_provider)),
    )
    for label, caller, provider in cases:
        imports = imports_by_dll(caller)
        check(f"{label} imports", set(imports), {provider.name.casefold()})
        check(f"{label} symbols", set(imports[provider.name.casefold()]),
              {"XMemIsToolsMemAvailable", "xmem_tools_note", "xmem_tools_note_value", "xmem_tools_calls"})

        native = subprocess.run([str(caller)], cwd=str(caller.parent), timeout=20)
        check(f"{label} native", native.returncode, EXPECTED[label])
        print(f"{label}: native exit {native.returncode:#x}")

        mapper = Mapper(mapper_path)
        base, status, detail, _ = mapper(caller, {})
        if base:
            mapper.release(base)
            raise AssertionError(f"{label}: mapped without the provider")
        check(f"{label} missing-provider status", status, STATUS["unresolved"])
        if not detail.startswith("unresolved import ") or provider.name.casefold() not in detail.casefold():
            raise AssertionError(f"{label} missing-provider diagnostic: {detail}")
        print(f"{label}: before entry, {detail}")

        module = ctypes.WinDLL(str(provider))
        module.xmem_tools_reset()
        mapped = run_mapped(mapper_path, caller, {provider.name.casefold(): module})
        check(f"{label} mapped", mapped, native.returncode)
        print(f"{label}: mapped exit {mapped:#x}")


if __name__ == "__main__":
    compare(*sys.argv[1:])