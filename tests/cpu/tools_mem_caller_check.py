"""Start the XMemIsToolsMemAvailable caller under the Windows loader and pe_map; compare what it does next.

tests/cpu/tools_mem_caller.asm imports XMemIsToolsMemAvailable by name from xmem.dll and finish_exe
(exe_startup.asm) from cpu_exe_io.dll. Here xmem.dll is the test-only recording provider
(tests/cpu/tools_mem_provider.cpp), not the desktop shim. The provider returns the BOOL chosen by
TOOLS_MEM_RESULT and appends one line per call to TOOLS_MEM_RECORD. The caller calls three times, then
records its branch as the exit code: 0x7100 (FALSE, continue without tools memory) or 0x7101 (TRUE).

FALSE models a desktop without a tools partition. That is proposed desktop policy, following the public
reimplementation cited on #870; it does not settle the console ordering, and this fixture does not show
that any real startup caller continues on FALSE.
"""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from exe_process import launch
from native_map_check import Mapper, STATUS

NAME = "XMemIsToolsMemAvailable"
XMEM, IO = "xmem.dll", "cpu_exe_io.dll"
REQUESTS = sorted([[XMEM, NAME], [IO, "finish_exe"]])
NO_TOOLS, TOOLS = 0x7100, 0x7101
ENTRYPOINT_NOT_FOUND = 0xC0000139
CASES = {
    # mode: (label, expected exit, expected record lines)
    "0": ("FALSE: desktop without a tools partition (proposed desktop policy)", NO_TOOLS,
          ["query 1 0", "query 2 0", "query 3 0"]),
    "1": ("TRUE: tools memory reported", TOOLS, ["query 1 1", "query 2 1", "query 3 1"]),
    # Negative controls: an answer that changes between calls, and an unconfigured provider.
    "flip": ("control: answer changes between calls", 0xF02, ["query 1 0", "query 2 1"]),
    "unset": ("control: unconfigured provider", 0xF01, ["query 1 2"]),
}


def import_names(image):
    pe = pefile.PE(str(image), fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    rows = {}
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        rows[entry.dll.decode().casefold()] = [(item.name.decode() if item.name else None, item.import_by_ordinal)
                                               for item in entry.imports]
    pe.close()
    return rows


def export_names(image):
    pe = pefile.PE(str(image), fast_load=True)
    pe.parse_data_directories([pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])
    names = sorted(item.name.decode() for item in pe.DIRECTORY_ENTRY_EXPORT.symbols if item.name)
    pe.close()
    return names


def environment(mode, record):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("TOOLS_MEM_")}
    if mode != "unset":
        env["TOOLS_MEM_RESULT"] = mode
    env["TOOLS_MEM_RECORD"] = str(record)
    return env


def take_record(record):
    lines = record.read_text().splitlines() if record.exists() else []
    record.unlink(missing_ok=True)
    return lines


def native(image, env):
    row = launch([image], b"", env=env, timeout=30)
    if not row["process_signaled"]:
        raise AssertionError(f"{image.name}: process did not end: {row}")
    if row["stdout"]["bytes"] or row["stderr"]["bytes"]:
        raise AssertionError(f"{image.name}: unexpected output {row['stdout']['bytes']!r} {row['stderr']['bytes']!r}")
    return row["code"] & 0xFFFFFFFF


def child(mapper_path, image, xmem, io):
    """Map and start the EXE in this (child) process; print the observation as JSON."""
    mapper = Mapper(mapper_path)
    run_entry = ctypes.CDLL(str(mapper_path)).pe_map_run
    run_entry.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    run_entry.restype = ctypes.c_int
    provider = ctypes.WinDLL(str(xmem))
    provider.tools_mem_calls.restype = ctypes.c_long
    bindings = {XMEM: provider, IO.casefold(): ctypes.CDLL(str(io))}
    base, status, detail, requests = mapper(Path(image), bindings)
    if not base:
        print(json.dumps({"status": status, "detail": detail, "requests": requests,
                          "calls": provider.tools_mem_calls()}))
        return
    code = ctypes.c_uint32(0xdeadbeef)
    started = run_entry(base, ctypes.byref(code))
    mapper.release(base)
    print(json.dumps({"status": status, "started": started, "exit": code.value, "requests": requests,
                      "calls": provider.tools_mem_calls()}))


def mapped(mapper_path, image, xmem, io, env):
    result = subprocess.run([sys.executable, "-B", __file__, "--child", str(mapper_path), str(image),
                             str(xmem), str(io)], capture_output=True, text=True, timeout=60, env=env)
    if result.returncode or result.stderr:
        raise AssertionError(f"mapped {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def main(mapper_path, caller, provider, provider_missing, io):
    imports = import_names(caller)
    if imports.get(XMEM) != [(NAME, False)]:
        raise AssertionError(f"caller import table: {XMEM} entries {imports.get(XMEM)}, expected {NAME} by name")
    print(f"import table: {XMEM}!{NAME} by name")
    exports, control_exports = export_names(provider), export_names(provider_missing)
    if exports != sorted([NAME, "tools_mem_calls"]) or NAME in control_exports:
        raise AssertionError(f"provider exports {exports}, missing-export control {control_exports}")

    with TemporaryDirectory() as temp:
        temp = Path(temp)
        good, missing = temp / "good", temp / "missing"
        for folder, xmem in ((good, provider), (missing, provider_missing)):
            folder.mkdir()
            for name, source in ((caller.name, caller), (XMEM, xmem), (IO, io)):
                shutil.copyfile(source, folder / name)
        record = temp / "record.txt"
        branches = {}
        for mode, (label, expected, lines) in CASES.items():
            env = environment(mode, record)
            launched = native(good / caller.name, env)
            native_record = take_record(record)
            result = mapped(mapper_path, good / caller.name, good / XMEM, good / IO, env)
            mapped_record = take_record(record)
            if result.get("status") != STATUS["ok"] or result.get("started") != 1:
                raise AssertionError(f"mapped {mode}: {result}")
            requests = sorted([dll.casefold(), symbol] for dll, symbol in result["requests"])
            if requests != REQUESTS:
                raise AssertionError(f"mapped {mode}: import requests {result['requests']}")
            if launched != expected or result["exit"] != expected:
                raise AssertionError(f"{label}: Windows launch exited {launched:#x}, pe_map {result['exit']:#x}, "
                                     f"expected {expected:#x} for both")
            if native_record != lines or mapped_record != lines or result["calls"] != len(lines):
                raise AssertionError(f"{label}: records native {native_record}, pe_map {mapped_record} "
                                     f"({result['calls']} calls), expected {lines}")
            branches[mode] = (launched, result["exit"])
            print(f"{label}: Windows launch {launched:#x}, pe_map {result['exit']:#x}, record {native_record}")

        # Swapping the outcome changes the recorded branch under both loaders.
        if any(false == true for false, true in zip(branches["0"], branches["1"])):
            raise AssertionError(f"swapped outcome kept the branch: FALSE {branches['0']}, TRUE {branches['1']}")
        print(f"swapped outcome: FALSE {branches['0'][0]:#x}, TRUE {branches['1'][0]:#x} under both loaders")

        # A provider without the export stops both loaders before the entry point runs.
        env = environment("0", record)
        launched = native(missing / caller.name, env)
        native_record = take_record(record)
        result = mapped(mapper_path, missing / caller.name, missing / XMEM, missing / IO, env)
        mapped_record = take_record(record)
        if launched != ENTRYPOINT_NOT_FOUND or native_record:
            raise AssertionError(f"provider without {NAME}: Windows launch {launched:#x}, record {native_record}")
        if result.get("status") != STATUS["unresolved"] or "started" in result or mapped_record or \
                result.get("calls") != 0 or result.get("detail") != f"unresolved import {XMEM}!{NAME}":
            raise AssertionError(f"provider without {NAME}: pe_map {result}, record {mapped_record}")
        print(f"provider without {NAME}: Windows launch {launched:#x}, pe_map {result['detail']}, no calls")
    print("XMemIsToolsMemAvailable caller: PASS")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:6])
    else:
        main(*map(Path, sys.argv[1:]))
