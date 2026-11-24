"""Run the lifecycle of two DLLs built with the static MSVC runtime under the Windows loader and under pe_map.

Each case runs in fresh child processes with a timeout. The child loads the event sink as an ordinary DLL, loads
or maps A and B, calls their exports, unloads them, reloads them and unloads them again, and reports the
events and values of every stage. The same scenario runs through LoadLibrary/FreeLibrary and through
pe_map_image/pe_map_export/pe_map_release, and the reports are compared stage by stage. The harness, not
pe_map, orders the mapping and the release of the images and serves the import of A; Windows does both itself."""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, Resolver

SINK, A, B = "cpu_life_sink.dll", "cpu_life_a.dll", "cpu_life_b.dll"
TIMEOUT = 30
PROCESSES = 3  # fresh child processes per loader and case
EXTRA_CYCLES = 2  # load/unload cycles after the first reload
MEM_FREE = 0x10000
LOAD_DEFAULT_AND_DLL_DIR = 0x1000 | 0x100  # LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
kernel.LoadLibraryExW.restype = wintypes.HMODULE
kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
kernel.FreeLibrary.restype = wintypes.BOOL
kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel.GetModuleHandleW.restype = wintypes.HMODULE
kernel.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_char_p]
kernel.GetProcAddress.restype = ctypes.c_void_p
kernel.VirtualQuery.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
kernel.VirtualQuery.restype = ctypes.c_size_t


class MemoryInfo(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wintypes.DWORD), ("PartitionId", wintypes.WORD),
                ("RegionSize", ctypes.c_size_t), ("State", wintypes.DWORD), ("Protect", wintypes.DWORD),
                ("Type", wintypes.DWORD)]


def is_free(address):
    info = MemoryInfo()
    return bool(kernel.VirtualQuery(address, ctypes.byref(info), ctypes.sizeof(info)) and info.State == MEM_FREE)


class Events:
    """Reads the sink and returns only the tokens logged since the previous call."""

    def __init__(self, sink):
        self.snapshot = sink.sink_snapshot
        self.snapshot.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        self.snapshot.restype = ctypes.c_uint
        self.seen = 0

    def new(self):
        buffer = ctypes.create_string_buffer(1024)
        length = self.snapshot(buffer, len(buffer))  # fill the buffer before slicing it
        tokens = buffer.raw[:length].decode().split(",")[:-1]
        fresh, self.seen = tokens[self.seen:], len(tokens)
        return fresh


class Native:
    """LoadLibrary and FreeLibrary: the loader finds A as B's dependency and decides every order."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.b = None

    def load(self):
        self.b = kernel.LoadLibraryExW(str(self.directory / B), None, LOAD_DEFAULT_AND_DLL_DIR)
        if not self.b:
            raise OSError(f"LoadLibraryExW failed: {ctypes.get_last_error()}")

    def function(self, name, restype=ctypes.c_int, argtypes=()):
        owner = kernel.GetModuleHandleW(A if name.startswith("a_") else B)
        return ctypes.CFUNCTYPE(restype, *argtypes)(kernel.GetProcAddress(owner, name.encode()))

    def unload(self):
        if not kernel.FreeLibrary(self.b):
            raise OSError("FreeLibrary failed")
        self.b = None

    def resident(self):
        return {"A": bool(kernel.GetModuleHandleW(A)), "B": bool(kernel.GetModuleHandleW(B))}

    def base(self, name):
        return kernel.GetModuleHandleW(name)


class Mapped:
    """pe_map: the harness maps A before B, serves B's import of A, and releases B before A."""

    def __init__(self, directory, mapper_path):
        self.directory = Path(directory)
        self.mapper = Mapper(mapper_path)
        self.live, self.sink_module, self.system, self.bases = {}, None, {}, {}

    def _resolve(self, dll, name, ordinal):
        dll = dll.decode()
        if dll == A:
            return self.mapper.export(self.live[A], name) if A in self.live and name else None
        if dll == SINK:
            return kernel.GetProcAddress(self.sink_module._handle, name) if name else None
        module = self.system.setdefault(dll, ctypes.WinDLL(dll))
        return ctypes.cast(getattr(module, name.decode()), ctypes.c_void_p).value if name else None

    def map_one(self, name):
        self.callback = Resolver(self._resolve)
        status, detail = ctypes.c_uint32(0xffffffff), ctypes.create_string_buffer(256)
        base = self.mapper.map(str(self.directory / name), self.callback, ctypes.byref(status), detail, len(detail))
        if not base:
            raise OSError(f"pe_map_image({name}) failed with {status.value}: {detail.value.decode()}")
        self.live[name] = base

    def load(self):
        for name in dependency_order(self.directory, (A, B)):
            self.map_one(name)

    def function(self, name, restype=ctypes.c_int, argtypes=()):
        owner = A if name.startswith("a_") else B
        return ctypes.CFUNCTYPE(restype, *argtypes)(self.mapper.export(self.live[owner], name.encode()))

    def unload(self):
        bases = dict(self.live)
        for name in reversed(dependency_order(self.directory, (A, B))):
            self.mapper.release(self.live.pop(name))
        # Asked right after the releases, before anything else can reuse the address space.
        self.after_release = {"A": not is_free(bases[A]), "B": not is_free(bases[B])}

    def resident(self):
        return self.after_release

    def base(self, name):
        return self.live[name]


def dependency_order(directory, names):
    """Dependencies first, from the import tables of the files."""
    order = []

    def visit(name):
        if name in order:
            return
        with pefile.PE(str(Path(directory) / name), fast_load=True) as pe:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            wanted = [e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
        for dependency in wanted:
            if dependency in names:
                visit(dependency)
        order.append(name)

    for name in names:
        visit(name)
    return order


def crt_dispatch(directory, name):
    """The function the entry point tail-jumps to, as an RVA range, plus the image's size and preferred base.

    The PE entry point is the runtime's _DllMainCRTStartup. It ends in `jmp rel32` to the function that
    dispatches DLL_PROCESS_ATTACH and DETACH (running the global constructors and destructors), so the
    entry point itself is not on the constructor's stack but that function is."""
    with pefile.PE(str(Path(directory) / name), fast_load=True) as pe:
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXCEPTION"]])
        entry = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        ranges = [(f.struct.BeginAddress, f.struct.EndAddress) for f in pe.DIRECTORY_ENTRY_EXCEPTION]
        begin, end = next((b, e) for b, e in ranges if b == entry)
        tail = pe.get_data(end - 5, 5)
        if tail[0] != 0xE9:
            raise AssertionError(f"{name}: the entry-point function does not end in a jmp rel32: {tail.hex()}")
        target = end + int.from_bytes(tail[1:], "little", signed=True)
        dispatch = next((b, e) for b, e in ranges if b == target)
        return dispatch, pe.OPTIONAL_HEADER.SizeOfImage, pe.OPTIONAL_HEADER.ImageBase


def called_from_entry(backend, name):
    """True if the constructor's stack holds the CRT's dispatch function, reached from the entry point, and the
    frame above it is outside the image: the caller is the loader (or pe_map), not code of the DLL."""
    (begin, end), size, _ = crt_dispatch(backend.directory, name)
    frames = (ctypes.c_ulonglong * 16)()
    count = backend.function("a_ctor_frames" if name == A else "b_ctor_frames", ctypes.c_uint,
                             (ctypes.c_void_p,))(ctypes.addressof(frames))
    base = backend.base(name)
    rvas = [frames[i] - base for i in range(count)]
    inside = [i for i, rva in enumerate(rvas) if begin <= rva < end]
    return bool(inside) and inside[0] + 1 < count and not 0 <= rvas[inside[0] + 1] < size


def values(backend):
    return {"a_value": backend.function("a_value")(),
            "a_ctor_runs": backend.function("a_ctor_runs", ctypes.c_uint)(),
            "b_value": backend.function("b_value")(), "b_via_a": backend.function("b_via_a")(),
            "a_ctor_called_from_crt_entry": called_from_entry(backend, A),
            "b_ctor_called_from_crt_entry": called_from_entry(backend, B)}


def scenario(backend, events):
    """Load, call, change A's state, unload, reload, unload. Every stage lists the events it caused."""
    stages = []
    backend.load()
    bases = {"A": backend.base(A), "B": backend.base(B)}
    stages.append({"stage": "load", "events": events.new(), "values": values(backend)})
    backend.function("a_set", None, (ctypes.c_int,))(99)
    stages.append({"stage": "after a_set(99)", "events": events.new(), "values": values(backend)})
    backend.unload()
    stages.append({"stage": "unload", "events": events.new(), "resident": backend.resident()})
    backend.load()
    stages.append({"stage": "reload", "events": events.new(), "values": values(backend)})
    backend.unload()
    stages.append({"stage": "unload again", "events": events.new(), "resident": backend.resident()})
    for cycle in range(3, 3 + EXTRA_CYCLES):
        backend.load()
        stages.append({"stage": f"cycle {cycle} load", "events": events.new(), "values": values(backend)})
        backend.unload()
        stages.append({"stage": f"cycle {cycle} unload", "events": events.new(), "resident": backend.resident()})
    return stages, bases


def refcount_native(backend, events):
    """What the Windows loader does on its own: count references and hold A while B imports it."""
    backend.load()
    first = events.new()
    again = kernel.LoadLibraryExW(str(backend.directory / B), None, LOAD_DEFAULT_AND_DLL_DIR)
    same_handle, after_second_load = again == backend.b, events.new()
    a_extra = kernel.LoadLibraryExW(str(backend.directory / A), None, LOAD_DEFAULT_AND_DLL_DIR)
    kernel.FreeLibrary(again)
    after_one_free = {"events": events.new(), "resident": backend.resident()}
    backend.unload()  # B's last reference: B is destroyed, A stays because the harness still holds it
    b_gone_a_held = {"events": events.new(), "resident": backend.resident()}
    kernel.FreeLibrary(a_extra)
    return {"first_load": first, "same_handle": same_handle, "after_second_load": after_second_load,
            "after_one_free": after_one_free, "b_gone_a_held": b_gone_a_held,
            "a_released": {"events": events.new(), "resident": backend.resident()}}


def second_copy_mapped(backend, events):
    """pe_map has no module table: mapping B twice gives two images, each with its own constructor run."""
    backend.load()
    first = events.new()
    first_base = backend.live[B]
    backend.map_one(B)
    second = {"events": events.new(), "distinct_base": backend.live[B] != first_base}
    state = [backend.function("b_value")(), backend.function("b_via_a")()]
    backend.mapper.release(backend.live.pop(B))
    backend.mapper.release(first_base)
    backend.mapper.release(backend.live.pop(A))
    return {"first_load": first, "second_map": second, "second_copy_values": state}


def child(mode, mapper, directory):
    os.add_dll_directory(str(directory))
    sink = ctypes.WinDLL(str(Path(directory) / SINK))  # stays loaded for the whole child, under both loaders
    events = Events(sink)
    if mode in ("native", "refcount"):
        backend = Native(directory)
    else:
        backend = Mapped(directory, mapper)
        backend.sink_module = sink
    if mode in ("native", "mapped"):
        stages, bases = scenario(backend, events)
        print(json.dumps({"stages": stages, "sink_loaded": bool(kernel.GetModuleHandleW(SINK)), "bases": bases}))
    elif mode == "refcount":
        print(json.dumps(refcount_native(backend, events)))
    else:
        print(json.dumps(second_copy_mapped(backend, events)))


def run(mode, mapper, directory):
    command = [sys.executable, "-B", __file__, "--child", mode, str(mapper), str(directory)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT)
    if done.returncode or done.stderr:
        raise AssertionError(f"{mode}: child exited {done.returncode:#x}: {done.stderr}")
    return json.loads(done.stdout)


def stage_directory(temp, sink, a, b):
    """A directory holding the sink and the chosen A and B under the names the DLLs import."""
    for name, path in ((SINK, sink), (A, a), (B, b)):
        shutil.copy(path, Path(temp) / name)
    return temp


FIRST = ["A:ctor", "A:attach", "B:ctor", "B:attach"]
FIRST_UNLOAD = ["B:detach", "B:dtor a=7", "A:detach", "A:dtor"]
NONE_RESIDENT = {"A": False, "B": False}


def loaded(a_value=7, b_via_a=21):
    return {"a_value": a_value, "a_ctor_runs": 1, "b_value": 14, "b_via_a": b_via_a,
            "a_ctor_called_from_crt_entry": True, "b_ctor_called_from_crt_entry": True}


LIFECYCLE = [
    {"stage": "load", "events": FIRST, "values": loaded()},
    {"stage": "after a_set(99)", "events": [], "values": loaded(99, 113)},
    {"stage": "unload", "events": ["B:detach", "B:dtor a=9", "A:detach", "A:dtor"], "resident": NONE_RESIDENT},
    {"stage": "reload", "events": FIRST, "values": loaded()},
    {"stage": "unload again", "events": FIRST_UNLOAD, "resident": NONE_RESIDENT},
] + [stage for cycle in range(3, 3 + EXTRA_CYCLES) for stage in (
    {"stage": f"cycle {cycle} load", "events": FIRST, "values": loaded()},
    {"stage": f"cycle {cycle} unload", "events": FIRST_UNLOAD, "resident": NONE_RESIDENT})]


def run_all(mapper, directory, mode, processes=1):
    """The same scenario in fresh child processes; every run must report the same stages."""
    reports = [run(mode, mapper, directory) for _ in range(processes)]
    for report in reports[1:]:
        if report["stages"] != reports[0]["stages"]:
            raise AssertionError(f"{mode}: two fresh processes disagree:\n  {reports[0]['stages']}\n  {report['stages']}")
    if not all(report["sink_loaded"] for report in reports):
        raise AssertionError(f"{mode}: the sink was not loaded at the end of the run")
    return reports


def check(mapper, sink, a, b, a_noctor, b_nodtor):
    cases = {"original": (a, b), "without A's constructor": (a_noctor, b), "without B's destructor": (a, b_nodtor)}
    reports = {}
    for name, (dll_a, dll_b) in cases.items():
        with TemporaryDirectory() as temp:
            directory = stage_directory(temp, sink, dll_a, dll_b)
            processes = PROCESSES if name == "original" else 1
            reports[name] = {mode: run_all(mapper, directory, mode, processes) for mode in ("native", "mapped")}
            preferred = crt_dispatch(directory, B)[2]
        for native, mapped in zip(reports[name]["native"][0]["stages"], reports[name]["mapped"][0]["stages"]):
            if native != mapped:
                raise AssertionError(f"{name}: first difference, stage {native['stage']!r}:\n"
                                     f"  Windows {native}\n  pe_map  {mapped}")
    native_stages = reports["original"]["native"][0]["stages"]
    if native_stages != LIFECYCLE:
        raise AssertionError(f"Windows lifecycle changed: {native_stages}")
    print(f"original: Windows and pe_map agree on every event and value in {PROCESSES} fresh processes each, over "
          f"load, a_set, unload, reload, unload and {EXTRA_CYCLES} more cycles. The first cycle:")
    for stage in LIFECYCLE[:5]:
        print(f"  {stage['stage']}: {stage['events']} {stage.get('values', stage.get('resident'))}")
    print("  a_ctor_called_from_crt_entry and b_ctor_called_from_crt_entry are true: the function that "
          "_DllMainCRTStartup tail-jumps to is on the constructor's stack, directly below the loader's (or "
          "pe_map's) frame.")
    mapped_bases = [report["bases"] for report in reports["original"]["mapped"]]
    if any(bases["B"] in (0, preferred) for bases in mapped_bases):
        raise AssertionError(f"pe_map mapped B at its preferred base or at 0: {mapped_bases}")
    print(f"mapped B bases: {[hex(b['B']) for b in mapped_bases]} (preferred {preferred:#x}); the image is relocated "
          "on every mapping. Windows bases: "
          f"{[hex(r['bases']['B']) for r in reports['original']['native']]}.")

    ctor = reports["without A's constructor"]["native"][0]["stages"]
    zero = {"a_value": 0, "a_ctor_runs": 0, "b_value": 0, "b_via_a": 0,
            "a_ctor_called_from_crt_entry": False, "b_ctor_called_from_crt_entry": True}
    if ctor[0]["events"] != ["A:attach", "B:ctor", "B:attach"] or ctor[0]["values"] != zero:
        raise AssertionError(f"control without A's constructor: {ctor[0]}")
    dtor = reports["without B's destructor"]["native"][0]["stages"]
    if dtor[2]["events"] != ["B:detach", "A:detach", "A:dtor"]:
        raise AssertionError(f"control without B's destructor: {dtor[2]}")
    print("controls, identical under both loaders: without A's constructor the load logs "
          f"{ctor[0]['events']} and every value is 0; without B's destructor the unload logs {dtor[2]['events']}")

    with TemporaryDirectory() as temp:
        directory = stage_directory(temp, sink, a, b)
        native = run("refcount", mapper, directory)
        mapped = run("second-copy", mapper, directory)
    expected_native = {
        "first_load": FIRST, "same_handle": True, "after_second_load": [],
        "after_one_free": {"events": [], "resident": {"A": True, "B": True}},
        "b_gone_a_held": {"events": ["B:detach", "B:dtor a=7"], "resident": {"A": True, "B": False}},
        "a_released": {"events": ["A:detach", "A:dtor"], "resident": NONE_RESIDENT}}
    if native != expected_native:
        raise AssertionError(f"loader reference counting: {native}")
    expected_mapped = {"first_load": FIRST, "second_map": {"events": ["B:ctor", "B:attach"], "distinct_base": True},
                       "second_copy_values": [14, 21]}
    if mapped != expected_mapped:
        raise AssertionError(f"second map of B: {mapped}")
    print("Windows counts references: a second LoadLibrary of B returns the same module and runs nothing, one "
          "FreeLibrary keeps it, and A stays loaded while the harness or B holds it.")
    print("pe_map keeps no module table and no reference counts: mapping B twice gives a second image that runs "
          "its constructor again. Mapping order, release order and the import of A are the harness's work.")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(*sys.argv[2:5])
    else:
        check(*map(Path, sys.argv[1:7]))
