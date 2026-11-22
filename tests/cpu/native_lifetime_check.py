"""Check thread notification order, release ownership and teardown of mapped images."""

import argparse
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory

from native_map_check import Mapper, STATUS, lookup_entry
from native_tls_check import Call, DETACH, ENTRY, TLS, load_sink, new_thread_calls
import pefile


THREAD_ATTACH = 2
THREAD_DETACH = 3
CASES = ("two-images-native", "two-images", "release-other-during-notification",
         "release-self-during-notification", "release-unowned-addresses", "concurrent-release",
         "release-during-notification-on-another-thread")
RELEASE_CASES = CASES[4:]
RACE_ROUNDS = 50
RACE_THREADS = 8
MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
PAGE_NOACCESS, PAGE_READWRITE = 0x01, 0x04
HOOK = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
CHILD_TIMEOUT = 60


def notification(reason, module):
    return [[TLS, reason, module, 0], [ENTRY, reason, module, 0]]


def expect(label, observed, expected):
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {observed!r}")


def process_detaches(sink, base):
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    if count >= len(buffer):
        raise AssertionError("sink log reached capacity")
    return [[call.caller, call.reason, "module" if call.module == base else hex(call.module),
             call.reserved] for call in buffer[:count] if call.reason == DETACH]


class Kernel:
    """The kernel32 calls the release cases need."""

    def __init__(self):
        self.dll = ctypes.WinDLL("kernel32", use_last_error=True)
        self.dll.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        self.dll.VirtualAlloc.restype = ctypes.c_void_p
        self.dll.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
        self.dll.VirtualQuery.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.dll.VirtualQuery.restype = ctypes.c_size_t
        self.dll.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                            ctypes.POINTER(ctypes.c_uint32)]

    def state(self, address):
        """(State, Protect) of the page holding address."""
        info = (ctypes.c_uint64 * 6)()  # MEMORY_BASIC_INFORMATION is 48 bytes on x64
        if not self.dll.VirtualQuery(address, info, ctypes.sizeof(info)):
            raise AssertionError(f"VirtualQuery failed at {address:#x}")
        raw = bytes(info)
        return int.from_bytes(raw[32:36], "little"), int.from_bytes(raw[36:40], "little")

    def unwind_entry(self, address):
        image_base = ctypes.c_uint64()
        return lookup_entry(address, ctypes.byref(image_base), None)

    def protect(self, address, protection):
        previous = ctypes.c_uint32()
        if not self.dll.VirtualProtect(address, 4096, protection, ctypes.byref(previous)):
            raise AssertionError(f"VirtualProtect failed at {address:#x}")
        return previous.value


def image_pages(image):
    """(entry RVA, quiet page RVA). The entry point lies inside a .pdata function, so its unwind entry
    shows the table is live. The quiet page is .rsrc, which no notification or release touches, so it
    can be made unreadable while other threads in the process start and stop."""
    with pefile.PE(str(image), fast_load=True) as pe:
        if not pe.OPTIONAL_HEADER.DATA_DIRECTORY[3].Size:
            raise AssertionError("the fixture needs .pdata for the unwind-table checks")
        quiet = [section.VirtualAddress for section in pe.sections if section.Name.rstrip(b"\0") == b".rsrc"]
        if not quiet:
            raise AssertionError("the fixture needs a .rsrc section for the unreadable interior page")
        return pe.OPTIONAL_HEADER.AddressOfEntryPoint, quiet[0]


def run(case, mapper_path, image, sink_path):
    sink = load_sink(sink_path)
    sink.sink_on_thread_attach.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    sink.sink_on_thread_attach.restype = None
    mapper = Mapper(mapper_path)
    mapper.release.restype = ctypes.c_int
    bindings = {sink_path.name.casefold(): sink}

    def map_copy(directory, name):
        copy = Path(directory) / name
        shutil.copyfile(image, copy)
        base, status, detail, _ = mapper(copy, bindings)
        if status != STATUS["ok"] or not base:
            raise AssertionError(f"{name}: mapping failed (status {status}, detail {detail!r})")
        return base

    if case == "two-images-native":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        with TemporaryDirectory() as a_dir, TemporaryDirectory() as b_dir:
            copies = [Path(a_dir) / "tls_a.dll", Path(b_dir) / "tls_b.dll"]
            for copy in copies:
                shutil.copyfile(image, copy)
            a, b = (kernel.LoadLibraryW(str(copy)) for copy in copies)
            if not a or not b or a == b:
                raise AssertionError(f"native loads failed: {a=} {b=}")
            try:
                expect("native worker thread", new_thread_calls(sink, a),
                       notification(THREAD_ATTACH, "module") + notification(THREAD_ATTACH, hex(b)) +
                       notification(THREAD_DETACH, hex(b)) + notification(THREAD_DETACH, "module"))
            finally:
                kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
                kernel.FreeLibrary(b)
                kernel.FreeLibrary(a)
        return

    if case in RELEASE_CASES:
        run_release_case(case, mapper, sink, map_copy, *image_pages(image))
        return

    with TemporaryDirectory() as a_dir, TemporaryDirectory() as b_dir:
        a = map_copy(a_dir, "tls_a.dll")
        b = None
        if case != "release-self-during-notification":
            b = map_copy(b_dir, "tls_b.dll")
            if b == a:
                raise AssertionError(f"A and B share a module base {a:#x}")

        a_attach = notification(THREAD_ATTACH, "module")
        a_detach = notification(THREAD_DETACH, "module")
        a_release = notification(DETACH, "module")
        b_release = notification(DETACH, hex(b)) if b is not None else []

        if case == "two-images":
            # Measured with LoadLibraryW in two-images-native: attach in load order, detach reversed.
            expect("worker thread", new_thread_calls(sink, a),
                   a_attach + notification(THREAD_ATTACH, hex(b)) +
                   notification(THREAD_DETACH, hex(b)) + a_detach)
            expect("release B", mapper.release(b), 1)
            expect("process detach after release B", process_detaches(sink, a), b_release)
            expect("release A", mapper.release(a), 1)
            expect("process detach after release A", process_detaches(sink, a), b_release + a_release)
        else:
            target = b if case == "release-other-during-notification" else a
            # The sink calls pe_map_release directly from the guest TLS callback.
            hook = ctypes.cast(mapper.release, ctypes.c_void_p)
            arm = lambda thread: sink.sink_on_thread_attach(hook, target, thread)
            if case == "release-other-during-notification":
                # B is released from A's TLS callback while the notifier still holds B's reference.
                # B then gets no thread notification; its process detach runs when the notifier
                # drops that reference, right after A's attach, and B is gone before the detach.
                expect("worker thread", new_thread_calls(sink, a, arm), a_attach + b_release + a_detach)
                expect("second release B", mapper.release(b), 0)
                expect("process detach after second release B", process_detaches(sink, a), b_release)
                expect("release A", mapper.release(a), 1)
                expect("process detach after release A", process_detaches(sink, a), b_release + a_release)
            else:
                expect("worker thread", new_thread_calls(sink, a, arm), a_attach + a_release)
                expect("second release A", mapper.release(a), 0)
                expect("process detach after second release A", process_detaches(sink, a), a_release)


def run_release_case(case, mapper, sink, map_copy, entry, quiet):
    kernel = Kernel()
    mapper.release.argtypes = [ctypes.c_void_p]
    a_release = notification(DETACH, "module")
    with TemporaryDirectory() as directory:
        if case == "release-unowned-addresses":
            a = map_copy(directory, "tls_a.dll")
            # A foreign page that release must not read: any access to it faults the child.
            guard = kernel.dll.VirtualAlloc(None, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS)
            # A foreign page whose content and state release must not change.
            buffer = kernel.dll.VirtualAlloc(None, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
            if not guard or not buffer:
                raise AssertionError("VirtualAlloc failed")
            ctypes.memset(buffer, 0xA5, 4096)
            try:
                for label, address in (("null", None), ("interior", a + 0x10), ("base + 1", a + 1),
                                       ("unreadable foreign page", guard), ("foreign buffer", buffer)):
                    expect(f"release of {label}", mapper.release(address), 0)
                # An interior address on a page made unreadable: release must not read it either.
                interior = a + quiet
                previous = kernel.protect(interior, PAGE_NOACCESS)
                try:
                    expect("release of an unreadable interior page", mapper.release(interior), 0)
                finally:
                    kernel.protect(interior, previous)
                expect("process detach after unowned releases", process_detaches(sink, a), [])
                expect("foreign buffer", ctypes.string_at(buffer, 4096), bytes([0xA5]) * 4096)
                expect("foreign buffer page", kernel.state(buffer), (MEM_COMMIT, PAGE_READWRITE))
                expect("unwind entry before release", bool(kernel.unwind_entry(a + entry)), True)
                expect("release of the base", mapper.release(a), 1)
                # Without a notification in flight, teardown finishes before release returns.
                expect("process detach after release", process_detaches(sink, a), a_release)
                expect("unwind entry after release", kernel.unwind_entry(a + entry), None)
                expect("released base is no longer committed", kernel.state(a)[0] != MEM_COMMIT, True)
                expect("second release of the base", mapper.release(a), 0)
            finally:
                kernel.dll.VirtualFree(guard, 0, MEM_RELEASE)
                kernel.dll.VirtualFree(buffer, 0, MEM_RELEASE)
            return

        if case == "concurrent-release":
            # Native threads spin on one flag and then release together, without the GIL, and the
            # sink returns only after every racer has exited, thread detach included.
            sink.sink_race_release.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_int)]
            sink.sink_race_release.restype = ctypes.c_int
            release = ctypes.cast(mapper.release, ctypes.c_void_p)
            previous_base = None
            for round_index in range(RACE_ROUNDS):
                sink.sink_clear()  # bases repeat between rounds, so start each round with an empty log
                a = map_copy(directory, f"tls_race_{round_index}.dll")
                results = (ctypes.c_int * RACE_THREADS)()
                expect(f"round {round_index} racers", sink.sink_race_release(release, a, RACE_THREADS, results),
                       RACE_THREADS)
                expect(f"round {round_index} results", sorted(results), [0] * (RACE_THREADS - 1) + [1])
                expect(f"round {round_index} process detach", process_detaches(sink, a), a_release)
                expect(f"round {round_index} unwind entry", kernel.unwind_entry(a + entry), None)
                previous_base = a
            # Addresses are the only identity: a reused address releases the mapping now there.
            sink.sink_clear()
            reused = map_copy(directory, "tls_reuse.dll")
            # pe_map lets VirtualAlloc choose the base, so the new mapping may or may not reuse it.
            if reused == previous_base:
                expect("stale pointer to a reused address", mapper.release(previous_base), 1)
                print("the new mapping reused the address; the stale pointer released it")
            else:
                expect("stale pointer to a freed address", mapper.release(previous_base), 0)
                expect("process detach after the stale release", process_detaches(sink, reused), [])
                expect("release of the new mapping", mapper.release(reused), 1)
                print("the new mapping took another address; the stale pointer released nothing")
            expect("process detach of the new mapping", process_detaches(sink, reused), a_release)
            print(f"{RACE_ROUNDS} rounds of {RACE_THREADS} native concurrent releases: one winner each")
            return

        # release-during-notification-on-another-thread: a worker's TLS callback blocks inside its
        # thread-attach notification while this thread releases the image.
        a = map_copy(directory, "tls_a.dll")
        entered, proceed = threading.Event(), threading.Event()

        @HOOK
        def block(_argument):
            entered.set()
            proceed.wait(CHILD_TIMEOUT / 2)

        observed = []
        arm = lambda thread: sink.sink_on_thread_attach(ctypes.cast(block, ctypes.c_void_p), None, thread)
        worker = threading.Thread(target=lambda: observed.extend(new_thread_calls(sink, a, arm)))
        worker.start()
        try:
            if not entered.wait(CHILD_TIMEOUT / 2):
                raise AssertionError("worker never reached the TLS callback")
            # The admitted notification holds a reference: release unregisters and returns at once.
            expect("release while a notification is in flight", mapper.release(a), 1)
            expect("second release while in flight", mapper.release(a), 0)
            expect("process detach while in flight", process_detaches(sink, a), [])
            expect("unwind entry while in flight", bool(kernel.unwind_entry(a + entry)), True)
        finally:
            proceed.set()  # never leave the hook blocked under the loader lock
        worker.join(CHILD_TIMEOUT / 2)
        if worker.is_alive():
            raise AssertionError("worker did not finish")
        # The admitted notification finishes with its entry-point call, then its reference performs
        # the teardown once, on the worker. The thread detach that follows is not admitted.
        expect("worker thread", observed, notification(THREAD_ATTACH, "module") + a_release)
        expect("process detach after the worker", process_detaches(sink, a), a_release)
        expect("unwind entry after the worker", kernel.unwind_entry(a + entry), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=CASES)
    for name in ("mapper", "image", "sink"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    dlls = [path.resolve() for path in (args.mapper, args.image, args.sink)]
    if args.run:
        run(args.run, *dlls)
        return 0

    for case in CASES:
        command = [sys.executable, "-B", __file__, "--run", case, *map(str, dlls)]
        try:
            child = subprocess.run(command, capture_output=True, text=True, timeout=CHILD_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"{case}: FAIL (child timed out after {CHILD_TIMEOUT}s)", flush=True)
            return 1
        if child.returncode:
            detail = " ".join((child.stderr or child.stdout).split())
            print(f"{case}: FAIL (child exited {child.returncode:#x}) {detail}".rstrip(), flush=True)
            return 1
        print(f"{case}: PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
