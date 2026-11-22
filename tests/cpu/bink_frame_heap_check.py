"""Exercise frame allocation from the outer caller through the named Windows imports."""
import ctypes as C
from pathlib import Path
import sys

import pefile

from bink_frame_planes_check import Frames
from bink_heap_check import ALLOC, ATTRIBUTES, FREE, PTR, WORD


class Context(C.Structure):
    _fields_ = [("before_account", C.c_byte * 0xe0), ("accounted", WORD),
                ("before_format", C.c_byte * (0x118 - 0xe8)), ("format", C.c_uint32),
                ("before_owner", C.c_byte * (0x6a8 - 0x11c)), ("owner", WORD)]


def load_driver(path, provider):
    with pefile.PE(str(path), fast_load=True) as pe:
        pe.parse_data_directories(directories=[1])
        names = {i.name for d in pe.DIRECTORY_ENTRY_IMPORT if d.dll.lower() == provider
                 for i in d.imports}
        assert {b"XMemAlloc", b"XMemFree"} <= names, (path, names)
    lib = C.CDLL(str(path))
    lib.frame_allocate.argtypes = [C.POINTER(Context), C.POINTER(Frames), C.c_uint32]
    lib.frame_allocate.restype = C.c_uint32
    lib.heap_release.argtypes, lib.heap_release.restype = [PTR], None
    return lib


class Recorder:
    def __init__(self, xmem):
        self.xmem = xmem
        self.calls = []
        self.fail = False
        self.alloc_hook = ALLOC(self.allocate)
        self.free_hook = FREE(self.release)

    def allocate(self, size, flags):
        base = None if self.fail else self.xmem.XMemAlloc(size, flags)
        self.calls.append(("alloc", size, flags, base))
        return base

    def release(self, base, flags):
        self.calls.append(("free", base, flags))
        self.xmem.XMemFree(base, flags)


def scenario(lib, recorder, masks, aligned, alignment, fail=False):
    frames = Frames(len(masks), 79, 7, 29, 3)
    for i, planes in enumerate(frames.planes):
        for p, plane in enumerate(planes):
            plane.enabled = int(i < len(masks) and masks[i] & (1 << p) != 0)
            plane.destination = 0xf001 + 5*i + p
            plane.pitch = 0xabcd
    context = Context()
    context.format = int.from_bytes(b"KB2f", "little") if aligned else 0
    context.accounted = 17
    expected = Frames.from_buffer_copy(frames)
    expected.count = min(2, len(masks))
    total = 0
    payloads = []
    for i, planes in enumerate(expected.planes):
        for p, plane in enumerate(planes):
            if i >= expected.count:
                if p < 3:
                    plane.destination = 0
                continue
            if not plane.enabled:
                continue
            small = p in (1, 2)
            width = frames.small_width if small else frames.width
            quantum = max(64, alignment) if aligned else (16 if small else 1)
            plane.pitch = (width + quantum - 1) // quantum * quantum
            size = plane.pitch * (frames.small_height if small else frames.height)
            payloads.append((i, p, 288 + total, size))
            # Smallest 32-byte unit count with the descriptor's required residue.
            units = (size + 31) // 32
            units += (total // 32 + 1 - units) % 32
            total += 32 * units

    accounted = 264 + total if payloads else 0
    # A failed attempt leaves the selected pitches and accounting visible.
    # Retry the same context and table to verify that they are still usable.
    for failing in ([True, False] if fail else [False]):
        if recorder:
            recorder.fail = failing
            recorder.calls.clear()
        initial = Context.from_buffer_copy(context)
        result = lib.frame_allocate(C.byref(context), C.byref(frames), alignment)
        assert result == int(not failing)
        initial.accounted += accounted
        if payloads and not failing:
            assert context.owner and context.owner % 32 == 0
            initial.owner = context.owner
        assert bytes(initial) == bytes(context)
        if recorder:
            if payloads:
                raw = None if failing else context.owner - 64  # shared heap base aligned to 32
                assert recorder.calls == [("alloc", 352 + total, ATTRIBUTES, raw)], recorder.calls
            else:
                assert not recorder.calls
        if context.owner:
            for i, p, offset, size in payloads:
                expected.planes[i][p].destination = context.owner + offset
        assert bytes(frames) == bytes(expected)
        if context.owner:
            try:
                assert C.string_at(context.owner, 288 + total) == bytes(288 + total)
                memory = bytearray(288 + total)
                for i, p, offset, size in payloads:
                    color = 5*i + p + 1
                    C.memset(frames.planes[i][p].destination, color, size)
                    memory[offset:offset+size] = bytes([color]) * size
                assert C.string_at(context.owner, len(memory)) == memory
                # The existing owner guard rejects a second allocation before any store.
                assert lib.frame_allocate(C.byref(context), C.byref(frames), alignment) == 0
                assert bytes(initial) == bytes(context) and bytes(frames) == bytes(expected)
            finally:
                lib.heap_release(context.owner)
            if recorder:
                assert recorder.calls[1:] == [("free", context.owner - 64, ATTRIBUTES)]


def main(direct, observed, xmem_path, provider_path):
    xmem = C.CDLL(str(xmem_path))
    xmem.XMemAlloc.argtypes, xmem.XMemAlloc.restype = [WORD, WORD], PTR
    xmem.XMemFree.argtypes, xmem.XMemFree.restype = [PTR, WORD], None
    provider = C.CDLL(str(provider_path))
    provider.bind_heap_observers.argtypes, provider.bind_heap_observers.restype = [ALLOC, FREE], None
    recorder = Recorder(xmem)
    for path, module, trace in ((direct, b"xmem.dll", None),
                                (observed, b"cpu_bink_heap_provider.dll", recorder)):
        lib = load_driver(path, module)
        if trace:
            provider.bind_heap_observers(trace.alloc_hook, trace.free_hook)
        try:
            scenario(lib, trace, [], False, 32)
            scenario(lib, trace, [0, 0], True, 128)
            for mask in range(32):
                scenario(lib, trace, [mask], False, 16)
            scenario(lib, trace, [31, 21, 31], True, 32, fail=bool(trace))
            scenario(lib, trace, [9, 31], True, 128)
        finally:
            if trace:
                provider.bind_heap_observers(ALLOC(), FREE())
        print(f"{path.name}: 36 frame cases passed; injected failure and retry={bool(trace)}")


if __name__ == "__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
