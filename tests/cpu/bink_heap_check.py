"""Check the native control and offline translations against a real Windows process heap."""
import ctypes as C
from pathlib import Path
import sys


WORD = C.c_uint64
PTR = C.c_void_p
ALLOC = C.CFUNCTYPE(PTR, WORD, WORD)
FREE = C.CFUNCTYPE(None, PTR, WORD)
ATTRIBUTES = 0x05e50000
GUARD = 0xad


class HeapTrace:
    def __init__(self, provider):
        self.provider = provider
        self.calls = []
        self.live = {}
        self.residue = 0
        self.fail = False
        self.allocate_hook = ALLOC(self.allocate)
        self.release_hook = FREE(self.release)

    def allocate(self, size, flags):
        # Vary the address handed to the caller while retaining two guard zones
        # in a real process-heap allocation. Cap unexpected requests in the probe.
        raw = None
        base = None
        if not self.fail and size <= 8192:
            raw = self.provider.windows_allocate(size + 96, flags)
        if raw:
            C.memset(raw, GUARD, size + 96)
            base = raw + 32 + ((self.residue - raw) % 32)
            self.live[base] = (raw, size)
        self.calls.append(("alloc", size, flags, base))
        return base

    def release(self, base, flags):
        record = self.live.get(base)
        freed = bool(record and self.provider.windows_release(record[0], flags))
        if freed:
            del self.live[base]
        self.calls.append(("free", base, flags, freed))


def check(path, provider):
    lib = C.CDLL(str(path.resolve()))
    lib.heap_request.argtypes, lib.heap_request.restype = [WORD], PTR
    lib.heap_release.argtypes, lib.heap_release.restype = [PTR], None

    # Also exercise the desktop binding directly, with no interposed hooks.
    pointer = lib.heap_request(33)
    assert pointer and pointer % 32 == 0
    C.memset(pointer, 0x47, 33)
    lib.heap_release(pointer)

    trace = HeapTrace(provider)
    provider.bind_heap_observers(trace.allocate_hook, trace.release_hook)
    try:
        for size in (0, 0xffffffff):
            assert lib.heap_request(size) is None
        lib.heap_release(None)
        assert trace.calls == [], trace.calls

        trace.fail = True
        for size in (1, 0xfffffffe, 0x100000001):
            assert lib.heap_request(size) is None
            lib.heap_release(None)
            assert trace.calls == [("alloc", size + 64, ATTRIBUTES, None)], trace.calls
            trace.calls.clear()
        trace.fail = False

        # Keep all 32 entries live, then release in reverse allocation order.
        pointers = []
        sizes = (1, 15, 31, 32, 33, 64, 257, 4096)
        for residue in range(32):
            trace.residue = residue
            size = sizes[residue % len(sizes)]
            pointer = lib.heap_request(size)
            assert len(trace.calls) == 1, trace.calls
            base = trace.calls[0][-1]
            assert base and base % 32 == residue
            assert trace.calls == [("alloc", size + 64, ATTRIBUTES, base)]
            trace.calls.clear()
            offset = 64 - residue
            assert pointer == base + offset and pointer % 32 == 0
            raw, granted = trace.live[base]
            C.memset(pointer, residue + 1, size)
            expected = bytearray([GUARD]) * (granted + 96)
            start = pointer - raw
            expected[start-2:start] = bytes((0, offset))
            expected[start:start+size] = bytes([residue + 1]) * size
            assert C.string_at(raw, len(expected)) == expected, (size, residue)
            pointers.append((pointer, base))
        for pointer, base in reversed(pointers):
            lib.heap_release(pointer)
            assert trace.calls == [("free", base, ATTRIBUTES, True)], trace.calls
            trace.calls.clear()
        assert not trace.live
    finally:
        provider.bind_heap_observers(ALLOC(), FREE())
        for raw, _ in trace.live.values():
            provider.windows_release(raw, ATTRIBUTES)
    print(f"{path.name}: 32 base residues, 32 matching releases, 3 allocator failures, "
          "2 early rejects and null release passed")


def main(provider_path, *images):
    provider = C.CDLL(str(Path(provider_path).resolve()))
    provider.windows_allocate.argtypes, provider.windows_allocate.restype = [WORD, WORD], PTR
    provider.windows_release.argtypes, provider.windows_release.restype = [PTR, WORD], C.c_bool
    provider.bind_heap_observers.argtypes, provider.bind_heap_observers.restype = [ALLOC, FREE], None
    for path in images:
        check(Path(path), provider)


if __name__ == "__main__":
    main(*sys.argv[1:])
