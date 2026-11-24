"""Check the native control and offline translations against a real Windows process heap."""
import ctypes as C
from pathlib import Path
import sys


WORD = C.c_uint64
PTR = C.c_void_p
ALLOC = C.CFUNCTYPE(PTR, WORD, WORD)
FREE = C.CFUNCTYPE(None, PTR, WORD)
CUSTOM_ALLOC = C.CFUNCTYPE(PTR, WORD)
CUSTOM_FREE = C.CFUNCTYPE(None, PTR)
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


class CallbackPair:
    def __init__(self, provider, residue):
        self.heap = HeapTrace(provider)
        self.heap.residue = residue
        # The custom callbacks take one argument. The test uses Windows storage
        # behind them and keeps the callback objects alive across replacements.
        self.allocate = CUSTOM_ALLOC(lambda size: self.heap.allocate(size, ATTRIBUTES))
        self.release = CUSTOM_FREE(lambda base: self.heap.release(base, ATTRIBUTES))


def check_block(pointer, size, heap, release=None):
    base, = heap.live
    assert heap.calls == [("alloc", size + 64, ATTRIBUTES, base)], heap.calls
    assert base % 32 == heap.residue
    offset = 64 - heap.residue
    assert pointer == base + offset and pointer % 32 == 0
    raw, granted = heap.live[base]
    start = pointer - raw
    expected = bytearray([GUARD]) * (granted + 96)
    expected[start-2:start] = bytes((3 if release else 0, offset))
    if release:
        address = C.cast(release, PTR).value
        expected[start-16:start-8] = address.to_bytes(8, "little")
    C.memset(pointer, 0x59, size)
    expected[start:start+size] = bytes([0x59]) * size
    assert C.string_at(raw, len(expected)) == expected
    return base


def check_custom(lib, provider, fallback):
    a = CallbackPair(provider, 7)
    b = CallbackPair(provider, 19)
    lib.BinkSetMemory.argtypes, lib.BinkSetMemory.restype = [CUSTOM_ALLOC, CUSTOM_FREE], None
    try:
        lib.BinkSetMemory(a.allocate, a.release)
        old = lib.heap_request(33)
        old_base = check_block(old, 33, a.heap, a.release)
        lib.BinkSetMemory(b.allocate, b.release)
        new = lib.heap_request(129)
        new_base = check_block(new, 129, b.heap, b.release)
        lib.heap_release(old)
        assert a.heap.calls == [("alloc", 97, ATTRIBUTES, old_base),
                                ("free", old_base, ATTRIBUTES, True)]
        assert b.heap.calls == [("alloc", 193, ATTRIBUTES, new_base)]
        lib.BinkSetMemory(CUSTOM_ALLOC(), CUSTOM_FREE())
        lib.heap_release(new)
        assert b.heap.calls == [("alloc", 193, ATTRIBUTES, new_base),
                                ("free", new_base, ATTRIBUTES, True)]
        assert not a.heap.live and not b.heap.live and fallback.calls == []

        # Keep the relative call order at the custom-to-default boundary.
        attempts = []
        def decline(size):
            attempts.append((size, len(fallback.calls)))
            return None
        decline_hook = CUSTOM_ALLOC(decline)
        lib.BinkSetMemory(decline_hook, a.release)
        pointer = lib.heap_request(65)
        base = check_block(pointer, 65, fallback)
        lib.BinkSetMemory(b.allocate, b.release)
        lib.heap_release(pointer)
        assert fallback.calls == [("alloc", 129, ATTRIBUTES, base),
                                  ("free", base, ATTRIBUTES, True)]
        assert len(a.heap.calls) == len(b.heap.calls) == 2

        lib.BinkSetMemory(decline_hook, a.release)
        fallback.fail = True
        assert lib.heap_request(0x100000001) is None
        assert fallback.calls[2:] == [("alloc", 0x100000041, ATTRIBUTES, None)]
        assert attempts == [(129, 0), (0x100000041, 2)]
        # The other failure return from the custom allocator ends the request.
        refused = []
        def refuse(size):
            refused.append(size)
            return WORD(-1).value
        refuse_hook = CUSTOM_ALLOC(refuse)
        lib.BinkSetMemory(refuse_hook, b.release)
        assert lib.heap_request(9) is None
        lib.heap_release(None)
        assert refused == [73] and len(fallback.calls) == 3
        assert len(a.heap.calls) == len(b.heap.calls) == 2
    finally:
        lib.BinkSetMemory(CUSTOM_ALLOC(), CUSTOM_FREE())
        for heap in (a.heap, b.heap):
            for raw, _ in heap.live.values():
                provider.windows_release(raw, ATTRIBUTES)


def check(path, provider):
    lib = C.CDLL(str(path.resolve()))
    lib.heap_request.argtypes, lib.heap_request.restype = [WORD], PTR
    lib.heap_release.argtypes, lib.heap_release.restype = [PTR], None

    # Also exercise the desktop binding directly, with no interposed hooks.
    pointer = lib.heap_request(33)
    assert pointer and pointer % 32 == 0
    C.memset(pointer, 0x47, 33)
    lib.heap_release(pointer)

    # The binding serves the request from the shared ERA heap: the alignment exponent in the word
    # selects 32 bytes and the block comes back zeroed.
    for size in (97, 129):
        raw = provider.windows_allocate(size, ATTRIBUTES)
        assert raw and raw % 32 == 0
        assert C.string_at(raw, size) == bytes(size)
        assert provider.windows_release(raw, ATTRIBUTES)

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
        check_custom(lib, provider, trace)
    finally:
        provider.bind_heap_observers(ALLOC(), FREE())
        for raw, _ in trace.live.values():
            provider.windows_release(raw, ATTRIBUTES)
    print(f"{path.name}: 32 base residues, 32 matching releases, 3 allocator failures, "
          "2 early rejects and null release passed; custom A/B ownership, null fallback, "
          "double failure and custom refusal passed")


def main(provider_path, *images):
    provider = C.CDLL(str(Path(provider_path).resolve()))
    provider.windows_allocate.argtypes, provider.windows_allocate.restype = [WORD, WORD], PTR
    provider.windows_release.argtypes, provider.windows_release.restype = [PTR, WORD], C.c_bool
    provider.bind_heap_observers.argtypes, provider.bind_heap_observers.restype = [ALLOC, FREE], None
    for path in images:
        check(Path(path), provider)


if __name__ == "__main__":
    main(*sys.argv[1:])
