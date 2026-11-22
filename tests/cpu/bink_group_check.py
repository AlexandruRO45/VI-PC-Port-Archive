"""Compare grouped Bink requests and pointer placement using owned Windows buffers."""
import ctypes as C
from pathlib import Path
import sys

from bink_heap_check import ALLOC, ATTRIBUTES, FREE, HeapTrace, PTR, WORD


class Group(C.Structure):
    _fields_ = [("unused0", WORD), ("destinations", C.POINTER(WORD)),
                ("unused1", WORD), ("lengths", C.POINTER(WORD)),
                ("extra", WORD), ("count", C.c_uint32)]


def check(image: Path, provider: C.CDLL) -> None:
    lib = C.CDLL(str(image.resolve()))
    lib.heap_group.argtypes, lib.heap_group.restype = [C.POINTER(Group), WORD], PTR
    lib.heap_release.argtypes, lib.heap_release.restype = [PTR], None
    trace = HeapTrace(provider)
    provider.bind_heap_observers(trace.allocate_hook, trace.release_hook)
    try:
        for size, aligned in ((0, 0), (31, 32), (32, 32), (33, 64)):
            pointer = lib.heap_group(None, size)
            if aligned:
                assert pointer and trace.calls == [("alloc", aligned + 64, ATTRIBUTES, next(iter(trace.live)))]
                lib.heap_release(pointer)
            else:
                assert pointer is None and not trace.calls
            assert not trace.live
            trace.calls.clear()

        empty = Group()
        pointer = lib.heap_group(C.byref(empty), 1)
        assert pointer and trace.calls[0][:3] == ("alloc", 96, ATTRIBUTES)
        lib.heap_release(pointer)
        assert not trace.live
        trace.calls.clear()

        for fail in (True, False):
            external = WORD(0x1234)
            destinations = (WORD * 2)(8, C.addressof(external))
            lengths = (WORD * 2)(17, 32)
            group = Group(0, destinations, 0, lengths, 48, 2)
            trace.fail = fail
            pointer = lib.heap_group(C.byref(group), 33)
            assert group.extra == 0 and group.count == 0
            assert len(trace.calls) == 1 and trace.calls[0][:3] == ("alloc", 176, ATTRIBUTES)
            if fail:
                assert pointer is None and external.value == 0x1234
            else:
                assert pointer
                assert WORD.from_address(pointer + 8).value == pointer + 64
                assert external.value == pointer + 80
                lib.heap_release(pointer)
            assert not trace.live
            trace.calls.clear()
    finally:
        provider.bind_heap_observers(ALLOC(), FREE())
        for raw, _ in trace.live.values():
            provider.windows_release(raw, ATTRIBUTES)


if __name__ == "__main__":
    provider = C.CDLL(str(Path(sys.argv[1]).resolve()))
    provider.windows_allocate.argtypes, provider.windows_allocate.restype = [WORD, WORD], PTR
    provider.windows_release.argtypes, provider.windows_release.restype = [PTR, WORD], C.c_bool
    provider.bind_heap_observers.argtypes = [ALLOC, FREE]
    for image in sys.argv[2:]:
        check(Path(image), provider)
