"""Windows differential check for a provider-context slot and wrapper cleanup."""
import ctypes as C
from pathlib import Path
import sys


Ptr = C.c_void_p
Word = C.c_uint64
Destroy = C.CFUNCTYPE(C.c_uint32, C.POINTER(Ptr), Ptr)
Dealloc = C.CFUNCTYPE(None, Ptr, Ptr)
Free = C.CFUNCTYPE(None, Ptr)


class Binding(C.Structure):
    _fields_ = [("table", Ptr), ("destroy", Destroy), ("allocator", Ptr),
                ("dealloc", Dealloc), ("release", Free)]


class Observation(C.Structure):
    _fields_ = [("slot", Ptr), ("allocator", Ptr), ("calls", C.c_uint32), ("status", C.c_uint32)]


def check(original, translated, provider):
    native = C.CDLL(str(original)).release_context
    native.argtypes = [C.POINTER(Ptr), Ptr]
    native.restype = C.c_uint32
    lifted = C.CDLL(str(translated)).try_destroy
    lifted.argtypes = [C.POINTER(Ptr), Ptr, C.POINTER(Word), C.POINTER(Binding)]
    lifted.restype = Word
    module = C.CDLL(str(provider))
    destroy = Destroy(("ffxDestroyContext", module))
    table = (Ptr * 2)(None, C.cast(destroy, Ptr))
    crt = C.CDLL("ucrtbase")
    crt.malloc.argtypes = [C.c_size_t]
    crt.malloc.restype = Ptr
    crt.free.argtypes = [Ptr]
    crt.free.restype = None

    for empty in (None, C.pointer(Ptr())):
        result = Word()
        assert native(empty, None) == 6
        assert lifted(empty, None, C.byref(result), None) == 1 and result.value == 6

    for custom in (False, True):
        for expected in (0, 6, 0x80000001, 0xffffffff):
            for use_lifted in (False, True):
                observation = Observation(status=expected)
                events = []

                def cleanup(user, wrapper):
                    inner = (Ptr * 2).from_address(wrapper)[1]
                    events.append((user, wrapper, observation.calls, inner))
                    crt.free(wrapper)

                custom_free = Dealloc(cleanup)
                default_free = Free(lambda p: cleanup(None, p))
                allocator = (Ptr * 3)(C.addressof(observation), None, C.cast(custom_free, Ptr))
                alloc = C.addressof(allocator) if custom else None
                binding = Binding(C.addressof(table), destroy, alloc, custom_free, default_free)
                address = crt.malloc(16)
                assert address
                handle = Ptr(address)
                wrapper = (Ptr * 2).from_address(address)
                wrapper[:] = [C.addressof(table), C.addressof(observation)]
                if use_lifted:
                    result = Word(0x1234567887654321)
                    # Refuse a mismatched binding before either provider or cleanup runs.
                    for wrong in (Binding(None, destroy, alloc, custom_free, default_free),
                                  Binding(C.addressof(table), Destroy(), alloc, custom_free, default_free),
                                  Binding(C.addressof(table), destroy, alloc, Dealloc(), Free())):
                        assert lifted(C.byref(handle), alloc, C.byref(result), C.byref(wrong)) == 0
                        assert (result.value, observation.calls, events) == (0x1234567887654321, 0, [])
                    assert lifted(C.byref(handle), alloc, C.byref(result), C.byref(binding)) == 1
                    actual = result.value
                else:
                    actual = native(C.byref(handle), alloc)
                assert actual == expected and observation.calls == 1
                assert observation.slot == address + C.sizeof(Ptr) and observation.allocator == alloc
                if custom or use_lifted:
                    expected_user = C.addressof(observation) if custom else None
                    assert events == [(expected_user, address, 1, None)]
    print("Two null exits, both cleanup routes, exact arguments and four uint32 status codes matched.")


if __name__ == "__main__":
    check(*map(Path, sys.argv[1:]))
