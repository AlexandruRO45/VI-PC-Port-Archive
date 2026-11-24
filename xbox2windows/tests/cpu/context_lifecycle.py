"""Compare create-to-destroy ownership using real CRT storage and a controlled provider."""
import ctypes as C
from pathlib import Path
import sys

from destroy_check import Binding, Dealloc, Destroy, Free, Ptr, Word

Create = C.CFUNCTYPE(C.c_uint32, C.POINTER(Ptr), Ptr, Ptr)
Allocate = C.CFUNCTYPE(Ptr, Ptr, Word)
Calloc = C.CFUNCTYPE(Ptr, Word, Word)


class Allocator(C.Structure):
    _fields_ = [("user", Ptr), ("allocate", Allocate), ("deallocate", Dealloc)]


class Provider(C.Structure):
    _fields_ = [("create", Create), ("destroy", Destroy)]


class Host(C.Structure):
    _fields_ = [("table", C.POINTER(Provider)), ("zero_alloc", Calloc)]


def run_case(create, destroy, crt, *, custom, create_status=0, destroy_status=0, oom=False):
    events, errors, live = [], [], {}
    allocated = {}
    desc = (Word * 2)(0x10007, 0)
    user = C.addressof(desc) + 8
    expected_user = user if custom else None
    inner_size, wrapper_size = 64, 16

    def allocate(sender, size):
        events.append(("alloc", size))
        if sender != expected_user or size not in (inner_size, wrapper_size):
            errors.append(("allocation arguments", sender, size))
            return None
        if oom and size == wrapper_size:
            return None
        p = crt.calloc(1, size)
        if not p:
            errors.append("unexpected CRT allocation failure")
            return None
        allocated[size] = p
        live[p] = size
        return p

    def zero_alloc(count, size):
        if count != 1:
            errors.append(("calloc count", count))
        return allocate(None, size)

    def release(sender, p):
        size = live.pop(p, None)
        events.append(("free", size))
        if size is None or sender != expected_user:
            errors.append(("release arguments", sender, p))
        if size == wrapper_size and (Ptr * 2).from_address(p)[1] is not None:
            errors.append("wrapper released before provider cleared its field")
        if size is not None:
            crt.free(p)

    allocator = Allocator(user, Allocate(allocate), Dealloc(release))
    alloc_address = C.addressof(allocator) if custom else None
    default_alloc = Calloc(zero_alloc)
    default_free = Free(lambda p: release(None, p))

    @Create
    def provider_create(out, descriptor, callbacks):
        events.append(("create", create_status))
        if out[0] or descriptor != C.addressof(desc) or callbacks != alloc_address:
            errors.append("provider create arguments")
            return 6
        if not create_status:
            if callbacks:
                cb = Allocator.from_address(callbacks)
                out[0] = cb.allocate(cb.user, inner_size)
            else:
                out[0] = default_alloc(1, inner_size)
        return create_status

    @Destroy
    def provider_destroy(slot, callbacks):
        events.append(("destroy", destroy_status))
        if callbacks != alloc_address or slot[0] != allocated.get(inner_size):
            errors.append("provider destroy arguments")
            return 6
        if not oom and C.addressof(slot.contents) != allocated[wrapper_size] + 8:
            errors.append("destroy received a copy of the inner field")
        if callbacks:
            cb = Allocator.from_address(callbacks)
            cb.deallocate(cb.user, slot[0])
        else:
            default_free(slot[0])
        slot[0] = None
        return destroy_status

    table = Provider(provider_create, provider_destroy)
    host = Host(C.pointer(table), default_alloc)
    binding = Binding(C.addressof(table), provider_destroy, alloc_address,
                      allocator.deallocate, default_free)
    handle = Ptr(0x1234)
    try:
        actual = create(C.byref(handle), C.addressof(desc), alloc_address, C.byref(host))
        assert actual == (create_status or (5 if oom else 0)), (actual, events)
        expected = [("create", create_status)]
        if create_status:
            assert handle.value is None
        else:
            expected += [("alloc", inner_size), ("alloc", wrapper_size)]
            if oom:
                assert handle.value is None
            else:
                assert handle.value == allocated[wrapper_size]
                assert tuple((Ptr * 2).from_address(handle.value)) == (C.addressof(table), allocated[inner_size])
                status = Word(0x5555555555555555)
                assert destroy(C.byref(handle), alloc_address, C.byref(status), C.byref(binding)) == 1
                assert status.value == destroy_status
            expected += [("destroy", destroy_status), ("free", inner_size)]
            if not oom:
                expected.append(("free", wrapper_size))
        assert events == expected and not live and not errors, (events, expected, live, errors)
    finally:
        for p in live:
            crt.free(p)


def check(native_path, lifted_path, destroy_path):
    crt = C.CDLL("ucrtbase")
    crt.calloc.argtypes, crt.calloc.restype = [Word, Word], Ptr
    crt.free.argtypes, crt.free.restype = [Ptr], None
    destroy = C.CDLL(str(destroy_path.resolve())).try_destroy
    destroy.argtypes, destroy.restype = [C.POINTER(Ptr), Ptr, C.POINTER(Word), C.POINTER(Binding)], Word
    for path, symbol in ((native_path, "fixture_create"), (lifted_path, "create_context")):
        create = getattr(C.CDLL(str(path.resolve())), symbol)
        create.argtypes, create.restype = [C.POINTER(Ptr), Ptr, Ptr, C.POINTER(Host)], C.c_uint32
        for custom in (False, True):
            for code in (0, 0x80000001, 0xffffffff):
                run_case(create, destroy, crt, custom=custom, destroy_status=code)
            for code in (6, 0x80000001, 0xffffffff):
                run_case(create, destroy, crt, custom=custom, create_status=code)
            run_case(create, destroy, crt, custom=custom, oom=True)
        print(f"{path.name}: 14 lifecycle cases matched status, pointers, allocation order and zero live allocations")


if __name__ == "__main__":
    check(*map(Path, sys.argv[1:]))
