"""Exercise the ctypes version-query ABI on the source fixture or a local package lift."""
import ctypes as C
from pathlib import Path
import sys

Word = C.c_uint64
Ptr = C.c_void_p
GUARD = 0x57201db83511f4c3


class Query(C.Structure):
    _fields_ = [("type", Word), ("next", Ptr), ("effect", Word), ("device", Ptr),
                ("count", Ptr), ("ids", Ptr), ("names", Ptr)]


def check(dll, export, expected_id, expected_name):
    fn = getattr(C.CDLL(str(dll.resolve())), export)
    fn.argtypes = [C.POINTER(Query)]
    fn.restype = C.c_uint32
    assert fn(None) == 6

    def query(capacity, ids, names, effect=0x10000, count=True, device=None):
        n = Word(capacity)
        id_buffer = (Word * 4)(*[GUARD]*4)
        name_buffer = (Word * 4)(*[GUARD]*4)
        q = Query(4, None, effect, device, C.addressof(n) if count else None,
                  C.addressof(id_buffer)+8 if ids else None,
                  C.addressof(name_buffer)+8 if names else None)
        status = fn(C.byref(q))
        # The loader clears the descriptor's count pointer on a routing error.
        if effect != 0x10000 and effect != 0x10007:
            assert (status, q.count, n.value) == (4 if effect == 0x20000 else 2, None, capacity)
        else:
            assert status == 0 and q.count == (C.addressof(n) if count else None)
            assert n.value == (1 if count else capacity)
        fill = status == 0 and count and capacity > 0
        assert list(id_buffer) == [GUARD, expected_id if fill and ids else GUARD, GUARD, GUARD]
        assert name_buffer[0] == name_buffer[2] == name_buffer[3] == GUARD
        if fill and names:
            assert name_buffer[1] not in (0, GUARD)
            name = C.string_at(name_buffer[1]).decode("ascii")
            assert name == expected_name
        else:
            assert name_buffer[1] == GUARD
            name = None
        return n.value, id_buffer[1], name

    count, _, _ = query(0, False, False)
    _, version_id, version_name = query(count, True, True)
    for capacity in (0, 1, 2):
        for ids, names in ((True, False), (False, True), (True, True), (False, False)):
            query(capacity, ids, names)
    query(2, True, True, count=False)
    query(1, True, True, effect=0x10007, device=1)
    query(2, True, True, effect=0x20000)
    query(2, True, True, effect=0x90000)
    print(f"{dll.name}: count={count} id=0x{version_id:016x} name={version_name}; capacity and guards passed")


if __name__ == "__main__":
    path, symbol, value, name = sys.argv[1:]
    check(Path(path), symbol, int(value, 0), name)
