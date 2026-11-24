"""Compare post-query counter writes against signed differences modulo 2**64."""
import ctypes as C
from pathlib import Path
import sys


class Input(C.Structure):
    _fields_ = [("before", C.c_uint64), ("after", C.c_uint64),
                ("status", C.c_uint32), ("secondary", C.c_uint32)]


def check(image: Path) -> None:
    lib = C.CDLL(str(image.resolve()))
    mask = (1 << 64) - 1
    for name in ("create_delta", "release_delta"):
        update = getattr(lib, name)
        update.argtypes = [C.POINTER(Input), C.POINTER(C.c_uint64)]
        update.restype = None
        for before, after in ((23, 70), (70, 23), (0, 0), (mask - 2, 7)):
            for status in (0, 1, 0x80000000):
                for flag in (0, 1, 256):
                    counters = (C.c_uint64 * 2)(11, 101)
                    update(C.byref(Input(before, after, status, flag)), counters)
                    delta = (after if status else 0) - before
                    enabled = flag if name == "create_delta" else flag & 255
                    expected = [(11 + delta) & mask, (101 + (delta if enabled else 0)) & mask]
                    assert list(counters) == expected, (name, before, after, status, flag, list(counters))


if __name__ == "__main__":
    for image in sys.argv[1:]:
        check(Path(image))
