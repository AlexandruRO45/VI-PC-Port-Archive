"""Check the two descriptor arrays using storage owned by the test."""
import ctypes as C
from pathlib import Path


def check(image: Path, translated: bool) -> None:
    lib = C.CDLL(str(image.resolve()))
    word = C.c_uint64
    lib.group_init.argtypes = [C.c_void_p, C.c_uint32]
    lib.group_init.restype = None
    lib.group_append.argtypes = [C.c_void_p, word, word]
    lib.group_append.restype = word
    # (input size, expected stored stride) in successive append operations.
    entries = ((0, 32), (33, 64), (1, 128), (1025, 1280), (32, 512))
    capacity = len(entries)
    sentinel = 0xdeadbeefdeadbeef
    storage = (word * (8 + 2 * capacity + 1))(*([sentinel] * (9 + 2 * capacity)))
    base = C.addressof(storage)
    lib.group_init(base, capacity)
    assert storage[:4] == [base + 64, base + 64, base + 64 + 8 * capacity,
                           base + 64 + 8 * capacity]
    assert storage[4] == 0 and storage[5] == capacity << 32
    assert storage[8:] == [sentinel] * (2 * capacity + 1)
    total = 0
    destinations = []
    strides = []
    for index, (size, stride) in enumerate(entries):
        destination = 0x100000040 + 8 * index
        assert lib.group_append(base, destination, size) == 1
        total += stride
        destinations.append(destination)
        strides.append(stride)
        untouched = [sentinel] * (capacity - index - 1)
        assert storage[8:8 + capacity] == destinations + untouched
        assert storage[8 + capacity:] == strides + untouched + [sentinel]
        assert storage[4] == total and storage[5] == (capacity << 32) + index + 1

    if translated:
        full = bytes(storage)
        assert lib.group_append(base, 0, 1) == 0
        assert bytes(storage) == full


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path)
    parser.add_argument("translations", nargs="+", type=Path)
    args = parser.parse_args()
    if args.native:
        check(args.native, False)
    for image in args.translations:
        check(image, True)
