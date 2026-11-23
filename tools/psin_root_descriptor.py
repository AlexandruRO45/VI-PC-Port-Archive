"""Read retail-style PSIN static root descriptors from a local AMD64 PE."""
import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_inspect import PeImage, PeFormatError


def unpack(image, fmt, rva):
    offset = image.offset(rva, struct.calcsize(fmt))
    if offset is None:
        raise PeFormatError(f"unmapped descriptor data at RVA {rva:#x}")
    return struct.unpack_from(fmt, image.data, offset)


def descriptor(image, rva):
    if not image.is64 or image.machine != 0x8664:
        raise PeFormatError("expected AMD64 PE32+")
    ptr, root, size, flags = unpack(image, "<QIIH", rva + 8)
    metadata = ptr - image.image_base
    count, = unpack(image, "<H", metadata + 1)
    fields = []
    for i in range(count):
        delta, = unpack(image, "<H", metadata + 3 + 2 * i)
        identifier, kind = unpack(image, "<IB", metadata + delta + 0x18)
        offset, = unpack(image, "<I", rva + 0x1a + 4 * i)
        fields.append((offset, kind, identifier))
    return root, size, 1 << ((flags >> 7) & 7), fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("rva", type=lambda value: int(value, 0), nargs="+")
    args = parser.parse_args()
    try:
        image = PeImage(args.image.read_bytes())
        for rva in args.rva:
            root, size, alignment, fields = descriptor(image, rva)
            print(f"root={root:#x} size={size} alignment={alignment} members={len(fields)}")
            for offset, kind, identifier in fields:
                print(f"  offset={offset:#x} kind={kind:#x} member_id={identifier:#010x}")
    except (OSError, PeFormatError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
