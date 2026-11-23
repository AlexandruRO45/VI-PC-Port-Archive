"""List AMD64 PE exception-directory ranges in ascending RVA order as CSV."""
import argparse
import csv
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_inspect import DIR_EXCEPTION, PeFormatError, PeImage


def inventory(image):
    if image.machine != 0x8664 or not image.is64:
        raise PeFormatError("expected an AMD64 PE32+ image")
    rva, size = image.directory(DIR_EXCEPTION)
    if (rva, size) == (0, 0):
        return []
    start = image.offset(rva, size)
    if not rva or start is None or size % 12:
        raise PeFormatError("invalid exception directory")
    rows = []
    for begin, end, unwind in struct.iter_unpack("<III", memoryview(image.data)[start:start + size]):
        section = next((s for s in image.sections if s["rva"] <= begin < end <=
                        s["rva"] + (s["vsize"] or s["rawsize"])), None)
        header = image.offset(unwind, 4)
        if section is None or not unwind or unwind % 4 or header is None:
            raise PeFormatError(f"invalid runtime function at {begin:#x}")
        flags = image.data[header] >> 3
        rows.append((begin, end, section["name"], end - begin, unwind, flags))
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    try:
        rows = inventory(PeImage(args.image.read_bytes()))
    except (PeFormatError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("begin_rva", "end_rva", "section", "size", "unwind_rva", "unwind_flags"))
    for begin, end, section, size, unwind, flags in rows:
        writer.writerow((hex(begin), hex(end), section, size, hex(unwind), flags))


if __name__ == "__main__":
    main()
