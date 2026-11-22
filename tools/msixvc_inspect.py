#!/usr/bin/env python3
"""Read-only structure report for an MSIXVC container (the Store's `.EAppx` delivery format).

This tool only *reads* the file: it dumps the header, cross-checks candidate offset/size fields against the file
size, scans for plaintext container signatures, and measures where the bytes stop looking like text. It contains no
decryption, no key handling and no licence handling: a Store package's payload is protected and this project does
not go past that line (AGENTS.md, docs/design.md).

    python -I tools/msixvc_inspect.py <package.EAppx> [--full]

`--full` scans the whole file for signatures; without it only the first 64 MiB and a few sampled windows are read.
Standard library only, no external tools.
"""
import argparse
import math
import os
import struct
import sys

SIGNATURES = [
    (b"EXPH", "EXPHX header"),
    (b"PK\x03\x04", "ZIP local file header"),
    (b"PK\x01\x02", "ZIP central directory entry"),
    (b"PK\x05\x06", "ZIP end of central directory"),
    (b"AppxBlockMap.xml", "Appx block map"),
    (b"AppxManifest.xml", "Appx manifest"),
    (b"AppxSignature.p7x", "Appx signature"),
    (b"[Content_Types].xml", "Appx content types"),
    (b"<?xml", "XML text"),
    (b"<BlockMap", "block map XML element"),
    (b"<NetworkManifest", "Xbox network manifest"),
    (b"This program cannot be run in DOS mode", "PE/DOS stub"),
    (b"AXPC", "MSIXVC chunk table"),
    (b"AXCT", "MSIXVC content table"),
]


def entropy(window):
    if not window:
        return 0.0
    counts = [0] * 256
    for byte in window:
        counts[byte] += 1
    total = len(window)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def read_at(f, offset, size):
    f.seek(offset)
    return f.read(size)


def dump_header(f, size):
    print(f"file: {size:,} bytes")
    head = read_at(f, 0, 512)
    print("\nheader, first 128 bytes:")
    for base in range(0, 128, 16):
        chunk = head[base:base + 16]
        hexed = " ".join(f"{b:02x}" for b in chunk)
        text = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  0x{base:04x}  {hexed:<47}  {text}")

    print("\nfields (candidates, cross-checked against the file size):")
    magic = head[0:4]
    print(f"  magic[4]              = {magic!r}")
    fifth = head[4:5]
    print(f"  byte at 0x04          = 0x{head[4]:02x} ({head[4]!r})")
    for offset, kind in [(0x04, "u32"), (0x08, "u32"), (0x0C, "u32"), (0x10, "u32"), (0x14, "u32"),
                         (0x18, "u32"), (0x1C, "u32"), (0x20, "u32"), (0x24, "u32"), (0x28, "u32"),
                         (0x30, "u32"), (0x38, "u32")]:
        if offset + 4 > len(head):
            continue
        (value,) = struct.unpack_from("<I", head, offset)
        note = ""
        if value and value < 0x10000:
            note = "small (version/count?)"
        elif value <= size:
            note = "offset or size within the file"
        else:
            note = "larger than the file"
        print(f"  u32 at 0x{offset:04x}        = {value:>12} (0x{value:08x})  {note}")
    for offset in (0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38):
        if offset + 8 > len(head):
            continue
        (value,) = struct.unpack_from("<Q", head, offset)
        if not value:
            continue
        note = "within the file" if value <= size else "larger than the file"
        print(f"  u64 at 0x{offset:04x}        = {value:>12} (0x{value:016x})  {note}")


def scan(f, size, limit, step):
    """Counts each signature over [0, limit) and remembers the first few offsets."""
    hits = {name: [] for _, name in SIGNATURES}
    overlap = 64
    offset = 0
    while offset < limit:
        chunk = read_at(f, offset, min(step, limit - offset) + overlap)
        for needle, name in SIGNATURES:
            start = 0
            while True:
                found = chunk.find(needle, start)
                if found < 0 or len(hits[name]) >= 8:
                    break
                absolute = offset + found
                if absolute < limit:
                    hits[name].append(absolute)
                start = found + 1
        offset += step
    return hits


def extract_plaintext(f, size, hits, directory):
    """Writes every plaintext XML document found in the package; print-only elsewhere."""
    os.makedirs(directory, exist_ok=True)
    written = []
    for offset in hits["XML text"]:
        f.seek(offset)
        data = f.read(1024 * 1024)
        # A document ends at the first byte a text file does not contain; the biggest run of text is taken.
        end = len(data)
        for index, byte in enumerate(data):
            if byte < 0x20 and byte not in (0x09, 0x0A, 0x0D):
                end = index
                break
        text = data[:end]
        if b"</" not in text or len(text) < 32:
            continue
        cut = text.rfind(b">")
        if cut < 0:
            continue
        text = text[:cut + 1]
        name = f"plaintext_{offset:012x}.xml"
        path = os.path.join(directory, name)
        with open(path, "wb") as out:
            out.write(text)
        written.append((path, len(text)))
    print(f"\nplaintext documents extracted to {directory}:")
    if not written:
        print("  none")
    for path, length in written:
        print(f"  {path}  ({length} bytes)")
    return written


def classify(f, size, hits):
    """Tells a real structure from a magic string that merely occurs in random bytes. Returns (zip, xml) counts."""
    plausible_zip = 0
    plausible_xml = 0
    print("\nsignature hits, validated (a 4-byte magic in 10 GB of random data occurs by chance):")
    for offset in hits["ZIP local file header"]:
        f.seek(offset)
        head = f.read(30 + 64)
        if len(head) < 30:
            print(f"  0x{offset:x}  ZIP local header: truncated, not a header")
            continue
        version, flags, method = struct.unpack_from("<HHH", head, 4)
        name_len, extra_len = struct.unpack_from("<HH", head, 26)
        name = head[30:30 + min(name_len, 64)]
        printable = all(32 <= b < 127 for b in name) and 0 < name_len <= 256
        if printable and method in (0, 8):
            plausible_zip += 1
        print(f"  0x{offset:x}  ZIP local header: version={version} flags=0x{flags:04x} method={method} "
              f"name_len={name_len} name={name!r} -> {'plausible' if printable and method in (0, 8) else 'random'}")
    for offset in hits["ZIP end of central directory"]:
        f.seek(offset)
        record = f.read(22)
        if len(record) < 22:
            print(f"  0x{offset:x}  ZIP EOCD: truncated")
            continue
        comment_len = struct.unpack_from("<H", record, 20)[0]
        central = struct.unpack_from("<I", record, 16)[0]
        f.seek(central)
        signature = f.read(4)
        real = signature == b"PK\x01\x02" and central + 22 <= offset and offset + 22 + comment_len <= size
        print(f"  0x{offset:x}  ZIP EOCD: comment_len={comment_len} central_dir=0x{central:x} "
              f"signature={signature!r} -> {'plausible' if real else 'random'}")
    for offset in hits["XML text"]:
        f.seek(offset)
        snippet = f.read(48)
        printable = sum(1 for b in snippet if 32 <= b < 127) / max(len(snippet), 1)
        if printable > 0.9:
            plausible_xml += 1
        print(f"  0x{offset:x}  '<?xml': {snippet[:40]!r} -> "
              f"{'plausible' if printable > 0.9 else 'random'}")
    for name in ("Appx block map", "Appx manifest", "Appx signature", "Appx content types", "MSIXVC chunk table",
                 "MSIXVC content table"):
        for offset in hits[name]:
            print(f"  0x{offset:x}  {name}: hit")
    return plausible_zip, plausible_xml


def entropy_profile(f, size, window, count, limit):
    """Entropy of up to `count` windows spread evenly over [0, limit)."""
    print(f"\nentropy, {window // 1024} KiB windows, at most {count} samples over the first "
          f"{limit // (1024 * 1024)} MiB:")
    span = min(limit, size)
    if not span:
        return
    stride = max(window, span // count)
    for offset in range(0, span, stride):
        data = read_at(f, offset, window)
        if not data:
            break
        print(f"  0x{offset:010x}  {entropy(data):.3f} bits/byte")


def text_runs(data, minimum=4):
    """Printable ASCII and UTF-16LE runs in `data`, with their offsets."""
    runs = []
    current, start = [], None
    for index, byte in enumerate(data):
        if 32 <= byte < 127:
            if not current:
                start = index
            current.append(chr(byte))
        else:
            if len(current) >= minimum:
                runs.append((start, "ascii", "".join(current)))
            current = []
    if len(current) >= minimum:
        runs.append((start, "ascii", "".join(current)))
    current, start = [], None
    for index in range(0, len(data) - 1, 2):
        low, high = data[index], data[index + 1]
        if high == 0 and 32 <= low < 127:
            if not current:
                start = index
            current.append(chr(low))
        else:
            if len(current) >= minimum:
                runs.append((start, "utf16", "".join(current)))
            current = []
    if len(current) >= minimum:
        runs.append((start, "utf16", "".join(current)))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package")
    parser.add_argument("--full", action="store_true", help="scan the whole file for signatures")
    parser.add_argument("--window", type=int, default=4096, help="entropy window in bytes (default 4096)")
    parser.add_argument("--extract-plaintext", metavar="DIR",
                        help="write the plaintext XML documents the scan finds into DIR")
    args = parser.parse_args()
    path = args.package
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        dump_header(f, size)

        head = read_at(f, 0, 4096)
        print("\nprintable strings in the first 4 KiB:")
        for offset, kind, text in text_runs(head):
            print(f"  0x{offset:04x}  {kind:<5} {text[:96]!r}")

        scan_limit = size if args.full else min(size, 64 * 1024 * 1024)
        hits = scan(f, size, scan_limit, 8 * 1024 * 1024)
        print(f"\nplaintext signatures over the first {(scan_limit + 0xFFFFF) // (1024 * 1024)} MiB:")
        for _, name in SIGNATURES:
            offsets = hits[name]
            where = ", ".join(f"0x{o:x}" for o in offsets) if offsets else "-"
            print(f"  {name:<28} {len(offsets):>3}  {where}")
        plausible_zip, plausible_xml = (0, 0)
        if hits["ZIP local file header"] or hits["XML text"] or hits["ZIP end of central directory"]:
            plausible_zip, plausible_xml = classify(f, size, hits)
        if args.extract_plaintext:
            extract_plaintext(f, size, hits, args.extract_plaintext)

        # A 1 MiB window is dominated by the payload; the header is what the small windows show.
        entropy_profile(f, size, 4096, 16, min(size, 65536))
        entropy_profile(f, size, 1024 * 1024, 24, scan_limit)
        entropy_profile(f, size, 1024 * 1024, 24, size)

        first = read_at(f, 0, 1)
        pe = len(hits["PE/DOS stub"])
        print(f"\nconclusion: the file starts with {first!r}; "
              f"{plausible_zip} validated ZIP structure(s), {pe} PE image(s), {plausible_xml} plaintext XML "
              f"document(s) in the scanned range. This report is read-only: the protected payload is not decrypted "
              "here (docs/design.md).")


if __name__ == "__main__":
    sys.exit(main())
