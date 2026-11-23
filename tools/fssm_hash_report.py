"""Compare each inflated FSSM descriptor with its enclosing DXBC HASH part."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fssm_unpack import _inflate_xbch, parts, xdat_entries_all


def inspect_container(container):
    counts = Counter(containers=1)
    try:
        if container[:4] != b"DXBC":
            raise ValueError("invalid magic")
        chunks = parts(container)
        hash_part = chunks.get(b"HASH")
        expected = None
        if hash_part is None:
            counts["missing_hash"] += 1
        elif len(hash_part) != 20:
            counts["invalid_hash"] += 1
        else:
            digest = hash_part[4:]
            expected = digest[:2] + digest[4:8] + digest[8:10] + digest[12:16]
        if b"DXIL" not in chunks:
            counts["missing_dxil"] += 1
            return counts
        dxil = chunks[b"DXIL"]
        if b"XDAT" not in dxil:
            counts["missing_xdat"] += 1
            return counts
        entries = xdat_entries_all(dxil)
    except ValueError:
        counts["invalid_container"] += 1
        return counts

    for tag, value in entries:
        if tag != 0x0D:
            continue
        counts["payloads"] += 1
        try:
            blob = _inflate_xbch(value)
        except (ValueError, struct.error, zlib.error):
            blob = None
        if blob is None:
            counts["invalid_payload"] += 1
            continue
        if len(blob) < 0xC8:
            counts["invalid_descriptor"] += 1
            continue
        start = struct.unpack_from("<Q", blob, 0xC0)[0] + 0x10
        block = blob[start:start + 24]
        if len(block) != 24:
            counts["invalid_descriptor"] += 1
            continue
        # Low halves of w0 and w3, plus the full w2 and w5, in digest order.
        projection = block[:2] + block[8:12] + block[12:14] + block[20:24]
        if expected is None:
            counts["no_comparable_hash"] += 1
        elif projection == expected:
            counts["matched"] += 1
        else:
            counts["mismatched"] += 1
    if not counts["payloads"]:
        counts["missing_xbch"] += 1
    elif counts["payloads"] > 1:
        counts["multi_payload_containers"] += 1
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory of carved .dxbc files")
    args = parser.parse_args(argv)
    counts = Counter(containers=0, payloads=0, matched=0, mismatched=0)
    try:
        for path in args.directory.iterdir():
            if path.suffix.lower() == ".dxbc" and path.is_file():
                try:
                    counts.update(inspect_container(path.read_bytes()))
                except OSError:
                    counts["containers"] += 1
                    counts["unreadable"] += 1
    except OSError as error:
        parser.error(str(error))
    print(json.dumps(counts, sort_keys=True))
    return int(not counts["containers"] or counts["payloads"] != counts["matched"]
               or any(counts[k] for k in ("invalid_container", "missing_dxil", "missing_xdat",
                                          "missing_xbch", "unreadable")))


if __name__ == "__main__":
    sys.exit(main())
