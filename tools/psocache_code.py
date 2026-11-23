"""Decode the code blob of a PSO/shader cache as RDNA2 and count what does not decode.

``docs/pso_cache.md`` locates a code blob and a shader index of ``(u32 offset, u32 size)`` pairs over it, and
leaves open whether the blob is raw RDNA2: the evidence was only the ``s_endpgm`` and ``s_code_end`` marker words.
This tool runs the repository's own decoder (``tools/rdna2_decode.py``) over every shader-index entry and reports
how much of it decodes.

Each entry starts with a 36-byte header that is not code (``--header``, measured on the retail cache; see
``docs/pso_cache.md``). Per entry, the *program* part is the bytes after the header through its last ``s_endpgm``; the *tail* is the
rest of the entry. The program part is decoded instruction by instruction (instruction lengths come from the
encodings, so literal constants and 64-bit encodings are skipped correctly); a word the decoder rejects counts as
invalid and decoding resumes at the next dword. The tail is counted by word: ``s_code_end``, zero, or other.

The report is counts only: entries, programs (``s_endpgm`` reached by the decoder, not by a byte search), decoded
instructions, invalid words, entries that decode with no invalid word, and a histogram of encoding families
(opcode prefixes ``s``, ``v``, ``image`` ...). It never prints an instruction, an operand or a byte of the blob.

Usage:
    python -I psocache_code.py <cache> [--limit N] [--header BYTES] [--json] [--mcpu gfx1013|gfx1030]

Exit code: 0 when every decoded entry has no invalid word, 1 when some entry has one, 2 when the file cannot be
read or does not start with the cache magic.
"""

import argparse
import collections
import json
import mmap
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import psocache_inspect as P  # noqa: E402
import rdna2_decode as D  # noqa: E402

ENTRY_HEADER = 36  # measured: the code of 19,825 of 21,522 entries decodes cleanly from here
END_PGM = 0xBF810000
CODE_END = 0xBF9F0000


def _code_region(m, size):
    counts = [P.u32(m, off) for off in P.COUNT_OFFS]
    meta_size, code_size = (P.u32(m, off) for off in P.SIZE_OFFS)
    shader_index = (
        P.HEADER_SIZE
        + counts[0] * P.GRAPHICS_STRIDE
        + counts[1] * P.COMPUTE_STRIDE
        + counts[2] * P.INDEX_STRIDE
    )
    code = shader_index + counts[3] * P.INDEX_STRIDE + meta_size
    if code + code_size > size:
        raise ValueError("code blob ends at 0x%x, past the file end 0x%x" % (code + code_size, size))
    return shader_index, counts[3], code, code_size


def _last_endpgm(buf):
    """Byte offset just past the last s_endpgm dword in ``buf`` (dword-aligned), or 0 when there is none."""
    pos = len(buf) & ~3
    marker = struct.pack("<I", END_PGM)
    while pos >= 4:
        hit = buf.rfind(marker, 0, pos)
        if hit < 0:
            return 0
        if hit % 4 == 0:
            return hit + 4
        pos = hit + 3
    return 0


def _family(ins):
    """The opcode's prefix (``s``, ``v``, ``image``, ``buffer``, ``ds``, ``exp`` ...); never its bytes."""
    return ins["op"].split("_", 1)[0]


def analyse(path, limit=None, mcpu="gfx1013", header=ENTRY_HEADER):
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < P.HEADER_SIZE:
            raise ValueError("file is %d bytes, shorter than the header" % size)
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if bytes(m[:4]) != P.MAGIC:
                raise ValueError("no PS\\0\\x09 magic at 0x0")
            sidx, nentries, code, code_size = _code_region(m, size)
            r = {
                "entries": 0,
                "entries_out_of_bounds": 0,
                "entries_without_endpgm": 0,
                "entries_clean": 0,
                "entries_with_invalid": 0,
                "programs": 0,
                "instructions": 0,
                "invalid_words": 0,
                "program_bytes": 0,
                "tail_words": {"s_code_end": 0, "zero": 0, "other": 0},
                "families": collections.Counter(),
                "decoder_mcpu": mcpu,
                "entry_header": header,
                "entries_shorter_than_header": 0,
            }
            n = nentries if limit is None else min(limit, nentries)
            for i in range(n):
                off, length = struct.unpack_from("<2I", m, sidx + i * 8)
                r["entries"] += 1
                if off + length > code_size:
                    r["entries_out_of_bounds"] += 1
                    continue
                if length < header:
                    r["entries_shorter_than_header"] += 1
                    continue
                buf = bytes(m[code + off + header : code + off + length])
                end = _last_endpgm(buf)
                if end == 0:
                    r["entries_without_endpgm"] += 1
                for (w,) in struct.iter_unpack("<I", buf[end : len(buf) & ~3]):
                    key = "s_code_end" if w == CODE_END else ("zero" if w == 0 else "other")
                    r["tail_words"][key] += 1
                if end == 0:
                    continue
                d = D.decode_bytes(buf[:end], mcpu=mcpu)
                r["program_bytes"] += end
                r["instructions"] += len(d["insns"])
                r["invalid_words"] += d["invalid"]
                for ins in d["insns"]:
                    r["families"][_family(ins)] += 1
                    if ins["op"] == "s_endpgm":
                        r["programs"] += 1
                if d["invalid"]:
                    r["entries_with_invalid"] += 1
                else:
                    r["entries_clean"] += 1
            r["shader_index_entries"] = nentries
            r["families"] = dict(r["families"].most_common())
            return r


def print_report(r, out=sys.stdout):
    w = out.write
    w("shader-index entries: %d (decoded %d)\n" % (r["shader_index_entries"], r["entries"]))
    w("  out of bounds: %d, no s_endpgm: %d\n" % (r["entries_out_of_bounds"], r["entries_without_endpgm"]))
    w("  decode with no invalid word: %d, with invalid words: %d\n" % (r["entries_clean"], r["entries_with_invalid"]))
    w("program bytes: %d, instructions: %d, invalid words: %d\n" % (r["program_bytes"], r["instructions"], r["invalid_words"]))
    w("programs (s_endpgm decoded on an instruction boundary): %d\n" % r["programs"])
    t = r["tail_words"]
    w("tail words: s_code_end %d, zero %d, other %d\n" % (t["s_code_end"], t["zero"], t["other"]))
    w("encoding families (%s):\n" % r["decoder_mcpu"])
    for fam, c in r["families"].items():
        w("  %-8s %d\n" % (fam, c))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cache")
    ap.add_argument("--limit", type=int, help="decode only the first N shader-index entries")
    ap.add_argument("--mcpu", default="gfx1013", choices=("gfx1013", "gfx1030"))
    ap.add_argument("--header", type=int, default=ENTRY_HEADER, help="bytes of entry header to skip (default %(default)s)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1")
    if a.header < 0 or a.header % 4:
        ap.error("--header must be a non-negative multiple of 4")
    try:
        r = analyse(a.cache, a.limit, a.mcpu, a.header)
    except (OSError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    if a.json:
        json.dump(r, sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print_report(r)
    return 1 if r["entries_with_invalid"] or r["entries_out_of_bounds"] else 0


if __name__ == "__main__":
    sys.exit(main())
