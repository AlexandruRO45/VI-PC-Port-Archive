"""Carve shader containers out of an RPF9 archive without reading its (encrypted) table of contents.

Each shader in shaders.rpf is stored as:

    <name> 00  <u8 kind>  <u32 stored>  <u32 id>  <u32 stored>  <u32 size>  <raw deflate, `stored` bytes>

and inflates to a DXBC container (`size` bytes) holding DXIL plus an Xbox PRIV part.

Usage:
    python -I shader_carve.py <archive.rpf> <out_dir>

Writes <out_dir>/<name>.dxbc and <out_dir>/index.json (name, offset, stage, shader model, parts).

Reruns stage all files on the output volume. Publishers using this tool acquire .shader-publish by
creating that directory exclusively. A competing publisher fails if the directory is already present.
The index rename commits a run. An I/O failure before commit rolls back the installed files, including
new names. Unrelated files remain in place. Readers needing a consistent snapshot must finish before
publication or wait until the writer exits: stable DXBC paths change individually during publication.
There is no crash recovery. A crash or a second I/O failure during rollback can leave .shader-publish
with the remaining original files. Inspect the output and recover it before clearing the guard.
A cleanup error after commit leaves the new index and payloads published.
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib

ENTRY = re.compile(rb"([\x21-\x7e]{3,200})\x00(.)(....)(....)\3(....)", re.DOTALL)
STAGES = {0: "ps", 1: "vs", 2: "gs", 3: "hs", 4: "ds", 5: "cs", 6: "lib", 13: "ms", 14: "as"}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def parts(dxbc: bytes) -> dict:
    if len(dxbc) < 32:
        raise ValueError("truncated DXBC header")
    count = struct.unpack_from("<I", dxbc, 28)[0]
    table_end = 32 + 4 * count
    if table_end > len(dxbc):
        raise ValueError("truncated DXBC part table")
    out = {}
    for entry in range(32, table_end, 4):
        off = struct.unpack_from("<I", dxbc, entry)[0]
        if not table_end <= off <= len(dxbc) - 8:
            raise ValueError("DXBC part header outside the part data")
        fourcc, size = struct.unpack_from("<4sI", dxbc, off)
        if off + 8 + size > len(dxbc):
            raise ValueError("truncated DXBC part payload")
        out[fourcc.decode("latin-1")] = (off + 8, size)
    return out


def shader_model(dxbc: bytes, p: dict):
    if "DXIL" not in p:
        return None, None
    off, size = p["DXIL"]
    if size < 4:
        raise ValueError("truncated DXIL version and stage")
    ver, kind = struct.unpack_from("<HH", dxbc, off)
    return STAGES.get(kind, str(kind)), f"{ver >> 4}.{ver & 0xF}"


def file_name(name: str, seen: dict, used: set) -> str:
    """A file name for `name` that no earlier shader has, as Windows compares them: without case, and with the
    device names (CON, NUL, COM1...) that open a device instead of a file. Same name twice means distinct
    variants; the second gets ~1, the third ~2."""
    base = re.sub(r'[<>:"/\\|?*]', "_", name)
    if base.split(".")[0].upper() in RESERVED:
        base = "_" + base
    key = base.lower()
    n = seen.get(key, 0)
    while True:
        fname = base + (f"~{n}" if n else "") + ".dxbc"
        if fname.lower() not in used:
            break
        n += 1
    seen[key] = n + 1
    used.add(fname.lower())
    return fname


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Carve DXBC shader containers out of an RPF9 archive.")
    ap.add_argument("src", help="the shaders.rpf archive")
    ap.add_argument("out", help="folder for the .dxbc files and index.json")
    a = ap.parse_args(argv)
    src, out = a.src, a.out
    if not os.path.isfile(src):
        ap.error(f"{src} is not a file")
    try:
        if os.path.getsize(src) == 0:
            ap.error(f"{src} is empty")
    except OSError as e:
        ap.error(f"cannot read {src}: {e.strerror or e}")
    if os.path.isfile(out):
        ap.error(f"{out} is a file, not a folder")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:  # a regular file where a parent folder is expected
        ap.error(f"cannot create {out}: {e.strerror or e}")
    try:
        with tempfile.TemporaryDirectory(dir=out, prefix=".shader-stage-") as staging:
            return carve(src, out, staging)
    except OSError as e:
        print(f"ERROR: cannot write {out}: {e}", file=sys.stderr)
        return 1


def publish(index, staging, out):
    """Publish staged files under their stable names; restore the previous files on failure."""
    backup = os.path.join(out, ".shader-publish")
    os.mkdir(backup)  # Exclusive publication, including rollback, for cooperating writers.
    undo = []
    try:
        for row in index:
            name = row["file"]
            dest = os.path.join(out, name)
            old = os.path.join(backup, name)
            if os.path.isdir(dest):
                raise IsADirectoryError(dest)
            if os.path.lexists(dest):
                os.replace(dest, old)
                undo.append((dest, old))
                os.replace(os.path.join(staging, name), dest)
            else:
                os.replace(os.path.join(staging, name), dest)
                undo.append((dest, None))
        os.replace(os.path.join(staging, "index.json"), os.path.join(out, "index.json"))
    except OSError:
        # If restoration itself fails, leave the backups and writer guard in place.
        for dest, old in reversed(undo):
            if old is None:
                os.unlink(dest)
            else:
                os.replace(old, dest)
        os.rmdir(backup)
        raise
    # The index commit succeeded. Move backups to our private staging area for cleanup.
    os.rename(backup, os.path.join(staging, "previous"))


def carve(src, out, staging):
    index, seen, used, bad = [], {}, set(), 0
    try:
        with open(src, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            pos = 0
            while (mt := ENTRY.search(m, pos)) is not None:
                # A rejected header is rescanned past its name (a real entry can start inside it); a carved
                # shader resumes after its stored bytes, so deflate output never becomes a name or a match.
                pos = mt.end(1) + 1
                name = mt.group(1).decode("latin-1")
                stored = struct.unpack("<I", mt.group(3))[0]
                size = struct.unpack("<I", mt.group(5))[0]
                if not (16 < stored < 64 << 20 and 32 < size < 256 << 20):
                    continue
                start = mt.end()
                decompressor = zlib.decompressobj(-15)
                try:
                    data = decompressor.decompress(m[start:start + stored], size + 1)
                except zlib.error:
                    bad += 1
                    continue
                if not decompressor.eof or data[:4] != b"DXBC" or len(data) != size:
                    bad += 1
                    continue
                try:
                    p = parts(data)
                    stage, sm = shader_model(data, p)
                except ValueError:
                    bad += 1
                    continue
                pos = start + stored
                fname = file_name(name, seen, used)
                out_path = os.path.join(staging, fname)
                try:
                    with open(out_path, "wb") as f:
                        f.write(data)
                except OSError as e:
                    print(f"ERROR: cannot write {out_path}: {e.strerror or e}", file=sys.stderr)
                    return 1
                index.append({"name": name, "file": fname, "offset": mt.start(), "kind": mt.group(2)[0],
                              "stage": stage, "sm": sm, "size": size, "parts": {k: v[1] for k, v in p.items()}})
    except OSError as e:
        print(f"ERROR: cannot read {src}: {e.strerror or e}", file=sys.stderr)
        return 1
    if not index:
        print(f"ERROR: no shaders carved from {src} (rejected candidates: {bad}).", file=sys.stderr)
        return 1
    # make_game.py takes index.json as proof the carve finished: write it last, under another name, then rename.
    with open(os.path.join(staging, "index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    publish(index, staging, out)
    from collections import Counter
    print(f"{len(index)} shaders carved, {bad} candidates rejected")
    print("stages:", dict(Counter(e["stage"] for e in index)))
    print("shader models:", dict(Counter(e["sm"] for e in index)))
    print("parts:", dict(Counter(k for e in index for k in e["parts"])))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
