"""Carve shader containers out of an RPF9 archive without reading its (encrypted) table of contents.

Each shader in shaders.rpf is stored as:

    <name> 00  <u8 kind>  <u32 stored>  <u32 id>  <u32 stored>  <u32 size>  <raw deflate, `stored` bytes>

and inflates to a DXBC container (`size` bytes) holding DXIL plus an Xbox PRIV part.

Usage:
    python -I shader_carve.py <archive.rpf> <out_dir>

Writes <out_dir>/<name>.dxbc and <out_dir>/index.json (name, offset, stage, shader model, parts).
"""
import argparse
import mmap
import os
import re
import struct
import sys
import tempfile
import zlib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from json.encoder import encode_basestring_ascii as _quote

# `python -I` leaves the script's directory off sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replace_retry import replace as replace_retry  # noqa: E402

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


def index_entry(name, fname, offset, kind, stage, sm, size, p) -> str:
    """One index.json row, as the text `json.dump(..., indent=1)` writes for the same dict one level in.
    Formatting it here skips the pure-Python encoder `indent` forces and the dict kept per shader."""
    q = _quote
    rows = ",\n".join(f"   {q(k)}: {v[1]}" for k, v in p.items())
    return (f' {{\n  "name": {q(name)},\n  "file": {q(fname)},\n  "offset": {offset},\n  "kind": {kind},\n'
            f'  "stage": {"null" if stage is None else q(stage)},\n  "sm": {"null" if sm is None else q(sm)},\n'
            f'  "size": {size},\n  "parts": ' + (f"{{\n{rows}\n  }}\n }}" if rows else "{}\n }"))


def write_dxbc(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


class WriteBehind:
    """Creates the .dxbc files on worker threads (file creation is slow kernel work that drops the GIL) while the
    caller inflates the next shader. The data still waiting for its write is bounded by `throttle`. The first
    `inline` files are written by the caller: a small archive has nothing to gain from the threads' fixed memory."""

    def __init__(self, workers: int = 4, inline: int = 64):
        self.workers = workers
        self.inline = inline
        self.pool = None
        self.queue = deque()  # (future, size) in submission order
        self.held = 0         # bytes of the queued entries whose write has not been collected yet

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True)  # also on an error: no worker is left writing into the folder being reported

    def submit(self, path: str, data: bytes) -> None:
        if self.inline > 0:
            self.inline -= 1
            write_dxbc(path, data)
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.workers)
        self.queue.append((self.pool.submit(write_dxbc, path, data), len(data)))
        self.held += len(data)

    def throttle(self, limit: int) -> None:
        """Wait until at most `limit` bytes are still queued; raise the OSError of a write that failed."""
        q = self.queue
        while q and (q[0][0].done() or self.held > limit):
            fut, n = q.popleft()
            fut.result()
            self.held -= n

    def drain(self) -> None:
        self.throttle(-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Carve DXBC shader containers out of an RPF9 archive.")
    ap.add_argument("src", help="the shaders.rpf archive")
    ap.add_argument("out", help="folder for the .dxbc files and index.json")
    a = ap.parse_args(argv)
    src, out = a.src, a.out
    if not os.path.isfile(src):
        ap.error(f"{src} is not a file")
    if os.path.getsize(src) == 0:
        ap.error(f"{src} is empty")
    if os.path.isfile(out):
        ap.error(f"{out} is a file, not a folder")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:  # a regular file where a parent folder is expected
        ap.error(f"cannot create {out}: {e.strerror or e}")
    rows, seen, used, bad = [], {}, set(), 0
    stages, models, part_names = Counter(), Counter(), Counter()
    with WriteBehind() as writer, open(src, "rb") as fh, \
            mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m, memoryview(m) as view:
        pos = last = 0
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
            # The loop used to hold the last inflated shader plus a copy of the next stored bytes; the queue may hold
            # as much, so the peak memory stays what it was. The shader just queued is no longer held by the loop.
            data = None
            writer.throttle(last + stored)
            decompressor = zlib.decompressobj(-15)
            try:
                data = decompressor.decompress(view[start:start + stored], size + 1)
            except zlib.error:
                bad += 1
                continue
            last = len(data)
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
            writer.submit(os.path.join(out, fname), data)
            rows.append(index_entry(name, fname, mt.start(), mt.group(2)[0], stage, sm, size, p))
            stages[stage] += 1
            models[sm] += 1
            part_names.update(p.keys())
        writer.drain()
    if not rows:
        print(f"ERROR: no shaders carved from {src} (rejected candidates: {bad}).", file=sys.stderr)
        return 1
    # make_game.py takes index.json as proof the carve finished: write it last, under another name, then rename.
    with tempfile.TemporaryDirectory(dir=out, prefix=".shader-index-") as staging:
        tmp = os.path.join(staging, "index.json")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write("[\n" + ",\n".join(rows) + "\n]")
            f.flush()
            os.fsync(f.fileno())
        replace_retry(tmp, os.path.join(out, "index.json"))
    print(f"{len(rows)} shaders carved, {bad} candidates rejected")
    print("stages:", dict(stages))
    print("shader models:", dict(models))
    print("parts:", dict(part_names))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
