"""Aggregate byte equality for complete, single-instance BLAS streams."""
import contextlib
import functools
import hashlib
import itertools
import os
import sqlite3
import zlib

import bvh_index


BLOCK = 64 * 1024


class _Archive:
    """Keep a bounded read cache for the scan and the two interleaved comparison streams."""

    WINDOW = 1 << 20

    def __init__(self, file):
        window_size = self.WINDOW

        @functools.lru_cache(maxsize=4)
        def window(index):
            file.seek(index * window_size)
            # One overlap byte lets the two-byte zlib header straddle windows.
            return file.read(window_size + 1)
        self.window = window

    def read(self, start, end):
        index, offset = divmod(start, self.WINDOW)
        count = end - start
        data = self.window(index)[offset:offset + count]
        if len(data) < count:
            data += self.window(index + 1)[1:1 + count - len(data)]
        return data

    def next_stream(self, pos, end):
        while pos < end:
            index, offset = divmod(pos, self.WINDOW)
            base = index * self.WINDOW
            match = bvh_index.ZHDR.search(self.window(index), offset, end - base)
            if match:
                return base + match.start()
            pos = base + self.WINDOW
        return None


def _blocks(archive, start, end):
    """Yield fixed-size inflated blocks; return compressed bytes consumed at EOF."""
    decoder = zlib.decompressobj()
    pos = start
    block = bytearray()
    while not decoder.eof:
        data = decoder.unconsumed_tail
        if not data:
            if pos == end:
                raise ValueError("truncated compressed stream")
            data = archive.read(pos, min(pos + BLOCK, end))
            if not data:
                raise ValueError("archive ended during scan")
            pos += len(data)
        block.extend(decoder.decompress(data, BLOCK - len(block)))
        if len(block) == BLOCK:
            yield bytes(block)
            block.clear()
    if block:
        yield bytes(block)
    return pos - start - len(decoder.unused_data)


def _inspect(archive, start, end):
    blocks = _blocks(archive, start, end)
    head = b""
    digest = hashlib.sha256()
    total = 0
    while True:
        try:
            block = next(blocks)
        except StopIteration as done:
            return head, total, digest.digest(), done.value
        if not total:
            head = block[:bvh_index.HEAD]
        total += len(block)
        digest.update(block)


def _index(archive, start, end, db, counts):
    pending = None
    pos = start
    while (offset := archive.next_stream(pos, end)) is not None:
        if pending is not None and any(archive.read(i, min(i + BLOCK, offset)).strip(b"\0")
                                       for i in range(pos, offset, BLOCK)):
            counts["uncompared_instances"] += 1
            pending = None
        try:
            head, total, digest, used = _inspect(archive, offset, end)
        except (ValueError, zlib.error):
            if pending is not None:
                counts["uncompared_instances"] += 1
                pending = None
            pos = offset + 1
            continue
        pos = offset + used
        row = bvh_index.classify(head, total)
        if pending is not None:
            entry_size = int.from_bytes(pending[0xa0:0xa4], "little")
            if row["kind"] == "bvh4" and entry_size == ((row["declared_bytes"] + 255) & ~255):
                signature = head[0x100:0x140] + pending[0x50:0x80]
                db.execute("INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
                           (signature, pending[0x40:0x48], digest, offset, pos))
                counts["complete_pairs"] += 1
            else:
                counts["uncompared_instances"] += 1
        pending = head if row["kind"] == "instance208" else None
    if pending is not None:
        counts["uncompared_instances"] += 1


def _compare(archive, db, counts):
    db.execute("CREATE INDEX by_group ON observations(signature, digest, descriptor_key)")
    rows = db.execute("""
        SELECT signature, digest, descriptor_key, start, end FROM observations
        WHERE signature IN (SELECT signature FROM observations GROUP BY signature
                            HAVING min(descriptor_key) != max(descriptor_key))
        ORDER BY signature, digest, descriptor_key
    """)
    for _, group in itertools.groupby(rows, key=lambda r: r[0]):
        counts["shared_header_transform_groups"] += 1
        variants = 0
        for _, matches in itertools.groupby(group, key=lambda r: r[1]):
            variants += 1
            _, _, first_key, start, end = next(matches)
            cross_key = False
            for _, _, key, other_start, other_end in matches:
                pairs = itertools.zip_longest(_blocks(archive, start, end),
                                              _blocks(archive, other_start, other_end))
                if any(left != right for left, right in pairs):
                    raise ValueError("digest collision during full byte comparison")
                cross_key |= key != first_key
            counts["exact_payloads_shared_across_keys"] += int(cross_key)
        counts["groups_with_multiple_payloads"] += int(variants > 1)


def scan(path, start=None, limit=None):
    counts = dict.fromkeys(("complete_pairs", "uncompared_instances", "shared_header_transform_groups",
                            "groups_with_multiple_payloads", "exact_payloads_shared_across_keys"), 0)
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size
        if start is not None and not 0 <= start <= size:
            raise ValueError("start offset is outside the archive")
        if not size:
            return counts
        archive = _Archive(file)
        with contextlib.closing(sqlite3.connect("")) as db:
            if start is None:
                start = int.from_bytes(archive.read(4, 8), "little") if size >= 8 else size
            if start > size:
                raise ValueError("start offset is outside the archive")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE observations(signature BLOB, descriptor_key BLOB, digest BLOB, "
                       "start INTEGER, end INTEGER)")
            _index(archive, start, min(size, start + limit) if limit else size, db, counts)
            _compare(archive, db, counts)
    return counts
