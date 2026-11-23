"""Inflate one zlib candidate from an archive scan without letting its output grow without bound.

The archive scanners find a zlib header, inflate from it in 1 MiB input blocks and keep the stream when it ends.
`zlib.decompressobj().decompress(block)` has no output limit, and one 1 MiB block of a highly compressed stream
can inflate to about 1 GiB; a false candidate or a huge stream then grows in memory until the scan dies
(docs/story.md: the full story000.rpf walk stopped at 72% with one candidate at 8.5 GB). This inflates the same
way but stops once the output passes `max_out` bytes, and reads at most `step` bytes of output per call, so the
peak stays near `max_out` + one step.

A stream that passes the cap is reported as `"too_big"` and the scanners skip it like a candidate that does not
end in the window (they continue at the next byte and count it in `skipped_candidates`); every stream at or
under the cap inflates to the same bytes and the same consumed length as before. restofworld_index.py, which
already stops at its own cap, uses the same 512 MiB default.
"""
import zlib

MAX_INFLATE = 512 << 20          # bytes of output; restofworld_index.py's default
STEP = 1 << 20                   # input block and per-call output bound


def inflate(buf, start, end, max_out=None, step=STEP):
    """Inflate the zlib stream that starts at buf[start], reading no input at or past `end`.

    Returns (data, used, status):
      - ("ok"): the stream ended inside the window and inflated to at most `max_out` bytes; `data` is the
        inflated bytes and `used` the number of stored bytes the stream consumed.
      - (None, 0, "too_big"): the output passed `max_out` before the stream ended.
      - (None, 0, "truncated"): the window (or the buffer) ended before the stream did.
    Raises zlib.error for bytes that are not a valid zlib stream, as decompress() does.
    `max_out` defaults to MAX_INFLATE, read at call time."""
    if max_out is None:
        max_out = MAX_INFLATE
    end = min(end, len(buf))
    d = zlib.decompressobj()
    parts, total, p = [], 0, start
    while not d.eof and p < end:
        block = buf[p:min(p + step, end)]
        if not block:
            break
        p += len(block)
        while block and not d.eof:
            chunk = d.decompress(block, step)
            block = d.unconsumed_tail
            if chunk:
                total += len(chunk)
                if total > max_out:
                    return None, 0, "too_big"
                parts.append(chunk)
    if not d.eof:
        return None, 0, "truncated"
    data = b"".join(parts)
    return data, p - start - len(d.unused_data), "ok"
