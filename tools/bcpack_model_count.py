"""Count which entropy model the BCPack chunks of a game copy need (ASSET-519).

Two decoders read the lane streams of a BCPack chunk:

  * the strict Python model, tools/bcpack.py `decode_lanes`, which shifts up to 8 "virtual zero bytes" into a
    flushed fragment whose trailing zero bytes were omitted (docs/textures.md, `flush = 1`);
  * the Rust rANS decoder in third_party/bcpack-rs (`rans::decode_slice`), reached through the `unpack` export of
    tools/bcpack_calibration, which does not have them and refuses such a stream.

This tool walks every texture pool below a game folder (the same discovery as tools/ttpf_mips.py), runs both on each
BCPack chunk and prints COUNTS only: chunks, fragments, accepted by both, only by Python, only by Rust, by neither
(per error class), accepted only thanks to virtual zero bytes, and symbol disagreements where both accept. It never
prints, stores or writes a byte, symbol, file name, path or content hash of the game; it only reads the pools.

    python -I tools/bcpack_model_count.py --game-dir GAME --work WORK [--rs-lib LIB] [--python-only]
                                          [--python-stride N] [--limit-chunks N] [--no-vz-check]

--work is a folder for the counts file (bcpack_model_count.json); it must not be inside the game folder. Without the
compiled Rust library (build it as tools/bcpack_calibration/README.md describes, or pass --rs-lib) the run stops with
exit 77, unless --python-only asks for the Python counts alone. The Python model is a cycle-level simulation and slow
(a few chunks per second): the Rust decoder runs on every chunk, the Python model on every chunk the Rust decoder does
not fully accept (so "only Python" and "neither" are exact) and, with --python-stride N, on every Nth of the others
(so "both" and the symbol disagreements are counted over that sample); --rust-only skips Python altogether. The counts
say how many chunks each model saw.

"accepted_only_thanks_to_virtual_zero_bytes" counts the chunks the strict Python model accepts but rejects when it may
shift no virtual zero byte at all (MAX_VIRTUAL_ZEROS = 0). It includes uses the Rust decoder handles too; the chunks the
Rust decoder refuses are "only_python". Exit codes: 0 done, 1 symbol disagreements, 2 bad arguments or no pools, 77 Rust
library unavailable.

The decoders' outcome for a chunk (`classify`) and the counting (`tally`) are plain functions; tests/test_bcpack_model_count.py
drives them with hand-built streams and a stand-in for the Rust side, with no game data.
"""
import argparse
import ctypes
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # `python -I` implies -P, so the script's directory is not on sys.path

import bcpack  # noqa: E402

# The four width-9 probability tables tools/bcpack_tables.py and third_party/bcpack-rs/src/tables.rs disagree on.
DIVERGENT_W9_Q = (8, 10, 13, 14)

# message fragment of a BcpackError -> a class name that carries nothing of the stream
ERROR_CLASSES = (
    ("input ran out", "input ran out"),
    ("did not end at its home state", "fragment not at home state"),
    ("probability table for this width and q is not known", "no probability table"),
    ("symbol outside its width", "symbol outside its width"),
    ("symbol width out of range", "width out of range"),
    ("non-zero data left over", "non-zero trailing data"),
    ("not a whole number", "size not a whole number of blocks"),
    ("unsupported BCPack codec", "unsupported codec"),
)


class RsUnavailable(Exception):
    """The compiled Rust decoder is not there; the caller turns it into exit 77."""


def error_class(exc):
    text = str(exc)
    for needle, name in ERROR_CLASSES:
        if needle in text:
            return name
    return "other: " + type(exc).__name__


class ModelCounter:
    """Counts the fragments a decode opens: tools/bcpack.py asks for a probability model once per fragment header."""

    def __init__(self):
        self.fragments = 0
        self.width9 = 0
        self.width9_divergent = 0

    def __enter__(self):
        self._real = bcpack._get_model

        def counting(w, q):
            self.fragments += 1
            if w == 9:
                self.width9 += 1
                if q in DIVERGENT_W9_Q:
                    self.width9_divergent += 1
            return self._real(w, q)
        bcpack._get_model = counting
        return self

    def __exit__(self, *exc):
        bcpack._get_model = self._real
        return False


def python_decode(raw, codec, size, virtual_zeros=True):
    """('ok', lanes, bytes_used, counter) or ('error', class, None, counter) from the strict Python model.
    ``virtual_zeros=False`` runs the same model with no virtual zero bytes (MAX_VIRTUAL_ZEROS = 0)."""
    saved = bcpack.MAX_VIRTUAL_ZEROS
    bcpack.MAX_VIRTUAL_ZEROS = saved if virtual_zeros else 0
    try:
        with ModelCounter() as counter:
            try:
                lanes, used = bcpack.decode_lanes(raw, codec, size)
            except bcpack.BcpackError as exc:
                return "error", error_class(exc), None, counter
        return "ok", lanes, used, counter
    finally:
        bcpack.MAX_VIRTUAL_ZEROS = saved


class RsDecoder:
    """The `unpack` export of tools/bcpack_calibration (third_party/bcpack-rs `rans::decode_slice`)."""

    def __init__(self, path):
        if not os.path.isfile(path):
            raise RsUnavailable("the bcpack_calibration library is not built (see tools/bcpack_calibration/README.md)")
        try:
            self.lib = ctypes.CDLL(path)
        except OSError as exc:
            raise RsUnavailable(f"the bcpack_calibration library does not load: {type(exc).__name__}") from exc
        self.lib.unpack.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
                                    ctypes.c_size_t, ctypes.c_void_p]
        self.lib.unpack.restype = ctypes.c_ssize_t

    def decode(self, raw, codec, size, want_symbols=True):
        """('full'|'short'|'error', per-lane symbol lists or None, bytes_used). 'short' consumed fewer bytes than the
        chunk has: tools/calibrated_texture.py then falls back to the Python model, as for an error. The symbol lists
        are only built when asked for (they are the slow part of a big chunk)."""
        widths = bcpack.LANE_WIDTHS[codec]
        nl, nsym = len(widths), size // bcpack.ELEMENT_BYTES
        out = (ctypes.c_ubyte * (nsym * nl))()
        raw = bytes(raw)
        used = self.lib.unpack(raw, len(raw), bytes(w - 1 for w in widths), nl, nsym, ctypes.addressof(out))
        if used < 0:
            return "error", None, used
        lanes = [list(out[lane::nl]) for lane in range(nl)] if want_symbols else None
        return ("full" if used == len(raw) else "short"), lanes, used


def classify(raw, codec, size, rs=None, vz_check=True, run_python=True, python_if=None):
    """Both decoders on one chunk. ``rs`` is an object with decode(raw, codec, size, want_symbols) like RsDecoder, or
    None. ``python_if(rs_status)``, when given, decides after the Rust decode whether the Python model runs (the scan
    uses it to run Python on every refusal and on a sample of the rest). Returns a dict of plain values, no stream
    content."""
    result = {"rs": "n/a", "py": "n/a", "fragments": 0, "w9": 0, "w9_divergent": 0, "vz_needed": None, "agree": None}
    if rs is not None:
        result["rs"] = rs.decode(raw, codec, size, want_symbols=False)[0]
    if run_python and python_if is not None:
        run_python = bool(python_if(result["rs"]))
    if run_python:
        status, lanes, used, counter = python_decode(raw, codec, size)
        result["fragments"], result["w9"], result["w9_divergent"] = counter.fragments, counter.width9, counter.width9_divergent
        result["py"] = "ok" if status == "ok" else lanes          # the error class
        if status == "ok" and vz_check:
            strict, strict_lanes, _, _ = python_decode(raw, codec, size, virtual_zeros=False)
            result["vz_needed"] = strict != "ok" or strict_lanes != lanes
        if status == "ok" and result["rs"] == "full":
            result["agree"] = rs.decode(raw, codec, size, want_symbols=True)[1] == lanes
    return result


def tally(counts, result):
    """Add one classify() result to a Counter of named counts."""
    counts["chunks_classified"] += 1
    counts["fragments"] += result["fragments"]
    counts["width9_fragments"] += result["w9"]
    counts["width9_divergent_q_fragments"] += result["w9_divergent"]
    rs, py = result["rs"], result["py"]
    if rs != "n/a":
        counts[f"rust_{rs}"] += 1
    if py == "n/a":
        return
    counts["python_run"] += 1
    py_ok = py == "ok"
    counts["python_accepts" if py_ok else "python_rejects"] += 1
    if not py_ok:
        counts[f"python_error[{py}]"] += 1
    if result["vz_needed"]:
        counts["accepted_only_thanks_to_virtual_zero_bytes"] += 1
        if rs != "n/a" and rs == "full":
            counts["accepted_by_both_although_python_shifts_virtual_zeros"] += 1
    if rs == "n/a":
        return
    rs_full = rs == "full"
    if py_ok and rs_full:
        counts["accepted_by_both"] += 1
        if result["agree"] is False:
            counts["symbol_disagreements"] += 1
    elif py_ok:
        counts["only_python"] += 1
        counts[f"only_python_rust_{rs}"] += 1
        if result["vz_needed"]:
            counts["only_python_and_needs_virtual_zero_bytes"] += 1
    elif rs_full:
        counts["only_rust"] += 1
    else:
        counts["neither"] += 1
        counts[f"neither[python: {py}; rust: {rs}]"] += 1


def default_rs_library():
    name = {"win32": "bcpack_calibration.dll", "darwin": "libbcpack_calibration.dylib"}.get(
        sys.platform, "libbcpack_calibration.so")
    return os.path.join(HERE, "bcpack_calibration", "target", "release", name)


def inside(path, folder):
    path, folder = os.path.realpath(path), os.path.realpath(folder)
    return os.path.commonpath([path, folder]) == folder


def scan(game_dir, rs, stride=1, limit=None, vz_check=True, run_python=True, clock=time.time):
    """Walk every pool below game_dir and tally its BCPack chunks. Returns (Counter, seconds). With a Rust decoder the
    Python model runs on every chunk it does not fully accept and on every ``stride``-th of the others; without one
    (--python-only) on every ``stride``-th chunk."""
    import ttpf_mips
    counts = Counter()
    started = clock()
    pools = ttpf_mips.find_pools(game_dir)
    counts["pools_found"] = len(pools)
    seen = 0
    for path in pools:
        try:
            pool = ttpf_mips.Pool(path)
        except Exception as exc:  # noqa: BLE001 - counted by type only, never by name
            counts[f"pool_not_opened[{type(exc).__name__}]"] += 1
            continue
        try:
            counts["pools_read"] += 1
            for virt, size, start, packed, codec, pad in pool.chunks:
                counts["chunks_total"] += 1
                if codec not in bcpack.LANE_WIDTHS:
                    counts["chunks_not_bcpack"] += 1
                    continue
                counts["chunks_bcpack"] += 1
                if limit is not None and seen >= limit:
                    continue
                position = seen
                seen += 1
                tally(counts, classify(pool.data[start:start + packed], codec, size, rs, vz_check, run_python,
                                       lambda status, n=position: (status not in ("full", "n/a")) or n % stride == 0))
        finally:
            pool.close()
    return counts, clock() - started


def render(counts, seconds, rs_note):
    order = ["pools_found", "pools_read", "chunks_total", "chunks_not_bcpack", "chunks_bcpack", "chunks_classified",
             "python_run", "fragments", "accepted_by_both", "only_python", "only_rust", "neither",
             "accepted_only_thanks_to_virtual_zero_bytes", "symbol_disagreements", "width9_fragments",
             "width9_divergent_q_fragments"]
    lines = [f"rust decoder: {rs_note}"]
    lines += [f"{k}: {counts[k]}" for k in order]
    lines += [f"{k}: {counts[k]}" for k in sorted(counts) if k not in order]
    lines.append(f"seconds: {seconds:.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game-dir", required=True, help="the game copy (only read)")
    ap.add_argument("--work", required=True, help="folder for the counts file; not inside the game folder")
    ap.add_argument("--rs-lib", default=None, help="the compiled bcpack_calibration library")
    ap.add_argument("--python-only", action="store_true", help="count the Python model alone (no Rust library needed)")
    ap.add_argument("--python-stride", type=int, default=1,
                    help="run the Python model on every Nth chunk the Rust decoder fully accepts (it always runs on "
                         "the others; with --python-only, on every Nth chunk)")
    ap.add_argument("--rust-only", action="store_true", help="run the Rust decoder alone (no Python model)")
    ap.add_argument("--limit-chunks", type=int, default=None, help="classify at most N BCPack chunks")
    ap.add_argument("--no-vz-check", action="store_true", help="skip the second Python decode without virtual zero bytes")
    args = ap.parse_args(argv)
    if args.python_only and args.rust_only:
        print("--python-only and --rust-only exclude each other", file=sys.stderr)
        return 2
    if args.python_stride < 1 or (args.limit_chunks is not None and args.limit_chunks < 0):
        print("--python-stride must be at least 1 and --limit-chunks not negative", file=sys.stderr)
        return 2
    if not os.path.isdir(args.game_dir):
        print("the game folder does not exist", file=sys.stderr)
        return 2
    if inside(args.work, args.game_dir):
        print("--work must not be inside the game folder", file=sys.stderr)
        return 2
    rs, rs_note = None, "not used (--python-only)"
    if not args.python_only:
        try:
            rs = RsDecoder(args.rs_lib or default_rs_library())
            rs_note = "loaded"
        except RsUnavailable as exc:
            print(f"Rust decoder unavailable: {exc}", file=sys.stderr)
            return 77
    counts, seconds = scan(args.game_dir, rs, args.python_stride, args.limit_chunks, not args.no_vz_check,
                           not args.rust_only)
    if counts["pools_found"] == 0:
        print("no texture pool found below the game folder", file=sys.stderr)
        return 2
    text = render(counts, seconds, rs_note)
    print(text)
    os.makedirs(args.work, exist_ok=True)
    with open(os.path.join(args.work, "bcpack_model_count.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"counts": dict(sorted(counts.items())), "seconds": round(seconds, 1), "rust": rs_note,
                   "python_stride": args.python_stride}, f, indent=2)
        f.write("\n")
    return 1 if counts["symbol_disagreements"] else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
