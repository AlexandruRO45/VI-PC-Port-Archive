# Native candidate scan for rpf9_carve

An optional speed-up for `tools/rpf9_carve.py`. With no dependencies outside the Rust standard library, it builds
one shared library that the carve loads with `ctypes` when it is present:

```text
cargo build --release --locked --offline --manifest-path tools/rpf9_carve_native/Cargo.toml
python -I tests/test_rpf9_carve_native.py
```

Nothing else changes: the same command line, the same `index.json` and the same carved files, byte for byte. Without
the library (not built, fails to load, wrong interface version) or with `LEONIDA_RPF9_CARVE_NATIVE=0`, the carve
runs its pure-Python code, which stays the reference.

## What it does

`rpf9_next` (`src/lib.rs`) runs the part of the carve loop that finds the next stream to keep: the `ZHDR` header
search (`78 01`, `78 5e`, `78 9c`, `78 da`, `KB2` plus a lowercase letter), the Bink 2 header check and the zlib
inflate of each candidate. It works on the archive's memory map in place, so a false header inside compressed
data or noise costs a few bytes of decoding instead of a 1 MiB copy and a Python `decompressobj`. `rpf9_all_zero`
and `rpf9_byte_counts` do the gap checks; the entropy of an unread range is still summed in Python, over the
counts in `collections.Counter`'s order, so it is the same float.

Python keeps the rest: `ModelSystemPages`, the rows of the index, the progress lines and the file writes.

The inflate is written for this scan, not taken from a library. It keeps exactly the streams `zlib.decompressobj()`
inflates to their end inside the scan window, with the same bytes and the same consumed length, following zlib's
`inflate.c` and `inftrees.c` in their default build:

- the header must pass its check, use method 8 with a window of at most 32 KiB and ask for no preset dictionary;
- a code-length code must be complete; a literal/length or distance code may only be incomplete when it is one
  code of length 1, and the distance code may be empty (then any match is an error);
- at most 286 literal/length and 30 distance codes, a repeat code 16 needs a previous length, repeats stay inside
  the code lengths, and the end-of-block code must have a length;
- symbols 286, 287 and distances 30, 31 of the fixed code are errors, as is a distance before the start of the
  stream or a stored block whose length and its complement disagree;
- the Adler-32 trailer must match.

Whether a rejected stream would have been one error or another, or simply cut off by the window, does not matter:
the Python loop moves one byte on in every case.

## Safety

Every exported function checks its ranges, returns a negative value to refuse, and runs under `catch_unwind`, so a
panic never crosses into Python; the output buffer grows with `try_reserve_exact`, so a failed allocation is a
refusal too. Python takes any refusal as "do this step in Python" and goes on from the same offset.

## Tests

`cargo test` has the unit tests of the inflate and the header search. `tests/test_rpf9_carve_native.py` compares
the library with the Python code on seeded synthetic data (zlib streams of every level, strategy and window size,
hand-encoded deflate streams with codes zlib's compressor never writes, Bink 2-like headers, padding, noise, bit
flips and cut-off streams), carves whole synthetic archives both ways and compares the folders, forces refusals in
the middle of a scan, and checks the fallbacks. No game data is involved.
