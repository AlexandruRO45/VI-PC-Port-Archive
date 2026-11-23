# amd_swizzle_native

Optional native kernels for [`tools/amd_swizzle.py`](../amd_swizzle.py): the element offset grid of `offsets`,
the element gather of `detile`, and the neighboring color differences used to score candidate layouts. The
crate uses only the Rust standard library and builds a `cdylib` loaded through `ctypes`.

```powershell
cargo build --release --manifest-path tools/amd_swizzle_native/Cargo.toml
cargo test --release --manifest-path tools/amd_swizzle_native/Cargo.toml
python -I tests/test_amd_swizzle_native.py   # compares it with the numpy code (needs numpy)
```

The library is found at `tools/amd_swizzle_native/target/release/` (`amd_swizzle_native.dll`,
`libamd_swizzle_native.so` or `libamd_swizzle_native.dylib`). Nothing else changes: without it, `amd_swizzle.py`
runs its numpy code exactly as before.

## What it computes

Each in-block address bit `i` is `parity(x & xmask[i]) ^ parity(y & ymask[i])`, so the byte offset of element
`(x, y)` is `(row[y] ^ col[x]) + (y / bh) * blocks_x * block_bytes + (x / bw) * block_bytes`, with `row` and
`col` built per axis as `_axis_offsets` does. `offsets` returns that grid as int64; `detile` copies element
`offset / bpe` of the source for every cell (numpy's shifted `_grid(..., shift)` and `_grid(..., 0) // bpe` both
reduce to that) and returns `None` when one lies past the data. Grids of 2^15 elements or more are split by rows
over up to 16 threads (one per 2^17 elements); `ctypes` releases the GIL for the call.

Candidate scoring computes the same horizontal and vertical RGB differences as `feature` and `tile_probe.score`.
It writes each axis in NumPy's row-major order as float32. NumPy still takes the two means and adds them in the
same order, preserving rounding and ranking. ABI 1 libraries without the new export still provide offset and
detile acceleration, with NumPy handling the entire score.

## When the numpy code runs instead

- `LEONIDA_AMD_SWIZZLE_NATIVE=0`, or the library is missing, fails to load, or reports another ABI version;
- the first-use self-check finds any difference in 24 seeded offset/detile layouts or the seeded score checks
  for each supported element size;
- an argument the library does not take: an equation that is not a list/tuple of pairs of Python `int` masks in
  `[0, 2^63)`, more than 63 address bits, a non-`int` or negative size, an element larger than the block, or an
  offset that would not fit in an int64 (numpy would wrap there).

`detile` reads the data with the same `np.frombuffer(...).reshape(-1, bpe)` line before calling the library, so
inputs numpy rejects raise the same exception.
