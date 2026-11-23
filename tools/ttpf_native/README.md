# Native TTPF kernels

An optional Rust build of the hot loops in `tools/ttpf_mips.py`. `src/lib.rs` ports, step for step:

- the element copy of `detile` and `detile_volume_slice` (the X/Y bit tables and row bases are the ones
  `ttpf_mips.py` computes and passes in);
- the BC1, BC2/BC3 colour, BC4, BC5, BC6H (UF16 and SF16) and BC7 block decoders behind `decode_pixels`;
- `block_means` and `predict_child_means` for BC mips.

Every integer step and every double operation is the Python one, in the same order, so the detiled bytes and
the IEEE bits of every decoded or averaged double are identical (including -0.0). The BPTC partition and anchor
tables and the BC6H header bit plans are not copied: `ttpf_mips.py` passes `bcpack_tables_bptc.load_bc7()` and
`bcpack._BC6H_INFO` / `bcpack._bc6h_plan`. The crate has no dependencies.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/ttpf_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/ttpf_native/Cargo.toml
python -I tests/test_ttpf_native.py
```

`ttpf_mips.py` loads `target/release/ttpf_native.dll` (`libttpf_native.so`, `libttpf_native.dylib`) when it is
there and runs the Python code otherwise. It also runs the Python code for a call the library refuses (an element
outside the tiled allocation, a short block payload, a buffer type other than bytes, bytearray or a byte
memoryview), so those inputs keep their Python result or error. `LEONIDA_TTPF_NATIVE=0` turns the native path off.

How the pieces fit:

- `decode_pixels` still returns a list of `(r, g, b)` tuples. For BC4, BC5, BC6H and BC7 the library decodes the
  blocks and returns each distinct channel value and each distinct pixel once, so equal pixels share one tuple and
  one float, as the Python BC1/BC4 decoders already do. BC1-BC3 keep the Python decoder: it already shares a tuple
  per palette entry, and building the list from native output is no faster.
- The list is a `list` subclass that remembers the blocks it was decoded from. `block_means` and
  `predict_child_means` can decode those blocks again in Rust and average them without reading the tuples back.
  Child prediction follows the visible mip bounds even when the stored block grid is larger. Editing the list
  drops the remembered blocks and sends both operations through Python. Older ABI 1 libraries without the
  prediction export still provide the original kernels.
- The first time each block kind is used, `ttpf_mips.py` decodes and averages 64 seeded random blocks (every BC6H
  mode code, every BC7 mode and the reserved one, both BC1 colour orders) both ways, including child prediction
  when the library has it, and keeps that kind in Python
  if any double differs. A change to a Python decoder that is not made here too therefore turns the native path
  off instead of giving different numbers; `tests/test_ttpf_native.py` fails on it as well.

`tests/test_ttpf_native.py` compares the two paths on random synthetic data: 2D mip chains of every element size
with 4 KiB and 64 KiB tiles, including 1x1, 3x700 and 1023x5 surfaces and short allocations; 3D volume slices of
every element size, including 1x1x1 and 33x1x2; every BC format at 1x1, 1x37, 37x1, 2x3, 13x7 and 64x64 blocks,
BC6H extreme endpoints under every mode code, bytearray and memoryview inputs and truncated payloads; and
`block_means` under several crops, block sizes and grids, and child prediction for clipped and padded mips. It also
checks the fallbacks. Its comparisons skip when
the library is not built.

## Speed

Best of five (three for the slower rows) on an AMD Ryzen (Zen 3, family 25 model 33), Windows 11, Python 3.12,
Rust 1.100, random synthetic data; Python is the same code with `LEONIDA_TTPF_NATIVE=0`.

| Call | Python | Native | Speed-up |
|---|---:|---:|---:|
| `detile` BC1 1024x1024, 4 KiB tiles | 7.35 ms | 0.385 ms | 19x |
| `detile` BC7 1024x1024, 64 KiB tiles | 7.65 ms | 0.593 ms | 13x |
| `detile` RGBA8 1024x1024, 4 KiB tiles | 60.0 ms | 2.23 ms | 27x |
| `detile_volume` 256x256x32, 8-byte voxels | 272 ms | 16.5 ms | 16x |
| `block_means` of a BC1 512x512 decode | 62.9 ms | 5.03 ms | 13x |
| `block_means` of a BC7 512x512 decode | 69.8 ms | 6.60 ms | 11x |
| `decode_pixels` BC7 512x512, 512 distinct blocks | 386 ms | 39.4 ms | 10x |
| `decode_pixels` BC6H UF16 512x512, 512 distinct blocks | 563 ms | 45.4 ms | 12x |
| `decode_pixels` BC7 512x512, every block random | 370 ms | 82.6 ms | 4.5x |
| `decode_pixels` BC6H SF16 512x512, every block random | 568 ms | 114 ms | 5.0x |
| `decode_pixels` BC4 512x512, every block random | 48.6 ms | 21.5 ms | 2.3x |
| `decode_pixels` BC1 512x512 (Python decoder in both) | 39.6 ms | 47.5 ms | - |

A decoded surface is a Python list with one tuple per pixel, so `decode_pixels` costs at least the time to build
that list: about 0.1-0.3 µs per pixel when every pixel differs (the random rows above), much less when pixels
repeat. `detile` returns bytes and `block_means` one tuple per block, so they gain more.
