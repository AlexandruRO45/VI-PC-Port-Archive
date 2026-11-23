# navmesh_native

Optional Rust kernels for the `--deep` walks of [`tools/navmesh_decode.py`](../navmesh_decode.py): the
quadtree walk, the polygon chain, the obstacle volume pairs and the grid hash-table checks (value bijection,
octree keys, inner payload words). The crate has no dependencies.

```sh
cargo build --release --manifest-path tools/navmesh_native/Cargo.toml
```

`tools/_navmesh_native.py` loads `target/release/navmesh_native.dll` (`libnavmesh_native.so`,
`libnavmesh_native.dylib`) with `ctypes`. When the library is missing, fails to load, reports another ABI or
declines a call, or when `LEONIDA_NAVMESH_NATIVE=0` is set, `navmesh_decode.py` runs its Python reference, which
stays the specification. Both paths give the same report, byte for byte, including the insertion order of every
counter.

`tests/test_navmesh_native.py` checks that on seeded synthetic inputs (1,500 per kernel, plus whole deep scans of
an archive and a carved folder), the fallback cases, and prints kernel timings.
