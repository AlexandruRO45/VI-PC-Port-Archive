# Native XDXR BVH traversal

This optional Rust library runs `trace` (closest hit in a TLAS or BLAS) and `brute_force` for
[`tools/xdxr_bvh.py`](../xdxr_bvh.py), the two loops that dominate `xdxr_bvh.py check-build` and `selftest`. It has
no crate dependencies. It repeats the Python code step by step: the same float64 operations in the same order,
Python's `min`/`max` tie and NaN rules, and the same `Memory.read` region lookup for every read. The results
(hit dict, float bits and the `stats` step count) are identical.

From the repository root:

```text
cargo build --release --locked --offline --manifest-path tools/xdxr_bvh_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/xdxr_bvh_native/Cargo.toml
python -I tests/test_xdxr_bvh_native.py
```

`xdxr_bvh.py` loads `target/release/xdxr_bvh_native.dll` (`libxdxr_bvh_native.so`, `libxdxr_bvh_native.dylib`)
when it exists. Set `LEONIDA_XDXR_BVH_NATIVE=0` to run Python only. The Python path also runs in these cases:

- the library is missing, ctypes is unavailable, or the interpreter is 32-bit;
- an input is not a plain one: a `Memory` subclass or an overridden `read`, a generator, a vector that is not three
  numbers, an int past 2**53 or an address past 2**62;
- the library declines. That covers every case where Python would raise: a read outside every buffer, an instance
  root that is not its BLAS header's, a triangle node in a TLAS, or a coordinate `f32()` cannot hold. It also
  declines a leaf whose geometry flags have the internal QUAD_PRIM or FLIP_TRI1 bits, which Python defines (#8913).

When the library declines, nothing has been changed yet, so Python returns its own result or raises its own
exception. A panic is caught at the FFI boundary, and an allocation that fails becomes a decline instead of an
abort.

`brute_force` keeps the float32 arrays of the last 32 triangle lists it has seen. Each call checks again that a
list has not changed: it compares the list to a copy of its nesting (the copy holds the same number objects) and
checks the sign of each zero coordinate.

On Windows 11 with Python 3.12, `tests/test_xdxr_bvh_native.py` measured these runs:

- `trace` over a 2,000-triangle BOX16 BLAS: 23–32x faster;
- `brute_force`: 41–47x faster;
- a `check-build`-style loop (trace plus brute force per ray, 4,000 triangles): 57x faster, 2.29 s down to 40 ms.

`selftest` gains less (8.7x), because there `build_blas` runs in Python.
