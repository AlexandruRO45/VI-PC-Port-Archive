# Optional native groom kernel

Build the standard-library-only Rust `cdylib` from the repository root:

```sh
cargo build --release --locked --offline --manifest-path tools/smh_groom_native/Cargo.toml
cargo test --release --locked --offline --manifest-path tools/smh_groom_native/Cargo.toml
python -I tests/test_smh_groom_native.py
```

On CPython 3.12–3.14, `smh_groom.py` loads the platform library from this crate's
`target/release` directory. Set `LEONIDA_GROOM_NATIVE=0` to use the Python
reference. Python also handles a missing or incompatible library, unsupported
input shapes, and non-finite values or float32 overflow refused by Rust.

The blend kernel accepts scattered roots and blends, trims, and calculates
UVs in binary64. Python controls scattering and the output dictionary.
The geometry kernel builds tube, ribbon, curve, and line-strip buffers,
performs the up-axis conversion, and writes little-endian float32 attributes
and uint16 or uint32 indices. Python supplies tube ring trigonometry and
writes the GLB accessors, materials, and metadata. The norm implementation
uses compensated sums of squares and the final square-root correction to
match `math.hypot` at rounding boundaries.

The parity tests build every input from scratch. They compare binary64 strand
values, positions, normals, indices, accessor bounds, and complete GLBs, and
exercise the Python fallback.
