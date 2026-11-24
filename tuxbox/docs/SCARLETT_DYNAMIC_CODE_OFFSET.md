# Bounded runtime embedded-table offsets

The compiler previously refused nonconstant SOFFSET on an embedded-code
scalar load. This change permits the compute-only case where a local
GETPC pair reaches the load unchanged, and a bounded ordinary SGPR mask
and shift prove that every dword lies inside the table, with dword
alignment. Unknown clobbers, branch entries, empty tables, overflow,
indirect control flow, packed and mesh paths remain unsupported.
It does not change existing static loads or permit arbitrary root
descriptor offsets.

The unit test uses synthetic gfx1030 instruction encodings and checks
rejection of invalid bounds, high-half clobber, unaligned offset,
multi-dword overrun and a bypass branch. For GPU readback extract the
HLSL between BEGIN_HLSL and END_HLSL from:

```sh
cargo test --target x86_64-unknown-linux-gnu --manifest-path tools/shader-compiler/Cargo.toml --lib bounded_runtime_code_offset -- --nocapture
```

Pass the extracted file to `tools/check_dynamic_code_offset.py --hlsl ...`
with the usual provider, Proton, prefix, Windows compiler-dir, work,
output and adapter arguments. The check adds only a uniform input and
readback stores to the actual emitted HLSL. It compares all 384 words
of each buffer, including input and guards. No title data is required.

The RTX 4080 control passed 64 dispatches, 24,576 words including 4,096
result words, with zero mismatches. This is a bounded synthetic pattern,
not a measured reduction in retail refusal counts. See the receipt.
