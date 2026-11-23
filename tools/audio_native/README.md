# audio_native

Optional Rust kernels for [`tools/rsc8_audio.py`](../rsc8_audio.py): the AWC (`ADAT`) bank header parse
(`parse_adat_header_bytes`) and the word loops of the `0x5a`, `0xd9` and `0xbd` companion-table decoders
(`decode_grid20_table`, `decode_index_runs`, `decode_marker_records`, and the region-cue check before
`pair_marker_regions`). The crate has no dependencies.

```sh
cargo build --release --manifest-path tools/audio_native/Cargo.toml
```

`tools/_audio_native.py` loads `target/release/audio_native.dll` (`libaudio_native.so`, `libaudio_native.dylib`)
with `ctypes`. When the library is missing, fails to load, reports another ABI or declines a call, or when
`LEONIDA_AUDIO_NATIVE=0` is set, `rsc8_audio.py` runs its Python reference, which stays the specification. Tables
shorter than the adapter's `MIN_*` sizes also stay in Python, where the call overhead would cost as much as the loop.
Both paths give the same result, including the first-seen order of the marker names.

`tests/test_audio_native.py` checks that on seeded synthetic inputs (1,500 per kernel, valid and broken, plus whole
banks through `decode_awc_bank`), the fallback cases, and with `LEONIDA_AUDIO_NATIVE_BENCH=1` prints kernel timings.
