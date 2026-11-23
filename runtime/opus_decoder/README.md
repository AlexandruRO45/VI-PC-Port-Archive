# Native Opus packet decoder

This C ABI decodes one raw Opus packet per call at 48 kHz, with one or two interleaved float32 output channels. It uses the BSD-3-Clause `opusdec` crate at the Git revision already pinned by the sibling XMA2 decoder.

The decoder's port attribution and redistribution terms are in its [NOTICE](https://github.com/omonkeyz/xbox-emulator/blob/27122f7dfcceb9a899555362ee0dde2d530e033c/crates/compat/opus/NOTICE) and [COPYING](https://github.com/omonkeyz/xbox-emulator/blob/27122f7dfcceb9a899555362ee0dde2d530e033c/crates/compat/opus/COPYING).

Create one handle per logical stream. Pass the current destination address and its capacity in bytes on each submission. The caller can rotate destinations independently of stream identity. Publish a successful completion with the returned byte count after decoding succeeds. On insufficient capacity the caller can retry the same packet on the same handle. The caller owns stream scheduling, the command and result records, connection setup and any later PCM conversion. See `leonida_opus_decoder.h` for buffer and lifetime requirements.

Build with `cargo build --locked --manifest-path runtime/opus_decoder/Cargo.toml`. This produces both a static library for native linking and a shared library for the ABI test. On Windows run the comparison from the repository root:

```powershell
python -I runtime/opus_decoder/tests/compare_ffmpeg.py runtime/opus_decoder/target/debug/leonida_opus_decoder.dll
```

On Linux pass `runtime/opus_decoder/target/debug/libleonida_opus_decoder.so`. The comparison needs FFmpeg with its libopus encoder and native Opus decoder. It creates two original 20 ms packets per stream from generated tones, compares every output sample, swaps the destination buffers between packets and checks that a short-buffer retry preserves the signal. All test audio is generated in a temporary directory.
