//! Platform-level data formats, shared by every Xbox Series title.
//!
//! Each module is pure (bytes in, bytes out, no I/O): [`texture`] (the
//! GPU's tiled texture layouts and copies to and from linear rows),
//! [`bcpack`] (BC1-BC7 block streams as DirectStorage packs them) and
//! [`xma`] (XMA2 packets and frames, decoded to PCM). Opus streams and the
//! zlib streams used by hardware decompression are decoded where they are
//! read, in `victory-hle` (`apu` and `dstorage`), by the `opus-pure` and
//! `miniz_oxide` crates.

pub mod bcpack;
pub mod texture;
pub mod xma;
