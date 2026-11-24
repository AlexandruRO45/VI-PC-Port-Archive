//! Platform-level data formats, shared by every Xbox Series title.
//!
//! Each module is pure (bytes in, bytes out, no I/O): [`texture`] (the
//! GPU's tiled texture layouts and copies to and from linear rows),
//! [`bcpack`] (BCPack block streams, as hardware decompression unpacks
//! them) and [`xma`] (the XMA2 audio codec). Planned: Opus.

pub mod bcpack;
pub mod texture;
pub mod xma;
