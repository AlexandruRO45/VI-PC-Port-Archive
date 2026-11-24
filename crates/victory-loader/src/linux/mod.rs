//! The platform under title code on Linux: what the Windows loader and
//! kernel give a title on Windows. See `docs/linux.md`.
//!
//! - [`teb`]: each thread's TEB at its GS base, last error.
//! - [`vm`]: the region table behind `VirtualAlloc` and friends.

pub mod teb;
pub mod vm;
