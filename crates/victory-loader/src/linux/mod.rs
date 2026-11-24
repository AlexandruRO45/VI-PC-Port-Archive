//! The platform under title code on Linux: what the Windows loader and
//! kernel give a title on Windows. See `docs/linux.md`.
//!
//! - [`teb`]: each thread's TEB at its GS base, last error.
//! - [`thread`]: guest threads.
//! - [`seh`]: exception dispatch and unwinding.
//! - [`vm`]: the region table behind `VirtualAlloc` and friends.

pub mod seh;
pub mod teb;
pub mod thread;
pub mod vm;
