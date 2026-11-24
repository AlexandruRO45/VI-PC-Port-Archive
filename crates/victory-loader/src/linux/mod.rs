//! The platform under title code on Linux: what the Windows loader and
//! kernel give a title on Windows. See `docs/linux.md`.
//!
//! - [`image`]: mapping PE images, binding imports, entry points.
//! - [`teb`]: each thread's TEB at its GS base, the PEB, last error.
//! - [`thread`]: guest threads.
//! - [`tls`]: static TLS.
//! - [`seh`]: exception dispatch and unwinding.
//! - [`vm`]: the region table behind `VirtualAlloc` and friends.

pub mod image;
pub mod seh;
pub mod teb;
pub mod thread;
pub mod tls;
pub mod vm;
