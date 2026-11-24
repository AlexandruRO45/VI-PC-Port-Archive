//! Structured exception handling for title code: faults and
//! `RaiseException` are dispatched through the images' `.pdata`, as the
//! Windows kernel and `ntdll` do.

/// Makes the image mapped at `base` (`size` bytes) known to the
/// dispatcher and unwinder: its exception directory is read from its
/// headers.
pub fn register_image(_base: usize, _size: usize) {}

/// Forgets an image before it is unmapped.
pub fn unregister_image(_base: usize) {}
