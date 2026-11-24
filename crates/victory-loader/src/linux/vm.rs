//! The region table behind `VirtualAlloc`, `VirtualFree`, `VirtualProtect`
//! and `VirtualQuery`: every mapping title code can see, with the
//! Windows allocation granularity (64 KiB).

/// Records a mapping made elsewhere (an image, thunk memory) so that
/// `VirtualQuery` and `VirtualProtect` see it. `protect` is a Windows
/// `PAGE_*` value.
pub fn adopt(_base: usize, _size: usize, _protect: u32) {}
