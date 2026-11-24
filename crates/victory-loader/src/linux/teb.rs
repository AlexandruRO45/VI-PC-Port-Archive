//! Each thread that runs title code has a TEB at its GS base, as on
//! Windows (`gs:[0x30]` self, `gs:[0x58]` static TLS, `gs:[0x68]` last
//! error).

use std::cell::Cell;

thread_local! {
    // ponytail: until threads get a TEB, last error lives here; then it moves to gs:[0x68].
    static LAST_ERROR: Cell<u32> = const { Cell::new(0) };
}

/// `GetLastError` of the calling thread.
pub fn last_error() -> u32 {
    LAST_ERROR.get()
}

/// `SetLastError` of the calling thread.
pub fn set_last_error(code: u32) {
    LAST_ERROR.set(code);
}

thread_local! {
    // ponytail: until threads get a TEB, TLS slots live here; then they are gs:[0x1480].
    static TLS_SLOTS: std::cell::UnsafeCell<[usize; 64]> = const { std::cell::UnsafeCell::new([0; 64]) };
}

/// The calling thread's `TlsSlots` (TEB +0x1480): the 64 values behind
/// `TlsGetValue`, valid for as long as the thread lives.
pub fn tls_slots() -> *mut usize {
    TLS_SLOTS.with(|s| s.get().cast())
}
