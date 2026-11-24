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
