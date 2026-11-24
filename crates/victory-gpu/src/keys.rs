//! Keys the title's own window saw pressed, from its window procedure.
//!
//! A window receives `WM_KEYDOWN`/`WM_KEYUP` for a person's keyboard and for
//! input another process injects with `SendInput`, and it receives them only
//! while it is the foreground window — the same gate the async key state has.
//! The async state is what a person's keyboard writes; an injected key does not
//! appear there at all. Measured, on a focused window of our own: one
//! `WM_KEYDOWN` arrived and zero async-state samples saw the key down.
//!
//! So the input source unions the two. A press from a bot and a press from a
//! person end in the same state, go through the same translation, and reach the
//! game by the same route — which is what keeps a harness from passing a screen
//! a person could not.

use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

/// Down and not yet released, in no particular order.
static HELD: Mutex<BTreeSet<u8>> = Mutex::new(BTreeSet::new());

/// The window was told this key is down. Repeats are harmless: the set holds
/// one entry per key.
pub(crate) fn down(virtual_key: u8) {
    let _ = held().insert(virtual_key);
}

/// The window was told this key is up.
pub(crate) fn up(virtual_key: u8) {
    let _ = held().remove(&virtual_key);
}

/// The window lost the foreground, so it will not hear the releases of keys
/// still down: a session that alt-tabs with a key held would keep it held.
pub(crate) fn release_all() {
    held().clear();
}

/// The virtual keys the window has been told are down and has not been told
/// are up.
pub fn pressed() -> Vec<u8> {
    held().iter().copied().collect()
}

fn held() -> MutexGuard<'static, BTreeSet<u8>> {
    HELD.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one test that touches the shared set, so nothing else in this crate
    /// can make it pass or fail.
    #[test]
    fn a_key_is_held_from_its_down_to_its_up() {
        release_all();
        down(0x0D);
        assert_eq!(pressed(), vec![0x0D]);
        // A repeat of the same key is one entry.
        down(0x0D);
        assert_eq!(pressed(), vec![0x0D]);
        up(0x0D);
        assert!(pressed().is_empty(), "{:?}", pressed());
        // A release for a key nobody pressed changes nothing.
        up(0x41);
        assert!(pressed().is_empty(), "{:?}", pressed());
        // Losing the foreground releases what is still down.
        down(0x41);
        release_all();
        assert!(pressed().is_empty(), "{:?}", pressed());
    }
}
