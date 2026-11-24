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
    // Whether the message arrives at all is the question a keyed run has to
    // answer before it can say anything about the game's input: a press that
    // lands in the window and never reaches here is a different bug from a
    // press the game reads and ignores.
    #[cfg(feature = "diagnostics")]
    tracing::debug!(target: "victory_gpu::keys", virtual_key, "window key down");
    let _ = held().insert(virtual_key);
}

/// The window was told this key is up.
pub(crate) fn up(virtual_key: u8) {
    #[cfg(feature = "diagnostics")]
    tracing::debug!(target: "victory_gpu::keys", virtual_key, "window key up");
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

/// The low-level hook that sees the keys the title's own window would think
/// it owned.
///
/// The console creates `sgaWindowClass` and puts our caption on it, and
/// `WM_KEYDOWN` reaches *that* window procedure, not ours: six presses landed
/// on it with focus proven and not one arrived at [`down`] (measured, 2026-11-24).
/// A `WH_KEYBOARD_LL` hook sits in front of every window, so an injected key
/// and a person's key take the same route into the same set — which is the
/// property that keeps a harness from passing a screen a person could not.
///
/// It reads only while the foreground window belongs to this process: a key
/// pressed in another application is not the title's input, and a hook that
/// collected it would be reading the person's typing.
#[cfg(windows)]
pub(crate) mod hook {
    use windows_sys::Win32::Foundation::{LPARAM, WPARAM};
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, GetForegroundWindow, GetWindowThreadProcessId, KBDLLHOOKSTRUCT,
        SetWindowsHookExW, UnhookWindowsHookEx, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN,
        WM_SYSKEYUP,
    };

    /// Whether a key event belongs to this process: the foreground window's
    /// process is ours, and a foreground nobody owns is nobody's.
    pub(crate) fn ours(foreground_pid: u32, this_pid: u32) -> bool {
        foreground_pid != 0 && foreground_pid == this_pid
    }

    /// The installed hook. Dropping it removes it: a low-level hook that
    /// outlives its owner keeps the machine paying for every key press.
    pub(crate) struct Installed(*mut core::ffi::c_void);

    impl Drop for Installed {
        fn drop(&mut self) {
            // SAFETY: a hook this value installed and no one removed.
            unsafe { UnhookWindowsHookEx(self.0) };
        }
    }

    /// Installs the hook on this thread, which must have the message loop the
    /// hook is served by.
    pub(crate) fn install() -> Installed {
        // SAFETY: our own thread, a callback that outlives the hook, and no
        // module handle because the callback is ours.
        let installed = unsafe {
            SetWindowsHookExW(
                WH_KEYBOARD_LL as i32,
                Some(callback),
                std::ptr::null_mut(),
                0,
            )
        };
        // A hook that failed to install is the whole reason a run can press
        // into a window and see nothing, so the outcome is recorded rather
        // than assumed.
        #[cfg(feature = "diagnostics")]
        tracing::debug!(
            target: "victory_gpu::keys",
            installed = !installed.is_null(),
            last_error = unsafe { windows_sys::Win32::Foundation::GetLastError() },
            "keyboard hook"
        );
        Installed(installed)
    }

    /// SAFETY: called by the system with a `KBDLLHOOKSTRUCT` in `lparam` when
    /// `code` is non-negative.
    unsafe extern "system" fn callback(code: i32, wparam: WPARAM, lparam: LPARAM) -> isize {
        if code >= 0 && lparam != 0 {
            let mut pid = 0;
            // SAFETY: reading the foreground window's process.
            unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut pid) };
            // SAFETY: this process's own id.
            let ours_now = ours(pid, unsafe { GetCurrentProcessId() });
            if ours_now {
                // SAFETY: the system's own structure for this event.
                let key = unsafe { (*(lparam as *const KBDLLHOOKSTRUCT)).vkCode } as u8;
                match wparam as u32 {
                    WM_KEYDOWN | WM_SYSKEYDOWN => super::down(key),
                    WM_KEYUP | WM_SYSKEYUP => super::up(key),
                    _ => {}
                }
            } else {
                // The foreground left us, so the releases will not: a key held
                // through an alt-tab must not stay held into the next run.
                super::release_all();
            }
        }
        // SAFETY: the next hook in the chain, as the API requires.
        unsafe { CallNextHookEx(std::ptr::null_mut(), code, wparam, lparam) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one test that touches the shared set, so nothing else in this crate
    /// can make it pass or fail.
    /// The filter the hook runs before it records anything: a key pressed in
    /// another application is not the title's input, and a foreground nobody
    /// owns is nobody's.
    #[cfg(windows)]
    #[test]
    fn a_key_is_taken_only_while_the_foreground_is_this_process() {
        assert!(super::hook::ours(4321, 4321));
        assert!(!super::hook::ours(4321, 1234), "another process's key is not ours");
        assert!(!super::hook::ours(0, 1234), "no foreground is nobody's");
    }

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
