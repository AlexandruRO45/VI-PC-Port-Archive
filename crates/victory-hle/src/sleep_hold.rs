//! Opt-in clock adjustment for the title watchdog's two one-second sleeps.

use std::sync::OnceLock;
use std::time::Duration;

use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
use windows_sys::Win32::System::Threading::Sleep as NativeSleep;

const BUILD: &str = "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93";
const WATCHDOG_RETURNS: [usize; 2] = [0x2215b0b, 0x2215cb0];
const API_SET: &str = "api-ms-win-core-synch-l1-2-0.dll";
const TICK_MS: u32 = 1000;
// Leave margin below the watchdog's 2 s wall-clock discontinuity check.
const MAX_EXTENSION: Duration = Duration::from_millis(500);

static TITLE: OnceLock<usize> = OnceLock::new();

fn eligible(option: Option<&str>, hash: &str) -> bool {
    option == Some("1") && hash == BUILD
}

/// Configures Sleep before the executable's imports are bound.
pub fn configure(base: usize, original_hash: &str) {
    let option = std::env::var("VICTORY_SLEEP_HOLD").ok();
    if eligible(option.as_deref(), original_hash) {
        victory_gpu::pipeline::track_pipeline_holds();
        if TITLE.set(base).is_ok() {
            tracing::info!("title watchdog Sleep holds enabled");
        }
    } else if option.as_deref() == Some("1") {
        tracing::warn!("title watchdog Sleep holds unavailable for this build");
    }
}

fn matches_import(dll: &str, symbol: &str) -> bool {
    dll.eq_ignore_ascii_case(API_SET) && symbol == "Sleep"
}

/// Only overrides a Sleep import that the resolver already allows on the host.
pub(crate) fn resolve(dll: &str, symbol: &str) -> Option<usize> {
    (TITLE.get().is_some() && matches_import(dll, symbol)).then_some(sleep as *const () as usize)
}

fn is_watchdog_tick(ms: u32, base: usize, caller: usize) -> bool {
    ms == TICK_MS && caller.checked_sub(base).is_some_and(|rva| WATCHDOG_RETURNS.contains(&rva))
}

// Keep this frame so skipping one frame in RtlCaptureStackBackTrace finds
// the title return PC. The import thunk jumps here, leaving no extra frame.
#[inline(never)]
extern "system" fn sleep(ms: u32) {
    if ms == TICK_MS && let Some(&base) = TITLE.get() {
        let mut caller = std::ptr::null_mut();
        // SAFETY: we request one return address and supply room for one.
        let count = unsafe { RtlCaptureStackBackTrace(1, 1, &mut caller, std::ptr::null_mut()) };
        if count == 1 && is_watchdog_tick(ms, base, caller as usize) {
            let extension = victory_gpu::pipeline::sleep_with_pipeline_holds(
                Duration::from_millis(u64::from(ms)),
                MAX_EXTENSION,
            );
            if !extension.is_zero() {
                tracing::debug!(extension_ms = extension.as_millis(), "extended title watchdog tick for pipeline waits");
            }
            return;
        }
    }
    // SAFETY: forwards the original duration to the host API, not our IAT hook.
    unsafe { NativeSleep(ms) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_exact_title_identity_and_opt_in() {
        assert!(eligible(Some("1"), BUILD));
        for option in [None, Some("0"), Some("true")] {
            assert!(!eligible(option, BUILD));
        }
        assert!(!eligible(Some("1"), &BUILD[..16]));
        assert!(!eligible(Some("1"), "different build"));
    }

    #[test]
    fn only_the_two_verified_returns_and_one_second_sleep_qualify() {
        let base = 0x140000000;
        for rva in WATCHDOG_RETURNS {
            assert!(is_watchdog_tick(1000, base, base + rva));
            assert!(!is_watchdog_tick(1000, base, base + rva - 1));
            for ms in [0, 1, 99, 100, 999, 1001, u32::MAX] {
                assert!(!is_watchdog_tick(ms, base, base + rva));
            }
        }
        assert!(!is_watchdog_tick(1000, base, base - 1));
        assert!(matches_import(API_SET, "Sleep"));
        assert!(!matches_import(API_SET, "SleepEx"));
        assert!(!matches_import("kernel32.dll", "Sleep"));
    }
}
