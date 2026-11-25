//! Low-rate system memory samples for runtime pipeline workers.

use std::ptr::null_mut;
use std::time::{Duration, Instant};
use windows_sys::Win32::System::Performance::*;
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1024 * 1024;
const PAUSE_MIB: u64 = 1536;
const RESUME_MIB: u64 = 2048;

pub(super) fn paused(previous: bool, available: Option<u64>) -> bool {
    available.map_or(previous, |mib| {
        if previous { mib < RESUME_MIB } else { mib < PAUSE_MIB }
    })
}

pub(super) fn available_mib() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a local buffer with its size set for the Win32 call.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullAvailPhys / MIB)
}

struct PagesInput {
    query: PDH_HQUERY,
    counter: PDH_HCOUNTER,
}

impl PagesInput {
    fn open() -> Option<Self> {
        let mut this = Self { query: null_mut(), counter: null_mut() };
        // SAFETY: PDH owns the query and counter until Drop. The English
        // counter path is independent of the Windows display language.
        unsafe {
            if PdhOpenQueryW(std::ptr::null(), 0, &mut this.query) != 0
                || PdhAddEnglishCounterW(
                    this.query,
                    windows_sys::w!("\\Memory\\Pages Input/sec"),
                    0,
                    &mut this.counter,
                ) != 0
            {
                return None;
            }
            PdhCollectQueryData(this.query);
        }
        Some(this)
    }

    fn sample(&self) -> Option<f64> {
        let mut value = PDH_FMT_COUNTERVALUE::default();
        // SAFETY: live handles used by their owning sampler thread only.
        let rate = unsafe {
            if PdhCollectQueryData(self.query) != 0
                || PdhGetFormattedCounterValue(self.counter, PDH_FMT_DOUBLE, null_mut(), &mut value) != 0
                || !matches!(value.CStatus, PDH_CSTATUS_VALID_DATA | PDH_CSTATUS_NEW_DATA)
            {
                return None;
            }
            value.Anonymous.doubleValue
        };
        (rate.is_finite() && rate >= 0.0).then_some(rate)
    }
}

impl Drop for PagesInput {
    fn drop(&mut self) {
        if !self.query.is_null() {
            // SAFETY: the query owns and closes the counter as well.
            unsafe { PdhCloseQuery(self.query) };
        }
    }
}

pub(super) fn sample_loop(
    mut paused_now: bool,
    mut observe: impl FnMut(bool),
    mut report: impl FnMut(Option<u64>, Option<f64>, bool),
) {
    let counter = PagesInput::open();
    if counter.is_none() {
        tracing::warn!("PDH Memory/Pages Input/sec unavailable; reporting physical memory only");
    }
    let mut next_report = Instant::now() + Duration::from_secs(5);
    let mut pages = 0.0;
    let mut seconds = 0.0;
    let mut last = Instant::now();
    loop {
        std::thread::sleep(Duration::from_secs(1));
        let available = available_mib();
        paused_now = paused(paused_now, available);
        observe(paused_now);
        let rate = counter.as_ref().and_then(PagesInput::sample);
        let now = Instant::now();
        let span = now.duration_since(last).as_secs_f64();
        last = now;
        if let Some(rate) = rate {
            pages += rate * span;
            seconds += span;
        }
        if now >= next_report {
            let rate = (seconds > 0.0).then(|| pages / seconds);
            report(available, rate, paused_now);
            pages = 0.0;
            seconds = 0.0;
            next_report = now + Duration::from_secs(5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_has_hysteresis_and_ignores_missing_samples() {
        assert!(!paused(false, Some(1536)));
        assert!(paused(false, Some(1535)));
        assert!(paused(true, Some(2047)));
        assert!(!paused(true, Some(2048)));
        assert!(paused(true, None));
        assert!(!paused(false, None));
    }

    #[test]
    fn physical_memory_is_available_from_windows() {
        assert!(available_mib().is_some());
    }
}
