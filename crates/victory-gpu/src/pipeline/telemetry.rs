//! Phase breadcrumbs and interval counters, sampled off the render thread.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

const MAX_WORKERS: usize = 16;
static SLOTS: [AtomicU64; MAX_WORKERS] = [const { AtomicU64::new(0) }; MAX_WORKERS];
static START: OnceLock<Instant> = OnceLock::new();
thread_local! { static SLOT: Cell<Option<&'static AtomicU64>> = const { Cell::new(None) }; }

#[derive(Clone, Copy)]
#[repr(u8)]
pub(super) enum Phase {
    Build = 1,
    Shader,
    Root,
    LibraryLock,
    LibraryLoad,
    DriverCreate,
    SignatureLock,
    SignatureCreate,
    Save,
}

impl Phase {
    const LABELS: [&str; 10] = ["idle", "build", "shader", "root", "library-lock", "load", "create", "signature-lock", "signature-create", "save"];
}

pub(super) fn register(index: usize) {
    debug_assert!(index < MAX_WORKERS);
    SLOT.set(Some(&SLOTS[index]));
}

fn millis() -> u64 {
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

pub(super) struct Guard(Option<(&'static AtomicU64, u64)>);

pub(super) fn phase(phase: Phase) -> Guard {
    Guard(SLOT.get().map(|slot| {
        let previous = slot.swap((millis() << 8) | u64::from(phase as u8), Relaxed);
        (slot, previous)
    }))
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some((slot, previous)) = self.0 {
            slot.store(previous, Relaxed);
        }
    }
}

/// Occupied workers and the oldest resident wall time in each phase.
pub(super) fn snapshot() -> String {
    let now = millis();
    let mut counts = [(0, 0); Phase::LABELS.len()];
    for slot in &SLOTS {
        let data = slot.load(Relaxed);
        if data == 0 { continue; }
        let (n, age) = &mut counts[(data & 0xff) as usize];
        *n += 1;
        *age = (*age).max(now.saturating_sub(data >> 8));
    }
    counts.iter().zip(Phase::LABELS).filter(|&(&(n, _), _)| n != 0)
        .map(|(&(n, age), label)| format!("{label}:{n}/{age}ms"))
        .collect::<Vec<_>>().join(",")
}

pub(super) static MISSES: AtomicU64 = AtomicU64::new(0);
pub(super) static CREATES: AtomicU64 = AtomicU64::new(0);
pub(super) static FAILED_CREATES: AtomicU64 = AtomicU64::new(0);
pub(super) static JOBS: AtomicU64 = AtomicU64::new(0);
pub(super) static JOB_WALL_MS: AtomicU64 = AtomicU64::new(0);
pub(super) static JOB_CPU_MS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_worker_phases_restore_the_outer_phase() {
        std::thread::spawn(|| {
            static TEST_SLOT: AtomicU64 = AtomicU64::new(0);
            let slot = &TEST_SLOT;
            SLOT.set(Some(slot));
            let outer = phase(Phase::Build);
            assert_eq!(slot.load(Relaxed) as u8, Phase::Build as u8);
            {
                let _inner = phase(Phase::LibraryLoad);
                assert_eq!(slot.load(Relaxed) as u8, Phase::LibraryLoad as u8);
            }
            assert_eq!(slot.load(Relaxed) as u8, Phase::Build as u8);
            drop(outer);
            assert_eq!(slot.load(Relaxed), 0);
        }).join().unwrap();
    }
}
