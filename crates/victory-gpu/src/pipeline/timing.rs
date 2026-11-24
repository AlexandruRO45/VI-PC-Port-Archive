//! Where the time of a host pipeline goes, printed once a second
//! (`VICTORY_PIPELINE_TIMING=1`, diagnostics builds only).
//!
//! The queue that never drains is either slow work or slow handing: a
//! pipeline waits in the queue, then its shaders are translated (and a
//! translation another thread is already making is waited for), then a root
//! signature is made, then the library is asked for the PSO and the driver
//! compiles it, then it is stored back. Those are five different fixes, so
//! each is counted and timed on its own, and the line prints the rate they
//! add up to and what the queue would take to drain at it.
//!
//! Worker busy-seconds are counted too: the workers run below normal
//! priority, so a rate that is low because the title's own threads have the
//! processors is a different problem from a rate that is low because the
//! driver's compile is slow.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Duration;

/// One counter per stage of a pipeline's making.
#[derive(Clone, Copy)]
pub(crate) enum Part {
    /// Queued to claimed by a worker.
    Queue,
    /// Translation work this thread did.
    Translate,
    /// Waiting for a translation another thread was making.
    TranslateWait,
    /// `CreateRootSignature`.
    RootSignature,
    /// Asked the library for the PSO (a load attempt, hit or miss).
    LibraryLoad,
    /// The driver compiled it (`CreateGraphicsPipelineState` and friends).
    Create,
    /// Stored back into the library.
    Store,
    /// A whole pipeline, in its worker.
    Whole,
}

const PARTS: usize = 8;
static US: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
static COUNT: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];
/// Pipelines a worker finished with and without a host pipeline.
static MADE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// The longest single queue wait since the start, and since the last line.
static LONGEST: AtomicU64 = AtomicU64::new(0);
/// Workers started (the busy fraction is against these).
static WORKERS: AtomicU64 = AtomicU64::new(0);

/// Whether the line prints (`VICTORY_PIPELINE_TIMING=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_TIMING").is_some_and(|v| v == "1"))
}

/// `part` of a pipeline took `elapsed`.
pub(crate) fn add(part: Part, elapsed: Duration) {
    let us = elapsed.as_micros() as u64;
    US[part as usize].fetch_add(us, Relaxed);
    COUNT[part as usize].fetch_add(1, Relaxed);
}

/// A whole pipeline was finished (with a host pipeline or without).
pub(crate) fn made(ok: bool) {
    MADE[usize::from(!ok)].fetch_add(1, Relaxed);
}

/// A pipeline waited `us` before a worker claimed it.
pub(crate) fn queued(us: u64) {
    add(Part::Queue, Duration::from_micros(us));
    LONGEST.fetch_max(us, Relaxed);
}

/// The workers were started, `workers` of them.
pub(crate) fn workers(workers: u64) {
    WORKERS.store(workers, Relaxed);
}

/// Starts the once-a-second line if the switch is on and it is not running.
pub(crate) fn report_thread() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        if !on() {
            return;
        }
        tracing::info!("host pipeline timing: a line a second, by stage");
        let spawned = std::thread::Builder::new()
            .name("victory-pipeline-timing".into())
            .spawn(|| {
                // The counters start again from here: the line is a rate.
                let _ = Snapshot::take();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    report(&Snapshot::take(), Duration::from_secs(1));
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no host pipeline timing line");
        }
    });
}

struct Snapshot {
    us: [u64; PARTS],
    count: [u64; PARTS],
    made: [u64; 2],
}

impl Snapshot {
    fn take() -> Snapshot {
        let mut us = [0; PARTS];
        let mut count = [0; PARTS];
        let mut made = [0; 2];
        for i in 0..PARTS {
            us[i] = US[i].swap(0, Relaxed);
            count[i] = COUNT[i].swap(0, Relaxed);
        }
        for i in 0..2 {
            made[i] = MADE[i].swap(0, Relaxed);
        }
        Snapshot { us, count, made }
    }
}

/// Milliseconds per count of `part`, and the count.
fn mean(s: &Snapshot, part: Part) -> (f64, u64) {
    let n = s.count[part as usize];
    if n == 0 {
        (0.0, 0)
    } else {
        (s.us[part as usize] as f64 / n as f64 / 1e3, n)
    }
}

fn report(now: &Snapshot, over: Duration) {
    let made = now.made[0];
    let refused = now.made[1];
    let secs = over.as_secs_f64();
    let (queue, queued) = mean(now, Part::Queue);
    let (translate, translated) = mean(now, Part::Translate);
    let (wait, waits) = mean(now, Part::TranslateWait);
    let (root, _) = mean(now, Part::RootSignature);
    let (load, loads) = mean(now, Part::LibraryLoad);
    let (create, creates) = mean(now, Part::Create);
    let (store, _) = mean(now, Part::Store);
    let (whole, wholes) = mean(now, Part::Whole);
    let waiting = super::make::waiting();
    let workers = WORKERS.load(Relaxed);
    // Worker-seconds spent in a pipeline against worker-seconds available.
    let busy = if workers == 0 {
        0.0
    } else {
        (now.us[Part::Whole as usize] as f64 / 1e6) / (secs * workers as f64) * 100.0
    };
    // What the queue would take to drain at this rate, adding nothing.
    let drain = if made == 0 {
        f64::INFINITY
    } else {
        waiting as f64 / (made as f64 / secs)
    };
    let longest = LONGEST.swap(0, Relaxed) as f64 / 1e3;
    tracing::info!(
        "pipelines {made_per_s:.1}/s made ({made} made, {refused} refused), \
         {translated_per_s:.1} shaders translated/s; \
         per pipeline: whole {whole:.1} ms ({wholes}), queue {queue:.1} ms ({queued}, longest {longest:.0} ms), \
         translate {translate:.1} ms ({translated}), waiting for a translation {wait:.1} ms ({waits}), \
         root {root:.1} ms, library load {load:.1} ms ({loads}), create {create:.1} ms ({creates}), store {store:.1} ms; \
         {waiting} queued, {workers} workers {busy:.0}% busy, drains in {drain:.0} s",
        made_per_s = made as f64 / secs,
        translated_per_s = translated as f64 / secs,
    );
}
