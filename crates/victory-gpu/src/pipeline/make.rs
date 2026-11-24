//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// When it entered the queue, for the queue wait (diagnostics).
    #[cfg(feature = "diagnostics")]
    queued_at: AtomicU64,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            #[cfg(feature = "diagnostics")]
            queued_at: AtomicU64::new(0),
        }
    }

    fn make(&self) {
        let started = Instant::now();
        self.made
            .get_or_init(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
        note_typical(started.elapsed());
        #[cfg(feature = "diagnostics")]
        {
            super::timing::add(super::timing::Part::Whole, started.elapsed());
            super::timing::made(self.made.get().is_some_and(|made| made.is_some()));
        }
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// How long a pipeline takes to make, in microseconds: the average of the
/// recent ones, weighted towards the last, and 0 until one is done. A
/// pipeline that cannot be made within what is left of a frame's budget is
/// dropped by the caller anyway, so [`wait`] uses this to tell a wait worth
/// taking (a menu's 30 ms pipeline) from one that only costs the frame (the
/// world's 3 s stages; measured: 2.7 s of driver time per pipeline in the
/// world against 0.4 s in loading, `VICTORY_PIPELINE_TIMING=1`).
static TYPICAL_US: AtomicU64 = AtomicU64::new(0);

/// One more make took `elapsed`.
fn note_typical(elapsed: Duration) {
    let us = elapsed.as_micros() as u64;
    let previous = TYPICAL_US.load(Ordering::Relaxed);
    let next = if previous == 0 {
        us
    } else {
        (previous / 8 * 7).saturating_add(us / 8)
    };
    TYPICAL_US.store(next, Ordering::Relaxed);
}

/// Whether a wait of up to `left` is worth taking for a pipeline whose
/// making typically takes `typical` (0: nothing is known yet, so as
/// before). Twice the typical make, so a pipeline already half made still
/// pays off and one with seconds to go does not: the 500 ms of budget a
/// world frame spent on stages that take 2.7 s completed nothing.
fn worth_waiting(typical: Duration, left: Duration) -> bool {
    typical.is_zero() || typical <= left * 2
}

/// Whether a wait is skipped for a pipeline whose making is known to take
/// longer than what is left of the frame's budget. On by default;
/// `VICTORY_PIPELINE_WAIT_BY_DURATION=0` waits as before (runs up to
/// 2026-11-24 spent the whole 500 ms of the budget in every world frame and
/// completed no pipeline with it).
fn wait_by_duration() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_BY_DURATION").is_none_or(|v| v != "0"))
}

/// Waits skipped since the start.
static SKIPPED: AtomicU64 = AtomicU64::new(0);
/// Skip lines written, so that a run logs the first few and the rest is
/// counted.
static SKIP_LINES: AtomicU64 = AtomicU64::new(0);
/// The first skip lines of a run.
const SKIP_LINES_AT: u64 = 8;

/// A wait was skipped because the pipeline takes longer to make than the
/// frame has budget left: logged for the first few (the pipeline, the
/// typical make, what was left), counted for all of them. Without it a
/// skipped wait leaves no trace, and a draw that was never waited for reads
/// exactly like one that timed out.
fn note_skipped(p: &Pending, typical: Duration, left: Duration) {
    SKIPPED.fetch_add(1, Ordering::Relaxed);
    if SKIP_LINES.fetch_add(1, Ordering::Relaxed) < SKIP_LINES_AT {
        tracing::info!(
            typical_ms = typical.as_secs_f64() * 1e3,
            left_ms = left.as_secs_f64() * 1e3,
            shaders = %format!("{:?}", p.shaders),
            "no wait for a host pipeline: it takes longer to make than the frame's budget has left"
        );
    }
}

/// Waits skipped since the start, or 0 with the duration rule off.
pub(crate) fn skipped() -> u64 {
    SKIPPED.load(Ordering::Relaxed)
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let budget = wait_budget();
    if budget.is_zero() {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now).
    static SPENT: Mutex<(u64, Duration)> = Mutex::new((u64::MAX, Duration::ZERO));
    let shared = shared_budget();
    let left = {
        let mut spent = SPENT.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    if wait_by_duration() {
        let typical = Duration::from_micros(TYPICAL_US.load(Ordering::Relaxed));
        if !worth_waiting(typical, left) {
            note_skipped(p, typical, left);
            return;
        }
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    drop(held);
    let waited = started.elapsed().min(left);
    {
        let mut spent = SPENT.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        #[cfg(feature = "diagnostics")]
        if prev == NOT_QUEUED {
            p.queued_at.store(crate::host::now_ms(), Ordering::Relaxed);
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        let p = self.requested.pop_front().or_else(|| {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        })?;
        #[cfg(feature = "diagnostics")]
        {
            let queued = p.queued_at.swap(0, Ordering::Relaxed);
            let now = crate::host::now_ms();
            if queued != 0 && now >= queued {
                super::timing::queued(now - queued);
            }
        }
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: half the logical processors, 2 to 8;
/// `VICTORY_PIPELINE_WORKERS=<n>` sets it (the driver compiles in parallel
/// across threads, so the count is throughput in the loading phase, where
/// the world's first frame finds every pipeline it will ask for).
fn worker_count() -> usize {
    std::env::var("VICTORY_PIPELINE_WORKERS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|n| (1..=64).contains(n))
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8)))
}

/// Starts the workers on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count();
        #[cfg(feature = "diagnostics")]
        {
            super::timing::workers(workers as u64);
            super::timing::report_thread();
        }
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        next().make();
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame waits for the pipelines it can get and drops the draws
    /// whose pipelines are seconds away: the world's stages take 2.7 s of
    /// driver time each, so the 500 ms a frame spends waiting completed
    /// nothing (measured, 2026-11-24). A menu's 30 ms pipeline is still
    /// waited for, and with nothing known yet nothing changes.
    #[test]
    fn a_wait_is_taken_only_for_a_pipeline_that_can_arrive() {
        let budget = Duration::from_millis(500);
        // Nothing measured yet: as before, the frame waits.
        assert!(worth_waiting(Duration::ZERO, budget));
        // A menu pipeline (30 ms): the draw is worth waiting for.
        assert!(worth_waiting(Duration::from_millis(30), budget));
        // Half made inside the budget: still worth it.
        assert!(worth_waiting(Duration::from_millis(1000), budget));
        // A world stage (2.7 s) against the whole budget: two and a half
        // budgets to go, so the wait would only cost the frame.
        assert!(!worth_waiting(Duration::from_millis(2700), budget));
        // And the same when most of the frame's budget is spent.
        assert!(!worth_waiting(Duration::from_millis(300), Duration::from_millis(120)));
    }

    /// A skipped wait is counted, and only the first few are logged: a
    /// world frame skips thousands a second, and "a skip is a finding" is
    /// true of the reason, not of 4,000 lines a second.
    #[test]
    fn skipped_waits_are_counted_and_the_first_few_logged() {
        let before = skipped();
        let lines_before = SKIP_LINES.load(Ordering::Relaxed);
        let p = Pending::new(Vec::new(), None);
        note_skipped(&p, Duration::from_millis(2_700), Duration::from_millis(10));
        assert_eq!(skipped(), before + 1);
        let logged = SKIP_LINES.load(Ordering::Relaxed) - lines_before;
        assert!(logged <= SKIP_LINES_AT, "at most the first {SKIP_LINES_AT} are logged: {logged}");
    }

    /// The typical make is an average of the recent ones, weighted to the
    /// last, and it moves with them: a run that starts with menu pipelines
    /// follows them into the world's slower ones.
    #[test]
    fn the_typical_make_follows_the_recent_ones() {
        // The counter is the process's, so this test makes no claim about
        // where it starts; only that slow makes move it and enough of them
        // settle it near their own time.
        note_typical(Duration::from_millis(30));
        let fast = TYPICAL_US.load(Ordering::Relaxed);
        for _ in 0..40 {
            note_typical(Duration::from_millis(3_000));
        }
        let slow = TYPICAL_US.load(Ordering::Relaxed);
        assert!(slow > fast, "slow makes move it up: {slow} > {fast}");
        assert!(slow > 2_500_000, "enough 3 s makes settle near 3 s: {slow} us");
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &earlier));
        assert!(queue.pop().is_none());
    }
}
