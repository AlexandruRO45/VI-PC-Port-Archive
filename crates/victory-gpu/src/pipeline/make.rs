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
//! Passes over the screen (no vertex input, no depth) have workers and a
//! longer budget of their own: the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth
//! of pipelines they waited minutes; a run that lost them had no diffuse
//! light at all.
//!
//! shortcut: the budgets (half a second of waiting per title frame, 20 s
//! for screen passes) are guesses; another pass the title runs only once
//! and whose pipeline takes longer is lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// Its worker was lost to it: it will not be made.
    lost: AtomicBool,
    /// A pass over the screen (see [`GraphicsState::screen_pass`]).
    screen: bool,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
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
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            lost: AtomicBool::new(false),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        // Not `get_or_init`: a worker lost in it would leave it locked.
        let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
        let _ = self.made.set(made);
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
const BUDGET: Duration = Duration::from_millis(500);

/// The same for passes over the screen (below the title's 30 s watchdog).
const SCREEN_BUDGET: Duration = Duration::from_secs(20);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p, REQUESTED);
        if p.lost.load(Ordering::Relaxed) {
            return None;
        }
        // Also with a world's worth queued: passes the title runs only once
        // set up what later ones loop over.
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers at `priority`, promoting an unstarted early entry.
fn start(p: &Arc<Pending>, priority: u8) {
    workers(p.screen);
    let pool = usize::from(p.screen);
    let mut queue = QUEUES[pool].lock().unwrap_or_else(|e| e.into_inner());
    enqueue_into(&mut queue, &AVAILABLE[pool], p, priority);
}

fn enqueue_into(queue: &mut Queue, available: &Condvar, p: &Arc<Pending>, priority: u8) {
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        available.notify_one();
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Waits for `p` while frame `frame` has budget of its kind left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: [AtomicU64; 2] = [const { AtomicU64::new(u64::MAX) }; 2];
    static WAITED_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let kind = usize::from(p.screen);
    let budget = [BUDGET, SCREEN_BUDGET][kind];
    let waited = &WAITED_US[kind];
    if FRAME[kind].swap(frame, Ordering::Relaxed) != frame {
        waited.store(0, Ordering::Relaxed);
    }
    let left = budget.saturating_sub(Duration::from_micros(waited.load(Ordering::Relaxed)));
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
    waited.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// Work queue for one pool (`false`: ordinary pipelines, `true`: screen passes).
struct Queue {
    early: VecDeque<Arc<Pending>>,
    early_depth: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            early_depth: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued early.
    /// Returns whether it is new to the queue, or `None` if already queued
    /// at that priority or above.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if p
            .graphics
            .as_ref()
            .is_some_and(|state| state.writes_depth_only())
        {
            // Shadow maps and depth prepasses supply later draws. Rebuild their
            // previously used PSOs first while no draw is requesting one.
            self.early_depth.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    /// Next pipeline to build: requested draws first, then early depth-only
    /// passes, then ordinary early pipelines.
    fn pop(&mut self) -> Option<Arc<Pending>> {
        let p = self.requested.pop_front().or_else(|| {
            // An early entry may have been promoted to `requested` already.
            std::iter::from_fn(|| {
                self.early_depth
                    .pop_front()
                    .or_else(|| self.early.pop_front())
            })
            .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        })?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

/// What the workers take from: the ordinary pipelines (`0`), the screen passes (`1`).
static QUEUES: [Mutex<Queue>; 2] = [const { Mutex::new(Queue::new()) }; 2];
static AVAILABLE: [Condvar; 2] = [const { Condvar::new() }; 2];

fn next(screen: bool) -> Arc<Pending> {
    let pool = usize::from(screen);
    let mut queue = QUEUES[pool].lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE[pool]
            .wait(queue)
            .unwrap_or_else(|e| e.into_inner());
    }
}

/// What the workers (by thread) are making.
static AT_WORK: Mutex<Vec<(u32, Arc<Pending>)>> = Mutex::new(Vec::new());

/// Starts worker `i`, for screen passes or for the others.
fn worker(i: usize, screen: bool) {
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{i}"))
        .spawn(move || {
            // Below the title's own threads (loading, recording).
            // SAFETY: a plain call on our own thread.
            unsafe {
                use windows::Win32::System::Threading::*;
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            }
            loop {
                let p = next(screen);
                // SAFETY: a plain call.
                let id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).push((id, p.clone()));
                p.make();
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
}

/// A worker is lost to a compile that crashed or never ends: another takes
/// its place (its pipeline stays unmade). With all of them lost, the rest of
/// a world's pipelines waited forever.
pub(super) fn worker_lost(id: u32) {
    static LOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
    let mut lost = None;
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| {
        if w.0 == id {
            w.1.lost.store(true, Ordering::Relaxed);
            lost = Some(w.1.screen);
        }
        w.0 != id
    });
    if let Some(screen) = lost {
        WAITING.fetch_sub(1, Ordering::Relaxed);
        worker(LOST.fetch_add(1, Ordering::Relaxed), screen);
    }
}

/// Starts workers (half the logical processors, 2 to 8, and two
/// for screen passes) on first use.
fn workers(screen: bool) {
    static STARTED: [OnceLock<()>; 2] = [const { OnceLock::new() }; 2];
    STARTED[usize::from(screen)].get_or_init(|| {
        let workers = if screen {
            2
        } else {
            std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8))
        };
        for i in 0..workers {
            worker(i + if screen { 50 } else { 0 }, screen);
        }
    });
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

        // Screen passes use their own pool and workers too.
        let sp = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::test_screen_pass()))));
        assert!(sp.screen);
        assert!(ready(&sp).is_none());
        let started = Instant::now();
        while sp.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(sp.made.get(), Some(None)));
    }

    #[test]
    fn queue_prioritizes_requested_then_early_depth_then_early_normal() {
        let normal_a = Arc::new(Pending::new(Vec::new(), None));
        let depth = Arc::new(Pending::new(Vec::new(), Some(Box::new(GraphicsState::test_depth_only()))));
        let normal_b = Arc::new(Pending::new(Vec::new(), None));
        let mut q = Queue::new();

        assert_eq!(q.push(&normal_a, EARLY), Some(true));
        assert_eq!(q.push(&normal_a, EARLY), None);
        assert_eq!(q.push(&depth, EARLY), Some(true));
        assert_eq!(q.push(&normal_b, EARLY), Some(true));
        // Promoting `normal_b` keeps it counted once (`Some(false)`) and jumps ahead.
        assert_eq!(q.push(&normal_b, REQUESTED), Some(false));
        assert_eq!(q.push(&normal_b, REQUESTED), None);

        let first = q.pop().unwrap();
        assert!(Arc::ptr_eq(&first, &normal_b));
        assert_eq!(first.queue_state.load(Ordering::Relaxed), MAKING);
        // Once making, it cannot be pushed or promoted again.
        assert_eq!(q.push(&normal_b, REQUESTED), None);

        // Depth-only early pass comes before ordinary early pass.
        let second = q.pop().unwrap();
        assert!(Arc::ptr_eq(&second, &depth));
        let third = q.pop().unwrap();
        assert!(Arc::ptr_eq(&third, &normal_a));
        assert!(q.pop().is_none(), "promoted duplicate in early queue is skipped");
    }

    #[test]
    fn early_starts_wanted_pipeline_and_tracks_peak() {
        while waiting() != 0 {
            std::thread::yield_now();
        }
        PEAK.store(0, Ordering::Relaxed);
        let mut local_q = Queue::new();
        let local_cv = Condvar::new();
        let peak_p = Arc::new(Pending::new(Vec::new(), None));
        enqueue_into(&mut local_q, &local_cv, &peak_p, EARLY);
        assert_eq!(waiting(), 1);
        assert_eq!(PEAK.swap(0, Ordering::Relaxed), 1);
        WAITING.fetch_sub(1, Ordering::Relaxed);

        let shader = super::super::shader("compute", b"early-test");
        let p = Arc::new(Pending::new(vec![shader], None));
        let before = super::super::library::EARLY.load(Ordering::Relaxed);
        early(&p);
        assert_eq!(super::super::library::EARLY.load(Ordering::Relaxed), before);
        assert_eq!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);

        super::super::library::note_used(&p.key);
        early(&p);
        assert_eq!(
            super::super::library::EARLY.load(Ordering::Relaxed),
            before + 1
        );
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }
}
