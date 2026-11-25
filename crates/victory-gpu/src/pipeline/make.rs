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
//! Pipelines a draw waits for go ahead of those started early (at the
//! title's creation, because an earlier run drew them).
//!
//! The default budget is half a second per title frame. A one-shot pass
//! whose pipeline takes longer may still be lost.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::hold_clock::{CLOCK, HoldClock};
use super::host::{GraphicsState, HostPipeline, MeshPass1};
use super::{pressure, telemetry as stats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    /// Whether it went into the queue of pipelines a draw waits for, and
    /// whether a worker took it (it may be in both queues).
    now: AtomicBool,
    taken: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    mesh_pass1: Mutex<Option<MeshPass1>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let key = key(&shaders, graphics.as_deref());
        #[cfg(feature = "diagnostics")]
        log("created", &key, || {
            let state = graphics.as_ref().map(|g| g.describe()).unwrap_or_default();
            format!("{shaders:?}\t{state}")
        });
        Pending {
            key,
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            now: AtomicBool::new(false),
            taken: AtomicBool::new(false),
            made: OnceLock::new(),
            mesh_pass1: Mutex::new(None),
            used: AtomicBool::new(false),
        }
    }

    // The queue grants one worker exclusive ownership of this attempt.
    fn make(&self, mode: CompileMode) -> Option<Deferred> {
        let _phase = stats::phase(stats::Phase::Build);
        let (started, cpu) = (Instant::now(), super::thread_cpu());
        let mut compile = Compile::new(mode);
        compile.requested = Some(&self.now);
        compile.mesh_pass1 = self.mesh_pass1.lock().unwrap_or_else(|e| e.into_inner()).take();
        let made = HostPipeline::build(&self.shaders, self.graphics.as_deref(), &mut compile);
        let (user, kernel) = super::thread_cpu_since(cpu);
        if matches!(mode, CompileMode::Worker) {
            stats::JOBS.fetch_add(1, Ordering::Relaxed);
            stats::JOB_WALL_MS.fetch_add(started.elapsed().as_millis() as u64, Ordering::Relaxed);
            stats::JOB_CPU_MS.fetch_add(user + kernel, Ordering::Relaxed);
        }
        if let Some(deferred) = compile.deferred {
            *self.mesh_pass1.lock().unwrap_or_else(|e| e.into_inner()) = compile.mesh_pass1.take();
            return Some(deferred);
        }
        #[cfg(feature = "diagnostics")]
        log_built(&self.key);
        tracing::debug!(
            shaders = ?self.shaders,
            ms = started.elapsed().as_millis() as u64,
            user_ms = user,
            kernel_ms = kernel,
            "pipeline made"
        );
        let _ = self.made.set(made);
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
        None
    }
}

/// What identifies a title pipeline across runs: its shader containers
/// and graphics state.
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(graphics.map(|g| g.describe()).unwrap_or_default().as_bytes());
    key.finalize().to_hex()[..32].to_owned()
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

/// Logs the library name of the host pipeline this thread just made or
/// loaded for `key` (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log_built(key: &str) {
    if let Some((name, loaded)) = super::library::LAST.take() {
        log("built", key, || format!("{name}\t{}", if loaded { "loaded" } else { "made" }));
    }
}

/// `VICTORY_PIPELINE_LOG=<file>` (diagnostics): a line per title pipeline
/// key created (`created`, its shaders, its state as described) and first
/// drawn or dispatched with (`used`), and its host pipeline's library name
/// once made or loaded (`built`, also from the pre-build), to compare what
/// a run creates with what the pre-build makes.
#[cfg(feature = "diagnostics")]
fn log(what: &str, key: &str, rest: impl FnOnce() -> String) {
    use std::collections::HashSet;
    use std::io::Write;
    static LOG: Mutex<Option<(std::fs::File, HashSet<String>)>> = Mutex::new(None);
    static PATH: OnceLock<Option<std::ffi::OsString>> = OnceLock::new();
    let Some(path) = PATH.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_LOG")) else {
        return;
    };
    let mut log = LOG.lock().unwrap_or_else(|e| e.into_inner());
    if log.is_none() {
        *log = std::fs::File::create(path).ok().map(|f| (f, HashSet::new()));
    }
    let Some((file, seen)) = log.as_mut() else { return };
    if seen.insert(format!("{what} {key}")) {
        let _ = writeln!(file, "{what}\t{key}\t{}", rest());
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

const DEFAULT_WAIT_MS: u64 = 500;
const MAX_WAIT_MS: u64 = 5000;

fn parse_wait_ms(value: &str) -> Option<u64> {
    value.trim().parse().ok().filter(|&ms| ms <= MAX_WAIT_MS)
}

fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let ms = setting("VICTORY_PIPELINE_WAIT_MS", DEFAULT_WAIT_MS, parse_wait_ms);
        tracing::info!(ms, "host pipeline wait budget per frame");
        Duration::from_millis(ms)
    })
}

#[derive(Default)]
struct WaitBudget {
    frame: u64,
    spent: Duration,
}

impl WaitBudget {
    fn reserve(&mut self, frame: u64, budget: Duration) -> Duration {
        if frame < self.frame {
            return Duration::ZERO;
        }
        if frame != self.frame {
            *self = Self { frame, spent: Duration::ZERO };
        }
        let remaining = budget.saturating_sub(self.spent);
        self.spent += remaining;
        remaining
    }

    fn refund(&mut self, frame: u64, reserved: Duration, elapsed: Duration) {
        if frame == self.frame {
            self.spent -= reserved.saturating_sub(elapsed);
        }
    }
}

/// Outstanding pipelines (queued, deferred, or in progress).
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Outstanding pipelines, for the frame line and worker priority.
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // Listed for an early start in later runs from its first use, made or
    // not: one a run never got made is the one it should start early.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        #[cfg(feature = "diagnostics")]
        log("used", &p.key, String::new);
    }
    if p.made.get().is_none() {
        start(p, true);
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Queues `p` for the workers: once, and once more ahead of the early
/// starts if a draw waits for it (`now`) while it is still queued.
fn start(p: &Arc<Pending>, now: bool) {
    // A previously requested job is already queued, running or deferred.
    if now && p.now.load(Ordering::Relaxed) {
        return;
    }
    if let Some(now) = enqueue(p, now) {
        queue(p.clone(), now);
    } else if now {
        // Wake workers if the first draw promoted an already-deferred job.
        // Synchronize with their empty-queue check before signaling.
        let _queue = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner());
        QUEUES.1.notify_one();
    }
}

/// Which queue `p` goes into now (true: the one draws wait for), if any.
fn enqueue(p: &Pending, now: bool) -> Option<bool> {
    let first = !p.queued.swap(true, Ordering::Relaxed);
    if first {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
    }
    let ahead = now && !p.now.load(Ordering::Relaxed) && !p.now.swap(true, Ordering::Relaxed);
    (first || ahead && !p.taken.load(Ordering::Relaxed)).then_some(now)
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    super::library::note_created(&p.key);
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    static SPENT: Mutex<WaitBudget> = Mutex::new(WaitBudget { frame: 0, spent: Duration::ZERO });
    let budget = wait_budget();
    if budget.is_zero() || p.made.get().is_some() {
        return;
    }
    let left = SPENT.lock().unwrap_or_else(|e| e.into_inner()).reserve(frame, budget);
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    wait_pending(p, started, left, CLOCK.get());
    SPENT.lock().unwrap_or_else(|e| e.into_inner()).refund(frame, left, started.elapsed());
}

fn wait_pending(p: &Pending, started: Instant, left: Duration, clock: Option<&HoldClock>) {
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut pipeline_hold = None;
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        if let Some(clock) = clock {
            pipeline_hold.get_or_insert_with(|| clock.enter());
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    // No watchdog credit once the wait ends, before the budget refund.
    drop(pipeline_hold);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lane {
    Large,
    Huge,
}

impl Lane {
    const ALL: [Self; 2] = [Self::Large, Self::Huge];

    fn for_bytes(bytes: usize) -> Option<Self> {
        if bytes >= 4 * 1024 * 1024 {
            Some(Self::Huge)
        } else if bytes >= 256 * 1024 {
            Some(Self::Large)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum CompileMode {
    Sync,
    Worker,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Deferred {
    Lane(Lane),
    Pressure,
}

/// One build attempt. A deferral is neither a D3D12 error nor a refusal.
pub(super) struct Compile<'a> {
    mode: CompileMode,
    requested: Option<&'a AtomicBool>,
    deferred: Option<Deferred>,
    /// Pass 1, if pass 2 was deferred. Tied to a single pending pipeline.
    pub(super) mesh_pass1: Option<MeshPass1>,
}

impl Compile<'_> {
    pub(super) fn new(mode: CompileMode) -> Self {
        Self { mode, requested: None, deferred: None, mesh_pass1: None }
    }

    fn defer_for_memory(&mut self, paused: bool) -> bool {
        let defer = paused && matches!(self.mode, CompileMode::Worker)
            && self.requested.is_some_and(|now| !now.load(Ordering::Relaxed));
        if defer {
            self.deferred = Some(Deferred::Pressure);
        }
        defer
    }

    pub(super) fn cold<T>(
        &mut self,
        name: &str,
        bytes: usize,
        create: impl FnOnce() -> windows::core::Result<T>,
    ) -> windows::core::Result<Option<T>> {
        if self.defer_for_memory(PREWARM_PAUSED.load(Ordering::Relaxed)) {
            return Ok(None);
        }
        let lane = Lane::for_bytes(bytes).filter(|_| matches!(self.mode, CompileMode::Worker));
        let permit = if let Some(lane) = lane {
            let Some(permit) = Permit::acquire(lane) else {
                self.deferred = Some(Deferred::Lane(lane));
                return Ok(None);
            };
            Some(permit)
        } else {
            None
        };
        let phase = stats::phase(stats::Phase::DriverCreate);
        let started = Instant::now();
        let result = create();
        drop(phase);
        drop(permit);
        stats::CREATES.fetch_add(1, Ordering::Relaxed);
        if result.is_err() {
            stats::FAILED_CREATES.fetch_add(1, Ordering::Relaxed);
        }
        if lane.is_some() {
            tracing::debug!(name, ?lane, bytes, ms = started.elapsed().as_millis() as u64, "cold host pipeline compile");
        }
        result.map(Some)
    }
}

struct Permit(Lane);

impl Permit {
    fn acquire(lane: Lane) -> Option<Self> {
        let available = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner()).take_lane(lane);
        if available { Some(Self(lane)) } else { None }
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let mut queues = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner());
        queues.busy[self.0 as usize] = false;
        QUEUES.1.notify_all();
    }
}

struct Queues {
    fresh: [VecDeque<Arc<Pending>>; 2],
    deferred: [VecDeque<Arc<Pending>>; 2],
    busy: [bool; 2],
    pressure: VecDeque<Arc<Pending>>,
}

impl Queues {
    const fn new() -> Self {
        Self {
            fresh: [VecDeque::new(), VecDeque::new()],
            deferred: [VecDeque::new(), VecDeque::new()],
            busy: [false; 2],
            pressure: VecDeque::new(),
        }
    }

    fn take_lane(&mut self, lane: Lane) -> bool {
        !std::mem::replace(&mut self.busy[lane as usize], true)
    }

    fn retry(&mut self, urgent: bool) -> Option<Arc<Pending>> {
        for lane in Lane::ALL {
            let i = lane as usize;
            if self.busy[i] {
                continue;
            }
            if let Some(at) = self.deferred[i].iter().position(|p| p.now.load(Ordering::Relaxed) == urgent) {
                return self.deferred[i].remove(at);
            }
        }
        None
    }

    fn claim(&mut self, index: usize) -> Option<Arc<Pending>> {
        while let Some(p) = self.fresh[index].pop_front() {
            if !p.taken.swap(true, Ordering::Relaxed) {
                return Some(p);
            }
        }
        None
    }

    fn defer(&mut self, p: Arc<Pending>, reason: Deferred) {
        match reason {
            Deferred::Lane(lane) => self.deferred[lane as usize].push_back(p),
            Deferred::Pressure => self.pressure.push_back(p),
        }
    }

    fn next(&mut self, paused: bool) -> Option<Arc<Pending>> {
        self.retry(true)
            .or_else(|| self.claim(0))
            .or_else(|| {
                let at = self.pressure.iter().position(|p| p.now.load(Ordering::Relaxed))?;
                self.pressure.remove(at)
            })
            .or_else(|| if paused { None } else { self.retry(false) })
            .or_else(|| if paused { None } else { self.pressure.pop_front() })
            .or_else(|| self.claim(1)) // Keep probing new early jobs for cache hits.
    }
}

static QUEUES: (Mutex<Queues>, Condvar) = (Mutex::new(Queues::new()), Condvar::new());
static PREWARM_PAUSED: AtomicBool = AtomicBool::new(false);

fn start_sampler() {
    let initial = pressure::paused(false, pressure::available_mib());
    PREWARM_PAUSED.store(initial, Ordering::Relaxed);
    let spawn = std::thread::Builder::new().name("victory-pso-sampler".into()).spawn(move || {
        use super::library::{LOADED, MADE};
        let mut last = [0; 8];
        pressure::sample_loop(initial, |paused| {
            if PREWARM_PAUSED.load(Ordering::Relaxed) != paused {
                let _queue = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner());
                PREWARM_PAUSED.store(paused, Ordering::Relaxed);
                QUEUES.1.notify_all();
            }
        }, |available_mib, pages_input_s, paused| {
            let total = [
                LOADED.load(Ordering::Relaxed), stats::MISSES.load(Ordering::Relaxed),
                MADE.load(Ordering::Relaxed), stats::CREATES.load(Ordering::Relaxed),
                stats::FAILED_CREATES.load(Ordering::Relaxed), stats::JOBS.load(Ordering::Relaxed),
                stats::JOB_WALL_MS.load(Ordering::Relaxed), stats::JOB_CPU_MS.load(Ordering::Relaxed),
            ];
            let delta: [u64; 8] = std::array::from_fn(|i| total[i].saturating_sub(last[i]));
            last = total;
            let (urgent, early, parked, lanes, busy) = {
                let q = QUEUES.0.lock().unwrap_or_else(|e| e.into_inner());
                let waiting = |i: usize| q.fresh[i].iter().filter(|p| !p.taken.load(Ordering::Relaxed)
                    && (i == 0 || !p.now.load(Ordering::Relaxed))).count();
                (waiting(0), waiting(1), q.pressure.len(), q.deferred.each_ref().map(VecDeque::len), q.busy)
            };
            tracing::info!(?available_mib, ?pages_input_s, prewarm_paused = paused,
                outstanding = waiting(), urgent, early, pressure_deferred = parked,
                large_deferred = lanes[0], huge_deferred = lanes[1], large_busy = busy[0], huge_busy = busy[1],
                phases = %stats::snapshot(), hits = delta[0], misses = delta[1], made = delta[2],
                creates_finished = delta[3], creates_failed = delta[4], attempts = delta[5],
                worker_wall_ms = delta[6], worker_cpu_ms = delta[7], "host pipeline queue sample");
        });
    });
    if let Err(error) = spawn {
        PREWARM_PAUSED.store(false, Ordering::Relaxed);
        tracing::warn!(%error, "pipeline memory sampler unavailable; prewarm not throttled");
    }
}

const BOOST_ABOVE: u64 = 64;
const LOWER_AT: u64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Priority {
    Auto,
    Normal,
    BelowNormal,
}

impl Priority {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "normal" => Some(Self::Normal),
            "below-normal" | "below_normal" => Some(Self::BelowNormal),
            _ => None,
        }
    }

    fn normal(self, outstanding: u64, was_normal: bool) -> bool {
        match self {
            Self::Auto => outstanding > BOOST_ABOVE || (was_normal && outstanding > LOWER_AT),
            Self::Normal => true,
            Self::BelowNormal => false,
        }
    }
}

fn setting<T: Copy>(key: &str, default: T, parse: impl FnOnce(&str) -> Option<T>) -> T {
    std::env::var(key).ok().map_or(default, |value| {
        parse(&value).unwrap_or_else(|| {
            tracing::warn!(key, %value, "invalid pipeline setting; using default");
            default
        })
    })
}

fn set_priority(previous: &mut Option<bool>, normal: bool) {
    if *previous == Some(normal) {
        return;
    }
    *previous = Some(normal);
    use windows::Win32::System::Threading::*;
    let priority = if normal { THREAD_PRIORITY_NORMAL } else { THREAD_PRIORITY_BELOW_NORMAL };
    // SAFETY: change only the current worker's priority.
    if let Err(error) = unsafe { SetThreadPriority(GetCurrentThread(), priority) } {
        tracing::warn!(%error, normal, "could not set pipeline worker priority");
    } else {
        tracing::debug!(normal, pending = waiting(), "set pipeline worker priority");
    }
}

/// Hands `p` to the workers, started on first use.
fn queue(p: Arc<Pending>, now: bool) {
    static WORKERS: OnceLock<usize> = OnceLock::new();
    let workers = *WORKERS.get_or_init(|| {
        let default = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        let n = setting("VICTORY_PIPELINE_WORKERS", default, |value| {
            value.trim().parse::<usize>().ok().filter(|n| (1..=16).contains(n))
        });
        let priority = setting("VICTORY_PIPELINE_PRIORITY", Priority::Auto, Priority::parse);
        tracing::info!(workers = n, ?priority, "host pipeline worker configuration");
        start_sampler();
        (0..n)
            .filter(|&i| {
                let spawned = std::thread::Builder::new()
                    .name(format!("victory-pipelines-{i}"))
                    .spawn(move || {
                        stats::register(i);
                        work(priority);
                    });
                if let Err(e) = &spawned {
                    tracing::warn!(error = %e, "no pipeline worker");
                }
                spawned.is_ok()
            })
            .count()
    });
    if workers == 0 {
        // No worker: made here instead.
        if !p.taken.swap(true, Ordering::Relaxed) {
            let _ = p.make(CompileMode::Sync);
            WAITING.fetch_sub(1, Ordering::Relaxed);
        }
        return;
    }
    let (lock, work) = &QUEUES;
    lock.lock().unwrap_or_else(|e| e.into_inner()).fresh[usize::from(!now)].push_back(p);
    work.notify_one();
}

/// A worker: makes queued pipelines, those draws wait for first, and saves
/// the cache when idle (first uses of pipelines made before).
fn work(priority: Priority) {
    let mut normal = None;
    set_priority(&mut normal, priority.normal(0, false));
    let (lock, work) = &QUEUES;
    loop {
        let p = {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(p) = queues.next(PREWARM_PAUSED.load(Ordering::Relaxed)) {
                    break p;
                }
                let (held, waited) = work
                    .wait_timeout(queues, Duration::from_secs(20))
                    .unwrap_or_else(|e| e.into_inner());
                queues = held;
                if waited.timed_out() {
                    drop(queues);
                    super::library::save(false);
                    queues = lock.lock().unwrap_or_else(|e| e.into_inner());
                }
            }
        };
        let next_priority = priority.normal(waiting(), normal == Some(true));
        set_priority(&mut normal, next_priority);
        if let Some(reason) = p.make(CompileMode::Worker) {
            let mut queues = lock.lock().unwrap_or_else(|e| e.into_inner());
            queues.defer(p, reason);
            work.notify_one();
            continue;
        }
        let remaining = WAITING.fetch_sub(1, Ordering::Relaxed) - 1;
        if priority == Priority::Auto && remaining <= LOWER_AT {
            set_priority(&mut normal, false);
        }
        if remaining == 0 {
            drained();
        } else {
            super::library::save(false);
        }
    }
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

    #[test]
    fn wait_budget_is_shared_and_refunds_only_the_current_frame() {
        let mut spent = WaitBudget::default();
        let ms = Duration::from_millis;
        assert_eq!(parse_wait_ms("0"), Some(0));
        assert_eq!(parse_wait_ms(" 24 "), Some(24));
        assert_eq!(parse_wait_ms("5001"), None);
        let budget = ms(DEFAULT_WAIT_MS);
        assert_eq!(spent.reserve(1, budget), budget);
        assert!(spent.reserve(1, budget).is_zero());
        spent.refund(1, budget, ms(24));
        let remainder = spent.reserve(1, budget);
        assert_eq!(remainder, budget - ms(24));
        assert_eq!(spent.reserve(2, budget), budget);
        spent.refund(1, remainder, Duration::ZERO);
        assert!(spent.reserve(2, budget).is_zero());
        assert!(spent.reserve(1, budget).is_zero());
    }

    #[test]
    fn auto_priority_tracks_backlog_with_hysteresis() {
        let p = Priority::Auto;
        assert!(!p.normal(64, false));
        assert!(p.normal(65, false));
        assert!(p.normal(33, true));
        assert!(!p.normal(32, true));
        assert!(Priority::Normal.normal(0, false));
        assert!(!Priority::BelowNormal.normal(65, true));
    }

    #[test]
    fn worker_defers_busy_lanes_but_sync_and_small_compiles_continue() {
        let large = 256 * 1024;
        let huge = 4 * 1024 * 1024;
        let permit = Permit::acquire(Lane::Large).unwrap();
        assert!(Permit::acquire(Lane::Large).is_none());
        let mut worker = Compile::new(CompileMode::Worker);
        assert_eq!(worker.cold("large", large, || panic!("compile ran")).unwrap(), None::<()>);
        assert_eq!(worker.deferred, Some(Deferred::Lane(Lane::Large)));
        let mut sync = Compile::new(CompileMode::Sync);
        assert_eq!(sync.cold("prebuild", large, || Ok(1)).unwrap(), Some(1));
        let mut worker = Compile::new(CompileMode::Worker);
        assert_eq!(worker.cold("small", large - 1, || Ok(2)).unwrap(), Some(2));
        assert_eq!(worker.cold("huge", huge, || Ok(3)).unwrap(), Some(3));
        assert_eq!(worker.deferred, None);
        drop(permit);
        assert_eq!(worker.cold("retry", large, || Ok(4)).unwrap(), Some(4));
    }

    #[test]
    fn deferred_work_keeps_urgency_without_blocking_other_jobs() {
        let mut queues = Queues::new();
        let p = Arc::new(Pending::new(Vec::new(), None));
        let early = Arc::new(Pending::new(Vec::new(), None));
        let ready = Arc::new(Pending::new(Vec::new(), None));
        queues.deferred[Lane::Large as usize].push_back(early.clone());
        queues.deferred[Lane::Large as usize].push_back(p.clone());
        queues.fresh[0].push_back(ready.clone());
        assert!(queues.take_lane(Lane::Large));
        p.now.store(true, Ordering::Relaxed);
        assert!(Arc::ptr_eq(&queues.next(false).unwrap(), &ready));
        assert!(queues.next(false).is_none());
        queues.busy[Lane::Large as usize] = false;
        assert!(Arc::ptr_eq(&queues.next(false).unwrap(), &p));
        assert!(Arc::ptr_eq(&queues.next(false).unwrap(), &early));
    }

    #[test]
    fn pressure_defers_cold_early_work_but_promotion_and_hits_stay_eligible() {
        let mut queues = Queues::new();
        let early = Arc::new(Pending::new(Vec::new(), None));
        let promote = Arc::new(Pending::new(Vec::new(), None));
        let probe = Arc::new(Pending::new(Vec::new(), None));
        let mut compile = Compile::new(CompileMode::Worker);
        compile.requested = Some(&promote.now);
        assert!(!compile.defer_for_memory(false));
        assert!(compile.defer_for_memory(true));
        assert_eq!(compile.deferred, Some(Deferred::Pressure));
        assert!(!Compile::new(CompileMode::Sync).defer_for_memory(true));
        queues.defer(early.clone(), Deferred::Pressure);
        queues.defer(promote.clone(), Deferred::Pressure);
        queues.fresh[1].push_back(probe.clone());
        assert!(Arc::ptr_eq(&queues.next(true).unwrap(), &probe));
        assert!(queues.next(true).is_none());
        promote.now.store(true, Ordering::Relaxed);
        assert!(!compile.defer_for_memory(true));
        assert!(Arc::ptr_eq(&queues.next(true).unwrap(), &promote));
        assert!(queues.next(true).is_none());
        assert!(Arc::ptr_eq(&queues.next(false).unwrap(), &early));
    }

    /// An early start is queued once; a draw waiting for it puts it ahead,
    /// once, unless a worker has taken it.
    #[test]
    fn draws_go_ahead_of_early_starts() {
        let p = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&p, false), Some(false));
        assert_eq!(enqueue(&p, false), None);
        assert_eq!(enqueue(&p, true), Some(true));
        assert_eq!(enqueue(&p, true), None);
        let q = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&q, false), Some(false));
        q.taken.store(true, Ordering::Relaxed);
        assert_eq!(enqueue(&q, true), None);
        let r = Pending::new(Vec::new(), None);
        assert_eq!(enqueue(&r, true), Some(true));
        assert_eq!(enqueue(&r, false), None);
        // Counted as waiting by `enqueue`, never handed to a worker.
        WAITING.fetch_sub(3, Ordering::Relaxed);
    }

    #[test]
    fn tracks_only_the_block_on_a_pending_pipeline() {
        let pending = Pending::new(Vec::new(), None);
        let clock = HoldClock::default();
        wait_pending(&pending, Instant::now(), Duration::ZERO, Some(&clock));
        assert!(clock.unoccupied(), "no budget left");
        wait_pending(&pending, Instant::now(), Duration::from_millis(5), Some(&clock));
        assert!(!clock.unoccupied(), "spent time waiting for a pending PSO");

        assert!(pending.made.set(None).is_ok());
        let clock = HoldClock::default();
        wait_pending(&pending, Instant::now(), Duration::from_secs(1), Some(&clock));
        assert!(clock.unoccupied(), "PSO already answered");
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }
}
