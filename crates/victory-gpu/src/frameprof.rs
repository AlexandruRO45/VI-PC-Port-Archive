//! Frame-attributed function timings for diagnostics builds
//! (`VICTORY_FRAME_PROFILE=<file>`).
//!
//! A probe is a scope: [`enter`] with a [`Site`] opens it and the guard it
//! returns closes it. The time is added to the thread that ran it, and the
//! frame it belongs to is the one that ends next: a frame is the stretch
//! between two presents, and its wall time is the frame budget the report
//! divides by. A scope that starts before a present and ends after it counts
//! in the next frame (the one its close lands in), which is what an
//! accountant would do and needs no locking on the hot path.
//!
//! The report ranks functions by the share of the frame budget they take
//! (`self`, nested calls taken out), then the same by what they call
//! (`inclusive`), the worst frames' own breakdown, and what the frames are
//! made of by kind of time: work on the CPU, waiting for the host GPU,
//! waiting for a lock, waiting for a frame to be paced, and the time no
//! scope covers (the title's own code, the system, and host code without a
//! probe). See `docs/frametime-profiler.md`.
//!
//! Why scope timers next to the sampler in [`crate::profile`]: the sampler
//! answers *which* functions take the time, statistically, with no idea of
//! frames; this answers *what one frame's budget went to* exactly, for the
//! host code we can open and close a scope in. Title code cannot be
//! instrumented: it shows up as "outside any scope" and in the sampler.
//!
//! Observer effect: every open and close reads the clock and touches
//! atomics. Measured on an idle machine, a pair costs tens of nanoseconds
//! (`crates/victory-gpu/src/frameprof.rs`'s overhead test prints it), which
//! a frame of 150 ms cannot see; on a per-draw probe of 9,000 draws a frame
//! it is under 1 ms, and the probe of a hot loop that is shorter than that
//! pair reads as the pair's own cost, not as the function's. Probes are
//! therefore put on frame-sized work (submission, copies, plans), not on
//! inner loops, and the report is meant to be read with an eye on the
//! calls/frame column: a function with a huge count and a tiny self time is
//! where the measurement itself starts to matter.
//!
//! The module is portable (standard library only, no Windows calls) so its
//! tests run on any host; `lib.rs` compiles it in with the `diagnostics`
//! feature, and call sites add nothing without it.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Most functions one run can profile: site ids are indexes into the
/// per-thread counters, so the space is fixed and dense.
const MOST_SITES: usize = 64;
/// Most threads whose time is kept apart; a thread past this is not
/// measured (its scopes close as if profiling were off).
const MOST_THREADS: usize = 16;
/// Most frames kept for the per-frame part of the report. The totals and
/// the ranking are not capped; the worst frames are.
const MOST_FRAMES: usize = 2048;
/// Most (function, thread) rows kept for one frame.
const MOST_ROWS: usize = 256;
/// Deepest nesting of probes one thread records.
const MOST_DEPTH: usize = 64;
/// No id assigned to a site yet.
const NO_ID: usize = usize::MAX;
/// A site left out by `VICTORY_FRAME_PROFILE_TAGS` (or past [`MOST_SITES`]).
const OFF: usize = usize::MAX - 1;
/// No thread slot assigned yet.
const NO_SLOT: u32 = u32::MAX;
/// The kind of a probe's time, for the report's breakdown.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// Work on the CPU: recording, preparing, copying on this thread.
    Cpu,
    /// Waiting for the host GPU to finish what was submitted.
    Gpu,
    /// Copying title memory or GPU memory: bandwidth, not instructions.
    Memory,
    /// Waiting for a lock another thread holds.
    Sync,
    /// Waiting for the frame pacer or the display, not work at all.
    Idle,
}

impl Kind {
    const COUNT: usize = 5;

    fn index(self) -> usize {
        match self {
            Kind::Cpu => 0,
            Kind::Gpu => 1,
            Kind::Memory => 2,
            Kind::Sync => 3,
            Kind::Idle => 4,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Kind::Cpu => "cpu",
            Kind::Gpu => "gpu",
            Kind::Memory => "memory",
            Kind::Sync => "sync",
            Kind::Idle => "idle",
        }
    }
}

/// One function a probe can time. Declare it once, next to its call site:
///
/// ```ignore
/// pub(crate) static SUBMIT: Site = Site::new("host::submit", "host", Kind::Cpu);
/// // ...
/// #[cfg(feature = "diagnostics")]
/// let _prof = crate::frameprof::enter(&crate::frameprof::SUBMIT);
/// ```
///
/// The id is assigned when the site is first entered, so [`enter`] on a hot
/// path costs one atomic load after the first call.
pub(crate) struct Site {
    /// The name the report shows; keep it stable, it is the key for
    /// before/after comparison.
    pub(crate) name: &'static str,
    /// The `VICTORY_FRAME_PROFILE_TAGS` group this site belongs to.
    pub(crate) tag: &'static str,
    /// What its time is, for the report's breakdown.
    pub(crate) kind: Kind,
    id: AtomicUsize,
}

impl Site {
    /// A site, declared as a `static` in the function's module.
    pub(crate) const fn new(name: &'static str, tag: &'static str, kind: Kind) -> Site {
        Site {
            name,
            tag,
            kind,
            id: AtomicUsize::new(NO_ID),
        }
    }
}

/// What the switches asked for, read once.
pub(crate) struct Config {
    file: Option<PathBuf>,
    /// Tag prefixes to profile; empty means every site.
    tags: Vec<String>,
    /// How often the report is rewritten.
    every: Duration,
    /// Stop after this many frames (0: never stop on the count).
    frames: u64,
    /// A report from an earlier run to compare against.
    compare: Option<PathBuf>,
}

impl Config {
    /// `VICTORY_FRAME_PROFILE=<file>`: the report's path. More than one path
    /// is not supported; an empty value, `0`, turns profiling off, as does
    /// the absence of the variable.
    ///
    /// `VICTORY_FRAME_PROFILE_TAGS=host,copy`: only sites whose tag or name
    /// starts with one of these; `-` leaves every probe out (the frames
    /// still measured, so one run can price the probes against another).
    /// `VICTORY_FRAME_PROFILE_SECONDS=<n>`: how
    /// often the report is written (default 20). `VICTORY_FRAME_PROFILE_FRAMES=<n>`:
    /// stop after `n` frames and write last. `VICTORY_FRAME_PROFILE_COMPARE=<file>`:
    /// show the change from that report.
    pub(crate) fn from_env() -> Config {
        let file = std::env::var_os("VICTORY_FRAME_PROFILE")
            .map(PathBuf::from)
            .filter(|p| {
                let s = p.as_os_str();
                !s.is_empty() && s != "0"
            });
        let list = |name: &str| {
            std::env::var(name)
                .map(|v| {
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default()
        };
        let number = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
        };
        Config {
            file,
            tags: list("VICTORY_FRAME_PROFILE_TAGS"),
            every: Duration::from_secs(
                number("VICTORY_FRAME_PROFILE_SECONDS").unwrap_or(20).max(1),
            ),
            frames: number("VICTORY_FRAME_PROFILE_FRAMES").unwrap_or(0),
            compare: std::env::var_os("VICTORY_FRAME_PROFILE_COMPARE").map(PathBuf::from),
        }
    }

    /// Whether a site is in the selected tags (a prefix of its tag or name).
    fn selected(&self, site: &Site) -> bool {
        if self.tags.iter().any(|tag| tag == "-") {
            return false;
        }
        self.tags.is_empty()
            || self.tags.iter().any(|tag| {
                site.tag == tag
                    || site.name == tag
                    || site.tag.starts_with(&format!("{tag}::"))
                    || site.name.starts_with(&format!("{tag}::"))
            })
    }

    /// A configuration for the tests: a file in the temporary directory,
    /// every `every`, no frame limit, no comparison.
    #[cfg(test)]
    fn for_test(file: PathBuf, tags: &[&str]) -> Config {
        Config {
            file: Some(file),
            tags: tags.iter().map(|t| (*t).to_owned()).collect(),
            every: Duration::from_secs(3600),
            frames: 0,
            compare: None,
        }
    }
}

/// The counters of one (site, thread): since the process started, so that
/// the frame boundary reads deltas and nothing is ever reset on the hot path.
#[derive(Default)]
struct Totals {
    incl: AtomicU64,
    own: AtomicU64,
    calls: AtomicU64,
}

/// One function's share of one frame.
struct Row {
    site: usize,
    slot: usize,
    incl_ns: u64,
    self_ns: u64,
    calls: u64,
}

/// One frame between two presents.
struct Frame {
    presented: u64,
    title: u64,
    wall_ns: u64,
    rows: Vec<Row>,
}

/// A function's totals since the run started.
#[derive(Default, Clone, Copy)]
struct Agg {
    self_ns: u64,
    incl_ns: u64,
    calls: u64,
    /// One bit per thread the function ran on.
    slots: u16,
}

/// A thread's totals since the run started.
#[derive(Default, Clone, Copy)]
struct SlotAgg {
    self_ns: u64,
    kind: [u64; Kind::COUNT],
}

/// What the report is built from.
struct History {
    frames: VecDeque<Frame>,
    agg: Vec<Agg>,
    slots: Vec<SlotAgg>,
    frames_closed: u64,
    /// Whether the first present, whose frame is not counted, has passed.
    begun: bool,
    wall_ns: u64,
    /// Frame time no scope accounts for: the title's own code, the system,
    /// and host code without a probe.
    outside_ns: u64,
}

/// The profiler: one per process, read once from the environment.
pub(crate) struct Profiler {
    cfg: Config,
    running: AtomicBool,
    sites: Mutex<Vec<&'static Site>>,
    totals: Vec<Totals>,
    /// Each counter's last value, to read deltas at the frame boundary.
    seen: Mutex<Vec<[u64; 3]>>,
    last: Mutex<Instant>,
    last_report: Mutex<Instant>,
    history: Mutex<History>,
    names: Mutex<Vec<String>>,
    next_slot: AtomicUsize,
}

impl Profiler {
    /// A profiler that does nothing if `cfg` names no file.
    pub(crate) fn new(cfg: Config) -> Profiler {
        let running = cfg.file.is_some();
        let now = Instant::now();
        Profiler {
            cfg,
            running: AtomicBool::new(running),
            sites: Mutex::new(Vec::new()),
            totals: (0..MOST_SITES * MOST_THREADS)
                .map(|_| Totals::default())
                .collect(),
            seen: Mutex::new(vec![[0; 3]; MOST_SITES * MOST_THREADS]),
            last: Mutex::new(now),
            last_report: Mutex::new(now),
            history: Mutex::new(History {
                frames: VecDeque::new(),
                agg: vec![Agg::default(); MOST_SITES],
                slots: vec![SlotAgg::default(); MOST_THREADS],
                frames_closed: 0,
                begun: false,
                wall_ns: 0,
                outside_ns: 0,
            }),
            names: Mutex::new(vec![String::new(); MOST_THREADS]),
            next_slot: AtomicUsize::new(0),
        }
    }

    /// The site's id, registering it the first time. `None` for a site the
    /// tags leave out.
    fn site_id(&self, site: &'static Site) -> Option<usize> {
        match site.id.load(Ordering::Acquire) {
            NO_ID => self.register(site),
            OFF => None,
            id => Some(id),
        }
    }

    fn register(&self, site: &'static Site) -> Option<usize> {
        let mut sites = self.sites.lock().unwrap_or_else(|e| e.into_inner());
        // Another thread may have registered it while this one waited.
        match site.id.load(Ordering::Acquire) {
            NO_ID => {}
            OFF => return None,
            id => return Some(id),
        }
        if !self.cfg.selected(site) || sites.len() >= MOST_SITES {
            site.id.store(OFF, Ordering::Release);
            return None;
        }
        let id = sites.len();
        sites.push(site);
        site.id.store(id, Ordering::Release);
        Some(id)
    }

    /// The slot of the calling thread, claiming one the first time.
    fn slot(&self) -> Option<usize> {
        let me = std::ptr::from_ref(self) as usize;
        let cached = SLOT.with(|s| match s.get() {
            (owner, slot) if owner == me && slot != NO_SLOT => Some(slot as usize),
            _ => None,
        });
        if let Some(slot) = cached {
            return Some(slot);
        }
        let slot = self.claim_slot()?;
        SLOT.with(|s| s.set((me, slot as u32)));
        Some(slot)
    }

    fn claim_slot(&self) -> Option<usize> {
        let slot = self.next_slot.fetch_add(1, Ordering::Relaxed);
        if slot >= MOST_THREADS {
            return None;
        }
        let thread = std::thread::current();
        let name = thread
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{:?}", thread.id()));
        let mut names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(at) = names.get_mut(slot) {
            *at = name;
        }
        Some(slot)
    }

    /// Opens a probe, or nothing when profiling is off, the site is left
    /// out, the thread has no slot, or the nesting is too deep.
    fn open(self: &'static Profiler, site: &'static Site) -> Option<Guard> {
        if !self.running.load(Ordering::Relaxed) {
            return None;
        }
        let id = self.site_id(site)?;
        let slot = self.slot()?;
        let start = Instant::now();
        let pushed = OPEN.with(|stack| {
            let mut stack = stack.borrow_mut();
            if stack.len() >= MOST_DEPTH {
                return false;
            }
            stack.push(Open {
                site: id,
                start,
                children: 0,
            });
            true
        });
        pushed.then_some(Guard {
            prof: self,
            site: id,
            slot,
            _thread: std::marker::PhantomData,
        })
    }

    fn add(&self, site: usize, slot: usize, incl: u64, own: u64) {
        let totals = &self.totals[site * MOST_THREADS + slot];
        totals.incl.fetch_add(incl, Ordering::Relaxed);
        totals.own.fetch_add(own, Ordering::Relaxed);
        totals.calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Closes the frame that began at the last present and opened the next.
    /// `presented`: the count [`crate::queue::presented`] is at; `title`:
    /// the title's own frame token ([`crate::frame::current`]).
    pub(crate) fn close_frame(&self, presented: u64, title: u64) {
        if !self.running.load(Ordering::Relaxed) {
            return;
        }
        let now = Instant::now();
        let wall = {
            let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
            let wall = now.saturating_duration_since(*last);
            *last = now;
            wall
        };
        if wall.is_zero() {
            return;
        }
        let wall_ns = wall.as_nanos().min(u128::from(u64::MAX)) as u64;
        let presenter = self.slot();
        let rows = self.deltas();
        let presenter_self = rows
            .iter()
            .filter(|row| Some(row.slot) == presenter)
            .map(|row| row.self_ns)
            .sum::<u64>();

        let mut history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        if !history.begun {
            // The stretch from the first probe (and so from profiling being
            // turned on) to the first present is not a frame: it may hold
            // minutes of loading, and it is not what the report is about.
            history.begun = true;
            return;
        }
        history.frames_closed += 1;
        history.wall_ns = history.wall_ns.saturating_add(wall_ns);
        history.outside_ns = history
            .outside_ns
            .saturating_add(wall_ns.saturating_sub(presenter_self));
        for row in &rows {
            let agg = &mut history.agg[row.site];
            agg.self_ns = agg.self_ns.saturating_add(row.self_ns);
            agg.incl_ns = agg.incl_ns.saturating_add(row.incl_ns);
            agg.calls = agg.calls.saturating_add(row.calls);
            agg.slots |= 1 << row.slot;
            let slot = &mut history.slots[row.slot];
            slot.self_ns = slot.self_ns.saturating_add(row.self_ns);
            slot.kind[self.kind_of(row.site).index()] =
                slot.kind[self.kind_of(row.site).index()].saturating_add(row.self_ns);
        }
        history.frames.push_back(Frame {
            presented,
            title,
            wall_ns,
            rows,
        });
        if history.frames.len() > MOST_FRAMES {
            history.frames.pop_front();
        }
        let limit = self.cfg.frames != 0 && history.frames_closed >= self.cfg.frames;
        let due = {
            let last = self.last_report.lock().unwrap_or_else(|e| e.into_inner());
            now.saturating_duration_since(*last) >= self.cfg.every
        };
        if due || limit {
            let text = self.render(&history);
            drop(history);
            self.write(&text);
            // The write is not the frame's work: the next frame starts after
            // it, so its wall time does not carry the report.
            *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            *self.last_report.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
            if limit {
                self.running.store(false, Ordering::Relaxed);
            }
        }
    }

    /// What every counter gained since the last frame, as rows.
    fn deltas(&self) -> Vec<Row> {
        let sites = self.sites.lock().unwrap_or_else(|e| e.into_inner()).len();
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut rows = Vec::new();
        for site in 0..sites {
            for slot in 0..MOST_THREADS {
                let at = site * MOST_THREADS + slot;
                let totals = &self.totals[at];
                let (incl, own, calls) = (
                    totals.incl.load(Ordering::Relaxed),
                    totals.own.load(Ordering::Relaxed),
                    totals.calls.load(Ordering::Relaxed),
                );
                let before = &mut seen[at];
                if (incl, own, calls) == (before[0], before[1], before[2]) {
                    continue;
                }
                rows.push(Row {
                    site,
                    slot,
                    incl_ns: incl.saturating_sub(before[0]),
                    self_ns: own.saturating_sub(before[1]),
                    calls: calls.saturating_sub(before[2]),
                });
                *before = [incl, own, calls];
            }
        }
        if rows.len() > MOST_ROWS {
            rows.sort_unstable_by_key(|row| std::cmp::Reverse(row.self_ns));
            rows.truncate(MOST_ROWS);
        }
        rows
    }

    fn kind_of(&self, site: usize) -> Kind {
        let sites = self.sites.lock().unwrap_or_else(|e| e.into_inner());
        sites.get(site).map_or(Kind::Cpu, |site| site.kind)
    }

    /// Writes the report now (the frame limit and the interval call this).
    #[cfg(test)]
    fn write_report(&self) {
        let text = {
            let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
            self.render(&history)
        };
        self.write(&text);
    }

    /// Writes the report, making its directory.
    fn write(&self, text: &str) {
        let Some(path) = &self.cfg.file else {
            return;
        };
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, text);
    }

    fn name(&self, site: usize) -> &'static str {
        let sites = self.sites.lock().unwrap_or_else(|e| e.into_inner());
        sites.get(site).map_or("(gone)", |site| site.name)
    }

    /// The names of the threads a site ran on, joined by `+`.
    fn threads(&self, slots: u16) -> String {
        let names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = String::new();
        for slot in 0..MOST_THREADS {
            if slots & (1 << slot) == 0 {
                continue;
            }
            if !out.is_empty() {
                out.push('+');
            }
            let name = names.get(slot).map_or("", String::as_str);
            out.push_str(if name.is_empty() { "?" } else { name });
        }
        out
    }

    /// The report: the text written to the file.
    fn render(&self, history: &History) -> String {
        let frames = history.frames_closed.max(1) as f64;
        let wall_ms = history.wall_ns as f64 / 1e6;
        let mut text = String::new();
        let seconds = wall_ms / 1e3;
        let fps = if seconds > 0.0 { frames / seconds } else { 0.0 };
        let _ = writeln!(
            text,
            "VIctory frame profile: {} frames in {:.2} s ({:.1} frames/s), {} sites",
            history.frames_closed,
            seconds,
            fps,
            history.agg.iter().filter(|a| a.calls > 0).count()
        );
        let mut walls: Vec<f64> = history
            .frames
            .iter()
            .map(|f| f.wall_ns as f64 / 1e6)
            .collect();
        walls.sort_by(f64::total_cmp);
        let median = walls.get(walls.len() / 2).copied().unwrap_or(0.0);
        let worst = walls.last().copied().unwrap_or(0.0);
        let over = walls.iter().filter(|&&ms| ms > 16.7).count();
        let _ = writeln!(
            text,
            "frame time: {:.1} ms mean, {:.1} ms median, {:.1} ms max, {} of {} frames over 16.7 ms",
            wall_ms / frames,
            median,
            worst,
            over,
            walls.len()
        );
        if history.frames.len() == MOST_FRAMES {
            let _ = writeln!(
                text,
                "median and max are over the {MOST_FRAMES} frames still kept; the mean is over all {} frames",
                history.frames_closed
            );
        }
        if let (Some(first), Some(last)) = (history.frames.front(), history.frames.back()) {
            let _ = writeln!(
                text,
                "presented frames {} to {}, title frames {} to {}",
                first.presented, last.presented, first.title, last.title
            );
        }
        let _ = writeln!(
            text,
            "wall time {:.2} s, outside any scope {:.1}%",
            seconds,
            100.0 * history.outside_ns as f64 / history.wall_ns.max(1) as f64
        );

        self.render_ranked(&mut text, history, frames, false);
        self.render_ranked(&mut text, history, frames, true);
        self.render_frames(&mut text, history);
        self.render_composition(&mut text, history);
        self.render_threads(&mut text, history, frames);
        self.render_compare(&mut text, history, frames);
        self.render_machine(&mut text, history, frames);
        text
    }

    /// The self or inclusive table, ranked by the share of the frame budget.
    fn render_ranked(&self, text: &mut String, history: &History, frames: f64, inclusive: bool) {
        let mut rows: Vec<(usize, Agg)> = history
            .agg
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, agg)| agg.calls > 0)
            .collect();
        if inclusive {
            rows.sort_unstable_by_key(|(_, agg)| std::cmp::Reverse(agg.incl_ns));
        } else {
            rows.sort_unstable_by_key(|(_, agg)| std::cmp::Reverse(agg.self_ns));
        }
        if inclusive {
            let _ = writeln!(
                text,
                "-- with what it calls, ranked by inclusive time (nested functions count once each)"
            );
            let _ = writeln!(
                text,
                "  share  incl ms/f  self ms/f  calls/f  kind    thread  site"
            );
        } else {
            let _ = writeln!(
                text,
                "-- self time, ranked by the share of the frame budget it takes"
            );
            let _ = writeln!(
                text,
                "  share  self ms/f  incl ms/f  calls/f  kind    thread  site"
            );
        }
        for (site, agg) in rows.iter().take(24) {
            let ranked = if inclusive { agg.incl_ns } else { agg.self_ns };
            let share = 100.0 * ranked as f64 / history.wall_ns.max(1) as f64;
            let (first, second) = if inclusive {
                (
                    agg.incl_ns as f64 / frames / 1e6,
                    agg.self_ns as f64 / frames / 1e6,
                )
            } else {
                (
                    agg.self_ns as f64 / frames / 1e6,
                    agg.incl_ns as f64 / frames / 1e6,
                )
            };
            let _ = writeln!(
                text,
                "  {share:5.1}%  {:8.1}  {:10.1}  {:7.1}  {:<6}  {}  {}",
                first,
                second,
                agg.calls as f64 / frames,
                self.kind_of(*site).label(),
                self.threads(agg.slots),
                self.name(*site),
            );
        }
        if rows.len() > 24 {
            let _ = writeln!(text, "  ({} more)", rows.len() - 24);
        }
    }

    fn render_frames(&self, text: &mut String, history: &History) {
        if history.frames.is_empty() {
            return;
        }
        let _ = writeln!(text, "-- the worst frames");
        let mut frames: Vec<&Frame> = history.frames.iter().collect();
        frames.sort_unstable_by_key(|frame| std::cmp::Reverse(frame.wall_ns));
        for frame in frames.iter().take(5) {
            let _ = writeln!(
                text,
                "  frame {} (title {}): {:.1} ms",
                frame.presented,
                frame.title,
                frame.wall_ns as f64 / 1e6
            );
            let mut rows: Vec<&Row> = frame.rows.iter().collect();
            rows.sort_unstable_by_key(|row| std::cmp::Reverse(row.self_ns));
            for row in rows.iter().take(6) {
                let _ = writeln!(
                    text,
                    "    {:5.1}%  {:8.1} ms  {} ({})",
                    100.0 * row.self_ns as f64 / frame.wall_ns.max(1) as f64,
                    row.self_ns as f64 / 1e6,
                    self.name(row.site),
                    self.thread(row.slot),
                );
            }
        }
    }

    fn render_composition(&self, text: &mut String, history: &History) {
        let mut by_kind = [0u64; Kind::COUNT];
        for (site, agg) in history.agg.iter().enumerate() {
            by_kind[self.kind_of(site).index()] += agg.self_ns;
        }
        let wall = history.wall_ns.max(1) as f64;
        let _ = writeln!(text, "-- what the frames are made of (self time)");
        let mut line = String::new();
        for (index, kind) in [
            (0, Kind::Cpu),
            (1, Kind::Gpu),
            (2, Kind::Memory),
            (3, Kind::Sync),
            (4, Kind::Idle),
        ] {
            let _ = write!(
                line,
                "{} {:.1}%  ",
                kind.label(),
                100.0 * by_kind[index] as f64 / wall
            );
        }
        let _ = writeln!(
            line,
            "outside any scope {:.1}%",
            100.0 * history.outside_ns as f64 / wall
        );
        text.push_str(&line);
    }

    fn render_threads(&self, text: &mut String, history: &History, frames: f64) {
        let _ = writeln!(text, "-- threads");
        let names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        for (slot, agg) in history.slots.iter().enumerate() {
            if agg.self_ns == 0 {
                continue;
            }
            let name = names.get(slot).map_or("", String::as_str);
            let _ = writeln!(
                text,
                "  {:<20} {:8.1} ms/frame, {:5.1}% of the budget (cpu {:.1}%, gpu {:.1}%, \
                 memory {:.1}%, sync {:.1}%, idle {:.1}%)",
                if name.is_empty() { "?" } else { name },
                agg.self_ns as f64 / frames / 1e6,
                100.0 * agg.self_ns as f64 / history.wall_ns.max(1) as f64,
                100.0 * agg.kind[0] as f64 / history.wall_ns.max(1) as f64,
                100.0 * agg.kind[1] as f64 / history.wall_ns.max(1) as f64,
                100.0 * agg.kind[2] as f64 / history.wall_ns.max(1) as f64,
                100.0 * agg.kind[3] as f64 / history.wall_ns.max(1) as f64,
                100.0 * agg.kind[4] as f64 / history.wall_ns.max(1) as f64,
            );
        }
    }

    /// The change from the report `VICTORY_FRAME_PROFILE_COMPARE` names:
    /// before, after and the difference of the same functions, by self time
    /// per frame. Absent or unreadable, the section is left out.
    fn render_compare(&self, text: &mut String, history: &History, frames: f64) {
        let Some(path) = &self.cfg.compare else {
            return;
        };
        let Ok(old) = std::fs::read_to_string(path) else {
            let _ = writeln!(text, "-- compared with {}: not readable", path.display());
            return;
        };
        let before = machine_section(&old);
        if before.is_empty() {
            let _ = writeln!(
                text,
                "-- compared with {}: no machine section",
                path.display()
            );
            return;
        }
        let _ = writeln!(
            text,
            "-- compared with {} (self time per frame)",
            path.display()
        );
        let _ = writeln!(text, "  before     after    change   site");
        let mut rows: Vec<(String, f64, f64)> = Vec::new();
        for (site, agg) in history.agg.iter().enumerate() {
            if agg.calls == 0 {
                continue;
            }
            let name = self.name(site).to_owned();
            let Some(&old_ms) = before.get(&name) else {
                continue;
            };
            let new_ms = agg.self_ns as f64 / frames / 1e6;
            rows.push((name, old_ms, new_ms));
        }
        // Biggest change first, either way: an improvement is as much the
        // point of the comparison as a regression.
        rows.sort_unstable_by(|a, b| {
            let change = |(_, old, new): &(String, f64, f64)| new - old;
            change(b).abs().total_cmp(&change(a).abs())
        });
        for (name, old, new) in rows.iter().take(24) {
            let change = new - old;
            let percent = if *old > 0.0 {
                100.0 * change / old
            } else {
                0.0
            };
            let _ = writeln!(
                text,
                "  {old:8.1}  {new:8.1}  {change:+8.1}  ({percent:+.1}%)  {name}"
            );
        }
        let _ = writeln!(text, "  (ms per frame; a row needs the same site name)");
    }

    /// The machine-readable section the next run's comparison reads.
    fn render_machine(&self, text: &mut String, history: &History, frames: f64) {
        let _ = writeln!(
            text,
            "-- machine (tab separated: site, self ns per frame, calls per frame, share)"
        );
        let _ = writeln!(text, "site\tself_ns_per_frame\tcalls_per_frame\tshare");
        let mut rows: Vec<(usize, Agg)> = history
            .agg
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, agg)| agg.calls > 0)
            .collect();
        rows.sort_unstable_by_key(|(_, agg)| std::cmp::Reverse(agg.self_ns));
        for (site, agg) in rows {
            let _ = writeln!(
                text,
                "{}\t{:.1}\t{:.3}\t{:.4}",
                self.name(site),
                agg.self_ns as f64 / frames,
                agg.calls as f64 / frames,
                100.0 * agg.self_ns as f64 / history.wall_ns.max(1) as f64,
            );
        }
    }

    fn thread(&self, slot: usize) -> String {
        let names = self.names.lock().unwrap_or_else(|e| e.into_inner());
        names
            .get(slot)
            .filter(|name| !name.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("#{slot}"))
    }
}

/// The site names and self milliseconds per frame of a report's machine
/// section, for [`Profiler::render_compare`].
fn machine_section(report: &str) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    let mut inside = false;
    for line in report.lines() {
        if line.starts_with("-- machine") {
            inside = true;
            continue;
        }
        if !inside || line.starts_with("site\t") {
            continue;
        }
        let mut fields = line.split('\t');
        let (Some(name), Some(ns), Some(_calls), Some(_share)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if let Ok(ns) = ns.parse::<f64>() {
            // Stored as nanoseconds per frame; the comparison prints ms.
            out.insert(name.to_owned(), ns / 1e6);
        }
    }
    out
}

/// One open probe on this thread, with the time its children took.
struct Open {
    site: usize,
    start: Instant,
    children: u64,
}

/// A probe that is open: dropping it closes the scope.
#[must_use = "the probe closes when this is dropped"]
pub(crate) struct Guard {
    prof: &'static Profiler,
    site: usize,
    slot: usize,
    /// A guard closes the probe its own thread opened: it must not move to
    /// another thread (and cannot, being neither `Send` nor `Sync`).
    _thread: std::marker::PhantomData<*const ()>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let now = Instant::now();
        let closed = OPEN.with(|stack| {
            let mut stack = stack.borrow_mut();
            let entry = stack.pop()?;
            let elapsed = now.saturating_duration_since(entry.start).as_nanos() as u64;
            if let Some(parent) = stack.last_mut() {
                parent.children = parent.children.saturating_add(elapsed);
            }
            Some((entry, elapsed))
        });
        let Some((entry, elapsed)) = closed else {
            return;
        };
        debug_assert_eq!(
            entry.site, self.site,
            "probes closed out of order: {} closed while {} was innermost",
            self.site, entry.site
        );
        let own = elapsed.saturating_sub(entry.children);
        self.prof.add(self.site, self.slot, elapsed, own);
    }
}

thread_local! {
    /// The thread's slot in `Profiler::totals`, with the profiler it is for.
    static SLOT: Cell<(usize, u32)> = const { Cell::new((0, NO_SLOT)) };
    /// The thread's open probes, innermost last.
    static OPEN: RefCell<Vec<Open>> = const { RefCell::new(Vec::new()) };
}

/// The process's profiler, if the environment asked for one.
fn global() -> Option<&'static Profiler> {
    static GLOBAL: OnceLock<Option<&'static Profiler>> = OnceLock::new();
    *GLOBAL.get_or_init(|| {
        let cfg = Config::from_env();
        cfg.file.as_ref()?;
        Some(Box::leak(Box::new(Profiler::new(cfg))))
    })
}

/// Opens a probe for `site`. Call it as
///
/// ```ignore
/// #[cfg(feature = "diagnostics")]
/// let _prof = crate::frameprof::enter(&crate::frameprof::SUBMIT);
/// ```
///
/// and keep the value alive in a named local until the scope ends (`_prof`,
/// not `_`: the latter drops it at once).
pub(crate) fn enter(site: &'static Site) -> Option<Guard> {
    global()?.open(site)
}

/// A frame passed: the one that ended is closed and the next begins.
/// Called at present, before anything of the frame is measured.
pub(crate) fn frame_boundary(presented: u64, title: u64) {
    if let Some(prof) = global() {
        prof.close_frame(presented, title);
    }
}

// The probes the host opens. Names are the report's keys: keep them.
pub(crate) static SUBMIT: Site = Site::new("host::submit", "host", Kind::Cpu);
pub(crate) static SUBMIT_LOCK: Site = Site::new("host::submit.lock", "host", Kind::Sync);
pub(crate) static PREWARM: Site = Site::new("host::submit.prewarm", "host", Kind::Cpu);
pub(crate) static DRAIN: Site = Site::new("host::submit.drain", "host", Kind::Gpu);
pub(crate) static PREPARE: Site = Site::new("host::submit.prepare", "host", Kind::Cpu);
pub(crate) static EXECUTE: Site = Site::new("host::submit.execute", "host", Kind::Cpu);
pub(crate) static WAIT_GPU: Site = Site::new("host::submit.wait", "host", Kind::Gpu);
pub(crate) static COPY_BACK: Site = Site::new("host::finish.copy_back", "copy", Kind::Memory);
pub(crate) static READBACKS: Site = Site::new("host::finish.readbacks", "copy", Kind::Memory);
pub(crate) static WRITES: Site = Site::new("host::finish.writes", "host", Kind::Cpu);
pub(crate) static PARALLEL: Site = Site::new("host::copy.parallel", "copy", Kind::Memory);
pub(crate) static PRESENT: Site = Site::new("host::present", "present", Kind::Idle);
pub(crate) static SHADOW_UPDATE: Site = Site::new("shadow::update", "shadow", Kind::Memory);
pub(crate) static SHADOW_COPY_BACK: Site = Site::new("shadow::copy_back", "shadow", Kind::Memory);
pub(crate) static PRELUDE: Site = Site::new("list::prelude", "list", Kind::Cpu);
pub(crate) static DEFERRED: Site = Site::new("list::run_deferred", "list", Kind::Cpu);
pub(crate) static PLAN: Site = Site::new("descriptor::plan", "descriptor", Kind::Cpu);
pub(crate) static RECORD: Site = Site::new("descriptor::record", "descriptor", Kind::Cpu);
pub(crate) static BINDLESS_LOCK: Site = Site::new("descriptor::lock", "descriptor", Kind::Sync);
pub(crate) static FRAME_WAIT: Site = Site::new("frame::wait", "frame", Kind::Idle);
pub(crate) static DRAW: Site = Site::new("draw::record", "draw", Kind::Cpu);
pub(crate) static PRESENT_SHOW: Site = Site::new("queue::present", "present", Kind::Cpu);

#[cfg(test)]
mod tests {
    use super::*;

    fn site(name: &'static str, tag: &'static str, kind: Kind) -> &'static Site {
        Box::leak(Box::new(Site::new(name, tag, kind)))
    }

    fn profiler(name: &str, tags: &[&str]) -> &'static Profiler {
        let path = std::env::temp_dir().join(format!(
            "victory-frameprof-test-{}-{name}.txt",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        Box::leak(Box::new(Profiler::new(Config::for_test(path, tags))))
    }

    fn close(prof: &Profiler, presented: u64) {
        prof.close_frame(presented, 0);
    }

    /// The first frame after profiling starts is not counted (see
    /// `Profiler::close_frame`); tests close it before measuring.
    fn warm_up(prof: &Profiler) {
        std::thread::sleep(Duration::from_millis(1));
        close(prof, 0);
    }

    fn rows(prof: &Profiler, frame: usize) -> Vec<(usize, usize, u64, u64)> {
        let history = prof.history.lock().unwrap();
        history.frames[frame]
            .rows
            .iter()
            .map(|row| (row.site, row.slot, row.self_ns, row.calls))
            .collect()
    }

    #[test]
    fn scopes_land_in_the_frame_they_close_in() {
        let prof = profiler("frames", &[]);
        warm_up(prof);
        let a = site("a", "host", Kind::Cpu);
        let b = site("b", "host", Kind::Cpu);
        {
            let _a = prof.open(a).expect("profiling");
            std::thread::sleep(Duration::from_millis(2));
        }
        {
            let _b = prof.open(b).expect("profiling");
            std::thread::sleep(Duration::from_millis(2));
        }
        close(prof, 1);
        {
            let _b = prof.open(b).expect("profiling");
            std::thread::sleep(Duration::from_millis(2));
        }
        close(prof, 2);
        let first = rows(prof, 0);
        let second = rows(prof, 1);
        assert_eq!(first.iter().filter(|row| row.0 == 0).count(), 1, "a once");
        assert_eq!(first.iter().filter(|row| row.0 == 1).count(), 1, "b once");
        assert_eq!(second.iter().filter(|row| row.0 == 1).count(), 1, "b again");
        assert_eq!(
            second.iter().filter(|row| row.0 == 0).count(),
            0,
            "a closed"
        );
        for row in first.iter().chain(&second) {
            assert!(row.2 > 1_000_000, "a 2 ms scope measured {} ns", row.2);
        }
    }

    #[test]
    fn self_time_leaves_the_children_out() {
        let prof = profiler("self", &[]);
        warm_up(prof);
        let outer = site("outer", "host", Kind::Cpu);
        let inner = site("inner", "host", Kind::Cpu);
        {
            let _outer = prof.open(outer).expect("profiling");
            std::thread::sleep(Duration::from_millis(2));
            {
                let _inner = prof.open(inner).expect("profiling");
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        close(prof, 1);
        let history = prof.history.lock().unwrap();
        let agg = &history.agg;
        let outer_agg = agg[outer.id.load(Ordering::Acquire)];
        let inner_agg = agg[inner.id.load(Ordering::Acquire)];
        assert!(inner_agg.self_ns > 1_000_000, "inner measured");
        assert!(
            outer_agg.self_ns + inner_agg.self_ns <= outer_agg.incl_ns + 1_000_000,
            "self times add up to the inclusive time"
        );
        assert!(
            outer_agg.self_ns > 1_000_000,
            "outer's own 2 ms: {} ns",
            outer_agg.self_ns
        );
    }

    #[test]
    fn tags_leave_sites_out() {
        let prof = profiler("tags", &["copy"]);
        let copied = site("shadow::copy_back", "shadow", Kind::Memory);
        let other = site("host::submit", "host", Kind::Cpu);
        assert!(prof.open(copied).is_none(), "not selected");
        assert_eq!(copied.id.load(Ordering::Acquire), OFF);
        assert!(prof.open(other).is_none(), "not selected either");
        let selected = profiler("tags-selected", &["copy"]);
        let in_copy = site("copy::thing", "copy", Kind::Memory);
        assert!(selected.open(in_copy).is_some(), "selected by tag");
        // `-` is how the probes are priced against a run without them.
        let none = profiler("tags-none", &["-"]);
        let anything = site("copy::other", "copy", Kind::Memory);
        assert!(none.open(anything).is_none(), "no probe at all");
    }

    #[test]
    fn threads_are_kept_apart() {
        let prof = profiler("threads", &[]);
        warm_up(prof);
        let a = site("a", "host", Kind::Cpu);
        let b = site("b", "host", Kind::Cpu);
        let threads: Vec<_> = [a, b]
            .into_iter()
            .map(|site| {
                std::thread::spawn(move || {
                    let _guard = prof.open(site).expect("profiling");
                    std::thread::sleep(Duration::from_millis(2));
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        close(prof, 1);
        let history = prof.history.lock().unwrap();
        assert_eq!(history.slots.iter().filter(|s| s.self_ns > 0).count(), 2);
        let a_slot = history.agg[a.id.load(Ordering::Acquire)].slots;
        let b_slot = history.agg[b.id.load(Ordering::Acquire)].slots;
        assert_ne!(a_slot, b_slot, "one slot each");
    }

    #[test]
    fn report_ranks_by_share_of_the_budget() {
        let prof = profiler("report", &[]);
        warm_up(prof);
        let slow = site("slow::thing", "slow", Kind::Cpu);
        let fast = site("fast::thing", "fast", Kind::Cpu);
        for frame in 1..=3 {
            {
                let _slow = prof.open(slow).expect("profiling");
                std::thread::sleep(Duration::from_millis(8));
            }
            {
                let _fast = prof.open(fast).expect("profiling");
                std::thread::sleep(Duration::from_millis(2));
            }
            close(prof, frame);
        }
        // Force a write by asking for the report directly.
        let text = {
            let history = prof.history.lock().unwrap();
            prof.render(&history)
        };
        let line = |name: &str| {
            text.lines()
                .find(|line| line.ends_with(name))
                .unwrap_or_else(|| panic!("{name} in report:\n{text}"))
                .to_owned()
        };
        let slow_line = line("slow::thing");
        let fast_line = line("fast::thing");
        let share = |line: &str| {
            line.split_whitespace()
                .next()
                .unwrap()
                .trim_end_matches('%')
                .parse::<f64>()
                .unwrap()
        };
        assert!(
            share(&slow_line) > share(&fast_line),
            "slow ranks first:\n{text}"
        );
        assert!(text.contains("-- the worst frames"), "{text}");
        assert!(text.contains("-- machine"), "{text}");
        // The file itself is written with the same ranking.
        prof.write_report();
        let written = std::fs::read_to_string(prof.cfg.file.as_ref().unwrap()).unwrap();
        assert!(written.contains("slow::thing"), "{written}");
    }

    #[test]
    fn compare_reads_an_earlier_report() {
        let path = std::env::temp_dir().join(format!(
            "victory-frameprof-compare-{}.txt",
            std::process::id()
        ));
        let old = "-- machine (tab separated: site, self ns per frame, calls per frame, share)\n\
                   site\tself_ns_per_frame\tcalls_per_frame\tshare\n\
                   thing::slow\t20000000.0\t2.000\t0.2000\n";
        std::fs::write(&path, old).unwrap();
        let report = std::env::temp_dir().join(format!(
            "victory-frameprof-compare-out-{}.txt",
            std::process::id()
        ));
        let prof = Box::leak(Box::new(Profiler::new(Config {
            file: Some(report.clone()),
            tags: Vec::new(),
            every: Duration::from_secs(3600),
            frames: 0,
            compare: Some(path.clone()),
        })));
        warm_up(prof);
        let slow = site("thing::slow", "thing", Kind::Cpu);
        {
            let _slow = prof.open(slow).expect("profiling");
            std::thread::sleep(Duration::from_millis(4));
        }
        close(prof, 1);
        let text = {
            let history = prof.history.lock().unwrap();
            prof.render(&history)
        };
        println!("{text}");
        assert!(text.contains("-- compared with"), "{text}");
        assert!(text.contains("thing::slow"), "{text}");
        assert!(
            text.lines().any(|line| line.contains("thing::slow")
                && line.contains('%')
                && line.contains('(')),
            "a change row:\n{text}"
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn overhead_is_small_when_on_and_none_when_off() {
        let prof = profiler("overhead", &[]);
        let work = site("work", "host", Kind::Cpu);
        let count = 200_000u32;
        // Off: the running flag is the only work after the first call.
        prof.running.store(false, Ordering::Relaxed);
        let mut sink = 0u64;
        let started = Instant::now();
        for _ in 0..count {
            if let Some(guard) = prof.open(work) {
                std::hint::black_box(&guard);
                sink += 1;
            }
        }
        let off = started.elapsed().as_nanos() as f64 / f64::from(count);
        prof.running.store(true, Ordering::Relaxed);
        let started = Instant::now();
        for _ in 0..count {
            let guard = prof.open(work);
            std::hint::black_box(&guard);
            drop(guard);
            sink += 1;
        }
        let on = started.elapsed().as_nanos() as f64 / f64::from(count);
        println!(
            "probe cost: {off:.1} ns open+close with profiling off, {on:.1} ns with it on \
             ({} probes)",
            sink / 2
        );
        assert!(off < 100.0, "off costs {off} ns");
        assert!(on < 2_000.0, "on costs {on} ns");
    }

    #[test]
    fn a_frame_without_a_scope_has_no_rows() {
        let prof = profiler("empty", &[]);
        warm_up(prof);
        std::thread::sleep(Duration::from_millis(2));
        close(prof, 1);
        let history = prof.history.lock().unwrap();
        assert_eq!(history.frames_closed, 1);
        assert!(history.frames[0].rows.is_empty());
        assert!(history.outside_ns > 1_000_000, "all outside a scope");
    }

    #[test]
    fn the_entry_point_costs_nothing_when_profiling_is_off() {
        if std::env::var_os("VICTORY_FRAME_PROFILE").is_some() {
            println!("skipped: this run has VICTORY_FRAME_PROFILE set");
            return;
        }
        let site = site("off::thing", "off", Kind::Cpu);
        let count = 1_000_000u32;
        // The first call reads the environment and initialises the global:
        // not what a frame pays.
        std::hint::black_box(enter(site));
        let started = Instant::now();
        for _ in 0..count {
            std::hint::black_box(enter(site));
        }
        let ns = started.elapsed().as_nanos() as f64 / f64::from(count);
        println!("entry point with profiling off: {ns:.2} ns per probe");
        assert!(ns < 100.0, "off costs {ns} ns per probe");
    }

    #[test]
    fn frame_walls_add_up_to_the_run() {
        let prof = profiler("walls", &[]);
        // The clock starts before the warm-up, which the frames do not
        // include: the frames are the run minus that first stretch.
        let started = Instant::now();
        warm_up(prof);
        for frame in 1..=5u64 {
            std::thread::sleep(Duration::from_millis(6));
            close(prof, frame);
        }
        let elapsed = started.elapsed().as_nanos() as u64;
        let wall = prof.history.lock().unwrap().wall_ns;
        assert!(wall <= elapsed, "the frames cannot outlast the run");
        assert!(
            wall + 5_000_000 >= elapsed,
            "the frames {wall} ns of a run of {elapsed} ns"
        );
    }

    #[test]
    fn a_sample_report() {
        let prof = profiler("sample", &[]);
        warm_up(prof);
        let submit = site("host::submit", "host", Kind::Cpu);
        let copy = site("shadow::copy_back", "shadow", Kind::Memory);
        let wait = site("host::submit.wait", "host", Kind::Gpu);
        for frame in 1..=6u64 {
            let _submit = prof.open(submit).expect("profiling");
            std::thread::sleep(Duration::from_millis(4));
            {
                let _copy = prof.open(copy).expect("profiling");
                std::thread::sleep(Duration::from_millis(3));
            }
            {
                let _wait = prof.open(wait).expect("profiling");
                std::thread::sleep(Duration::from_millis(2));
            }
            drop(_submit);
            close(prof, frame);
        }
        let text = {
            let history = prof.history.lock().unwrap();
            prof.render(&history)
        };
        println!("{text}");
        assert!(text.contains("VIctory frame profile"));
    }
}
