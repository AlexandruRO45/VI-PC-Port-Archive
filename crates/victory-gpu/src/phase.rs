//! Where a frame's wall clock goes, printed in the meter's line every second
//! (`VICTORY_PHASE=1`).
//!
//! The frame line accounts for what we submit (preparing, the card, copying
//! back) and the copy line for the recording threads' list calls, and then
//! leaves most of a frame unattributed: in the world at 2,731 draws a frame
//! and 3,554.9 ms a frame, 1,402.6 ms was submitting and 612.3 ms recording,
//! and 2,152 ms was in no counter at all. This splits the rest where it runs —
//! the binding a draw does before it records, the shadow and address-space
//! bookkeeping that keeps track of what the title's own memory holds, a
//! pipeline's making, the submission itself, and the queue's loop around it.
//!
//! Nested phases are summed separately and said so: a draw is inside the
//! recording, binding and the shadow lookups are inside the draw, and a
//! submission is inside the queue's loop.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

/// `VICTORY_PHASE=1`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_PHASE").is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// The phases, in the order the line names them.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    /// A title command-list call, from the call to the end of its recording.
    Record = 0,
    /// Of a record: a draw, a dispatch or an indirect execution.
    Draw,
    /// Of a draw: binding its pipeline, roots and tables.
    Bind,
    /// The shadow and address-space lookups, wherever they run.
    Shadow,
    /// A whole pipeline made in its worker, translation and driver included.
    Translate,
    /// A whole submission, preparing to copying back.
    Submit,
    /// A queue's `ExecuteCommandLists`, the submission and its loop inside.
    Loop,
}

const PHASES: usize = 7;

static US: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];
static CALLS: [AtomicU64; PHASES] = [const { AtomicU64::new(0) }; PHASES];

/// Times the block it is held for as `phase`; does nothing at all when the
/// switch is off.
pub struct Span(usize, Option<Instant>);

impl Drop for Span {
    fn drop(&mut self) {
        let Some(started) = self.1 else { return };
        US[self.0].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
    }
}

pub fn at(phase: Phase) -> Span {
    let slot = phase as usize;
    if !on() {
        return Span(slot, None);
    }
    CALLS[slot].fetch_add(1, Relaxed);
    Span(slot, Some(Instant::now()))
}

/// What one phase cost a frame, and how many calls it took.
fn take(phase: Phase) -> (u64, u64) {
    let slot = phase as usize;
    (US[slot].swap(0, Relaxed), CALLS[slot].swap(0, Relaxed))
}

/// The line, in the meter's shape: milliseconds a frame and the calls they
/// took, for the parts the frame line leaves out.
///
/// The phases are summed over threads, so they do not add up to a frame's
/// wall clock and are not presented as if they did: a pipeline is made on a
/// worker while the title's own threads record, and the queue's loop runs on
/// the submitting thread beside them.
pub(crate) fn line(frames: f64, ms: f64) {
    if !on() {
        return;
    }
    let per = |phase: Phase| {
        let (us, calls) = take(phase);
        (us as f64 / 1000.0 / frames, calls as f64 / frames)
    };
    let (record, record_calls) = per(Phase::Record);
    let (draw, draw_calls) = per(Phase::Draw);
    let (bind, _) = per(Phase::Bind);
    let (shadow, shadow_calls) = per(Phase::Shadow);
    let (translate, translate_calls) = per(Phase::Translate);
    let (submit, submit_calls) = per(Phase::Submit);
    let (queue, _) = per(Phase::Loop);
    tracing::info!(
        "where a frame's time goes per frame, of {ms:.1} ms: {queue:.1} ms in the queues' loop, {submit:.1} ms of it submitting ({submit_calls:.1} submissions); beside it on the title's threads {record:.1} ms recording {record_calls:.0} list calls, of it {draw:.1} ms in {draw_calls:.0} draws and dispatches ({bind:.1} ms binding them, {shadow:.1} ms in {shadow_calls:.0} shadow and address-space lookups); and {translate:.1} ms making {translate_calls:.1} pipelines. These are sums over threads: recording and making run beside the loop, so they can exceed the frame's wall clock"
    );
}
