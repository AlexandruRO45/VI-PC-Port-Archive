//! Wall-clock time for which at least one title bind is waiting for a PSO.

use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

pub(super) static CLOCK: OnceLock<HoldClock> = OnceLock::new();

#[derive(Default)]
pub(super) struct HoldClock {
    state: Mutex<Intervals>,
    changed: Condvar,
}

#[derive(Default)]
struct Intervals {
    active: usize,
    since: Option<Instant>,
    total: Duration,
}

impl Intervals {
    fn enter(&mut self, now: Instant) {
        if self.active == 0 {
            self.since = Some(now);
        }
        self.active += 1;
    }

    fn leave(&mut self, now: Instant) {
        self.active -= 1;
        if self.active == 0 {
            self.total = self.occupied_at(now);
            self.since = None;
        }
    }

    fn occupied_at(&self, now: Instant) -> Duration {
        self.total + self.since.map_or(Duration::ZERO, |at| now - at)
    }
}

pub(super) struct PipelineHold<'a>(&'a HoldClock);

impl HoldClock {
    pub(super) fn enter(&self) -> PipelineHold<'_> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.enter(Instant::now());
        if state.active == 1 {
            self.changed.notify_all();
        }
        PipelineHold(self)
    }

    #[cfg(test)]
    pub(super) fn unoccupied(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.active == 0 && state.total.is_zero()
    }

    pub(super) fn sleep(&self, requested: Duration, cap: Duration) -> Duration {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let tick = Tick::start(&state, Instant::now(), requested, cap);
        loop {
            let (until, extension) = tick.remaining(&state, Instant::now());
            if until.is_zero() {
                return extension;
            }
            // The same mutex protects both the calculation and the wait,
            // so a begin/end transition cannot get lost between them.
            state = self.changed.wait_timeout(state, until).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

impl Drop for PipelineHold<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        state.leave(Instant::now());
        if state.active == 0 {
            self.0.changed.notify_all();
        }
    }
}

struct Tick {
    started: Instant,
    prior: Duration,
    requested: Duration,
    cap: Duration,
}

impl Tick {
    fn start(state: &Intervals, now: Instant, requested: Duration, cap: Duration) -> Self {
        Self { started: now, prior: state.occupied_at(now), requested, cap }
    }

    /// Until the next necessary wake, and the extension accrued so far.
    fn remaining(&self, state: &Intervals, now: Instant) -> (Duration, Duration) {
        let elapsed = now - self.started;
        let extension = (state.occupied_at(now) - self.prior).min(self.cap);
        let left = (self.requested + extension).saturating_sub(elapsed);
        if !left.is_zero() && state.active != 0 {
            // While a hold continues, the remaining guest time cannot
            // decrease until the cap is reached. Wake on a transition or
            // at the absolute deadline, not repeatedly on the guest time.
            ((self.requested + self.cap).saturating_sub(elapsed), extension)
        } else {
            (left, extension)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn a_hold_during_a_tick_extends_it_only_until_the_bind_unblocks() {
        let start = Instant::now();
        let at = |n| start + ms(n);
        let mut state = Intervals::default();
        let tick = Tick::start(&state, at(0), ms(1000), ms(500));
        assert_eq!(tick.remaining(&state, at(0)), (ms(1000), ms(0)));
        state.enter(at(400));
        assert_eq!(tick.remaining(&state, at(400)), (ms(1100), ms(0)));
        state.leave(at(580));
        assert_eq!(tick.remaining(&state, at(580)), (ms(600), ms(180)));
        assert_eq!(tick.remaining(&state, at(1180)), (ms(0), ms(180)));
    }

    #[test]
    fn overlapping_and_separate_holds_count_only_their_union_inside_the_tick() {
        let start = Instant::now();
        let at = |n| start + ms(n);
        let mut state = Intervals::default();
        state.enter(at(0));
        let tick = Tick::start(&state, at(100), ms(1000), ms(500));
        state.enter(at(200));
        state.leave(at(240));
        state.leave(at(310));
        state.enter(at(400));
        state.leave(at(460));
        assert_eq!(tick.remaining(&state, at(460)), (ms(910), ms(270)));
        assert_eq!(tick.remaining(&state, at(1370)), (ms(0), ms(270)));
    }

    #[test]
    fn the_tick_expires_even_if_a_bind_never_unblocks() {
        let start = Instant::now();
        let mut state = Intervals::default();
        state.enter(start);
        let tick = Tick::start(&state, start, ms(1000), ms(500));
        assert_eq!(tick.remaining(&state, start + ms(1499)), (ms(1), ms(500)));
        assert_eq!(tick.remaining(&state, start + ms(1500)), (ms(0), ms(500)));
    }

    #[test]
    fn dropping_nested_guards_leaves_the_clock_idle() {
        let clock = HoldClock::default();
        let outer = clock.enter();
        let inner = clock.enter();
        drop(outer);
        assert_eq!(clock.state.lock().unwrap().active, 1);
        drop(inner);
        let state = clock.state.lock().unwrap();
        assert_eq!(state.active, 0);
        assert!(state.since.is_none());
    }
}
