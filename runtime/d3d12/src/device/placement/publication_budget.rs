//! A shared wall-clock budget for CPU waits on overlapping placed images.
use std::time::{Duration, Instant};

pub(super) struct Budget {
    limit: Duration,
    start: Option<Instant>,
}

impl Budget {
    pub(super) fn new(ms: u32) -> Self {
        Self { limit: Duration::from_millis(u64::from(ms)), start: None }
    }

    // Start only at the first incomplete publication. Already completed
    // fences are checked by the caller even after exhaustion.
    pub(super) fn remaining_ms(&mut self, now: Instant) -> Option<u32> {
        let start = *self.start.get_or_insert(now);
        let left = self.limit.checked_sub(now.saturating_duration_since(start))?;
        if left.is_zero() { return None; }
        // Round down so a sub-millisecond remainder cannot start a new wait.
        let ms = left.as_millis() as u32;
        (ms != 0).then_some(ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_is_shared_across_every_overlap() {
        let t = Instant::now();
        let mut budget = Budget::new(2000);
        assert_eq!(budget.remaining_ms(t), Some(2000));
        assert_eq!(budget.remaining_ms(t + Duration::from_millis(750)), Some(1250));
        assert_eq!(budget.remaining_ms(t + Duration::from_millis(1999)), Some(1));
        for n in 2000..10000 {
            assert_eq!(budget.remaining_ms(t + Duration::from_millis(n)), None);
        }
    }

    #[test]
    fn submillisecond_and_zero_cannot_restart_wait() {
        let t = Instant::now();
        let mut budget = Budget::new(2);
        assert_eq!(budget.remaining_ms(t), Some(2));
        assert_eq!(budget.remaining_ms(t + Duration::from_micros(1500)), None);
        assert_eq!(Budget::new(0).remaining_ms(t), None);
    }
}
