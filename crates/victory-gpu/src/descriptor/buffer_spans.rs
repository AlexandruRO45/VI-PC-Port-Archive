use std::collections::HashMap;
use std::sync::Arc;

pub(crate) type BufferRange<T> = (Arc<T>, u64, u64);

/// The ranges to refresh for an indexed draw. A pass keeps its snapshot
/// alive after releasing the heap lock, even if a later pass replaces it.
pub(crate) struct BufferSpans<T> {
    pub(crate) ranges: Vec<BufferRange<T>>,
    pub(crate) views: u32,
    pub(crate) bytes: u64,
}

/// Slot resolutions and their grouped spans belong to the descriptor heap
/// whose indices they describe. Dropping the heap releases both.
pub(crate) struct BufferSlots<T> {
    passes: u64,
    slots: HashMap<u32, BufferRange<T>>,
    spans: Option<Arc<BufferSpans<T>>>,
}

impl<T> Default for BufferSlots<T> {
    fn default() -> Self {
        Self {
            passes: 0,
            slots: HashMap::new(),
            spans: None,
        }
    }
}

impl<T> BufferSlots<T> {
    pub(crate) fn pass(
        &mut self,
        changed: Vec<u32>,
        all: impl FnOnce() -> Vec<u32>,
        mut resolve: impl FnMut(usize, u32, &[u32]) -> Option<BufferRange<T>>,
    ) -> (usize, Arc<BufferSpans<T>>) {
        // A resource may go stale or gain a host copy without a slot write.
        let full = self.passes.is_multiple_of(16);
        let slots = if full {
            self.spans = None;
            self.slots.clear();
            all()
        } else {
            changed
        };
        self.passes += 1;
        for (n, &slot) in slots.iter().enumerate() {
            match resolve(n, slot, &slots) {
                Some(range) => {
                    if !full
                        && self.slots.get(&slot).is_some_and(|old| {
                            Arc::ptr_eq(&old.0, &range.0) && old.1 == range.1 && old.2 == range.2
                        })
                    {
                        continue;
                    }
                    self.slots.insert(slot, range);
                }
                None if self.slots.remove(&slot).is_none() => continue,
                None => {}
            }
            self.spans = None;
        }

        let spans = self.spans.get_or_insert_with(|| {
            let mut groups = HashMap::new();
            for (shadow, from, to) in self.slots.values() {
                let key = Arc::as_ptr(shadow);
                let span = groups
                    .entry(key)
                    .or_insert_with(|| (shadow.clone(), *from, *to));
                span.1 = span.1.min(*from);
                span.2 = span.2.max(*to);
            }
            let ranges: Vec<_> = groups.into_values().collect();
            let bytes = ranges.iter().map(|(_, from, to)| to - from).sum();
            Arc::new(BufferSpans {
                ranges,
                views: self.slots.len() as u32,
                bytes,
            })
        });
        (slots.len(), Arc::clone(spans))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(
        cache: &mut BufferSlots<u8>,
        slots: &[Option<BufferRange<u8>>],
        changed: &[u32],
    ) -> (usize, Arc<BufferSpans<u8>>) {
        cache.pass(
            changed.to_vec(),
            || {
                slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.as_ref().map(|_| i as u32))
                    .collect()
            },
            |n, i, visit| {
                assert_eq!(visit[n], i);
                slots[i as usize].clone()
            },
        )
    }

    fn sorted(snapshot: &BufferSpans<u8>) -> Vec<(u8, u64, u64)> {
        let mut result: Vec<_> = snapshot
            .ranges
            .iter()
            .map(|(v, lo, hi)| (**v, *lo, *hi))
            .collect();
        result.sort_unstable();
        result
    }

    #[test]
    fn unchanged_ranges_reuse_the_snapshot_and_changes_replace_it() {
        let (a, b) = (Arc::new(1), Arc::new(2));
        let mut slots = vec![
            Some((a.clone(), 10, 15)),
            Some((a.clone(), 20, 26)),
            Some((b.clone(), 10, 18)),
            None,
        ];
        let mut cache = BufferSlots::default();
        let (visited, first) = pass(&mut cache, &slots, &[]);
        assert_eq!((visited, first.views, first.bytes), (3, 3, 24));
        assert_eq!(sorted(&first), [(1, 10, 26), (2, 10, 18)]);
        let (visited, same) = pass(&mut cache, &slots, &[1, 3]);
        assert_eq!(visited, 2);
        assert!(Arc::ptr_eq(&first, &same));

        slots[1] = None;
        let (_, removed) = pass(&mut cache, &slots, &[1]);
        assert_eq!(sorted(&removed), [(1, 10, 15), (2, 10, 18)]);
        slots[0] = Some((b, 10, 15)); // Same addresses, different host copy.
        let (_, replaced) = pass(&mut cache, &slots, &[0]);
        assert_eq!(sorted(&replaced), [(2, 10, 18)]);
        slots[2] = Some((a, 7, 19));
        let (_, moved) = pass(&mut cache, &slots, &[2]);
        assert_eq!(sorted(&moved), [(1, 7, 19), (2, 10, 15)]);
        assert_eq!((moved.views, moved.bytes), (2, 17));
        let replacement = Arc::new(1); // Equal payload, distinct underlying allocation.
        slots[2] = Some((replacement.clone(), 7, 19));
        let (_, new_owner) = pass(&mut cache, &slots, &[2]);
        assert!(!Arc::ptr_eq(&moved, &new_owner));
        assert!(
            new_owner
                .ranges
                .iter()
                .any(|r| Arc::ptr_eq(&r.0, &replacement))
        );
        // A pass may still be refreshing its earlier immutable snapshot.
        assert_eq!(sorted(&first), [(1, 10, 26), (2, 10, 18)]);
    }

    #[test]
    fn each_heap_rescans_on_its_own_sixteenth_interval_and_releases_its_copies() {
        let copy = Arc::new(1);
        let weak = Arc::downgrade(&copy);
        let mut slots = vec![Some((copy, 0, 8)), None];
        let mut first = BufferSlots::default();
        let mut second = BufferSlots::default();
        let (visited, previous) = pass(&mut first, &slots, &[]);
        assert_eq!(visited, 1);
        let previous_span = Arc::downgrade(&previous);
        drop(previous);
        slots[0] = None; // This slot is absent from the next full enumeration.
        slots[1] = Some((Arc::new(2), 16, 20));
        assert_eq!(sorted(&pass(&mut second, &slots, &[]).1), [(2, 16, 20)]);
        for _ in 0..15 {
            let (visited, current) = pass(&mut first, &slots, &[]);
            assert_eq!(visited, 0);
            assert!(Arc::ptr_eq(&current, &previous_span.upgrade().unwrap()));
        }
        let (visited, current) = pass(&mut first, &slots, &[]);
        assert_eq!(visited, 1);
        assert_eq!(sorted(&current), [(2, 16, 20)]);
        assert!(weak.upgrade().is_none());
        assert!(previous_span.upgrade().is_none());
        let remaining = Arc::downgrade(&slots[1].as_ref().unwrap().0);
        drop(slots);
        drop(current);
        drop(first);
        assert!(remaining.upgrade().is_some());
        drop(second);
        assert!(remaining.upgrade().is_none());
    }
}
