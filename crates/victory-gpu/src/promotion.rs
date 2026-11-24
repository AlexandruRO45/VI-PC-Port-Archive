//! The read states a command list's own bindings promote title buffers'
//! host copies to.
//!
//! Those copies stay in the common state, and every use promotes them
//! implicitly (`host`). Promotion to read states adds up, so after a
//! list binds a buffer as vertex data and as a root shader resource, it is
//! in both states. A later explicit transition of that buffer has to start
//! from exactly that combination. The indirect path makes such a
//! transition, so each list notes what its bindings promoted. The list
//! ends by returning each promoted buffer to the common state, which
//! leaves the next list of the same submission free to promote it to
//! anything.
//!
//! shortcut: only the list's own root and input-assembler bindings are
//! noted. Accesses through descriptor tables are not, as the host does not
//! track them either (without GPU-based validation).

use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Per list: by host buffer, the buffer and the read states the list's
/// bindings promoted it to, or `Held` while the list keeps it in an
/// explicit state of its own.
pub(crate) struct Promotions<R> {
    buffers: HashMap<usize, (Use, R)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    Read(u32),
    Held,
}

impl<R> Default for Promotions<R> {
    fn default() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }
}

impl<R> Promotions<R> {
    /// A binding reads buffer `key` in read state `state`. Nothing while
    /// the list holds the buffer in an explicit state.
    pub(crate) fn read(&mut self, key: usize, state: u32, buffer: impl FnOnce() -> R) {
        let (u, _) = self
            .buffers
            .entry(key)
            .or_insert_with(|| (Use::Read(0), buffer()));
        if let Use::Read(s) = u {
            *s |= state;
        }
    }

    /// The list takes buffer `key` out of implicit promotion: the read
    /// states it is in (0: still common). It is held until `release`.
    pub(crate) fn hold(&mut self, key: usize, buffer: impl FnOnce() -> R) -> u32 {
        match self.buffers.entry(key) {
            Entry::Vacant(v) => {
                v.insert((Use::Held, buffer()));
                0
            }
            Entry::Occupied(mut o) => match std::mem::replace(&mut o.get_mut().0, Use::Held) {
                Use::Read(s) => s,
                Use::Held => 0,
            },
        }
    }

    /// The list returned buffer `key` to the common state.
    pub(crate) fn release(&mut self, key: usize) {
        self.buffers.remove(&key);
    }

    /// At the list's end: each buffer still promoted, with its read states.
    /// Everything is forgotten. A buffer still held is left out, as the
    /// list did not return it.
    pub(crate) fn drain(&mut self) -> Vec<(R, u32)> {
        self.buffers
            .drain()
            .filter_map(|(_, (u, r))| match u {
                Use::Read(s) if s != 0 => Some((r, s)),
                _ => None,
            })
            .collect()
    }

    /// Whether a buffer is still held (a list that ends so has a bug).
    pub(crate) fn holds_any(&self) -> bool {
        self.buffers.values().any(|(u, _)| *u == Use::Held)
    }
}

#[cfg(test)]
mod tests {
    use super::Promotions;

    const VERTEX: u32 = 0x1;
    const INDEX: u32 = 0x2;
    const NON_PIXEL: u32 = 0x40;

    #[test]
    fn reads_add_up_until_held() {
        let mut p = Promotions::<()>::default();
        p.read(1, VERTEX, || ());
        p.read(1, NON_PIXEL, || ());
        assert_eq!(p.hold(1, || ()), VERTEX | NON_PIXEL);
        // While held, bindings do not promote it.
        p.read(1, INDEX, || ());
        assert!(p.holds_any());
        p.release(1);
        assert!(!p.holds_any());
        assert!(p.drain().is_empty());
    }

    #[test]
    fn an_untouched_buffer_is_held_from_common() {
        let mut p = Promotions::<()>::default();
        assert_eq!(p.hold(7, || ()), 0);
        p.release(7);
        // After its release, it is promoted again.
        p.read(7, INDEX, || ());
        assert_eq!(p.drain(), vec![((), INDEX)]);
    }

    #[test]
    fn the_end_returns_each_promoted_buffer_once() {
        let mut p = Promotions::<u8>::default();
        p.read(1, VERTEX, || 1);
        p.read(2, INDEX, || 2);
        p.read(2, NON_PIXEL, || 2);
        p.hold(3, || 3);
        let mut ended = p.drain();
        ended.sort();
        assert_eq!(ended, vec![(1, VERTEX), (2, INDEX | NON_PIXEL)]);
        assert!(p.drain().is_empty());
    }
}
