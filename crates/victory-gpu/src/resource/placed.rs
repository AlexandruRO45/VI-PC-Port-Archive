//! Owners for textures the title lays out without creating a resource object.

use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering::{Acquire, Release};
use std::sync::{Arc, Mutex, Weak};

const COLD_BYTES: u64 = 256 << 20;
type Key = (u64, [u32; 14]);

pub(crate) struct Placed(object::Held);
// SAFETY: this holder is used only for Resource, whose mutable host state
// is atomic or locked. Dropping the last Arc releases its one COM reference.
unsafe impl Sync for Placed {}

impl Placed {
    pub(crate) fn get(&self) -> This {
        self.0.get()
    }

    pub(crate) fn stale(&self) -> bool {
        was_remapped(self.get()) || (self.reconstructible() && !current(self.get()))
    }

    fn reconstructible(&self) -> bool {
        resource(self.get()).materialized().is_some_and(|c| {
            crate::texture::title_written(&c.texture)
                && !c.host_written.load(Acquire)
                && !matches!(c.twin.get(), Some(Some(_)))
        })
    }

    fn remap(&self) {
        let r = resource(self.get());
        r.remapped.store(true, Release);
        let (Some(key), Some(copy)) = (r.key, r.materialized()) else {
            return;
        };
        let mut memory = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = memory.as_mut()
            && map
                .get(&key)
                .is_some_and(|w| w.as_ptr() == Arc::as_ptr(copy))
        {
            map.remove(&key);
        }
    }
}

struct Entry {
    owner: Arc<Placed>,
    end: u64,
    /// Nominal D3D12 bytes of a potentially reconstructible entry.
    bytes: u64,
    used: u64,
}

type Detached = (u64, u64, Weak<Placed>);

#[derive(Default)]
struct Cache {
    entries: BTreeMap<Key, Entry>,
    detached: Vec<Detached>,
    clock: u64,
    bytes: u64,
}

static CACHE: std::sync::LazyLock<Mutex<Cache>> = std::sync::LazyLock::new(Default::default);

impl Cache {
    fn detach(&mut self, key: &Key) -> Option<Arc<Placed>> {
        let e = self.entries.remove(key)?;
        self.bytes -= e.bytes;
        Some(e.owner)
    }

    fn trim(&mut self, budget: u64, removed: &mut Vec<Arc<Placed>>) {
        if self.bytes <= budget {
            return;
        }
        // A slot and an in-progress lookup hold Placed; recorded lists hold
        // HostTexture. Drop neither, even if we cannot reach the target.
        let mut cold: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(key, e)| {
                if e.bytes == 0 || Arc::strong_count(&e.owner) != 1 || !e.owner.reconstructible() {
                    return None;
                }
                let host = resource(e.owner.get()).materialized()?;
                let len = usize::try_from(e.end - key.0).ok()?;
                // Do not discard the last readable copy if the mapping is
                // incomplete. A subsequent real remap invalidates it instead.
                (Arc::strong_count(host) == 1 && crate::texture::committed(key.0 as *const u8, len))
                    .then_some((e.used, *key))
            })
            .collect();
        cold.sort_unstable();
        for (_, key) in cold {
            if self.bytes <= budget {
                break;
            }
            removed.extend(self.detach(&key));
        }
    }
}

/// Made on first use. Acquire the caller's owner under the cache lock;
/// neither a replaced entry nor remap may free its resource under a reader.
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<Arc<Placed>> {
    let mut desc = *desc;
    // These two words are padding in the console description.
    (desc[1], desc[13]) = (0, 0);
    // SAFETY: 56 initialized bytes in ResourceDesc's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    let key = (address, desc);
    // Drop outside the cache lock: Release unlinks descriptor/alias state.
    let mut removed = Vec::new();
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    cache.clock += 1;
    let now = cache.clock;
    if let Some(e) = cache.entries.get_mut(&key)
        && !e.owner.stale()
    {
        e.used = now;
        let owner = e.owner.clone();
        if now.is_multiple_of(64) {
            cache.trim(COLD_BYTES, &mut removed);
        }
        return Some(owner);
    }
    if let Some(e) = cache.entries.get(&key) {
        let old = (address, e.end, Arc::downgrade(&e.owner));
        if cache.detached.len().is_power_of_two() {
            cache.detached.retain(|(_, _, w)| w.strong_count() != 0);
        }
        cache.detached.push(old);
        removed.extend(cache.detach(&key));
    }
    let mut out = std::ptr::null();
    if create_at_address(
        std::ptr::null(),
        address,
        &d,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    ) != S_OK
    {
        return None;
    }
    // SAFETY: successful creation returned one owned reference.
    let owner = Arc::new(Placed(unsafe { object::Held::adopt(out) }));
    let Some(copy) = resource(out).materialize() else {
        removed.push(owner);
        return None;
    };
    let layout = texture_layout(&d);
    let end = address.saturating_add(layout.as_ref().map_or(u64::MAX, |t| t.size.max(1)));
    let bytes = if copy.bytes != u64::MAX
        && owner.reconstructible()
        && d.sample_count <= 1
        && layout.is_some_and(|t| t.planes.len() == 1 && !t.planes[0].layout.desc.swizzle.xor())
    {
        copy.bytes
    } else {
        0
    };
    cache.bytes += bytes;
    cache.entries.insert(
        key,
        Entry {
            owner: owner.clone(),
            end,
            bytes,
            used: now,
        },
    );
    // Limit cold bytes on insertion. Pins may force us to keep more than
    // the target; periodic hits retry after those pins are released.
    cache.trim(COLD_BYTES, &mut removed);
    Some(owner)
}

/// Drop cache ownership of old generations (including partial overlaps).
/// Old descriptors/lists continue to own their copies, but cannot upload
/// from the new mapping into an old generation.
pub(crate) fn invalidate_placed(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (at, end) = (at as u64, at.saturating_add(len) as u64);
    let mut removed = Vec::new();
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<_> = cache
        .entries
        .range(..(end, [0; 14]))
        .filter(|(_, e)| e.end > at)
        .map(|(k, _)| *k)
        .collect();
    for key in keys {
        let owner = cache.detach(&key).expect("entry under the cache lock");
        owner.remap();
        removed.push(owner);
    }
    cache.detached.retain(|(start, old_end, weak)| {
        if *start < end && *old_end > at {
            if let Some(owner) = weak.upgrade() {
                owner.remap();
                removed.push(owner);
            }
            false
        } else {
            weak.strong_count() != 0
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba(flags: u32) -> [u32; 14] {
        [
            3,
            0,
            0,
            0,
            1024,
            0,
            1024,
            1 | 1 << 16,
            28,
            1,
            0,
            0x105,
            flags,
            0,
        ]
    }

    #[test]
    fn placed_trim_and_remap_preserve_pins() {
        let backing = Memory::zeroed(16 << 20, 1 << 16).unwrap();
        let start = backing.address();
        let a = texture_at(start, &rgba(0)).unwrap();
        let b = texture_at(start + (4 << 20), &rgba(0)).unwrap();
        let c = texture_at(start + (8 << 20), &rgba(0)).unwrap();
        let d = texture_at(start + (12 << 20), &rgba(4)).unwrap();
        let oldest = Arc::downgrade(&a);
        let gpu = Arc::downgrade(&d);
        let list_copy = host_copy(b.get()).unwrap();
        let b_weak = Arc::downgrade(&b);
        let bytes = list_copy.bytes;
        let mut local = Cache::default();
        // Exercise the production budget with an isolated set of entries.
        for (i, owner) in [a, b, c.clone(), d].into_iter().enumerate() {
            let start = start + ((i as u64) << 22);
            local.entries.insert(
                (start, rgba(if i == 3 { 4 } else { 0 })),
                Entry {
                    owner,
                    end: start + (4 << 20),
                    bytes,
                    used: i as u64,
                },
            );
        }
        local.bytes = bytes * 4;
        invalidate_placed(start as usize, 16 << 20);
        // The live owners are still valid after invalidation. The local
        // cache is now the sole owner of the oldest entry.
        assert!(was_remapped(c.get()));
        assert_eq!(
            host_copy(b_weak.upgrade().unwrap().get()).unwrap().bytes,
            bytes
        );
        let next = texture_at(start + (8 << 20), &rgba(0)).unwrap();
        assert!(!Arc::ptr_eq(&c, &next));
        assert!(!Arc::ptr_eq(
            &host_copy(c.get()).unwrap(),
            &host_copy(next.get()).unwrap()
        ));
        let mut removed = Vec::new();
        local.trim(bytes, &mut removed);
        drop(removed);
        assert!(oldest.upgrade().is_none());
        assert!(b_weak.upgrade().is_some()); // command list pin
        assert!(gpu.upgrade().is_some()); // GPU-only, never LRU eligible
        assert_eq!(local.bytes, bytes * 3);
        eprintln!(
            "PLACED fixture: nominal cache bytes {} -> {}; live slot, list and UAV excluded",
            4 * bytes,
            local.bytes
        );
        invalidate_placed(start as usize, 16 << 20);
    }
}
