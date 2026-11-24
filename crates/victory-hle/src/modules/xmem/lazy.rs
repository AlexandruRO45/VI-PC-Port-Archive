//! Private XMem reservations committed in 64 KiB chunks on first access.
//! Registered ranges are removed before the host is allowed to free them.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{OnceLock, RwLock};
use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
use windows_sys::Win32::System::Diagnostics::Debug::{
    AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
    EXCEPTION_POINTERS,
};
use windows_sys::Win32::System::Memory::*;

pub(super) const MIN: usize = 2 << 20;
const CHUNK: usize = 64 << 10;
const PAGE: usize = 4096;

#[derive(Clone, Copy)]
struct Range {
    end: usize,
    allocation: usize,
    protect: u32,
}
static RANGES: RwLock<BTreeMap<usize, Range>> = RwLock::new(BTreeMap::new());

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        if super::on_wine() || std::env::var_os("VICTORY_LAZY_COMMIT").is_some_and(|v| v == "0") {
            return false;
        }
        // SAFETY: process-lifetime callback; does not allocate on fault.
        !unsafe { AddVectoredExceptionHandler(1, Some(fault)) }.is_null()
    })
}

fn query(at: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: valid output storage; the queried address need not be mapped.
    (unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) } != 0)
        .then_some(info)
}

fn readable(p: u32) -> bool {
    p & PAGE_GUARD == 0
        && p & (PAGE_READONLY
            | PAGE_READWRITE
            | PAGE_WRITECOPY
            | PAGE_EXECUTE_READ
            | PAGE_EXECUTE_READWRITE
            | PAGE_EXECUTE_WRITECOPY)
            != 0
}

fn writable(p: u32) -> bool {
    p & PAGE_GUARD == 0
        && p & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)
            != 0
}

/// Only private reservations, never a mapped file or physical placeholder.
/// For a fixed-address commit, validate that the whole range is reserved.
pub(super) fn register(at: usize, len: usize, protect: u32) -> bool {
    let start = at & !(PAGE - 1);
    let Some(end) = at
        .checked_add(len)
        .and_then(|v| v.checked_next_multiple_of(PAGE))
    else {
        return false;
    };
    if start == 0 || start >= end || !matches!(protect, PAGE_READWRITE | PAGE_READONLY) {
        return false;
    }
    let mut map = RANGES.write().unwrap_or_else(|e| e.into_inner());
    let Some(info) = query(start).filter(|i| i.Type == MEM_PRIVATE && i.State == MEM_RESERVE)
    else {
        return false;
    };
    if (info.BaseAddress as usize).saturating_add(info.RegionSize) < end {
        return false;
    }
    trim(&mut map, start, end);
    map.insert(
        start,
        Range {
            end,
            allocation: info.AllocationBase as usize,
            protect,
        },
    );
    true
}

fn trim(map: &mut BTreeMap<usize, Range>, from: usize, to: usize) {
    if from >= to {
        return;
    }
    let overlaps: Vec<_> = map
        .range(..to)
        .filter(|&(_, r)| from < r.end)
        .map(|(&a, &r)| (a, r))
        .collect();
    for (a, r) in overlaps {
        map.remove(&a);
        if a < from {
            map.insert(a, Range { end: from, ..r });
        }
        if to < r.end {
            map.insert(to, r);
        }
    }
}

fn cover(map: &BTreeMap<usize, Range>, at: usize) -> Option<(usize, Range)> {
    map.range(..=at)
        .next_back()
        .filter(|&(_, r)| at < r.end)
        .map(|(&a, &r)| (a, r))
}

/// Serializes an explicit OS commit/free with the VEH, and touches the
/// registry only after a successful OS operation.
pub(super) fn explicit(at: usize, len: usize, free: u32, op: impl FnOnce() -> bool) -> bool {
    let mut map = RANGES.write().unwrap_or_else(|e| e.into_inner());
    let info = (len == 0 && free & MEM_DECOMMIT != 0)
        .then(|| query(at))
        .flatten();
    if !op() {
        return false;
    }
    if free & MEM_RELEASE != 0 {
        map.retain(|_, r| r.allocation != at);
    } else if let Some(info) = info {
        map.retain(|_, r| r.allocation != info.AllocationBase as usize);
    } else if let Some(end) = at
        .checked_add(len)
        .and_then(|v| v.checked_next_multiple_of(PAGE))
    {
        trim(&mut map, at & !(PAGE - 1), end);
    }
    true
}

pub(super) fn logical_query(at: usize, info: &mut MEMORY_BASIC_INFORMATION) {
    if info.State == MEM_RESERVE
        && let Some((start, r)) = cover(&RANGES.read().unwrap_or_else(|e| e.into_inner()), at)
        && info.AllocationBase as usize == r.allocation
    {
        let lo = start.max(info.BaseAddress as usize);
        let hi = r
            .end
            .min((info.BaseAddress as usize).saturating_add(info.RegionSize));
        info.BaseAddress = lo as *mut c_void;
        info.RegionSize = hi - lo;
        info.State = MEM_COMMIT;
        info.Protect = r.protect;
    }
}

/// Kernel writes cannot fault into the user-mode handler. Only commit
/// private writable ranges; do not change existing protections.
pub(super) fn before_write(at: usize, len: usize) {
    let Some(end) = at.checked_add(len) else {
        return;
    };
    if len == 0 {
        return;
    }
    let map = RANGES.read().unwrap_or_else(|e| e.into_inner());
    let first = cover(&map, at).map_or(at, |(start, _)| start);
    for (&base, r) in map.range(first..end) {
        if !writable(r.protect) {
            continue;
        }
        let mut p = base.max(at & !(PAGE - 1));
        let hi = r
            .end
            .min(end.checked_next_multiple_of(PAGE).unwrap_or(r.end));
        while p < hi {
            let Some(info) = query(p) else { break };
            let next = hi.min((info.BaseAddress as usize).saturating_add(info.RegionSize));
            if next <= p {
                break;
            }
            if info.State == MEM_RESERVE && info.AllocationBase as usize == r.allocation {
                // SAFETY: still inside this live reservation under RANGES's lock.
                unsafe { VirtualAlloc(p as *const c_void, next - p, MEM_COMMIT, r.protect) };
            }
            p = next;
        }
    }
}

unsafe extern "system" fn fault(p: *mut EXCEPTION_POINTERS) -> i32 {
    // SAFETY: Windows supplies the exception record to the callback.
    let e = unsafe { &*(*p).ExceptionRecord };
    if e.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || e.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let access = |protect| match e.ExceptionInformation[0] {
        0 => readable(protect),
        1 => writable(protect),
        _ => false,
    };
    let at = e.ExceptionInformation[1];
    let map = RANGES.read().unwrap_or_else(|e| e.into_inner());
    let Some((base, range)) = cover(&map, at) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    let Some(info) = query(at).filter(|i| i.AllocationBase as usize == range.allocation) else {
        return EXCEPTION_CONTINUE_SEARCH;
    };
    if info.State == MEM_COMMIT && access(info.Protect) {
        return EXCEPTION_CONTINUE_EXECUTION; // another writer committed it
    }
    if info.State != MEM_RESERVE || !access(range.protect) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let aligned = at & !(CHUNK - 1);
    let run = info.BaseAddress as usize;
    let lo = aligned.max(base).max(run);
    let hi = aligned
        .saturating_add(CHUNK)
        .min(range.end)
        .min(run.saturating_add(info.RegionSize));
    // SAFETY: only the reserved run, pinned by the shared registry lock.
    if hi > lo
        && !unsafe { VirtualAlloc(lo as *const c_void, hi - lo, MEM_COMMIT, range.protect) }
            .is_null()
    {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_touch_and_kernel_write_commit_only_their_spans() {
        if !enabled() {
            return;
        }
        let p = super::super::XMemVirtualAlloc(
            std::ptr::null_mut(),
            MIN,
            MEM_RESERVE | MEM_COMMIT,
            3,
            PAGE_READWRITE,
        )
        .cast::<u8>();
        assert!(!p.is_null());
        let base = p as usize;
        assert_eq!(query(base).unwrap().State, MEM_RESERVE);
        let mut logical = query(base).unwrap();
        logical_query(base, &mut logical);
        assert_eq!(logical.State, MEM_COMMIT);
        // SAFETY: these bytes are in the owned registered reservation.
        unsafe { p.add(CHUNK + 3).write_volatile(0xa5) };
        assert_eq!(query(base + CHUNK).unwrap().RegionSize, CHUNK);
        assert_eq!(query(base + CHUNK).unwrap().State, MEM_COMMIT);
        assert_eq!(query(base + 2 * CHUNK).unwrap().State, MEM_RESERVE);
        before_write(base + 3 * CHUNK + 1, 10);
        assert_eq!(query(base + 3 * CHUNK).unwrap().RegionSize, PAGE);
        assert_eq!(query(base + 3 * CHUNK).unwrap().State, MEM_COMMIT);
        assert_ne!(super::super::VirtualFree(p.cast(), 0, MEM_RELEASE), 0);
        assert!(cover(&RANGES.read().unwrap(), base).is_none());
    }
}
