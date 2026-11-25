//! Bounded crash context, independent of the detailed DRED command notes.
use std::fmt;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::{Direct3D12::ID3D12Device, Dxgi::*};
use windows_sys::Win32::System::{ProcessStatus::*, Threading::GetCurrentProcess};

const SUBMISSIONS: usize = 24;
const LISTS: usize = 12;
const OPS: usize = 8;
const STAGES: usize = 6;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CRUMBS").is_some_and(|v| v == "1"))
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Title,
    Prelude,
    Upload,
    Present,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Command {
    Draw,
    Indexed,
    Dispatch,
    Mesh,
    Indirect,
    MeshFallbackDispatch,
    MeshFallbackDraw,
    MeshFallbackIndirectDispatch,
    MeshFallbackIndirectDraw,
}

#[derive(Clone, Copy)]
pub(crate) struct Op {
    command: Command,
    args: [u32; 3],
    shaders: [[u8; 16]; STAGES],
    stages: u8,
}

impl Op {
    pub(crate) fn new(
        command: Command,
        args: [u32; 3],
        hashes: impl Iterator<Item = impl AsRef<str>>,
    ) -> Self {
        let mut op = Self {
            command,
            args,
            shaders: [[0; 16]; STAGES],
            stages: 0,
        };
        for (slot, hash) in op.shaders.iter_mut().zip(hashes) {
            let hash = hash.as_ref().as_bytes();
            let n = hash.len().min(slot.len());
            slot[..n].copy_from_slice(&hash[..n]);
            op.stages += 1;
        }
        op
    }
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}{:?}", self.command, self.args)?;
        for hash in &self.shaders[..usize::from(self.stages)] {
            let n = hash.iter().position(|&b| b == 0).unwrap_or(hash.len());
            write!(f, " {}", std::str::from_utf8(&hash[..n]).unwrap_or("?"))?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Tail<T: Copy, const N: usize> {
    slots: [Option<T>; N],
    count: usize,
}

impl<T: Copy, const N: usize> Default for Tail<T, N> {
    fn default() -> Self {
        Self {
            slots: [None; N],
            count: 0,
        }
    }
}

impl<T: Copy, const N: usize> Tail<T, N> {
    pub(crate) fn push(&mut self, value: T) {
        self.slots[self.count % N] = Some(value);
        self.count += 1;
    }

    fn iter(&self) -> impl Iterator<Item = &T> {
        (self.count.saturating_sub(N)..self.count).filter_map(|i| self.slots[i % N].as_ref())
    }
}

pub(crate) type ListOps = Tail<Op, OPS>;

#[derive(Clone, Copy)]
pub(crate) struct List {
    kind: Kind,
    raw: usize,
    ops: ListOps,
}

impl List {
    pub(crate) fn new(kind: Kind, raw: usize, ops: ListOps) -> Self {
        Self { kind, raw, ops }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Outcome {
    Pending,
    Complete,
    Removed,
    WaitFailed,
    SignalFailed,
    EventFailed,
}

struct Submission {
    fence: u64,
    frame: u64,
    lists: Tail<List, LISTS>,
    start: Instant,
    elapsed: Option<Duration>,
    observed: Option<u64>,
    outcome: Outcome,
}

struct Ring {
    // Heap allocated once, then overwritten in place; never grows.
    entries: Box<[Option<Submission>]>,
    next: usize,
    count: usize,
}

impl Ring {
    fn new() -> Self {
        Self {
            entries: (0..SUBMISSIONS).map(|_| None).collect(),
            next: 0,
            count: 0,
        }
    }

    fn begin(&mut self, fence: u64, frame: u64, lists: impl Iterator<Item = List>) -> usize {
        let mut tail = Tail::default();
        for list in lists {
            tail.push(list);
        }
        let slot = self.next;
        self.entries[slot] = Some(Submission {
            fence,
            frame,
            lists: tail,
            start: Instant::now(),
            elapsed: None,
            observed: None,
            outcome: Outcome::Pending,
        });
        self.next = (slot + 1) % SUBMISSIONS;
        self.count = (self.count + 1).min(SUBMISSIONS);
        slot
    }

    fn finish(&mut self, slot: usize, observed: Option<u64>, outcome: Outcome) {
        if let Some(entry) = &mut self.entries[slot] {
            entry.elapsed = Some(entry.start.elapsed());
            entry.observed = observed;
            entry.outcome = outcome;
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Submission> {
        let first = self.next + SUBMISSIONS - self.count;
        (first..first + self.count).filter_map(|i| self.entries[i % SUBMISSIONS].as_ref())
    }
}

static RING: OnceLock<Mutex<Ring>> = OnceLock::new();

pub(crate) fn begin(fence: u64, frame: u64, lists: impl Iterator<Item = List>) -> usize {
    RING.get_or_init(|| Mutex::new(Ring::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .begin(fence, frame, lists)
}

pub(crate) fn finish(slot: Option<usize>, observed: Option<u64>, outcome: Outcome) {
    if let Some(slot) = slot {
        RING.get()
            .unwrap()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .finish(slot, observed, outcome);
    }
}

/// Only called on a host failure. No per-command or per-submission I/O during play.
pub(crate) fn flush() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if let Some(ring) = RING.get() {
        ONCE.call_once(|| {
            let ring = ring.lock().unwrap_or_else(|e| e.into_inner());
            for entry in ring.iter() {
                let elapsed_ms = entry.elapsed.unwrap_or_else(|| entry.start.elapsed()).as_secs_f64() * 1e3;
                tracing::error!(id = entry.fence, fence_expected = entry.fence, fence_observed = ?entry.observed,
                    frame = entry.frame, elapsed_ms, outcome = ?entry.outcome, lists = entry.lists.count,
                    omitted_lists = entry.lists.count.saturating_sub(LISTS), "crash submission");
                for (index, list) in entry.lists.iter().enumerate() {
                    let index = index + entry.lists.count.saturating_sub(LISTS);
                    tracing::error!(id = entry.fence, index, kind = ?list.kind, raw = format_args!("{:#x}", list.raw),
                        ops = list.ops.count, tail = ?list.ops.iter().collect::<Vec<_>>(), "crash list");
                }
            }
        });
    }
}

/// Poll outside the GPU submission path. Node zero is the node this host uses.
pub(crate) fn start_sampler(device: &ID3D12Device) {
    if !enabled() {
        return;
    }
    // SAFETY: a live device and a plain adapter ID query.
    let luid = unsafe { device.GetAdapterLuid() };
    let thread = std::thread::Builder::new().name("victory-crumbs-sampler".into()).spawn(move || {
        // SAFETY: standard factory creation, using the D3D12 device's exact LUID.
        let adapter = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(luid))
        };
        if let Err(e) = &adapter {
            tracing::warn!(error = %e, "crash sampler has no DXGI budget adapter");
        }
        let pid = std::process::id() as usize;
        let mut faults = Vec::new();
        let mut previous = None;
        let mut last = Instant::now();
        loop {
            let start = Instant::now();
            let sample_ms = start.duration_since(last).as_secs_f64() * 1e3;
            last = start;
            let memory = |segment| {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                // SAFETY: live adapter and valid output for the host's node.
                unsafe { adapter.as_ref().ok()?.QueryVideoMemoryInfo(0, segment, &mut info) }.ok()?;
                Some((info.CurrentUsage, info.Budget))
            };
            let local = memory(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
            let non_local = memory(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
            let mut process = PROCESS_MEMORY_COUNTERS_EX { cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32, ..Default::default() };
            let mut system = PERFORMANCE_INFORMATION { cb: size_of::<PERFORMANCE_INFORMATION>() as u32, ..Default::default() };
            // SAFETY: our process handle and writable, sized PSAPI structures.
            let proc_ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut process).cast(), process.cb) } != 0;
            let system_ok = unsafe { K32GetPerformanceInfo(&mut system, system.cb) } != 0;
            let current = hard_faults(pid, &mut faults);
            let hard_fault_delta = current.zip(previous).map(|(now, last): (u32, u32)| now.wrapping_sub(last));
            previous = current;
            tracing::info!(luid_hi = luid.HighPart, luid_lo = luid.LowPart, node = 0, sample_ms,
                local_usage_bytes = ?local.map(|v| v.0), local_budget_bytes = ?local.map(|v| v.1),
                non_local_usage_bytes = ?non_local.map(|v| v.0), non_local_budget_bytes = ?non_local.map(|v| v.1),
                ws_bytes = ?proc_ok.then_some(process.WorkingSetSize), private_commit_bytes = ?proc_ok.then_some(process.PrivateUsage),
                physical_available_bytes = ?system_ok.then(|| system.PhysicalAvailable * system.PageSize),
                system_commit_bytes = ?system_ok.then(|| system.CommitTotal * system.PageSize),
                system_commit_limit_bytes = ?system_ok.then(|| system.CommitLimit * system.PageSize),
                ?hard_fault_delta, "GPU memory sample");
            std::thread::sleep(Duration::from_secs(1).saturating_sub(start.elapsed()));
        }
    });
    if let Err(e) = thread {
        tracing::warn!(error = %e, "crash memory sampler could not start");
    }
}

fn hard_faults(pid: usize, buf: &mut Vec<u8>) -> Option<u32> {
    if !cfg!(target_arch = "x86_64") {
        return None; // The layout below is NT x64 only.
    }
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    if buf.is_empty() {
        buf.resize(256 << 10, 0);
    }
    loop {
        let mut returned = 0u32;
        // SAFETY: the buffer has `len` writable bytes, and returned is our out parameter.
        let status = unsafe {
            NtQuerySystemInformation(5, buf.as_mut_ptr(), buf.len() as u32, &mut returned)
        };
        if status >= 0 {
            return parse_faults(buf.get(..returned as usize)?, pid);
        }
        const LENGTH_MISMATCH: i32 = 0xc0000004_u32 as i32;
        let needed = (returned as usize).max(buf.len() * 2);
        if status != LENGTH_MISMATCH || needed > 4 << 20 {
            return None;
        }
        buf.resize(needed, 0);
    }
}

fn parse_faults(mut buf: &[u8], pid: usize) -> Option<u32> {
    // SYSTEM_PROCESS_INFORMATION (x64): NextEntryOffset, HardFaultCount, UniqueProcessId.
    while buf.len() >= 0x58 {
        let next = u32::from_le_bytes(buf[..4].try_into().ok()?) as usize;
        if u64::from_le_bytes(buf[0x50..0x58].try_into().ok()?) == pid as u64 {
            return Some(u32::from_le_bytes(buf[0x10..0x14].try_into().ok()?));
        }
        if next < 0x58 {
            break;
        }
        buf = buf.get(next..)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_the_recent_submissions_and_failed_fence() {
        let mut ring = Ring::new();
        for fence in 1..=SUBMISSIONS as u64 + 2 {
            let slot = ring.begin(fence, 10, std::iter::empty());
            let removed = fence == SUBMISSIONS as u64 + 2;
            ring.finish(
                slot,
                Some(if removed { u64::MAX } else { fence }),
                if removed {
                    Outcome::Removed
                } else {
                    Outcome::Complete
                },
            );
        }
        assert_eq!(ring.iter().count(), SUBMISSIONS);
        assert_eq!(ring.iter().next().unwrap().fence, 3);
        let failure = ring.iter().last().unwrap();
        assert_eq!(failure.observed, Some(u64::MAX));
        assert_eq!(failure.frame, 10);
        assert!(failure.elapsed.is_some());
        assert!(matches!(failure.outcome, Outcome::Removed));
    }

    #[test]
    fn command_tail_keeps_latest_in_order() {
        let mut tail = ListOps::default();
        for i in 0..OPS + 2 {
            tail.push(Op::new(
                Command::Draw,
                [i as u32, 1, 0],
                ["0123456789abcdef"].into_iter(),
            ));
        }
        assert_eq!(tail.count, OPS + 2);
        assert_eq!(
            tail.iter().map(|op| op.args[0]).collect::<Vec<_>>(),
            (2..OPS as u32 + 2).collect::<Vec<_>>()
        );
        assert_eq!(
            format!("{:?}", tail.iter().next().unwrap()),
            "Draw[2, 1, 0] 0123456789abcdef"
        );
    }

    #[test]
    fn enabled_host_submission_keeps_a_completed_fence_and_command_tail() {
        if !enabled() {
            return;
        }
        use windows::core::Interface;
        let host = crate::host::host().expect("D3D12 test host");
        let list = host.new_list().unwrap();
        list.crumb(|| Op::new(Command::Dispatch, [2, 3, 4], ["0123456789abcdef"].into_iter()));
        list.close();
        host.execute(&[&list]);
        let ring = RING.get().unwrap().lock().unwrap();
        let entry = ring.iter().last().unwrap();
        assert_eq!(entry.observed, Some(entry.fence));
        assert!(entry.elapsed.is_some());
        assert!(matches!(entry.outcome, Outcome::Complete));
        let context = entry.lists.iter().last().unwrap();
        assert!(matches!(context.kind, Kind::Title));
        assert_eq!(context.raw, list.list.as_raw() as usize);
        let op = context.ops.iter().last().unwrap();
        assert!(matches!(op.command, Command::Dispatch));
        assert_eq!(op.args, [2, 3, 4]);
        assert_eq!(&op.shaders[0], b"0123456789abcdef");
    }

    #[test]
    fn hard_fault_counter_parser_checks_bounds_and_pid() {
        let mut entries = [0u8; 0xb0];
        entries[..4].copy_from_slice(&0x58u32.to_le_bytes());
        entries[0xa8..0xb0].copy_from_slice(&42u64.to_le_bytes());
        entries[0x68..0x6c].copy_from_slice(&17u32.to_le_bytes());
        assert_eq!(parse_faults(&entries, 42), Some(17));
        assert_eq!(parse_faults(&entries[..0xaf], 42), None);
        assert_eq!(parse_faults(&entries, 43), None);
    }
}
