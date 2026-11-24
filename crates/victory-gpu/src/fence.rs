//! Fences: a 64-bit value and the events and queues waiting for it. Three
//! things move a value: the title's CPU signals, our queues' signals (a
//! queue's work runs to completion before its signal, so a queue not held
//! back by a wait signals at once) and bottom-of-pipe writes a command list
//! makes at the fence's GPU address. All three wake the waits, so an
//! event, a blocking wait or a queue's wait ends once the value it asked
//! for is reached, whichever moved it.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;
use crate::timeline::Timeline;

pub(crate) const FENCE_IIDS: [Iid; 4] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
    /// Wakes blocking waits (`SetEventOnCompletion` with a null event)
    /// whenever the value moves.
    moved: Condvar,
    /// Queues holding work back until the value reaches theirs
    /// (`ID3D12CommandQueue::Wait`). Locked after `waiters`, never before.
    queues: Mutex<Vec<(u64, Weak<Timeline>)>>,
}

impl Drop for Fence {
    fn drop(&mut self) {
        live().remove(&((&raw const self.value) as usize));
    }
}

/// Live fences by the address of their value (their GPU address), for
/// bottom-of-pipe writes that land on one. Locked before a fence's
/// `waiters`, never after.
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// How often a blocking wait looks at the value without being woken (a
/// value moved by a path we do not see), and how long it waits before
/// saying it is stuck.
const RECHECK: Duration = Duration::from_millis(100);
const STALL: Duration = Duration::from_secs(5);
/// How often the background watcher re-checks live fences while at least one
/// event or queue wait is pending (for direct CPU/shader memory writes).
const WATCH_POLL: Duration = Duration::from_millis(10);

struct Watcher {
    /// Incremented whenever an event or queue wait is added. Pure leaf lock:
    /// never held while acquiring `LIVE`, fence locks, or a timeline lock.
    generation: Mutex<u64>,
    added: Condvar,
}

static WATCHER: Watcher = Watcher {
    generation: Mutex::new(0),
    added: Condvar::new(),
};

fn watch_notify() {
    static THREAD: std::sync::Once = std::sync::Once::new();
    {
        let mut g = WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        *g = g.wrapping_add(1);
        WATCHER.added.notify_one();
    }
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-fence-watch".into())
            .spawn(watch_loop);
    });
}

fn watch_loop() {
    loop {
        let seen = *WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        let (mut woken, any_waiting) = scan_live_fences();
        let (more, mem_waiting) = poll_memwaits();
        woken.extend(more);
        let any_waiting = any_waiting || mem_waiting;
        if !woken.is_empty() {
            run_held(woken);
            continue;
        }
        let g = WATCHER.generation.lock().unwrap_or_else(|e| e.into_inner());
        if *g != seen {
            continue;
        }
        if any_waiting {
            drop(
                WATCHER
                    .added
                    .wait_timeout(g, WATCH_POLL)
                    .unwrap_or_else(|e| e.into_inner()),
            );
        } else {
            drop(WATCHER.added.wait(g).unwrap_or_else(|e| e.into_inner()));
        }
    }
}

/// Checks every live fence for event or queue waits whose value has been
/// reached outside `Signal` / bottom-of-pipe writes (for example a CPU write
/// to the fence's memory).
fn scan_live_fences() -> (Vec<Arc<Timeline>>, bool) {
    let live = live();
    let mut woken = Vec::new();
    let mut any_waiting = false;
    for &ptr in live.values() {
        // SAFETY: registered fences stay alive until `Fence::drop`, which
        // takes the `LIVE` lock we hold here.
        let fence = unsafe { &*(ptr as *const Fence) };
        let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
        let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
        if waiters.is_empty() && queues.is_empty() {
            continue;
        }
        let (more, moved) = wake_locked(fence, &mut waiters, &mut queues);
        if moved {
            fence.moved.notify_all();
        }
        woken.extend(more);
        any_waiting |= !waiters.is_empty() || !queues.is_empty();
    }
    (woken, any_waiting)
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
        moved: Condvar::new(),
        queues: Mutex::new(Vec::new()),
    });
    let at = (&raw const fence.value) as usize;
    live().insert(at, (&raw const *fence) as usize);
    // Dropped (and so unregistered) here if the title asked for an
    // interface a fence does not have.
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    tracing::debug!(fence = ?this, value, reached = fence.value.load(Ordering::Acquire), event = ?event, "fence event wait");
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if !event.is_null() {
        waiters.push((value, event as usize));
        drop(waiters);
        watch_notify();
        return S_OK;
    }
    // A null event: block until the value is reached, as D3D12 does.
    let started = Instant::now();
    let mut warned = Duration::ZERO;
    while fence.value.load(Ordering::Acquire) < value {
        waiters = fence
            .moved
            .wait_timeout(waiters, RECHECK)
            .unwrap_or_else(|e| e.into_inner())
            .0;
        let waited = started.elapsed();
        if waited >= warned + STALL {
            warned = waited;
            tracing::warn!(
                value,
                completed = fence.value.load(Ordering::Acquire),
                seconds = waited.as_secs(),
                "blocking fence wait still waiting"
            );
        }
    }
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = fence(this);
    let waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    tracing::debug!(fence = ?this, value, "fence signaled");
    fence.value.store(value, Ordering::Release);
    let queues = wake(fence, waiters);
    run_held(queues);
    S_OK
}

/// Wakes any events and queues on `fence` whose value has been reached;
/// returns the timelines to resume once locks are dropped and whether
/// anything changed.
#[must_use]
fn wake_locked(
    fence: &Fence,
    waiters: &mut Vec<(u64, usize)>,
    queues: &mut Vec<(u64, Weak<Timeline>)>,
) -> (Vec<Arc<Timeline>>, bool) {
    let value = fence.value.load(Ordering::Acquire);
    let mut moved = false;
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
            moved = true;
        }
        !reached
    });
    let mut woken = Vec::new();
    queues.retain(|(wanted, queue)| {
        if *wanted <= value {
            woken.extend(queue.upgrade());
            moved = true;
            false
        } else {
            queue.strong_count() > 0
        }
    });
    (woken, moved)
}

/// Sets the events whose value the fence has reached, wakes the blocking
/// waits and returns the queues to wake, for [`run_held`] once no fence
/// lock is held.
#[must_use]
fn wake(fence: &Fence, mut waiters: MutexGuard<'_, Vec<(u64, usize)>>) -> Vec<Arc<Timeline>> {
    let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
    let (woken, _) = wake_locked(fence, &mut waiters, &mut queues);
    fence.moved.notify_all();
    woken
}

/// Runs the work queues held back until a value now reached. With no fence
/// lock held: the work may move fences itself.
fn run_held(queues: Vec<Arc<Timeline>>) {
    for queue in queues {
        queue.wake();
    }
}

/// Queues a bottom-of-pipe write woke, to run once the writer holds no
/// lock: their work submits lists itself (`Host::execute`'s lock).
#[derive(Default)]
#[must_use = "run the woken queues once no lock is held"]
pub(crate) struct Woken(Vec<Arc<Timeline>>);

impl Woken {
    pub(crate) fn add(&mut self, other: Woken) {
        self.0.extend(other.0);
    }

    pub(crate) fn run(self) {
        run_held(self.0);
    }
}

/// A reference to a fence held by a queue's work, given back on drop.
pub(crate) struct Held(object::Held);

impl Held {
    pub(crate) fn new(this: This) -> Held {
        Held(object::Held::new(this))
    }

    pub(crate) fn object(&self) -> This {
        self.0.get()
    }

    pub(crate) fn reached(&self, value: u64) -> bool {
        fence(self.0.get()).value.load(Ordering::Acquire) >= value
    }

    pub(crate) fn signal(&self, value: u64) {
        signal(self.0.get(), value);
    }

    /// Whether the value is reached; if not, `queue` is woken once it is.
    /// Checked under the lock a signal takes, so no signal is missed.
    pub(crate) fn wait_on_gpu(&self, value: u64, queue: &Arc<Timeline>) -> bool {
        let fence = fence(self.0.get());
        let mut queues = fence.queues.lock().unwrap_or_else(|e| e.into_inner());
        if fence.value.load(Ordering::Acquire) >= value {
            return true;
        }
        let queue = Arc::downgrade(queue);
        if !queues.iter().any(|(v, q)| *v == value && q.ptr_eq(&queue)) {
            queues.push((value, queue));
        }
        drop(queues);
        watch_notify();
        false
    }
}

/// A list's `wait_value` parked on a queue's timeline (`Op::WaitValue`):
/// the queue resumes once `*addr cmp value` reads true — a bottom-of-pipe
/// write, or the title's own CPU write the watcher polls for.
struct MemWait {
    addr: usize,
    cmp: u32,
    value: u64,
    width: u8,
    queue: Weak<Timeline>,
}

/// The queue waits parked on memory cells. Leaf lock: taken after a
/// timeline's state and after `Host::submitted`, never before — its
/// waiters' queues run once it is dropped.
static MEMWAITS: Mutex<Vec<MemWait>> = Mutex::new(Vec::new());

/// The `comparison` of a list's `wait_value` (the console's comparison
/// funcs): `None` for a value outside them.
pub(crate) fn compare_u64(cmp: u32, have: u64, want: u64) -> Option<bool> {
    Some(match cmp {
        1 => false,          // NEVER
        2 => have < want,    // LESS
        3 => have == want,   // EQUAL
        4 => have <= want,   // LESS_EQUAL
        5 => have > want,    // GREATER
        6 => have != want,   // NOT_EQUAL
        7 => have >= want,   // GREATER_EQUAL
        8 => true,           // ALWAYS
        _ => return None,
    })
}

/// Whether `*addr cmp value` reads true now (a cell that cannot be read
/// counts as not reached).
pub(crate) fn mem_wait_reached(addr: usize, cmp: u32, value: u64, width: u8) -> bool {
    let value = if width == 4 { u64::from(value as u32) } else { value };
    read_cell(addr, width).is_some_and(|have| compare_u64(cmp, have, value) == Some(true))
}

pub(crate) fn read_cell(addr: usize, width: u8) -> Option<u64> {
    #[cfg(windows)]
    {
        let mut bytes = [0; 8];
        if !matches!(width, 4 | 8)
            || !crate::host::read_into(addr as *const u8, &mut bytes[..usize::from(width)])
        {
            return None;
        }
        Some(u64::from_le_bytes(bytes))
    }
    #[cfg(not(windows))]
    {
        let _ = (addr, width);
        None
    }
}

/// Whether `*addr cmp value` holds; if not, `queue` is woken once a write
/// makes it hold. Checked and registered under `MEMWAITS` — a write lands
/// in memory before it takes the lock to wake, so one between the checks
/// is not missed.
pub(crate) fn memwait_on_gpu(
    addr: usize,
    cmp: u32,
    value: u64,
    width: u8,
    queue: &Arc<Timeline>,
) -> bool {
    let mut waits = MEMWAITS.lock().unwrap_or_else(|e| e.into_inner());
    if mem_wait_reached(addr, cmp, value, width) {
        return true;
    }
    let queue = Arc::downgrade(queue);
    if !waits
        .iter()
        .any(|w| w.addr == addr && w.cmp == cmp && w.value == value && w.width == width && w.queue.ptr_eq(&queue))
    {
        waits.push(MemWait { addr, cmp, value, width, queue });
    }
    drop(waits);
    watch_notify();
    false
}

/// Wakes the parked list waits whose cell `[at, at + bytes)` overlaps and
/// whose condition now reads true; returns their queues.
fn scan_memwaits(at: usize, bytes: usize) -> Vec<Arc<Timeline>> {
    let end = at.saturating_add(bytes);
    let mut waits = MEMWAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut woken = Vec::new();
    waits.retain(|w| {
        if w.addr < end
            && w.addr.saturating_add(usize::from(w.width)) > at
            && mem_wait_reached(w.addr, w.cmp, w.value, w.width)
        {
            woken.extend(w.queue.upgrade());
            return false;
        }
        w.queue.strong_count() > 0
    });
    woken
}

/// The watcher's pass over parked list waits, for writes it does not see
/// (the title's own CPU writes to the cell).
fn poll_memwaits() -> (Vec<Arc<Timeline>>, bool) {
    let mut waits = MEMWAITS.lock().unwrap_or_else(|e| e.into_inner());
    let mut woken = Vec::new();
    waits.retain(|w| {
        if mem_wait_reached(w.addr, w.cmp, w.value, w.width) {
            woken.extend(w.queue.upgrade());
            return false;
        }
        w.queue.strong_count() > 0
    });
    (woken, !waits.is_empty())
}

/// After a bottom-of-pipe write of `bytes` at `at`: wakes the waits of a
/// fence whose value the write touched, and the list waits whose cell it
/// covered; returns their held queues.
pub(crate) fn written(at: usize, bytes: usize) -> Woken {
    let mut woken = Woken(scan_memwaits(at, bytes));
    let end = at.saturating_add(bytes);
    let live = live();
    if let Some((&value_at, &fence)) = live.range(..end).next_back()
        && value_at + size_of::<u64>() > at
    {
        // SAFETY: registered fences are alive until their drop, which takes
        // the registry lock we hold.
        let fence = unsafe { &*(fence as *const Fence) };
        tracing::debug!(fence = format_args!("{:p}", fence), reached = fence.value.load(Ordering::Acquire), "write to a fence GPU address");
        let queues = wake(
            fence,
            fence.waiters.lock().unwrap_or_else(|e| e.into_inner()),
        );
        drop(live);
        woken.add(Woken(queues));
    }
    woken
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    type Wait = extern "system" fn(This, u64, *mut c_void) -> HRESULT;
    type Signal = extern "system" fn(This, u64) -> HRESULT;

    fn new_fence(initial: u64) -> This {
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), initial, 0, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        f
    }

    fn release_fence(f: This) {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    fn address(f: This) -> usize {
        slot::<extern "system" fn(This) -> u64>(f, 0x60)(f) as usize
    }

    /// Runs a blocking wait for `value` on another thread; the receiver
    /// gets its result once it returns.
    fn blocking_wait(f: This, value: u64) -> std::sync::mpsc::Receiver<HRESULT> {
        let (done, result) = std::sync::mpsc::channel();
        let f = Held::new(f);
        std::thread::spawn(move || {
            let wait = slot::<Wait>(f.object(), 0x48);
            let result = wait(f.object(), value, std::ptr::null_mut());
            drop(f);
            let _ = done.send(result);
        });
        result
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(1);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<Wait>(f, 0x48);
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // Grand Theft Auto VI reads the value at the fence's address.
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(address(f) as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[test]
    fn blocking_wait_returns_at_once_when_reached() {
        let f = new_fence(7);
        let wait = slot::<Wait>(f, 0x48);
        assert_eq!(wait(f, 7, std::ptr::null_mut()), S_OK);
        assert_eq!(wait(f, 3, std::ptr::null_mut()), S_OK);
        release_fence(f);
    }

    /// Used to answer `E_NOTIMPL` at once, so the title went on before the
    /// value was reached.
    #[test]
    fn blocking_wait_blocks_until_signalled() {
        let f = new_fence(0);
        let result = blocking_wait(f, 2);
        let short = Duration::from_millis(200);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned before any signal"
        );
        let signal = slot::<Signal>(f, 0x50);
        assert_eq!(signal(f, 1), S_OK);
        assert!(
            result.recv_timeout(short).is_err(),
            "returned below its value"
        );
        assert_eq!(signal(f, 2), S_OK);
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[test]
    fn bottom_of_pipe_write_releases_blocking_wait() {
        let f = new_fence(0);
        let result = blocking_wait(f, 0x1_0000_0003);
        std::thread::sleep(Duration::from_millis(50));
        // The low half alone does not reach the value.
        crate::queue::Written::Value(3)
            .write(address(f) as *mut u8, 4)
            .run();
        assert!(result.recv_timeout(Duration::from_millis(200)).is_err());
        crate::queue::Written::Value(0x1_0000_0003)
            .write(address(f) as *mut u8, 8)
            .run();
        assert_eq!(result.recv_timeout(Duration::from_secs(5)), Ok(S_OK));
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn bottom_of_pipe_write_sets_waiting_event() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(0);
        // SAFETY: plain manual-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        assert_eq!(slot::<Wait>(f, 0x48)(f, 9, event), S_OK);
        crate::queue::Written::Value(9)
            .write(address(f) as *mut u8, 8)
            .run();
        // SAFETY: a valid event.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn direct_memory_write_sets_waiting_event() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let f = new_fence(0);
        // SAFETY: plain manual-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        assert!(!event.is_null());
        assert_eq!(slot::<Wait>(f, 0x48)(f, 11, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        // Direct write to the fence's GPU address, bypassing `Signal` and
        // `Written::write`.
        // SAFETY: `address(f)` is `Fence::value`, an `AtomicU64`.
        unsafe { (*(address(f) as *const AtomicU64)).store(11, Ordering::Release) };
        // SAFETY: a valid event.
        assert_eq!(unsafe { WaitForSingleObject(event, 2_000) }, WAIT_OBJECT_0);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        release_fence(f);
    }

    #[cfg(windows)]
    #[test]
    fn memory_wait_uses_its_width_and_resumes_without_another_submission() {
        use crate::timeline::Op;
        let cell = Arc::new(AtomicU64::new(0x5_0000_0000));
        let at = Arc::as_ptr(&cell) as usize;
        let narrow = Arc::new(Timeline::default());
        let wide = Arc::new(Timeline::default());
        let (low_tx, low) = std::sync::mpsc::channel();
        let (full_tx, full) = std::sync::mpsc::channel();
        narrow.submit_batch(vec![Op::WaitValue(at, 3, 7, 4), Op::Run(Box::new(move || { let _ = low_tx.send(()); }))]);
        wide.submit_batch(vec![Op::WaitValue(at, 3, 0x6_0000_0007, 8), Op::Run(Box::new(move || { let _ = full_tx.send(()); }))]);
        assert!(low.try_recv().is_err());
        assert!(full.try_recv().is_err());
        crate::queue::Written::Value(0x5_0000_0007).write(at as *mut u8, 8).run();
        assert_eq!(low.recv_timeout(Duration::from_secs(2)), Ok(()));
        assert!(full.try_recv().is_err());
        // A title-side write bypassing the signal APIs is picked up by the watcher.
        cell.store(0x6_0000_0007, Ordering::Release);
        assert_eq!(full.recv_timeout(Duration::from_secs(2)), Ok(()));
    }

    #[cfg(windows)]
    #[test]
    fn memory_wait_reads_only_the_mapped_word_at_a_page_boundary() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        // SAFETY: reserve two pages, but commit only the first. Its final
        // four bytes are readable; reading eight would cross the boundary.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_READWRITE);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            let at = base.cast::<u8>().add(4092).cast::<u32>();
            at.write(12);
            let narrow = mem_wait_reached(at as usize, 3, 12, 4);
            let wide = mem_wait_reached(at as usize, 3, 12, 8);
            VirtualFree(base, 0, MEM_RELEASE);
            assert!(narrow);
            assert!(!wide);
        }
    }

    #[test]
    fn released_fences_leave_the_registry() {
        let f = new_fence(0);
        let at = address(f);
        assert!(live().contains_key(&at));
        release_fence(f);
        assert!(!live().contains_key(&at));
        // A write next to (not on) a live fence's value wakes nothing and
        // does not fault.
        let g = new_fence(0);
        let mut other = [0u64; 2];
        written(other.as_mut_ptr() as usize, 16).run();
        written(address(g) + 8, 8).run();
        release_fence(g);
    }

    /// A bottom-of-pipe write at a fence a held queue waits for does not
    /// run that queue's work inside the write: the writer (`Host::execute`)
    /// still holds its submission lock, which the work takes again. The old
    /// code ran it there (a deadlock on the real lock).
    #[test]
    fn bottom_of_pipe_writes_leave_woken_work_to_the_writer() {
        use crate::timeline::{Op, Timeline};
        let f = new_fence(0);
        let submission = Arc::new(Mutex::new(()));
        let ran = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let queue = Arc::new(Timeline::default());
        queue.submit(Op::Wait(Held::new(f), 1));
        let (lock, seen) = (submission.clone(), ran.clone());
        queue.submit(Op::Run(Box::new(move || {
            let free = lock.try_lock().is_ok();
            seen.store(if free { 1 } else { 2 }, Ordering::SeqCst);
        })));
        let held = submission.lock().unwrap();
        let woken = crate::queue::Written::Value(1).write(address(f) as *mut u8, 8);
        assert_eq!(ran.load(Ordering::SeqCst), 0, "ran inside the write");
        drop(held);
        woken.run();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        release_fence(f);
    }
}
