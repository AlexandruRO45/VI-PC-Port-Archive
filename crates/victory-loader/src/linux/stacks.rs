//! Where every thread is, on Linux: `VICTORY_STACKS=<seconds>[,...]` as on
//! Windows. Most title threads wait inside our host code, which has no
//! `.pdata`, so a signal makes each thread with a TEB scan its own stack for
//! return addresses into registered images (a call instruction just before
//! them), innermost first.
// ponytail: a stack scan, not an unwind; stale return addresses can show up.
// Unwind from the innermost title frame if that misleads.

use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::{asm, seh, unwind};

const MAX_FRAMES: usize = 24;

/// The scanning thread's answer: its rip, then the return addresses found.
static FRAMES: [AtomicU64; MAX_FRAMES + 1] = [const { AtomicU64::new(0) }; MAX_FRAMES + 1];
static COUNT: AtomicUsize = AtomicUsize::new(0);
static DONE: AtomicU32 = AtomicU32::new(0);
/// The registered images' code ranges, set by the dumper before signalling.
static IMAGES: AtomicPtr<Vec<(usize, usize)>> = AtomicPtr::new(std::ptr::null_mut());

fn signal() -> libc::c_int {
    libc::SIGRTMIN() + 7
}

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let mut times: Vec<f64> = list.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    times.sort_by(f64::total_cmp);
    let start = Instant::now();
    std::thread::spawn(move || {
        for t in times {
            let at = Duration::from_secs_f64(t.max(0.0));
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// Whether the bytes before `ret` are a call: `E8 rel32`, or `FF /2`
/// with a register, or a memory operand of up to 6 more bytes.
fn after_call(ret: u64) -> bool {
    let Some(before) = asm::read64(ret - 8) else {
        return false;
    };
    let byte = |back: u64| (before >> (8 * (8 - back))) as u8;
    byte(5) == 0xE8 || [2, 3, 4, 6, 7].iter().any(|&k| byte(k) == 0xFF && byte(k - 1) >> 3 & 7 == 2)
}

extern "C" fn on_signal(_: libc::c_int, _: *mut libc::siginfo_t, raw: *mut libc::c_void) {
    // SAFETY: the kernel hands an SA_SIGINFO handler its ucontext.
    let g = unsafe { &(*raw.cast::<libc::ucontext_t>()).uc_mcontext.gregs };
    let (rip, rsp) = (g[libc::REG_RIP as usize] as u64, g[libc::REG_RSP as usize] as u64);
    FRAMES[0].store(rip, Ordering::Relaxed);
    let mut n = 1;
    let images = IMAGES.load(Ordering::Acquire);
    if let (Some((_, high)), false) = (seh::stack_bounds(), images.is_null()) {
        // SAFETY: published lists are never freed.
        let images = unsafe { &*images };
        let mut at = rsp & !7;
        while at + 8 <= high && n <= MAX_FRAMES {
            if let Some(v) = asm::read64(at)
                && images.iter().any(|&(l, h)| (l..h).contains(&(v as usize)))
                && after_call(v)
            {
                FRAMES[n].store(v, Ordering::Relaxed);
                n += 1;
            }
            at += 8;
        }
    }
    COUNT.store(n, Ordering::Relaxed);
    DONE.store(1, Ordering::Release);
}

/// Prints every other thread's innermost location and the title return
/// addresses on its stack to stderr.
pub fn dump_stacks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: the handler only reads its own stack and registers and
        // writes atomics.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal(), &action, std::ptr::null_mut());
        }
    });
    // Leaked: one list per dump.
    IMAGES.store(Box::into_raw(Box::new(unwind::image_ranges())), Ordering::Release);
    // SAFETY: plain query.
    let (pid, me) = unsafe { (libc::getpid(), libc::gettid()) };
    let mut ids: Vec<i32> = std::fs::read_dir("/proc/self/task")
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
        .filter(|&id| id != me)
        .collect();
    ids.sort();
    eprintln!("VIctory: stacks of {} threads", ids.len());
    for id in ids {
        let name = std::fs::read_to_string(format!("/proc/self/task/{id}/comm")).unwrap_or_default();
        DONE.store(0, Ordering::Release);
        // SAFETY: tgkill to one of our own threads with our own signal.
        if unsafe { libc::syscall(libc::SYS_tgkill, pid, id, signal()) } != 0 {
            continue;
        }
        let until = Instant::now() + Duration::from_millis(200);
        while DONE.load(Ordering::Acquire) == 0 && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(1));
        }
        if DONE.load(Ordering::Acquire) == 0 {
            eprintln!("  thread {id} \"{}\": no answer", name.trim());
            continue;
        }
        eprintln!("  thread {id} \"{}\"", name.trim());
        for (i, pc) in FRAMES[..COUNT.load(Ordering::Relaxed)].iter().enumerate() {
            eprintln!("    #{i:<2} {}", seh::locate(pc.load(Ordering::Relaxed) as usize));
        }
    }
}
