//! Walks a thread's stack into return addresses by the unwind data of each
//! image, for the crash report and the stack dump. It does not name the
//! frames (`crash::locate` does) or suspend other threads (`stacks` does),
//! and it stops where `rsp` lies outside the thread's committed stack.

use std::ops::Range;
use std::ptr::null_mut;
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, RtlLookupFunctionEntry, RtlVirtualUnwind, UNW_FLAG_NHANDLER,
};
use windows_sys::Win32::System::Kernel::NT_TIB;

/// The most frames a walk records, the starting instruction's included.
pub(crate) const MAX_FRAMES: usize = 40;

/// The committed part of the stack the NT_TIB at `tib` describes, from its
/// stack limit up to its stack base. Below it lie the thread's guard pages
/// and pages reserved but not committed. A reserved page faults when any
/// thread reads it; a guard page faults when another thread reads it.
///
/// # Safety
/// `tib` points at a mapped NT_TIB.
pub(crate) unsafe fn tib_stack(tib: *const NT_TIB) -> Range<usize> {
    // SAFETY: the caller's; the fields are read through the raw pointer.
    let (base, limit) = unsafe { ((*tib).StackBase, (*tib).StackLimit) };
    limit.addr()..base.addr()
}

/// This thread's committed stack, read from its own TEB: no system call, so
/// it cannot fail.
fn current_thread_stack() -> Range<usize> {
    let tib: *const NT_TIB;
    // SAFETY: on x64 Windows `gs` holds this thread's TEB, whose NT_TIB keeps
    // its own address in `Self` (how winnt.h's `NtCurrentTeb` reads it); the
    // read touches no stack and no flags.
    unsafe {
        std::arch::asm!(
            "mov {}, gs:[{}]",
            out(reg) tib,
            const std::mem::offset_of!(NT_TIB, Self_),
            options(nostack, readonly, preserves_flags),
        );
    }
    // SAFETY: this thread's own TEB, mapped while it runs.
    unsafe { tib_stack(tib) }
}

/// Return addresses of the faulting thread's frames, from the faulting
/// instruction up, by the unwind data of each image (the Windows loader
/// registered it for title images too). Code without unwind data (our
/// thunks, leaf functions) is taken to have its return address at `rsp`.
/// The stack is its committed part, so `rsp` in a reserved page ends the
/// walk.
pub(crate) fn unwind(context: CONTEXT) -> Vec<u64> {
    let stack = current_thread_stack();
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    unwind_into(context, stack.start, stack.end, &mut frames);
    frames
}

/// A `CONTEXT` at the 16-byte alignment the unwind and capture functions need.
#[repr(C, align(16))]
pub(crate) struct Aligned16(pub(crate) CONTEXT);

/// [`unwind`] for any thread of the process, its stack being `low..high`;
/// pushes no more than `MAX_FRAMES`, so it does not allocate when `frames`
/// has that capacity (another thread may be suspended holding the heap).
pub(crate) fn unwind_into(context: CONTEXT, low: usize, high: usize, frames: &mut Vec<u64>) {
    // `RtlVirtualUnwind` writes the context with 16-byte aligned stores, and
    // windows-sys declares `CONTEXT` without that alignment: a by-value copy on
    // the stack can land 8 bytes off and fault (seen in a debug build).
    let mut aligned = Aligned16(context);
    let context = &mut aligned.0;
    frames.push(context.Rip);
    while frames.len() < MAX_FRAMES {
        // Stop where reading the stack would fault inside this handler.
        let rsp = context.Rsp as usize;
        if rsp < low || rsp + 8 > high {
            break;
        }
        let mut image = 0;
        // SAFETY: looks up the function table entry covering `Rip`, if any.
        let entry = unsafe { RtlLookupFunctionEntry(context.Rip, &mut image, null_mut()) };
        if entry.is_null() {
            // SAFETY: `rsp` is inside this thread's stack.
            context.Rip = unsafe { (rsp as *const u64).read() };
            context.Rsp += 8;
        } else {
            let (mut data, mut frame) = (null_mut(), 0);
            // SAFETY: `entry` belongs to `image`; the context is our copy.
            unsafe {
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    image,
                    context.Rip,
                    entry,
                    context,
                    &mut data,
                    &mut frame,
                    null_mut(),
                )
            };
        }
        if context.Rip == 0 {
            break;
        }
        frames.push(context.Rip);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        RtlCaptureContext, RtlCaptureStackBackTrace,
    };
    use windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits;

    use crate::crash::memory_at;

    /// One of this test binary's tests, run alone in a child process with
    /// one environment variable set.
    struct Child<'a> {
        test: &'a str,
        variable: &'a str,
        value: &'a str,
    }

    impl Child<'_> {
        /// Runs it; returns its exit code and its stderr.
        fn run(&self) -> (Option<i32>, String) {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", self.test, "--nocapture"])
                .env(self.variable, self.value)
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stderr).into_owned();
            (out.status.code(), text)
        }
    }

    /// An address in the reserved, never committed part of this thread's
    /// stack.
    fn reserved_stack_address() -> usize {
        let (mut low, mut high) = (0, 0);
        // SAFETY: writes the current thread's stack bounds, its whole
        // reservation.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        let reserved = low + 0x2000;
        assert_eq!(memory_at(reserved), "reserved, never committed");
        reserved
    }

    /// In a child process of the next test: walks from `rsp` in the reserved,
    /// uncommitted part of its stack, with `rip` where no function table
    /// reaches, so the walk would read the return address at `rsp`; that read
    /// ends the process if the walk's bounds allow it.
    #[test]
    fn stack_pointer_child() {
        if std::env::var_os("VICTORY_STACK_POINTER_CHILD").is_none() {
            return;
        }
        // SAFETY: a zeroed CONTEXT is valid.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        context.Rip = 0x1234;
        context.Rsp = reserved_stack_address() as u64;
        // On stderr: libtest writes test names to stdout, without a newline
        // before the test runs when it runs one test at a time.
        eprintln!("{} frames", unwind(context).len());
        std::process::exit(0);
    }

    #[test]
    fn a_stack_pointer_in_the_uncommitted_stack_ends_the_walk() {
        let (code, text) = Child {
            test: "unwind::tests::stack_pointer_child",
            variable: "VICTORY_STACK_POINTER_CHILD",
            value: "1",
        }
        .run();
        assert_eq!(code, Some(0), "{text}");
        assert!(text.lines().any(|l| l == "1 frames"), "{text}");
    }

    /// What [`capture_frames`] saw: our [`unwind`], then the system's walk.
    type Captured = (Vec<u64>, Vec<usize>);

    /// A frame we expect [`unwind`] to report: the helper it is in, and the
    /// helper's name for the message.
    type Expected = (&'static str, fn() -> Captured);

    /// The runtime function entry start covering `at`: the image base the
    /// lookup writes, plus the entry's `BeginAddress`. Asking the system for
    /// both sides of the comparison means a thunk or an incremental-link jump
    /// is not mistaken for a different function.
    fn entry_start_at(at: u64) -> Option<u64> {
        let mut image = 0;
        // SAFETY: looks up the function table entry covering `at`; nothing
        // keeps a reference to the entry past this call.
        let entry = unsafe { RtlLookupFunctionEntry(at, &mut image, null_mut()) };
        if entry.is_null() {
            return None;
        }
        // SAFETY: a non-null entry belongs to `image`, which the lookup wrote,
        // and `BeginAddress` is an offset into that image.
        Some(image + u64::from(unsafe { (*entry).BeginAddress }))
    }

    #[test]
    fn unwinds_like_the_system_walk() {
        // Asking the system's own walk whether the stack looks right is not
        // reliable: on Windows 10 x64 debug builds `RtlCaptureStackBackTrace`
        // returns no frames about half the time. The frames above us are
        // known instead (see `level_a`), so check those first.
        let (frames, walked) = level_a();
        // Frame 0 is in `capture_frames`, which took the context; then its
        // callers in call order, each named by its runtime function entry.
        let helpers: [Expected; 4] = [
            ("capture_frames", capture_frames),
            ("level_c", level_c),
            ("level_b", level_b),
            ("level_a", level_a),
        ];
        assert!(frames.len() >= helpers.len(), "{frames:?}");
        for (i, (name, helper)) in helpers.iter().enumerate() {
            let want = entry_start_at(*helper as usize as u64)
                .unwrap_or_else(|| panic!("no unwind data for {name}"));
            let got = entry_start_at(frames[i])
                .unwrap_or_else(|| panic!("no unwind data for frame {i} at {:#x}", frames[i]));
            assert_eq!(got, want, "frame {i} should be {name}: {frames:?}");
        }
        // A walk that repeats an address is not a walk.
        let mut distinct = frames[..helpers.len()].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), helpers.len(), "repeated: {frames:?}");
        // Where the system's walk does return frames, it must still agree with
        // ours: frame 0 differs (both are in `capture_frames`, at different
        // call sites), and the callers above are the same.
        if walked.len() >= helpers.len() {
            assert!(frames.len() >= walked.len(), "{frames:?} vs {walked:?}");
            let ours: Vec<usize> = frames[1..walked.len()]
                .iter()
                .map(|&f| f as usize)
                .collect();
            assert_eq!(ours.as_slice(), &walked[1..]);
        }
    }

    /// The frames of this call, both ways: our [`unwind`], and the system's
    /// walk, which Windows 10 can come up empty on.
    #[inline(never)]
    fn capture_frames() -> Captured {
        // SAFETY: a zeroed CONTEXT is valid and both calls write buffers of
        // ours; `RtlCaptureContext` wants the context 16-byte aligned too.
        unsafe {
            let mut context = Aligned16(std::mem::zeroed());
            RtlCaptureContext(&mut context.0);
            let mut walked = [null_mut(); 8];
            let n = RtlCaptureStackBackTrace(0, 8, walked.as_mut_ptr(), null_mut());
            let frames = unwind(context.0);
            let walked = walked[..usize::from(n)]
                .iter()
                .map(|&f| f as usize)
                .collect();
            (frames, walked)
        }
    }

    /// `level_a` → `level_b` → `level_c` → [`capture_frames`]: three known
    /// callers above the capture. `black_box` after each call keeps it from
    /// becoming a tail call, which would erase the caller's frame.
    #[inline(never)]
    fn level_a() -> Captured {
        let captured = level_b();
        std::hint::black_box(&captured);
        captured
    }

    /// See [`level_a`].
    #[inline(never)]
    fn level_b() -> Captured {
        let captured = level_c();
        std::hint::black_box(&captured);
        captured
    }

    /// See [`level_a`].
    #[inline(never)]
    fn level_c() -> Captured {
        let captured = capture_frames();
        std::hint::black_box(&captured);
        captured
    }
}
