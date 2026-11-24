//! Guest threads: host threads that run title code, each with a TEB at
//! its GS base, a stack with a guard page, and the attached images' TLS
//! callbacks on start and exit.

use std::ffi::c_void;

/// A title thread's start routine (`LPTHREAD_START_ROUTINE`).
pub type StartRoutine = unsafe extern "win64" fn(*mut c_void) -> u32;

/// What [`spawn`] reports once the thread exists.
#[derive(Debug, Clone, Copy)]
pub struct Spawned {
    /// The Windows thread id (`GetCurrentThreadId` on that thread).
    pub id: u32,
}

/// Starts `start(param)` on a new guest thread with `stack_size` bytes of
/// stack (0: the image's default). `on_exit` runs on that thread with its
/// exit code once title code is done with it (the start routine returned,
/// or it called `ExitThread`). Returns once the thread has its TEB; the
/// attach callbacks run on the new thread.
pub fn spawn(
    start: StartRoutine,
    param: *mut c_void,
    stack_size: usize,
    on_exit: Box<dyn FnOnce(u32) + Send>,
) -> std::io::Result<Spawned> {
    // ponytail: a plain host thread until guest threads get a TEB and TLS.
    let param = param as usize;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut builder = std::thread::Builder::new();
    if stack_size != 0 {
        builder = builder.stack_size(stack_size);
    }
    builder.spawn(move || {
        let _ = tx.send(current_id());
        // SAFETY: title code given to us as a thread start routine.
        let code = unsafe { start(param as *mut c_void) };
        on_exit(code);
    })?;
    let id = rx.recv().map_err(std::io::Error::other)?;
    Ok(Spawned { id })
}

/// The calling thread's Windows thread id.
pub fn current_id() -> u32 {
    // SAFETY: gettid has no preconditions.
    unsafe { libc::gettid() as u32 }
}
