//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// included in `QUEUED` (flipped back to false once in `complete`)
    counted: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

fn finish_pending(p: &Pending, real: usize, queued: &AtomicUsize) -> bool {
    let mut done = p.done.lock().unwrap();
    let orphan = p.orphan.load(Ordering::Acquire);
    if !orphan {
        p.real.store(real, Ordering::Release);
    }
    *done = true;
    if p.counted.swap(false, Ordering::AcqRel) {
        queued.fetch_sub(1, Ordering::Relaxed);
    }
    p.cv.notify_all();
    orphan
}

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = finish_pending(p, real, &QUEUED);
    // outside `done`: the Release override takes PENDING
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// Pipelines currently queued for or building on the background workers (excludes dormant lazy stand-ins).
static QUEUED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&p, job());
                }
            });
        }
        Mutex::new(tx)
    })
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, false);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        counted: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        orphan: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
    }
}

/// Mark `p` sent and increment `queued` once; returns `true` iff `p` should be sent to workers
/// (returns `false` if already sent or already completed inline by `resolve_wait`).
fn mark_enqueued(p: &Pending, queued: &AtomicUsize) -> bool {
    if !p.sent.swap(true, Ordering::AcqRel) {
        let done = p.done.lock().unwrap();
        if !*done {
            p.counted.store(true, Ordering::Release);
            queued.fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Send a pipeline to the workers once (skipped if already completed inline by `resolve_wait`).
fn enqueue(p: Arc<Pending>) {
    if mark_enqueued(&p, &QUEUED) {
        let _ = queue().lock().unwrap().send(p);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            enqueue(p);
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(&p, job());
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, queued or building on workers)
pub fn counts() -> (usize, usize, usize) {
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), QUEUED.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_pending(compute: bool) -> Pending {
        Pending {
            job: Mutex::new(Some(Box::new(|| 0))),
            sent: std::sync::atomic::AtomicBool::new(false),
            counted: std::sync::atomic::AtomicBool::new(false),
            real: AtomicUsize::new(0),
            done: Mutex::new(false),
            cv: Condvar::new(),
            compute,
            orphan: Default::default(),
        }
    }

    #[test]
    fn dormant_lazy_pipelines_are_not_counted_as_queued() {
        // Isolated local counter: immune to parallel tests touching the global `QUEUED`.
        let queued = AtomicUsize::new(0);

        let dormant = new_pending(false);
        assert_eq!(queued.load(Ordering::Relaxed), 0, "unrequested lazy pipeline must not inflate queued count");

        // Enqueuing once increments queued; a duplicate enqueue is a no-op; completion drains it.
        assert!(mark_enqueued(&dormant, &queued));
        assert_eq!(queued.load(Ordering::Relaxed), 1);
        assert!(!mark_enqueued(&dormant, &queued));
        assert_eq!(queued.load(Ordering::Relaxed), 1);
        assert!(!finish_pending(&dormant, 0, &queued));
        assert_eq!(queued.load(Ordering::Relaxed), 0);

        // Inline completion (resolve_wait) followed by resolve/enqueue must never increment or leak queued.
        let inline = new_pending(true);
        assert!(!finish_pending(&inline, 0, &queued));
        assert_eq!(queued.load(Ordering::Relaxed), 0);
        assert!(!mark_enqueued(&inline, &queued));
        assert_eq!(queued.load(Ordering::Relaxed), 0);
    }
}
