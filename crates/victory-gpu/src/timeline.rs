//! A queue's timeline: its work in submission order, held back behind a
//! GPU-side wait (`ID3D12CommandQueue::Wait`) until the fence reaches the
//! value.
//!
//! Work runs on the thread that submits it, as before, unless the queue is
//! held back: then the submission returns at once and the work runs later,
//! on the thread that moves the fence (a signal or a bottom-of-pipe write
//! from another queue, or the title's CPU signal). Nothing blocks a
//! submitting thread behind a wait, so a thread may submit a queue's wait
//! before the signal it waits for.
//!
//! Locks: a timeline's state is taken before a fence's waiters, never
//! after; work runs with no lock held.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::fence::Held;

/// One step of a queue's work.
pub(crate) enum Op {
    /// Hold back what follows until the fence reaches the value.
    Wait(Held, u64),
    /// Command lists to run, a fence to signal.
    Run(Box<dyn FnOnce() + Send>),
}

#[derive(Default)]
pub(crate) struct Timeline {
    state: Mutex<State>,
    /// Signalled whenever work is done or the queue stops running it.
    moved: Condvar,
    /// The latest of this queue's waits (by ticket) a fence released, set
    /// under that fence's lock. A released wait holds even if the value goes
    /// back down before the queue resumes (D3D12: a wait a signal satisfied
    /// stays satisfied); a later wait has a later ticket.
    released: AtomicU64,
}

#[derive(Default)]
struct State {
    ops: VecDeque<Op>,
    /// Ops submitted and ops done (a submission's ticket is its count).
    pushed: u64,
    done: u64,
    /// A thread is running the queue's work.
    busy: bool,
    /// The first op is a wait not reached yet, registered with its fence.
    held: bool,
}

impl Timeline {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs it unless the queue is held back.
    /// Work another thread is running for the queue is waited for, as a
    /// submission's own work is.
    pub(crate) fn submit(self: &Arc<Self>, op: Op) {
        let mut s = self.lock();
        s.ops.push_back(op);
        s.pushed += 1;
        let ticket = s.pushed;
        while s.busy && s.done < ticket {
            s = self.moved.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        if s.done >= ticket {
            return;
        }
        // Still held back unless the value was reached by a path that does
        // not wake us (a CPU write to the fence's memory).
        if s.held
            && let Some(Op::Wait(fence, value)) = s.ops.front()
            && !fence.reached(*value)
        {
            return;
        }
        self.run(s);
    }

    /// A fence's value reached the wait with ticket `wait`. Called under the
    /// fence's lock; [`Self::wake`] follows once it is dropped.
    pub(crate) fn release(&self, wait: u64) {
        self.released.fetch_max(wait, Ordering::AcqRel);
    }

    /// Whether a fence released the wait with ticket `wait`.
    pub(crate) fn is_released(&self, wait: u64) -> bool {
        self.released.load(Ordering::Acquire) >= wait
    }

    /// The fence this queue waits for moved: runs what it held back.
    pub(crate) fn wake(self: &Arc<Self>) {
        let s = self.lock();
        if !s.busy {
            self.run(s);
        }
    }

    fn run<'a>(self: &'a Arc<Self>, mut s: MutexGuard<'a, State>) {
        s.busy = true;
        s.held = false;
        while let Some(op) = s.ops.pop_front() {
            match op {
                Op::Wait(fence, value) => {
                    // Checked and registered under the fence's lock, so a
                    // signal after this check wakes us. The ticket is the
                    // op's place in submission order (as in `submit`).
                    if !fence.wait_on_gpu(value, s.done + 1, self) {
                        s.ops.push_front(Op::Wait(fence, value));
                        s.held = true;
                        break;
                    }
                }
                Op::Run(work) => {
                    drop(s);
                    work();
                    s = self.lock();
                }
            }
            s.done += 1;
            self.moved.notify_all();
        }
        s.busy = false;
        self.moved.notify_all();
    }
}
