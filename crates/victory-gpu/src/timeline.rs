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
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::fence::Held;

/// One step of a queue's work.
pub(crate) enum Op {
    /// Hold back what follows until the fence reaches the value.
    Wait(Held, u64),
    /// Hold back what follows until a memory cell a list recorded a
    /// `wait_value` on reads `cmp value` true `(address, comparison,
    /// value, width)`.
    #[cfg_attr(not(windows), allow(dead_code))]
    WaitValue(usize, u32, u64, u8),
    /// Command lists to run, a fence to signal.
    Run(Box<dyn FnOnce() + Send>),
}

#[derive(Default)]
pub(crate) struct Timeline {
    state: Mutex<State>,
    /// Signalled whenever work is done or the queue stops running it.
    moved: Condvar,
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
    /// Correlation key for this queue's trace events.
    pub(crate) fn id(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs it unless the queue is held back.
    /// Work another thread is running for the queue is waited for, as a
    /// submission's own work is.
    pub(crate) fn submit(self: &Arc<Self>, op: Op) {
        self.submit_batch(vec![op]);
    }

    /// Enqueues a submission and its gates together, so another caller
    /// cannot insert work between a list's wait and the list itself.
    pub(crate) fn submit_batch(self: &Arc<Self>, ops: Vec<Op>) {
        let mut s = self.lock();
        s.pushed += ops.len() as u64;
        s.ops.extend(ops);
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
            && match s.ops.front() {
                Some(Op::Wait(fence, value)) => !fence.reached(*value),
                Some(&Op::WaitValue(addr, cmp, value, width)) => {
                    !crate::fence::mem_wait_reached(addr, cmp, value, width)
                }
                _ => false,
            }
        {
            return;
        }
        self.run(s);
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
        let mut was_held = std::mem::take(&mut s.held);
        while let Some(op) = s.ops.pop_front() {
            let queue = self.id();
            match op {
                Op::Wait(fence, value) => {
                    // Checked and registered under the fence's lock, so a
                    // signal after this check wakes us.
                    if !fence.wait_on_gpu(value, self) {
                        if !was_held {
                            tracing::debug!(
                                target: "victory_gpu::xsync", queue = format_args!("{queue:#x}"),
                                fence = ?fence.object(), value, "queue fence wait parked"
                            );
                        }
                        s.ops.push_front(Op::Wait(fence, value));
                        s.held = true;
                        break;
                    }
                    if was_held {
                        tracing::debug!(
                            target: "victory_gpu::xsync", queue = format_args!("{queue:#x}"),
                            fence = ?fence.object(), value, "queue fence wait released"
                        );
                    }
                }
                Op::WaitValue(addr, cmp, value, width) => {
                    // Checked and registered under `MEMWAITS`, so a write
                    // after this check wakes us.
                    if !crate::fence::memwait_on_gpu(addr, cmp, value, width, self) {
                        if !was_held {
                            tracing::debug!(
                                target: "victory_gpu::xsync", queue = format_args!("{queue:#x}"),
                                address = format_args!("{addr:#x}"), cmp, value, width,
                                "queue memory wait parked"
                            );
                        }
                        s.ops.push_front(Op::WaitValue(addr, cmp, value, width));
                        s.held = true;
                        break;
                    }
                    if was_held {
                        tracing::debug!(
                            target: "victory_gpu::xsync", queue = format_args!("{queue:#x}"),
                            address = format_args!("{addr:#x}"), cmp, value, width,
                            "queue memory wait released"
                        );
                    }
                }
                Op::Run(work) => {
                    drop(s);
                    work();
                    s = self.lock();
                }
            }
            was_held = false;
            s.done += 1;
            self.moved.notify_all();
        }
        s.busy = false;
        self.moved.notify_all();
    }
}
