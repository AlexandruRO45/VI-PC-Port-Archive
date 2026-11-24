use std::sync::Mutex;

/// One submission at a time, protected by the host's submission lock.
/// This separate mutex linearizes descriptor retirement with fence completion.
#[derive(Default)]
pub(super) struct Retirement(Mutex<Option<Vec<Box<dyn Send>>>>);

impl Retirement {
    pub(super) fn begin(&self) {
        let mut active = self.0.lock().unwrap_or_else(|e| e.into_inner());
        assert!(active.is_none(), "overlapping submissions on one host");
        *active = Some(Vec::new());
    }

    pub(super) fn retire(&self, resource: impl Send + 'static) {
        let mut active = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(active) = &mut *active {
            active.push(Box::new(resource));
        } else {
            drop(active);
            drop(resource);
        }
    }

    pub(super) fn finish(&self) {
        let ready = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(ready);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    struct Payload(Vec<u8>, Arc<AtomicUsize>);
    impl Drop for Payload {
        fn drop(&mut self) {
            self.1.fetch_sub(self.0.len(), Relaxed);
        }
    }

    #[test]
    fn idle_and_concurrent_retirement_do_not_strand_resources() {
        let queue = Retirement::default();
        let live = Arc::new(AtomicUsize::new(0));
        let payload = |n| {
            live.fetch_add(n, Relaxed);
            Payload(vec![1; n], live.clone())
        };
        queue.retire(payload(1024));
        assert_eq!(live.load(Relaxed), 0);
        queue.begin();
        queue.retire(payload(4 << 20));
        assert_eq!(live.load(Relaxed), 4 << 20);
        queue.finish();
        assert_eq!(live.load(Relaxed), 0);
        std::thread::scope(|s| {
            for _ in 0..128 {
                queue.begin();
                let p = payload(1);
                let queue = &queue;
                let worker = s.spawn(move || queue.retire(p));
                queue.finish();
                worker.join().unwrap();
                assert_eq!(live.load(Relaxed), 0);
            }
        });
    }
}
