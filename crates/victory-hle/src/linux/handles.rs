//! The handle table: every kernel object title code holds (files, events,
//! threads, ...) behind one kind of `HANDLE`, so `CloseHandle`,
//! `DuplicateHandle` and the waits work on all of them.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

/// A title `HANDLE` value.
pub(crate) type Handle = usize;

/// A kernel object. Waitable objects say so through `as_waitable`.
pub(crate) trait Object: Any + Send + Sync {
    /// `Some` for objects `WaitForSingleObject` accepts.
    fn as_waitable(&self) -> Option<&dyn Waitable> {
        None
    }
}

/// An object a thread can wait for.
pub(crate) trait Waitable: Send + Sync {
    /// Whether a wait would be satisfied now; takes the object (an
    /// auto-reset event's signal, a semaphore count) when it is.
    fn try_acquire(&self) -> bool;
}

static TABLE: LazyLock<Mutex<Table>> = LazyLock::new(Default::default);

#[derive(Default)]
struct Table {
    next: usize,
    objects: HashMap<Handle, Arc<dyn Object>>,
}

// ponytail: one global lock and no waits yet; the sync slice gives this its real shape.

/// A new handle to `object`. Handle values are multiples of 4 from 4.
pub(crate) fn insert<T: Object>(object: Arc<T>) -> Handle {
    let mut table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.next += 4;
    let handle = table.next;
    table.objects.insert(handle, object);
    handle
}

/// The object behind `handle`, if it is a `T`.
pub(crate) fn get<T: Object>(handle: Handle) -> Option<Arc<T>> {
    let object: Arc<dyn Any + Send + Sync> = object(handle)?;
    object.downcast().ok()
}

/// The object behind `handle`, whatever it is.
pub(crate) fn object(handle: Handle) -> Option<Arc<dyn Object>> {
    let table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.objects.get(&handle).cloned()
}

/// Closes `handle`; false if it was not open.
pub(crate) fn close(handle: Handle) -> bool {
    let mut table = TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.objects.remove(&handle).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Thing(u32);
    impl Object for Thing {}

    #[test]
    fn handles_find_their_object_until_closed() {
        let h = insert(Arc::new(Thing(7)));
        assert_eq!(h % 4, 0);
        assert_eq!(get::<Thing>(h).map(|t| t.0), Some(7));
        assert!(object(h).unwrap().as_waitable().is_none());
        assert!(close(h));
        assert!(get::<Thing>(h).is_none() && !close(h));
    }
}
