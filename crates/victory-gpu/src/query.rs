//! Query heaps. With a host GPU, occlusion and timestamp heaps have a host
//! query heap behind them: `BeginQuery`/`EndQuery` run on the host list and
//! `ResolveQueryData` copies the host's results into the title's buffer
//! once the list has run. Other heaps, and queries the host never ended,
//! answer with the plausible values of `list::resolve_queries`.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

/// Heap types (`D3D12_QUERY_HEAP_TYPE`).
const HEAP_OCCLUSION: u32 = 0;
const HEAP_TIMESTAMP: u32 = 1;

/// Query types (`D3D12_QUERY_TYPE`).
pub(crate) const OCCLUSION: u32 = 0;
pub(crate) const BINARY_OCCLUSION: u32 = 1;
pub(crate) const TIMESTAMP: u32 = 2;

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    #[cfg(windows)]
    host: Option<host::HostHeap>,
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(windows)]
        host: host::HostHeap::new(desc),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

/// Whether a query of type `kind` belongs in a heap of type `heap`.
fn fits(heap: u32, kind: u32) -> bool {
    match heap {
        HEAP_OCCLUSION => kind == OCCLUSION || kind == BINARY_OCCLUSION,
        HEAP_TIMESTAMP => kind == TIMESTAMP,
        _ => false,
    }
}

#[cfg(windows)]
pub(crate) use host::{Open, begin, end, end_open, resolve};

#[cfg(windows)]
mod host {
    use std::sync::{Arc, Mutex};

    use windows::Win32::Graphics::Direct3D12::*;

    use super::{Desc, HEAP_OCCLUSION, HEAP_TIMESTAMP, Heap, TIMESTAMP, fits};
    use crate::host::HostList;
    use crate::object::This;
    use crate::queue::Written;

    /// What the host recorded for one query, as lists were recorded.
    #[derive(Debug, Clone, Copy, Default, PartialEq)]
    pub(crate) enum Mark {
        /// Never ended on the host: its slot holds nothing to resolve.
        #[default]
        Never,
        /// Ended; `dropped` if a draw or dispatch inside it did not run on
        /// the host, so its sample count would undercount the title's.
        Ended { dropped: bool },
    }

    /// The host side of a title heap.
    pub(crate) struct HostHeap {
        pub(crate) heap: ID3D12QueryHeap,
        kind: u32,
        pub(crate) marks: Arc<Mutex<Vec<Mark>>>,
    }

    impl HostHeap {
        pub(super) fn new(desc: Desc) -> Option<HostHeap> {
            if !matches!(desc.kind, HEAP_OCCLUSION | HEAP_TIMESTAMP) || desc.count == 0 {
                return None;
            }
            let host = crate::host::host()?;
            let mut heap: Option<ID3D12QueryHeap> = None;
            // SAFETY: plain object creation.
            let made = unsafe {
                host.device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE(desc.kind as i32),
                        Count: desc.count,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
            };
            if let Err(e) = made {
                tracing::warn!(error = %e, ?desc, "no host query heap; its queries answer with plausible values");
                return None;
            }
            Some(HostHeap {
                heap: heap?,
                kind: desc.kind,
                marks: Arc::new(Mutex::new(vec![Mark::Never; desc.count as usize])),
            })
        }
    }

    /// The host heap behind a title heap, if a query of `kind` at `index`
    /// fits it.
    fn host_heap<'a>(heap: This, kind: u32, index: u32, count: u32) -> Option<&'a HostHeap> {
        if heap.is_null() {
            return None;
        }
        // SAFETY: the title passes one of our heaps.
        let heap = unsafe { &*heap.cast::<Heap>() };
        let host = heap.host.as_ref()?;
        (fits(host.kind, kind) && u64::from(index) + u64::from(count) <= u64::from(heap.desc.count))
            .then_some(host)
    }

    /// A query left open in a list: ended when the list closes.
    pub(crate) struct Open {
        heap: ID3D12QueryHeap,
        marks: Arc<Mutex<Vec<Mark>>>,
        kind: u32,
        index: u32,
        /// The list's dropped-work count when the query began.
        dropped: u64,
    }

    /// `BeginQuery` on the host (occlusion only: timestamps only end).
    pub(crate) fn begin(list: &HostList, heap: This, kind: u32, index: u32) {
        let Some(host) = host_heap(heap, kind, index, 1) else {
            return;
        };
        if kind == TIMESTAMP {
            return;
        }
        let mut open = list.open_queries();
        if open.iter().any(|o| o.heap == host.heap && o.index == index) {
            return;
        }
        // SAFETY: an open list and a live heap; the index fits it.
        unsafe {
            list.list
                .BeginQuery(&host.heap, D3D12_QUERY_TYPE(kind as i32), index)
        };
        // The title may release its heap before the list runs.
        list.keep(&host.heap);
        open.push(Open {
            heap: host.heap.clone(),
            marks: host.marks.clone(),
            kind,
            index,
            dropped: list.dropped(),
        });
    }

    /// `EndQuery` on the host: the query's slot now holds a result.
    pub(crate) fn end(list: &HostList, heap: This, kind: u32, index: u32) {
        let Some(host) = host_heap(heap, kind, index, 1) else {
            return;
        };
        let dropped = if kind == TIMESTAMP {
            // Occlusion heaps were kept when the query began.
            list.keep(&host.heap);
            false
        } else {
            let mut open = list.open_queries();
            let Some(at) = open
                .iter()
                .position(|o| o.heap == host.heap && o.index == index)
            else {
                // Ended without a begin on this list: nothing to end.
                return;
            };
            let o = open.swap_remove(at);
            o.dropped != list.dropped()
        };
        // SAFETY: an open list and a live heap; the index fits it.
        unsafe {
            list.list
                .EndQuery(&host.heap, D3D12_QUERY_TYPE(kind as i32), index)
        };
        host.marks.lock().unwrap_or_else(|e| e.into_inner())[index as usize] =
            Mark::Ended { dropped };
    }

    /// At list close: queries still open end there, as not to be trusted.
    pub(crate) fn end_open(list: &HostList, open: Vec<Open>) {
        for o in open {
            // SAFETY: an open list and a live heap; the index fit it.
            unsafe {
                list.list
                    .EndQuery(&o.heap, D3D12_QUERY_TYPE(o.kind as i32), o.index)
            };
            o.marks.lock().unwrap_or_else(|e| e.into_inner())[o.index as usize] =
                Mark::Ended { dropped: true };
        }
    }

    /// `ResolveQueryData` through the host: true if the results at `dst`
    /// (8 bytes each) will be the host's. A query whose work was partly
    /// dropped reports `fallback` instead, so missing draws never make the
    /// title cull what it should draw.
    pub(crate) fn resolve(
        list: &HostList,
        heap: This,
        (kind, first, count): (u32, u32, u32),
        dst: *mut u8,
        fallback: Written,
    ) -> bool {
        let Some(host) = host_heap(heap, kind, first, count) else {
            return false;
        };
        let marks: Vec<Mark> = host.marks.lock().unwrap_or_else(|e| e.into_inner())
            [first as usize..(first + count) as usize]
            .to_vec();
        // Resolving a slot never ended is invalid on the host.
        if count == 0 || marks.contains(&Mark::Never) {
            return false;
        }
        let Some((back, data)) = list.readback(u64::from(count) * 8) else {
            return false;
        };
        list.keep(&host.heap);
        // SAFETY: an open list, a live heap and a read-back buffer of
        // `count` results; every slot was ended.
        unsafe {
            list.list.ResolveQueryData(
                &host.heap,
                D3D12_QUERY_TYPE(kind as i32),
                first,
                count,
                &back,
                0,
            )
        };
        if kind == TIMESTAMP {
            crate::host::calibrate();
        }
        static FIRST: [std::sync::Once; 3] = [const { std::sync::Once::new() }; 3];
        FIRST[kind.min(2) as usize]
            .call_once(|| tracing::info!(kind, "queries resolved on the host"));
        for (i, mark) in marks.iter().enumerate() {
            let what = match mark {
                Mark::Ended { dropped: true } => fallback,
                _ => Written::Read {
                    at: data as usize + i * 8,
                    timestamp: kind == TIMESTAMP,
                },
            };
            // SAFETY: inside the title's buffer (checked by the caller).
            list.defer_write(unsafe { dst.add(i * 8) }, what, 8);
        }
        true
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::queue::Written;

    /// A title heap as `create` makes it, without handing it out.
    fn heap(kind: u32, count: u32) -> Box<Heap> {
        let desc = Desc {
            kind,
            count,
            nodes: 0,
        };
        Box::new(Heap {
            object: Object::new(&FUNCTIONS),
            desc,
            host: host::HostHeap::new(desc),
        })
    }

    #[test]
    fn occlusion_counts_come_from_the_host_unless_work_was_dropped() {
        let Some(gpu) = crate::host::host() else {
            return;
        };
        let heap = heap(HEAP_OCCLUSION, 3);
        let this = (&raw const *heap).cast_mut().cast();
        let list = gpu.new_list().unwrap();
        // Nothing drawn: the host counts no samples.
        begin(&list, this, OCCLUSION, 0);
        end(&list, this, OCCLUSION, 0);
        // A draw dropped inside: the title must not cull on this one.
        begin(&list, this, BINARY_OCCLUSION, 1);
        list.note_dropped();
        end(&list, this, BINARY_OCCLUSION, 1);
        // Left open: ended at close, and not trusted either.
        begin(&list, this, OCCLUSION, 2);
        let mut results = [u64::MAX; 3];
        let dst = results.as_mut_ptr().cast::<u8>();
        assert!(resolve(
            &list,
            this,
            (OCCLUSION, 0, 1),
            dst,
            Written::Value(1 << 20)
        ));
        // SAFETY: inside `results`.
        let second = unsafe { dst.add(8) };
        assert!(resolve(
            &list,
            this,
            (BINARY_OCCLUSION, 1, 1),
            second,
            Written::Value(1)
        ));
        list.close();
        assert_eq!(gpu.log_messages(), None, "debug layer");
        gpu.execute(&[&list]);
        assert_eq!(results[..2], [0, 1]);
        assert_eq!(results[2], u64::MAX, "resolved before it ended");

        // The query left open was ended at close, as dropped.
        let next = gpu.new_list().unwrap();
        // SAFETY: inside `results`.
        let third = unsafe { dst.add(16) };
        assert!(resolve(
            &next,
            this,
            (OCCLUSION, 2, 1),
            third,
            Written::Value(1 << 20)
        ));
        next.close();
        assert_eq!(gpu.log_messages(), None, "debug layer");
        gpu.execute(&[&next]);
        assert_eq!(results[2], 1 << 20);
    }

    #[test]
    fn queries_never_ended_or_of_another_type_fall_back() {
        let Some(gpu) = crate::host::host() else {
            return;
        };
        let heap = heap(HEAP_OCCLUSION, 2);
        let this = (&raw const *heap).cast_mut().cast();
        let list = gpu.new_list().unwrap();
        let mut results = [0u64; 2];
        let dst = results.as_mut_ptr().cast::<u8>();
        assert!(!resolve(
            &list,
            this,
            (OCCLUSION, 0, 2),
            dst,
            Written::Value(1)
        ));
        assert!(!resolve(
            &list,
            this,
            (TIMESTAMP, 0, 1),
            dst,
            Written::Value(1)
        ));
        assert!(!resolve(
            &list,
            this,
            (OCCLUSION, 1, 2),
            dst,
            Written::Value(1)
        ));
        list.close();
        assert_eq!(gpu.log_messages(), None, "debug layer");
        gpu.execute(&[&list]);
    }

    #[test]
    fn timestamps_are_on_the_titles_gpu_clock() {
        let Some(gpu) = crate::host::host() else {
            return;
        };
        let heap = heap(HEAP_TIMESTAMP, 2);
        let this = (&raw const *heap).cast_mut().cast();
        let list = gpu.new_list().unwrap();
        end(&list, this, TIMESTAMP, 0);
        end(&list, this, TIMESTAMP, 1);
        let mut results = [0u64; 3];
        let dst = results.as_mut_ptr().cast::<u8>();
        assert!(resolve(
            &list,
            this,
            (TIMESTAMP, 0, 2),
            dst,
            Written::GpuTimestamp
        ));
        list.close();
        assert_eq!(gpu.log_messages(), None, "debug layer");
        gpu.execute(&[&list]);
        // The title's clock right after the list ran.
        Written::GpuTimestamp.write(results[2..].as_mut_ptr().cast(), 8);
        let [first, second, now] = results;
        assert!(first > 0 && second >= first, "{results:?}");
        // 100 MHz ticks: within a second of the CPU's reading of the clock.
        assert!(now.abs_diff(second) < 100_000_000, "{results:?}");
    }
}
