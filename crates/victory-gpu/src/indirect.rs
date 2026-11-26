//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours ([`crate::compute`],
//! which Wine's compiler builds too: without the clamp every indirect
//! execution is dropped).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension except a dispatch's x, 2^20 per dispatch,
//! one-dimensional mesh dispatches, the budget) cut legitimate work beyond
//! them; raise them if the title needs more. A dispatch over the host's
//! per-dimension limit is clamped into 17 chunks (kind 3), each with its
//! own base for the shader's group id.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::compute::{compute, constants, root_view};
use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint counted; };
// The title's records, read only; the clamped ones go to the list's own.
// In words: the caller refuses addresses and strides not multiples of 4.
StructuredBuffer<uint> args : register(t0);
// The title's record count, when `counted` (chunked dispatch only).
StructuredBuffer<uint> counts : register(t1);
RWStructuredBuffer<uint> dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWStructuredBuffer<uint> budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// A dispatch larger than D3D12's per-dimension limit runs in chunks of x.
static const uint CHUNK = 65535;
static const uint CHUNKS = 17;
uint3 load3(uint at) {
    return uint3(args[at], args[at + 1], args[at + 2]);
}
void store3(uint at, uint3 v) {
    dest[at] = v.x;
    dest[at + 1] = v.y;
    dest[at + 2] = v.z;
}
// A whole dispatch's work, before its x is split. Compare by division so
// that x * y * z cannot wrap before the 2^20-groups test.
uint dispatch_cost(uint at) {
    uint x = min(args[at], GROUPS);
    uint y = min(args[at + 1], CHUNK);
    uint z = min(args[at + 2], CHUNK);
    uint yz = y * z;
    if (yz != 0 && x > GROUPS / yz) {
        x = 0;
    }
    return x * yz * 64;
}
// Clamps the title's record at word `at` into `dest` at word `to`; `host`
// is the host record's index (a chunk of one title record for kind 3).
// Returns the work charged to the budget.
uint clamped(uint at, uint to, uint host) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args[at], 1u << 24);
        uint instances = min(args[at + 1], (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest[to] = count;
        dest[to + 1] = instances;
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        dest[to + 2] = args[at + 2];
        dest[to + 3] = args[at + 3];
        if (kind == 1) {
            dest[to + 4] = args[at + 4];
        }
    } else if (kind == 2) {
        uint3 g = min(load3(at), uint3(CHUNK, CHUNK, CHUNK));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        store3(to, g);
    } else if (kind == 3) {
        // Chunked dispatch: the command signature reads a base and three
        // group counts from each 16-byte host record.
        uint chunk = host % CHUNKS;
        uint base = chunk * CHUNK;
        uint x = args[at];
        dest[to] = base;
        dest[to + 1] = x > base ? min(x - base, CHUNK) : 0;
        dest[to + 2] = min(args[at + 1], CHUNK);
        dest[to + 3] = min(args[at + 2], CHUNK);
        // Charge the whole title record once, to its first chunk.
        cost = chunk == 0 ? dispatch_cost(at) : 0;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = load3(at);
        g = uint3(min(g.x, CHUNK), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        store3(to, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    // A chunked record rejected by the budget: all its chunks run empty.
    uint reject = 0xffffffffu;
    if (lane == 0) {
        left = BUDGET - min(budget[0], BUDGET);
    }
    uint total = kind == 3 ? records * CHUNKS : records;
    for (uint tile = 0; tile < total / 64 + min(total % 64, 1u); tile++) {
        uint host = tile * 64 + lane;
        uint at = kind == 3 ? (host / CHUNKS) * (stride / 4) : host * (stride / 4);
        uint to = kind == 3 ? host * 4 : at;
        uint cost = 0;
        if (host < total) {
            cost = clamped(at, to, host);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (kind == 3) {
                    uint h = tile * 64 + i;
                    if (h < total && h % CHUNKS == 0 && costs[i] > left) {
                        reject = h;
                        costs[i] = PAST;
                        left = 0;
                    }
                } else if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
            if (kind == 3) {
                for (uint j = 0; j < 64; j++) {
                    uint h = tile * 64 + j;
                    if (h < total && h - h % CHUNKS == reject) {
                        costs[j] = PAST;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dest[to] = 0;
            if (kind == 3) {
                dest[to + 1] = 0;
                dest[to + 2] = 0;
                dest[to + 3] = 0;
            }
        }
    }
    if (lane == 0) {
        budget[0] = BUDGET - left;
        if (kind == 3 && counted != 0) {
            // ExecuteIndirect reads this as its count: one title record
            // became CHUNKS host records.
            dest[records * CHUNKS * 4] = min(counts[0], records) * CHUNKS;
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

/// The clamp, its root signature the record (b0), the title's records
/// (t0) and count (t1), the budget (u1) and the clamped records (u0), as
/// [`bound`] binds.
fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let params = [
        constants(0, 4),
        root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0, 0),
        root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 1, 0),
        root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0),
        root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1, 0),
    ];
    let (root, pso) = compute(device, s!("argument clamp"), SOURCE, 0, &params)?;
    Ok(Clamp { root, pso })
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering: 3 is a chunked dispatch) `stride` bytes apart at
/// host address `gpu` (a title buffer's copy, only read) into the list's
/// own argument buffer, left ready to be read as arguments: that buffer,
/// for [`done`] after the execution. `count` is the title's count buffer's
/// address, read for kind 3 and written into the output's last word as
/// the host record count. Replaces the list's compute root signature and
/// pipeline (callers bind theirs after). `None` if the clamp is not
/// available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<u64>,
) -> Option<ID3D12Resource> {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let chunked = kind == 3;
    if count.is_some_and(|at| !at.is_multiple_of(4)) {
        return None;
    }
    let host_stride = if chunked { 16 } else { stride };
    let words = (host_stride / 4) as u64 * u64::from(records.max(1));
    let count_word = chunked && count.is_some();
    let out = list.arguments(4 * (words + u64::from(count_word)))?;
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(count.is_some())];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        l.SetComputeRootShaderResourceView(4, count.unwrap_or(gpu));
        // One group walks every host record.
        l.Dispatch(
            (if chunked {
                records.saturating_mul(17)
            } else {
                records
            })
            .min(1),
            1,
            1,
        );
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps the records of `kind` in `title`, `per` words apart, in a
    /// list of their own: every record word of the list's argument buffer
    /// after, and the list's budget spent. `None` without a host GPU.
    fn every_word(kind: u32, per: usize, title: &[u32]) -> Option<(Vec<u32>, u32)> {
        let host = super::host()?;
        let records = (title.len() / per) as u32;
        let (buffer, data) = host
            .shared_buffer(std::mem::size_of_val(title) as u64)
            .unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs; the
        // list's argument buffer read after it has.
        unsafe {
            data.cast::<u32>()
                .copy_from_nonoverlapping(title.as_ptr(), title.len());
            let out = super::bound(
                &list,
                buffer.GetGPUVirtualAddress(),
                (kind, per as u32 * 4, records),
                None,
            )
            .unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out.Map(0, None, Some(&mut p)).unwrap();
            let words = std::slice::from_raw_parts(p.cast::<u32>(), title.len()).to_vec();
            Some((words, list.budget_spent().unwrap()))
        }
    }

    /// Every word the clamp writes, of each kind: counts and groups
    /// bounded, the rest of a record copied as it is (5-word records
    /// 20 bytes apart: no words of one land in another), the work charged
    /// to the list's budget; records past it run empty with their other
    /// words kept, and one of no work still runs.
    #[test]
    fn clamped_records_hold_every_word() {
        const MANY: u32 = 1 << 24;
        // Draws: vertex count, instances, first vertex, first instance.
        let Some(draws) = every_word(0, 4, &[MANY + 7, 5, 11, 13, 3, 2, 1, 9]) else {
            return; // no host GPU here
        };
        assert_eq!(draws, (vec![MANY, 4, 11, 13, 3, 2, 1, 9], (1 << 26) + 6));
        // Indexed: index count, instances, first index, base vertex, first
        // instance.
        let indexed = every_word(
            1,
            5,
            &[10, 3, 100, -16i32 as u32, 7, u32::MAX, u32::MAX, 1, 2, 3],
        )
        .unwrap();
        assert_eq!(
            indexed,
            (
                vec![10, 3, 100, -16i32 as u32, 7, MANY, 4, 1, 2, 3],
                (1 << 26) + 30
            )
        );
        // Dispatches: each dimension bounded, and a dispatch of more than
        // 2^20 groups none.
        let dispatches = every_word(2, 3, &[3, 2, 1, 65_536, 17, 2, 70_000, 7, 2]).unwrap();
        assert_eq!(
            dispatches,
            (vec![3, 2, 1, 0, 0, 0, 65_535, 7, 2], (6 + 65_535 * 14) * 64)
        );
        // Mesh dispatches: one-dimensional.
        let meshes = every_word(10, 3, &[70_000, 7, 2, 5, 0, 3]).unwrap();
        assert_eq!(meshes, (vec![65_535, 1, 1, 5, 0, 1], 65_535 * 128));
        // Two draws of 2^26 vertex instances spend the budget; a third, a
        // small one after it, run empty; one of no work runs.
        let mut title = [MANY, 4, 0, 0].repeat(3);
        title.extend([1, 1, 5, 6, 0, 9, 1, 2]);
        let past = every_word(0, 4, &title).unwrap();
        let mut want = [MANY, 4, 0, 0].repeat(2);
        want.extend([0, 4, 0, 0, 0, 1, 5, 6, 0, 9, 1, 2]);
        assert_eq!(past, (want, 1 << 27));
    }

    /// A title dispatch over the host's per-dimension limit becomes 17
    /// host records: three nonempty chunks of x, then empty ones with
    /// their bases still set. The title's count becomes the host count.
    #[test]
    fn chunked_dispatch_records_hold_every_word() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, title_data) = host.shared_buffer(12).unwrap();
        let (counts, count_data) = host.shared_buffer(4).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: mapped buffers written before the list runs, read after.
        unsafe {
            title_data
                .cast::<u32>()
                .copy_from_nonoverlapping([140_000u32, 1, 1].as_ptr(), 3);
            count_data.cast::<u32>().write(1);
            let out = super::bound(
                &list,
                title.GetGPUVirtualAddress(),
                (3, 12, 1),
                Some(counts.GetGPUVirtualAddress()),
            )
            .unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out.Map(0, None, Some(&mut p)).unwrap();
            let words = std::slice::from_raw_parts(p.cast::<u32>(), 17 * 4 + 1).to_vec();
            let mut want = vec![0, 65_535, 1, 1, 65_535, 65_535, 1, 1, 131_070, 8_930, 1, 1];
            for chunk in 3..17 {
                want.extend([chunk * 65_535, 0, 1, 1]);
            }
            want.push(17);
            assert_eq!(words, want);
        }
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, None).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), None).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: a mapped buffer of ours, the lists have run.
            let title = unsafe { data.cast::<[u32; 3]>().read_volatile() };
            assert_eq!(title, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }
}
