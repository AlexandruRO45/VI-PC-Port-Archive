//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
RWByteAddressBuffer args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` in place; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args.Load(at), 1u << 24);
        uint instances = min(args.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        args.Store2(at, uint2(count, instances));
    } else if (kind == 2) {
        uint3 g = min(args.Load3(at), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = args.Load3(at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args.Store3(at, g);
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
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            args.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
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

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize())
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> bool {
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    let l = &list.list;
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and a live buffer
    // the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
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
    use windows::Win32::Graphics::Direct3D12::*;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). One record more follows them and must stay as
    /// written. `None` without a host GPU.
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
            for _ in 0..calls {
                assert!(super::bound(&list, &buffer, gpu, spec));
                super::done(&list, &buffer);
            }
            list.close();
            host.execute(&[&list]);
            for at in all - per..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| data.add(i * per).read_volatile();
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

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    #[test]
    fn bound_rejects_unaligned_gpu_or_stride() {
        let Some(host) = super::host() else {
            return;
        };
        if super::clamp().is_none() {
            return;
        }
        let list = host.new_list().unwrap();
        let buf = committed_buffer(
            &host.device,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            256,
        );
        // SAFETY: valid default buffer.
        let gpu = unsafe { buf.GetGPUVirtualAddress() };
        assert!(!super::bound(&list, &buf, gpu + 2, (0, 16, 1)));
        assert!(!super::bound(&list, &buf, gpu, (0, 14, 1)));
        list.close();
    }

    #[test]
    fn argument_clamp_bounds_and_budget_on_gpu() {
        let Some(host) = super::host() else {
            return;
        };
        if super::clamp().is_none() {
            return;
        }

        // Draw (kind = 0, stride = 16):
        // 0: normal draw stays untouched.
        // 1: oversized vertices & instances clamp to (1<<24, 4), keeping start args.
        // Total cost = 400 + (1<<26) <= BUDGET (1<<27), so neither is zeroed.
        let draw_in: [[u32; 4]; 2] = [[100, 4, 1, 2], [u32::MAX, 100, 7, 9]];
        // Second draw on the same HostList pushes its accumulated budget past
        // BUDGET (400 + 2*(1<<26) > 1<<27), so its count is deterministically zeroed.
        let draw_over_in: [[u32; 4]; 1] = [[1 << 24, 5, 3, 4]];
        let (draw_out, draw_over_out) = run_clamp_pair(host, 0, 16, &draw_in, &draw_over_in);
        assert_eq!(draw_out[0], [100, 4, 1, 2]);
        assert_eq!(draw_out[1], [1 << 24, 4, 7, 9]);
        assert_eq!(draw_over_out[0][0], 0);
        assert_eq!(draw_over_out[0][2..], [3, 4]);

        // Dispatch (kind = 2, stride = 16):
        // 0: normal dispatch stays untouched.
        // 1: per-dimension limit clamps X to 65535 (65535*2*2 <= 1<<20).
        // 2: product exceeding GROUPS (1<<20) is zeroed.
        // 3: 3D overflow guard zeroes (65535, 65535, 65535).
        let disp_in: [[u32; 4]; 4] = [
            [10, 2, 3, 0],
            [70_000, 2, 2, 0],
            [2048, 1024, 1, 0],
            [65_535, 65_535, 65_535, 0],
        ];
        let disp_out = run_clamp(host, 2, 16, &disp_in);
        assert_eq!(disp_out[0], [10, 2, 3, 0]);
        assert_eq!(disp_out[1], [65_535, 2, 2, 0]);
        assert_eq!(disp_out[2], [0, 0, 0, 0]);
        assert_eq!(disp_out[3], [0, 0, 0, 0]);

        // Mesh dispatch (kind = 3, stride = 16): clamped to (65535, 1, 1).
        let mesh_in: [[u32; 4]; 1] = [[70_000, 4, 8, 5]];
        let mesh_out = run_clamp(host, 3, 16, &mesh_in);
        assert_eq!(mesh_out[0], [65_535, 1, 1, 5]);
    }

    fn run_clamp(
        host: &crate::host::Host,
        kind: u32,
        stride: u32,
        records: &[[u32; 4]],
    ) -> Vec<[u32; 4]> {
        let list = host.new_list().unwrap();
        let (stage, readback) = record_clamp(host, &list, kind, stride, records);
        list.close();
        host.execute(&[&list]);
        let out = read_back(&readback, records.len());
        drop(stage);
        out
    }

    fn run_clamp_pair(
        host: &crate::host::Host,
        kind: u32,
        stride: u32,
        first: &[[u32; 4]],
        second: &[[u32; 4]],
    ) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
        let list = host.new_list().unwrap();
        let (stage1, rb1) = record_clamp(host, &list, kind, stride, first);
        let (stage2, rb2) = record_clamp(host, &list, kind, stride, second);
        list.close();
        host.execute(&[&list]);
        let out1 = read_back(&rb1, first.len());
        let out2 = read_back(&rb2, second.len());
        drop((stage1, stage2));
        (out1, out2)
    }

    fn record_clamp(
        host: &crate::host::Host,
        list: &crate::host::HostList,
        kind: u32,
        stride: u32,
        records: &[[u32; 4]],
    ) -> ((ID3D12Resource, ID3D12Resource), ID3D12Resource) {
        let bytes = std::mem::size_of_val(records) as u64;
        let upload = committed_buffer(
            &host.device,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            bytes,
        );
        let uav = committed_buffer(
            &host.device,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
            bytes,
        );
        let readback = committed_buffer(
            &host.device,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
            bytes,
        );

        let mut mapped = std::ptr::null_mut();
        // SAFETY: upload buffer is `bytes` bytes on an upload heap.
        unsafe {
            upload.Map(0, None, Some(&mut mapped)).unwrap();
            std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), mapped.cast(), bytes as usize);
            upload.Unmap(0, None);
        }

        // SAFETY: valid open list and live buffers sized `bytes`.
        let gpu = unsafe {
            list.list.CopyBufferRegion(&uav, 0, &upload, 0, bytes);
            list.list.ResourceBarrier(&[super::transition(
                &uav,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            uav.GetGPUVirtualAddress()
        };
        assert!(super::bound(list, &uav, gpu, (kind, stride, records.len() as u32)));
        super::done(list, &uav);
        // SAFETY: `done` leaves `uav` in UNORDERED_ACCESS; copy back to readback.
        unsafe {
            list.list.ResourceBarrier(&[super::transition(
                &uav,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyBufferRegion(&readback, 0, &uav, 0, bytes);
        }
        ((upload, uav), readback)
    }

    fn read_back(readback: &ID3D12Resource, count: usize) -> Vec<[u32; 4]> {
        let mut out = vec![[0u32; 4]; count];
        let bytes = std::mem::size_of_val(out.as_slice());
        let mut back = std::ptr::null_mut();
        // SAFETY: `host.execute` waits for completion; readback buffer is `bytes` bytes.
        unsafe {
            readback.Map(0, None, Some(&mut back)).unwrap();
            std::ptr::copy_nonoverlapping(back.cast::<u8>(), out.as_mut_ptr().cast(), bytes);
            readback.Unmap(0, None);
        }
        out
    }

    fn committed_buffer(
        device: &ID3D12Device,
        heap: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
        bytes: u64,
    ) -> ID3D12Resource {
        let heap_props = D3D12_HEAP_PROPERTIES {
            Type: heap,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut res = None;
        // SAFETY: valid device, heap properties, and buffer resource descriptor.
        unsafe {
            device
                .CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut res)
                .unwrap();
        }
        res.unwrap()
    }
}
