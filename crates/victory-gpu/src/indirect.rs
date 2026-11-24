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
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    uint at = id.x * stride;
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
    uint before;
    budget.InterlockedAdd(0, cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        args.Store(at, 0);
    }
}
"#;

// Each lane converts one record. The first 32 bytes of the output
// contain the draw arguments and allocator cursor, then [base, x, y, z].
const MESH_PREPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1), UAV(u2), UAV(u3)"
cbuffer Info : register(b0) { uint stride, records, has_count, cap; };
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer count_buffer : register(u1);
RWByteAddressBuffer output : register(u2);
RWByteAddressBuffer budget : register(u3);
static const uint BUDGET = 1u << 27;
[RootSignature(RS)]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint r = tid.x;
    if (r >= records) return;
    uint4 rec = uint4(0, 0, 1, 1);
    uint count = has_count ? min(records, count_buffer.Load(0)) : records;
    if (r < count) {
        uint3 g = args.Load3(r * stride);
        uint n = g.y != 0 && g.z != 0 ? min(g.x, 65535u) : 0;
        uint cost = n * 128;
        // Saturate before adding: untrusted record totals cannot wrap the
        // shared budget back to zero.
        uint before = budget.Load(0);
        while (before < BUDGET && cost != 0) {
            uint next = min(BUDGET, before + cost);
            uint seen;
            budget.InterlockedCompareExchange(0, before, next, seen);
            if (seen == before) {
                if (cost <= BUDGET - before) {
                    uint base;
                    output.InterlockedAdd(16, n, base);
                    if (base < cap) {
                        n = min(n, cap - base);
                        output.InterlockedMax(4, base + n);
                        rec.xy = uint2(base, n);
                    }
                }
                break;
            }
            before = seen;
        }
    }
    output.Store4(32 + 16 * r, rec);
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
        .get_or_init(|| match make(SOURCE, s!("clamp")) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str, label: windows::core::PCSTR) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            label,
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
        l.Dispatch(records.div_ceil(64), 1, 1);
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

pub(crate) fn prepare_mesh(
    list: &HostList,
    args: u64,
    count: Option<u64>,
    stride: u32,
    records: u32,
    output: &ID3D12Resource,
    cap: u32,
) -> bool {
    if stride < 12
        || !stride.is_multiple_of(4)
        || !args.is_multiple_of(4)
        || count.is_some_and(|c| !c.is_multiple_of(4))
        || records == 0
    {
        return false;
    }
    static PREP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    let Some(prep) = PREP
        .get_or_init(|| {
            make(MESH_PREPARE, s!("mesh_prepare"))
                .inspect_err(|e| tracing::warn!(error = %e, "mesh prepare PSO refused"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    let Some(budget) = list.budget() else {
        return false;
    };
    let constants = [stride, records, u32::from(count.is_some()), cap];
    let l = &list.list;
    // SAFETY: arguments and scratch are live UAV buffers held by the list;
    // host scratch sizing limits the output and dispatch dimensions.
    unsafe {
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&prep.root);
        l.SetPipelineState(&prep.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, args);
        l.SetComputeRootUnorderedAccessView(2, count.unwrap_or(args));
        l.SetComputeRootUnorderedAccessView(3, output.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    true
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

    #[test]
    fn mesh_prepare_respects_gpu_count_capacity_and_leaves_input_intact() {
        use super::*;
        let Some(h) = host() else { return };
        let list = h.new_list().unwrap();
        let input = [[2u32, 1, 1, 99], [2, 1, 1, 77], [8, 1, 1, 55]];
        let (args, ap) = h.shared_buffer(64).unwrap();
        let (count, cp) = h.shared_buffer(16).unwrap();
        let (out, op) = h.shared_buffer(96).unwrap();
        unsafe {
            ap.cast::<[[u32; 4]; 3]>().write_unaligned(input);
            cp.cast::<u32>().write_unaligned(2);
            op.cast::<[u32; 8]>().write_unaligned([3, 0, 0, 0, 0, 0, 0, 0]);
            list.list.ResourceBarrier(&[transition(&out, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
        }
        assert!(prepare_mesh(&list, unsafe { args.GetGPUVirtualAddress() },
            Some(unsafe { count.GetGPUVirtualAddress() }), 16, 3, &out, 3));
        done(&list, &out);
        list.close();
        h.execute(&[&list]);
        // Atomic reservation order is unspecified; check the occupied range.
        let words = unsafe { op.cast::<[u32; 20]>().read_unaligned() };
        assert_eq!(&words[..5], &[3, 3, 0, 0, 4]);
        let mut ranges = [(words[8], words[9]), (words[12], words[13])];
        ranges.sort();
        assert_eq!(ranges, [(0, 2), (2, 1)]);
        assert_eq!(words[17], 0, "the count buffer excludes record 2");
        assert_eq!(unsafe { ap.cast::<[[u32; 4]; 3]>().read_unaligned() }, input);
    }
}
