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

/// Rectangle-list draws, rewritten as the triangle-list draws of their
/// expanded vertices (`draw::State::rectangles`: six per rectangle, from
/// the first vertex of the bound buffers on) into records of our own, so
/// the title's records stay as they are. A record must start on a
/// rectangle and stay within the `vertices` expanded; others draw nothing.
const RECTANGLES: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint stride; uint records; uint vertices; uint unused; };
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer triangles : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= records) return;
    // Vertex count, instances, first vertex, first instance.
    uint4 a = args.Load4(id.x * stride);
    uint rects = a.x / 3;
    bool whole = a.z % 3 == 0 && a.z <= vertices && rects * 3 <= vertices - a.z;
    triangles.Store4(id.x * 16, whole ? uint4(rects * 6, a.y, a.z / 3 * 6, a.w) : uint4(0, 0, 0, 0));
}
"#;

/// One of our compute passes: its root signature (inside the shader) and
/// pipeline.
struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn rectangle_pass() -> Option<&'static Pass> {
    static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    PASS.get_or_init(|| match make(RECTANGLES) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no rectangle pass: indirect rectangles dropped");
            None
        }
    })
    .as_ref()
}

fn clamp() -> Option<&'static Pass> {
    static CLAMP: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make(SOURCE) {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make(source: &str) -> windows::core::Result<Pass> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("victory"),
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
            tracing::warn!("our compute pass: {}", String::from_utf8_lossy(text));
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
        Ok(Pass { root, pso })
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

/// Records, in `list`, the rewrite of `records` rectangle-list draw records
/// `stride` bytes apart at host address `gpu` (in unordered access, as
/// [`done`] leaves them) into triangle-list records over the first
/// `vertices` vertices expanded (see [`RECTANGLES`]): 16-byte records in
/// the list's argument memory, left ready to be read as arguments ([`done`]
/// after the execution). Replaces the list's compute root signature and
/// pipeline. None if the pass or the memory is not available.
pub(crate) fn rectangles(
    list: &HostList,
    gpu: u64,
    (stride, records, vertices): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let p = rectangle_pass()?;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) || stride < 16 {
        return None;
    }
    let (out, offset) = list.argument_space(u64::from(records) * 16)?;
    let l = &list.list;
    let constants = [stride, records, vertices, 0];
    // SAFETY: an open list; our root signature, pipeline, the caller's live
    // buffer and the list's own argument memory.
    unsafe {
        let to = out.GetGPUVirtualAddress() + offset;
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&p.root);
        l.SetPipelineState(&p.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootUnorderedAccessView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, to);
        l.Dispatch(records.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                &out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    Some((out, offset))
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
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Rectangle records become records of their expanded triangles: six
    /// vertices per rectangle, from the rectangle's own start. Records that
    /// start inside a rectangle, or reach past the expanded vertices, draw
    /// nothing; the title's records are left as they were.
    #[test]
    fn rectangle_records_become_triangle_records() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        // 20-byte records (4 bytes past the draw), 9 expanded vertices.
        let records: [[u32; 5]; 4] = [
            [6, 2, 0, 1, 0xdead],
            [7, 1, 3, 0, 0],
            [3, 1, 1, 0, 0],
            [3, 1, 9, 0, 0],
        ];
        let (input, data) = host.shared_buffer(80).unwrap();
        // SAFETY: the buffer's mapping holds 80 bytes.
        unsafe { data.cast::<[[u32; 5]; 4]>().write(records) };
        let list = host.new_list().unwrap();
        // SAFETY: a live buffer.
        let gpu = unsafe { input.GetGPUVirtualAddress() };
        let (out, offset) = super::rectangles(&list, gpu, (20, 4, 9)).unwrap();
        list.close();
        host.execute(&[&list]);
        let read = |buffer: &super::ID3D12Resource, at: u64| {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped CPU-visible buffer holding the records read.
            unsafe {
                buffer.Map(0, None, Some(&mut p)).unwrap();
                let v = p
                    .cast::<u8>()
                    .add(at as usize)
                    .cast::<[[u32; 4]; 4]>()
                    .read_unaligned();
                buffer.Unmap(0, None);
                v
            }
        };
        assert_eq!(
            read(&out, offset),
            [[12, 2, 0, 1], [12, 1, 6, 0], [0; 4], [0; 4]]
        );
        // SAFETY: as above.
        assert_eq!(unsafe { data.cast::<[[u32; 5]; 4]>().read() }, records);
    }
}
