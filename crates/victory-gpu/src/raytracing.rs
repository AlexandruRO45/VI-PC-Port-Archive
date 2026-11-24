//! Native DXR 1.1 acceleration structure support (`ID3D12Device5` and
//! `ID3D12GraphicsCommandList4`).
//!
//! Console titles place bottom-level (BLAS) and top-level (TLAS) acceleration
//! structures in unified title memory and reference them by title GPU virtual
//! address in build descriptors, TLAS instance records, copies, and shader
//! resource views. Host D3D12 requires acceleration structures to live in
//! `D3D12_HEAP_TYPE_DEFAULT` buffers in
//! `D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE`, and build scratch
//! buffers to live in `DEFAULT` UAV buffers.
//!
//! This module maps title acceleration structure addresses to host `DEFAULT`
//! buffers, translates BLAS geometry and TLAS instance buffers through
//! [`crate::shadow`] / scratch upload memory, and records native DXR 1.1
//! prebuild queries, builds, and copies when the host supports at least
//! `D3D12_RAYTRACING_TIER_1_0` (falling back cleanly on WARP or non-RT hosts).

use std::collections::BTreeMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::host::{HostList, host};
use crate::object::This;

/// Console/GDK `D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO` layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrebuildInfo {
    pub result_data_max_size_in_bytes: u64,
    pub scratch_data_size_in_bytes: u64,
    pub update_scratch_data_size_in_bytes: u64,
}
const _: () = assert!(
    size_of::<PrebuildInfo>() == size_of::<D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO>()
);

/// Host acceleration structure buffer tracked by title GPU virtual address.
struct AsEntry {
    resource: ID3D12Resource,
    host_va: u64,
    size: u64,
}

static AS_BUFFERS: Mutex<BTreeMap<u64, AsEntry>> = Mutex::new(BTreeMap::new());

/// Looks up the host resource and host GPU virtual address for an acceleration
/// structure built at `title_va`.
pub fn lookup_as(title_va: u64) -> Option<(ID3D12Resource, u64)> {
    if title_va == 0 {
        return None;
    }
    let map = AS_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    map.get(&title_va)
        .map(|entry| (entry.resource.clone(), entry.host_va))
}

/// Returns the host GPU virtual address of a built acceleration structure, if
/// present.
pub fn host_acceleration_structure_va(title_va: u64) -> Option<u64> {
    lookup_as(title_va).map(|(_, va)| va)
}

/// Evicts host acceleration structure buffers whose title address range
/// overlaps `[start, end)`.
pub(crate) fn evict_as_range(start: u64, end: u64) {
    if start >= end {
        return;
    }
    let mut map = AS_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    map.retain(|&addr, entry| {
        let entry_end = addr.saturating_add(entry.size.max(1));
        !(addr < end && entry_end > start)
    });
}

fn resolve_or_alloc_as(title_va: u64, required_size: u64) -> Option<(ID3D12Resource, u64)> {
    if title_va == 0 {
        return None;
    }
    let aligned = required_size.max(256).next_multiple_of(256);
    {
        let map = AS_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = map.get(&title_va)
            && entry.size >= aligned
        {
            return Some((entry.resource.clone(), entry.host_va));
        }
    }
    let host = host()?;
    let resource = host.default_uav_buffer(
        aligned,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
    )?;
    crate::host::name(&resource, || format!("dxr as {title_va:#x}+{aligned:#x}"));
    // SAFETY: `resource` is a live buffer.
    let host_va = unsafe { resource.GetGPUVirtualAddress() };
    let mut map = AS_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    map.insert(
        title_va,
        AsEntry {
            resource: resource.clone(),
            host_va,
            size: aligned,
        },
    );
    Some((resource, host_va))
}

fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

/// Upper bound on geometries accepted in a single BLAS build/prebuild query.
/// Rejects corrupt `NumDescs` values before allocating host staging buffers.
const MAX_GEOMETRY_DESCS: usize = 1 << 20;

/// Reads the `NumDescs` geometry descriptions from `inputs` (supporting both
/// contiguous arrays and arrays of pointers), validating `NumDescs`, address
/// overflow, and committed/readable backing memory via `ReadProcessMemory`
/// before allocating or dereferencing caller pointers.
fn read_geometries(
    inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
) -> Option<Vec<D3D12_RAYTRACING_GEOMETRY_DESC>> {
    let count = inputs.NumDescs as usize;
    if count == 0 {
        return Some(Vec::new());
    }
    if count > MAX_GEOMETRY_DESCS {
        return None;
    }
    const GEOM_BYTES: usize = size_of::<D3D12_RAYTRACING_GEOMETRY_DESC>();
    match inputs.DescsLayout {
        D3D12_ELEMENTS_LAYOUT_ARRAY => {
            // SAFETY: union member selected by `DescsLayout == ARRAY`.
            let ptr = unsafe { inputs.Anonymous.pGeometryDescs };
            let addr = ptr as usize;
            let bytes = count.checked_mul(GEOM_BYTES)?;
            if ptr.is_null()
                || addr.checked_add(bytes).is_none()
                || !crate::texture::committed(ptr.cast::<u8>(), bytes)
            {
                return None;
            }
            let mut out = vec![D3D12_RAYTRACING_GEOMETRY_DESC::default(); count];
            // SAFETY: `out` is a contiguous writable buffer of `bytes` bytes.
            let dst =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), bytes) };
            crate::host::read_into(ptr.cast::<u8>(), dst).then_some(out)
        }
        D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS => {
            // SAFETY: union member selected by `DescsLayout == ARRAY_OF_POINTERS`.
            let ptrs = unsafe { inputs.Anonymous.ppGeometryDescs };
            let ptrs_addr = ptrs as usize;
            let ptr_bytes = count.checked_mul(size_of::<usize>())?;
            if ptrs.is_null()
                || ptrs_addr.checked_add(ptr_bytes).is_none()
                || !crate::texture::committed(ptrs.cast::<u8>(), ptr_bytes)
            {
                return None;
            }
            let mut raw_ptrs = vec![0usize; count];
            // SAFETY: `raw_ptrs` is a contiguous writable buffer of `ptr_bytes` bytes.
            let ptr_dst = unsafe {
                std::slice::from_raw_parts_mut(raw_ptrs.as_mut_ptr().cast::<u8>(), ptr_bytes)
            };
            if !crate::host::read_into(ptrs.cast::<u8>(), ptr_dst) {
                return None;
            }
            let mut out = Vec::with_capacity(count);
            for p in raw_ptrs {
                if p == 0 || p.checked_add(GEOM_BYTES).is_none() {
                    return None;
                }
                let mut desc = D3D12_RAYTRACING_GEOMETRY_DESC::default();
                // SAFETY: `desc` is a writable `GEOM_BYTES`-byte local buffer.
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(
                        std::ptr::from_mut(&mut desc).cast::<u8>(),
                        GEOM_BYTES,
                    )
                };
                if !crate::host::read_into(p as *const u8, dst) {
                    return None;
                }
                out.push(desc);
            }
            Some(out)
        }
        _ => None,
    }
}

/// Translates console vertex formats in BLAS geometries for host prebuild
/// queries (without resolving GPU VAs).
fn sanitize_geometries_for_prebuild(geoms: &mut [D3D12_RAYTRACING_GEOMETRY_DESC]) {
    for g in geoms {
        if g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            // SAFETY: union member selected by `g.Type == TRIANGLES`.
            let tri = unsafe { &mut g.Anonymous.Triangles };
            tri.VertexFormat =
                DXGI_FORMAT(crate::resource::pc_format(tri.VertexFormat.0 as u32) as i32);
        }
    }
}

/// Translates BLAS geometry buffer VAs from title memory to host `Shadow` GPU
/// VAs and schedules refreshes on `list`.
fn translate_geometries_for_build(
    list: &HostList,
    geoms: &mut [D3D12_RAYTRACING_GEOMETRY_DESC],
) -> bool {
    for g in geoms {
        match g.Type {
            D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES => {
                // SAFETY: union member selected by `g.Type == TRIANGLES`.
                let tri = unsafe { &mut g.Anonymous.Triangles };
                tri.VertexFormat =
                    DXGI_FORMAT(crate::resource::pc_format(tri.VertexFormat.0 as u32) as i32);
                if tri.Transform3x4 != 0 {
                    let Some(shadow) = crate::shadow::find(tri.Transform3x4, 48) else {
                        return false;
                    };
                    shadow.refresh(list, tri.Transform3x4, 48);
                    tri.Transform3x4 = shadow.gpu(tri.Transform3x4);
                }
                if tri.IndexCount > 0 && tri.IndexBuffer != 0 {
                    let elem = match tri.IndexFormat {
                        DXGI_FORMAT_R16_UINT => 2u64,
                        DXGI_FORMAT_R32_UINT => 4u64,
                        _ => return false,
                    };
                    let Some(len) = u64::from(tri.IndexCount).checked_mul(elem) else {
                        return false;
                    };
                    let Some(shadow) = crate::shadow::find(tri.IndexBuffer, len) else {
                        return false;
                    };
                    shadow.refresh(list, tri.IndexBuffer, len);
                    tri.IndexBuffer = shadow.gpu(tri.IndexBuffer);
                }
                if tri.VertexCount > 0 && tri.VertexBuffer.StartAddress != 0 {
                    let elem =
                        crate::resource::element(tri.VertexFormat.0 as u32).map_or(12, |(b, _)| b);
                    let stride = tri.VertexBuffer.StrideInBytes.max(elem);
                    let Some(len) = u64::from(tri.VertexCount).checked_mul(stride) else {
                        return false;
                    };
                    let addr = tri.VertexBuffer.StartAddress;
                    let Some(shadow) = crate::shadow::find(addr, len) else {
                        return false;
                    };
                    shadow.refresh(list, addr, len);
                    tri.VertexBuffer.StartAddress = shadow.gpu(addr);
                }
            }
            D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS => {
                // SAFETY: union member selected by `g.Type == PROCEDURAL_PRIMITIVE_AABBS`.
                let aabbs = unsafe { &mut g.Anonymous.AABBs };
                if aabbs.AABBCount > 0 && aabbs.AABBs.StartAddress != 0 {
                    let stride = aabbs.AABBs.StrideInBytes.max(24);
                    let Some(len) = aabbs.AABBCount.checked_mul(stride) else {
                        return false;
                    };
                    let addr = aabbs.AABBs.StartAddress;
                    let Some(shadow) = crate::shadow::find(addr, len) else {
                        return false;
                    };
                    shadow.refresh(list, addr, len);
                    aabbs.AABBs.StartAddress = shadow.gpu(addr);
                }
            }
            _ => return false,
        }
    }
    true
}

/// Translates TLAS instance descriptors into list scratch upload memory,
/// rewriting each instance's title BLAS address to its host BLAS GPU VA at
/// submission time.
fn translate_tlas_instances(
    list: &HostList,
    inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
) -> Option<u64> {
    let count = inputs.NumDescs as usize;
    if count == 0 {
        return Some(0);
    }
    // SAFETY: union member selected for `TOP_LEVEL`.
    let src_va = unsafe { inputs.Anonymous.InstanceDescs };
    if src_va == 0 {
        return None;
    }
    const INSTANCE_BYTES: usize = size_of::<D3D12_RAYTRACING_INSTANCE_DESC>();
    const _: () = assert!(INSTANCE_BYTES == 64);
    let total_bytes = (count as u64).checked_mul(INSTANCE_BYTES as u64)?;
    let (gpu, to) = list.scratch(total_bytes)?;
    let to_addr = to as usize;
    let array_of_ptrs = inputs.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS;
    list.defer_host(move || {
        // SAFETY: `to_addr` holds `count` writable `D3D12_RAYTRACING_INSTANCE_DESC` slots
        // in this list's scratch upload memory.
        let dst = unsafe {
            std::slice::from_raw_parts_mut(to_addr as *mut D3D12_RAYTRACING_INSTANCE_DESC, count)
        };
        if array_of_ptrs {
            let mut ptrs = vec![0u64; count];
            crate::host::copy_mapped(
                ptrs.as_mut_ptr().cast::<u8>(),
                src_va as *const u8,
                count * size_of::<u64>(),
                false,
            );
            for (slot, ptr) in dst.iter_mut().zip(ptrs) {
                *slot = D3D12_RAYTRACING_INSTANCE_DESC::default();
                if ptr != 0 {
                    crate::host::copy_mapped(
                        std::ptr::from_mut(slot).cast::<u8>(),
                        ptr as *const u8,
                        INSTANCE_BYTES,
                        false,
                    );
                }
            }
        } else {
            crate::host::copy_mapped(
                dst.as_mut_ptr().cast::<u8>(),
                src_va as *const u8,
                count * INSTANCE_BYTES,
                false,
            );
        }
        for inst in dst {
            inst.AccelerationStructure =
                host_acceleration_structure_va(inst.AccelerationStructure).unwrap_or(0);
        }
    });
    Some(gpu)
}

/// Queries host DXR 1.1 acceleration structure prebuild sizes via
/// `ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo`.
/// Returns `None` when the host lacks DXR support or the inputs are invalid.
pub fn prebuild_info(
    inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS,
) -> Option<PrebuildInfo> {
    let device5 = host()?.device5()?;
    let mut host_inputs = *inputs;
    let _geoms = if inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL {
        let mut g = read_geometries(inputs)?;
        sanitize_geometries_for_prebuild(&mut g);
        host_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        host_inputs.Anonymous.pGeometryDescs = if g.is_empty() {
            std::ptr::null()
        } else {
            g.as_ptr()
        };
        g
    } else if inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL {
        Vec::new()
    } else {
        return None;
    };
    let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
    // SAFETY: `host_inputs` and `geoms` stay alive for the duration of the call.
    unsafe { device5.GetRaytracingAccelerationStructurePrebuildInfo(&host_inputs, &mut info) };
    Some(PrebuildInfo {
        result_data_max_size_in_bytes: info.ResultDataMaxSizeInBytes,
        scratch_data_size_in_bytes: info.ScratchDataSizeInBytes,
        update_scratch_data_size_in_bytes: info.UpdateScratchDataSizeInBytes,
    })
}

/// Records a native DXR 1.1 `BuildRaytracingAccelerationStructure` onto `list`
/// (translating title GPU VAs to host `Shadow` / `DEFAULT` AS / scratch
/// buffers) followed by a UAV barrier on the destination acceleration structure.
pub(crate) fn build_acceleration_structure(
    list: &HostList,
    desc: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    postbuild: &[D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC],
) -> bool {
    let Some(host) = host() else {
        return false;
    };
    if host.raytracing_tier() < D3D12_RAYTRACING_TIER_1_0.0 as u32 {
        return false;
    }
    let Ok(list4) = list.list.cast::<ID3D12GraphicsCommandList4>() else {
        return false;
    };
    let Some(info) = prebuild_info(&desc.Inputs) else {
        return false;
    };
    let is_update = (desc.Inputs.Flags.0
        & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE.0)
        != 0;
    let scratch_bytes = if is_update {
        info.update_scratch_data_size_in_bytes
    } else {
        info.scratch_data_size_in_bytes
    };
    let Some((dest_res, dest_va)) = resolve_or_alloc_as(
        desc.DestAccelerationStructureData,
        info.result_data_max_size_in_bytes,
    ) else {
        return false;
    };
    let Some(scratch_res) = host.default_uav_buffer(scratch_bytes, D3D12_RESOURCE_STATE_COMMON)
    else {
        return false;
    };
    crate::host::name(&scratch_res, || format!("dxr scratch {scratch_bytes:#x}"));
    // SAFETY: `scratch_res` is a live buffer.
    let scratch_va = unsafe { scratch_res.GetGPUVirtualAddress() };

    let source_va = if desc.SourceAccelerationStructureData != 0 {
        let Some((src_res, va)) = lookup_as(desc.SourceAccelerationStructureData) else {
            return false;
        };
        list.keep(&src_res);
        va
    } else {
        0
    };

    let mut host_inputs = desc.Inputs;
    let _geoms = match desc.Inputs.Type {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL => {
            let Some(mut g) = read_geometries(&desc.Inputs) else {
                return false;
            };
            if !translate_geometries_for_build(list, &mut g) {
                return false;
            }
            host_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            host_inputs.Anonymous.pGeometryDescs = if g.is_empty() {
                std::ptr::null()
            } else {
                g.as_ptr()
            };
            g
        }
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL => {
            let Some(instances_va) = translate_tlas_instances(list, &desc.Inputs) else {
                return false;
            };
            host_inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            host_inputs.Anonymous.InstanceDescs = instances_va;
            Vec::new()
        }
        _ => return false,
    };

    let mut host_postbuild = Vec::with_capacity(postbuild.len());
    for p in postbuild {
        let bytes = match p.InfoType {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_SERIALIZATION => 16u64,
            _ => 8u64,
        };
        let Some(shadow) = crate::shadow::find(p.DestBuffer, bytes) else {
            return false;
        };
        shadow.refresh(list, p.DestBuffer, bytes);
        shadow.gpu_writes(list, p.DestBuffer, bytes);
        host_postbuild.push(
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: shadow.gpu(p.DestBuffer),
                InfoType: p.InfoType,
            },
        );
    }

    let host_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
        DestAccelerationStructureData: dest_va,
        Inputs: host_inputs,
        SourceAccelerationStructureData: source_va,
        ScratchAccelerationStructureData: scratch_va,
    };
    list.keep(&dest_res);
    list.keep(&scratch_res);
    // SAFETY: `list4` is an open list; all referenced resources are kept alive
    // until list reset, and `geoms`/`host_postbuild` outlive the call.
    unsafe {
        list4.BuildRaytracingAccelerationStructure(&host_desc, Some(&host_postbuild));
        list.list.ResourceBarrier(&[uav_barrier(&dest_res)]);
    }
    true
}

/// Records a batch of acceleration structure builds on `list`.
pub(crate) fn build_acceleration_structures_batched(
    list: &HostList,
    descs: &[D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC],
) -> bool {
    descs
        .iter()
        .all(|desc| build_acceleration_structure(list, desc, &[]))
}

/// Records a native DXR 1.1 `CopyRaytracingAccelerationStructure` onto `list`
/// followed by a UAV barrier on the destination acceleration structure.
pub(crate) fn copy_acceleration_structure(
    list: &HostList,
    dest_title_va: u64,
    src_title_va: u64,
    mode: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE,
) -> bool {
    let Some(host) = host() else {
        return false;
    };
    if host.raytracing_tier() < D3D12_RAYTRACING_TIER_1_0.0 as u32 {
        return false;
    }
    let Ok(list4) = list.list.cast::<ID3D12GraphicsCommandList4>() else {
        return false;
    };
    let Some((src_res, src_va, src_size)) = ({
        let map = AS_BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        map.get(&src_title_va)
            .map(|e| (e.resource.clone(), e.host_va, e.size))
    }) else {
        return false;
    };
    let Some((dest_res, dest_va)) = resolve_or_alloc_as(dest_title_va, src_size) else {
        return false;
    };
    list.keep(&src_res);
    list.keep(&dest_res);
    // SAFETY: `list4` is an open list; both AS resources are kept alive until reset.
    unsafe {
        list4.CopyRaytracingAccelerationStructure(dest_va, src_va, mode);
        list.list.ResourceBarrier(&[uav_barrier(&dest_res)]);
    }
    true
}

/// Public entry point recording an acceleration structure build on a title
/// command list (`ID3D12GraphicsCommandList4`).
pub fn build_on_list(
    title_list: This,
    desc: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC,
    postbuild: &[D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC],
) -> bool {
    let Some(list) = crate::list::host_list(title_list) else {
        return false;
    };
    build_acceleration_structure(list, desc, postbuild)
}

/// Public entry point recording a batch of acceleration structure builds on a
/// title command list.
pub fn build_batched_on_list(
    title_list: This,
    descs: &[D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC],
) -> bool {
    let Some(list) = crate::list::host_list(title_list) else {
        return false;
    };
    build_acceleration_structures_batched(list, descs)
}

/// Public entry point recording an acceleration structure copy on a title
/// command list.
pub fn copy_on_list(
    title_list: This,
    dest_title_va: u64,
    src_title_va: u64,
    mode: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE,
) -> bool {
    let Some(list) = crate::list::host_list(title_list) else {
        return false;
    };
    copy_acceleration_structure(list, dest_title_va, src_title_va, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_geometry(vb_va: u64, ib_va: u64) -> D3D12_RAYTRACING_GEOMETRY_DESC {
        D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: 0,
                    IndexFormat: if ib_va != 0 {
                        DXGI_FORMAT_R16_UINT
                    } else {
                        DXGI_FORMAT_UNKNOWN
                    },
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: if ib_va != 0 { 3 } else { 0 },
                    VertexCount: 3,
                    IndexBuffer: ib_va,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: vb_va,
                        StrideInBytes: 12,
                    },
                },
            },
        }
    }

    #[test]
    fn options5_query_reports_host_raytracing_tier() {
        use crate::object::{S_OK, slot};
        let device = crate::device::device();
        type Check = extern "system" fn(This, u32, *mut std::ffi::c_void, u32) -> i32;
        let check = slot::<Check>(device, 0x68);
        let mut out = [0u32; 3];
        assert_eq!(check(device, 27, out.as_mut_ptr().cast(), 12), S_OK);
        let expected_tier = host().map_or(0, |h| h.raytracing_tier());
        assert_eq!(out[2], expected_tier);
        // Wrong buffer size must return E_INVALIDARG.
        assert_ne!(check(device, 27, out.as_mut_ptr().cast(), 8), S_OK);
    }

    #[test]
    fn prebuild_info_sizes_blas_and_tlas_on_dxr_host() {
        let Some(host) = host() else {
            return;
        };
        if host.raytracing_tier() < D3D12_RAYTRACING_TIER_1_0.0 as u32 {
            return; // skip cleanly on WARP / non-DXR adapters
        }
        let geom = triangle_geometry(0, 0);
        let blas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &geom,
            },
        };
        let blas_info = prebuild_info(&blas_inputs).expect("BLAS prebuild info");
        assert!(blas_info.result_data_max_size_in_bytes > 0);
        assert!(blas_info.scratch_data_size_in_bytes > 0);

        // Also test ARRAY_OF_POINTERS layout.
        let geom_ptr = &raw const geom;
        let ptr_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                ppGeometryDescs: &geom_ptr,
            },
            ..blas_inputs
        };
        assert_eq!(prebuild_info(&ptr_inputs), Some(blas_info));

        let tlas_inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: 0 },
        };
        let tlas_info = prebuild_info(&tlas_inputs).expect("TLAS prebuild info");
        assert!(tlas_info.result_data_max_size_in_bytes > 0);
        assert!(tlas_info.scratch_data_size_in_bytes > 0);
    }

    #[test]
    fn build_and_copy_blas_and_tlas_on_dxr_host() {
        let Some(host) = host() else {
            return;
        };
        if host.raytracing_tier() < D3D12_RAYTRACING_TIER_1_0.0 as u32 {
            return; // skip cleanly on WARP / non-DXR adapters
        }
        // Title memory holding vertices, indices, instance desc, postbuild info,
        // and virtual addresses for BLAS, cloned BLAS, and TLAS.
        let mem = crate::resource::Memory::zeroed(256 << 10, 64 << 10).unwrap();
        let base = mem.address();
        let vb_va = base;
        let ib_va = base + 0x100;
        let inst_va = base + 0x200;
        let postbuild_va = base + 0x300;
        let blas_title_va = base + (64 << 10);
        let clone_title_va = base + (128 << 10);
        let tlas_title_va = base + (192 << 10);

        let verts: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]];
        let indices: [u16; 3] = [0, 1, 2];
        let instance = D3D12_RAYTRACING_INSTANCE_DESC {
            Transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            _bitfield1: 0xFF << 24, // InstanceID=0, InstanceMask=0xFF
            _bitfield2: 0,
            AccelerationStructure: clone_title_va,
        };
        // SAFETY: offsets lie within the 256 KiB zeroed allocation `mem`.
        unsafe {
            (vb_va as *mut [[f32; 3]; 3]).write(verts);
            (ib_va as *mut [u16; 3]).write(indices);
            (inst_va as *mut D3D12_RAYTRACING_INSTANCE_DESC).write(instance);
            (postbuild_va as *mut u64).write(0);
        }

        let geom = triangle_geometry(vb_va, ib_va);
        let blas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: blas_title_va,
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
                    | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &geom,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: 0,
        };
        let postbuild = [
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC {
                DestBuffer: postbuild_va,
                InfoType: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_CURRENT_SIZE,
            },
        ];
        let tlas_desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            DestAccelerationStructureData: tlas_title_va,
            Inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: inst_va,
                },
            },
            SourceAccelerationStructureData: 0,
            ScratchAccelerationStructureData: 0,
        };

        let list = host.new_list().unwrap();
        assert!(build_acceleration_structure(&list, &blas_desc, &postbuild));
        assert!(copy_acceleration_structure(
            &list,
            clone_title_va,
            blas_title_va,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        ));
        assert!(build_acceleration_structures_batched(&list, &[tlas_desc]));
        list.close();
        host.execute(&[&list]);
        list.reset();

        assert!(host_acceleration_structure_va(blas_title_va).is_some());
        assert!(host_acceleration_structure_va(clone_title_va).is_some());
        assert!(host_acceleration_structure_va(tlas_title_va).is_some());
        // SAFETY: `host.execute` copied GPU postbuild info back into `postbuild_va`.
        let current_size = unsafe { (postbuild_va as *const u64).read_volatile() };
        assert!(
            current_size > 0,
            "DXR postbuild CURRENT_SIZE must be written back to title memory"
        );

        // Remapping title memory must evict overlapping host AS entries.
        crate::watch::title_memory_remapped(blas_title_va as usize, 64 << 10);
        assert!(host_acceleration_structure_va(blas_title_va).is_none());
        assert!(host_acceleration_structure_va(tlas_title_va).is_some());

        // Clean up remaining AS entries before `mem` drops.
        crate::watch::title_memory_remapped(base as usize, 256 << 10);
        assert!(host_acceleration_structure_va(clone_title_va).is_none());
        assert!(host_acceleration_structure_va(tlas_title_va).is_none());
    }

    #[test]
    fn read_geometries_rejects_null_oversized_and_uncommitted_inputs() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };

        struct Guard(*mut std::ffi::c_void);
        impl Drop for Guard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    // SAFETY: `self.0` was returned by `VirtualAlloc(..., MEM_RESERVE, ...)`.
                    unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
                }
            }
        }

        let make_inputs =
            |layout, count, anon| D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
                NumDescs: count,
                DescsLayout: layout,
                Anonymous: anon,
            };

        let geom = triangle_geometry(0x1000, 0x2000);
        let geom_ptr = &raw const geom;

        // Oversized counts must fail before allocating or reading memory.
        for bad_count in [MAX_GEOMETRY_DESCS as u32 + 1, u32::MAX] {
            assert!(
                read_geometries(&make_inputs(
                    D3D12_ELEMENTS_LAYOUT_ARRAY,
                    bad_count,
                    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: &geom,
                    },
                ))
                .is_none()
            );
            assert!(
                read_geometries(&make_inputs(
                    D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
                    bad_count,
                    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        ppGeometryDescs: &geom_ptr,
                    },
                ))
                .is_none()
            );
        }

        // Null pointers with non-zero count must fail.
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY,
                1,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: std::ptr::null(),
                },
            ))
            .is_none()
        );
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
                1,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    ppGeometryDescs: std::ptr::null(),
                },
            ))
            .is_none()
        );

        // Reserve 2 pages (8 KiB) uncommitted, commit only the first 4 KiB page.
        // SAFETY: reserving an anonymous 8 KiB address range.
        let base = unsafe { VirtualAlloc(std::ptr::null(), 8192, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!base.is_null());
        let _guard = Guard(base);
        // SAFETY: committing the first 4 KiB page inside our reservation.
        let committed = unsafe { VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(committed, base);
        let uncommitted = (base as usize + 4096) as *const D3D12_RAYTRACING_GEOMETRY_DESC;

        // Uncommitted ARRAY and ARRAY_OF_POINTERS bases must fail.
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY,
                1,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: uncommitted,
                },
            ))
            .is_none()
        );
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
                1,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    ppGeometryDescs: uncommitted.cast(),
                },
            ))
            .is_none()
        );

        // Committed ARRAY straddling into the uncommitted second page must fail.
        let straddle_count = (4096 / size_of::<D3D12_RAYTRACING_GEOMETRY_DESC>() + 2) as u32;
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY,
                straddle_count,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: base.cast(),
                },
            ))
            .is_none()
        );

        // Committed pointer array holding a null or uncommitted element must fail.
        let null_elem: [*const D3D12_RAYTRACING_GEOMETRY_DESC; 2] = [&geom, std::ptr::null()];
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
                2,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    ppGeometryDescs: null_elem.as_ptr(),
                },
            ))
            .is_none()
        );
        let bad_elem: [*const D3D12_RAYTRACING_GEOMETRY_DESC; 2] = [&geom, uncommitted];
        assert!(
            read_geometries(&make_inputs(
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
                2,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    ppGeometryDescs: bad_elem.as_ptr(),
                },
            ))
            .is_none()
        );

        // Valid ARRAY and ARRAY_OF_POINTERS inputs still decode cleanly.
        let valid_arr = read_geometries(&make_inputs(
            D3D12_ELEMENTS_LAYOUT_ARRAY,
            1,
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &geom,
            },
        ))
        .expect("valid ARRAY");
        assert_eq!(valid_arr.len(), 1);
        let valid_ptrs = read_geometries(&make_inputs(
            D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS,
            1,
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                ppGeometryDescs: &geom_ptr,
            },
        ))
        .expect("valid ARRAY_OF_POINTERS");
        assert_eq!(valid_ptrs.len(), 1);
    }
}
