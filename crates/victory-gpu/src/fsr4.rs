//! Fast bilinear bypass for FidelityFX FSR4 neural-network upscaling passes.
//!
//! On Xbox Series S, `amd_fidelityfx_upscaler_xbox.dll` executes a multi-pass
//! FP8/INT8 neural network (`ACCC`/`FSPX` compute shaders) every frame. On host
//! GPUs without native RDNA4 matrix instructions (such as RTX 2060), those
//! emulated compute passes dominate GPU frame time (~385 ms/frame). When
//! `VICTORY_FAST_FSR4=1` (or `VICTORY_FAST_COMPUTE` is enabled without
//! `VICTORY_FAST_FSR4=0`), intermediate model passes are skipped and the final
//! postpass is replaced with a single bilinear compute upscale from
//! `r_input_color` to `rw_mlsr_output_color`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;

/// Root parameter and descriptor-table dword offset of `r_input_color` (`t3`) in FSR4 postpass.
pub(crate) const INPUT_COLOR: (u8, u32) = (1, 0x18);
/// Root parameter and descriptor-table dword offset of `rw_mlsr_output_color` (`u2`) in FSR4 postpass.
pub(crate) const OUTPUT_COLOR: (u8, u32) = (0, 0x10);
/// Root parameter and descriptor-table dword offset of the shared `ScratchBuffer` UAV (`u11`).
const SCRATCH_UAV: (u8, u32) = (0, 0x58);

/// Classification of an `ACCC` (`FSPX`) compute shader in the FSR4 upscaler chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PassKind {
    /// Not an FSR4 neural-network model pass (e.g. SPD exposure, RCAS, or title shader).
    None,
    /// Prepass or intermediate neural-network pass writing `ScratchBuffer` (`u11`).
    Model,
    /// Final model postpass reading `r_input_color` (`t3`) and writing `rw_mlsr_output_color` (`u2`).
    Postpass,
}

/// Classifies a compute shader from its `ACCC` container flag and declared resources.
pub(crate) fn classify(is_accc: bool, used: &[UsedResource]) -> PassKind {
    if !is_accc {
        return PassKind::None;
    }
    let has = |kind: ResourceType, (slot, offset): (u8, u32)| {
        used.iter().any(|u| {
            u.kind == kind
                && u.root == RootType::Table
                && u.slot == slot
                && u.table_offset_dwords == offset
        })
    };
    if !has(ResourceType::Uav, SCRATCH_UAV) {
        return PassKind::None;
    }
    if has(ResourceType::Srv, INPUT_COLOR) && has(ResourceType::Uav, OUTPUT_COLOR) {
        PassKind::Postpass
    } else {
        PassKind::Model
    }
}

fn parse_bool(value: Option<&str>) -> Option<bool> {
    let trimmed = value?.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("off")
    {
        return Some(false);
    }
    if trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("on")
    {
        return Some(true);
    }
    None
}

fn parse_enabled(fsr4_env: Option<&str>, fast_compute_env: Option<&str>) -> bool {
    if let Some(explicit) = parse_bool(fsr4_env) {
        return explicit;
    }
    let Some(fc) = fast_compute_env.map(str::trim) else {
        return false;
    };
    if fc.is_empty()
        || fc == "0"
        || fc.eq_ignore_ascii_case("false")
        || fc.eq_ignore_ascii_case("off")
    {
        return false;
    }
    fc == "1"
        || fc.eq_ignore_ascii_case("true")
        || fc.eq_ignore_ascii_case("on")
        || fc.parse::<u64>().is_ok_and(|n| n > 1)
}

/// Whether the fast bilinear FSR4 bypass is active.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_enabled(
            std::env::var("VICTORY_FAST_FSR4").ok().as_deref(),
            std::env::var("VICTORY_FAST_COMPUTE").ok().as_deref(),
        );
        if on {
            tracing::info!("VICTORY_FAST_FSR4 active: replacing FSR4 neural passes with bilinear upscale");
        }
        on
    })
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), DescriptorTable(SRV(t0), UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_LINEAR_MIP_POINT, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"
cbuffer Params : register(b0) { float2 uv_scale; float2 uv_max; };
Texture2D<float4> src : register(t0);
RWTexture2D<float4> dst : register(u0);
SamplerState lin : register(s0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    dst.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    float2 uv = min((float2(at.xy) + 0.5) / float2(w, h) * uv_scale, uv_max);
    dst[at.xy] = src.SampleLevel(lin, uv, 0);
}
"#;

const SLOTS: u32 = 1024;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    step: u32,
    next: AtomicU32,
}

// SAFETY: D3D12 interfaces in `Pass` are thread-safe to share after creation.
unsafe impl Send for Pass {}
// SAFETY: descriptor slot allocation uses an atomic counter.
unsafe impl Sync for Pass {}

fn compile() -> Result<Vec<u8>, String> {
    let mut blob: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `SOURCE` is valid UTF-8 HLSL and output pointers are valid locals.
    let hr = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("fsr4_bilinear.hlsl"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut blob,
            Some(&mut errors),
        )
    };
    if let Err(e) = hr {
        let detail = errors
            .map(|b| {
                // SAFETY: `D3DCompile` returns a valid error buffer.
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
                };
                String::from_utf8_lossy(bytes).trim().to_owned()
            })
            .unwrap_or_default();
        return Err(format!("{e}: {detail}"));
    }
    let blob = blob.ok_or_else(|| "D3DCompile returned no shader".to_owned())?;
    // SAFETY: `blob` owns the compiled DXBC bytecode.
    Ok(unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    })
}

fn make(device: &ID3D12Device) -> Result<Pass, String> {
    let code = compile()?;
    // SAFETY: root signature is embedded in the compiled CS blob.
    unsafe {
        let root: ID3D12RootSignature = device
            .CreateRootSignature(0, &code)
            .map_err(|e| format!("root signature: {e}"))?;
        let pso = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .map_err(|e| format!("compute PSO: {e}"))?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: SLOTS * 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .map_err(|e| format!("descriptor heap: {e}"))?;
        Ok(Pass {
            root,
            pso,
            views,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn view_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R8G8B8A8_TYPELESS | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R32G32B32A32_TYPELESS => DXGI_FORMAT_R32G32B32A32_FLOAT,
        _ => format,
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: caller holds an open direct graphics command list.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// Computes `(uv_scale, uv_max)` for sampling an active DRS sub-rect `(render_w, render_h)`
/// inside a backing texture of size `(src_w, src_h)` without bleeding past the active border.
fn drs_uv_params(src_w: u32, src_h: u32, render: Option<(u32, u32)>) -> [f32; 4] {
    let sw = src_w.max(1);
    let sh = src_h.max(1);
    let (rw, rh) = match render {
        Some((w, h)) if w > 0 && w <= sw && h > 0 && h <= sh => (w, h),
        _ => (sw, sh),
    };
    let sx = rw as f32 / sw as f32;
    let sy = rh as f32 / sh as f32;
    let mx = ((rw as f32 - 0.5).max(0.5)) / sw as f32;
    let my = ((rh as f32 - 0.5).max(0.5)) / sh as f32;
    [sx, sy, mx, my]
}

fn read_render_size(cb0: Option<u64>) -> Option<(u32, u32)> {
    let addr = cb0?;
    let mut raw = [0u8; 8];
    if !crate::host::read_into(addr as *const u8, &mut raw) {
        return None;
    }
    let w = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let h = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    Some((w, h))
}

/// Bilinearly upscales `src` (`r_input_color`) into `dst` (`rw_mlsr_output_color`).
/// Returns `true` when it overrides the list's descriptor heap, root signature, and PSO.
pub(crate) fn upscale(list: &HostList, src: This, dst: This, cb0: Option<u64>) -> bool {
    let Some(host) = host() else {
        return false;
    };
    let (Some(src_res), Some(dst_res)) = (
        crate::resource::host_resource(src),
        crate::resource::host_resource(dst),
    ) else {
        return false;
    };
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no fast FSR4 upscale pass"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    // SAFETY: `src_res` and `dst_res` are live resources retained on `list` below.
    let (sd, dd) = unsafe { (src_res.GetDesc(), dst_res.GetDesc()) };
    let (src_fmt, dst_fmt) = (view_format(sd.Format), view_format(dd.Format));
    let render = read_render_size(cb0);
    let params = drs_uv_params(sd.Width as u32, sd.Height, render);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            src_w = sd.Width,
            src_h = sd.Height,
            src_fmt = src_fmt.0,
            render = ?render,
            dst_w = dd.Width,
            dst_h = dd.Height,
            dst_fmt = dst_fmt.0,
            uv_scale = ?[params[0], params[1]],
            "fast FSR4 bilinear upscale active"
        );
    });
    let slot = pass.next.fetch_add(1, Ordering::Relaxed) % SLOTS;
    let base = slot * 2;
    list.keep(src_res);
    list.keep(dst_res);
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: `l` is open, resources are kept by `list` and transitioned by caller, and `slot` cycles safely.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let srv_at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (base * pass.step) as usize,
        };
        let uav_at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + ((base + 1) * pass.step) as usize,
        };
        let mut srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: src_fmt,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        srv.Anonymous.Texture2D = D3D12_TEX2D_SRV {
            MipLevels: 1,
            ..Default::default()
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: dst_fmt,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device.CreateShaderResourceView(src_res, Some(&srv), srv_at);
        host.device
            .CreateUnorderedAccessView(dst_res, None::<&ID3D12Resource>, Some(&uav), uav_at);
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(base * pass.step),
            },
        );
        l.Dispatch((dd.Width as u32).div_ceil(8), dd.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_res(kind: ResourceType, slot: u8, offset: u32) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: offset,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot,
        }
    }

    #[test]
    fn parse_enabled_respects_explicit_and_fast_compute() {
        assert!(!parse_enabled(None, None));
        assert!(!parse_enabled(None, Some("0")));
        assert!(parse_enabled(None, Some("1")));
        assert!(parse_enabled(None, Some("8")));
        assert!(!parse_enabled(Some("0"), Some("1")));
        assert!(!parse_enabled(Some("off"), Some("on")));
        assert!(parse_enabled(Some("1"), None));
        assert!(parse_enabled(Some("true"), Some("0")));
    }

    #[test]
    fn classify_identifies_model_and_postpass() {
        let scratch = table_res(ResourceType::Uav, 0, 0x58);
        let input = table_res(ResourceType::Srv, 1, 0x18);
        let output = table_res(ResourceType::Uav, 0, 0x10);
        assert_eq!(classify(false, &[scratch, input, output]), PassKind::None);
        assert_eq!(classify(true, &[input, output]), PassKind::None);
        assert_eq!(classify(true, &[scratch]), PassKind::Model);
        assert_eq!(classify(true, &[scratch, input, output]), PassKind::Postpass);
    }

    #[test]
    fn view_format_unwraps_typeless_and_srgb() {
        assert_eq!(view_format(DXGI_FORMAT_R16G16B16A16_TYPELESS), DXGI_FORMAT_R16G16B16A16_FLOAT);
        assert_eq!(view_format(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB), DXGI_FORMAT_R8G8B8A8_UNORM);
        assert_eq!(view_format(DXGI_FORMAT_R11G11B10_FLOAT), DXGI_FORMAT_R11G11B10_FLOAT);
    }

    #[test]
    fn drs_uv_params_scales_active_rect_and_falls_back_cleanly() {
        let p = drs_uv_params(1280, 720, Some((960, 540)));
        assert!((p[0] - 0.75).abs() < 1e-6);
        assert!((p[1] - 0.75).abs() < 1e-6);
        assert!(p[2] < p[0] && p[3] < p[1]);
        let full = drs_uv_params(1280, 720, Some((0, 0)));
        assert!((full[0] - 1.0).abs() < 1e-6);
        assert!((full[1] - 1.0).abs() < 1e-6);
        let oob = drs_uv_params(1280, 720, Some((4096, 2160)));
        assert!((oob[0] - 1.0).abs() < 1e-6);
        assert!((oob[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fsr4_pass_builds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        super::make(&host.device).expect("the fast FSR4 bilinear pass");
    }
}
