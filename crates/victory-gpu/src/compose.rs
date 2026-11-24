//! Local build, from XCL's `present_compose`: multi-plane presentation. The console's display
//! composes the planes a title presents; Grand Theft Auto VI's Series S profile presents its
//! 1080p game image as plane 0 and its 4K interface as plane 1, so plane 0 alone is a black
//! screen in the menus and a game without HUD. The window's back buffer gets plane 0
//! stretched over it, then plane 1 blended on top (premultiplied alpha, as the display plane).
//!
//! `VICTORY_COMPOSE=0` turns it off (plane 0 copied as before).

use std::ffi::c_void;
use std::mem::transmute_copy;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

const HLSL: &str = r#"
Texture2D t0 : register(t0);
SamplerState s0 : register(s0);
struct V { float4 p : SV_Position; float2 uv : TEXCOORD0; };
V vs(uint id : SV_VertexID) {
    V o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.uv = uv;
    o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}
float4 ps_base(V i) : SV_Target { return float4(t0.Sample(s0, i.uv).rgb, 1); }
float4 ps_over(V i) : SV_Target { return t0.Sample(s0, i.uv); }
"#;

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_COMPOSE").map_or(true, |v| v.trim() != "0"))
}

pub(crate) struct Compositor {
    pub(crate) format: DXGI_FORMAT,
    rs: ID3D12RootSignature,
    base: ID3D12PipelineState,
    over: ID3D12PipelineState,
    srv: ID3D12DescriptorHeap,
    srv_inc: u32,
    rtv: ID3D12DescriptorHeap,
}

// SAFETY: D3D12 objects are free-threaded; used under the presenter's lock.
unsafe impl Send for Compositor {}

unsafe fn compile(entry: &str, target: &str) -> Result<ID3DBlob, String> {
    let e = std::ffi::CString::new(entry).unwrap();
    let t = std::ffi::CString::new(target).unwrap();
    let mut code: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    // SAFETY: plain compilation of our source.
    let r = unsafe {
        D3DCompile(
            HLSL.as_ptr() as *const c_void,
            HLSL.len(),
            PCSTR::null(),
            None,
            None,
            PCSTR(e.as_ptr() as *const u8),
            PCSTR(t.as_ptr() as *const u8),
            0,
            0,
            &mut code,
            Some(&mut err),
        )
    };
    if let Err(x) = r {
        let msg = err
            // SAFETY: the error blob's bytes.
            .map(|b| unsafe { String::from_utf8_lossy(std::slice::from_raw_parts(b.GetBufferPointer() as *const u8, b.GetBufferSize())).into_owned() })
            .unwrap_or_default();
        return Err(format!("D3DCompile {entry}: {x} {msg}"));
    }
    code.ok_or_else(|| format!("D3DCompile {entry}: no code"))
}

fn bytecode(b: &ID3DBlob) -> D3D12_SHADER_BYTECODE {
    // SAFETY: a live blob.
    unsafe { D3D12_SHADER_BYTECODE { pShaderBytecode: b.GetBufferPointer(), BytecodeLength: b.GetBufferSize() } }
}

impl Compositor {
    pub(crate) fn new(dev: &ID3D12Device, format: DXGI_FORMAT) -> Result<Compositor, String> {
        // SAFETY: plain object creation.
        unsafe {
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 { DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE { NumDescriptorRanges: 1, pDescriptorRanges: &range } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxLOD: f32::MAX,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, NumStaticSamplers: 1, pStaticSamplers: &sampler, Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE };
            let mut ser: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).map_err(|x| format!("serialize root signature: {x}"))?;
            let ser = ser.ok_or("serialize root signature")?;
            let rs: ID3D12RootSignature = dev
                .CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize()))
                .map_err(|x| format!("CreateRootSignature: {x}"))?;
            let vs = compile("vs", "vs_5_0")?;
            let ps_base = compile("ps_base", "ps_5_0")?;
            let ps_over = compile("ps_over", "ps_5_0")?;
            let pso = |ps: &ID3DBlob, blend: bool| -> Result<ID3D12PipelineState, String> {
                let mut pd = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: transmute_copy(&rs),
                    VS: bytecode(&vs),
                    PS: bytecode(ps),
                    SampleMask: u32::MAX,
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let rt = &mut pd.BlendState.RenderTarget[0];
                rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
                if blend {
                    rt.BlendEnable = true.into();
                    rt.SrcBlend = D3D12_BLEND_ONE;
                    rt.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt.BlendOp = D3D12_BLEND_OP_ADD;
                    rt.SrcBlendAlpha = D3D12_BLEND_ZERO;
                    rt.DestBlendAlpha = D3D12_BLEND_ONE;
                    rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
                pd.RasterizerState.DepthClipEnable = true.into();
                pd.RTVFormats[0] = format;
                dev.CreateGraphicsPipelineState(&pd).map_err(|x| format!("CreateGraphicsPipelineState: {x}"))
            };
            let base = pso(&ps_base, false)?;
            let over = pso(&ps_over, true)?;
            let srv: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 2, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 })
                .map_err(|x| format!("SRV heap: {x}"))?;
            let rtv: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE, NodeMask: 0 })
                .map_err(|x| format!("RTV heap: {x}"))?;
            let srv_inc = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            tracing::info!(format = format.0, "plane composition ready (plane 1 over plane 0)");
            Ok(Compositor { format, rs, base, over, srv, srv_inc, rtv })
        }
    }

    fn view(&self, dev: &ID3D12Device, slot: u32, res: &ID3D12Resource, format: DXGI_FORMAT) {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: 0, MipLevels: 1, PlaneSlice: 0, ResourceMinLODClamp: 0.0 } },
        };
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut h = self.srv.GetCPUDescriptorHandleForHeapStart();
            h.ptr += (slot * self.srv_inc) as usize;
            dev.CreateShaderResourceView(res, Some(&desc), h);
        }
    }

    /// Records onto `list`: `bb` (in PRESENT) gets `base` stretched over it and `over` blended
    /// on top, both read as shader resources (the caller moves them there), and goes back to
    /// PRESENT.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        dev: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        bb: &ID3D12Resource,
        w: u32,
        h: u32,
        base: (&ID3D12Resource, DXGI_FORMAT),
        over: Option<(&ID3D12Resource, DXGI_FORMAT)>,
    ) {
        self.view(dev, 0, base.0, base.1);
        if let Some(o) = over {
            self.view(dev, 1, o.0, o.1);
        }
        // SAFETY: recording onto an open list of ours.
        unsafe {
            let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(bb, None, rtv);
            list.ResourceBarrier(&[crate::host::transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(self.srv.clone())]);
            list.SetGraphicsRootSignature(&self.rs);
            list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: w as f32, Height: h as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: w as i32, bottom: h as i32 }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let gpu = self.srv.GetGPUDescriptorHandleForHeapStart();
            list.SetPipelineState(&self.base);
            list.SetGraphicsRootDescriptorTable(0, gpu);
            list.DrawInstanced(3, 1, 0, 0);
            if over.is_some() {
                list.SetPipelineState(&self.over);
                list.SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu.ptr + self.srv_inc as u64 });
                list.DrawInstanced(3, 1, 0, 0);
            }
            list.ResourceBarrier(&[crate::host::transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)]);
        }
    }
}
