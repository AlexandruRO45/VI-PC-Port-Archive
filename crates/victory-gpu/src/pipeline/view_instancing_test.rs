//! Synthetic SM 6.6 DXIL shaders, render targets and readback for view
//! instancing (pipeline stream subobject 22) through the title entry points.

use super::*;
use crate::object::{S_OK, slot};
use victory_shader::native::StageIo;
use victory_shader::{Container, HwStage, NativeProgram, emit, io, ir, resource, structure};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::core::{GUID, Interface};

#[repr(C, align(8))]
struct ViewInstancingSub(u32, D3D12_VIEW_INSTANCING_DESC);
const _: () = assert!(size_of::<ViewInstancingSub>() == 32);

#[repr(C, align(8))]
struct Depth1Sub(u32, D3D12_DEPTH_STENCIL_DESC1);
const _: () = assert!(size_of::<Depth1Sub>() == 64);

#[repr(C, align(8))]
struct RtFormatsSub(u32, D3D12_RT_FORMAT_ARRAY);
const _: () = assert!(size_of::<RtFormatsSub>() == 40);

#[test]
fn view_instancing_changes_the_stream_state() {
    let mut g = host::GraphicsState::default();
    let default = g.describe();

    let read = |g: &mut host::GraphicsState, desc: D3D12_VIEW_INSTANCING_DESC| {
        let sub = ViewInstancingSub(22, desc);
        // SAFETY: a whole kind-22 subobject.
        unsafe { g.read(22, std::ptr::from_ref(&sub).cast()) };
    };

    let two_viewports = [
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 0 },
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 1, RenderTargetArrayIndex: 0 },
    ];
    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: two_viewports.len() as u32,
            pViewInstanceLocations: two_viewports.as_ptr(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_NONE,
        },
    );
    let keyed_vp = g.describe();
    assert_ne!(keyed_vp, default, "enabled view instancing must change the pipeline key");

    let two_slices = [
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 0 },
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 1 },
    ];
    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: two_slices.len() as u32,
            pViewInstanceLocations: two_slices.as_ptr(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_NONE,
        },
    );
    let keyed_rt = g.describe();
    assert_ne!(keyed_rt, keyed_vp, "different view locations must change the pipeline key");

    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: two_slices.len() as u32,
            pViewInstanceLocations: two_slices.as_ptr(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
        },
    );
    assert_ne!(g.describe(), keyed_rt, "view instancing flags must change the pipeline key");

    // ViewInstanceCount == 0 disables view instancing and restores the default key.
    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: 0,
            pViewInstanceLocations: std::ptr::null(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_NONE,
        },
    );
    assert_eq!(g.describe(), default);

    // Null locations or counts above D3D12_MAX_VIEW_INSTANCE_COUNT are ignored safely.
    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: 2,
            pViewInstanceLocations: std::ptr::null(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_NONE,
        },
    );
    assert_eq!(g.describe(), default);

    let too_many = [D3D12_VIEW_INSTANCE_LOCATION::default(); 5];
    read(
        &mut g,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: too_many.len() as u32,
            pViewInstanceLocations: too_many.as_ptr(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_NONE,
        },
    );
    assert_eq!(g.describe(), default);
}

/// Builds a minimal `ISG1` / `OSG1` signature chunk.
fn sig_chunk(elements: &[(&str, u32, u32, u32, u8, u8)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((elements.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    let mut next = 8 + 32 * elements.len() as u32;
    let mut names = Vec::new();
    for &(name, sys_val, comp, row, mask, rw) in elements {
        let offset = next;
        next += name.len() as u32 + 1;
        names.push(name);
        for v in [0u32, offset, 0, sys_val, comp, row] {
            out.extend(v.to_le_bytes());
        }
        out.extend([mask, rw, 0, 0]);
        out.extend(0u32.to_le_bytes());
    }
    for name in names {
        out.extend(name.as_bytes());
        out.push(0);
    }
    out
}

/// Builds a minimal DXBC container wrapping the given FourCC parts.
fn dxbc_container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let table = 0x20 + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend(b"DXBC");
    out.extend([0u8; 16]);
    out.extend(1u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend((offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend(*fourcc);
        out.extend((d.len() as u32).to_le_bytes());
        out.extend(*d);
    }
    out
}

/// Translates a synthetic GFX10 shader to SM 6.6 DXIL (with `PSV0` v2
/// validation metadata required by D3D12 view instancing).
fn emit_shader(
    stage_name: &'static str,
    native: &NativeProgram,
    isg1: &[u8],
    osg1: &[u8],
) -> Arc<Shader> {
    let container = dxbc_container(&[(b"ISG1", isg1), (b"OSG1", osg1)]);
    let c = Container::parse(&container).unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, native).unwrap();
    let host = emit::emit(&c, native, &p, &st, &io, &res).unwrap();
    let shader = shader(stage_name, &container);
    shader.translated.get_or_init(|| {
        Some(Translated {
            host,
            used: native.used.clone(),
        })
    });
    shader
}

/// Fullscreen triangle vertex shader (`SV_VertexID` in `v0` -> `SV_Position`
/// at `z = 0.25`, clockwise front-facing).
fn fullscreen_vs() -> Arc<Shader> {
    let words: [u32; 14] = [
        0x3602_0081, // v_and_b32 v1, 1, v0          (id & 1)
        0x3604_0082, // v_and_b32 v2, 2, v0          (id & 2)
        0x7E02_0D01, // v_cvt_f32_u32 v1, v1         (0.0, 1.0, 0.0)
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2         (0.0, 0.0, 2.0)
        0x1002_02F6, // v_mul_f32 v1, 4.0, v1        (0.0, 4.0, 0.0)
        0x0602_02F3, // v_add_f32 v1, -1.0, v1       (-1.0, 3.0, -1.0)
        0x1004_04F5, // v_mul_f32 v2, -2.0, v2       (0.0, 0.0, -4.0)
        0x0604_04F2, // v_add_f32 v2, 1.0, v2        (1.0, 1.0, -3.0)
        0x7E06_02FF,
        0x3E80_0000, // v_mov_b32 v3, 0.25
        0x7E08_02F2, // v_mov_b32 v4, 1.0
        0xF800_08CF,
        0x0403_0201, // exp pos0, v1, v2, v3, v4 done
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Vs,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: Vec::new(),
            outputs: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg1 = sig_chunk(&[("SV_VertexID", 6, 1, u32::MAX, 0x1, 0x1)]);
    let osg1 = sig_chunk(&[("SV_Position", 1, 3, 0, 0xF, 0x0)]);
    emit_shader("vertex", &native, &isg1, &osg1)
}

/// Solid-green pixel shader (`SV_Target0 = float4(0, 1, 0, 1)`).
fn green_ps() -> Arc<Shader> {
    let words: [u32; 5] = [
        0x7E00_0280, // v_mov_b32 v0, 0.0
        0x7E02_02F2, // v_mov_b32 v1, 1.0
        0xF800_180F,
        0x0100_0100, // exp mrt0, v0, v1, v0, v1 done vm
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let isg1 = sig_chunk(&[("SV_Position", 1, 3, 0, 0xF, 0x0)]);
    let osg1 = sig_chunk(&[("SV_Target", 64, 3, 0, 0xF, 0x0)]);
    emit_shader("pixel", &native, &isg1, &osg1)
}

fn made(pso: This) -> &'static host::HostPipeline {
    let started = std::time::Instant::now();
    loop {
        if let Some(p) = host_pipeline(pso) {
            return p;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(60),
            "the host refused the view instancing pipeline, or did not make it in a minute"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn adapter_name(device: &ID3D12Device) -> String {
    // SAFETY: plain adapter queries on a live device.
    let desc = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        adapter.GetDesc1().unwrap()
    };
    String::from_utf16_lossy(&desc.Description)
        .trim_end_matches('\0')
        .to_owned()
}

fn make_resource(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    kind: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
    let mut result: Option<ID3D12Resource> = None;
    // SAFETY: valid resource description and output pointer.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut result)
            .unwrap();
    }
    result.unwrap()
}

fn make_list() -> (This, This) {
    let null = std::ptr::null();
    let mut allocator = null;
    let mut list = null;
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list,
        ),
        S_OK
    );
    (allocator, list)
}

fn release_objects(objects: &[This]) {
    for &object in objects {
        assert_eq!(slot::<extern "system" fn(This) -> u32>(object, 0x10)(object), 0);
    }
}

fn make_color_pso(
    vs: &Arc<Shader>,
    ps: &Arc<Shader>,
    locations: Option<&[D3D12_VIEW_INSTANCE_LOCATION]>,
    flags: D3D12_VIEW_INSTANCING_FLAGS,
    depth_bounds: bool,
) -> This {
    let mut rts = D3D12_RT_FORMAT_ARRAY {
        NumRenderTargets: 1,
        ..Default::default()
    };
    rts.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

    let vi_desc = match locations {
        Some(locs) => D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: locs.len() as u32,
            pViewInstanceLocations: locs.as_ptr(),
            Flags: flags,
        },
        None => D3D12_VIEW_INSTANCING_DESC::default(),
    };

    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth1Sub,
        topology: [u32; 2],
        targets: RtFormatsSub,
        dsv: [u32; 2],
        vi: ViewInstancingSub,
    }
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode { code: vs.container.as_ptr(), size: vs.container.len() },
        ps_kind: 2,
        ps: Bytecode { code: ps.container.as_ptr(), size: ps.container.len() },
        depth: Depth1Sub(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: depth_bounds.into(),
                DepthWriteMask: if depth_bounds {
                    D3D12_DEPTH_WRITE_MASK_ALL
                } else {
                    D3D12_DEPTH_WRITE_MASK_ZERO
                },
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: depth_bounds.into(),
                ..Default::default()
            },
        ),
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        targets: RtFormatsSub(15, rts),
        dsv: [
            16,
            if depth_bounds {
                DXGI_FORMAT_D32_FLOAT.0 as u32
            } else {
                DXGI_FORMAT_UNKNOWN.0 as u32
            },
        ],
        vi: ViewInstancingSub(22, vi_desc),
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let null = std::ptr::null();
    let mut pso = null;
    assert_eq!(create_stream(null, &desc, &PIPELINE_IIDS[0], &mut pso), S_OK);
    assert!(!made(pso).mesh);
    pso
}

const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 0];

#[test]
fn view_instancing_routes_one_title_draw_to_multiple_viewports() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped view instancing viewport render: no host D3D12 device");
        return;
    };
    let supported = host.view_instancing_supported;
    eprintln!("view instancing on {}: {supported}", adapter_name(&host.device));

    let vs = fullscreen_vs();
    let ps = green_ps();
    let two_viewports = [
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 0 },
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 1, RenderTargetArrayIndex: 0 },
    ];
    let pso_plain = make_color_pso(&vs, &ps, None, D3D12_VIEW_INSTANCING_FLAG_NONE, false);
    let pso_vi = make_color_pso(
        &vs,
        &ps,
        Some(&two_viewports),
        D3D12_VIEW_INSTANCING_FLAG_NONE,
        false,
    );
    let pso_vi_masked = make_color_pso(
        &vs,
        &ps,
        Some(&two_viewports),
        D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
        false,
    );

    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let readback_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: rt_desc.SampleDesc,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let (allocator, list) = make_list();
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;

    let viewports = [
        D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: 2.0, Height: 1.0, MinDepth: 0.0, MaxDepth: 1.0 },
        D3D12_VIEWPORT { TopLeftX: 2.0, TopLeftY: 0.0, Width: 2.0, Height: 1.0, MinDepth: 0.0, MaxDepth: 1.0 },
    ];
    let scissors = [
        RECT { left: 0, top: 0, right: 2, bottom: 1 },
        RECT { left: 2, top: 0, right: 4, bottom: 1 },
    ];

    let run_case = |pso: This, first: bool| -> [[u8; 4]; 4] {
        if !first {
            assert_eq!(
                slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                    list,
                    allocator,
                    std::ptr::null(),
                ),
                S_OK
            );
        }
        let rt = make_resource(
            &host.device,
            &rt_desc,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let readback = make_resource(
            &host.device,
            &readback_desc,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let rtv = host.rtv(&rt, None);
        // SAFETY: live render target and open command list.
        unsafe {
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
        }
        slot::<extern "system" fn(This, This)>(list, 0xc8)(list, pso);
        slot::<extern "system" fn(This, u32)>(list, 0xa0)(list, 4); // triangle list
        slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(
            list,
            viewports.len() as u32,
            viewports.as_ptr(),
        );
        slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0)(
            list,
            scissors.len() as u32,
            scissors.as_ptr(),
        );
        slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60)(list, 3, 1, 0, 0);

        let mut from = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..Default::default()
        };
        let mut to = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: rt_desc.Format,
                        Width: 4,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
            ..Default::default()
        };
        // SAFETY: locations borrow live resources for the copy.
        unsafe {
            from.pResource = std::mem::transmute_copy(&rt);
            to.pResource = std::mem::transmute_copy(&readback);
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        host.execute(&[commands]);

        let mut mapped = std::ptr::null_mut();
        // SAFETY: GPU copy finished; readback holds at least 16 bytes.
        unsafe {
            readback
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut mapped))
                .unwrap();
            let pixels = mapped.cast::<[[u8; 4]; 4]>().read();
            readback.Unmap(0, Some(&D3D12_RANGE::default()));
            pixels
        }
    };

    let plain_pixels = run_case(pso_plain, true);
    let vi_pixels = run_case(pso_vi, false);
    let plain_after_vi = run_case(pso_plain, false);
    let masked_pixels = run_case(pso_vi_masked, false);
    eprintln!(
        "viewport readback plain={plain_pixels:?} vi={vi_pixels:?} plain_after_vi={plain_after_vi:?} masked={masked_pixels:?}"
    );

    assert_eq!(plain_pixels, [GREEN, GREEN, BLACK, BLACK]);
    let expected_vi = if supported {
        [GREEN, GREEN, GREEN, GREEN]
    } else {
        [GREEN, GREEN, BLACK, BLACK]
    };
    assert_eq!(vi_pixels, expected_vi);
    assert_eq!(
        plain_after_vi,
        [GREEN, GREEN, BLACK, BLACK],
        "a plain pipeline bound after view instancing must still render only view 0"
    );
    assert_eq!(masked_pixels, expected_vi);

    release_objects(&[list, allocator, pso_plain, pso_vi, pso_vi_masked]);
}

#[test]
fn view_instancing_routes_one_title_draw_to_multiple_array_slices() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped view instancing slice render: no host D3D12 device");
        return;
    };
    let supported = host.view_instancing_supported;

    let vs = fullscreen_vs();
    let ps = green_ps();
    let two_slices = [
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 0 },
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 1 },
    ];
    let pso = make_color_pso(
        &vs,
        &ps,
        Some(&two_slices),
        D3D12_VIEW_INSTANCING_FLAG_NONE,
        false,
    );

    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 2,
        Height: 1,
        DepthOrArraySize: 2,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let readback_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 512,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: rt_desc.SampleDesc,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let rt = make_resource(
        &host.device,
        &rt_desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    let readback = make_resource(
        &host.device,
        &readback_desc,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                MipSlice: 0,
                FirstArraySlice: 0,
                ArraySize: 2,
                PlaneSlice: 0,
            },
        },
    };
    let rtv = host.rtv(&rt, Some(&rtv_desc));

    let (allocator, list) = make_list();
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;
    // SAFETY: live 2-slice render target array and open list.
    unsafe {
        l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
        l.OMSetRenderTargets(1, Some(&rtv), false, None);
    }
    slot::<extern "system" fn(This, This)>(list, 0xc8)(list, pso);
    slot::<extern "system" fn(This, u32)>(list, 0xa0)(list, 4);
    let vp = D3D12_VIEWPORT { Width: 2.0, Height: 1.0, MaxDepth: 1.0, ..Default::default() };
    let sc = RECT { left: 0, top: 0, right: 2, bottom: 1 };
    slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(list, 1, &vp);
    slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0)(list, 1, &sc);
    slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60)(list, 3, 1, 0, 0);

    // SAFETY: copy both array slices into consecutive 256-byte rows.
    unsafe {
        l.ResourceBarrier(&[crate::host::transition(
            &rt,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for slice in 0u32..2 {
            let mut from = D3D12_TEXTURE_COPY_LOCATION {
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: slice },
                ..Default::default()
            };
            let mut to = D3D12_TEXTURE_COPY_LOCATION {
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: u64::from(slice) * 256,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: rt_desc.Format,
                            Width: 2,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
                ..Default::default()
            };
            from.pResource = std::mem::transmute_copy(&rt);
            to.pResource = std::mem::transmute_copy(&readback);
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
    }
    assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
    host.execute(&[commands]);

    let mut mapped = std::ptr::null_mut();
    // SAFETY: GPU copy finished; readback holds both 256-byte slice rows.
    let (slice0, slice1) = unsafe {
        readback
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 512 }), Some(&mut mapped))
            .unwrap();
        let base = mapped.cast::<u8>();
        let s0 = base.cast::<[[u8; 4]; 2]>().read();
        let s1 = base.add(256).cast::<[[u8; 4]; 2]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        (s0, s1)
    };
    eprintln!("array slice readback slice0={slice0:?} slice1={slice1:?}");
    assert_eq!(slice0, [GREEN, GREEN]);
    assert_eq!(slice1, if supported { [GREEN, GREEN] } else { [BLACK, BLACK] });

    release_objects(&[list, allocator, pso]);
}

#[test]
fn view_instancing_combines_with_depth_bounds() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped combined depth bounds + view instancing: no host D3D12 device");
        return;
    };
    let vs = fullscreen_vs();
    let ps = green_ps();
    let two_viewports = [
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: 0 },
        D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 1, RenderTargetArrayIndex: 0 },
    ];
    let pso = make_color_pso(
        &vs,
        &ps,
        Some(&two_viewports),
        D3D12_VIEW_INSTANCING_FLAG_NONE,
        true,
    );

    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let depth_desc = D3D12_RESOURCE_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..rt_desc
    };
    let readback_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: rt_desc.SampleDesc,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let rt = make_resource(
        &host.device,
        &rt_desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_RENDER_TARGET,
    );
    let depth = make_resource(
        &host.device,
        &depth_desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
    );
    let readback = make_resource(
        &host.device,
        &readback_desc,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let rtv = host.rtv(&rt, None);
    let dsv = host.dsv(&depth, None);

    let (allocator, list) = make_list();
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;
    let newer = crate::list::has_newer_table(list);
    let set_bounds =
        slot::<extern "system" fn(This, f32, f32)>(list, if newer { 0x230 } else { 0x220 });
    let viewports = [
        D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: 2.0, Height: 1.0, MinDepth: 0.0, MaxDepth: 1.0 },
        D3D12_VIEWPORT { TopLeftX: 2.0, TopLeftY: 0.0, Width: 2.0, Height: 1.0, MinDepth: 0.0, MaxDepth: 1.0 },
    ];
    let scissors = [
        RECT { left: 0, top: 0, right: 2, bottom: 1 },
        RECT { left: 2, top: 0, right: 4, bottom: 1 },
    ];

    let read_pixels = |first: bool, min_z: f32, max_z: f32| -> [[u8; 4]; 4] {
        if !first {
            assert_eq!(
                slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(
                    list,
                    allocator,
                    std::ptr::null(),
                ),
                S_OK
            );
            // SAFETY: transition RT back from COPY_SOURCE to RENDER_TARGET.
            unsafe {
                l.ResourceBarrier(&[crate::host::transition(
                    &rt,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
        }
        // SAFETY: live targets and open command list.
        unsafe {
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
            l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
            l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
        }
        slot::<extern "system" fn(This, This)>(list, 0xc8)(list, pso);
        slot::<extern "system" fn(This, u32)>(list, 0xa0)(list, 4);
        slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(
            list,
            viewports.len() as u32,
            viewports.as_ptr(),
        );
        slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0)(
            list,
            scissors.len() as u32,
            scissors.as_ptr(),
        );
        set_bounds(list, min_z, max_z);
        slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60)(list, 3, 1, 0, 0);

        let mut from = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            ..Default::default()
        };
        let mut to = D3D12_TEXTURE_COPY_LOCATION {
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: rt_desc.Format,
                        Width: 4,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
            ..Default::default()
        };
        // SAFETY: locations borrow live resources for the copy.
        unsafe {
            from.pResource = std::mem::transmute_copy(&rt);
            to.pResource = std::mem::transmute_copy(&readback);
            l.ResourceBarrier(&[crate::host::transition(
                &rt,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        assert_eq!(slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list), S_OK);
        host.execute(&[commands]);

        let mut mapped = std::ptr::null_mut();
        // SAFETY: GPU copy finished; readback holds at least 16 bytes.
        unsafe {
            readback
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut mapped))
                .unwrap();
            let pixels = mapped.cast::<[[u8; 4]; 4]>().read();
            readback.Unmap(0, Some(&D3D12_RANGE::default()));
            pixels
        }
    };

    let excluded = read_pixels(true, 0.0, 0.25);
    let included = read_pixels(false, 0.4, 0.6);
    eprintln!("combined depth-bounds + view-instancing readback excluded={excluded:?} included={included:?}");

    let painted = if host.view_instancing_supported {
        [GREEN, GREEN, GREEN, GREEN]
    } else {
        [GREEN, GREEN, BLACK, BLACK]
    };
    assert_eq!(
        excluded,
        if host.depth_bounds_supported {
            [BLACK, BLACK, BLACK, BLACK]
        } else {
            painted
        }
    );
    assert_eq!(included, painted);

    release_objects(&[list, allocator, pso]);
}
