//! Synthetic WARP draw adapted from fxceless/victory-fork #604.

use super::super::{
    Bytecode, PIPELINE_IIDS, StreamDesc, Translated, create_stream, library, make, pipeline,
    release, shader,
};
use super::{GraphicsState, HostPipeline};
use crate::host::{host, transition};
use crate::object::{S_OK, This};
use std::sync::Arc;
use victory_shader::{HwStage, emit::HostShader};
use windows::Win32::Foundation::{CloseHandle, RECT, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter1, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

const VS: &str = "float4 main(uint id : SV_VertexID) : SV_Position {
    const float2 vertices[] = {float2(-1, -1), float2(-1, 3), float2(3, -1)};
    return float4(vertices[id], 0, 1);
}";
const PS: &str = "float4 main() : SV_Target { return float4(1, 0, 0, 1); }";
const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
fn compile(dxc: &std::path::Path, dir: &std::path::Path, source: &str, profile: &str) -> Vec<u8> {
    let input = dir.join(format!("{profile}.hlsl"));
    let output = dir.join(format!("{profile}.dxil"));
    std::fs::write(&input, source).unwrap();
    let result = std::process::Command::new(dxc)
        .args(["-T", profile, "-E", "main", "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .expect("run Windows SDK DXC");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    std::fs::read(output).unwrap()
}

fn shader_bytes(dxc: &std::path::Path, dir: &std::path::Path) -> [Vec<u8>; 2] {
    [
        (VS, "vs_6_1", "vertex", HwStage::Vs),
        (PS, "ps_6_1", "pixel", HwStage::Ps),
    ]
    .map(|(source, profile, name, stage)| {
        let bytes = compile(dxc, dir, source, profile);
        // Supply synthetic host shaders at the translation boundary.
        // PSO creation, caching, and binding use the normal path.
        shader(name, &bytes).translated.get_or_init(|| {
            Some(Translated {
                host: HostShader {
                    stage,
                    dxil: bytes.clone(),
                    user_data: vec![],
                    samplers: vec![],
                    static_samplers: vec![],
                    root_cbvs: vec![],
                    root_srvs: vec![],
                    root_uavs: vec![],
                    memory: false,
                    memory_writes: false,
                    view_addresses: false,
                    heap: false,
                    heap_writes: false,
                    view_marks: false,
                },
                used: vec![],
            })
        });
        bytes
    })
}

fn create(
    shaders: &[Vec<u8>; 2],
    routes: &[(u32, u32)],
    flags: D3D12_VIEW_INSTANCING_FLAGS,
) -> Arc<make::Pending> {
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        views: Sub<D3D12_VIEW_INSTANCING_DESC>,
        vs: Sub<Bytecode>,
        ps: Sub<Bytecode>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
    }
    let [vs, ps] = shaders.each_ref().map(|bytes| Bytecode {
        code: bytes.as_ptr(),
        size: bytes.len(),
    });
    let locations: Vec<_> = routes
        .iter()
        .map(|&(viewport, slice)| D3D12_VIEW_INSTANCE_LOCATION {
            ViewportArrayIndex: viewport,
            RenderTargetArrayIndex: slice,
        })
        .collect();
    let depth = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: false.into(),
        ..GraphicsState::default().depth
    };
    let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
    formats[0] = FORMAT;
    let stream = Stream {
        views: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,
            D3D12_VIEW_INSTANCING_DESC {
                ViewInstanceCount: locations.len() as u32,
                pViewInstanceLocations: locations.as_ptr(),
                Flags: flags,
            },
        ),
        vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        raster: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
        ),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, depth),
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            D3D12_RT_FORMAT_ARRAY {
                RTFormats: formats,
                NumRenderTargets: 1,
            },
        ),
    };
    let desc = StreamDesc {
        size: size_of::<Stream>(),
        stream: (&raw const stream).cast(),
    };
    let mut raw: This = std::ptr::null();
    assert_eq!(
        create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut raw),
        S_OK
    );
    let pending = pipeline(raw).host.clone();
    release(raw);
    pending
}

fn draw(pso: &HostPipeline) -> [[usize; 2]; 2] {
    let host = host().unwrap();
    let device = &host.device;
    const WIDTH: usize = 8;
    const HEIGHT: usize = 4;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: WIDTH as u64,
        Height: HEIGHT as u32,
        DepthOrArraySize: 2,
        MipLevels: 1,
        Format: FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    // SAFETY: all objects and memory are owned by this draw. The fence
    // completes the readback before the buffer is mapped.
    unsafe {
        let mut target = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
            .unwrap();
        let target: ID3D12Resource = target.unwrap();
        let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut bytes = 0;
        device.GetCopyableFootprints(
            &desc,
            0,
            2,
            0,
            Some(footprints.as_mut_ptr()),
            None,
            None,
            Some(&mut bytes),
        );
        let mut readback = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut readback,
            )
            .unwrap();
        let readback: ID3D12Resource = readback.unwrap();
        let heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        device.CreateRenderTargetView(&target, None, rtv);

        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.SetGraphicsRootSignature(&pso.root);
        list.SetPipelineState(&pso.pso);
        list.RSSetViewports(&[0.0, 4.0].map(|x| D3D12_VIEWPORT {
            TopLeftX: x,
            Width: 4.0,
            Height: 4.0,
            MaxDepth: 1.0,
            ..Default::default()
        }));
        list.RSSetScissorRects(&[0, 4].map(|x| RECT {
            left: x,
            top: 0,
            right: x + 4,
            bottom: 4,
        }));
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.DrawInstanced(3, 1, 0, 0);
        list.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for (slice, footprint) in footprints.iter().enumerate() {
            let source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: slice as u32,
                },
            };
            let dest = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: *footprint,
                },
            };
            list.CopyTextureRegion(&dest, 0, 0, 0, &source, None);
        }
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        let waited = fence
            .SetEventOnCompletion(1, event)
            .map(|()| WaitForSingleObject(event, 10_000));
        CloseHandle(event).unwrap();
        assert_eq!(waited.unwrap(), WAIT_OBJECT_0);
        device.GetDeviceRemovedReason().unwrap();

        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        let image = std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize);
        let pixels = footprints.map(|footprint| {
            let mut red = [0; 2];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    let at = footprint.Offset as usize
                        + y * footprint.Footprint.RowPitch as usize
                        + x * 4;
                    red[x / (WIDTH / 2)] += usize::from(image[at..at + 4] == [255, 0, 0, 255]);
                }
            }
            red
        });
        readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        pixels
    }
}

#[test]
fn graphics_view_selection_uses_effective_library_state() {
    let plain = GraphicsState::default();
    let plain_state = plain.describe();
    assert!(plain.effective_views(|| panic!("no view query for plain graphics")).is_empty());

    let mut graphics = GraphicsState {
        views: vec![D3D12_VIEW_INSTANCE_LOCATION {
            ViewportArrayIndex: 1,
            RenderTargetArrayIndex: 0,
        }],
        ..GraphicsState::default()
    };
    let single = graphics.effective_views(|| panic!("no tier requirement for one view"));
    assert_eq!(single, graphics.views);
    assert_ne!(graphics.describe_with_views(single), plain_state);

    graphics.views.push(D3D12_VIEW_INSTANCE_LOCATION {
        ViewportArrayIndex: 0,
        RenderTargetArrayIndex: 1,
    });
    for tier in [Some(D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED), None] {
        let views = graphics.effective_views(|| tier);
        assert!(views.is_empty());
        assert_eq!(graphics.describe_with_views(views), plain_state);
    }
    for tier in [
        D3D12_VIEW_INSTANCING_TIER_1,
        D3D12_VIEW_INSTANCING_TIER_2,
        D3D12_VIEW_INSTANCING_TIER_3,
    ] {
        let views = graphics.effective_views(|| Some(tier));
        assert_eq!(views, graphics.views);
        assert_eq!(graphics.describe_with_views(views), graphics.describe());
    }
    assert_ne!(graphics.describe(), plain_state);
}

#[test]
#[ignore = "requires --features diagnostics, VICTORY_D3D12_WARP=1 and Windows SDK DXC"]
fn static_views_through_create_stream() {
    assert_eq!(std::env::var("VICTORY_D3D12_WARP").as_deref(), Ok("1"));
    let device = &host().expect("WARP device").device;
    // SAFETY: querying the selected adapter and its view instancing tier.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        let desc = adapter.GetDesc1().unwrap();
        assert_ne!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .unwrap();
        assert_ne!(
            options.ViewInstancingTier,
            D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED
        );
        eprintln!(
            "DXGI {:04X}:{:04X}, ViewInstancingTier={}",
            desc.VendorId, desc.DeviceId, options.ViewInstancingTier.0
        );
    }
    let dir = tempfile::tempdir().unwrap();
    library::use_pipeline_cache(dir.path().to_path_buf());
    let dxc = std::env::var_os("VICTORY_PC_DXC")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            victory_shader::emit::validator::Validator::default_path()
                .map(|path| path.with_file_name("dxc.exe"))
        })
        .expect("Windows SDK DXC");
    let shaders = shader_bytes(&dxc, dir.path());
    let across = [(0, 0), (1, 0)];
    let layered = [(0, 0), (0, 1)];
    let none = D3D12_VIEW_INSTANCING_FLAG_NONE;

    let plain_pso = create(&shaders, &[], none);
    let plain = [[16, 0], [0, 0]];
    assert_eq!(draw(make::ready(&plain_pso).expect("plain PSO")), plain);

    let (made, loaded) = super::super::pipeline_counts();
    let across_pso = create(&shaders, &across, none);
    let across_pixels = draw(make::ready(&across_pso).expect("two-viewport PSO"));
    eprintln!("two viewports, red pixels [left, right] by slice: {across_pixels:?}");

    let layered_pso = create(&shaders, &layered, none);
    let layered_pixels = draw(make::ready(&layered_pso).expect("two-slice PSO"));
    eprintln!("two slices, red pixels [left, right] by slice: {layered_pixels:?}");
    eprintln!(
        "title keys: plain={}, viewports={}, slices={}",
        plain_pso.key, across_pso.key, layered_pso.key
    );
    assert_eq!(across_pixels, [[16, 16], [0, 0]]);
    assert_eq!(layered_pixels, [[16, 0], [16, 0]]);
    assert_ne!(across_pso.key, layered_pso.key);
    assert_ne!(plain_pso.key, across_pso.key);
    assert_eq!(super::super::pipeline_counts(), (made + 2, loaded));

    for (routes, first) in [(across, across_pso), (layered, layered_pso)] {
        let again = create(&shaders, &routes, none);
        assert_eq!(again.key, first.key);
        make::ready(&again).expect("library PSO");
    }
    assert_eq!(super::super::pipeline_counts(), (made + 2, loaded + 2));
    eprintln!("both view variants created once and reloaded from the pipeline library");

    let masked = create(
        &shaders,
        &across,
        D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
    );
    assert_eq!(masked.key, plain_pso.key);
    assert_eq!(
        draw(make::ready(&masked).expect("masked view fallback PSO")),
        plain
    );

    let single = create(&shaders, &[(1, 1)], none);
    let single_pixels = draw(make::ready(&single).expect("single explicit view PSO"));
    assert_eq!(single_pixels, [[0, 0], [0, 16]]);
    eprintln!("one explicit view, red pixels [left, right] by slice: {single_pixels:?}");
}
