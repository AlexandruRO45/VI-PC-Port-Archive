//! Synthetic shaders, depth target and readback for a pipeline stream and
//! draws that go through the title entry points.

use super::*;
use crate::object::{S_OK, slot};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};
use windows::core::{GUID, Interface, PCSTR, s};

#[repr(C, align(8))]
struct Depth1(u32, D3D12_DEPTH_STENCIL_DESC1);

#[test]
fn depth_bounds_flag_changes_the_stream_state() {
    let mut g = host::GraphicsState::default();
    let mut depth = Depth1(21, D3D12_DEPTH_STENCIL_DESC1::default());
    let read = |g: &mut host::GraphicsState, d: &Depth1| {
        // SAFETY: a whole kind-21 subobject.
        unsafe { g.read(21, std::ptr::from_ref(d).cast()) };
    };
    read(&mut g, &depth);
    let disabled = g.describe();
    depth.1.DepthBoundsTestEnable = true.into();
    read(&mut g, &depth);
    assert!(
        g.describe() != disabled,
        "enabled bounds must have a different pipeline key"
    );
    depth.1.DepthBoundsTestEnable = false.into();
    read(&mut g, &depth);
    assert_eq!(g.describe(), disabled);
}

fn fullscreen_vertex_shader() -> Arc<Shader> {
    let source = b"float4 main(uint id : SV_VertexID) : SV_Position {
        return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.25, 1);
    }";
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the arguments are live and code is the output pointer.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            s!("vs_5_1"),
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
    }
    let code = code.unwrap();
    // SAFETY: the blob owns these bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    host_shader("vertex", victory_shader::HwStage::Vs, bytes)
}

fn host_shader(
    stage: &'static str,
    hw_stage: victory_shader::HwStage,
    bytes: &[u8],
) -> Arc<Shader> {
    host_shader_used(stage, hw_stage, bytes, Vec::new(), Vec::new(), Vec::new())
}

fn host_shader_used(
    stage: &'static str,
    hw_stage: victory_shader::HwStage,
    bytes: &[u8],
    used: Vec<victory_shader::UsedResource>,
    user_data: Vec<victory_shader::emit::UserData>,
    root_srvs: Vec<u8>,
) -> Arc<Shader> {
    let shader = shader(stage, bytes);
    shader.translated.get_or_init(|| {
        Some(Translated {
            resource_heap: false,
            writes_memory: true,
            host: victory_shader::emit::HostShader {
                stage: hw_stage,
                dxil: bytes.to_vec(),
                user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs,
                root_uavs: Vec::new(),
                stride_table: false,
            },
            used,
        })
    });
    shader
}

#[test]
fn depth_bounds_control_a_title_draw() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped depth bounds render: no host D3D12 device");
        return;
    };
    render_with_depth_bounds(host, fullscreen_vertex_shader(), None, true);
}

/// The triangle mesh and empty pixel shader the simple mesh test compiles.
// WARP removed the device on a depth-only mesh dispatch without a
// pixel shader, even with depth bounds disabled.
const TRIANGLE_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

fn dxc_compile(
    dxc: &std::ffi::OsStr,
    source: &std::path::Path,
    profile: &str,
    entry: &str,
) -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let code = dir.path().join(profile);
    let compiled = std::process::Command::new(dxc)
        .args(["-T", profile, "-E", entry, "-Fo"])
        .arg(&code)
        .arg(source)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    std::fs::read(&code).unwrap()
}

fn simple_mesh_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, TRIANGLE_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

/// Mesh shaders with a per-primitive output: unread by the pixel shader
/// (`perprim-unused`) or read (`perprim-match`). The title's mesh carries
/// per-primitive outputs and RADV ends the process compiling such pairs.
const PERPRIM_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; };
        struct Primitive { float4 color : COLOR0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1],
                  out primitives Primitive prims[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
            prims[0].color = float4(1, 0, 0, 1);
        }
        void ps() {}
    "#;

const PERPRIM_MATCH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; };
        struct Primitive { float4 color : COLOR0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1],
                  out primitives Primitive prims[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
            prims[0].color = float4(1, 0, 0, 1);
        }
        void ps(primitives Primitive p) { if (p.color.x > 2) discard; }
    "#;

/// Mesh vertex outputs beyond SV_Position: unread (`tex-unused`) or read
/// (`tex-match`) by the pixel shader, as the title's mesh exports them.
const TEX_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; float2 tex2 : TEXCOORD1; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float4(0, 0, 0, 0); verts[1].tex = float4(1, 0, 0, 0); verts[2].tex = float4(0, 1, 0, 0);
            verts[0].tex2 = float2(0, 0); verts[1].tex2 = float2(1, 0); verts[2].tex2 = float2(0, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

const TEX_MATCH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; float2 tex2 : TEXCOORD1; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float4(0, 0, 0, 0); verts[1].tex = float4(1, 0, 0, 0); verts[2].tex = float4(0, 1, 0, 0);
            verts[0].tex2 = float2(0, 0); verts[1].tex2 = float2(1, 0); verts[2].tex2 = float2(0, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps(float4 tex : TEXCOORD0, float2 tex2 : TEXCOORD1) { if (tex.x + tex2.y > 2) discard; }
    "#;

fn tex_shaders(matched: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, if matched { TEX_MATCH_HLSL } else { TEX_HLSL }).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

/// A root constant buffer read by the mesh (`cbv-mesh`) or by the pixel
/// shader (`cbv-pixel`): descriptor bindings on a mesh pipeline, which the
/// title's shaders carry and the plain triangle does not.
const CBV_MESH_HLSL: &str = r#"
        cbuffer Root : register(b0) { float4 scale; };
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            if (scale.x > 100) verts[0].position = float4(2, 2, 2, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

const CBV_PIXEL_HLSL: &str = r#"
        cbuffer Root : register(b0) { float4 scale; };
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() { if (scale.x > 2) discard; }
    "#;

/// A bindless descriptor-heap load in the mesh (`heap-mesh`) or the pixel
/// shader (`heap-pixel`), as the title's shaders read their textures.
const HEAP_MESH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            Texture2D t = ResourceDescriptorHeap[0];
            float4 s = t.Load(int3(0, 0, 0));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = s; verts[1].tex = s; verts[2].tex = s;
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

const HEAP_PIXEL_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() { Texture2D t = ResourceDescriptorHeap[0]; if (t.Load(int3(0, 0, 0)).x > 2) discard; }
    "#;

fn heap_shaders(in_mesh: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(
        &source,
        if in_mesh {
            HEAP_MESH_HLSL
        } else {
            HEAP_PIXEL_HLSL
        },
    )
    .unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

/// A root raw-buffer (ByteAddressBuffer) load in the mesh (`srv-mesh`) or
/// the pixel shader (`srv-pixel`): the title reads vertex data this way.
const SRV_MESH_HLSL: &str = r#"
        ByteAddressBuffer data : register(t0);
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            float4 s = asfloat(data.Load4(0));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = s; verts[1].tex = s; verts[2].tex = s;
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

const SRV_PIXEL_HLSL: &str = r#"
        ByteAddressBuffer data : register(t0);
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() { if (asfloat(data.Load(0)) > 2) discard; }
    "#;

/// Wave (subgroup) operations in the mesh (`wave-mesh`): the title's mesh
/// joins per-lane descriptor loads with them.
const WAVE_MESH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(32, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1],
                  uint tid : SV_GroupIndex) {
            SetMeshOutputCounts(3, 1);
            uint lane = WaveGetLaneIndex();
            float s = WaveActiveSum(float(lane));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float4(s, 0, 0, 0); verts[1].tex = float4(s, 0, 0, 0); verts[2].tex = float4(s, 0, 0, 0);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

/// A texture bound at a console register space (`bind3-mesh`: t3 space3)
/// that the synthetic host root signature does not declare: the title's
/// mesh keeps its console spaces, the host root signature uses 0/1/2.
const BIND3_MESH_HLSL: &str = r#"
        Texture2D t : register(t3, space3);
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            float4 s = t.Load(int3(0, 0, 0));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = s; verts[1].tex = s; verts[2].tex = s;
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

/// A bindless raw buffer (ByteAddressBuffer from the descriptor heap) read
/// by the mesh (`heapraw-mesh`) or pixel (`heapraw-pixel`): the title's
/// vertex-data path, heap handle annotated as a raw buffer.
const HEAPRAW_MESH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            ByteAddressBuffer b = ResourceDescriptorHeap[0];
            float4 s = asfloat(b.Load4(0));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = s; verts[1].tex = s; verts[2].tex = s;
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

const HEAPRAW_PIXEL_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() { ByteAddressBuffer b = ResourceDescriptorHeap[0]; if (asfloat(b.Load(0)) > 2) discard; }
    "#;

/// The title mesh's shape (`big-mesh`): 128 threads, 128 verts / 128 prims,
/// looped output stores with dynamic indices, a per-primitive output, a
/// bindless raw-buffer read and a wave join.
const BIG_MESH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; float4 tex2 : TEXCOORD1; };
        struct Prim { float edge : EDGE; };
        [numthreads(128, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[128], out indices uint3 triangles[128],
                  out primitives Prim prims[128], uint tid : SV_GroupIndex) {
            SetMeshOutputCounts(128, 128);
            ByteAddressBuffer b = ResourceDescriptorHeap[0];
            uint first = WaveReadLaneFirst(tid);
            for (uint i = tid; i < 128; i += 128) {
                float4 s = asfloat(b.Load4(i * 16));
                verts[i].position = float4(-1 + float(i) * 0.01, 1, 0.25, 1);
                verts[i].tex = s;
                verts[i].tex2 = s + float(first);
                triangles[i] = uint3(i, (i + 1) % 128, (i + 2) % 128);
                prims[i].edge = float(i);
            }
        }
        void ps() {}
    "#;

/// Partial component masks across the mesh→pixel interface (`mask-mesh`):
/// TEXCOORD0 written xyz, TEXCOORD1 written xy, as the title's pair does.
const MASK_MESH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float3 tex : TEXCOORD1; float2 tex2 : TEXCOORD2; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float3(1, 2, 3); verts[1].tex = float3(1, 2, 3); verts[2].tex = float3(1, 2, 3);
            verts[0].tex2 = float2(4, 5); verts[1].tex2 = float2(4, 5); verts[2].tex2 = float2(4, 5);
            triangles[0] = uint3(0, 1, 2);
        }
        float4 ps(float3 tex : TEXCOORD1, float2 tex2 : TEXCOORD2) : SV_Target {
            return float4(tex, tex2.x) + tex2.y;
        }
    "#;

/// Wave joins in the mesh and helper-lane/derivative work in the pixel
/// (`waveboth`): both title stages carry subgroup operations.
const WAVEBOTH_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(32, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1],
                  uint tid : SV_GroupIndex) {
            SetMeshOutputCounts(3, 1);
            float s = WaveActiveSum(float(WaveGetLaneIndex()));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float4(s, 0, 0, 0); verts[1].tex = float4(s, 0, 0, 0); verts[2].tex = float4(s, 0, 0, 0);
            triangles[0] = uint3(0, 1, 2);
        }
        float4 ps(float4 tex : TEXCOORD0) : SV_Target {
            float d = ddx(tex.x) + ddy(tex.y);
            float w = WaveReadLaneFirst(d);
            return tex + w;
        }
    "#;

/// The title pixel's op mix (`fullpixel`): heap texture sample, discard,
/// derivatives and a wave join, paired with a wave-joined mesh.
const FULLPIXEL_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(32, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1],
                  uint tid : SV_GroupIndex) {
            SetMeshOutputCounts(3, 1);
            float s = WaveActiveSum(float(WaveGetLaneIndex()));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float4(s, 0, 0, 0); verts[1].tex = float4(s, 0, 0, 0); verts[2].tex = float4(s, 0, 0, 0);
            triangles[0] = uint3(0, 1, 2);
        }
        SamplerState samp : register(s0);
        float4 ps(float4 tex : TEXCOORD0) : SV_Target {
            Texture2D t = ResourceDescriptorHeap[0];
            float4 c = t.Sample(samp, tex.xy);
            float d = ddx(c.x) + ddy(c.y);
            float w = WaveReadLaneFirst(d);
            if (c.w + w < 0) discard;
            return c + tex;
        }
    "#;

/// The title pixel's whole op mix (`allpixel`): heap raw-buffer loads, a
/// heap texture sample, a legacy cbuffer, discard, derivatives and a wave
/// join, over the title's TEXCOORD1/TEXCOORD2 interface.
const ALLPIXEL_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float3 tex : TEXCOORD1; float2 tex2 : TEXCOORD2; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float3(1, 2, 3); verts[1].tex = float3(1, 2, 3); verts[2].tex = float3(1, 2, 3);
            verts[0].tex2 = float2(4, 5); verts[1].tex2 = float2(4, 5); verts[2].tex2 = float2(4, 5);
            triangles[0] = uint3(0, 1, 2);
        }
        cbuffer C : register(b0) { float4 k; };
        SamplerState samp : register(s0);
        float4 ps(float3 tex : TEXCOORD1, float2 tex2 : TEXCOORD2) : SV_Target {
            ByteAddressBuffer b = ResourceDescriptorHeap[1];
            float4 r = asfloat(b.Load4(uint(tex.x) * 16));
            Texture2D t = ResourceDescriptorHeap[0];
            float4 c = t.Sample(samp, tex2);
            float d = ddx(c.x) + ddy(c.y);
            float w = WaveReadLaneFirst(d);
            if (c.w + w + r.x + k.x < 0) discard;
            return c + float4(tex, tex2.x) + r + w;
        }
    "#;

/// An integer wave join in the pixel (`wavei32`): the title's joins are
/// `WaveReadLaneFirst` on i32, where the synthetic ones were float.
const WAVEI32_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float3 tex : TEXCOORD1; float2 tex2 : TEXCOORD2; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float3(1, 2, 3); verts[1].tex = float3(1, 2, 3); verts[2].tex = float3(1, 2, 3);
            verts[0].tex2 = float2(4, 5); verts[1].tex2 = float2(4, 5); verts[2].tex2 = float2(4, 5);
            triangles[0] = uint3(0, 1, 2);
        }
        float4 ps(float3 tex : TEXCOORD1, float2 tex2 : TEXCOORD2) : SV_Target {
            int w = WaveReadLaneFirst(int(tex.x));
            return float4(tex, tex2.x) + float(w);
        }
    "#;

/// A wave join inside divergent control flow (`wavediv`): the title's
/// joins sit under helper-lane guards, not in uniform flow.
const WAVEDIV_HLSL: &str = r#"
        struct Vertex { float4 position : SV_Position; float3 tex : TEXCOORD1; float2 tex2 : TEXCOORD2; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = float3(1, 2, 3); verts[1].tex = float3(1, 2, 3); verts[2].tex = float3(1, 2, 3);
            verts[0].tex2 = float2(4, 5); verts[1].tex2 = float2(4, 5); verts[2].tex2 = float2(4, 5);
            triangles[0] = uint3(0, 1, 2);
        }
        float4 ps(float3 tex : TEXCOORD1, float2 tex2 : TEXCOORD2) : SV_Target {
            float4 o = float4(tex, tex2.x);
            if (tex.x > 0) {
                int w = WaveReadLaneFirst(int(tex.y));
                o += float(w);
                if (tex2.y < 0) discard;
            }
            return o;
        }
    "#;

fn wavediv_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, WAVEDIV_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn wavei32_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, WAVEI32_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn allpixel_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, ALLPIXEL_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn fullpixel_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, FULLPIXEL_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn waveboth_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, WAVEBOTH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn mask_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, MASK_MESH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn big_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, BIG_MESH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn heapraw_shaders(in_mesh: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(
        &source,
        if in_mesh {
            HEAPRAW_MESH_HLSL
        } else {
            HEAPRAW_PIXEL_HLSL
        },
    )
    .unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn bind3_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, BIND3_MESH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

/// Both at once (`heapbind3-mesh`): a bindless heap load and an undeclared
/// console-space binding, as the title's mesh carries.
const HEAPBIND3_MESH_HLSL: &str = r#"
        Texture2D t : register(t3, space3);
        struct Vertex { float4 position : SV_Position; float4 tex : TEXCOORD0; };
        [numthreads(1, 1, 1)] [outputtopology("triangle")]
        void main(out vertices Vertex verts[3], out indices uint3 triangles[1]) {
            SetMeshOutputCounts(3, 1);
            Texture2D h = ResourceDescriptorHeap[0];
            float4 s = t.Load(int3(0, 0, 0)) + h.Load(int3(0, 0, 0));
            verts[0].position = float4(-1,  1, 0.25, 1);
            verts[1].position = float4( 3,  1, 0.25, 1);
            verts[2].position = float4(-1, -3, 0.25, 1);
            verts[0].tex = s; verts[1].tex = s; verts[2].tex = s;
            triangles[0] = uint3(0, 1, 2);
        }
        void ps() {}
    "#;

fn heapbind3_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, HEAPBIND3_MESH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn wave_shaders() -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(&source, WAVE_MESH_HLSL).unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

fn srv_shaders(in_mesh: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(
        &source,
        if in_mesh {
            SRV_MESH_HLSL
        } else {
            SRV_PIXEL_HLSL
        },
    )
    .unwrap();
    let srv = victory_shader::UsedResource {
        kind: victory_shader::ResourceType::Srv,
        root: victory_shader::RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let param = |index: u32| victory_shader::emit::UserData {
        index,
        slot: 0,
        root: victory_shader::RootType::Descriptor,
        dword: index,
    };
    let ms_bytes = dxc_compile(&dxc, &source, "ms_6_6", "main");
    let ps_bytes = dxc_compile(&dxc, &source, "ps_6_6", "ps");
    (
        host_shader_used(
            "mesh",
            victory_shader::HwStage::Ms,
            &ms_bytes,
            if in_mesh { vec![srv] } else { Vec::new() },
            if in_mesh {
                vec![param(0), param(1)]
            } else {
                Vec::new()
            },
            if in_mesh { vec![0] } else { Vec::new() },
        ),
        host_shader_used(
            "pixel",
            victory_shader::HwStage::Ps,
            &ps_bytes,
            if in_mesh { Vec::new() } else { vec![srv] },
            if in_mesh {
                Vec::new()
            } else {
                vec![param(0), param(1)]
            },
            if in_mesh { Vec::new() } else { vec![0] },
        ),
    )
}

fn cbv_shaders(in_mesh: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(
        &source,
        if in_mesh {
            CBV_MESH_HLSL
        } else {
            CBV_PIXEL_HLSL
        },
    )
    .unwrap();
    let cbv = victory_shader::UsedResource {
        kind: victory_shader::ResourceType::Cbv,
        root: victory_shader::RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    };
    let param = |index: u32| victory_shader::emit::UserData {
        index,
        slot: 0,
        root: victory_shader::RootType::Descriptor,
        dword: index,
    };
    let ms_bytes = dxc_compile(&dxc, &source, "ms_6_6", "main");
    let ps_bytes = dxc_compile(&dxc, &source, "ps_6_6", "ps");
    (
        host_shader_used(
            "mesh",
            victory_shader::HwStage::Ms,
            &ms_bytes,
            if in_mesh { vec![cbv] } else { Vec::new() },
            if in_mesh {
                vec![param(0), param(1)]
            } else {
                Vec::new()
            },
            Vec::new(),
        ),
        host_shader_used(
            "pixel",
            victory_shader::HwStage::Ps,
            &ps_bytes,
            if in_mesh { Vec::new() } else { vec![cbv] },
            if in_mesh {
                Vec::new()
            } else {
                vec![param(0), param(1)]
            },
            Vec::new(),
        ),
    )
}

fn perprim_shaders(matched: bool) -> (Arc<Shader>, Arc<Shader>) {
    let dxc = std::env::var_os("VICTORY_PC_DXC").expect("VICTORY_PC_DXC");
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("triangle.hlsl");
    std::fs::write(
        &source,
        if matched {
            PERPRIM_MATCH_HLSL
        } else {
            PERPRIM_HLSL
        },
    )
    .unwrap();
    (
        host_shader(
            "mesh",
            victory_shader::HwStage::Ms,
            &dxc_compile(&dxc, &source, "ms_6_6", "main"),
        ),
        host_shader(
            "pixel",
            victory_shader::HwStage::Ps,
            &dxc_compile(&dxc, &source, "ps_6_6", "ps"),
        ),
    )
}

#[test]
fn depth_bounds_control_a_title_mesh_dispatch() {
    let Some(host) = crate::host::host().filter(|h| h.mesh_shaders()) else {
        eprintln!("skipped depth bounds mesh render: no host mesh shaders");
        return;
    };
    if std::env::var_os("VICTORY_PC_DXC").is_none() {
        eprintln!("skipped depth bounds mesh render: VICTORY_PC_DXC is unset");
        return;
    };
    let (ms, ps) = simple_mesh_shaders();
    render_with_depth_bounds(host, ms, Some(ps), true);
}

/// The title's own mesh and pixel shaders, dumped with VICTORY_SHADER_DUMP
/// and named by VICTORY_MESH_CONSOLE / VICTORY_PIXEL_CONSOLE (console blobs,
/// translated here) or VICTORY_MESH_HOST / VICTORY_PIXEL_HOST (already
/// translated host DXIL, injected). Skipped without the blobs: the simple
/// mesh above passes under Wine where the title's pair ends the process.
#[test]
fn depth_bounds_control_a_title_mesh_blob() {
    let Some(host) = crate::host::host().filter(|h| h.mesh_shaders()) else {
        eprintln!("skipped title mesh blob: no host mesh shaders");
        return;
    };
    if std::env::var_os("VICTORY_MESH_CONSOLE").is_none()
        && std::env::var_os("VICTORY_MESH_HOST").is_none()
        && std::env::var_os("VICTORY_MESH_SIMPLE").is_none()
    {
        eprintln!("skipped title mesh blob: no VICTORY_MESH_CONSOLE/HOST/SIMPLE");
        return;
    }
    let blob = |console: &str, host: &str, stage: &'static str, hw| {
        if let Some(path) = std::env::var_os(host) {
            let bytes = std::fs::read(path).expect("host blob");
            return host_shader(stage, hw, &bytes);
        }
        let path = std::env::var_os(console).expect("VICTORY_MESH_CONSOLE or VICTORY_MESH_HOST");
        let bytes = std::fs::read(path).expect("console blob");
        let shader = shader(stage, &bytes);
        assert!(
            shader.translated().is_some(),
            "{stage} blob does not translate"
        );
        shader
    };
    let ms = if std::env::var_os("VICTORY_MESH_SIMPLE").is_some() {
        match std::env::var("VICTORY_MESH_VARIANT").as_deref() {
            Ok("perprim-unused") => perprim_shaders(false).0,
            Ok("perprim-match") => perprim_shaders(true).0,
            Ok("tex-unused") => tex_shaders(false).0,
            Ok("tex-match") => tex_shaders(true).0,
            Ok("cbv-mesh") => cbv_shaders(true).0,
            Ok("cbv-pixel") => cbv_shaders(false).0,
            Ok("heap-mesh") => heap_shaders(true).0,
            Ok("heap-pixel") => heap_shaders(false).0,
            Ok("srv-mesh") => srv_shaders(true).0,
            Ok("srv-pixel") => srv_shaders(false).0,
            Ok("wave-mesh") => wave_shaders().0,
            Ok("bind3-mesh") => bind3_shaders().0,
            Ok("heapbind3-mesh") => heapbind3_shaders().0,
            Ok("heapraw-mesh") => heapraw_shaders(true).0,
            Ok("heapraw-pixel") => heapraw_shaders(false).0,
            Ok("big-mesh") => big_shaders().0,
            Ok("mask-mesh") => mask_shaders().0,
            Ok("waveboth") => waveboth_shaders().0,
            Ok("fullpixel") => fullpixel_shaders().0,
            Ok("allpixel") => allpixel_shaders().0,
            Ok("wavei32") => wavei32_shaders().0,
            Ok("wavediv") => wavediv_shaders().0,
            _ => simple_mesh_shaders().0,
        }
    } else {
        blob(
            "VICTORY_MESH_CONSOLE",
            "VICTORY_MESH_HOST",
            "mesh",
            victory_shader::HwStage::Ms,
        )
    };
    if let Some(out) = std::env::var_os("VICTORY_MESH_HOST_OUT") {
        std::fs::write(out, &ms.translated().unwrap().host.dxil).expect("host dump");
    }
    let ps = if std::env::var_os("VICTORY_PIXEL_SIMPLE").is_some() {
        match std::env::var("VICTORY_MESH_VARIANT").as_deref() {
            Ok("perprim-unused") => Some(perprim_shaders(false).1),
            Ok("perprim-match") => Some(perprim_shaders(true).1),
            Ok("tex-unused") => Some(tex_shaders(false).1),
            Ok("tex-match") => Some(tex_shaders(true).1),
            Ok("cbv-mesh") => Some(cbv_shaders(true).1),
            Ok("cbv-pixel") => Some(cbv_shaders(false).1),
            Ok("heap-mesh") => Some(heap_shaders(true).1),
            Ok("heap-pixel") => Some(heap_shaders(false).1),
            Ok("srv-mesh") => Some(srv_shaders(true).1),
            Ok("srv-pixel") => Some(srv_shaders(false).1),
            Ok("wave-mesh") => Some(wave_shaders().1),
            Ok("bind3-mesh") => Some(bind3_shaders().1),
            Ok("heapbind3-mesh") => Some(heapbind3_shaders().1),
            Ok("heapraw-mesh") => Some(heapraw_shaders(true).1),
            Ok("heapraw-pixel") => Some(heapraw_shaders(false).1),
            Ok("big-mesh") => Some(big_shaders().1),
            Ok("mask-mesh") => Some(mask_shaders().1),
            Ok("waveboth") => Some(waveboth_shaders().1),
            Ok("fullpixel") => Some(fullpixel_shaders().1),
            Ok("allpixel") => Some(allpixel_shaders().1),
            Ok("wavei32") => Some(wavei32_shaders().1),
            Ok("wavediv") => Some(wavediv_shaders().1),
            _ => Some(simple_mesh_shaders().1),
        }
    } else {
        std::env::var_os("VICTORY_PIXEL_CONSOLE")
            .or_else(|| std::env::var_os("VICTORY_PIXEL_HOST"))
            .map(|_| {
                blob(
                    "VICTORY_PIXEL_CONSOLE",
                    "VICTORY_PIXEL_HOST",
                    "pixel",
                    victory_shader::HwStage::Ps,
                )
            })
    };
    if let (Some(out), Some(ps)) = (std::env::var_os("VICTORY_PIXEL_HOST_OUT"), &ps) {
        std::fs::write(out, &ps.translated().unwrap().host.dxil).expect("host dump");
    }
    // The title's mesh reads root bindings this harness does not supply, so
    // it may raster nothing; the point here is that the pair compiles and
    // dispatches, which is what the multi-`ret` DXIL used to crash.
    render_with_depth_bounds(host, ms, ps, false);
}

fn render_with_depth_bounds(
    host: &crate::host::Host,
    shader: Arc<Shader>,
    ps: Option<Arc<Shader>>,
    check_depth: bool,
) {
    let device = &host.device;
    // SAFETY: plain adapter queries on a live device.
    let adapter = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        adapter.GetDesc1().unwrap()
    };
    let name = String::from_utf16_lossy(&adapter.Description);
    let supported = host.depth_bounds_supported;
    let mesh = shader.stage == "mesh";
    eprintln!(
        "{} depth bounds on {}: {supported}",
        shader.stage,
        name.trim_end_matches('\0')
    );

    #[repr(C)]
    struct Stream {
        shader_kind: u64,
        shader: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth1,
        topology: [u32; 2],
        dsv: [u32; 2],
    }
    let stream = Stream {
        shader_kind: if mesh { 24 } else { 1 },
        shader: Bytecode {
            code: shader.container.as_ptr(),
            size: shader.container.len(),
        },
        ps_kind: 2,
        ps: ps.as_ref().map_or(
            Bytecode {
                code: std::ptr::null(),
                size: 0,
            },
            |ps| Bytecode {
                code: ps.container.as_ptr(),
                size: ps.container.len(),
            },
        ),
        depth: Depth1(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        ),
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        dsv: [16, DXGI_FORMAT_D32_FLOAT.0 as u32],
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let null = std::ptr::null();
    let mut pso = null;
    assert_eq!(
        create_stream(null, &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    assert_eq!(
        host_pipeline(pso)
            .expect("host refused the depth bounds pipeline")
            .mesh,
        mesh
    );

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
            &mut list
        ),
        S_OK
    );
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;
    let newer = crate::list::has_newer_table(list);
    let bounds =
        slot::<extern "system" fn(This, f32, f32)>(list, if newer { 0x230 } else { 0x220 });
    let scissors = slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0);
    let draw = slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60);
    let draw_pixel = |x| {
        scissors(
            list,
            1,
            &RECT {
                left: x,
                top: 0,
                right: x + 1,
                bottom: 1,
            },
        );
        if mesh {
            let dispatch = slot::<extern "system" fn(This, u32, u32, u32)>(
                list,
                if newer { 0x408 } else { 0x3f8 },
            );
            dispatch(list, 1, 1, 1);
        } else {
            draw(list, 3, 1, 0, 0);
        }
    };
    let depth_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let buffer_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 256,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: depth_desc.SampleDesc,
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let resource = |desc: &D3D12_RESOURCE_DESC, kind, state| {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let mut result: Option<ID3D12Resource> = None;
        // SAFETY: valid resource descriptions and output pointer.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    None,
                    &mut result,
                )
                .unwrap();
        }
        result.unwrap()
    };
    let depth = resource(
        &depth_desc,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
    );
    let readback = resource(
        &buffer_desc,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let dsv = host.dsv(&depth, None);
    let bind = |l: &ID3D12GraphicsCommandList| {
        // SAFETY: an open list and the live depth buffer's descriptor.
        unsafe { l.OMSetRenderTargets(0, None, false, Some(&dsv)) };
        slot::<extern "system" fn(This, This)>(list, 0xc8)(list, pso);
        if !mesh {
            slot::<extern "system" fn(This, u32)>(list, 0xa0)(list, 4); // triangle list
        }
        slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(
            list,
            1,
            &D3D12_VIEWPORT {
                Width: 4.0,
                Height: 1.0,
                MaxDepth: 1.0,
                ..Default::default()
            },
        );
    };
    let submit = |commands: &crate::host::HostList| {
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        host.execute(&[commands]);
    };
    // SAFETY: the list is open; the depth buffer is in DEPTH_WRITE.
    unsafe { l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None) };
    bind(l);
    draw_pixel(0); // Default bounds on a fresh list.
    bounds(list, 0.0, 0.25);
    draw_pixel(1); // Excludes the stored depth, includes the triangle depth.
    bounds(list, 0.5, 0.75);
    draw_pixel(2); // Includes the stored depth, excludes the triangle depth.
    bounds(list, 0.0, 0.25);
    submit(&commands);

    assert_eq!(
        slot::<extern "system" fn(This, This, This) -> HRESULT>(list, 0x50)(list, allocator, null),
        S_OK
    );
    // Resetting a list whose host list is still referenced (here by
    // `commands`) records on into a fresh host list.
    let commands = crate::list::host_list(list).unwrap();
    let l = &commands.list;
    bind(l);
    draw_pixel(3); // Reset restores the default bounds.
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
                    Format: depth_desc.Format,
                    Width: depth_desc.Width as u32,
                    Height: 1,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        },
        ..Default::default()
    };
    // SAFETY: the locations borrow live resources, and the copy covers the whole subresource.
    unsafe {
        from.pResource = std::mem::transmute_copy(&depth);
        to.pResource = std::mem::transmute_copy(&readback);
        l.ResourceBarrier(&[crate::host::transition(
            &depth,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    submit(&commands);
    let mut mapped = std::ptr::null_mut();
    // SAFETY: the GPU copy finished and the readback buffer holds at least four floats.
    let actual = unsafe {
        readback
            .Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 16 }),
                Some(&mut mapped),
            )
            .unwrap();
        let values = mapped.cast::<[f32; 4]>().read();
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
        values
    };
    eprintln!("depth readback (fresh, excluded, included, reset): {actual:?}");
    if check_depth {
        assert_eq!(
            actual,
            [0.25, if supported { 0.5 } else { 0.25 }, 0.25, 0.25]
        );
    }

    for object in [list, allocator, pso] {
        assert_eq!(
            slot::<extern "system" fn(This) -> u32>(object, 0x10)(object),
            0
        );
    }
}
