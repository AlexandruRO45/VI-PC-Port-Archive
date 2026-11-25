//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{
    MEMORY_SPACE, MESH_RECORD_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE,
};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    fallback: bool,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// Our root parameters of the title-memory page table and of the
    /// written-pages bitmap (`crate::memory`), if it reads or writes there.
    pub(crate) memory: Option<u32>,
    pub(crate) memory_dirty: Option<u32>,
    /// Our root parameter of the heap's view addresses (`crate::memory`).
    pub(crate) view_addresses: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        if self.fallback {
            &self
                .shader
                .mesh_fallback()
                .expect("translated mesh fallback")
                .0
        } else {
            self.shader.translated().expect("translated host shader")
        }
    }
}

pub(crate) struct MeshFallback {
    pub(crate) compute: Box<HostPipeline>,
    pub(crate) layout: victory_shader::emit::MeshRecordLayout,
    pub(crate) uav: u32,
    pub(crate) base: u32,
    pub(crate) srv: u32,
    pub(crate) dispatch: ID3D12CommandSignature,
}

/// Compute PSO and its bindings survive a deferred graphics cache miss.
pub(super) struct MeshPass1 {
    compute: Box<HostPipeline>,
    uav: u32,
    base: u32,
    dispatch: ID3D12CommandSignature,
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) mesh_fallback: Option<MeshFallback>,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// A graphics pipeline of the console's rectangle lists whose geometry
    /// shader completes each rectangle (`victory_shader::emit::rectangle_gs`):
    /// it draws them as triangles, one per rectangle.
    pub(crate) rectangles: bool,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

fn pixel_features_supported(dxil: &[u8], barycentrics: impl FnOnce() -> bool) -> bool {
    !victory_shader::emit::requires_barycentrics(dxil) || barycentrics()
}

#[cfg(test)]
#[test]
fn native_barycentrics_need_the_device_feature() {
    use victory_shader::{Container, FourCC, emit::host::stub_ps};

    let mut ps = stub_ps();
    assert!(pixel_features_supported(&ps.dxil, || panic!(
        "plain PS queried OPTIONS3"
    )));
    let container = Container::parse(&ps.dxil).unwrap();
    let bit = container.part(FourCC::SFI0).unwrap().offset + 8 + 2;
    ps.dxil[bit] |= 2;
    assert!(!pixel_features_supported(&ps.dxil, || false));
    assert!(pixel_features_supported(&ps.dxil, || true));
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let mut compile = super::make::Compile::new(super::make::CompileMode::Sync);
        Self::build(shaders, g, &mut compile)
    }

    /// The same descriptor and cache path for synchronous prebuild and workers.
    pub(super) fn build(
        shaders: &[Arc<Shader>],
        g: Option<&GraphicsState>,
        compile: &mut super::make::Compile,
    ) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        let host = host()?;
        if let Some(ps) = find("pixel")
            && !pixel_features_supported(&ps.translated()?.host.dxil, || {
                host.barycentrics_supported()
            })
        {
            return refuse(
                "SV_Barycentrics requires D3D12_OPTIONS3.BarycentricsSupported",
                shaders,
            );
        }
        if let Some(ms) = mesh
            && !host.mesh_shaders()
        {
            return mesh_fallback(ms, find("pixel"), g?, shaders, compile);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        // `SFI0`: 64-bit atomics on typed resources and on heap resources.
        let atomics64 = stages.iter().any(|(s, _)| {
            let Some(t) = s.translated() else {
                return false;
            };
            victory_shader::Container::parse(&t.host.dxil)
                .ok()
                .and_then(|c| {
                    c.part(victory_shader::FourCC::SFI0)?
                        .data
                        .get(..4)?
                        .try_into()
                        .ok()
                })
                .is_some_and(|b: [u8; 4]| u32::from_le_bytes(b) & 0x1040_0000 != 0)
        });
        if atomics64 && !host.heap_atomics64() {
            return refuse(
                "the host GPU has no 64-bit atomics on typed heap resources",
                shaders,
            );
        }

        // The root signature the translated shaders' bindings define.
        let mut params = Vec::new();
        let mut samplers = Vec::new();
        let bound: Vec<_> = stages
            .iter()
            .map(|&(shader, visibility)| {
                stage_params(shader, false, visibility, &mut params, &mut samplers)
            })
            .collect::<Option<_>>()?;
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        Self::create_pso(
            shaders,
            g,
            &stages,
            bound,
            root,
            root_blob,
            compute.is_some(),
            mesh.is_some(),
            compile,
        )
    }
}

fn stage_params(
    shader: &Arc<Shader>,
    fallback: bool,
    visibility: D3D12_SHADER_VISIBILITY,
    params: &mut Vec<D3D12_ROOT_PARAMETER1>,
    samplers: &mut Vec<D3D12_STATIC_SAMPLER_DESC>,
) -> Option<Stage> {
    let t = &if fallback {
        &shader.mesh_fallback()?.0
    } else {
        shader.translated()?
    }
    .host;
    let mut stage = Stage {
        shader: shader.clone(),
        fallback,
        user_data: None,
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: None,
        memory_dirty: None,
        view_addresses: None,
    };
    let dwords = t.user_data_dwords();
    if dwords > 0 {
        stage.user_data = Some((params.len() as u32, dwords));
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: dwords,
                },
            },
            ShaderVisibility: visibility,
        });
    }
    for (slots, kind, out) in [
        (
            &t.root_cbvs,
            D3D12_ROOT_PARAMETER_TYPE_CBV,
            &mut stage.root_cbvs,
        ),
        (
            &t.root_srvs,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            &mut stage.root_srvs,
        ),
        (
            &t.root_uavs,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            &mut stage.root_uavs,
        ),
    ] {
        for &slot in slots {
            out.push((slot, params.len() as u32));
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: kind,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: u32::from(slot),
                        RegisterSpace: ROOT_CBV_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: visibility,
            });
        }
    }
    for (used, kind, register, out) in [
        (
            t.memory,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            0,
            &mut stage.memory,
        ),
        (
            t.memory_writes,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            0,
            &mut stage.memory_dirty,
        ),
        (
            t.view_addresses,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            1,
            &mut stage.view_addresses,
        ),
    ] {
        if used {
            *out = Some(params.len() as u32);
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: kind,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: register,
                        RegisterSpace: MEMORY_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: visibility,
            });
        }
    }
    for (n, s) in t.static_samplers.iter().enumerate() {
        samplers.push(D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER(s.filter as i32),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
            AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
            AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
            MipLODBias: s.mip_lod_bias,
            MaxAnisotropy: s.max_anisotropy,
            ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
            BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
            MinLOD: s.min_lod,
            MaxLOD: s.max_lod,
            ShaderRegister: n as u32,
            RegisterSpace: SAMPLER_SPACE,
            ShaderVisibility: visibility,
        });
    }
    Some(stage)
}

impl HostPipeline {
    #[allow(clippy::too_many_arguments)]
    fn create_pso(
        shaders: &[Arc<Shader>],
        g: Option<&GraphicsState>,
        stages: &[(&Arc<Shader>, D3D12_SHADER_VISIBILITY)],
        bound: Vec<Stage>,
        root: ID3D12RootSignature,
        root_blob: Vec<u8>,
        compute: bool,
        mesh: bool,
        compile: &mut super::make::Compile,
    ) -> Option<Self> {
        let host = host()?;
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                desc.CS.BytecodeLength,
                compile,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso?,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                mesh_fallback: None,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if mesh {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso =
                match mesh_pipeline(&host.device, &root, [code(0), code(1)], g, &name, compile) {
                    Ok(pso) => pso?,
                    Err(e) => {
                        tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                        host.log_messages();
                        return None;
                    }
                };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                mesh_fallback: None,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
            });
        }

        // Rectangle lists: a geometry shader made for the vertex shader adds
        // each rectangle's fourth corner (else rectangles are expanded on
        // the CPU from vertex inputs, see `draw`).
        let gs = match (
            g.topology.0,
            stages.first().and_then(|(s, _)| s.translated()),
        ) {
            (RECTANGLES, Some(vs)) => match victory_shader::emit::rectangle_gs(&vs.host.dxil) {
                Ok(gs) => gs,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "no rectangle geometry shader");
                    Vec::new()
                }
            },
            _ => Vec::new(),
        };
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            GS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: if gs.is_empty() {
                    std::ptr::null()
                } else {
                    gs.as_ptr().cast()
                },
                BytecodeLength: gs.len(),
            },
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            PrimitiveTopologyType: host_topology(g),
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs],
            &state,
        );
        let started = std::time::Instant::now();
        let made = super::library::pipeline(
            &name,
            desc.VS
                .BytecodeLength
                .saturating_add(desc.PS.BytecodeLength)
                .saturating_add(desc.GS.BytecodeLength),
            compile,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso?,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(
            ?shaders,
            ms = started.elapsed().as_millis(),
            rectangles = !gs.is_empty(),
            "host pipeline"
        );
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            mesh_fallback: None,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            rectangles: !gs.is_empty(),
            slots: g.slots(),
            inputs: g.elements(),
        })
    }
}

fn mesh_fallback(
    ms: &Arc<Shader>,
    ps: Option<&Arc<Shader>>,
    g: &GraphicsState,
    shaders: &[Arc<Shader>],
    compile: &mut super::make::Compile,
) -> Option<HostPipeline> {
    let (cs, layout) = ms.mesh_fallback()?;
    let pixel = match ps {
        Some(ps) => Some(ps.translated()?),
        None => None,
    };
    let vs =
        victory_shader::emit::emit_passthrough_vs(layout, pixel.map(|p| p.host.dxil.as_slice()))
            .map_err(|error| tracing::warn!(%error, ?ms, "mesh fallback VS refused"))
            .ok()?;
    let atomics = |dxil: &[u8]| {
        victory_shader::Container::parse(dxil)
            .ok()
            .and_then(|c| {
                c.part(victory_shader::FourCC::SFI0)?
                    .data
                    .get(..4)?
                    .try_into()
                    .ok()
            })
            .is_some_and(|b: [u8; 4]| u32::from_le_bytes(b) & 0x1040_0000 != 0)
    };
    let host = host()?;
    if (atomics(&cs.host.dxil) || pixel.is_some_and(|p| atomics(&p.host.dxil)))
        && !host.heap_atomics64()
    {
        return refuse("mesh fallback needs host 64-bit heap atomics", shaders);
    }
    let device = &host.device;
    let name = format!("mesh fallback {shaders:?}");
    let code = |dxil: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: dxil.as_ptr().cast(),
        BytecodeLength: dxil.len(),
    };
    let record_view = |kind, visibility| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        ShaderVisibility: visibility,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                RegisterSpace: MESH_RECORD_SPACE,
                ShaderRegister: 0,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
    };
    let make = || -> windows::core::Result<Option<HostPipeline>> {
        let pass1 = if let Some(pass1) = compile.mesh_pass1.take() {
            pass1
        } else {
            let mut params = Vec::new();
            let mut samplers = Vec::new();
            let cs_stage = stage_params(
                ms,
                true,
                D3D12_SHADER_VISIBILITY_ALL,
                &mut params,
                &mut samplers,
            )
            .expect("fallback translated above");
            let uav = params.len() as u32;
            params.push(record_view(
                D3D12_ROOT_PARAMETER_TYPE_UAV,
                D3D12_SHADER_VISIBILITY_ALL,
            ));
            let base = params.len() as u32;
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: MESH_RECORD_SPACE,
                        Num32BitValues: 1,
                    },
                },
            });
            let (cs_root, cs_blob) = root_signature(device, &params, &samplers, true)?;
            let cs_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: unsafe { std::mem::transmute_copy(&cs_root) },
                CS: code(&cs.host.dxil),
                ..Default::default()
            };
            let key = super::library::name("mesh-fallback-cs", &cs_blob, &[&cs.host.dxil], "");
            // SAFETY: the descriptions point at live bytecode and root signatures.
            let Some(cs_pso) = super::library::pipeline(
                &key,
                cs.host.dxil.len(),
                compile,
                |library, name| unsafe { library.LoadComputePipeline(name, &cs_desc) },
                || unsafe { device.CreateComputePipelineState(&cs_desc) },
            )?
            else {
                return Ok(None);
            };
            #[cfg(feature = "diagnostics")]
            super::make::log_built(&super::make::key(shaders, Some(g)));
            let args = [
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
                    Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                        Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                            RootParameterIndex: base,
                            DestOffsetIn32BitValues: 0,
                            Num32BitValuesToSet: 1,
                        },
                    },
                },
                D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                    ..Default::default()
                },
            ];
            let mut dispatch = None;
            // SAFETY: the signature's constant belongs to this live compute root.
            unsafe {
                device.CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 16,
                        NumArgumentDescs: args.len() as u32,
                        pArgumentDescs: args.as_ptr(),
                        NodeMask: 0,
                    },
                    &cs_root,
                    &mut dispatch,
                )?
            };
            let dispatch = dispatch.ok_or_else(windows::core::Error::empty)?;
            let compute = Box::new(HostPipeline {
                compute: true,
                mesh: false,
                mesh_fallback: None,
                pso: cs_pso,
                root: cs_root,
                stages: vec![cs_stage],
                rectangles: false,
                slots: 0,
                inputs: Vec::new(),
                name: name.clone(),
            });
            MeshPass1 {
                compute,
                uav,
                base,
                dispatch,
            }
        };
        let mut params = Vec::new();
        let mut samplers = Vec::new();
        let stages: Vec<_> = ps
            .map(|ps| {
                stage_params(
                    ps,
                    false,
                    D3D12_SHADER_VISIBILITY_PIXEL,
                    &mut params,
                    &mut samplers,
                )
                .expect("pixel translated above")
            })
            .into_iter()
            .collect();
        let srv = params.len() as u32;
        params.push(record_view(
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            D3D12_SHADER_VISIBILITY_VERTEX,
        ));
        let (root, blob) = root_signature(device, &params, &samplers, false)?;
        let ps_bytes = pixel.map_or(&[][..], |p| p.host.dxil.as_slice());
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(&vs.dxil),
            PS: if ps_bytes.is_empty() {
                Default::default()
            } else {
                code(ps_bytes)
            },
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let key = super::library::name(
            "mesh-fallback-gfx",
            &blob,
            &[&vs.dxil, ps_bytes],
            &g.describe(),
        );
        let Some(pso) = super::library::pipeline(
            &key,
            vs.dxil.len() + ps_bytes.len(),
            compile,
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            || unsafe { device.CreateGraphicsPipelineState(&desc) },
        )?
        else {
            compile.mesh_pass1 = Some(pass1);
            return Ok(None);
        };
        let MeshPass1 {
            compute,
            uav,
            base,
            dispatch,
        } = pass1;
        Ok(Some(HostPipeline {
            compute: false,
            mesh: true,
            pso,
            root,
            stages,
            rectangles: false,
            slots: 0,
            inputs: Vec::new(),
            name,
            mesh_fallback: Some(MeshFallback {
                compute,
                layout: layout.clone(),
                uav,
                base,
                srv,
                dispatch,
            }),
        }))
    };
    make()
        .map_err(|error| {
            tracing::warn!(%error, ?shaders, "host mesh fallback pipeline refused");
            host.log_messages();
        })
        .ok()
        .flatten()
}

/// The host's topology type: the console's rectangle and quad types draw
/// as triangles on the host (see `draw`).
fn host_topology(g: &GraphicsState) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    match g.topology.0 {
        RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => g.topology,
    }
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    [ms, ps]: [D3D12_SHADER_BYTECODE; 2],
    g: &GraphicsState,
    name: &str,
    compile: &mut super::make::Compile,
) -> windows::core::Result<Option<ID3D12PipelineState>> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        // The type the host derives from the mesh shader anyway, given: a
        // pipeline library compares descriptions with it, and loads none
        // made from a stream without it.
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            host_topology(g),
        ),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        ms.BytecodeLength.saturating_add(ps.BytecodeLength),
        compile,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}
