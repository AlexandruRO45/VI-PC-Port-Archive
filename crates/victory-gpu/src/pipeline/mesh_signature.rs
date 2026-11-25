//! Share the mesh fallback's root-writing indirect signature across PSOs.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::emit::MESH_RECORD_SPACE;
use windows::Win32::Foundation::E_INVALIDARG;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Error, IUnknown, Interface, Result};

const DEST: u32 = 0;
const DWORDS: u32 = 1;

#[derive(Default)]
struct Cache {
    // The live signatures (and their roots) keep their device alive. Key by
    // COM identity and exact serialized bytes, never by a root pointer/hash.
    devices: HashMap<usize, HashMap<u32, HashMap<Vec<u8>, Entry>>>,
}

type Entry = (ID3D12RootSignature, ID3D12CommandSignature);
static CACHE: std::sync::LazyLock<Mutex<Cache>> = std::sync::LazyLock::new(Mutex::default);
static CREATED: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static BYPASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn validate(params: &[D3D12_ROOT_PARAMETER1], base: u32) -> Result<()> {
    let invalid = || Error::from_hresult(E_INVALIDARG);
    let param = params.get(base as usize).ok_or_else(invalid)?;
    if param.ParameterType != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
        || param.ShaderVisibility != D3D12_SHADER_VISIBILITY_ALL
    {
        return Err(invalid());
    }
    // SAFETY: checked the union tag above.
    let c = unsafe { param.Anonymous.Constants };
    if c.ShaderRegister != 0
        || c.RegisterSpace != MESH_RECORD_SPACE
        || DEST + DWORDS > c.Num32BitValues
    {
        return Err(invalid());
    }
    Ok(())
}

pub(super) fn get(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    blob: &[u8],
    params: &[D3D12_ROOT_PARAMETER1],
    base: u32,
    about: impl FnOnce(),
) -> Result<Entry> {
    validate(params, base)?;
    #[cfg(test)]
    if BYPASS.get() {
        return create(device, root, base, about).map(|sig| (root.clone(), sig));
    }
    use crate::pipeline::telemetry::{Phase, phase};
    let identity: IUnknown = device.cast()?;
    let _phase = phase(Phase::SignatureLock);
    let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let roots = cache
        .devices
        .entry(identity.as_raw() as usize)
        .or_default()
        .entry(base)
        .or_default();
    if let Some(entry) = roots.get(blob) {
        return Ok(entry.clone());
    }
    // Serialize real creations so concurrent cache misses for the same root
    // cannot each call into the AMD driver. No pipeline-library lock is held.
    let _create = phase(Phase::SignatureCreate);
    let sig = create(device, root, base, about)?;
    let entry = (root.clone(), sig);
    roots.insert(blob.to_vec(), entry.clone());
    Ok(entry)
}

fn create(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    base: u32,
    about: impl FnOnce(),
) -> Result<ID3D12CommandSignature> {
    let args = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: base,
                    DestOffsetIn32BitValues: DEST,
                    Num32BitValuesToSet: DWORDS,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let mut signature = None;
    about();
    // SAFETY: the parameter was checked against this root's description;
    // the arguments, root and output pointer live through the call.
    unsafe {
        device.CreateCommandSignature(
            &D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: (DWORDS + 3) * 4,
                NumArgumentDescs: args.len() as u32,
                pArgumentDescs: args.as_ptr(),
                NodeMask: 0,
            },
            root,
            &mut signature,
        )?;
    }
    let signature = signature.ok_or_else(Error::empty)?;
    let total = CREATED.fetch_add(1, Ordering::Relaxed) + 1;
    // Keep a cumulative count in normal logs without logging every PSO.
    if total.is_power_of_two() {
        tracing::info!(total, "distinct mesh indirect command signatures created");
    }
    Ok(signature)
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
#[ignore = "isolated RX hardware test; see environment checks"]
fn mesh_signature_stress() {
    use super::{GraphicsState, HostPipeline, Shader, Translated};
    use std::sync::Arc;
    use victory_shader::{
        emit::{MeshRecordLayout, RecordElement, host::stub_ps},
        native::HwStage,
    };
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
    };
    use windows::{
        Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob},
        core::s,
    };

    assert!(std::env::var_os("VICTORY_D3D12_DEBUG").is_some());
    assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none());
    let host = crate::host::host().expect("hardware host");
    assert!(!host.mesh_shaders(), "set VICTORY_NO_MESH_SHADERS=1");
    let queue: ID3D12InfoQueue = host.device.cast().expect("D3D12 debug layer");
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        assert_eq!(desc.VendorId, 0x1002);
        eprintln!(
            "adapter: {}",
            String::from_utf16_lossy(&desc.Description).trim_end_matches('\0')
        );
        queue.ClearStoredMessages();
    }
    let source = b"[numthreads(1,1,1)] void main() {}";
    let dxil = unsafe {
        let mut blob: Option<ID3DBlob> = None;
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("mesh-stress"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut blob,
            None,
        )
        .unwrap();
        let blob = blob.unwrap();
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
            .to_vec()
    };
    let mut cs = stub_ps();
    cs.stage = HwStage::Cs;
    cs.dxil = dxil;
    let layout = MeshRecordLayout {
        max_vertices: 3,
        max_primitives: 1,
        vert_dwords: 4,
        prim_idx_base: 16,
        prim_attr_base: 17,
        prim_dwords: 0,
        record_dwords: 20,
        vertices: vec![RecordElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            offset: 0,
        }],
        prims: vec![],
    };
    let mesh = Arc::new(Shader::new("mesh", source));
    assert!(
        mesh.mesh_fallback
            .set(Some((
                Translated {
                    host: cs,
                    used: vec![]
                },
                layout
            )))
            .is_ok()
    );
    let bypass = std::env::var_os("VICTORY_MESH_STRESS_BASELINE").is_some();
    BYPASS.set(bypass);
    let before = CREATED.load(Ordering::Relaxed);
    const N: u32 = 5000;
    let mut pipelines = Vec::new();
    for i in 0..N {
        let g = GraphicsState {
            sample_mask: i,
            ..Default::default()
        };
        pipelines.push(HostPipeline::new(std::slice::from_ref(&mesh), Some(&g)).unwrap());
        if (i + 1) % 500 == 0 {
            eprintln!(
                "{} pipelines; {} real signature creations",
                i + 1,
                CREATED.load(Ordering::Relaxed) - before
            );
        }
    }
    assert_eq!(
        CREATED.load(Ordering::Relaxed) - before,
        if bypass { u64::from(N) } else { 1 }
    );
    // Read the queue directly: Host::log_messages returns only errors, not warnings.
    // Only corruption and errors fail the run: warnings come from the synthetic
    // graphics state (no depth view) and from normal pipeline-library misses.
    unsafe {
        let mut unexpected = Vec::new();
        for i in 0..queue.GetNumStoredMessages() {
            let mut bytes = 0;
            let _ = queue.GetMessage(i, None, &mut bytes);
            let mut storage = vec![0usize; bytes.div_ceil(size_of::<usize>())];
            let ptr = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
            queue.GetMessage(i, Some(ptr), &mut bytes).unwrap();
            let message = &*ptr;
            if message.Severity.0 > D3D12_MESSAGE_SEVERITY_ERROR.0 {
                continue;
            }
            let text = std::slice::from_raw_parts(message.pDescription, message.DescriptionByteLength);
            unexpected.push(format!(
                "id={} severity={}: {}",
                message.ID.0,
                message.Severity.0,
                String::from_utf8_lossy(text).trim_end_matches('\0')
            ));
        }
        assert!(unexpected.is_empty(), "D3D12 messages: {:?}", &unexpected[..unexpected.len().min(5)]);
    }
}

#[cfg(test)]
#[test]
fn mesh_constant_must_target_a_live_root_dword() {
    let mut param = D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: MESH_RECORD_SPACE,
                Num32BitValues: 1,
            },
        },
    };
    assert!(validate(&[param], 0).is_ok());
    assert!(validate(&[param], 1).is_err());
    param.Anonymous.Constants.Num32BitValues = 0;
    assert!(validate(&[param], 0).is_err());
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    assert!(validate(&[param], 0).is_err());
}
