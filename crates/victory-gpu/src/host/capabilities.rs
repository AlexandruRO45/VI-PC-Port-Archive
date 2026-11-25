use anyhow::{Context, Result};
use std::fmt;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_FLAG_SOFTWARE;

/// The two hardware features required by the translated mesh pipelines.
pub(super) struct MeshSupport {
    model_6_6: Option<bool>,
    mesh_shaders: Option<bool>,
}

impl MeshSupport {
    pub(super) fn query(device: &ID3D12Device) -> Self {
        let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        // SAFETY: the output structures match their feature IDs and sizes.
        unsafe {
            let model_6_6 = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of_val(&model) as u32,
                )
                .ok()
                .map(|()| model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0);
            let mesh_shaders = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .ok()
                .map(|()| options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED);
            Self {
                model_6_6,
                mesh_shaders,
            }
        }
    }

    pub(super) fn supported(&self) -> bool {
        self.model_6_6 == Some(true) && self.mesh_shaders == Some(true)
    }
}

/// Properties of the adapter backing the renderer's actual D3D12 device.
pub struct GpuInfo {
    name: String,
    luid: (u32, u32),
    kind: AdapterKind,
    dedicated_bytes: u64,
    local_budget: Option<u64>,
    features: MeshSupport,
}

#[derive(Clone, Copy)]
enum AdapterKind {
    Hardware { unified: Option<bool> },
    Software,
    Warp,
}

impl AdapterKind {
    fn from_dxgi(software: bool, vendor: u32, device: u32, unified: Option<bool>) -> Self {
        match (software, vendor, device) {
            // DXGI's render-only Microsoft Basic Render Driver (WARP).
            (true, 0x1414, 0x008c) => Self::Warp,
            (true, _, _) => Self::Software,
            (false, _, _) => Self::Hardware { unified },
        }
    }

    fn hardware(self) -> bool {
        matches!(self, Self::Hardware { .. })
    }
}

impl GpuInfo {
    /// A hardware device was created at feature level 12.0 or later and both
    /// mesh pipeline feature queries confirmed support.
    pub fn meets_requirements(&self) -> bool {
        self.kind.hardware() && self.features.supported()
    }
}

/// Queries the same device used for rendering. Host initialization does not
/// create a window or swap chain or require a title executable.
pub fn gpu_info() -> Result<GpuInfo> {
    describe(super::host_result().as_ref())
}

fn describe(host: Result<&super::Host, &windows::core::Error>) -> Result<GpuInfo> {
    let host = host
        .map_err(Clone::clone)
        .context("could not initialize the D3D12 renderer (requires feature level 12.0)")?;
    // Host resolved this adapter from ID3D12Device::GetAdapterLuid.
    let adapter = host
        .adapter
        .as_ref()
        .context("DXGI could not identify the renderer's adapter by LUID")?;
    // SAFETY: the adapter is live and GetDesc1 returns a fully initialized descriptor.
    let desc =
        unsafe { adapter.GetDesc1() }.context("cannot read the renderer's adapter description")?;
    let length = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    // SAFETY: node 0 exists on this device and the structure matches the feature ID.
    let unified_memory = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_ARCHITECTURE1,
            (&raw mut arch).cast(),
            size_of_val(&arch) as u32,
        )
    }
    .ok()
    .map(|()| arch.UMA.as_bool());
    Ok(GpuInfo {
        name: String::from_utf16_lossy(&desc.Description[..length]),
        luid: (desc.AdapterLuid.HighPart as u32, desc.AdapterLuid.LowPart),
        kind: AdapterKind::from_dxgi(
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            desc.VendorId,
            desc.DeviceId,
            unified_memory,
        ),
        dedicated_bytes: desc.DedicatedVideoMemory as u64,
        local_budget: host.video_memory().map(|m| m[0]),
        features: MeshSupport::query(&host.device),
    })
}

impl fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self.kind {
            AdapterKind::Warp => "software, WARP",
            AdapterKind::Software => "software",
            AdapterKind::Hardware {
                unified: Some(true),
            } => "hardware, unified memory",
            AdapterKind::Hardware {
                unified: Some(false),
            } => "hardware, non-unified memory",
            AdapterKind::Hardware { unified: None } => "hardware",
        };
        let status = |feature| match feature {
            Some(true) => "yes",
            Some(false) => "no",
            None => "unknown (query unavailable)",
        };
        writeln!(f, "Renderer adapter: {} ({kind})", self.name)?;
        writeln!(
            f,
            "DXGI adapter LUID: {:08x}:{:08x}",
            self.luid.0, self.luid.1
        )?;
        writeln!(f, "D3D12 feature level 12.0 or later: yes")?;
        writeln!(
            f,
            "Shader model 6.6 or later: {}",
            status(self.features.model_6_6)
        )?;
        writeln!(f, "Mesh shaders: {}", status(self.features.mesh_shaders))?;
        if self.kind.hardware() {
            writeln!(
                f,
                "Dedicated video memory: {} MiB",
                self.dedicated_bytes >> 20
            )?;
            if let Some(budget) = self.local_budget {
                writeln!(
                    f,
                    "Current DXGI local memory budget for this process: {} MiB",
                    budget >> 20
                )?;
            }
        }
        write!(
            f,
            "Hardware rendering requirements confirmed: {}",
            status(Some(self.meets_requirements()))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_host_reports_the_required_d3d_feature_level_and_native_error() {
        let native = windows::core::Error::from_hresult(
            windows::Win32::Graphics::Dxgi::DXGI_ERROR_UNSUPPORTED,
        );
        let Err(error) = describe(Err(&native)) else {
            panic!("a missing renderer cannot report an adapter")
        };
        assert_eq!(
            error.to_string(),
            "could not initialize the D3D12 renderer (requires feature level 12.0)"
        );
        assert_eq!(
            error.downcast_ref::<windows::core::Error>().unwrap().code(),
            native.code()
        );
    }

    #[test]
    fn readiness_requires_hardware_and_two_confirmed_features() {
        for software in [true, false] {
            for model in [Some(true), Some(false), None] {
                for mesh in [Some(true), Some(false), None] {
                    let info = GpuInfo {
                        name: "Test adapter".into(),
                        luid: (1, 0x2abc),
                        kind: AdapterKind::from_dxgi(software, 0x1414, 0x008c, Some(true)),
                        dedicated_bytes: 128 << 20,
                        local_budget: None,
                        features: MeshSupport {
                            model_6_6: model,
                            mesh_shaders: mesh,
                        },
                    };
                    assert_eq!(
                        info.meets_requirements(),
                        !software && model == Some(true) && mesh == Some(true)
                    );
                    let output = info.to_string();
                    assert!(output.contains("DXGI adapter LUID: 00000001:00002abc"));
                    assert_eq!(
                        output.contains("unknown (query unavailable)"),
                        model.is_none() || mesh.is_none()
                    );
                    assert_eq!(output.contains("Dedicated video memory"), !software);
                    assert_eq!(output.contains("software, WARP"), software);
                    assert_eq!(output.contains("hardware, unified memory"), !software);
                    assert!(!output.contains("Current DXGI local memory budget"));
                }
            }
        }
    }
}
