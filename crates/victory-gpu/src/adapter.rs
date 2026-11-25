//! The GPU the host device is on, logged once when it is made. The device
//! takes DXGI's default adapter, which on a laptop with an integrated and a
//! discrete GPU is usually the integrated one: the log says so, and warns
//! when that GPU cannot run the console's waves.

use windows::Win32::Foundation::LUID;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

/// One DXGI adapter, as the log names it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Adapter {
    name: String,
    luid: (u32, i32),
    vendor: u32,
    software: bool,
    /// The user-mode driver's version, `a.b.c.d`.
    driver: Option<String>,
    dedicated_mib: u64,
}

/// Logs the device's adapter and lane range, and warns when another GPU
/// would serve the title better or this one cannot run its waves.
pub(crate) fn log(device: &ID3D12Device) {
    // SAFETY: DXGI queries and a feature query writing its own structure.
    let (used, preferred, lanes) = unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return;
        };
        let used = factory
            .EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid())
            .ok()
            .and_then(|a| describe(&a));
        let preferred = factory
            .EnumAdapterByGpuPreference::<IDXGIAdapter1>(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            .ok()
            .and_then(|a| describe(&a));
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let lanes = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            .then_some((options.WaveLaneCountMin, options.WaveLaneCountMax));
        (used, preferred, lanes)
    };
    let Some(used) = used else { return };
    tracing::info!(
        name = %used.name,
        vendor = format_args!("{:#06x}", used.vendor),
        driver = used.driver.as_deref().unwrap_or("unknown"),
        dedicated_mib = used.dedicated_mib,
        lanes = %lanes.map_or_else(|| "unknown".to_owned(), |(min, max)| format!("{min}-{max}")),
        "host GPU"
    );
    for warning in warnings(&used, preferred.as_ref(), lanes) {
        tracing::warn!("{warning}");
    }
}

/// What the log should warn about the device's adapter (`used`), given
/// DXGI's high-performance one and the device's wave lane range.
fn warnings(used: &Adapter, preferred: Option<&Adapter>, lanes: Option<(u32, u32)>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(p) = preferred.filter(|p| !p.software && p.luid != used.luid) {
        out.push(format!(
            "the host device is on {}, but {} is this PC's high-performance GPU; \
             set this program to \"High performance\" in Windows' graphics settings \
             (Settings > System > Display > Graphics) to run on it",
            used.name, p.name
        ));
    }
    // Translated compute programs that depend on lane layout declare the
    // console's wave size (`emit` module docs).
    if let Some((_, max)) = lanes.filter(|&(_, max)| max < 32) {
        out.push(format!(
            "{} runs waves of at most {max} lanes: compute programs that need the console's \
             32 or 64 get no pipeline on it",
            used.name
        ));
    }
    out
}

/// # Safety
/// `adapter` is a live DXGI adapter.
unsafe fn describe(adapter: &IDXGIAdapter1) -> Option<Adapter> {
    // SAFETY: plain adapter queries.
    let desc = unsafe { adapter.GetDesc1() }.ok()?;
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    // The UMD version, when the adapter answers for `IDXGIDevice`.
    let driver = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
        .ok()
        .map(|v| {
            let v = v as u64;
            format!(
                "{}.{}.{}.{}",
                v >> 48,
                (v >> 32) & 0xffff,
                (v >> 16) & 0xffff,
                v & 0xffff
            )
        });
    Some(Adapter {
        name: String::from_utf16_lossy(&desc.Description[..len]),
        luid: luid(desc.AdapterLuid),
        vendor: desc.VendorId,
        software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        driver,
        dedicated_mib: desc.DedicatedVideoMemory as u64 >> 20,
    })
}

fn luid(l: LUID) -> (u32, i32) {
    (l.LowPart, l.HighPart)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;

    fn adapter(name: &str, low: u32, software: bool) -> Adapter {
        Adapter {
            name: name.to_owned(),
            luid: (low, 0),
            vendor: 0,
            software,
            driver: None,
            dedicated_mib: 0,
        }
    }

    #[test]
    fn a_device_off_the_high_performance_gpu_is_warned_about() {
        let intel = adapter("Intel(R) UHD Graphics", 1, false);
        let nvidia = adapter("NVIDIA GeForce RTX 4050 Laptop GPU", 2, false);
        let warp = adapter("Microsoft Basic Render Driver", 3, true);
        let one = warnings(&intel, Some(&nvidia), Some((32, 32)));
        assert_eq!(one.len(), 1);
        assert!(one[0].contains("on Intel(R) UHD Graphics, but NVIDIA GeForce RTX 4050"));
        assert!(warnings(&nvidia, Some(&nvidia), Some((32, 32))).is_empty());
        // Only software is preferred (no GPU): nothing better to point at.
        assert!(warnings(&intel, Some(&warp), Some((32, 32))).is_empty());
        assert!(warnings(&intel, None, None).is_empty());
    }

    #[test]
    fn a_gpu_without_32_lane_waves_is_warned_about() {
        let intel = adapter("Intel(R) UHD Graphics", 1, false);
        let one = warnings(&intel, Some(&intel), Some((16, 16)));
        assert_eq!(one.len(), 1);
        assert!(one[0].contains("at most 16 lanes"));
        assert!(warnings(&intel, Some(&intel), Some((32, 64))).is_empty());
        assert!(warnings(&intel, Some(&intel), Some((4, 128))).is_empty());
    }

    /// On this machine's default device: its adapter is found and named.
    #[test]
    fn the_default_device_names_its_adapter() {
        // SAFETY: plain DXGI and device creation.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let used =
                describe(&factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap()).unwrap();
            let preferred = factory
                .EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    0,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                )
                .ok()
                .and_then(|a| describe(&a));
            eprintln!("device on {used:?}\nhigh-performance {preferred:?}");
            for warning in warnings(&used, preferred.as_ref(), None) {
                eprintln!("warning: {warning}");
            }
            assert!(!used.name.is_empty());
            assert_eq!(used.luid, luid(device.GetAdapterLuid()));
            log(&device);
        }
    }
}
