use super::*;
use crate::capture_diff::{compare, write_submission};
use windows::Win32::Foundation::RECT;

#[test]
fn a_readback_reports_one_changed_alpha_byte() {
    let Some(host) = crate::host::host() else {
        eprintln!("capture readback skipped: no D3D12 host");
        return;
    };
    // A 12-byte logical row in a 256-byte D3D12 row pitch.
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 3,
        Height: 2,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: all inputs describe a single render target and texture is a valid output.
    unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            None,
            &mut texture,
        )
    }
    .unwrap();
    let texture = texture.unwrap();
    let before = tempfile::tempdir().unwrap();
    let after = tempfile::tempdir().unwrap();
    let mut pixels = [0u8; 24];
    for (directory, alpha) in [(before.path(), 0), (after.path(), 255)] {
        let list = host.new_list().unwrap();
        let rtv = host.rtv(&texture, None);
        // SAFETY: the list is open, the texture is a render target and the rectangle fits.
        unsafe {
            list.list.ClearRenderTargetView(rtv, &[0.; 4], None);
            let last = RECT {
                left: 2,
                top: 1,
                right: 3,
                bottom: 2,
            };
            list.list.ClearRenderTargetView(
                rtv,
                &[0., 0., 0., f32::from(alpha) / 255.],
                Some(&[last]),
            );
        }
        list.close();
        host.execute(&[&list]);

        let target = submission_target(
            host,
            "test",
            (0, std::ptr::null(), "test writer"),
            &texture,
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
            None,
        )
        .unwrap();
        pixels[23] = alpha;
        assert_eq!(target.hash, Some(blake3::hash(&pixels)));
        write_submission(directory, 7, 12, &[target]).unwrap();
    }
    let difference = compare(before.path(), after.path()).unwrap().unwrap();
    assert!(
        difference.contains("target contents changed"),
        "{difference}"
    );
    assert!(difference.contains("test writer"), "{difference}");
}
