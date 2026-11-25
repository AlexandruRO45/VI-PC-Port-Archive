#![cfg(all(windows, feature = "host"))]

use std::process::Command;

use victory_shader::emit::host::{Host, stub_ps};
use windows::Win32::Graphics::Direct3D12::{D3D12GetDebugInterface, ID3D12Debug};

const CHILD_ENV: &str = "VICTORY_D3D12_DEBUG_CHILD";

fn debug_layer_available() -> bool {
    let mut layer: Option<ID3D12Debug> = None;
    // SAFETY: valid out-pointer to an optional `ID3D12Debug` interface.
    unsafe { D3D12GetDebugInterface(&mut layer) }.is_ok() && layer.is_some()
}

fn trigger_pipeline_refusal(host: &Host) -> (String, String, String) {
    let ps = stub_ps();
    let root = host
        .root_signature_for(&[])
        .expect("empty root signature should serialize");
    let compute_err = host
        .compute_pipeline_with(&root, &ps.dxil)
        .expect_err("pixel shader bytecode must fail compute pipeline creation");
    let graphics_err = host
        .pipeline_with(&root, &ps.dxil, &ps.dxil)
        .expect_err("pixel shader bytecode as VS must fail graphics pipeline creation");
    let mesh_err = host
        .mesh_pipeline_with(&root, &ps.dxil, &ps.dxil)
        .expect_err("pixel shader bytecode as MS must fail mesh pipeline creation");
    (compute_err, graphics_err, mesh_err)
}

fn assert_has_debug_reason(label: &str, msg: &str) {
    assert!(
        !msg.contains('\n') && !msg.contains('\r'),
        "{label}: expected single-line error, got: {msg:?}"
    );
    assert!(
        msg.contains("CreateComputePipelineState")
            || msg.contains("CreateGraphicsPipelineState")
            || msg.contains("CreatePipelineState")
            || msg.contains("CreateComputeShader")
            || msg.contains("CreateVertexShader")
            || msg.contains("CreateMeshShader")
            || msg.contains("Shader")
            || msg.contains("STATE_CREATION ERROR"),
        "{label}: expected D3D12 debug layer reason, got: {msg:?}"
    );
}

#[test]
fn d3d12_debug_layer_reports_pipeline_refusal_reasons() {
    if let Ok(mode) = std::env::var(CHILD_ENV) {
        match mode.as_str() {
            "debug_on" => {
                let warp = Host::new(true).expect("WARP host with debug layer");
                let (cs_err, gfx_err, mesh_err) = trigger_pipeline_refusal(&warp);
                eprintln!("[WARP debug=1] compute={cs_err}");
                eprintln!("[WARP debug=1] graphics={gfx_err}");
                eprintln!("[WARP debug=1] mesh={mesh_err}");
                if debug_layer_available() {
                    assert_has_debug_reason("WARP compute", &cs_err);
                    assert_has_debug_reason("WARP graphics", &gfx_err);
                    assert_has_debug_reason("WARP mesh", &mesh_err);
                }

                // Exercise concurrent pipeline creation across worker threads on the same adapter.
                let handles: Vec<_> = (0..4)
                    .map(|tid| {
                        std::thread::spawn(move || {
                            let h = Host::new(true).expect("worker WARP host");
                            let (c, g, m) = trigger_pipeline_refusal(&h);
                            if debug_layer_available() {
                                assert_has_debug_reason(&format!("worker {tid} compute"), &c);
                                assert_has_debug_reason(&format!("worker {tid} graphics"), &g);
                                assert_has_debug_reason(&format!("worker {tid} mesh"), &m);
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().expect("worker join");
                }

                if let Ok(hw) = Host::new(false) {
                    let (hw_cs, hw_gfx, hw_mesh) = trigger_pipeline_refusal(&hw);
                    eprintln!("[HW debug=1] compute={hw_cs}");
                    eprintln!("[HW debug=1] graphics={hw_gfx}");
                    eprintln!("[HW debug=1] mesh={hw_mesh}");
                    if debug_layer_available() {
                        assert_has_debug_reason("HW compute", &hw_cs);
                        assert_has_debug_reason("HW graphics", &hw_gfx);
                        assert_has_debug_reason("HW mesh", &hw_mesh);
                    }
                }
            }
            "debug_off" => {
                let warp = Host::new(true).expect("WARP host without debug layer");
                let (cs_err, gfx_err, mesh_err) = trigger_pipeline_refusal(&warp);
                eprintln!("[WARP debug=0] compute={cs_err}");
                eprintln!("[WARP debug=0] graphics={gfx_err}");
                eprintln!("[WARP debug=0] mesh={mesh_err}");
                for (label, err) in [("cs", &cs_err), ("gfx", &gfx_err), ("mesh", &mesh_err)] {
                    assert!(
                        !err.contains("STATE_CREATION ERROR") && !err.contains('\n'),
                        "{label}: expected plain HRESULT error when VICTORY_D3D12_DEBUG=0, got: {err:?}"
                    );
                }
            }
            other => panic!("unexpected {CHILD_ENV} mode: {other}"),
        }
        return;
    }

    let exe = std::env::current_exe().expect("current_exe");
    for (mode, val) in [("debug_on", "1"), ("debug_off", "0")] {
        let out = Command::new(&exe)
            .arg("--exact")
            .arg("d3d12_debug_layer_reports_pipeline_refusal_reasons")
            .arg("--nocapture")
            .env(CHILD_ENV, mode)
            .env("VICTORY_D3D12_DEBUG", val)
            .output()
            .expect("spawn child test process");
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        eprintln!("=== child {mode} ===\n{stdout}{stderr}");
        assert!(
            out.status.success(),
            "child {mode} failed with status {:?}:\nstdout:\n{stdout}\nstderr:\n{stderr}",
            out.status
        );
    }
}
