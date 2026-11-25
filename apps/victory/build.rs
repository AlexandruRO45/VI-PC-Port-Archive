//! Windows 10: the D3D12 Agility SDK opt-in.
//!
//! Windows 10's inbox D3D12 runtime stops at shader model 6.5, and the
//! translated shaders need 6.6: without a newer runtime the title's root
//! signatures (flags 0xc01) are refused and it crashes during start-up.
//! Microsoft's Agility SDK ships that runtime as `D3D12Core.dll`, and the
//! system's `d3d12.dll` loads it from beside an executable that exports
//! `D3D12SDKVersion` (the runtime's SDK version) and `D3D12SDKPath`.
//!
//! The exports are part of the image, so they are decided at build time: when
//! the repository's `D3D12/` folder holds `D3D12Core.dll` (git ignores DLLs),
//! this script reads the SDK version that runtime exports, adds both exports
//! with it and copies the folder's DLLs to `target/<profile>/D3D12/`. Without
//! the file the executable is built as before and uses the system runtime.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-check-cfg=cfg(victory_agility)");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || std::env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return;
    }
    // `apps/victory`: the repository is two directories up.
    let Some(source) = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|dir| dir.ancestors().nth(2).map(|root| root.join("D3D12")))
    else {
        return;
    };
    // The folder is tracked (its README), so it exists: watching it instead of
    // the DLL, Cargo does not rerun the script (and relink) on every build.
    println!("cargo:rerun-if-changed={}", source.display());
    let runtime = source.join("D3D12Core.dll");
    let Ok(data) = std::fs::read(&runtime) else {
        return;
    };
    let version = victory_pe::ImageInfo::parse(&data)
        .ok()
        .and_then(|image| image.exported_u32(&data, "D3D12SDKVersion"));
    let Some(version) = version else {
        println!(
            "cargo:warning={} exports no D3D12SDKVersion: not an Agility SDK runtime, the system runtime stays in use",
            runtime.display()
        );
        return;
    };
    // `target/<profile>/build/victory-<hash>/out`: the executable lands three
    // directories above `OUT_DIR`.
    let Some(profile) = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .and_then(|out| out.ancestors().nth(3).map(Path::to_path_buf))
    else {
        return;
    };
    let target = profile.join("D3D12");
    if let Err(e) = copy_dlls(&source, &target) {
        println!(
            "cargo:warning=could not copy the Agility SDK runtime to {}: {e}",
            target.display()
        );
    }
    println!("cargo:rustc-cfg=victory_agility");
    println!("cargo:rustc-env=VICTORY_AGILITY_SDK_VERSION={version}");
    println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion");
    println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath");
    println!(
        "cargo:warning=using the Agility SDK runtime {version} from {}, copied to {}",
        runtime.display(),
        target.display()
    );
}

/// Copies every DLL in `from` (the runtime, and the SDK layers when they are
/// there) to `to`.
fn copy_dlls(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("dll"))
        {
            std::fs::copy(&path, to.join(path.file_name().unwrap_or_default()))?;
        }
    }
    Ok(())
}
