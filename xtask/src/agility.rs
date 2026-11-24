//! `cargo xtask agility`: fetches the D3D12 Agility SDK runtime that
//! victory-shader's `agility` module names and puts it in `D3D12\` beside
//! the build outputs, where the executables' exports send the system's D3D12
//! loader. Needed on Windows 10, whose own runtime stops at shader model 6.5.
//!
//! The package is Microsoft's redistributable (`Microsoft.Direct3D.D3D12` on
//! NuGet). It is kept in the cache folder, never in the repository (it would
//! fail the asset policy anyway), and checked against a pinned hash.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};
use victory_shader::agility::{PACKAGE_VERSION, SDK_VERSION};

/// BLAKE3 of the `.nupkg` for [`PACKAGE_VERSION`] (SHA-256
/// 70489669c2bbeaf318b62d72b0ac3e753d521387bfc848c9763791c4e920bfcc).
const PACKAGE_BLAKE3: &str = "b5811248412481b3fa9d77fbe3cc1c7703f6b961beca45c4a956f3792fcb0d8b";
/// What the process needs: the runtime, and the debug layer that
/// `VICTORY_D3D12_DEBUG` asks for (it must match the runtime's version).
const FILES: [&str; 2] = ["D3D12Core.dll", "d3d12SDKLayers.dll"];
/// Build profiles whose outputs get the runtime, beside the executables
/// (`<profile>\`) and the test harnesses (`<profile>\deps\`).
const PROFILES: [&str; 3] = ["debug", "release", "diag"];

pub fn run(root: &Path) -> Result<()> {
    let config = victory_core::Config::load()?;
    let dir = config.cache_subdir("agility")?.join(PACKAGE_VERSION);
    if !FILES.iter().all(|f| dir.join(f).is_file()) {
        fetch(&dir)?;
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("target"));
    for profile in PROFILES {
        for sub in ["", "deps"] {
            let out = target.join(profile).join(sub).join("D3D12");
            std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
            for f in FILES {
                std::fs::copy(dir.join(f), out.join(f))
                    .with_context(|| format!("copying {f} to {}", out.display()))?;
            }
        }
    }
    println!(
        "Agility SDK {PACKAGE_VERSION} (D3D12SDKVersion {SDK_VERSION}) from {} placed in \
         {}\\{{{}}}[\\deps]\\D3D12",
        dir.display(),
        target.display(),
        PROFILES.join(",")
    );
    Ok(())
}

/// Downloads the package, checks it and extracts [`FILES`] into `dir`.
fn fetch(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // The system's own tools: Git's `tar` cannot read a zip.
    let system = PathBuf::from(std::env::var_os("SystemRoot").unwrap_or("C:\\Windows".into()))
        .join("System32");
    let package = dir.join(format!("microsoft.direct3d.d3d12.{PACKAGE_VERSION}.nupkg"));
    let url =
        format!("https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/{PACKAGE_VERSION}");
    println!("downloading {url}");
    let status = Command::new(system.join("curl.exe"))
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "3",
            "--output",
        ])
        .arg(&package)
        .arg(&url)
        .status()
        .context("running curl.exe")?;
    ensure!(status.success(), "downloading {url} failed");
    let hash = blake3::hash(&std::fs::read(&package)?).to_hex();
    if hash.as_str() != PACKAGE_BLAKE3 {
        std::fs::remove_file(&package).ok();
        anyhow::bail!("{url}: BLAKE3 {hash}, expected {PACKAGE_BLAKE3}");
    }
    let staging = dir.join("extract");
    std::fs::create_dir_all(&staging)?;
    let inner: Vec<String> = FILES
        .iter()
        .map(|f| format!("build/native/bin/x64/{f}"))
        .collect();
    let status = Command::new(system.join("tar.exe"))
        .arg("-xf")
        .arg(&package)
        .arg("-C")
        .arg(&staging)
        .args(&inner)
        .status()
        .context("running tar.exe")?;
    ensure!(status.success(), "extracting {} failed", package.display());
    for (f, path) in FILES.iter().zip(&inner) {
        std::fs::rename(staging.join(path), dir.join(f))?;
    }
    std::fs::remove_dir_all(&staging)?;
    std::fs::remove_file(&package)?;
    Ok(())
}
