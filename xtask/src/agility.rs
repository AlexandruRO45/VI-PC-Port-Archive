//! `cargo xtask agility`: puts the D3D12 Agility SDK runtime beside
//! `victory.exe` (`target\<profile>\D3D12\D3D12Core.dll`), where the
//! exports in `apps/victory/src/agility.rs` point the system's loader.
//!
//! The runtime is Microsoft's redistributable, not ours: it is fetched from
//! its NuGet package at build time and never enters the repository (see
//! docs/asset-policy.md). The package is pinned by version and SHA-256.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};

/// Must match `SDK_VERSION` in `apps/victory/src/agility.rs` (the minor
/// version: 1.620.2 → 620).
const VERSION: &str = "1.620.2";
const SHA256: &str = "70489669c2bbeaf318b62d72b0ac3e753d521387bfc848c9763791c4e920bfcc";
const FILES: [&str; 2] = ["D3D12Core.dll", "d3d12SDKLayers.dll"];

pub fn run(root: &Path) -> Result<()> {
    let target = root.join("target");
    let store = target.join("agility").join(VERSION);
    let package = store.join("package.nupkg");
    if !package.exists() || sha256(&package)? != SHA256 {
        std::fs::create_dir_all(&store)?;
        let url = format!(
            "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/{VERSION}/microsoft.direct3d.d3d12.{VERSION}.nupkg"
        );
        println!("fetching {url}");
        let status = Command::new("curl.exe")
            .args(["--fail", "--silent", "--show-error", "--location", "--output"])
            .arg(&package)
            .arg(&url)
            .status()
            .context("running curl.exe")?;
        ensure!(status.success(), "download failed");
        let got = sha256(&package)?;
        if got != SHA256 {
            let _ = std::fs::remove_file(&package);
            bail!("package hash mismatch: expected {SHA256}, got {got}");
        }
    }
    let unpacked = store.join("unpacked");
    if !unpacked.join("build/native/bin/x64/D3D12Core.dll").exists() {
        std::fs::create_dir_all(&unpacked)?;
        // Windows' own tar (bsdtar) reads zip archives; a tar from Git or
        // MSYS earlier on the path does not.
        let system = std::env::var_os("SystemRoot").context("SystemRoot not set")?;
        let tar = Path::new(&system).join("System32").join("tar.exe");
        let status = Command::new(tar)
            .arg("-xf")
            .arg(&package)
            .arg("-C")
            .arg(&unpacked)
            .status()
            .context("running tar.exe")?;
        ensure!(status.success(), "unpacking failed");
    }
    let bin = unpacked.join("build/native/bin/x64");
    // Every profile built so far, and release if none is.
    let mut profiles: Vec<_> = ["release", "diag", "debug"]
        .into_iter()
        .filter(|p| target.join(p).is_dir())
        .collect();
    if profiles.is_empty() {
        profiles.push("release");
    }
    for profile in profiles {
        let dest = target.join(profile).join("D3D12");
        std::fs::create_dir_all(&dest)?;
        for file in FILES {
            std::fs::copy(bin.join(file), dest.join(file))
                .with_context(|| format!("copying {file} to {}", dest.display()))?;
        }
        println!("Agility SDK {VERSION} installed in {}", dest.display());
    }
    Ok(())
}

/// A file's SHA-256 in lower-case hex, by the system's PowerShell.
fn sha256(file: &Path) -> Result<String> {
    let out = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command"])
        .arg(format!(
            "(Get-FileHash -Algorithm SHA256 -LiteralPath '{}').Hash",
            file.display().to_string().replace('\'', "''")
        ))
        .output()
        .context("running powershell.exe")?;
    ensure!(out.status.success(), "hashing {} failed", file.display());
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_ascii_lowercase())
}
