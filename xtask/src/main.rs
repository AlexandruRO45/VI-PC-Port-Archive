//! Repository maintenance. Run with `cargo xtask <command>`.

mod policy;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enforce the asset policy: the repository holds only small text files
    /// we wrote. Checks tracked and untracked-but-not-ignored files.
    Policy {
        /// Check only what is staged for commit (used by the pre-commit hook).
        #[arg(long)]
        staged: bool,
    },
    /// Point git at .githooks/ so every commit runs the policy check.
    InstallHooks,
    /// Download Microsoft's D3D12 Agility SDK and put its runtime in
    /// target/<profile>/D3D12/, next to victory.exe. Needed on Windows 10,
    /// whose own D3D12 lacks shader model 6.6, by a victory.exe built with
    /// `--features agility`.
    Agility,
}

fn main() -> Result<()> {
    let root = workspace_root();
    match Cli::parse().command {
        Cmd::Policy { staged } => policy::run(&root, staged),
        Cmd::InstallHooks => {
            let status = Command::new("git")
                .args(["config", "core.hooksPath", ".githooks"])
                .current_dir(&root)
                .status()
                .context("running git")?;
            anyhow::ensure!(
                status.success(),
                "git config failed (is this a git repository?)"
            );
            println!("hooks installed: commits now run `cargo xtask policy --staged`");
            Ok(())
        }
        Cmd::Agility => agility(&root),
    }
}

/// The Agility SDK package; its `D3D12_SDK_VERSION` (the middle number)
/// must match `D3D12SDKVersion` in apps/victory/src/main.rs.
const AGILITY_VERSION: &str = "1.620.2";

/// Fetches the Agility SDK from nuget.org (with Windows' own curl and tar)
/// and copies its runtime next to every built victory.exe.
fn agility(root: &Path) -> Result<()> {
    let dir = root.join("target").join("agility").join(AGILITY_VERSION);
    let core = dir.join("build/native/bin/x64/D3D12Core.dll");
    if !core.exists() {
        std::fs::create_dir_all(&dir)?;
        let package = dir.join("package.zip");
        let url = format!(
            "https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/\
             {AGILITY_VERSION}/microsoft.direct3d.d3d12.{AGILITY_VERSION}.nupkg"
        );
        println!("downloading {url}");
        run(Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(&package)
            .arg(&url))?;
        run(Command::new("tar")
            .arg("-xf")
            .arg(&package)
            .arg("-C")
            .arg(&dir))?;
        anyhow::ensure!(core.exists(), "the package has no {}", core.display());
    }
    let mut placed = 0;
    for profile in ["debug", "release", "diag"] {
        let out = root.join("target").join(profile);
        if !out.join("victory.exe").exists() {
            continue;
        }
        let to = out.join("D3D12");
        std::fs::create_dir_all(&to)?;
        for name in ["D3D12Core.dll", "d3d12SDKLayers.dll"] {
            std::fs::copy(core.with_file_name(name), to.join(name))
                .with_context(|| format!("copying {name} to {}", to.display()))?;
        }
        println!("Agility SDK {AGILITY_VERSION} -> {}", to.display());
        placed += 1;
    }
    anyhow::ensure!(
        placed > 0,
        "no victory.exe under target/; build it first (cargo build --release -p victory)"
    );
    Ok(())
}

fn run(command: &mut Command) -> Result<()> {
    let status = command
        .status()
        .with_context(|| format!("running {command:?}"))?;
    anyhow::ensure!(status.success(), "{command:?} failed: {status}");
    Ok(())
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace")
        .into()
}
