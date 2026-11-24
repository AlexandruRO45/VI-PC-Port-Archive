//! Exports the two symbols by which the system's D3D12 loader finds the
//! D3D12 Agility SDK runtime beside an executable (`D3D12SDKVersion`,
//! `D3D12SDKPath`, defined in `src/agility.rs`). A Rust executable exports
//! nothing on its own, so the linker is told to.
//!
//! The runtime itself (`D3D12\D3D12Core.dll` next to `victory.exe`) is not
//! in the repository: `cargo xtask agility` fetches it. Without it the
//! system's runtime is used, which on Windows 10 lacks shader model 6.6.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        for symbol in ["D3D12SDKVersion", "D3D12SDKPath"] {
            println!("cargo:rustc-link-arg-bin=victory=/EXPORT:{symbol},DATA");
        }
        // No import library for an executable's exports (nor its message).
        println!("cargo:rustc-link-arg-bin=victory=/NOIMPLIB");
        println!("cargo:rustc-link-arg-bin=victory=/NOEXP");
    }
}
