//! With feature `agility`, exports the D3D12 Agility SDK markers (see
//! `D3D12SDKVersion` in main.rs) from `victory.exe`; an executable exports
//! nothing by default.

fn main() {
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if msvc && std::env::var_os("CARGO_FEATURE_AGILITY").is_some() {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
