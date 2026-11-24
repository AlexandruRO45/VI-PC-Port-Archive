//! Exports the D3D12 Agility SDK values (see `D3D12SDKVersion` in main.rs)
//! from victory.exe: Rust does not export statics from an executable, and
//! Windows only reads them from the exe's export table.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
