// Exports `D3D12SDKVersion` and `D3D12SDKPath` from victory.exe, which is how
// a program opts into the D3D12 Agility SDK (see `src/agility.rs`).
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "windows") {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
