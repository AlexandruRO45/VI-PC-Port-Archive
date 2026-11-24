//! The unit-test harness creates host D3D12 devices, so it exports the
//! Agility SDK symbols (victory-shader's src/agility.rs). Same arguments as
//! victory-shader's build.rs, which explains them.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        for symbol in ["D3D12SDKVersion", "D3D12SDKPath"] {
            println!("cargo:rustc-link-arg=/EXPORT:{symbol},DATA");
        }
        println!("cargo:rustc-link-arg=/NOIMPLIB");
        println!("cargo:rustc-link-arg=/NOEXP");
    }
}
