//! `victory.exe` exports the Agility SDK symbols (victory-shader's
//! src/agility.rs), so that the system's D3D12 loader takes the runtime in
//! `D3D12\` beside it. Same arguments as victory-shader's build.rs, which
//! explains them; here they reach the binary and its unit-test harness, and
//! both define the symbols (`agility_exports!` in main.rs).

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
