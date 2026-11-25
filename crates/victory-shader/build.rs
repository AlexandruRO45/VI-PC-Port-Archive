use std::{env, fs, path::PathBuf};

fn main() {
    const VERSION: &str = "VICTORY_TEST_AGILITY_SDK_VERSION";
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed={VERSION}");
    println!("cargo::rustc-check-cfg=cfg(victory_test_agility)");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows")
        || env::var("CARGO_CFG_TARGET_ENV").as_deref() != Ok("msvc")
    {
        return;
    }
    let Some(version) = env::var_os(VERSION) else {
        return;
    };
    let version: u32 = version
        .to_str()
        .and_then(|version| version.parse().ok())
        .filter(|&version| version != 0)
        .expect("VICTORY_TEST_AGILITY_SDK_VERSION must be a positive SDK version");
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    fs::write(out.join("test_agility_version.rs"), format!("{version}u32"))
        .expect("write the Agility SDK version for shader tests");
    println!("cargo::rustc-cfg=victory_test_agility");

    // Cargo's scoped link-arg-tests directive only reaches integration
    // tests. The unscoped directive reaches the library's unit-test exe,
    // while ordinary rlib builds do not link.
    for name in ["D3D12SDKVersion", "D3D12SDKPath"] {
        println!("cargo::rustc-link-arg=/EXPORT:{name},DATA");
    }
}
