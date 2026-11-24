//! The unit-test harness creates D3D12 devices (`emit::host`), so it exports
//! the Agility SDK symbols (src/agility.rs) like the `victory` binary does.
//! `rustc-link-arg` (not `-tests`, which covers only `tests/` targets): the
//! library itself is an rlib and is never linked, so of this package only
//! the unit-test harness sees the arguments.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        for symbol in ["D3D12SDKVersion", "D3D12SDKPath"] {
            println!("cargo:rustc-link-arg=/EXPORT:{symbol},DATA");
        }
        // An executable with exports gets an import library by default,
        // and the linker says so on stdout (a `linker_messages` warning).
        println!("cargo:rustc-link-arg=/NOIMPLIB");
        println!("cargo:rustc-link-arg=/NOEXP");
    }
}
