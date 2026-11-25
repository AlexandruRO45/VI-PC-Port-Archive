fn main() {
    let arg = if std::env::var("CARGO_CFG_TARGET_ABI").as_deref() == Ok("llvm") {
        "-Wl,--delayload=victory-probe-delayed.dll"
    } else {
        "/DELAYLOAD:victory-probe-delayed.dll"
    };
    println!("cargo:rustc-link-arg-cdylib={arg}");
    println!("cargo:rustc-link-lib=delayimp");
}
