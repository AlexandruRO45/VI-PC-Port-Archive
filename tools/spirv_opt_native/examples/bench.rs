//! `cargo run --release --example bench -- <dir>`: times `optimize()` over every file in a folder of .spvasm.
fn main() {
    let dir = std::env::args().nth(1).expect("usage: bench <dir of .spvasm files>");
    let mut mods = Vec::new();
    for e in std::fs::read_dir(dir).expect("read_dir") {
        mods.push(std::fs::read(e.expect("entry").path()).expect("read"));
    }
    let bytes: usize = mods.iter().map(|m| m.len()).sum();
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = std::time::Instant::now();
        for m in &mods {
            std::hint::black_box(spirv_opt_native::optimize(m, 4).ok());
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{} modules, {:.2} MB: {:.1} ms ({:.0} MB/s)", mods.len(), bytes as f64 / 1e6, best * 1e3,
             bytes as f64 / 1e6 / best);
}
