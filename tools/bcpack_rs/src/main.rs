#![deny(clippy::all)]

mod bc;
mod dds;
mod rans;
mod swizzle;
mod tables;
mod ttpf;

use std::path::{Path, PathBuf};
use std::{env, fs, process};

fn usage(prog: &str) {
    eprintln!("Usage: {prog} <ttpf_file> [--out <dir>] [--mip <n>] [--entry <idx>]");
    eprintln!("  --out   output directory (default: same as input)");
    eprintln!("  --mip   extract only this mip level (default: 0)");
    eprintln!("  --entry extract only this entry index (default: all)");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        usage(&args[0]);
        process::exit(1);
    }

    let mut ttpf_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut mip_filter: Option<u32> = Some(0);
    let mut entry_filter: Option<usize> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(&args[i]));
            }
            "--mip" => {
                i += 1;
                mip_filter = Some(args[i].parse().unwrap_or(0));
            }
            "--entry" => {
                i += 1;
                entry_filter = Some(args[i].parse().unwrap_or(0));
            }
            s if !s.starts_with("--") => {
                ttpf_path = Some(PathBuf::from(s));
            }
            other => {
                eprintln!("Unknown flag: {other}");
                usage(&args[0]);
                process::exit(1);
            }
        }
        i += 1;
    }

    let ttpf_path = match ttpf_path {
        Some(p) => p,
        None => {
            usage(&args[0]);
            process::exit(1);
        }
    };

    let out_dir = out_dir.unwrap_or_else(|| {
        ttpf_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    });
    fs::create_dir_all(&out_dir).expect("cannot create output dir");

    let data = fs::read(&ttpf_path).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {e}", ttpf_path.display());
        process::exit(1);
    });

    let pack = ttpf::TtpfPack::parse(&data).unwrap_or_else(|e| {
        eprintln!("TTPF parse error: {e}");
        process::exit(1);
    });

    eprintln!(
        "TTPF: {} entries, format version {}",
        pack.entries.len(),
        pack.version
    );

    for (idx, entry) in pack.entries.iter().enumerate() {
        if let Some(ef) = entry_filter {
            if idx != ef {
                continue;
            }
        }

        eprintln!(
            "Entry {idx}: {}x{} fmt={:?} mips={} codec=0x{:02x}",
            entry.width, entry.height, entry.dxgi_format, entry.mip_count, entry.bcpack_codec
        );

        let mip = mip_filter.unwrap_or(0).min(entry.mip_count.saturating_sub(1));

        let decoded = match decode_entry(&data, entry, mip) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  skip entry {idx} mip {mip}: {e}");
                continue;
            }
        };

        let stem = ttpf_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let out_path = out_dir.join(format!("{stem}_{idx:04}_mip{mip}.dds"));

        let mip_w = (entry.width >> mip).max(1);
        let mip_h = (entry.height >> mip).max(1);

        let dds_bytes = dds::write_dds(mip_w, mip_h, entry.dxgi_format, &decoded);
        fs::write(&out_path, &dds_bytes).unwrap_or_else(|e| {
            eprintln!("Write error: {e}");
        });
        eprintln!("  written: {}", out_path.display());
    }
}

fn decode_entry(
    pack_data: &[u8],
    entry: &ttpf::TtpfEntry,
    mip: u32,
) -> Result<Vec<u8>, String> {
    let chunk = entry
        .mip_chunk(pack_data, mip)
        .ok_or("mip chunk out of bounds")?;

    let tiled = bc::bcpack_decode(chunk, entry.bcpack_codec, entry.width >> mip, entry.height >> mip)
        .map_err(|e| format!("bcpack: {e}"))?;

    let mip_w = (entry.width >> mip).max(1);
    let mip_h = (entry.height >> mip).max(1);
    let linear = swizzle::detile(
        &tiled,
        &swizzle::DetileParams {
            width: mip_w,
            height: mip_h,
            bytes_per_block: entry.bytes_per_block(),
            block_dim: entry.block_dim(),
            tile_mode: entry.tile_mode,
            d0: entry.d0,
            slices: entry.slices,
        },
    )
    .map_err(|e| format!("detile: {e}"))?;

    Ok(linear)
}
