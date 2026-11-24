//! `victory precompile`: build the pipeline library from a dump instead of
//! from a title run.
//!
//! A title writes the dump with `VICTORY_PIPELINE_DUMP` and the containers with
//! `VICTORY_SHADER_SAVE`; this command replays both through the runtime's own
//! pipeline creation, so the library it writes is the library the title would
//! have written for itself. It is a GPU session like any other: it creates a
//! device, compiles shaders and stores pipelines, so it takes the same care
//! with the card that the title does, and it writes the cache's own
//! adapter-named library file, so a `VICTORY_D3D12_WARP` run cannot leave the
//! card a blob it will reject with `0x887E0001`.

use std::path::PathBuf;

use anyhow::{Context, Result};

pub fn run(
    dump: PathBuf,
    shaders: Option<PathBuf>,
    cache: Option<PathBuf>,
    save_every: Option<usize>,
    dry_run: bool,
) -> Result<()> {
    let shaders = shaders
        .or_else(|| dump.parent().map(|dir| dir.to_path_buf()))
        .filter(|dir| !dir.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."));
    let summary = victory_gpu::pipeline::precompile::Dump::read(&dump, &shaders)
        .with_context(|| format!("reading {}", dump.display()))?;
    println!("dump           {}", dump.display());
    println!("shaders        {}", shaders.display());
    println!("lines          {}", summary.lines);
    println!("  replayable   {}", summary.replayable);
    println!("  unreadable   {}", summary.unreadable);
    println!("  no container {}", summary.missing);
    println!("  stage        {}", summary.unsupported);
    println!(
        "  of which     {} graphics, {} mesh, {} compute",
        summary.graphics, summary.mesh, summary.compute
    );
    if dry_run {
        return Ok(());
    }

    let cfg = victory_core::Config::load()?;
    let cache = match cache {
        Some(cache) => cache,
        None => cfg.cache_subdir("pipelines")?,
    };
    println!("library        {}", cache.display());
    let report = victory_gpu::pipeline::precompile::precompile(
        &dump,
        &shaders,
        &cache,
        save_every.unwrap_or(50),
    )
    .with_context(|| format!("replaying {}", dump.display()))?;
    println!();
    println!("lines          {}", report.lines);
    println!("  in library   {}", report.loaded);
    println!("  made         {}", report.made);
    println!("  refused      {}", report.refused);
    println!("  untranslated {}", report.untranslated);
    println!("  unreadable   {}", report.unreadable);
    println!("  no container {}", report.missing);
    println!("  stage        {}", report.unsupported);
    println!(
        "library bytes  {} -> {}",
        report.before, report.after
    );
    Ok(())
}
