//! `VICTORY_PRELOAD=1`: the pipelines an earlier run asked for, loaded into the
//! library's memory before the first frame asks for them.
//!
//! A pipeline is loaded where it is first asked for, which in the world is
//! inside a frame: that draw is dropped and the next frames carry the load or
//! the compile (`dropped 1586 without a host pipeline`). The manifest this
//! module writes at dump time — the lines `victory precompile` rebuilds, for
//! exactly the keys the early list holds, with each stage's container beside
//! them — lets a later run load them at startup instead: **load only, never a
//! create**, so a preload of pipelines this machine's library does not hold
//! costs a counted miss rather than an hour of driver compiles.
//!
//! The pass is bounded: past [`BUDGET_MS`] the rest are left to the frames that
//! ask for them and the line says how many were cut, so a slow startup is
//! visible rather than mysterious.

use std::path::Path;
use std::sync::Arc;

use super::Shader;
use super::host::{GraphicsState, HostPipeline};
use super::library;

/// The manifest, beside the library it describes.
const MANIFEST: &str = "preload.txt";
/// The stages' containers, beside the manifest.
const CONTAINERS: &str = "preload";
/// What a startup pass may spend; the rest is left to the frames that ask.
const BUDGET_MS: u128 = 5_000;

/// Whether runs preload at all: `VICTORY_PRELOAD=1`.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRELOAD").is_some_and(|v| v == "1"))
}

/// Appends the pipeline `key` to the manifest, and its stages' containers
/// beside it, when it is one an earlier run asked for (the early list) and
/// preloading is on.
pub(crate) fn record(shaders: &[Arc<Shader>], state: Option<&GraphicsState>, key: &str) {
    if !enabled() || !library::wanted_early(key) {
        return;
    }
    let Some(dir) = library::cache_dir() else { return };
    let containers = dir.join(CONTAINERS);
    if std::fs::create_dir_all(&containers).is_err() {
        return;
    }
    for shader in shaders {
        let path = containers.join(format!("{}-{}.bin", shader.stage, shader.hash));
        if !path.exists() {
            let _ = std::fs::write(&path, &shader.container);
        }
    }
    let stages: Vec<(String, String)> = shaders
        .iter()
        .map(|s| (s.stage.to_owned(), s.hash.clone()))
        .collect();
    let line = super::dump::line_of(&stages, state);
    static WRITTEN: std::sync::Mutex<Option<std::collections::HashSet<String>>> = std::sync::Mutex::new(None);
    let mut written = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let set = written.get_or_insert_with(|| {
        std::fs::read_to_string(dir.join(MANIFEST))
            .map(|text| text.lines().map(str::to_owned).collect())
            .unwrap_or_default()
    });
    if !set.insert(line.clone()) {
        return;
    }
    use std::io::Write;
    if let Ok(mut out) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(MANIFEST))
    {
        let _ = out.write_all(format!("{line}\n").as_bytes());
    }
}

/// Loads what the manifest lists and the library holds, logging
/// `preloaded N of M, missing K`.
pub(crate) fn run(dir: &Path) {
    let path = dir.join(MANIFEST);
    let Ok(text) = std::fs::read_to_string(&path) else {
        tracing::info!(path = %path.display(), "preload: no manifest here, nothing to preload");
        return;
    };
    let started = std::time::Instant::now();
    let (mut asked, mut preloaded, mut missing, mut cut) = (0usize, 0usize, 0usize, 0usize);
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if started.elapsed().as_millis() > BUDGET_MS {
            cut += 1;
            continue;
        }
        asked += 1;
        let Some((_, Some(stages), state)) = super::precompile::parse(line) else {
            missing += 1;
            continue;
        };
        let state = match state {
            Some(fields) => match GraphicsState::from_fields(fields) {
                Some(state) => Some(Box::new(state)),
                None => {
                    missing += 1;
                    continue;
                }
            },
            None => None,
        };
        let mut built = Vec::with_capacity(stages.len());
        for (stage, hash) in &stages {
            let Some(stage) = super::precompile::static_stage(stage) else {
                built.clear();
                break;
            };
            match std::fs::read(dir.join(CONTAINERS).join(format!("{stage}-{hash}.bin"))) {
                Ok(container) => built.push(Arc::new(Shader::of_container(stage, hash.clone(), container))),
                Err(_) => {
                    built.clear();
                    break;
                }
            }
        }
        if built.is_empty() {
            missing += 1;
            continue;
        }
        if library::while_preloading(|| HostPipeline::new(&built, state.as_deref()).is_some()) {
            preloaded += 1;
        } else {
            missing += 1;
        }
    }
    tracing::info!(
        asked,
        preloaded,
        missing,
        cut,
        ms = started.elapsed().as_millis() as u64,
        "preloaded {preloaded} of {asked}, missing {missing}"
    );
}
