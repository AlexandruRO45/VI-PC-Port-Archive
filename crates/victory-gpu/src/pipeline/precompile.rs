//! `victory precompile`: builds the pipeline library the title would have
//! built for itself, from a pipeline dump and the shader containers saved
//! beside it.
//!
//! A cold session compiles ~1,050 pipelines at 2.1 a second, 2,719 ms of driver
//! create plus 604 ms of translation each, and the title waits on them while it
//! draws. Every one of those descriptions is gone when the process exits: the
//! library keeps the compiled PSOs but nothing outside a title run can name
//! them. `VICTORY_PIPELINE_DUMP` writes them down ([`super::dump`]) and
//! `VICTORY_SHADER_SAVE` writes the containers, so this module can replay each
//! line through [`HostPipeline::new`] — the same function the runtime uses, so
//! the root signature, the description, the name and the store are the runtime's
//! own and not a second implementation of them.
//!
//! A line is replayed by loading it from the library if it is already there: a
//! second run of this command therefore skips what the first stored, and a run
//! killed halfway keeps what it had. The library is saved as it goes, for the
//! same reason: 48 minutes of driver time must survive an interruption.

use std::path::Path;
use std::sync::Arc;

use super::Shader;
use super::host::{GraphicsState, HostPipeline};
use super::library;

/// What a dump holds: how many lines, and which of them can be replayed.
#[derive(Default, Debug)]
pub struct Dump {
    pub lines: usize,
    /// Lines whose shaders and state are all present and readable.
    pub replayable: usize,
    /// Lines cut short or from another fields version.
    pub unreadable: usize,
    /// Lines whose `<hash>.<stage>.cdxil` is not in the shader directory.
    pub missing: usize,
    /// Lines naming a stage a host pipeline is not made from.
    pub unsupported: usize,
    /// Pipelines the dump names, by kind.
    pub graphics: usize,
    pub mesh: usize,
    pub compute: usize,
}

impl Dump {
    /// Reads `path`, looking for containers in `shaders`.
    pub fn read(path: &Path, shaders: &Path) -> std::io::Result<Dump> {
        let text = std::fs::read_to_string(path)?;
        let mut dump = Dump::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            dump.lines += 1;
            let Some((kind, stages, state)) = parse(line) else {
                dump.unreadable += 1;
                continue;
            };
            match kind {
                "graphics" => dump.graphics += 1,
                "mesh" => dump.mesh += 1,
                "compute" => dump.compute += 1,
                _ => {}
            }
            let Some(stages) = stages else {
                dump.unreadable += 1;
                continue;
            };
            if state.is_some_and(|text| GraphicsState::from_fields(text).is_none()) {
                dump.unreadable += 1;
                continue;
            }
            let mut usable = true;
            for (stage, hash) in &stages {
                if static_stage(stage).is_none() {
                    dump.unsupported += 1;
                    usable = false;
                    break;
                }
                if !shaders.join(format!("{hash}.{stage}.cdxil")).is_file() {
                    dump.missing += 1;
                    usable = false;
                    break;
                }
            }
            if usable {
                dump.replayable += 1;
            }
        }
        Ok(dump)
    }
}

/// What a run of the replay made of the dump.
#[derive(Default, Debug)]
pub struct Report {
    pub lines: usize,
    pub unreadable: usize,
    pub missing: usize,
    pub unsupported: usize,
    /// Shaders whose translation refused: the pipeline cannot be made, and the
    /// runtime reaches the same refusal by itself.
    pub untranslated: usize,
    /// Pipelines the library already held (a miss the title still sees is the
    /// proof it did not hold them: these are the ones it will not recompile).
    pub loaded: usize,
    pub made: usize,
    /// Pipelines the host refused to create.
    pub refused: usize,
    /// Bytes the library held when the run started and ended.
    pub before: u64,
    pub after: u64,
}

/// Replays `dump` into the pipeline library kept in `cache`, saving it every
/// `save_every` pipelines so an interrupted run keeps what it stored.
///
/// The library file is the cache's own, named for the adapter it was built on
/// (`host-pipelines.bin` on the card, `host-pipelines-warp.bin` on the software
/// adapter), so a run under `VICTORY_D3D12_WARP` can never write a blob the card
/// would reject with `0x887E0001`.
pub fn precompile(
    dump: &Path,
    shaders: &Path,
    cache: &Path,
    save_every: usize,
    jobs: usize,
) -> std::io::Result<Report> {
    let text = std::fs::read_to_string(dump)?;
    super::use_pipeline_cache(cache.to_path_buf());
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let before = library::BLOB.load(std::sync::atomic::Ordering::Relaxed);
    // Workers pull lines off one cursor: a line's driver compile is seconds
    // long, and the runtime itself makes pipelines on several threads, so one
    // line at a time would take two hours over the world's ~1,950.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let jobs = jobs.clamp(1, 16);
    let reports = std::sync::Mutex::new(Vec::with_capacity(jobs));
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                let mut report = Report::default();
                let mut since_save = 0usize;
                while let Some(at) = (next.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
                    .lt(&lines.len())
                    .then(|| next.load(std::sync::atomic::Ordering::Relaxed) - 1)
                {
                    replay(lines[at], shaders, &mut report);
                    since_save += 1;
                    if since_save >= save_every {
                        // Saved as it goes: 1,950 pipelines are more than one
                        // session's worth of driver time, and a run that is
                        // killed must keep what it stored.
                        library::save(true);
                        since_save = 0;
                    }
                }
                reports.lock().unwrap_or_else(|e| e.into_inner()).push(report);
            });
        }
    });
    library::save(true);
    let mut report = Report {
        before,
        after: library::BLOB.load(std::sync::atomic::Ordering::Relaxed),
        ..Report::default()
    };
    for one in reports.into_inner().unwrap_or_else(|e| e.into_inner()) {
        report.lines += one.lines;
        report.unreadable += one.unreadable;
        report.missing += one.missing;
        report.unsupported += one.unsupported;
        report.untranslated += one.untranslated;
        report.loaded += one.loaded;
        report.made += one.made;
        report.refused += one.refused;
    }
    Ok(report)
}

/// Replays one dump line into the library, counting what became of it.
fn replay(line: &str, shaders: &Path, report: &mut Report) {
    report.lines += 1;
    let Some((_, stages, state)) = parse(line) else {
        report.unreadable += 1;
        return;
    };
    let Some(stages) = stages else {
        report.unreadable += 1;
        return;
    };
    let state = match state {
        Some(text) => match GraphicsState::from_fields(text) {
            Some(state) => Some(Box::new(state)),
            None => {
                report.unreadable += 1;
                return;
            }
        },
        None => None,
    };
    let mut built = Vec::with_capacity(stages.len());
    for (stage, hash) in &stages {
        let Some(stage) = static_stage(stage) else {
            report.unsupported += 1;
            return;
        };
        match std::fs::read(shaders.join(format!("{hash}.{stage}.cdxil"))) {
            Ok(container) => built.push(Arc::new(Shader::of_container(stage, hash.clone(), container))),
            Err(_) => {
                report.missing += 1;
                return;
            }
        }
    }
    if built.iter().any(|s| s.translated().is_none()) {
        report.untranslated += 1;
        return;
    }
    let loaded = library::LOADED.load(std::sync::atomic::Ordering::Relaxed);
    let made = library::MADE.load(std::sync::atomic::Ordering::Relaxed);
    match HostPipeline::new(&built, state.as_deref()) {
        Some(_) => {
            if library::LOADED.load(std::sync::atomic::Ordering::Relaxed) > loaded {
                report.loaded += 1;
            } else if library::MADE.load(std::sync::atomic::Ordering::Relaxed) > made {
                report.made += 1;
            }
        }
        None => report.refused += 1,
    }
}

/// The stage names a host pipeline is made from, as the statics `Shader` holds.
fn static_stage(stage: &str) -> Option<&'static str> {
    match stage {
        "vertex" => Some("vertex"),
        "pixel" => Some("pixel"),
        "mesh" => Some("mesh"),
        "compute" => Some("compute"),
        _ => None,
    }
}

/// One dump line: its kind, its `stage:hash` pairs, and its state text.
///
/// The state is everything after `state=`, not one whitespace-separated field:
/// it is a list of numbers with spaces between them, so a line cannot be read by
/// splitting the whole of it on spaces.
fn parse(line: &str) -> Option<(&str, Option<Vec<(&str, String)>>, Option<&str>)> {
    let (head, state) = line.split_once(" state=")?;
    let state = (state != "-").then_some(state);
    let mut kind = None;
    let mut stages = None;
    for field in head.split(' ') {
        match field.split_once('=') {
            Some(("kind", v)) => kind = Some(v),
            Some(("stages", v)) => {
                let mut list = Vec::new();
                for pair in v.split(',') {
                    let (stage, hash) = pair.split_once(':')?;
                    list.push((stage, hash.to_owned()));
                }
                stages = Some(list);
            }
            // A field this version does not know: the line is not one of ours.
            _ => return None,
        }
    }
    Some((kind?, stages, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_is_its_kind_its_stages_and_its_state() {
        let (kind, stages, state) = parse("kind=compute stages=compute:deadbeef state=-").unwrap();
        assert_eq!(kind, "compute");
        assert_eq!(stages.unwrap(), vec![("compute", "deadbeef".to_owned())]);
        assert!(state.is_none());
        let (kind, stages, state) =
            parse("kind=graphics stages=vertex:aa,pixel:bb state=1 2 3").unwrap();
        assert_eq!(kind, "graphics");
        assert_eq!(stages.unwrap().len(), 2);
        // The state is the rest of the line: its own spaces are inside it.
        assert_eq!(state, Some("1 2 3"));
    }

    #[test]
    fn a_line_this_does_not_know_is_refused_rather_than_guessed() {
        // No state at all: the line was cut before its end.
        assert!(parse("kind=graphics stages=vertex:aa").is_none());
        // A field from another version.
        assert!(parse("colour=graphics stages=vertex:aa state=-").is_none());
        // A stage pair with no hash in it.
        assert!(parse("kind=graphics stages=vertex state=-").is_none());
        // An empty state is a compute line that lost its state marker.
        assert!(parse("kind=compute stages=compute:aa state=").is_some_and(|(_, _, s)| s == Some("")));
    }
}

