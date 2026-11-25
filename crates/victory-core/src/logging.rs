//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.
//!
//! A copy of every line also goes to `%LOCALAPPDATA%/victory/data/victory.log`
//! (or `VICTORY_LOG_FILE`). That file is truncated at process start so one
//! run is one file. Writes are synchronous so a TDR log is on disk before
//! the process exits. Set `VICTORY_LOG_FILE=off` to skip it.

use std::path::PathBuf;
use std::sync::Mutex;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

/// The most verbose level this build can log.
const CAP: LevelFilter = if cfg!(feature = "diagnostics") {
    LevelFilter::TRACE
} else {
    LevelFilter::INFO
};

pub fn init(default_level: &str) {
    let filter = std::env::var("VICTORY_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .ok()
        .and_then(|v| EnvFilter::try_new(v).ok())
        .unwrap_or_else(|| EnvFilter::new(default_level));
    let file = open_log_file().map(|file| {
        tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_ansi(false)
            .with_writer(Mutex::new(file))
    });
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .with(file)
        .try_init();
}

fn open_log_file() -> Option<std::fs::File> {
    let path = match std::env::var_os("VICTORY_LOG_FILE") {
        Some(v) if v.is_empty() || v == "off" || v == "0" => return None,
        Some(v) => PathBuf::from(v),
        None => {
            let base = std::env::var_os("LOCALAPPDATA")
                .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
            base.join("victory").join("data").join("victory.log")
        }
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok()?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&path)
        .ok()?;
    eprintln!("VIctory: logging to {}", path.display());
    Some(file)
}
