//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for.

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
    // Wine's console does not read ANSI escapes, and a redirected stream
    // shows them as rubbish; colour only on a terminal of genuine Windows
    // (or of the Linux host, where there is no Wine under our feet).
    let ansi = std::io::IsTerminal::is_terminal(&std::io::stderr())
        && crate::platform::wine_version().is_none();
    let console = std::env::var_os("VICTORY_LOG_CONSOLE")
        .is_some_and(|v| v == "1")
        // CONOUT$ reaches the Wine console window even when the process'
        // own stdout/stderr are redirected to a log file.
        .then(|| std::fs::OpenOptions::new().write(true).open("CONOUT$").ok())
        .flatten();
    let file = std::env::var("VICTORY_LOG_FILE").ok().filter(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .is_ok()
    });
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(ansi)
                .with_writer(std::io::stderr),
        )
        .with(console.map(|c| {
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(false)
                .with_writer(move || c.try_clone().expect("CONOUT$ clone"))
        }))
        .with(file.map(|p| {
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(false)
                .with_writer(move || {
                    std::fs::OpenOptions::new()
                        .append(true)
                        .open(&p)
                        .expect("log file")
                })
        }))
        .try_init();
}
