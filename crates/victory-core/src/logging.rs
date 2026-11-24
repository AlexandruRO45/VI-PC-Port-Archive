//! One tracing setup for every binary.
//!
//! Filter with `VICTORY_LOG` (falls back to `RUST_LOG`), e.g.
//! `VICTORY_LOG=victory_loader=trace,info`. Debug and trace output exist
//! only in builds with the `diagnostics` feature; a plain release build
//! logs info and above whatever the filter asks for. A filter that does not
//! parse is reported on stderr and the default level is used instead, and
//! so is one asking for debug or trace output from a build that cannot
//! produce it. A variable that is set but empty counts as unset.

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
    let (filter, warning) = choose_filter(
        std::env::var("VICTORY_LOG").ok(),
        std::env::var("RUST_LOG").ok(),
        default_level,
    );
    let capped = capped_warning(&filter, CAP);
    // The subscriber is not installed yet, so these cannot go through it.
    for warning in warning.into_iter().chain(capped) {
        eprintln!("victory: {warning}");
    }
    let _ = tracing_subscriber::registry()
        .with(CAP)
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_writer(std::io::stderr),
        )
        .try_init();
}

/// The filter from `VICTORY_LOG`, else `RUST_LOG`, else `default_level`.
/// When the variable that was chosen does not parse, the default is used and
/// the second value says why, so a typo is not silently ignored.
fn choose_filter(
    victory_log: Option<String>,
    rust_log: Option<String>,
    default_level: &str,
) -> (EnvFilter, Option<String>) {
    // `VICTORY_LOG=` (or `RUST_LOG=`) would otherwise be an empty filter
    // that hides everything but errors, and would shadow the other variable.
    let set = |v: Option<String>| v.filter(|v| !v.trim().is_empty());
    let chosen = set(victory_log)
        .map(|v| ("VICTORY_LOG", v))
        .or_else(|| set(rust_log).map(|v| ("RUST_LOG", v)));
    match chosen {
        None => (EnvFilter::new(default_level), None),
        Some((name, value)) => match EnvFilter::try_new(&value) {
            Ok(filter) => (filter, None),
            Err(e) => (
                EnvFilter::new(default_level),
                Some(format!(
                    "ignoring {name}={value:?} ({e}); logging at {default_level:?}"
                )),
            ),
        },
    }
}

/// Says so when `filter` asks for more detail than `cap` lets through, so
/// `VICTORY_LOG=trace` on a build without the `diagnostics` feature does not
/// look like the code has nothing to say.
fn capped_warning(filter: &EnvFilter, cap: LevelFilter) -> Option<String> {
    let wanted = filter.max_level_hint()?;
    (wanted > cap).then(|| {
        format!(
            "the log filter asks for {wanted} output, but this build logs {cap} and above; \
             rebuild with `--features diagnostics` for debug and trace"
        )
        .to_lowercase()
    })
}

#[cfg(test)]
mod tests {
    use super::{EnvFilter, LevelFilter, capped_warning, choose_filter};

    fn s(v: &str) -> Option<String> {
        Some(v.to_owned())
    }

    #[test]
    fn unset_uses_the_default_quietly() {
        let (filter, warning) = choose_filter(None, None, "info");
        assert_eq!(filter.to_string(), "info");
        assert_eq!(warning, None);
    }

    #[test]
    fn victory_log_wins_over_rust_log() {
        let (filter, warning) = choose_filter(s("victory_loader=trace"), s("warn"), "info");
        assert_eq!(filter.to_string(), "victory_loader=trace");
        assert_eq!(warning, None);
    }

    #[test]
    fn rust_log_is_the_fallback() {
        let (filter, warning) = choose_filter(None, s("warn"), "info");
        assert_eq!(filter.to_string(), "warn");
        assert_eq!(warning, None);
    }

    #[test]
    fn a_bad_filter_is_reported_and_the_default_used() {
        let (filter, warning) = choose_filter(s("victory_loader=loud"), None, "info");
        assert_eq!(filter.to_string(), "info");
        let warning = warning.expect("a warning");
        assert!(warning.contains("VICTORY_LOG"), "{warning}");
        assert!(warning.contains("victory_loader=loud"), "{warning}");
    }

    #[test]
    fn a_bad_rust_log_names_rust_log() {
        let (_, warning) = choose_filter(None, s("x=nope"), "info");
        assert!(warning.expect("a warning").contains("RUST_LOG"));
    }

    #[test]
    fn empty_values_count_as_unset() {
        let (filter, warning) = choose_filter(s(""), s("warn"), "info");
        assert_eq!(filter.to_string(), "warn");
        assert_eq!(warning, None);
        let (filter, warning) = choose_filter(s(" "), s(""), "info");
        assert_eq!(filter.to_string(), "info");
        assert_eq!(warning, None);
    }

    #[test]
    fn a_filter_above_the_cap_is_reported() {
        let filter = EnvFilter::new("victory_loader=trace,info");
        let warning = capped_warning(&filter, LevelFilter::INFO).expect("a warning");
        assert!(warning.contains("trace"), "{warning}");
        assert!(warning.contains("diagnostics"), "{warning}");
        assert_eq!(capped_warning(&filter, LevelFilter::TRACE), None);
        assert_eq!(
            capped_warning(&EnvFilter::new("warn"), LevelFilter::INFO),
            None
        );
        assert_eq!(
            capped_warning(&EnvFilter::new("info"), LevelFilter::INFO),
            None
        );
    }
}
