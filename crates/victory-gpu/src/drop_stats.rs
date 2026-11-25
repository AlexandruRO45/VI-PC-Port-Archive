//! One record per rejected host draw/dispatch, including indirect calls.
//! Totals and reasons are derived from one snapshot, not separate counters.

use std::collections::BTreeMap;
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Reason {
    NoHostList,
    NoPipelineSet,
    PipelinePending,
    ShaderTranslation,
    PipelineRefused,
    #[cfg(feature = "diagnostics")]
    ShaderSkipped,
    PipelineKind,
    Bind(&'static str),
    RootView(&'static str),
    Scratch(&'static str),
    Unsupported(&'static str),
    Indirect(&'static str),
    Unknown,
}

pub(crate) type Outcome = Result<(), Reason>;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    Direct,
    Indirect,
}

#[derive(Default)]
struct Census([BTreeMap<Reason, u64>; 2]);

impl Census {
    fn record(&mut self, kind: Kind, reason: Reason) {
        *self.0[kind as usize].entry(reason).or_default() += 1;
    }

    fn reasons(&self, kind: Kind) -> &BTreeMap<Reason, u64> {
        &self.0[kind as usize]
    }

    fn total(&self, kind: Kind) -> u64 {
        self.reasons(kind).values().sum()
    }
}

static DROPS: Mutex<Census> = Mutex::new(Census([BTreeMap::new(), BTreeMap::new()]));

pub(crate) fn record(kind: Kind, reason: Reason) {
    DROPS.lock().unwrap_or_else(|e| e.into_inner()).record(kind, reason);
}

pub(crate) fn log(frames: u64) {
    let drops = std::mem::take(&mut *DROPS.lock().unwrap_or_else(|e| e.into_inner()));
    let direct = drops.total(Kind::Direct);
    let indirect = drops.total(Kind::Indirect);
    tracing::info!(
        frames, direct, indirect,
        outstanding_psos = crate::pipeline::making(),
        "host draw/dispatch drops"
    );
    for kind in [Kind::Direct, Kind::Indirect] {
        for (&reason, &count) in drops.reasons(kind) {
            tracing::info!(?kind, ?reason, count, "host draw/dispatch drop reason");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_census_reconciles_each_kind_and_resets_windows() {
        let mut census = Census::default();
        for (kind, reason) in [
            (Kind::Direct, Reason::PipelinePending),
            (Kind::Direct, Reason::Bind("table")),
            (Kind::Direct, Reason::Bind("table")),
            (Kind::Direct, Reason::RootView("missing")),
            (Kind::Direct, Reason::Scratch("rectangle")),
            (Kind::Indirect, Reason::PipelinePending),
            (Kind::Indirect, Reason::ShaderTranslation),
            (Kind::Indirect, Reason::Unknown),
        ] {
            census.record(kind, reason);
        }
        let first = std::mem::take(&mut census);
        assert_eq!(first.total(Kind::Direct), 5);
        assert_eq!(first.total(Kind::Indirect), 3);
        assert_eq!(first.reasons(Kind::Direct)[&Reason::Bind("table")], 2);
        assert_eq!(first.reasons(Kind::Direct)[&Reason::PipelinePending], 1);
        assert_eq!(first.reasons(Kind::Indirect)[&Reason::PipelinePending], 1);
        assert_eq!(first.reasons(Kind::Indirect)[&Reason::Unknown], 1);
        assert_eq!(census.total(Kind::Direct), 0);
        assert_eq!(census.total(Kind::Indirect), 0);
        census.record(Kind::Direct, Reason::NoPipelineSet);
        assert_eq!(census.total(Kind::Direct), 1);
        assert_eq!(first.total(Kind::Direct), 5);
    }
}
