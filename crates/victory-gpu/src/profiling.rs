//! Optional numeric telemetry, independent of logging filters. The presenting
//! thread never waits for disk: a bounded queue drops records under pressure.
//! This records application Present cadence, not display scan-out or GPU time.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::time::Instant;

const CAPACITY: usize = 256;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const COLUMNS: [&str; 22] = [
    "sequence",
    "elapsed_ns",
    "interval_ns",
    "frames",
    "width",
    "height",
    "draws",
    "dropped_draws",
    "indirect",
    "prepare_us",
    "fence_wait_us",
    "copy_us",
    "upload_bytes",
    "texture_us",
    "texture_hashed_bytes",
    "back_us",
    "back_bytes",
    "shader_hits",
    "shader_misses",
    "pipelines_made",
    "pipelines_loaded",
    "lost_records",
];

#[derive(Clone, Copy)]
struct Sample {
    kind: &'static str,
    values: [u64; 22],
}

/// Existing one-second counters, before averaging per frame. These are
/// aggregate thread durations and may overlap, especially in relay mode.
pub(crate) struct Interval {
    pub frames: u64,
    pub duration_ns: u64,
    pub draws: u64,
    pub dropped_draws: u64,
    pub indirect: u64,
    pub submission_us: [u64; 2],
    pub preparing: [u64; 4],
    pub back: [u64; 2],
}

pub(crate) struct Profile {
    tx: Option<SyncSender<Sample>>,
    start: Instant,
    previous: Option<Instant>,
    sequence: u64,
    lost: u64,
}

impl Profile {
    pub fn from_env() -> Option<Self> {
        let dir = std::env::var_os("VICTORY_PROFILE_DIR")?;
        match Self::open(Path::new(&dir)) {
            Ok(profile) => Some(profile),
            Err(e) => {
                tracing::warn!(error = %e, "frame profiling disabled");
                None
            }
        }
    }

    fn open(dir: &Path) -> anyhow::Result<Self> {
        victory_core::ensure_outside_repo(dir)?;
        // The leaf must be new: never overwrite an earlier run, even partly.
        if let Some(parent) = dir.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::create_dir(dir)?;
        let create = |name| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.join(name))
        };
        let csv = create("samples.csv")?;
        let json = create("samples.jsonl")?;
        let (tx, rx) = sync_channel(CAPACITY);
        std::thread::Builder::new()
            .name("victory-profile".into())
            .spawn(move || {
                if let Err(e) = write_samples(csv, json, rx, MAX_BYTES) {
                    tracing::warn!(error = %e, "frame profiling writer stopped");
                }
            })?;
        Ok(Self {
            tx: Some(tx),
            start: Instant::now(),
            previous: None,
            sequence: 0,
            lost: 0,
        })
    }

    pub fn present(&mut self, now: Instant, width: u32, height: u32) {
        self.sequence += 1;
        let interval = self
            .previous
            .replace(now)
            .map_or(0, |old| ns(now.duration_since(old)));
        let mut values = [0; 22];
        values[..6].copy_from_slice(&[
            self.sequence,
            ns(now.duration_since(self.start)),
            interval,
            1,
            u64::from(width),
            u64::from(height),
        ]);
        self.send(Sample {
            kind: "present",
            values,
        });
    }

    pub fn interval(&mut self, interval: Interval) {
        let (hits, misses) = crate::shader_cache::counts();
        let (made, loaded) = crate::pipeline::cache_counts();
        self.send(Sample {
            kind: "interval",
            values: [
                self.sequence,
                ns(self.start.elapsed()),
                interval.duration_ns,
                interval.frames,
                0,
                0,
                interval.draws,
                interval.dropped_draws,
                interval.indirect,
                interval.submission_us[0],
                interval.submission_us[1],
                interval.preparing[0],
                interval.preparing[1],
                interval.preparing[2],
                interval.preparing[3],
                interval.back[0],
                interval.back[1],
                hits,
                misses,
                made,
                loaded,
                0,
            ],
        });
    }

    fn send(&mut self, mut sample: Sample) {
        let Some(tx) = &self.tx else { return };
        sample.values[21] = self.lost;
        match tx.try_send(sample) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                if self.lost == 0 {
                    tracing::warn!("frame profiling queue full; records lost");
                }
                self.lost += 1;
            }
            Err(TrySendError::Disconnected(_)) => {
                self.tx = None;
                tracing::warn!("frame profiling stopped; output is incomplete");
            }
        }
    }
}

fn ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn encode(sample: Sample) -> (String, String) {
    use std::fmt::Write;
    let mut csv = format!("1,{}", sample.kind);
    let mut json = format!("{{\"schema_version\":1,\"kind\":\"{}\"", sample.kind);
    for (name, value) in COLUMNS.iter().zip(sample.values) {
        write!(csv, ",{value}").unwrap();
        write!(json, ",\"{name}\":{value}").unwrap();
    }
    csv.push('\n');
    json.push_str("}\n");
    (csv, json)
}

fn write_samples(csv: File, json: File, rx: Receiver<Sample>, limit: usize) -> std::io::Result<()> {
    let (mut csv, mut json) = (BufWriter::new(csv), BufWriter::new(json));
    let header = format!("schema_version,kind,{}\n", COLUMNS.join(","));
    csv.write_all(header.as_bytes())?;
    csv.flush()?;
    let mut bytes = header.len();
    for sample in rx {
        let (line, object) = encode(sample);
        if bytes + line.len() + object.len() > limit {
            tracing::warn!(
                limit,
                "frame profiling size limit reached; output is incomplete"
            );
            break;
        }
        csv.write_all(line.as_bytes())?;
        json.write_all(object.as_bytes())?;
        // The worker flushes each pair so completed rows survive a forced
        // benchmark stop. No flush, formatting, or disk I/O on the producer.
        csv.flush()?;
        json.flush()?;
        bytes += line.len() + object.len();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_source_tree_and_existing_run() {
        assert!(Profile::open(Path::new(env!("CARGO_MANIFEST_DIR"))).is_err());
        let temp = tempfile::tempdir().unwrap();
        assert!(Profile::open(temp.path()).is_err());
    }

    #[test]
    fn pressure_does_not_block_and_reports_loss() {
        let (tx, rx) = sync_channel(1);
        let mut profile = Profile {
            tx: Some(tx),
            start: Instant::now(),
            previous: None,
            sequence: 0,
            lost: 0,
        };
        let now = Instant::now();
        profile.present(now, 1280, 720);
        profile.present(now + std::time::Duration::from_millis(20), 1280, 720);
        assert_eq!(rx.recv().unwrap().values[2], 0);
        profile.present(now + std::time::Duration::from_millis(50), 1280, 720);
        let next = rx.recv().unwrap();
        assert_eq!(next.values[0], 3);
        assert_eq!(next.values[2], 30_000_000);
        assert_eq!(next.values[21], 1);
        drop(rx);
        profile.present(
            Instant::now() + std::time::Duration::from_secs(1),
            1280,
            720,
        );
        assert!(profile.tx.is_none());
    }

    #[test]
    fn exports_numeric_rows_and_honors_storage_limit() {
        let temp = tempfile::tempdir().unwrap();
        let csv_path = temp.path().join("samples.csv");
        let json_path = temp.path().join("samples.jsonl");
        let (tx, rx) = sync_channel(2);
        let sample = Sample {
            kind: "present",
            values: [42; 22],
        };
        tx.send(sample).unwrap();
        tx.send(sample).unwrap();
        drop(tx);
        let header_bytes = format!("schema_version,kind,{}\n", COLUMNS.join(",")).len();
        let (row, json) = encode(sample);
        write_samples(
            File::create(&csv_path).unwrap(),
            File::create(&json_path).unwrap(),
            rx,
            header_bytes + row.len() + json.len(),
        )
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(csv_path).unwrap().lines().count(),
            2
        );
        assert_eq!(std::fs::read_to_string(json_path).unwrap(), json);
        assert!(json.contains("\"interval_ns\":42"));
        assert_eq!(row.trim().split(',').count(), COLUMNS.len() + 2);
    }
}
