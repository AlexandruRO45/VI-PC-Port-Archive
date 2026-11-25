# GPU: export optional frame cadence and existing meter counters

## Summary

Add diagnostics-only, opt-in CSV/JSONL capture at successful host Present calls,
reuse the existing meter totals, and provide reproducible Windows capture/summary
scripts. Correct the frame log's GPU-time label to identify CPU fence waiting.

## Problem

One-second frame averages and five-second log lines cannot recover individual
presentation intervals or meaningful 1%/0.1% lows. Existing logs label a CPU-side
submission/fence duration as time on the GPU, which can misdirect optimization.

## Root cause

`Meter::frame` discards the presentation distribution when reducing each window.
The submission timer brackets ExecuteCommandLists, Signal and a CPU wait rather
than reading GPU query timestamps. True per-command timestamp diagnostics already
exist separately and change the submission path.

## Implementation

- Compile the exporter only with diagnostics; require `VICTORY_PROFILE_DIR` to enable it.
- Count S_OK Present results, excluding failures and occlusion status.
- Export individual cadence/source dimensions plus raw existing meter-window totals
  and cumulative shader/PSO counts. Keep the existing counter resets and render work.
- Use a 256-record nonblocking producer queue and a worker for formatting/disk I/O.
  Report losses/failures and cap combined output at 64 MiB. Refuse source-tree and
  existing run destinations rather than overwriting captures.
- Add a benchmark wrapper with executable/Git/environment/hardware metadata,
  process sampling, optional device-wide NVIDIA sampling, and bounded process lifetime.
- Add a summarizer with median frame time, explicit tail-mean lows and nearest-rank percentiles; reject
  loss, gaps and partial required fields. Document observer effects and warm-up.

## Performance impact

This contribution enables measurement; it does not claim a gameplay speedup.
Plain release contains no exporter. Diagnostics with export disabled performs an
optional-state check at Present. Enabled export adds a timestamp and bounded enqueue
per presentation plus sampling of existing counts once per meter window. Actual
observer overhead still needs repeated, controlled off/on scene comparisons.

## Benchmark methodology

Reference: `90b529f878a470fec9bf07b7be316bc769a12232`, preserved diagnostic executable.
Local hardware: i7-8700K, RTX 5070 Ti (16,303 MiB reported), 32 GB RAM, Windows 11,
driver 621.01, Rust 1.98.1, DXC 1.9. Separate caches/outputs remain outside the source
tree. The existing shader cache keys by executable content: warm each binary separately.
See BENCHMARKING.md for controlled scene, save, resolution, cache and repetition rules.

## Before and after

Before: text averages and ambiguous GPU-time wording; no per-Present structured output.
After: numeric cadence/window records and accurately named CPU fence waits.
A 180-second runtime smoke capture produced 5,265 successful Present records and
173 interval records with no reported loss; every CSV field matched its JSONL peer.
Visual inspection identified an onboarding/legal menu at source resolution 3840x2160.
The 60-150 second menu window measured 30.0003 Present/s; this is not a gameplay
result or a before/after speed comparison. Further world observations belong in the
local progress report, with their scene and cache limitations stated explicitly.

## Tests

- Reference release/diagnostic builds succeeded; reference normal suite: 496 passed,
  0 failed, 7 ignored. Reference Clippy passed in an isolated exact-commit source copy.
- Candidate release and diagnostic builds succeeded.
- `cargo test --workspace --locked`: 496 passed, 0 failed, 7 ignored.
- `cargo test --workspace --all-features --locked -- --test-threads=1`: 507 passed,
  0 failed, 7 ignored. Includes three new exporter tests.
- GPU diagnostics in isolation: 120 passed, 1 ignored, both serial and parallel retry.
- Both CI Clippy configurations passed with `-D warnings`; OneDrive incremental
  cache hard-link fallback warnings appeared in a later run (no lint failure).
- Synthetic PowerShell distribution, loss, sequence-gap and partial-row checks passed.
- Asset policy and diff whitespace checks passed. Global fmt already fails at the
  reference; the new module passes rustfmt with edition 2024. No mass reformatting.

## Compatibility

No new external crates or CPU/GPU feature requirements. The added direct internal
dependency supplies the existing source-tree output guard. NVIDIA monitoring is an
optional script argument; it is not a runtime dependency. Existing scripts and
experimental graphics settings retain their behavior.

## Known limitations

Application Present cadence is not display scan-out or guaranteed unique rendered
frames. The output contains CPU phase totals, not new GPU execution timestamps.
Counters from different threads may overlap. GPU utilization from NVIDIA sampling
is device-wide. Forced termination can lose queued tail records; CSV/JSONL are not
an atomic file pair. Inspect warnings and use complete observed prefixes only.

An initial parallel all-feature run terminated with 0xc0000005; isolated serial and
parallel GPU retries passed. The identical original GPU suite separately showed a
timing-sensitive assertion in `frame.rs:241`, then passed on retry. The access
violation was not reproduced or assigned a root cause; these observations must not
be conflated. Loopback tests require the normal host permissions, not the restricted
execution sandbox. Long gameplay, multiple adapters and profiler overhead remain
additional validation work.

## Potential regressions

Optional telemetry may perturb scheduling or fill its queue under slow storage.
S_OK-only counting changes reported cadence when the swap chain is occluded; it does
not change submitted rendering. Existing consumers parsing the old English fence
label need updating. No synchronization, draw omission or cache-default change is
included. Upstream discussion: https://github.com/Zettrox/ztrx/pull/82.
