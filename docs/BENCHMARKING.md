# Reproducible frame profiling

Build with `cargo build --profile diag -p victory --features diagnostics --locked`.
Structured profiling is compiled only with diagnostics and is inactive unless
`VICTORY_PROFILE_DIR` names a new directory outside every source tree. Plain release
has no profiler, channel or per-frame export. Do not compare different build profiles.

Review `scripts/benchmark.ps1` before running it. It samples process CPU totals,
working set/private bytes, saves hardware/driver/environment/executable SHA-256 and
Git revision/dirty state, and stops only the process it launches at the deadline.
It sets info logging and restores the environment it changes. Run directories must
be new, and cache/output/save drives must have at least 16 GiB free at launch.
Existing experimental environment settings are recorded, not silently normalized.
Optionally pass `-NvidiaSmi 'C:\Windows\System32\nvidia-smi.exe'` to collect
device-wide GPU/memory utilization and VRAM use once a second. This requires that
installed NVIDIA utility, applies to all workloads on the device, and adds observer
overhead. Other adapters require their own external sampler; the runtime has no
NVIDIA-specific dependency. Monitor errors are retained in `gpu-monitor.log`.

```powershell
scripts/benchmark.ps1 -Game '<authorized executable>' -Output 'D:\ztrx-runs\candidate-01' -Cache 'D:\ztrx-cache\candidate' -Data 'D:\ztrx-saves\benchmark' -Seconds 300 -Scene 'fixed scene, resolution and camera' -CacheState warm -Profile
scripts/summarize-profile.ps1 -InputCsv 'D:\ztrx-runs\candidate-01\profile\samples.csv' -StartSeconds 60 -EndSeconds 240
```

Specify `-Exe` for a preserved reference binary. `-Keys` optionally uses the existing
diagnostic input schedule (seconds after input starts); it is not a deterministic
scene replay. Set up equivalent saves outside the repo, verify the actual scene and
resolution, and avoid first-boot/menu transitions inside the measured window.
The script never deletes caches or copies saves. `-CacheState` is a recorded label:
use a genuinely empty/new cache for cold runs, and a separate conditioning run for
each executable before warm runs. Runtime shader keys include executable content,
so a reference cache does not make a rebuilt candidate warm.

Run reference/candidate and candidate-off/on interleaved at least three times with
the same scene, cache state, settings and hardware. Finish builds and GPU tests
before captures. Capture an uninstrumented control to quantify observer overhead.
Record driver resets, dropped draws and visual differences as well as FPS. A short
boot run is a smoke test; it does not establish sustained gameplay performance.

## Export schema 1

`samples.csv` and `samples.jsonl` contain the same numeric records; JSONL is one JSON
object per line, allowing partial capture recovery. `kind=present` has sequence,
elapsed/interval nanoseconds, frames=1 and source width/height. The first interval
is zero and excluded from summaries. Counted calls return S_OK; failed or occluded
Present calls are excluded. This is successful application Present cadence, not
scan-out FPS, unique rendered images, or guaranteed displayed frames. Use PresentMon
for display/presentation modes when installed. Width/height describe the source.

`kind=interval` contains raw one-second meter-window totals (the first window may
include preparation before the first successful Present): frames, draws, dropped
draws, indirect commands, preparation/fence-wait/copy/texture/back microseconds,
upload bytes, texture bytes hashed, and bytes returned. `shader_hits/misses` and
`pipelines_made/loaded` are cumulative, not deltas. Fields not applicable to a kind
are zero; zero is not a measurement of missing GPU execution time. Upload bytes
are the existing title-memory-copy counter, not all D3D12 bus traffic; hashed texture
bytes are not texture-upload bytes. CPU durations can overlap and must not be summed
as a partition of frame wall time. There is no new timestamp instrumentation here.

The worker formats/writes/flushes records off the presenting thread. Queue capacity
is 256; total CSV+JSONL output is capped at 64 MiB. `lost_records` is cumulative;
sequence gaps and loss invalidate percentiles. A warning reports queue saturation,
writer failure or size exhaustion. Inspect stderr before accepting a summary.
Forced termination can lose queued tail records or leave a partial final line;
only complete records describe the observed prefix, not the entire requested run.
The files are not an atomic pair on disk failure. This exporter is not an asset cache.

The summarizer uses reciprocal mean interval for FPS, nearest-rank frame-time
percentiles, and reciprocal mean of the slowest ceil(N*1%) / ceil(N*0.1%) intervals
for lows. It excludes intervals crossing the selection boundary and refuses reported
loss/gaps. Small samples, especially fewer than 1,000 intervals, poorly estimate 0.1%
lows; the output flags this. Do not compute these metrics from one-second averages.
Selection times are relative to profiler creation, not process launch.

## Validation and tools

Run `scripts/test-profile-summary.ps1` for synthetic percentile/loss checks and
`cargo test -p victory-gpu --features diagnostics profiling --locked` for exporter
policy, queue, encoding and storage-limit checks. Full CI commands are in
`.github/workflows/ci.yml`. Global fmt failures already exist in the reference tree;
keep unrelated formatting out of the contribution.

MSVC/Windows SDK/DXC are installed on the development PC. PIX, Nsight Graphics and
PresentMon were not found in PATH or the checked conventional installation roots;
no claim is made that a portable/custom installation cannot exist. The process
sampler does not attribute per-process VRAM, GPU execution or I/O. Collect these
with a verified external profiler in the next controlled scene experiment.
