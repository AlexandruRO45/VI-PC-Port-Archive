# Runtime PSO queue and memory pressure

`host pipeline queue sample` is logged every five seconds after the first
runtime PSO request. It does not depend on frame submission. It shows:

- `available_mib`: system physical memory available now.
- `pages_input_s`: system-wide PDH `Memory\Pages Input/sec`, the interval
  average of pages read from disk to resolve hard faults. This is not a
  per-process count or a count of fault events. `None` means unavailable.
- Outstanding title jobs, fresh urgent and early jobs, jobs deferred by
  physical memory pressure or either DXIL lane, and lane occupancy.
- Active worker phases as `phase:count/oldest-ms`. Interval hits, misses,
  and driver creates are host PSO operations (including both mesh fallback
  passes); misses include retries. `attempts` and worker CPU/wall totals
  cover completed job attempts, including deferrals.

The runtime pauses **speculative cold creation** below 1,536 MiB of
available physical memory and resumes it at 2,048 MiB. An early request
can still obtain a cache hit; a draw/dispatch can promote a deferred job.
Urgent work, the existing large-PSO lane limits, and the wait budget are
unchanged. A compile already inside the driver is not interrupted. The
standalone `translate --cache --pipelines` command does not use this gate.

## Dropped host commands

`host draw/dispatch drops` reports exact counts of direct and indirect
commands dropped since the previous report, with `frames` giving the
number of presents in that window. The `host draw/dispatch drop reason`
lines sum to each total. One indirect command counts once even when it
would execute many records. A native or fallback draw can fail during
binding or scratch allocation even if the PSO is ready.

`PipelinePending` means the PSO was not ready when that command tried to
use it. `outstanding_psos` is a separate snapshot at report time, so it
may read zero after a window with pending-PSO drops. Translation and
terminal pipeline refusals have separate reasons; resource or binding
failures are not reported as missing PSOs.
