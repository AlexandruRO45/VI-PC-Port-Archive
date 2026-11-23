# RT-374 process-exit prototype

Review branch based on PR #3790, commit `bb561f532b42588838bf2a7d5a73c9fd64d9e4aa`.
This extends that PR's guest-thread startup and explicit rollback. It still needs integration with current main,
preserving main's newer guest atexit support. It is not ready for merging.

## Change

The first exit caller freezes loader mutation, publishes the exit status through the existing done-event protocol,
and waits. After the host prints its diagnostics, `finish_run` releases that caller to native `ExitProcess`.
The primary TLS helper forwards native process detach to the armed Loader, with mappings and TLS still live.
This preserves the exiting thread's TLS and lets Windows stop the other threads before the mapped callbacks run.

The process-termination path detaches DLLs before EXE TLS, passes the native non-null reserved pointer to DllMain,
and keeps TLS callback reserved null, matching the synthetic native oracle. It avoids the TLS registry lock and
startup tracing after native termination starts. Explicit unload/failed-startup rollback keeps its existing order
and null reserved pointer. Detach remains idempotent.

TLS helpers stay mapped through terminal notifications. Their clone files are reclaimed by the existing stale
clone sweep on a later multi-module load; they are not explicitly unloaded before process detach.

## Reproduce the focused test

From a native Windows x64 development environment, with Python available to CMake:

```powershell
cmake -S runtime -B runtime/build -A x64
cmake --build runtime/build --config RelWithDebInfo --target process_detach_driver
ctest --test-dir runtime/build -C RelWithDebInfo -R '^process_detach$' --output-on-failure
```

The driver uses the production PE loader and process shim, independently of graphics. The fixture EXE and DLL are
compiled from the sources alongside it. The standard-library Python runner compares native and mapped executions.
An optional `--evidence DIR` argument to `check_process_detach.py` saves per-case records. Repeated race cases
currently overwrite their earlier per-case JSON records; every iteration is still checked and reported.

## Local results

Windows 11 x64, MSVC 19.44, CPython 3.13.1, RelWithDebInfo:

- 93 subprocesses passed in the native oracle, including 20 independent exit races per implementation.
- Six ordinary exit routes, configured abort with fault reporting disabled, TerminateProcess, entry/worker exits,
  recursive ExitProcess, map-only loading, and ExitThread exit-status preservation were exercised.
- A worker heartbeat stayed stable during native/fixed detach. The early-detach negative control observed a
  changing heartbeat, confirming that the check detects callbacks running before worker quiescence.
- Actual `gta6rt` built with CMake/MSVC. The CPU build disabled Vulkan discovery and downloaded no SDL dependency.
- 7/7 selected CTest tests passed: loader, process_detach, loader_tls, process_cmdline, startup_rollback,
  tls_startup_native, tls_startup_mapped.
- 24 additional `gta6rt --run` processes passed: eight exit modes across entry, entry-with-worker, and worker-exit
  scenarios, checking callback order/thread/TLS and reserved-pointer behavior.
- The text writer line-ending suite passed, 3/3. Microsoft Defender lab source/binary scans reported no threats,
  with no antivirus exclusions.

All fixtures and evidence are synthetic. No game data, GPU path, or Wine was used.

## Review of the existing RT-374 PR

PR #5161 is already implementing RT-374. A separate local harness linked its exact production PE loader,
registry, process shim and TLS helper at `cca7612264e7443c9c13dde59dca1ada2cce6adf`, with the startup/shutdown
wiring from that head's main.cpp. This was a focused harness, not a build of that PR's full gta6rt target.
The same source-built synthetic EXE/DLL from this branch recorded worker progress during mapped DllMain detach:

| Exit route | Exit status | Heartbeat before detach wait | Heartbeat after 30 ms |
| --- | --- | --- | --- |
| `_exit` | 23 | 1 | 2 |
| `_Exit` | 23 | 1 | 4 |
| `quick_exit` | 23 | 1 | 3 |
| `ExitProcess` | 23 | 1 | 4 |

The worker remained able to execute during the shutdown callback. Native termination and this prototype kept
the heartbeat stable in the corresponding checks. This branch is offered as a review experiment for that existing
work; ownership and the integration approach remain with the authors/coordinator.

## Remaining qualification

- Coordinate RT-374 ownership/integration with the #3790 author before opening a follow-up PR.
- Integrate with current main without replacing its newer CRT/atexit changes with the older dependency snapshot.
- Audit/stress termination while workers own loader/registry lookup locks and detach callbacks reenter those APIs.
  The mutation barrier alone does not establish safety for arbitrary concurrent module churn.
- ExitThread is checked only for exit status/no hang. There is no live exit caller whose TLS can be retained;
  the TLS-heavy native fixture itself faults on that route, so it is not a TLS-parity oracle.
- Default abort/report-fault behavior, Wine, and a retail executable are not qualified.

References: [ExitProcess](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-exitprocess),
[DllMain](https://learn.microsoft.com/en-us/windows/win32/dlls/dllmain).
