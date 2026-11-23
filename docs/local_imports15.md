# Local import integration: supported behavior and remaining work

This branch integrates upstream PR #4348 (DirectStorage), adapts #6885 (AcpHal heap and contexts),
and adapts the XApu portion of #6981. It does not integrate their successful no-op hardware commands.
No game content or execution is needed by the tests. All work is local; no PR or issue comment was published.

The retail image gains bindings for 15 previously missing imports:

| Module | Imports | Current behavior |
|---|---:|---|
| AcpHal.dll | 6 Apu* | One host-backed heap, allocation/free, live-block checks and reversible 32-bit address mapping. No cached/non-cached separation. |
| AcpHal.dll | 2 shape-context exports | Allocate/zero/free the measured fifth context type. Other nonzero types return E_NOTIMPL. |
| AcpHal.dll | AcpHalCreate | Clears its output and returns E_NOTIMPL. No hardware object is fabricated. |
| xapu.dll | XApuConnect, XApuDisconnect | Allocate/free connection memory. Shared-buffer layout is an inference, not a public contract. |
| xapu.dll | XApuGetResultCount, XApuDequeueResult | No accepted commands or results: count is zero; dequeue returns E_PENDING without writing. Invalid handles are rejected. |
| xapu.dll | XApuEnqueueCommand | Validates the handle and arguments, then returns E_NOTIMPL. Nothing is silently accepted or dropped. |
| XCRDAPI.dll | DStorageGetFactory | Console factory with file/memory reads, ordered queues, status arrays, fences and events. See directstorage.md for unsupported request types and ABI assumptions. |

These are 15 bindings, not 15 complete emulated APIs. Hardware audio processing remains absent.
The two explicit unsupported entry points and the restricted queue behavior remain blockers even if
`--dry-run` no longer lists them as missing. A passing dry run proves binding only, not gameplay.

The address base used for APU addresses, cachedness interpretation and part of the shape-context
structure are hypotheses carried from the upstream implementation. Only the fifth context type is allocated.
Pointer ranges are compared numerically; only live allocations map; failed allocation clears both outputs;
unknown flags and alignments over the host allocation granularity are rejected. Context replacement
allocates first so a failure does not destroy the current contexts. XApu rejects duplicate disconnects.

No entry point, game TLS callback or game DllMain is invoked during validation. Runtime tests use synthetic files.
The active shader translation uses its separate original source directory and is unaffected by this branch.

Validation commands on Linux with the existing Windows cross toolchain and Wine prefix:

```text
cmake --build <windows build> --target gta6rt gta6rt_tls_host acphal_test xapu_test dstorage_test xmem_test gameinput_test unimplemented_policy_test --parallel 2
ctest --test-dir <windows build> -R "^(acphal|xapu|dstorage|xmem|gameinput|unimplemented_policy)$" --timeout 60 --output-on-failure
python3 -I tests/test_build_runtime_fallback.py
python3 -I scripts/check_no_game_data.py <new sources and tests>
git diff HEAD --check
```

The six tests above are targeted checks. The complete GPU and translator suites were not run for this import-only change.

## Local validation result

Linux, clang-cl 21 + xwin SDK 10.0.26100, Wine 10.0:
all six selected CTests pass. The final AcpHal test also covers unknown flags and
checks invalid alignment on a live allocation. The fallback-build suite passes
with its Windows-only PowerShell case skipped; Linux CMake configuration passes.
The local dry-run binding report goes from 22 missing imports to 7 (five chad DSP
exports and two xg texture-layout exports). This does not execute game code.

The upstream timing test used PAGE_GUARD to hold its copy worker. On this Wine
installation VirtualProtect reported success and VirtualQuery reported 0x104,
but the test did not receive the expected vectored exception. Changing memmove to
an intrinsic did not change that result and was reverted. The timing test now
uses an explicit before-copy hook compiled only into dstorage_test: the pending,
submitted-byte, completion and copied-content assertions remain active. Production
copy behavior is unchanged. Memory access error cases remain in the tests.
