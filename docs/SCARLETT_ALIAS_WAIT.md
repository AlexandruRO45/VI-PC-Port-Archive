# Bounded placed-texture alias waits

The main branch at cb185e0 degrades distinct placed-image overlaps instead
of poisoning the queue. Its two-second CPU wait was per overlapping
publication, and closed an event after a timeout even though D3D12 has
no cancellation API for the pending fence registration.

A synthetic blocked-writer control with two different slice images
reproduced two timeout diagnostics and a VKD3D failed-to-signal-event
error (E_FAIL), followed by guest fence completion failure. This is a
synthetic reproduction, not an attribution of a retail crash.

Share a lazy wall-clock budget across distinct-image publication waits
within one submission. Query completed values first, including after
budget exhaustion; poll incomplete fences in intervals of at most 2 ms
instead of registering an event that might outlive its handle. Preserve
the existing stale-image degradation on timeout, same-image GPU waits,
and publication order. This bounds requested CPU sleep/wait time, not
arbitrary driver calls or OS scheduling overhead.

Build with build.py d3d12 and use tools/check_d3d12_regressions.py
--case alias-budget --case copy-resource --case placement. The new
control is opt-in. It holds the writer behind a guest fence, reads two
distinct overlapping slices, releases the writer, verifies the earlier
image bytes, then verifies the newly published bytes. It checks the
shared wait time (1.5–3.5 s window to allow scheduling variation), two
timeout diagnostics and the absence of event signalling failures.

On RTX 4080 / NVIDIA 615.78.8 / Proton Experimental, both patched runs
observed 1999 ms. The earlier baseline did not complete its fence,
so no exact before/after speedup is claimed. 209 Python tests, two
standalone Rust budget tests, Windows-target D3D12 typecheck and full
provider build passed. Three GPU controls passed on the final provider.
No AMD or retail execution claim. The baseline fixture preceded a
minor change to timing-print placement and the order of two fence
waits; the baseline failure is not an exact paired timing measurement.

## Final synthetic receipt

```json
{
  "schema": 1,
  "scope": "Synthetic D3D12 provider controls; not retail execution",
  "adapter": "10de:2704",
  "proton": "1794497090 experimental-11.0-20261112-x86_64",
  "artifacts_sha256": {
    "d3d12_xs.dll": "51a93a737fc8ecfd3ac4ef495f987e9e0c9eddd257b6773cd21cdd81e370f116",
    "d3d12.dll": "fc2bbef02509fb11c334cc96dd211f64def94bb7b4eb82082a4808e6e0549543",
    "d3d12core.dll": "9356501320114773be279506644aabd9a1717ff451d480ac802a4fd4e94b2576",
    "dxgi.dll": "2814edc3350cae6d34a75b7900df19752889c310ef34255c7316c4a95e1b183f",
    "dxcompiler.dll": "0689e2fee0a1c9df8662a59680469f24e2a3ad7ce6bbbcc9284f7a83172bfc59",
    "dxil.dll": "64be2368211d257a038a7d4ea6bb3e24c878eb8d588c7b8414fccf759b69c2ac",
    "d3d12-copy-resource.exe": "fca9a89cda388e155d94b0654afd0b6788afcd6a1ed2aa8189b2948be3e49046",
    "d3d12-placement.exe": "2841d8c2e9ed1d863edccdf291cc39db43c768874eaee7da0503a52184d5ca28"
  },
  "unchanged": true,
  "cases": [
    {
      "case": "alias-budget",
      "passed": true,
      "exit": 0,
      "seconds": 14.92,
      "receipt": "d3d12-alias-budget PASS pci=10de:2704 overlaps=2 old=verified refreshed=verified shared_budget=1"
    },
    {
      "case": "copy-resource",
      "passed": true,
      "exit": 0,
      "seconds": 21.29,
      "receipt": "d3d12-copy-resource PASS pci=10de:2704 slot=84 buffer_bytes=1028 texture_subresources=8 nonzero_option=E_NOTIMPL self_copy=E_INVALIDARG source_and_destination=verified placed_uploads=2 placed_dst=verified same_va_alias=verified cpu_writeback=verified partial_copy_texels=2 overlap_mips=4 pending_reader=verified"
    },
    {
      "case": "placement",
      "passed": true,
      "exit": 0,
      "seconds": 18.04,
      "receipt": "d3d12-placement PASS pci=10de:2704 slot=44 VA-offsets=0,4096,0 passes=2 BC1-blocks=522 cube-blocks=16404 R8-texels=262144 rejected=3 configured_vm=0"
    }
  ],
  "passed": true
}
```
