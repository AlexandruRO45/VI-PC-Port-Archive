# The retail boot to the title's frame loop on Windows + NVIDIA

**Status: measured on one machine, 2026-11-23 (night). A measurement for RT-5, not a claim; RT-5 stays with its
owners on the coordination issue.** It continues [retail_boot_chain.md](retail_boot_chain.md), which ends at stop 13,
and #8166's section 5, which ends at the BVH builder. Each row below is the first failure with every fix above it in
place. The same rule as the boot chain holds: nothing here quotes the executable (no addresses, offsets, IID values or
bytes); only export names, public SDK types, vtable slot numbers, call shapes and counts.

Setup: `main` `0d31cdd64` with the pull requests below merged locally, in this order. Windows 11 26200, RTX 4070 SUPER
(NVIDIA 616.56), Ryzen 7 5700X3D, MSVC 2022 `RelWithDebInfo`, `LEONIDA_D3D12=vulkan`, own 1.0.2.1 copy, outbound
network blocked, `--overlay` so the package is never written. No translated shaders: make_game's shader stage was not
run on this machine.

Stack: #6582, #4313 (now #8379), #4348 (now #8461), #6964, #6954, #7076, #7075 with #6953, #7021, #7066, #7078,
#7052, #7110, #7235 (its own commit only, see below), #7516, #7163, #7115, #7323, #7330, #6891 (since merged),
#7201, #7755, #7286, #7403, #7417 and #7290. The physical-page stops are #6919's, already on `main`.

## 1. The stops after stop 13

| Order | Stop | What clears it |
|---|---|---|
| 13 | Heap slot 9: the title expects its descriptor heap at an address it chose | #7021 |
| 14 | Device slot 70 (a view from raw words) forwarded to the PC slot 70 | #7066 |
| 15 | Device slot 25 called with the return buffer first | #7078; #7110 cleans the descriptions |
| 16 | Device slot 127 is past the PC table | #7052 |
| 17 | An assert right after the first full-screen render targets: the title's allocation hook never saw their memory | #7235's own commit (slot 127 takes the memory through the title's XMem hook) |
| 18 | `XboxBvhBuilder.dll` throws a C++ exception at its first compute pipeline (here: no translated shaders) | #7516 serves the seven exports with the layer's builder |
| 19 | Device slot 41: the third of five startup command signatures has argument type 105 and a 12-byte stride, refused with `E_INVALIDARG`; then the fifth, a 12-byte `DISPATCH_MESH`, is `E_NOTIMPL` | #7163, then #7330 |
| 20 | Device slot 61 (`CreateSamplerX`) forwarded to the PC `CreateMetaCommand`: access violation in the runtime | #7115 |
| 21 | Device slot 74 after 99,776 physical pages: forwarded to the PC `ShaderCacheControl`, which fails | #7323 |
| 22 | `xg_xs.dll`'s two imports answered by the default policy (0); the title reads through the result | #6891 |
| 23 | Device slot 108 forwarded past the PC table (86 and 101 the same) | #7201 |
| 24 | `GetResourceAllocationInfo` for `A8_UNORM` (format 65): a 1D texture and a render target | #7755, #7403 |
| 25 | Queue slot 35 (`ExecuteCommandListsX`) answered as console-only, so nothing reaches the queue | #7286 |
| 26 | One assert right after the title's window is created, with or without pipeline stubs | Open; stepped over with `LEONIDA_SKIP_INT3=1` |
| 27 | Fence slot 12: `E_NOTIMPL` returned where the title reads the address of the completed value | #7290 |
| 28 | **Current stop**: `acphal.dll!ApuCreateHeap` and `AcpHalAllocateShapeContexts` are unimplemented, and the title writes through the result | Not tried: #7009 serves AcpHal |

About 50,000 graphics pipelines (device slot 86) have no translated shader here. With `LEONIDA_STUB_PIPELINES=1`
(#7417) they are created as no-op stubs, which is what lets the run go past stop 26. Device slot 67's 42 FSPX compute
programs fail the same way, and slot 57's lifetime trackers are refused; the title goes on after both.

## 2. What the frame loop did

With `LEONIDA_STUB_PIPELINES=1` and `LEONIDA_SKIP_INT3=1` (one assert stepped over, stop 26), one 18-second run
recorded:

- 50,471 `CreatePipelineState` calls (stubs) and 41 `CreateCommittedResource3` resources;
- 340 `ExecuteCommandListsX` submissions reaching the real queue, each with a `ClearRenderTargetView`;
- 1,015 recorded `DrawInstanced` calls, which draw nothing because their pipelines are stubs;
- 344 fence slot 12 calls, one `PresentX` setup on the title's window;
- a peak of 5.7 GiB private memory and 94 threads.

The frames are black: the pipelines are stubs, so this shows the submission path, not rendering.

## 3. Two fixes the stack needs that no single pull request has

- **#7286 with #7076.** #7286's `queue_execute_command_lists` calls the PC slot 10 with the caller's array, and it
  also takes the PC slot 10 over. With #7076's list proxy, slot 10 is where list proxies are unwrapped, so both
  slots 10 and 35 would hand proxies to the real queue. Unwrapping inside `queue_execute_command_lists` fixes both.
  #7286's `check_xbox_queue_slots` then needs readable mock lists instead of made-up pointers, and compares the
  entries instead of the array pointer (the PC slot gets the unwrapped copy).
- **#7115 with #7021.** `CreateSamplerX` passes the title's descriptor handle as it is. With #7021, a handle in the
  title's heap range has to be translated, so the call goes through slot 22's handler, which does that.

Both are posted on the two pull requests. #7235 was applied as its own commit, on top of the #7021 and #7052 merge
it was written against, instead of as the whole stacked branch; the rest of the stack merges with conflicts only in
the shared slot lists and test lists.

## 4. The CPU

The executable uses four instructions outside the x86-64 baseline (the RT-500 scan: SSE4a `INSERTQ`, `CLZERO`,
`MONITORX`/`MWAITX`, `CLWB`). Zen 3 has all four, and nothing on this Zen 3 X3D part stopped or behaved differently
from the reports on other machines.

## 5. What this does not show

A drawn frame, a present with content, audio, or anything past stop 28. Stop 26's assert is stepped over, not
explained. The counts are one run's.
