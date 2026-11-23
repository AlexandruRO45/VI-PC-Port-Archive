# Related projects and prior art

Leonida is **GPL-3.0-or-later**. Some useful reverse-engineering work lives under other licences, and some of it
cannot be mixed with this repository. This page records what we may learn from each project and where the line is.

The rule is the same for every entry: **facts are fine; code is not.** A format layout, an ABI observation or a
measured behaviour can be reimplemented independently. The source that expressed it cannot be copied.

## rpatty-dev/PercyStar

<https://github.com/rpatty-dev/PercyStar>

A source-available Rust explorer for RPF9/TTPF, with an RDNA2 disassembler, texture decoders and an `ADAT` Opus audio
decoder. Its format observations are relevant to the archive, texture and audio work here.

**Licence:** the "PercyStar Non-Commercial Attribution & Copyleft License (v1.2)". It is source-available, says
"All rights reserved", and adds non-commercial, attribution, naming and share-alike conditions. It is not
GPL-compatible, so **no PercyStar code, comments or tables may be copied into Leonida**.

**What we use it for:** reading about format facts, then implementing them independently. If a fact is useful,
record the observation in our own words, cite the project and verify it against the package or a public
specification. Do not paste even a small function "to save time".

## realdody/americas-recomp

<https://github.com/realdody/americas-recomp>

A **GPL-2.0-only** fork of [boykopovar/AnyPS5](https://github.com/boykopovar/AnyPS5), on its `xbox-series` branch. It
recompiles the same Xbox Series shaders and has run the package's DLLs against a PC D3D12 device behind an Xbox-ABI
proxy. Since 2026-11-22 its pull requests are limited to collaborators and its README points contributors here.

**Licence:** the repository is **GPL-2.0-only** (`LICENSE`), which is **incompatible with GPL-3.0-or-later**: GPLv2-only
code cannot be combined into a GPLv3 work, so none of it may be copied, and there is no "just this one file" exception.
Since 2026-11-22 its `REUSE.toml` on `xbox-series` (commit `f1c25d72`) lists **88 files under MIT**, each relicensed with
the agreement of everyone who wrote lines in it ([realdody/americas-recomp#179](https://github.com/realdody/americas-recomp/issues/179),
still open, so the list can change): 35 files at the top of its tools/xbox directory, 30 in tools/xbox/platform (for
example `xbox_dxil_lower.py`, `d3d12_xs.cpp`, `xmem.cpp`, `xbox_device_proxy.h`), one each under tools/xbox/msvc and
tools/xbox/platform/xgr, its 20 tools/tests/test_xbox_*.py tests and its docs/xbox/BVH_CONTAINER.md. Everything else,
including the recompiler in its core directory and the `xgameruntime` stand-in sources, stays GPL-2.0-only.

**Rule until the maintainer decides:** **no americas-recomp code is copied into Leonida**, MIT-listed or not. MIT is
compatible with GPL-3.0-or-later, so the listed files could be used the way `third_party/xtaskqueue` is (per file, only
paths listed in `REUSE.toml` at the commit used, the MIT notice kept); whether Leonida does that is the maintainer's
decision, not a contributor's.

**What we use it for:** the measured ABI facts only. The useful findings are summarised in
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi): the Xbox device entry point, the slot
deviations, the descriptor-heap start-handle return convention, the root-signature flag word and the BVH builder's
compute-pipeline path. Those are observations to verify, not code to port. The document paraphrases them; no code or
long passages are copied.

## breadbbbb/rpf9-explorer

<https://github.com/breadbbbb/rpf9-explorer>

A Windows viewer for RPF archives, and the source of the Core library behind `tools/rpf9dump` (models, textures,
TTPF pools and AWC audio). It is a `third_party` **submodule**, not code copied into this repository.

**Licence:** the repository has **no licence file** (GitHub reports no licence), so all rights are reserved by
default. A submodule is an external dependency; it is not a grant to copy its code. Do not paste its functions,
tables or comments into Leonida. If a format fact learned from it is useful, verify it against the package and
write our own implementation. The model-recovery gap measured in [models.md](models.md) is in that submodule; the
fix should go upstream, not into a copy.

## meetox80/6stars

<https://github.com/meetox80/6stars> — a macOS Swift/Metal engine that renders the world out of RPF9,
with a Swift `RPFKit` format library and 44 Python research scripts. Independent work: it likewise
does not decode the TOC.

**Licence: none.** The repository has no `LICENSE` file (GitHub reports `license: null`); only the
vendored cimgui/imgui carry MIT. With no grant, default copyright applies and the project is
**all rights reserved** — stricter than PercyStar. **No 6stars code, comment, identifier or table may
be copied into Leonida.**

**What we use it for:** corroboration and a small number of hypotheses. As an independent reading of
the same formats it matches our `CEntityDef` member hashes (9/9), the PRD0 header, the navmesh tile
header, the meshlet group header and the PSIN/LSCH field layout. Its stream-role predicates and
navmesh surface bits are recorded as hypotheses and tested from our own package, never copied.
Where it disagrees with us (ETPF format and tile mode, the RDS5 codec field, `atArray` type `0x5C`,
PSO member hashing) our measurements are the validated ones.
<https://github.com/meetox80/6stars>

A macOS Swift/Metal engine that reads the world from RPF9, with its own archive index and format readers.

**Licence: none.** There is no `LICENSE` file and GitHub reports no licence, so all rights are reserved.
**No 6stars code, comments, identifiers or tables may be copied into Leonida.**

**What we use it for:** corroboration. Its RPF9 stream-boundary predicate and PRD0 member type codes were
re-measured against this package before entering [rpf.md](rpf.md) and [world.md](world.md); its navmesh
road/pavement bit labels did not reproduce and are not used. The navmesh polygon `+4` packing it reads is the one
our decoder already validates ([navmesh.md](navmesh.md)).

## p-portela/SCARAB

<https://github.com/p-portela/SCARAB>

A GPL-3.0 "Scarlett Compatibility And Runtime API Bridge" that forwards the console D3D12 runtime to PC D3D12 (and
to vkd3d-proton on Linux). It carries a measured catalogue of the console `d3d12_xs.dll` export surface, standard
and console-only IIDs, extension interfaces, command-list interception and presentation entry points.

**Licence:** GPL-3.0. That is compatible with this project's GPL-3.0-or-later for study, and adapted code can be
used with attribution and the original licence notice, unlike the two projects above. Even so, **facts first**:
verify a claim against SCARAB's code and, where possible, this package; write the implementation in our style; and
say in the pull request which file a borrowed fact came from. Do not copy a whole component just because the
licence allows it.

**What we use it for:** the measured console runtime surface (see `docs/d3d12_xs_surface.md` when that study
lands), the IID/extension tables, and the call-sequence observations that Leonida's own
[d3d12_vulkan.md §7](d3d12_vulkan.md#7-prior-art-and-measured-xbox-abi) still lists as unknown. SCARAB forwards to
D3D12; Leonida translates to Vulkan. The ABI facts carry over, the forwarding architecture does not.

## holdmysocks/SixNG

<https://github.com/holdmysocks/SixNG>

A Rust extractor for the same package: models, textures, map placements, collision, navmesh, the lane graph and
audio, with format notes in its module headers.

**Licence:** GPL-3.0, compatible with this project's GPL-3.0-or-later. As with SCARAB, **facts first**: nothing of
it is copied here.

**What we use it for:** two navmesh facts, read in `crates/sixng-core/src/navmesh.rs` at commit `0619601` and
re-measured on this package by `tools/navmesh_nodes.py`
([navmesh.md](navmesh.md#polygon-flags-under-the-lane-graph)): its label "road surface" for polygon `+0x18` bit
13, which it bases on a visual check, and the vertex formula `tile minimum + u16 / 65535 * extent`. The formula is
confirmed by the lane-graph heights. The label is consistent with the count, but not settled by it: nine in ten
lane nodes stand on polygons with both bit 8 and bit 13, and which bit or combination means road surface is open.

## Graphics-runtime prior art: vkd3d-proton, DXVK, Wine, RADV, shadPS4, Xenia and SDL3

The D3D12 to Vulkan layer, its present path, its wave64 model and the GameInput shim were compared with these
projects for the first frame. The pins are the commits read in November 2026; every licence is compatible with
GPL-3.0-or-later, and the rule at the top of this page still applies - facts are borrowed, code is not.

### vkd3d-proton

<https://github.com/HansKristian-Work/vkd3d-proton> at `02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc`.
**Licence:** LGPL-2.1-or-later, compatible.

- **Feature answers come from the enabled Vulkan features, and this is the borrowing that can fail a PSO.**
  `DoublePrecisionFloatShaderOps` needs `shaderFloat64` plus preserved FP64 denorms plus independent denorm
  behaviour; `OutputMergerLogicOp` is `logicOp`; `PSSpecifiedStencilRefSupported` is
  `VK_EXT_shader_stencil_export`; `ROVsSupported` is `fragmentShaderPixelInterlock &&
  fragmentShaderSampleInterlock`; `VPAndRTArrayIndexFromAnyShaderFeedingRasterizer...` is
  `shaderOutputViewportIndex && shaderOutputLayer`
  ([device.c, `d3d12_device_caps_init_feature_options`](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/device.c#L9294-L9333)).
  The layer answers those five as `TRUE` unconditionally (`runtime/shims/d3d12_vulkan_device.cpp`, the
  `D3D12_FEATURE_D3D12_OPTIONS` case around lines 6770-6784), while the PSO path separately refuses a logic op
  whose `logicOp` feature is missing (around line 3294). A title that branches on the answer can build a PSO the
  layer then refuses; the answer and the refusal must read the same feature. **First frame: yes** - the first
  VS+PS PSO is the one at risk. [4.1](d3d12_vulkan.md#41-device-and-feature-queries) already states the rule.
- **Feature level follows the same options.** 12_1 needs ROVs and conservative raster tier 1; 12_2 needs mesh,
  RT 1.1, VRS tier 2, sampler feedback and binding tier 3
  ([`d3d12_device_caps_init_feature_level`](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/device.c#L9695-L9731)).
  Answering 12_1 while `OPTIONS5.RaytracingTier` and `OPTIONS7.MeshShaderTier` are "not supported" is coherent;
  answering 12_2 would not be.
- **Wave64 is a hard pipeline requirement, not emulated.** The shader's `cs_wave_size_min/max` is checked against
  `WaveLaneCountMin/Max`, and a PSO outside the range is `E_INVALIDARG`
  ([device.c](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/device.c#L10883-L10903));
  `debug_wave64_simulation` only forces a 64-lane request where the device can grant one. That is the opposite of
  this repo's tier A/B/C warn-and-draw, and it states what "exact" means: a tier C shader on 32-wide hardware is
  not exact, and drawing it is a deliberate first-frame trade.
- **Present.** `DXGI_STATUS_OCCLUDED` is returned when the surface reports a zero `maxImageExtent` (a minimised
  Win32 window), also for `DXGI_PRESENT_TEST`
  ([swapchain.c](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/swapchain.c#L1013-L1047));
  the present is queued on a present thread with a request ring, and the waitable-object flag changes the
  latency from 3 to 1
  ([#L1121-L1134](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/swapchain.c#L1121-L1134),
  [#L1572-L1621](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/swapchain.c#L1572-L1621)).
  On `VK_ERROR_SURFACE_LOST_KHR` it keeps rendering to nothing rather than failing the caller. After #5689 the
  DXGI shim matches the first two; surface lost still comes back as `E_FAIL`, which is the remaining difference.
  **First frame: after a resize or display change only.**
- **UMA** is decided from the memory types: every type except lazily-allocated must be `HOST_VISIBLE`
  (`HOST_COHERENT` too for coherent UMA)
  ([device.c](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/device.c#L4206-L4230)).
  The layer answers `UMA = FALSE` on every GPU, true only on a discrete PC. **First frame: no on a discrete GPU**,
  but it decides the title's upload/heap strategy on an iGPU.
- **Transform feedback and indirect draws are not first-frame work here.** A stream-output PSO needs
  `VK_EXT_transform_feedback` and the `ALLOW_STREAM_OUTPUT` root flag, and fails cleanly without them
  ([state.c](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/state.c#L5688-L5702));
  `ExecuteIndirect` signatures with root constants, VBV/IBV changes or predicates go through a patch shader that
  rewrites the argument buffer before the draw
  ([command.c](https://github.com/HansKristian-Work/vkd3d-proton/blob/02e8c2b221f3dea00dc7cf99c0c1c18554f6c1cc/libs/vkd3d/command.c#L18216-L18502)).
  Keep `D3D12_EXECUTE_INDIRECT_TIER_1_0` until such a path exists. **First frame: no** -
  [6.1](d3d12_vulkan.md#61-minimum-api-set-for-the-first-frame) excludes both.

### DXVK, Wine dxgi and Mesa RADV

**DXVK** <https://github.com/doitsujin/dxvk> at `9e57fa0b2c5b495ba1c890017d659935e5f1df65`, **zlib**. Present
errors the app cannot act on are softened: `VK_ERROR_OUT_OF_DATE_KHR` and `VK_ERROR_SURFACE_LOST_KHR` become
`VK_NOT_READY` and the presenter recreates next frame
([`softError`](https://github.com/doitsujin/dxvk/blob/9e57fa0b2c5b495ba1c890017d659935e5f1df65/src/dxvk/dxvk_presenter.cpp#L2221-L2229)).
Same lesson as vkd3d-proton: an out-of-date present is not a title error. **First frame: after a resize only.**
D3D11 has no wave intrinsics, so DXVK has no wave64 model to borrow.

**Wine** <https://gitlab.winehq.org/wine/wine> at `97027f879ba9a36dbe594720d4710818536e5433`, **LGPL-2.1-or-later**.
D3D12 present is asynchronous: `d3d12_swapchain_present` waits on a frame-latency semaphore for at most 2 s when
the chain is not waitable, queues the work to a worker and returns `S_OK`
([swapchain.c](https://gitlab.winehq.org/wine/wine/-/blob/97027f879ba9a36dbe594720d4710818536e5433/dlls/dxgi/swapchain.c#L2360-L2405));
the worker recreates the swapchain and presents again on `VK_ERROR_OUT_OF_DATE_KHR`
([#L2325-L2350](https://gitlab.winehq.org/wine/wine/-/blob/97027f879ba9a36dbe594720d4710818536e5433/dlls/dxgi/swapchain.c#L2325-L2350)).
`DXGI_PRESENT_TEST` returns `DXGI_STATUS_OCCLUDED` while the window is iconic
([#L341-L349](https://gitlab.winehq.org/wine/wine/-/blob/97027f879ba9a36dbe594720d4710818536e5433/dlls/dxgi/swapchain.c#L341-L349));
the test records that a normal flip-model present still returns `S_OK` while the bitblt model is occluded
([tests/dxgi.c:8558](https://gitlab.winehq.org/wine/wine/-/blob/97027f879ba9a36dbe594720d4710818536e5433/dlls/dxgi/tests/dxgi.c#L8558)).
The shim uses `DXGI_SWAP_EFFECT_FLIP_DISCARD`, so #5689's "skip and return `S_OK`" on a minimised window is the
native answer, not a shortcut.

**Mesa RADV** <https://gitlab.freedesktop.org/mesa/mesa> at `49f13b7fd40bbd5d6659c740a79b0608ae8d28ba`, **MIT**.
RADV models wave size as a pipeline key, not a lowering: `requiredSubgroupSize` must be 32 or 64 and becomes
`RADV_REQUIRED_WAVE32`/`RADV_REQUIRED_WAVE64`
([radv_pipeline.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/49f13b7fd40bbd5d6659c740a79b0608ae8d28ba/src/amd/vulkan/radv_pipeline.c#L150-L160)),
it exposes `subgroupSizeControl`
([radv_physical_device.c](https://gitlab.freedesktop.org/mesa/mesa/-/blob/49f13b7fd40bbd5d6659c740a79b0608ae8d28ba/src/amd/vulkan/radv_physical_device.c#L1140))
and comments that wave64 is recommended for pixel shaders
([#L2873](https://gitlab.freedesktop.org/mesa/mesa/-/blob/49f13b7fd40bbd5d6659c740a79b0608ae8d28ba/src/amd/vulkan/radv_physical_device.c#L2873)).
There is no path that runs a 64-wide shader on a 32-wide wave; on RDNA the width is real. The layer's
`requiredSubgroupSize = 64` is the same mechanism. **First frame: yes on AMD, no change.**

### shadPS4 and Xenia

**shadPS4** <https://github.com/shadps4-emu/shadPS4> at `b7c0f18bc3189065161074cec46f0e246d66c461`, **GPL-2.0**
(the `LICENSE` text is GPLv2; the file read carries a `GPL-2.0-or-later` SPDX header), compatible. It has a light
wave64 lowering for **compute only**: when the guest wave is 64 and the host subgroup is 32, it lowers `Ballot`,
`ReadLane`, `MaskedBitCount32` and `WaveId` inside uniform control flow through a small shared-memory scratch, one
32-bit word per subgroup and one barrier per operation
([lower_wave64_pass.cpp](https://github.com/shadps4-emu/shadPS4/blob/b7c0f18bc3189065161074cec46f0e246d66c461/src/shader_recompiler/ir/passes/lower_wave64_pass.cpp#L104-L200)).
Outside uniform flow it logs and leaves the value wrong. It is a cheaper special case of `--wave64-emulate`, whose
round scheduler is exact for any control flow and any cross-lane op at the cost of a barrier per segment.
**First frame: no** (compute only), but it confirms the two-subgroup + LDS + barrier construction independently.
Its present is a flip queue on a vblank thread: `sceVideoOutSubmitFlip` queues a flip, `flip_rate` presents every
rate+1 vblanks, `sceVideoOutIsFlipPending` reports the depth, more than 16 is
`ORBIS_VIDEO_OUT_ERROR_FLIP_QUEUE_FULL`, index -1 is a blank frame, and vblank events carry the counter
([driver.cpp](https://github.com/shadps4-emu/shadPS4/blob/b7c0f18bc3189065161074cec46f0e246d66c461/src/core/libraries/videoout/driver.cpp#L292-L410),
[video_out.cpp](https://github.com/shadps4-emu/shadPS4/blob/b7c0f18bc3189065161074cec46f0e246d66c461/src/core/libraries/videoout/video_out.cpp#L138-L185)).
This is the closest public analogue of `PresentX`'s token/plane model: a bounded queue, a rate and a blank present.
`FramePacer` has tokens and latency but an unbounded pending deque and no flip rate; if the title presents without
waiting, this is the model that says cap it. **First frame: the queue is empty**, so the cap matters later.

**Xenia** <https://github.com/xenia-project/xenia> at `95a5c3ee250f80c3b9d139658649d9ffb6db3eec`,
**BSD-3-Clause**, compatible. The guest's swap is a command packet, `XE_SWAP`, carrying a front-buffer pointer,
width and height
([command_processor.cc](https://github.com/xenia-project/xenia/blob/95a5c3ee250f80c3b9d139658649d9ffb6db3eec/src/xenia/gpu/command_processor.cc#L915-L937));
the backend's `IssueSwap` refreshes the guest output and the UI thread presents it
([d3d12_command_processor.cc](https://github.com/xenia-project/xenia/blob/95a5c3ee250f80c3b9d139658649d9ffb6db3eec/src/xenia/gpu/d3d12/d3d12_command_processor.cc#L1636-L1670)).
No guest swapchain, no vsync wait in the GPU thread and no guest present return code: the `PresentX` architecture,
where the title owns the colour texture and the host owns the swapchain. **First frame: yes as a design
confirmation**, no code. Its Vulkan presenter treats out-of-date, surface lost and
`VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT` as "connection outdated" (recreate, not an error), with device lost
separate
([vulkan_presenter.cc](https://github.com/xenia-project/xenia/blob/95a5c3ee250f80c3b9d139658649d9ffb6db3eec/src/xenia/ui/vulkan/vulkan_presenter.cc#L1395-L1412),
[#L2062-L2085](https://github.com/xenia-project/xenia/blob/95a5c3ee250f80c3b9d139658649d9ffb6db3eec/src/xenia/ui/vulkan/vulkan_presenter.cc#L2062-L2085)).
The full-screen-exclusive case cannot occur here: the layer never enables `VK_EXT_full_screen_exclusive`.

### SDL3 and the GameInput shim

<https://github.com/libsdl-org/SDL> at `31a04a734f2e8e6f02cb691c70b70a5fa9cbe972`. **Licence:** zlib, compatible.
The public joystick/gamepad API has `SDL_RumbleJoystickTriggers` (trigger rumble), `SDL_SetJoystickLED` (RGB) and
`SDL_SendJoystickEffect` (raw driver effect), with `RGB_LED` and `TRIGGER_RUMBLE` capability properties
([SDL_joystick.h](https://github.com/libsdl-org/SDL/blob/31a04a734f2e8e6f02cb691c70b70a5fa9cbe972/include/SDL3/SDL_joystick.h#L1395-L1432),
[#L770-L794](https://github.com/libsdl-org/SDL/blob/31a04a734f2e8e6f02cb691c70b70a5fa9cbe972/include/SDL3/SDL_joystick.h#L770-L794)).
There is **no typed adaptive-trigger API**: `SDL_SendJoystickEffect` is the only path, and the PS5 driver
implements it as the 47-byte `DS5EffectsState_t` block, with the right and left 11-byte trigger effects at
offsets 10 and 21
([SDL_hidapi_ps5.c](https://github.com/libsdl-org/SDL/blob/31a04a734f2e8e6f02cb691c70b70a5fa9cbe972/src/joystick/hidapi/SDL_hidapi_ps5.c#L164-L187),
[#L1061-L1146](https://github.com/libsdl-org/SDL/blob/31a04a734f2e8e6f02cb691c70b70a5fa9cbe972/src/joystick/hidapi/SDL_hidapi_ps5.c#L1061-L1146)).
`SDL_SendGamepadEffect` is the same call on a gamepad handle.

The GameInput shim already uses it: `SetRumbleState`'s trigger motors become an adaptive-trigger resistance effect
through a 47-byte block with `ucEnableBits1 = 0x04|0x08` (right/left trigger) and the effects at 10/21
(`runtime/shims/gameinput.cpp`, `kDsEffectSize` at line 69 and the block around line 1550), selected by
`LEONIDA_DS_TRIGGERS`, gain and frequency; the lightbar and player LEDs go through `SDL_SetGamepadLED` (around
line 1419). So the shim covers the lightbar and player LEDs, and covers adaptive triggers only as a **derived**
resistance from the Xbox impulse-trigger signal, not as authored PS5 effects. GameInput has no LED or
authored-trigger API of its own, so there is nothing further to map. **First frame: no.**

### Where that leaves the first frame

| Borrowing | Source | Licence | First frame |
|---|---|---|---|
| Feature booleans from enabled Vulkan features, matching the PSO refusals | vkd3d-proton `device.c` | LGPL-2.1-or-later | **yes** |
| Out-of-date present is recreated, not `E_FAIL`; minimised flip-model present is `S_OK` | #5689, vkd3d-proton, Wine | - | after a resize |
| Surface lost softened, not returned as `E_FAIL` | vkd3d-proton, DXVK, Xenia | LGPL-2.1-or-later, zlib, BSD-3-Clause | after a display change |
| Two-subgroup + LDS + barrier wave64 lowering | shadPS4 (compute), this repo (compute/mesh) | GPL-2.0-or-later | no (graphics stages need more) |
| Bounded flip queue and flip rate | shadPS4 `VideoOut` | GPL-2.0-or-later | later |
| Guest owns the texture, host owns the swapchain | Xenia `XE_SWAP` | BSD-3-Clause | design confirmation |
| Adaptive trigger via the 47-byte effect block, lightbar via LED | SDL3 PS5 driver | zlib | no |

The actionable order for the first frame: (1) audit the core `D3D12_OPTIONS` booleans against the same features
the PSO path checks; #5000 is doing the neighbouring heap tier and the core `OPTIONS` booleans are not in
D3D-86's list. (2) Leave present to #5689; the only remaining softening is surface lost. (3) Tier C stays an
accepted approximation: none of the projects read here has an exact graphics-stage wave64 technique beyond
running the stage as compute, and a pixel shader would need a compute rasteriser. (4) No GameInput change.

## microsoft/libHttpClient

<https://github.com/microsoft/libHttpClient>

Microsoft's HTTP/WebSocket library for GDK titles. Its `Source/Task` folder is the implementation of the GDK's
`XTaskQueue` and `XAsync`.

**Licence:** MIT, which is compatible with GPL-3.0-or-later.

**What we use:** the task queue's headers and Win32 sources, copied unmodified into `third_party/xtaskqueue` (commit
and file list in its README) with the licence file. `runtime/shims/xgameruntime.cpp` registers those functions as
`xgameruntime.dll` imports. The HTTP and WebSocket stack is not included. Unlike the projects above, this is code
use under a compatible licence, so the MIT notice must stay with the files.

## AMD machine-readable GPU ISA specification

<https://gpuopen.com/machine-readable-isa> (reader: <https://github.com/GPUOpen-Tools/isa_spec_manager>)

AMD's XML description of the RDNA and CDNA instruction sets: instructions, encodings, operands, data formats.

**Licence:** MIT (`<License>MIT</License>` in each XML file; `isa_spec_manager` is MIT too), compatible with
GPL-3.0-or-later.

**What we use it for:** `tools/isa_coverage.py` reads `amdgpu_isa_rdna2.xml` from a path the user supplies and counts
which instructions the translator has a rule for ([isa_coverage.md](isa_coverage.md)). The XML is not vendored (11 MB);
tests use a small fixture of our own in the same schema.

## Mesa (AMD register database and format tables) and LLVM's AMDGPU backend

<https://gitlab.freedesktop.org/mesa/mesa> (`src/amd/registers/*.json`, `src/amd/common/ac_descriptors.c`),
<https://github.com/llvm/llvm-project> (`llvm/lib/Target/AMDGPU/SIDefines.h`)

**Licence:** Mesa's AMD code is MIT, LLVM is Apache-2.0 with LLVM exceptions. Both are GPL-3.0-or-later compatible,
but nothing is copied.

**What we use:** field positions and enum values, written into this repository's own code. `tools/hw_oracle/tsharp.py`
builds the T# for the hardware oracle of `tools/image_fuzz.py` from Mesa's SQ_IMG_RSRC_WORD1..4 bit ranges
(`gfx10-rsrc.json`, `gfx11-rsrc.json`), the way `ac_build_gfx10_texture_descriptor` and `ac_set_mutable_tex_desc_fields`
fill them (linear images carry their pitch in WORD4), and the GFX10 and GFX11 format numbers (Mesa's `GFX10_FORMAT`,
and for GFX11 the order of LLVM's `UfmtGFX11::UnifiedFormat`). The GFX11 numbers were confirmed on a gfx1150 by the
hardware oracle itself (a wrong FORMAT changes the fetched words); the GFX10 numbers and layout are Mesa's and not yet
run on RDNA2 hardware.

## The line

| Fine | Not fine |
|---|---|
| A format layout, field offset or enum value | Copying the function that parses it |
| An ABI measurement (slot number, return register, flag word) | Copying the proxy, thunk or struct definitions |
| A behaviour observed by running a tool | Copying comments, identifiers or test code |
| A public specification or vendor documentation | A line-by-line port, a translated port, or a "cleaned up" copy |
| A table of facts you rebuilt from your own observations | A table copied from the source, even with renamed columns |

A table of constants is still expression. Rebuild it from your own measurements or a public specification. A
struct definition is still expression even if the field names are changed. If the only way to use a fact is to
copy the surrounding code, stop and find the fact another way.

When in doubt:

1. Write down the fact, not the code.
2. Cite the project and the section or file where you read it.
3. Verify the fact against the game package, a public specification or a test of your own.
4. Implement it independently, in this repository's style.

Never paste code from either project into a pull request, an issue, a commit message or a chat log that becomes
part of the repository history. If a contribution was written while looking at incompatible source, say so in the
pull request so the provenance can be checked.

Other dependencies keep their own licences; see the Credits and License sections of [README.md](../README.md).
