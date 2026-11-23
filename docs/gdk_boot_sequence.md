# A GDK title's boot path to the first `PresentX`, and what this runtime serves

Status: gap analysis, public sources only. This page answers one question: from process start to the first
`PresentX`, which calls does a GDK title make, and does the Vulkan-backed runtime answer each one? It uses
public sample code and public documentation, not the game. Struct layouts, vtable slots and the presentation
core are covered by [d3d12_vulkan.md](d3d12_vulkan.md) section 4.5 and [d3d12_command_objects.md](d3d12_command_objects.md);
this page does not repeat them. The shortest sample's order is the spine; the support matrix says where the
runtime stops.

Sources read on 2026-11-23:

- Microsoft `Xbox-GDK-Samples` at `main`: `Samples/IntroGraphics/SimpleDeviceAndSwapChain` (the shortest
  Microsoft sample that presents) and `Samples/Graphics/FramePacing`. MIT licence.
- DirectXTK12 at `main`: `Src/SpriteBatch.cpp`, `Src/EffectPipelineStateDescription.cpp`,
  `Inc/DirectXHelpers.h`. MIT licence. The `DeviceResourcesGXDK.cpp` in
  `walbourn/directxtk12modelviewer` is the same pattern. MIT licence.
- SDL2 `src/render/direct3d12/SDL_render_d3d12_xbox.cpp` (zlib licence).
- Microsoft's public GDK reference at <https://docs.xbox.com>, as indexed by
  `docs/xbox_d3d12_xs.md`.
- The public HLE `Freminet/ScarlettX` (README and `src/shims/d3d12_xs_shim.cpp`), which reports observed
  Xbox slots from a real title. It is a third-party measurement, not Microsoft documentation; its
  synthetic headers are not the real ABI.
- `hajimehoshi/ebiten`'s Apache-2.0 Go binding `internal/graphicsdriver/directx/d3d12_windows.go`, the only
  public field-by-field declaration of `D3D12XBOX_CREATE_DEVICE_PARAMETERS` found.

Nothing here is from the game, its shaders or its package. A call order read from a sample is **read from
the source**; a slot or ABI value from the public HLE is **reported**, not documented by Microsoft; the
consequences for this runtime are **derived** from its own code.

## 1. The shortest public sequence

`SimpleDeviceAndSwapChain` is the shortest Microsoft sample with a real present path. It is a Win32 program
that calls `XGameRuntimeInitialize`, creates one window, one device, one direct queue, two committed display
buffers, one depth buffer, one SpriteBatch PSO and one `PresentX` per frame. The order below is the order the
code executes, with the call shapes as written.

### 1.1 Process start

| Step | Call | Arguments | Where |
|---:|---|---|---|
| 1 | `wWinMain` | Win32 entry point | `Samples/IntroGraphics/SimpleDeviceAndSwapChain/Main.cpp:161` |
| 2 | `XGameRuntimeInitialize()` | 0 | `Main.cpp:52` |
| 3 | `RegisterClassExA` / `CreateWindowExA` / `ShowWindow` | Win32 | `Main.cpp:76-87` |
| 4 | `RegisterAppStateChangeNotification` | 3 (callback, context, out handle) | `Main.cpp:103` |
| 5 | `RegisterAppConstrainedChangeNotification` | 3 | `Main.cpp:123` |
| 6 | `Sample::Initialize(hwnd)` | 1 | `Main.cpp:96` |

`XGameRuntimeInitialize` is the only Gaming Runtime call before the device. The sample also calls
`ATG::SendLaunchTelemetry()` (`Main.cpp:94`), a sample helper, not a GDK requirement or part of the D3D path.
The sample asserts `GetACP() == CP_UTF8`; it is a PC-side assertion, not an Xbox API.

### 1.2 Device, queue and frame events

`Sample::CreateDevice` (`SimpleDeviceAndSwapChain.cpp:251`) runs this sequence:

| Step | Call | Arguments | Where |
|---:|---|---|---|
| 1 | `D3D12XboxCreateDevice(adapter, params, riid, out)` | 4; adapter is `nullptr`, `params` a named `D3D12XBOX_CREATE_DEVICE_PARAMETERS` | `SimpleDeviceAndSwapChain.cpp:301-304` |
| 2 | `CreateCommandQueue(desc, riid, out)` | 3; one `D3D12_COMMAND_QUEUE_DESC`, type `DIRECT`, flags `NONE` | `:318-322` |
| 3 | `CreateDescriptorHeap(desc, riid, out)` | 3, twice: RTV heap (2 descriptors) and DSV heap (1) | `:324-334` |
| 4 | `GetDescriptorHandleIncrementSize(type)` | 1, twice | `:336-337` |
| 5 | `CreateCommandAllocator(type, riid, out)` | 3, once per back buffer | `:340-343` |
| 6 | `CreateCommandList(nodeMask, type, allocator, initialState, riid, out)` | 6; `nodeMask` 0, `initialState` null | `:345-347` |
| 7 | `Close()` on the list | 0 | `:347` |
| 8 | `CreateFence(initial, flags, riid, out)` | 4; `initial` 0, flags `NONE` | `:349-351` |
| 9 | `QueryInterface(IDXGIDevice1)` | 2 | `:575` |
| 10 | `IDXGIDevice1::GetAdapter(out)` | 1 | `:579` |
| 11 | `IDXGIAdapter::EnumOutputs(0, out)` | 2 | `:583` |
| 12 | `SetFrameIntervalX(output, 60 Hz, bufferCount - 1, flags)` | 4; the third argument is the allowed latency, 1 here because the sample has two buffers | `:586-590` |
| 13 | `ScheduleFrameEventX(ORIGIN, 0, nullptr, flags)` | 4 | `:592-596` |
| 14 | `GraphicsMemory(device)` and `DescriptorHeap(device, N)` | DirectXTK12 helpers; committed upload buffers and a shader-visible heap | `:396-398` |

There is no `CreateSwapChainX` or `CreateSwapChainForHwnd` in this path. The two buffers are ordinary
committed resources and `PresentX` is the presentation call; DXGI is used only for the output object that
`SetFrameIntervalX` takes.

`RegisterFrameEvents` (`:571`) is called at `:393`, before any resource or PSO is created. Steps 9-13 are
therefore on the device-creation path, not a later frame-loop path. A failure in step 12 aborts in
`CreateDevice`, before the first `PresentX`.

FramePacing uses the same device call, then two queues: a PC `CreateCommandQueue` for graphics and an Xbox
`CreateCommandQueueX` for compute (`FramePacing.cpp:1730-1746`). Its `D3D12XBOX_COMMAND_QUEUE_DESC` adds
`EngineOrPipeIndex` and `QueueIndex`. The frame-event calls are identical in shape (`FramePacing.cpp:2036-2046`),
but its `SetFrameIntervalX` accepts a null output because the title chooses no specific display.

### 1.3 First frame resources and PSO

`Sample::CreateResources` (`SimpleDeviceAndSwapChain.cpp:402`) runs:

| Step | Call | Arguments | Where |
|---:|---|---|---|
| 1 | `WaitForGpu()`: `Signal` on the queue, `SetEventOnCompletion`, `WaitForSingleObjectEx` | 2, 2, 3 | `:540-555` |
| 2 | `PresentX(0, nullptr, nullptr)` | 3; the blank-screen present before anything is released | `:408` |
| 3 | `CreateCommittedResource(heapProps, heapFlags, desc, initialState, clear, riid, out)` | 7, twice; `D3D12_HEAP_FLAG_ALLOW_DISPLAY`, `B8G8R8A8_UNORM_SRGB`, `PRESENT` | `:421-447` |
| 4 | `CreateRenderTargetView(resource, desc, cpuHandle)` | 3, per buffer | `:455-462` |
| 5 | `CreateCommittedResource` for `D32_FLOAT` depth | 7; `DEPTH_WRITE` | `:470-495` |
| 6 | `CreateDepthStencilView(resource, desc, cpuHandle)` | 3 | `:499-503` |
| 7 | `ResourceUploadBatch::Begin()` / `CreateDDSTextureFromFileEx` / `End(queue)` | DirectXTK12 | `:506-532` |
| 8 | `SpriteBatch` constructor: two `CreateRootSignature` and one `CreateGraphicsPipelineState` | DirectXTK12; see section 5 | `:529-530` |
| 9 | `WaitForGpu()` again | as step 1 | `:534` |

The first `PresentX` in the whole program is the blank one at `:408`. The first present with a rendered
frame is in `Render` (`:192`):

| Step | Call | Arguments | Where |
|---:|---|---|---|
| 1 | `WaitFrameEventX(ORIGIN, INFINITE, nullptr, flags, &token)` | 5; token starts as `D3D12XBOX_FRAME_PIPELINE_TOKEN_NULL` | `:558-568` |
| 2 | reset allocator, `Reset(allocator, pso)` on the list | 2 | `:126-131` |
| 3 | `ResourceBarrier` PRESENT to RENDER_TARGET | 2 (count, array) | `:162-166` |
| 4 | `OMSetRenderTargets(1, &rtv, FALSE, &dsv)` / `ClearRenderTargetView` / `ClearDepthStencilView` | 4 / 4 / 4 | `:172-178` |
| 5 | `SetDescriptorHeaps`, `SetGraphicsRootSignature`, `SetGraphicsRootDescriptorTable`, `SetPipelineState`, `Draw*` | DirectXTK12 SpriteBatch | `:132-151` |
| 6 | `ResourceBarrier` RENDER_TARGET to PRESENT | 2 | `:196-200` |
| 7 | `Close()`; `ExecuteCommandLists(1, &list)` | 0; 2 | `:202-204` |
| 8 | `PresentX(1, &planeParameters, nullptr)` | 3; `ResourceCount` 1, `ppResources` points at the current buffer, `Token` from step 1 | `:206-213` |

The plane is a plain SDR plane: no `pSrcViewRects`, no `pDestPlacementBase`, default colour space and
filter. Shutdown presents zero planes again (`:68`, `:407`); suspend is `SuspendX(0)` and resume is
`ResumeX()` followed by `RegisterFrameEvents()` again (`:227-243`).

SDL's Xbox backend is the same order in fewer lines: `D3D12XboxCreateDevice` (`SDL_render_d3d12_xbox.cpp:54`),
`QueryInterface(IDXGIDevice1)` (`:60`), `GetAdapter` (`:66`), `EnumOutputs(0)` (`:72`), `SetFrameIntervalX`
(`:79`), `ScheduleFrameEventX` (`:85`), `CreateCommittedResource` with `ALLOW_DISPLAY`/`PRESENT`
(`:122-129`), `WaitFrameEventX` (`:136`), `PresentX(1, ...)` (`:147`). SDL defines the `ID3D12Device1` IID at
`:34` and asks for it at creation (`:54`), while the Microsoft sample uses `IID_GRAPHICS_PPV_ARGS` (the
`ID3D12Device` IID).

## 2. What the runtime implements, stubs or lacks

The Vulkan backend is selected with `LEONIDA_D3D12=vulkan`; without it `D3D12XboxCreateDevice` wraps the
host PC `d3d12.dll` device and the Vulkan layer is not in the path
(`runtime/shims/d3d12_xs.cpp:160-187`). The matrix below is for the Vulkan backend.

Status meanings: **implemented** means the call reaches working code; **core only** means the code exists
but no vtable slot or proxy calls it; **wrong slot** means the proxy forwards the call to an unrelated PC
method; **missing** means no handler exists. `runtime/shims/d3d12_com.cpp` contains no Xbox present or
frame-event code; it only resolves queue proxies at command-list calls (`:1777`, `:1800`). The generated
`d3d12_stubs.h` (`tools/gen_d3d12_stubs.py`) is PC-shaped as well: it generates the SDK interface chain and
has no `PresentX`, `SetFrameIntervalX` or `WaitFrameEventX`, so it cannot supply those slots.

| Boot call | Status | Evidence and consequence |
|---|---|---|
| `XGameRuntimeInitialize` / `Uninitialize` | implemented | `runtime/shims/xgameruntime.cpp:623`, `:629`; exported at `:1195-1196`. |
| `XSystemGetDeviceType` (4K sample path) | implemented | `runtime/shims/xgameruntime.cpp:645`; reports Series X. |
| `D3D12XboxCreateDevice` | implemented | `runtime/shims/d3d12_xs.cpp:124`; export table `:222-227`. The parameter block is logged, not interpreted. |
| QI for `IDXGIDevice1` | implemented | `runtime/shims/d3d12_xs_device.cpp:229-234`; one COM object with the device proxy. |
| `IDXGIDevice1::GetAdapter` | implemented | `d3d12_xs_device.cpp:320`; the Vulkan path installs the layer's own adapter provider at `runtime/shims/d3d12_vulkan_device.cpp:7961`. |
| `IDXGIAdapter::EnumOutputs` | implemented | `runtime/shims/dxgi_com.cpp:1138`; enumerates Win32 displays with the primary first. |
| `CreateCommandQueue` | implemented | `d3d12_xs_device.cpp:1041-1054`; returns a queue proxy. |
| `CreateCommandQueueX` | **wrong slot / missing** | No handler. The override list at `d3d12_xs_device.cpp:1517-1520` omits it, so the fallback at `:1370-1381` forwards an unlisted slot to the PC vtable. Public HLE observation puts the Xbox call at slot 48; PC slot 48 in SDK 10.0.26100 is `OpenExistingHeapFromAddress`, whose arguments do not match a queue desc. FramePacing calls it at `FramePacing.cpp:1746`. |
| `CreateDescriptorHeap` | implemented | `d3d12_xs_device.cpp:1175-1186`; returns a heap proxy that converts the Xbox handle-return convention (`:432-452`). |
| `CreateCommandAllocator` | implemented | `d3d12_xs_device.cpp:1100-1112`; allocator proxy. |
| `CreateCommandList` | implemented, with one open routing question | No override; falls through to PC slot 12 at `d3d12_xs_device.cpp:1370-1381`. The Vulkan device implements it at `runtime/shims/d3d12_vulkan_device.cpp:5161`. Whether the GDK header compiles the sample's `CreateCommandList` call to this slot or to the Xbox entry point below is not settled by the public samples. |
| `CreateCommandListX` | **wrong slot / missing** | No handler, same override list. Public HLE observation puts it at slot 73; PC slot 73 is `CreateShaderCacheSession`. The Go binding routes `CreateCommandList` to this Xbox entry point, so a title built that way never reaches PC slot 12. |
| `CreateFence` | implemented | `d3d12_xs_device.cpp:1073-1084`; fence proxy. |
| `SetFrameIntervalX` | **core only, wrong slot** | The pacing math exists at `runtime/shims/xbox_present.cpp:87-89`, but nothing outside `runtime/test/` constructs `XboxPresentation`. Public HLE observation puts the Xbox device call at slot 78; the proxy has no handler, so the fallback sends it to PC slot 78. On the Vulkan device, which answers `ID3D12Device11`, `detect_slot_count` returns 80 (`d3d12_xs_device.cpp:1409-1421`) and PC slot 78 is `CreateReservedResource2`. The Xbox output pointer is read as that method's resource-desc pointer, 16667 as its initial layout, and the latency as its clear-value pointer; expect `E_INVALIDARG` or a fault, not `S_OK`. |
| `ScheduleFrameEventX` | **core only, wrong slot** | `xbox_present.cpp:91-93`; public HLE observation slot 79, PC slot 79 is `CreateSampler2` on `ID3D12Device11`. |
| `WaitFrameEventX` | **core only, missing** | `xbox_present.cpp:95-97`; public HLE observation slot 80, which is outside the 80-slot PC vtable and is refused with `E_NOTIMPL` at `d3d12_xs_device.cpp:1372`. |
| `GetFrameStatisticsX` | **core only, missing** | `xbox_present.cpp:99-102`; no slot handler. |
| `CreateCommittedResource` | implemented | `d3d12_xs_device.cpp:1293-1317` (Xbox slot 76 to PC slot 27); Vulkan device `d3d12_vulkan_device.cpp:6231`. `ALLOW_DISPLAY` is an accepted committed-resource heap flag at `:1433` and `:1496`. |
| `CreateRootSignature` | implemented | PC slot 16 falls through to `d3d12_vulkan_device.cpp:5406-5459`; parses the `RTS0` blob and registers static samplers. |
| `CreateGraphicsPipelineState` | implemented for the game's shader form | `d3d12_vulkan_device.cpp:5479-5520`; see section 5 for the shader-source limit. |
| `PresentX` | **core only, missing** | `xbox_present.cpp:104-278`; the queue proxy forwards only 19 public slots (`d3d12_xs_device.cpp:993`) and refuses a higher one at `:899-911`. Public HLE observation puts `PresentX` at queue slots 30 and 32, so the call returns `E_NOTIMPL` and no frame reaches the window. |
| `SuspendX` / `ResumeX` | **core only, missing** | `xbox_present.cpp:282-290`; no queue slot. Only reached on PLM suspend, after the first frame. |
| `D3D12SerializeRootSignature` / `D3D12SerializeVersionedRootSignature` | implemented | `runtime/shims/d3d12_xs.cpp:56-104`; clears the measured high-byte flags on a copy and downgrades 1.2 to 1.1 when the host serializer cannot do 1.2. |
| `D3D12CreateRootSignatureDeserializer` | implemented | `d3d12_xs.cpp:106-118`; rewrites an `RTS0` blob's flags and DXBC hash. |
| `D3D12CreateDevice` (PC entry point) | missing from this shim | Not in the `SHIM_TABLE` at `d3d12_xs.cpp:222-228`. The samples never call it; a title that does is not this path. |
| `D3D12XboxSetProcessDebugFlags` | missing export | Public HLE lists it as a real `d3d12_xs.dll` entry point; no public sample calls it before a frame. |

The short version: the runtime serves the PC-shaped part of the boot path (device, queue, heaps,
allocators, lists, fences, resources, root signatures, PSOs) and the Xbox device entry point, but not the
frame-event and present part (frame intervals, frame events, `PresentX`, suspend/resume). The presentation core is written and tested,
but it is test-only: the only constructions in this tree are `runtime/test/xbox_present_test.cpp:337` and
`runtime/test/d3d12_boot_movie_test.cpp:450`.

## 3. Device-creation parameters

The public sample passes a named struct, not a raw word block:

```cpp
D3D12XBOX_CREATE_DEVICE_PARAMETERS params = {};
params.Version = D3D12_SDK_VERSION;
params.ProcessDebugFlags = ...;                 // debug/profile builds only
params.GraphicsCommandQueueRingSizeBytes = D3D12XBOX_DEFAULT_SIZE_BYTES;
params.GraphicsScratchMemorySizeBytes = ...;    // optional
params.ComputeScratchMemorySizeBytes = ...;     // optional
params.DisableGeometryShaderAllocations = TRUE;
params.DisableTessellationShaderAllocations = TRUE;
params.CreateDeviceFlags = D3D12XBOX_CREATE_DEVICE_FLAG_NONE;   // Scarlett block
params.AmplificationShaderIndirectArgsBufferSize = ...;         // optional, GDK 221000+
params.AmplificationShaderPayloadBufferSize = ...;              // optional
```

(`SimpleDeviceAndSwapChain.cpp:254-299`; the same fields in `FramePacing.cpp:1692-1708` and
`DeviceResourcesGXDK.cpp:70-92`; SDL sets only `Version`, `ProcessDebugFlags` and the three size fields,
`SDL_render_d3d12_xbox.cpp:42-52`.)

**The 24-word block.** No public sample shows or dumps 24 words. The samples initialize named fields and
pass a pointer. The only public field-by-field declaration found, ebiten's Go binding, has 15 fields and is
72 bytes (18 32-bit words) on x64, including `pOffchipTessellationBuffer`, `DisableDXR`,
`DisableAutomaticDPBBBreakBatchEvents`, `pDXRStackBuffer`, `CreateDeviceFlags`,
`AutoHDRPaperWhiteLevelNits` and `DisableAutomaticCommandSegmentChaining`. This does not prove the retail
GDK header is 72 bytes, but it does mean the public evidence does not establish a 24-word block. The
runtime's 24-word read is a prior-art measurement, not a public sample fact
(`runtime/shims/d3d12_xs.cpp:120-151`). Reading 96 bytes from a caller that supplied 72 would read 24 bytes
of adjacent stack; that is a diagnostic risk, not a first-frame blocker, because the runtime does not
interpret the block.

**The `0x03000000` root-signature bits.** No public sample sets them. The shortest sample's root signatures
use `D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT` and the `DENY_*` flags only
(`DirectXTK12_Src_SpriteBatch.cpp:338-344`), serialized as version 1.0
(`DirectXTK12_Inc_DirectXHelpers.h`, the `CreateRootSignature` helper: `D3D_ROOT_SIGNATURE_VERSION_1`). The
only support for the value found here is the measured game/BVH material in
[d3d12_vulkan.md](d3d12_vulkan.md) section 7; Microsoft's public reference does not publish the flags enum
([xbox_d3d12_xs.md](xbox_d3d12_xs.md) section 2.4). The runtime's mask is therefore a workaround, not a
publicly specified transformation: `runtime/shims/d3d12_xs_rootsig.h:14` defines
`kXboxRootSignatureFlagHighByte = 0xFF000000u`, `d3d12_xs_rootsig.cpp:8-10` clears it, and
`:127-163` also rewrites an `RTS0` blob and recomputes its DXBC hash.

**`D3D12XboxCreateDevice` flags.** Yes, one public flag appears: every sample read here that has a Scarlett
block assigns `D3D12XBOX_CREATE_DEVICE_FLAG_NONE` (`SimpleDeviceAndSwapChain.cpp:291`,
`FramePacing.cpp:1707`, `DeviceResourcesGXDK.cpp:85`). None of them shows another value, reads the field
back, or depends on the runtime honoring it. The runtime ignores the whole block, which is correct for
`NONE` and untested for anything else.

## 4. What fails first on the Vulkan layer

The order matters: the sample's first pacing call is inside `CreateDevice`, before a resource, PSO or
`PresentX` exists. So the first failure is not the present.

1. **`SetFrameIntervalX` reaches an unrelated PC method.** Call it at `SimpleDeviceAndSwapChain.cpp:586`.
   The device proxy detects 80 PC slots for the Vulkan device (`d3d12_xs_device.cpp:1409-1421`), has no
   handler for Xbox slot 78 (`:1517-1520`), and forwards through `:1370-1381`. PC slot 78 is
   `CreateReservedResource2`; the first Xbox argument is an `IDXGIOutput*`, while the PC method expects a
   heap-properties pointer. Depending on the pointer values this is `E_INVALIDARG` or an access violation.
   `ThrowIfFailed` at `:586` stops the sample. This is the top blocker.
2. **`ScheduleFrameEventX` and `WaitFrameEventX` cannot succeed even if 78 returned `S_OK`.** Slot 79 is
   forwarded to `CreateSampler2`; slot 80 is past the PC vtable and returns `E_NOTIMPL`
   (`d3d12_xs_device.cpp:1372`). The sample's `Tick` then calls `WaitForOrigin` first
   (`SimpleDeviceAndSwapChain.cpp:89`, `:563`), so `ThrowIfFailed` stops it before `Render`.
3. **`PresentX` is unreachable.** The queue proxy knows 19 slots (`d3d12_xs_device.cpp:993`); the public
   HLE observation puts `PresentX` at 30 and 32. The refusal at `:899-911` returns `E_NOTIMPL`. A title
   that presents without waiting gets no frame; a title that waits never gets there.
4. **The X queue/list variants hit wrong PC methods.** `CreateCommandQueueX` (slot 48) and
   `CreateCommandListX` (slot 73) have no handlers. FramePacing calls the former at
   `FramePacing.cpp:1746`; the Go binding uses the latter for `CreateCommandList` on Xbox. On the Vulkan
   device those slots are `OpenExistingHeapFromAddress` and `CreateShaderCacheSession`, so the call is not
   merely refused; it is misdirected. This is before the first present and, for a compute queue, before
   the first frame.
5. **A public sample's PSO may be the wrong shader form.** The runtime does not compile DXIL: its PSO path
   extracts an `FSSM` program from the shader bytecode and looks up translated SPIR-V
   (`d3d12_vulkan_device.cpp:70-72`, `:2985-3018`, `:5479-5520`). The sample's SpriteBatch shaders are the
   GDK-compiled `SpriteEffect` blobs (`DirectXTK12_Src_SpriteBatch.cpp:28-32`); no public source read here
   shows those blobs carrying an `FSSM` program. If they do not, `CreateGraphicsPipelineState` logs "shader
   bytecode holds no FSSM" and refuses. The game's own shaders are `FSSM`, so this is a sample-replication
   blocker, not necessarily a game blocker.
6. **A required feature query is not the problem.** The shortest sample does not call
   `CheckFeatureSupport` before its first present. The runtime's implementation covers the standard feature
   ids (`d3d12_vulkan_device.cpp:6727` onward), so a title that queries feature level or options gets an
   answer; the boot samples do not gate device creation on one.
7. **`SuspendX`/`ResumeX` are not a first-frame blocker.** They run on PLM suspend, after a frame has been
   presented (`SimpleDeviceAndSwapChain.cpp:227-243`). They are missing from the queue proxy, but a title
   reaches its first frame without them.
8. **The 24-word read is not a first-frame blocker.** It can read past a caller's 72-byte public-layout
   struct, but it does not change the device the title gets. It should be fixed before a retail trace is
   trusted.

Two things that are *not* blockers on this path: `CreateCommittedResource` accepts
`D3D12_HEAP_FLAG_ALLOW_DISPLAY` (`d3d12_vulkan_device.cpp:1433`, `:1496`), and the sample's formats
(`B8G8R8A8_UNORM_SRGB`, `D32_FLOAT`) are ordinary Vulkan formats. The DXGI output enumeration also works
through the layer's adapter provider (`d3d12_vulkan_device.cpp:7961`), so step 9-11 of the device sequence
does not fail before step 12.

## 5. First-frame resource and PSO pattern

The shortest sample does not use `D3D12XBOX_ROOT_SIGNATURE_1_2`, `D3D12XBOX_SHADER_COMPILER`, or any shader
library API. Those identifiers were not found in the Microsoft samples, DirectXTK12's GXDK path, SDL's Xbox
backend, Microsoft's public GDK reference, or this runtime. They should not be assumed to exist.

What the sample actually does:

- **Root signature.** DirectXTK12's `SpriteBatch` builds a 1.0 `D3D12_ROOT_SIGNATURE_DESC` with one CBV,
  one SRV table, one static sampler, and the standard `ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT`/`DENY_*` flags
  (`DirectXTK12_Src_SpriteBatch.cpp:336-389`). It serializes through DirectXHelpers'
  `CreateRootSignature`, which calls `D3D12SerializeRootSignature(..., D3D_ROOT_SIGNATURE_VERSION_1, ...)`
  then `ID3D12Device::CreateRootSignature(0, blob, size, iid, out)`. The GXDK variant adds
  `DENY_AMPLIFICATION_SHADER_ROOT_ACCESS` and `DENY_MESH_SHADER_ROOT_ACCESS`; no 1.2 features are used.
- **PSO.** `CreateGraphicsPipelineState` with a full `D3D12_GRAPHICS_PIPELINE_STATE_DESC`: precompiled VS
  and PS blobs, input layout, blend/rasterizer/depth-stencil state, RTV format and DSV format
  (`DirectXTK12_Src_SpriteBatch.cpp:473-496`). The shaders are the GDK-compiled `SpriteEffect` blobs; no
  compiler runs at runtime.
- **Resource upload.** `ResourceUploadBatch` uses a committed DEFAULT index buffer, a `GraphicsMemory`
  upload buffer, `Map`/`Unmap`, `CopyBufferRegion` and `CopyTextureRegion`, then a barrier to the read state
  (`DirectXTK12_Src_SpriteBatch.cpp:301-328`; `SimpleDeviceAndSwapChain.cpp:506-532`). All of these are in
  the runtime's tested command path.
- **First draw.** `SetDescriptorHeaps`, `SetGraphicsRootSignature`, `SetGraphicsRootDescriptorTable`,
  `SetPipelineState`, `Draw*`, then the barrier to `PRESENT`. The heap is shader-visible; the static
  sampler is in the root signature.

The runtime side matches the *shape*: `CreateRootSignature` parses `RTS0`
(`d3d12_vulkan_device.cpp:5406-5459`) and `CreateGraphicsPipelineState` consumes the game's `FSSM` form
(`:5479-5520`). It does not match the *source*: there is no DXIL compiler and no console shader-library
entry point. A title whose PSO bytecode is `FSSM` (the game, per the corpus notes in
[d3d12_vulkan.md](d3d12_vulkan.md) section 3.6) is on the supported path; a title whose PSO bytecode is
GDK DXC output with no `FSSM` program has nothing for that lookup to match. `D3D12XBOX_SHADER_COMPILER`
would be exactly such a missing API if it existed publicly; no public source names it.

## 6. First blocker candidates, ranked

| Rank | Candidate | Why it is first | Fix scope |
|---:|---|---|---|
| 1 | `SetFrameIntervalX` on Xbox device slot 78 | The sample calls it in `CreateDevice`; the proxy forwards it to PC `CreateReservedResource2` instead of pacing (`d3d12_xs_device.cpp:1409-1421`, `:1517-1520`, `:1370-1381`) | Add a device-slot handler that routes 78 to `XboxPresentation::SetFrameIntervalX`; test the proxy with a fake device before the real one |
| 2 | `ScheduleFrameEventX` / `WaitFrameEventX` on slots 79/80 | Even if 78 is fixed, the wait is the first GPU call in `Tick` and slot 80 returns `E_NOTIMPL` | Same handler table; a fake-clock test that arms, waits and gets a token |
| 3 | `PresentX` on queue slots 30/32 | No frame can reach the window while the queue proxy stops at 19 slots (`d3d12_xs_device.cpp:993`, `:899-911`) | Add queue-slot handlers for `PresentX`, `SuspendX`, `ResumeX` over `XboxPresentation`; test with a fake presenter |
| 4 | `CreateCommandQueueX` / `CreateCommandListX` | FramePacing and the Go binding use them; the fallback sends them to unrelated PC methods | Add the two desc types and handlers; test the argument layout against the public bindings |
| 5 | GXDK DXIL PSOs | A public sample's SpriteBatch PSO has no `FSSM`; the runtime refuses it | Only if sample replication is wanted; the game's `FSSM` path does not need it |
| 6 | 24-word parameter read | Reads past a 72-byte public-layout struct; not fatal today | Log at most the caller-declared size once it is known, or keep 24 words but document the over-read |

Ranks 1-3 are one change in practice: wire the presentation core into the device and queue proxies. The
core and its tests already exist (`xbox_present.cpp`, `runtime/test/xbox_present_test.cpp`,
`runtime/test/frame_pacer_test.cpp`); the gap is the vtable plumbing and the slot values, which are still
being measured. This page therefore makes no code change.

## 7. Verification

The document was checked with the repository's documentation tests:

```text
python -I tests/test_docs_links.py
python -I tests/test_docs_repo_paths.py
python -I tests/test_docs_repeated_lines.py
python -I tests/test_docs_table_ids.py
python -I tests/test_docs_table_rows.py
git diff --check
```

The public sample links are external and are not fetched by the tests; the file and line references were
read from the sources listed at the top on 2026-11-23. The runtime line references are to this tree at the
commit that adds this page.
