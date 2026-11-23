# Boot-path import coverage audit — main `3faac557b`

Status: audited at 3faac557b; every runtime gap is claimed by an open PR listed in the table.

This is a surface/code audit, not a first boot: the retail executable is still absent from the package (RT-5 / `docs/boot.md` §1), so the page audits the *documented* import surface and the code paths a first boot will take. Counts are from source, probe runs under `gta6rt --run`, or an open PR's measured result. No game data, keys or decrypted content is used, and no probe artifacts are included.

## 1. What was checked

- `docs/runtime_imports.md` (the measured 16-DLL table, the five documented thunk entry points plus the sixth measured in PR #5312, the class layouts, the save/store/offline classes).
- `docs/boot.md` §1–3 (the three passes and the triage order).
- `runtime/shims/registry.cpp`, `runtime/shims/xgameruntime*.cpp`, `runtime/shims/d3d12_xs.cpp`, `runtime/shims/dxgi_com.cpp`, `runtime/shims/xmem.cpp`, `runtime/shims/file_system.cpp`, `runtime/shims/process.cpp`, `runtime/shims/library_loader.cpp`.
- `runtime/src/pe_loader.cpp::resolve` and `library_loader.cpp` (how an unserved name binds).
- `tools/d3d12_xs_surface.py` (23 function exports + 93 GUID data exports) and `tools/gdk_runtime_classes.json` (24 runtime classes).
- Probe runs under `gta6rt --run` covered the six thunk names, the 24 class ids, the 23 `d3d12_xs` function names, `XNetworkingRegisterConnectivityHintChanged`, and `GetModuleFileNameW(NULL)`.

## 2. Boot order, and what runs before D3D12

`docs/boot.md` §1 and §3 define the order:

1. **`--dry-run`**: map the image, bind every import, apply TLS. It does *not* call TLS callbacks, `DllMain` or the entry point.
2. **`--init`**: TLS callbacks and `DllMain(PROCESS_ATTACH)`.
3. **`--run`**: the entry point on the game thread. For a normal MSVC title that means the CRT startup, then the title's `main`/`WinMain`, then subsystem init (memory, threads, files, user, save/package, networking), and only then the D3D12 device.

So the imports that matter *before the first D3D12 call* are the CRT/process/thread/file/memory ones, the `xgameruntime.lib` thunk helper, and the `QueryApiImpl` classes a title touches during subsystem init. The D3D12 device, root signatures and PSOs come after.

The loader resolves an import in this order (`runtime/src/pe_loader.cpp:1105-1141`):

1. the shim registry (`shims::find`),
2. a bundled module in `--game-dir` with that file name,
3. `LoadLibraryExA(dll, LOAD_LIBRARY_SEARCH_SYSTEM32)` + `GetProcAddress`,
4. an unimplemented-import thunk (`log`: return 0; `fail`: `E_NOTIMPL` for HRESULT names; `stop`: exit 3).

The runtime `LoadLibrary*` path is different: for a module the shims own (`owns_dll`), `library_loader.cpp:94-100` hands out a pseudo-handle and `GetProcAddress` resolves only through the registry. An unserved name is `NULL` + `ERROR_PROC_NOT_FOUND`; it never falls back to the host. This is the path `xgameruntime.lib` thunks use.

**Important loader gap on main:** `resolve()` itself does not check `owns_dll` before step 3. A *static import* of an unserved name from an owned module (`xgameruntime.dll`, `gameinput.dll`, an unserved `winhttp.dll` name, …) can therefore bind to the host namesake — including the PC Gaming Services runtime. PRs #2098 and #2490 both fix this; #2490 describes it as a host-binding fix.

## 3. The documented surface, in counts

| Surface | Documented | Served on main | Missing / stub |
|---|---:|---:|---|
| `xgameruntime.dll` named exports (`XGR` table) | 41 names + 2 `gameplatformservices.dll` names | 41 + 2 | `XErrorReport`; `gameplatformservices.dll!InitializeApiImpl` |
| `xgameruntime` runtime classes (`QueryApiImpl`) | 24 class ids | 15 | 9 classes (table below) |
| `xgameruntime` thunk entry points | 6 (`InitializeApiImpl`, `…Ex`, `…Ex2`, `QueryApiImpl`, `UninitializeApiImpl`, `XErrorReport`); `runtime_imports.md` still says five, PR #5312 measured the sixth | 5 | `XErrorReport` |
| `d3d12_xs.dll` function exports | 23 | 4 (`Serialize*` ×2, `CreateRootSignatureDeserializer`, `D3D12XboxCreateDevice`) | 19 |
| `d3d12_xs.dll` GUID data exports | 93 | 0 | 93 |
| `xmem.dll` | 7 measured + documented defaults/hooks | 7 | `XMemAllocDefault`, `XMemFreeDefault`, `XMemSetAllocationHooks`, `XMemGetTitleHeap`, … |
| `dxgi.dll` | `CreateDXGIFactory{,1,2}`, `DXGIGetDebugInterface1`, `DXGIDeclareAdapterRemovalSupport` | 0 shimmed | host-bound |
| `d3d12.dll` | `D3D12Create*`, `D3D12Serialize*`, … | 0 shimmed | host-bound |
| `kernel32` file API | 33 names × 5 modules | 33 × 5 | — |
| `kernel32` process/CRT | 17 UCRT × 3 + 9 msvcrt + 3 `ExitProcess` + 6 command-line | all | — |
| `kernel32` loader | 9 × 7 | 9 × 7 | `GetModuleFileName*`, `GetModuleHandleEx*` are **not** in the table |

The 15 served `xgameruntime` classes are `XThreadingImpl`, `XErrorImpl`, `XNetworkingImpl`, `XPersistentLocalStorageImpl`, `XSystemImpl`, `XPackageImpl`, `XGameImpl`, `XUserImpl`, `XGameSaveImpl`, `XStoreImpl`, `XGameEventImpl`, `XGameUiImpl`, `XGameStreamingImpl`, `XDisplayImpl`, `XUserDeviceImpl`. The 9 unserved are `XAccessibilityImpl`, `XAppCaptureImpl`, `XAppCaptureMetadataImpl`, `XGameActivationImpl`, `XGameInviteImpl`, `XGameProtocolImpl`, `XGameRuntimeFeatureImpl`, `XLauncherImpl`, `XSystemAnalyticsImpl`. A probe of those 9 returned `0x80070032` and cleared the out pointer for every one; `XThreadingImpl` served as the control.

## 4. Top 10 first-boot risks

Confidence: **confirmed** = reproduced by a probe or measured on the package's DLLs; **measured (sibling)** = an open PR measured the call in a title/package; **documented** = the GDK reference and the code shape agree, but no GTA VI trace; **conditional** = only if the executable imports it.

| # | Import / path | Status at `3faac557b` | Evidence | Boot risk | Cost | Owner PR (open) |
|---:|---|---|---|---|---|---|
| 1 | `xgameruntime.dll!XErrorReport` | **Missing.** `error_report` exists as `IXErrorImpl` slot 3 (`xgameruntime.cpp:236-240`) but the `SHIM_TABLE` (`:1098-1146`) has no `XErrorReport` entry. The thunk pseudo-handle returns `NULL`/`ERROR_PROC_NOT_FOUND`. | Probe under `--run`: five names served, `XErrorReport` `MISSING (err 127)`. PR #5312 measured `HCInitialize` → `0x89240101` (`E_GAMERUNTIME_DLL_NOT_FOUND`) and `curl_global_init` → `CURLE_FAILED_INIT`; the thunk helper refuses the runtime *before* calling any initializer. | **Fatal, confirmed.** If the executable's `xgameruntime.lib` is the same version, its very first runtime call fails. | **Cheap:** one `XGR` entry reusing the existing function. PR #5312 already does it + tests. | #5312 **dracbat** |
| 2 | `XNetworkingImpl` slot 16 (`XNetworkingRegisterConnectivityHintChanged`) | **Stub returning success.** `net_register` (`xgameruntime.cpp:301-306`) hands out a token and never calls back; unregister always true. | Probe under `--run`: `register hr=0`, `token=1`, `callbacks=0` after 50 dispatch rounds. PR #4551: XCurl calls slot 16 at boot and the usual network-init pattern waits for the initial callback, so offline it waits forever. | **Fatal hang, confirmed.** | Moderate: queue one initial callback on the completion port, handle unregister-before-dispatch and `wait`. PR #4551 (one function + tests). | #4551 **AirKyzzZ** |
| 3 | `GetModuleFileNameW(NULL)` / module identity | **Not shimmed.** The host answers for the loader's own `gta6rt.exe`. `GetModuleHandleEx*` and `GetProcAddress(NULL, …)` are likewise host answers. | Probe under `--run`: `GetModuleFileNameW(nullptr)` names the host loader, not the guest image. PR #2075: titles often derive their install folder from this; it also breaks `GetModuleFileName` on a mapped image's `HMODULE`. | **High, confirmed.** Any title that locates its data from its own module path reads the wrong folder before D3D12. | **Real work** (search rules, handles, paths, overlay policy). PR #2075 is substantial and has an open overlay-policy question. | #2075 **dracbat** |
| 4 | `XGameRuntimeFeatureImpl` (`XGameRuntimeIsFeatureAvailable`) | **Class missing.** Class id `8836fe87-…` is not in `QueryApiImpl`'s chain; returns `0x80070032`, out cleared. | Class probe: `XGameRuntimeFeatureImpl hr=0x80070032 out=null`. The thunk returns the HRESULT; a `bool` caller reads `al = 0x32` = true for **every** feature. PR #3476: returns true only for features the runtime serves and Microsoft documents on Windows (`XDisplay`, `XStore` false). | **Likely.** Feature-gating at boot can take a wrong path; `libHttpClient.GDK.dll` references the class. | **Cheap:** one class + one bool slot with a documented table. | #3476 **seblessa** |
| 5 | `XSystemAnalyticsImpl` (`XSystemGetAnalyticsInfo`) | **Named export served, class missing.** `XGR(XSystemGetAnalyticsInfo)` exists (`:1114`), but class id `b884675d-…` is not served. The thunk path gets `0x80070032` and the 144-byte struct is never written. | Class probe: `hr=0x80070032 out=null`; `XSystemAnalyticsInfo` is 144 bytes (`xgameruntime.cpp:675-682`). | **Possible.** Telemetry/analytics init reads an uninitialized struct; can crash or misconfigure. | **Cheap:** reuse the existing named export in one slot. | #3476 **seblessa** |
| 6 | `XGameProtocolImpl` + `XGameInviteImpl` | **Classes missing.** Ids `95fd18d2-…` / `0651aae2-…` return `0x80070032`. | Class probe: both `hr=0x80070032 out=null`. PR #4239: `XGameProtocolRegisterForActivation` and `XGameInviteRegisterForEvent` failed with `ERROR_NOT_SUPPORTED` at title start. | **Measured (sibling).** Activation/invite registration during boot can fail or leave an uninitialized token. | **Cheap-ish:** registration tokens, unregister true once, `E_NOTIMPL` for the undocumented `IXGameInviteImpl2` trio. PR #4239 (21 checks). | #4239 **evilclanker9000** |
| 7 | `XAccessibilityImpl` | **Class missing.** Id `3e241977-…` returns `0x80070032`. | Class probe. Methods: high contrast, closed captions, speech-to-text/synthesizer. | **Possible.** UI/accessibility init at startup gets `0x32` in low byte of an enum/bool and unwritten out params. | **Cheap-ish:** documented defaults (no high contrast, captions off, speech unavailable); 21 slots. | #4235 **evilclanker9000** |
| 8 | `XAppCaptureImpl` + `XAppCaptureMetadataImpl` | **Classes missing.** Ids `a4f1aee2-…` / `186d5592-…` return `0x80070032`. | Class probe. Methods include screenshots, recording, broadcast status, metadata. | **Possible.** Capture/broadcast init at boot; usually logged, but an uninitialized status can misroute UI. | **Moderate:** many slots, mostly "not available" answers. | #4307 **x3nnnonn** |
| 9 | `d3d12_xs.dll` standard forwards: `D3D12CreateVersionedRootSignatureDeserializer`, `D3D12EnableExperimentalFeatures`, `D3D12GetDebugInterface`, `D3D12CreateDevice` | **Missing.** The table (`d3d12_xs.cpp:221-227`) serves only 4 of 23 documented functions. | Probe under `--run`: `served=4 missing=19`. If imported, the loader makes an unimplemented thunk: returns 0 and leaves `*out` null. | **Conditional.** A first root-signature/deserializer call or an experimental-feature enable dies if the executable imports the name from `d3d12_xs.dll`. The package's DLLs import only the two root-signature names (served). | **Cheap:** forward the three standard functions; `D3D12CreateDevice` needs a backend decision (host PC device vs. Vulkan proxy). PR #2098 claims all four. | #2098 **SHAURYASANYAL3** |
| 10 | Memory setup: `XMemAlloc` alignment/page bits, `XMemVirtualQuery` +0x30, `XMemAllocDefault`/hooks/title heap | **Partly wrong / missing.** `XMemAlloc` always 64-byte aligned (`xmem.cpp:9-11,53-56`), ignoring the attribute word's alignment and 64 KiB page code. `XMemVirtualQuery` zero-fills after `MEMORY_BASIC_INFORMATION`, so +0x30 is 0. The documented default routines/hooks/heap queries are absent. | PR #3601 measured `amd_fidelityfx_upscaler_xbox.dll` passing `0xa8004000` (256 B / 64 KiB) and `XboxBvhBuilder.dll`'s four call sites requiring bits 0-1 at +0x30. `xmem.cpp:71-90`; PR #5253 lists the missing exports. | **Possible at heap/BVH init.** Engine allocators that expect 64 KiB/256 B alignment or a usable `XMemVirtualQuery` answer get a weaker/no answer before D3D12. | **Moderate:** decode the attribute word and write +0x30 (PR #3601); the default routines/hooks are mechanical (PR #5253). | #3601 **kotorvr**, #5253 **maxxgabriel** |

## 5. Additional findings (status at 3faac557b; served as a stub, limited, or host-bound)

These are not part of the ranked runtime-gap table above; rows without a PR are observations for the next pass.

| Import / path | Status at `3faac557b` | Evidence | Boot risk | Cost / owner PR (open) |
|---|---|---|---|---|
| `XPackageImpl` slots 5-6 (installation monitor create/close), 9, 12-21, 27-40 | `E_NOTIMPL`; slots 3/4/7/8/10/11/22-26/34 have real answers. | `xgameruntime.cpp:978-1004`, `:869-882`. | **Possible.** A title that creates an installation monitor at boot gets `E_NOTIMPL`; progress bytes are all 0 but launchable/completed. | Moderate; needs documented monitor semantics. No PR found; observation, not a ranked gap. |
| `XPackageMount` | Real, but only mounts ids listed in `LEONIDA_CONTENT_PACKAGES`; the process's own id is `E_ACCESSDENIED`; unlisted ids are `ERROR_NOT_FOUND`. | `xgameruntime.cpp:750-806`, README env table. | **Possible.** Content-package mount at boot fails unless the user configures the id. | Real work: resolve package metadata / `MicrosoftGame.Config`. |
| `XPersistentLocalStorageMountForPackage` | `E_NOTIMPL`, handle cleared. | `xgameruntime.cpp:467-471`. | **Possible** if save/storage init mounts another package. | Real work. |
| `XGameSaveReadBlobData` | Pads each blob to 8 bytes; a caller sizing the documented way gets `E_GS_PROVIDED_BUFFER_TOO_SMALL`. | PR #5189 (2 checks fail on main). | **Low on first boot** (empty save folder), **high on a save load**. | Cheap: dense packing (PR #5189 **AirKyzzZ**). |
| `XStoreGameLicense` sizes | 104-byte struct; `STORE_SKU_ID_SIZE` 18 and `TRIAL_UNIQUE_ID_MAX_SIZE` 64 corroborated; `SKU_ID_SIZE = 5` still unverified. | `docs/runtime_imports.md:276-278`; PR #4955 **bobcatchris**. | **Low/possible** license check at boot. | Needs the executable's buffer to confirm 5. |
| `XSystemGetAppMemoryUsageLimit` | Returns host total physical RAM, not the console's ~12 GiB title budget. | `xgameruntime.cpp:583-589`; `xmem.cpp:113-122` has the 12 GiB fallback only when `GlobalMemoryStatusEx` fails. | **Low/possible** over-sized memory pools. | Cheap cap, but the exact console budget is an assumption. |
| `XSystemHandleTrack` / `XSystemAllowFullDownloadBandwidth` | `S_OK` and no callback; no-op. | `xgameruntime.cpp:641-647`. | Low. | Cheap documented stub already. |
| `gameplatformservices.dll!InitializeApiImpl` | Not served; `QueryApiImpl` and `GetApiVersionImpl` are. | `xgameruntime.cpp:1106-1107`; PR #5312 says `curl_global_init` succeeds without it. | Low. | Cheap alias if wanted. |
| `d3d12_xs.dll` console-only: `D3D12ZeroMemoryX`, `D3DFlushEntireCpuCache`, `D3D12Cons*`, `PixGetPixieEntryPoints`, `D3D10CreateBlob` | Missing (part of the 19). | `tools/d3d12_xs_surface.py`; probe run under `--run`. | **Possible** at memory setup (`ZeroMemory`/`Flush`); the `Cons*` signatures are internal aliases, not verified wire names. | Cheap *if* signatures are accepted (`memset`, `_mm_sfence`, stored debug flags); otherwise real work. No PR for these; #2098 covers the standard four. |
| `dxgi.dll!CreateDXGIFactory*`, `DXGIGetDebugInterface1`, `DXGIDeclareAdapterRemovalSupport` | **Host-bound, not shimmed.** `registry.cpp:100-124` only classifies return types; there is no `SHIM_TABLE` entry, so `resolve` loads System32 `dxgi.dll`. The Vulkan factory (`d3d12_vulkan_device.cpp:7910`) is only reachable through the device proxy's `IDXGIDevice1`. | `pe_loader.cpp:1129-1137`; `d3d12_vulkan_device.cpp:7555-7560`; `d3d12_xs_device.cpp:229-271`. | **Low/possible** on Windows 11 (host DXGI works), but a host factory + Vulkan device swapchain can fail, and it violates the "no host DXGI" design (`d3d12_vulkan.md:98-99`). | Real work; PR #2098 **SHAURYASANYAL3**. |
| `d3d12.dll!D3D12SerializeVersionedRootSignature` / `...Deserializer` | Host-bound; the Xbox `0x03000000` flags are **not** cleared, so the host rejects the blob. | No `d3d12.dll` shim entry; the flag clear lives in `d3d12_xs.cpp:56-118`. | **Conditional first-frame.** If the executable imports the serializer from `d3d12.dll` rather than `d3d12_xs.dll`, the first root signature fails `E_INVALIDARG`. | Real work (host isolation + flag clear); PR #2098. |
| 93 `d3d12_xs.dll` GUID data exports | None served. | `tools/d3d12_xs_surface.py` `DATA_EXPORTS`; no data probe is included. | **Conditional.** A data import would become a code thunk. Unlikely from a normal PE import table. | Mechanical but large; PR #3675 **kotorvr**. |
| `XGameActivationImpl` | Missing; PR #4239 deliberately leaves it out because its signatures are undocumented. | PR #4239 body. | **Possible** activation at boot; needs signature recovery. | Real work. |
| `XLauncherImpl` | Missing; `XLaunchUri`, display timeout deferral. | PR #4248. | Low (launcher, not first boot). | Cheap-ish; PR #4248 **evilclanker9000**. |
| `XUserImpl` token/MSA/resolve calls | Fail synchronously; sign-out deferral unavailable; change callbacks never fire. | `xgameruntime_user.cpp`; docs lines 236-241. | Low/possible: a title that requires Xbox Live at boot takes its offline path. | Real work (auth) / acceptable offline. |
| `XTaskQueueIsEmpty` private slot 28; `XAsyncBeginAlloc` slot 8 | 0/false; `E_NOTIMPL`. | `xgameruntime.cpp:164-181,222`. | Low; lukewarm if the title polls the queue at exit. | Cheap only with a measured signature. |
| `gameinput.dll!GameInputCreate` | Built only with SDL3; our audit build had `LEONIDA_SDL3=OFF`, so it binds to the host's `GameInput.dll` if present, else unimplemented. | `runtime/CMakeLists.txt:333-384`; `registry.cpp:112-113`. | **Possible** input init; outside the requested families. | Build option, not a missing shim. |
| `xg_xs.dll`, `AcpHal.dll`, `dstorage_xs.dll`, `XAudio2_9.dll` | No shim; `dstorage_xs`/`xaudio2_9` only have HRESULT classification. | `docs/d3d12_vulkan.md:3223`; `registry.cpp:80-83,110-115`. | **Unknown**, outside scope: texture layout, audio APU, console DirectStorage. The package's 16 DLLs do not import them; the executable's table is unmeasured. | Real work, needs the executable's imports. |

## 6. Cheap versus real work

**A stub with a documented semantic (hours, not a subsystem):**

- `XErrorReport` — reuse `error_report` (`#5312`).
- `XSystemAnalyticsImpl` — reuse `XSystemGetAnalyticsInfo` (`#3476`).
- `XGameRuntimeFeatureImpl` — one bool slot with the documented per-feature answer (`#3476`).
- `XGameProtocolImpl` / `XGameInviteImpl` — token registration, true-once unregister (`#4239`).
- `XAccessibilityImpl` — no high contrast, captions off, speech unavailable (`#4235`).
- `XAppCaptureImpl` / `XAppCaptureMetadataImpl` — "capture/broadcast not available" answers (`#4307`).
- `XLauncherImpl` — log-and-succeed launch URI, deferral handle (`#4248`).
- `XMemAllocDefault` / `XMemFreeDefault` / `XMemSetAllocationHooks` / `XMemGetTitleHeap` — documented defaults (`#5253`).
- `gameplatformservices.dll!InitializeApiImpl` — alias.

**Real work:**

- `GetModuleFileName*` / `GetModuleHandleEx*` / `GetProcAddress(NULL, …)` identity (`#2075`): search rules, mapped-module paths, and the overlay policy for a package path.
- `XNetworking` initial connectivity callback (`#4551`): queue scheduling, unregister-before-dispatch and `wait` semantics.
- `XPackage` installation monitor and general content-package mount; `XPersistentLocalStorageMountForPackage`.
- `dxgi.dll` / `d3d12.dll` isolation and the standard `d3d12_xs` forwards (`#2098`); `D3D12CreateDevice` backend decision.
- `XGameActivationImpl` signatures; `D3D12ConfigureVirtualMemoryX`; `D3D12UploadCustomMicrocodeX`; `DXGIX*`; `InternalDriverCallX`.
- `XStore` purchases/tokens, `XUser` MSA/token flows (offline by design).
- 93 GUID data exports (`#3675`) — mechanical but large.

## 7. Build and test run

Windows 11, MSVC 19.44.35228.0 (VS 2022 Build Tools), CMake + Ninja, `RelWithDebInfo`, Vulkan SDK 1.4.363.0, `LEONIDA_VK_DEVICE='RTX 5070 Ti'`, `LEONIDA_D3D12='vulkan'`. SDL3 was off, so the GameInput shim is not in this build.

- **11/11 selected CTest tests passed:**
  - `xgameruntime`: 128 checks.
  - `xgameruntime_user`: 42 checks.
  - `xgameruntime_gamesave`: 50 checks.
  - `xgameruntime_store`: 22 checks.
  - `xgameruntime_offline`: 29 checks.
  - `xgameruntime_offline_hdr`: 29 checks.
  - `xgameruntime_user_gamertag`, `xgameruntime_user_gamertag_long`: pass.
  - `run_tiny_gdk_load`: 1 module, `shim=14 host=51 module=0 unimplemented=0`; PASS.
  - `run_tiny_atexit`: 1 module, `shim=11 host=47 module=0 unimplemented=0`; 8 termination paths match the Windows sequence.
  - `run_tiny_frame`: 1 module, `shim=10 host=46 module=0 unimplemented=0`; 16 guest `PASS` lines + `OK`; 19 `[d3d12-trace]` lines; 9 startup-trace events; device `NVIDIA GeForce RTX 5070 Ti`; 32/32 pixels clear; a second run with `LEONIDA_VK_VALIDATION=1` had 0 validation errors.
- **Probe results** (under `--run`; no artifacts are included):
  - six thunk names: 5 served, `XErrorReport` missing (`err 127`).
  - 9 class ids: all `0x80070032`, out cleared; `XThreadingImpl` served.
  - 23 `d3d12_xs` functions: 4 served, 19 missing.
  - `XNetworking` slot 16: `hr=0`, `token=1`, `callbacks=0`.
  - `GetModuleFileNameW(NULL)`: names the host loader, not the guest image.

## 8. Open PR overlap (owners)

The biggest gaps are already claimed. Do not duplicate these patches.

| Gap | PR | Owner |
|---|---|---|
| `XErrorReport` | #5312 | **dracbat** |
| `XGameRuntimeFeatureImpl`, `XSystemAnalyticsImpl` (and PLS/Event/Display rebased) | #3476 | **seblessa** |
| `XAccessibilityImpl` | #4235 | **evilclanker9000** |
| `XGameProtocolImpl`, `XGameInviteImpl` | #4239 | **evilclanker9000** |
| `XGameStreamingImpl`, `XLauncherImpl` | #4248 | **evilclanker9000** |
| `XAppCaptureImpl`, `XAppCaptureMetadataImpl` | #4307 | **x3nnnonn** |
| `dxgi` shim + `d3d12` isolation + standard `d3d12_xs` forwards | #2098 | **SHAURYASANYAL3** |
| 93 `d3d12_xs` GUID data exports | #3675 | **kotorvr** |
| Module identity (`GetModuleFileName*`, `GetModuleHandleEx*`) | #2075 | **dracbat** |
| Loader host-binding fix, import inventory, console-library catalogue | #2490 | **NolanCotter** |
| `XMemAlloc` alignment / `XMemVirtualQuery` +0x30 | #3601 | **kotorvr** |
| `XMemAllocDefault` / hooks / heap queries | #5253 | **maxxgabriel** |
| `XNetworking` initial callback | #4551 | **AirKyzzZ** |
| `XGameSaveReadBlobData` packing | #5189 | **AirKyzzZ** |
| `XGameSave` quota/atomic update | #5113 | **Dom-in** |
| `XGameSave` long paths | #5203 | **soldsawn** |
| `XUserFindForDevice` | #5166 | **milas6669** |
| `XErrorSetCallback` pair | #3963, #3927 | **traeb0n**, **x2Tweek** |
| `XSystemGetDeviceType` registry answer | #3775 | **kotorvr** |
| Series S by VRAM | #4821 | **DaniSBR1** |
| PLS out of `GAME:\` | #4327 | **AirKyzzZ** |
| PLS v1 | #3471 | **itsRevela** |
| Store sizes corroboration (docs) | #4955 | **bobcatchris** |
| Text-entry default | #4953 | **romainveurrier** |
| `runtime_imports.md` served-class reading (docs) | #5348 | **Vonsek** |

Unclaimed by any open PR found and **not ranked as runtime gaps** above: `XGameActivationImpl` (signatures undocumented), `XPackageCreateInstallationMonitor` semantics, `XPersistentLocalStorageMountForPackage`, the console-only `d3d12_xs` helpers (`D3D12ZeroMemoryX`, `D3DFlushEntireCpuCache`, `D3D12Cons*`, PIX) if their signatures are accepted, and the out-of-scope `xg_xs`/`AcpHal`/`dstorage_xs` modules. These are observations for the next pass, not duplicate-work candidates.

## 9. Limits

- The executable's import table is still unmeasured. The risk column is a surface audit plus the package DLLs' measured thunk calls, not a trace of GTA VI.
- "Named export missing" and "class missing" are different paths: a title linked through `xgameruntime.lib` reaches classes via `QueryApiImpl`; a title with a static import reaches named exports through `pe_loader::resolve`.
- The `d3d12_xs` 19 missing functions are only fatal if the executable imports them; the shipped package DLLs import two, both served.
- The `dxgi`/`d3d12` host bindings work on this Windows 11 machine, which is why they are not ranked above the confirmed `XErrorReport`/`XNetworking` failures.
