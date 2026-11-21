# Leonida development log

## 2026-11-21 — PE loader integration on latest upstream

### Baseline and preservation

- Created `codex/latest-upstream-boot-progress` from upstream `main` at `8bd0087d2d0e45fc18fa78fb40b70ecfc6648854` after checking the upstream branch. The original `C:\Windows\System32\Leonida` checkout and its `main` branch remain untouched.
- Preserved the `Leonida-dev` worktree and its uncommitted `codex/pe-loader-validation` changes, along with the earlier `Leonida-upstream-dev` worktree and its patch snapshots. No game package was opened.
- Carried forward only PE section validation, page-level permission merging, unresolved-import initialization preflight, and generated initialization-order fixtures. Existing upstream dependency ordering, TLS/DllMain initialization path, forwarding and delay imports, `.pdata` registration, Vulkan D3D12 backend, and base `xgameruntime`/`xtaskqueue` shims were reused rather than duplicated.

### Changes

- Added PE section layout checks for section/file alignment, mapped header overlap, section ordering and virtual overlap, image bounds, and complete raw-file extents. Mapping continues to use `VirtualSize` as the section's in-memory extent, with `SizeOfRawData` used when `VirtualSize` is zero.
- Merged section access requirements per Windows page before applying protections, so valid sub-page sections sharing a page retain the combined permissions. Unused image pages are set to `PAGE_NOACCESS`.
- Made `Loader::initialise()` preflight the loaded import graph and fail on unresolved imports before any TLS callback or DLL entry point can run.
- Added compiler-generated dependency and importer DLL fixtures that verify dependency-first `DllMain` ordering. Expanded synthetic PE tests for sub-page alignment, raw padding, shared-page protections, overlaps, truncated raw data, and unresolved-import preflight.
- Updated README status text to describe the verified generated-fixture scope and to avoid implying real-game compatibility.
- Reviewed `R0Y00/Xbox2Windows` and `omonkeyz/xbox-emulator` at the architecture/documentation level for PE, initialization, system-library and graphics approaches. No external code was copied; current upstream already contains relevant loader, runtime-shim, and Vulkan D3D12 components.

### Verification

- `scripts/build_runtime.ps1`: passed on the Visual Studio x64 build path after rebasing onto upstream `8bd0087`. The final rerun rebuilt and linked the updated `pe_loader_test` target (2 Ninja steps); the prior post-rebase full build compiled the runtime and tests. CMake did not find the Vulkan SDK, so Vulkan GPU tools and tests were not configured.
- Focused CTest on the current base: `loader`, `pe_loader`, and `xgameruntime` passed (3/3).
- Final full CTest: all 12 configured tests passed — `loader`, `pe_loader`, `device_proxy`, `root_sig`, `command_list`, `descriptor_table`, `bink_decode`, `bink_audio`, `xmem`, `xgameruntime`, `root_signature`, and `root_signature_oracle` (12/12, 0.73 seconds).
- Generated fixture CLI dry run: `tiny_exe.exe` and `tiny_dll.dll` mapped as two modules with zero unresolved imports. `--dry-run` was used; no fixture code was executed.
- All fixtures are locally compiler-generated and contain no game data. The optional package test was not enabled; no `LEONIDA_GAME_DIR` was set and no game executable was mapped or executed.

### Remaining blockers and next task

- No authorized complete GTA VI executable/package was supplied. The real executable's imports and startup sequence therefore remain unverified; no game-load, initialization, or main-menu milestone is claimed.
- The Vulkan SDK is absent, blocking Vulkan-backed GPU validation in this environment. The CPU loader path builds without it.
- The connected GitHub account currently exposes no owned Leonida fork, and the GitHub integration has no fork-creation action. The local branch cannot be published until a `kvdroxz/Leonida` fork exists and write access is available. No changes were pushed upstream.
- Next: once a fork is available, publish this focused branch and open a draft PR. For boot work, obtain an explicitly authorized complete game executable/package, then use the existing `first_look` import-analysis path without executing the game image; implement only the first evidenced missing startup import.
