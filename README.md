# Xbox2Windows

Xbox2Windows investigates **static recompilation of Xbox Series X|S (Scarlett) software for x64 Windows**. The CPU experiments translate source-built x64 functions from PE machine code through C++ to new Windows DLLs, including a direct call and cross-DLL imports by name and ordinal. The shader experiment translates selected gfx1013 pixel shaders through HLSL to desktop DXIL. Other experiments test the Windows runtime, render recovered texture levels and read script state data. Title CPU code execution, full-game startup and frame rate remain unverified.

New to the project? The [pinned guide in GitHub Discussions](https://github.com/R0Y00/Xbox2Windows/discussions/75) will help you choose a first contribution. Ask build and debugging questions in [Development Q&A](https://github.com/R0Y00/Xbox2Windows/discussions/categories/development-q-a).

## Current results and open work

| Area | Result on Windows | Open question or contribution |
|---|---|---|
| FidelityFX upscaler | The recompiled provider creates 11 resources at 720p → 1080p and a twelfth with the debug flag. On RTX 4060 Laptop and WARP, both the per-resource indices and the returned loader and provider handles passed. Recompiled graph pass 26 wrote the RCAS intermediate and the recompiled RCAS shader read it with five integer texel loads per output. Passes 24 and 25 supplied all 72 scratch reads on the 8×4 tile. Both adapters passed the zero and nonzero input comparisons through the final 256 pixels. [Scratch writers](docs/experiments.md#last-writers-of-the-pass-26-scratch-reads) · [Dispatch trace and replay](docs/experiments.md#fsr4-dispatch-through-the-rcas-input) · [RCAS format test](docs/experiments.md#rcas-intermediate-texel-access) · [Provider loop and desktop mappings](docs/experiments.md#provider-resource-loop-and-returned-handles) · [Recurrent texture pixel readback](docs/experiments.md#recurrent-texture-views-and-pixel-readback) | Exercise the full-wave LDS path of pass 26 with a larger tile using the verified scratch producers. |
| x64 context callbacks | The recompiled FidelityFX loader destroy function calls the bound provider and preserves its 32-bit result across both cleanup paths on Windows. The upscaler slot was identified in the local package. [Provider and Windows comparison](docs/experiments.md#fidelityfx-destroy-provider-and-cleanup) | Verify the upscaler's internal effect teardown with a successfully created graphics context. |
| CPU translation | A [leaf function](docs/experiments.md#offline-cpu-leaf-recompilation), a [direct call](docs/experiments.md#recompiling-a-direct-x64-call) and cross-DLL imports [by name](docs/experiments.md#calling-another-dll-through-a-named-pe-import) and [by ordinal](docs/experiments.md#calling-an-ordinal-only-export) match the original fixtures on Windows. The import fixtures and a relocated variant also run from an image [mapped without LoadLibrary](docs/experiments.md#mapping-an-image-without-loadlibrary), with imports bound through a host resolver. The [recompiled Bink allocation and release callers](docs/experiments.md#bink-allocation-and-release-on-the-windows-heap) passed size, alignment, failure and release checks against a Windows heap. | Verify allocation and release when the caller installs custom callbacks. |
| Shader translation | Supported pixel programs sample a texture, read constants and export color. D3D12 readback is checked against a CPU reference. [Translation and bindings](docs/experiments.md#offline-pixel-shader-recompilation) | Supply a source-built instruction case with known register inputs and an expected color. |
| BC7 textures | A translated shader samples recovered mip levels on hardware and WARP. Requests for missing high-resolution levels use the first resident level. [Mip fallback tests](docs/experiments.md#drawing-bc7-textures-with-a-recompiled-pixel-shader) | Match a variable BCPack lane against independently known BC7 blocks. [Issue #15](https://github.com/R0Y00/Xbox2Windows/issues/15) |
| Script states | A native x64 probe follows state-tree links and reproduces distinct camera limits for front and rear selections. The replay supplies the selection requests. [State replay](docs/experiments.md#state-linked-settings-in-a-script-archive) | Identify an event or caller that supplies a context and state ID, with a control input that takes a different branch. [Issue #19](https://github.com/R0Y00/Xbox2Windows/issues/19) |
| Windows runtime and GPU | Source-built checks cover PE imports, heap operations, root signatures, compute and ray tracing. Five Radeon GPUs passed the native wave64 BVH test. [Hardware coverage](#hardware-coverage) · [Experiment notes](docs/experiments.md) | Share a run on a new adapter or a driver retest in [the hardware thread](https://github.com/R0Y00/Xbox2Windows/issues/1). |

The public suite builds its own fixtures. A contributor can reproduce the checks below on a PC without a game installation. The standalone release packages the GPU check. To run the script, shader and texture development suite, build from source.

## Run the GPU check

Download the [Windows x64 GPU check and SHA-256](https://github.com/R0Y00/Xbox2Windows/releases/tag/gpu-check-v0.1.0), verify and extract the ZIP, then run this in PowerShell from the extracted folder. The package contains its own shaders and needs no compiler, console or game files.

```powershell
.\Run-GpuCheck.ps1
```

Review `gpu-report-hardware.md` and paste it in the [hardware coverage thread (#1)](https://github.com/R0Y00/Xbox2Windows/issues/1). Reports with a failure or capability skip are welcome. [Package instructions](tools/gpu-check/README.md)

### Hardware coverage

| Adapter or backend | Default native lanes | Forced wave64 | BVH differential | Evidence |
|---|---:|---|---|---|
| NVIDIA GeForce RTX 2060 | 32 | Skipped | 408 queries passed | [Maintainer baseline](docs/experiments.md#independent-bvh-checks-on-pc) |
| NVIDIA GeForce RTX 3070 (driver 32.0.16.1088) | 32 | Skipped, lane range 32–32 | 408 queries passed | [RTX 3070 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063566753) · [32.0.16.2101 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064780514) |
| NVIDIA GeForce RTX 4060 Laptop GPU | 32 | Skipped | 408 queries passed | [Second maintainer workstation](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7058907095) · [Independent Windows run, driver 32.0.16.2124](https://github.com/R0Y00/Xbox2Windows/issues/60) · [32.0.16.1742 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064580375) |
| NVIDIA GeForce RTX 5070 | 32 on both drivers | Skipped on both, lane range 32–32 | 408 queries passed per run | [32.0.16.1692, debug layer unavailable](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063361337) · [32.0.16.2135 retest, debug layer enabled](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063626309) |
| AMD Radeon RX 6600 (driver 32.0.21045.16010) | 32 | Passed, 64 lanes | 408 queries passed | [RX 6600 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063312495) · [32.0.11037.4004 retest](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063794169) · [32.0.21045.29003 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064667296) |
| AMD Radeon RX 6750 XT (driver 32.0.21045.29003) | 32 | Passed, 64 lanes | 408 queries passed | [RX 6750 XT report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7062253227) |
| AMD Radeon RX 6800 XT (driver 32.0.21043.5001) | 32 | Passed, 64 lanes | 408 queries passed | [RX 6800 XT report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063629409) |
| AMD Radeon RX 9070 (driver 32.0.22042.14002) | 64 | Passed, 64 lanes | 408 queries passed | [RX 9070 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7062384428) |
| AMD Radeon RX 9070 XT (driver 32.0.32015.16001) | 64 | Passed, 64 lanes | 408 queries passed | [RX 9070 XT report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063426420) · [32.0.32029.2010 retest, D3D12 warning 989](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063787553) · [32.0.32015.16001 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064744246) |
| NVIDIA GeForce RTX 4060 Ti (driver 32.0.16.1742) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063740292) |
| AMD Radeon RX 7900 GRE (driver 32.0.32015.7029) | 32 | Passed, 64 lanes | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063746749) |
| NVIDIA GeForce RTX 4070 Ti SUPER (driver 32.0.16.1088) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063783462) |
| NVIDIA GeForce RTX 5080 (driver 32.0.16.2124) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7063834333) |
| AMD Radeon RX 9060 XT (driver 32.0.32029.2010) | 64 | Passed, 64 lanes | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064143537) |
| NVIDIA GeForce RTX 4060 (driver 32.0.16.1088) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064173447) |
| NVIDIA GeForce RTX 4050 Laptop GPU (driver 32.0.16.1088) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064193927) |
| NVIDIA GeForce RTX 2060 SUPER (driver 32.0.16.2124) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Windows hardware report](https://github.com/R0Y00/Xbox2Windows/issues/57) |
| NVIDIA GeForce RTX 3060 (driver 32.0.16.2124) | 32 | Skipped, lane range 32–32 | 408 queries passed | [Windows hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064352679) |
| Intel Arc B570 (driver 32.0.101.9175) | 32 | Skipped, lane range 16–32 | 408 queries passed | [Windows hardware report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064376175) |
| AMD Radeon RX 6600 LE | 32 | Skipped, Shader Model 6.5 | 408 queries passed | [32.0.21045.29003 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064448893) |
| NVIDIA GeForce RTX 3080 Ti | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.1742 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064502245) |
| NVIDIA GeForce RTX 4070 SUPER | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.2135 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064571169) |
| NVIDIA GeForce GTX 1080 Ti | 32 | Skipped, lane range 32–32 | Skipped, DXR tier 1.0 | [32.0.15.8266 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064625762) |
| AMD Radeon RX 6700 | 32 | Passed, 64 lanes | 408 queries passed | [32.0.21045.16010 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064650872) |
| NVIDIA GeForce RTX 5060 | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.1742 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064676684) · [32.0.16.1656 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064685186) |
| NVIDIA GeForce RTX 4090 | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.2135 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064699143) |
| NVIDIA GeForce RTX 3080 Ti Laptop GPU | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.2135 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064722299) |
| AMD Radeon RX 6700 XT | 32 | Passed, 64 lanes | 408 queries passed | [32.0.21045.16010 report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064730965) |
| NVIDIA GeForce RTX 4090 Laptop GPU | 32 | Skipped, lane range 32–32 | 408 queries passed | [32.0.16.1656 report](https://github.com/R0Y00/Xbox2Windows/issues/64) |
| Microsoft WARP (software) | 4 | Skipped | 408 queries passed | [Maintainer baseline](docs/experiments.md#independent-bvh-checks-on-pc) |

On CachyOS, an RTX 3060 completed 408 comparisons through Proton and vkd3d-proton with NVIDIA driver 620.70.06. The adapter reported a 32-lane maximum and skipped forced wave64. [Linux translation-layer report](https://github.com/R0Y00/Xbox2Windows/issues/1#issuecomment-7064204388)

The eight community reports cover seven GPU models at commit `def386a`. Each Radeon completed all 38 forced-wave64 groups with zero mismatches. The RX 9070 XT, RTX 3070 and RTX 5070 driver retest reported zero D3D12 debug warnings and errors in their passing checks.

Maintainers link independent results to the original report and record the adapter, driver, commit and outcome. Names or handles are added on request. The hardware-report and independent-build milestones are complete. The remaining community milestone is a fix verified by the contributor who reported the problem.

## First build

On x64 Windows, install Visual Studio's **Desktop development with C++** workload, including x64 MSVC, the C++ CMake tools with Ninja, and Windows SDK 10.0.26100 with DXC. From the repository root in PowerShell, run:

```powershell
.\tools\dev.ps1
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error -R 'desktop_root_signature|era_heap_allocation|offscreen_warp'
```

These three tests exercise a D3D12 root signature, heap allocation and software rendering. To enable the script, shader, texture and PE fixture tests, install x64 Python 3.11 or newer and configure again. Windows CI uses Python 3.12.

```powershell
python -m venv out/pe-env
.\out\pe-env\Scripts\python.exe -m pip install -r requirements-dev.txt
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --no-tests=error
```

An independent contributor passed all three smoke tests from a clean checkout of `71bb9f4` on Windows 11 build 26300.9974 with VS Build Tools 2022 17.14 and Windows SDK 10.0.26100. The tests passed again after an incremental rebuild at `2465837`. With Python 3.14.10, the full suite had 66 and 68 tests respectively, with zero failures in either run. Each run skipped forced wave64 on the RTX 5080 and WARP because their reported lane ranges excluded 64. [Original clean-build report](https://github.com/R0Y00/Xbox2Windows/issues/2#issuecomment-7063351821)

## Choose an experiment

| Area | A contribution you can verify | Focused CTest command |
|---|---|---|
| Static CPU recompilation | Compare leaf, chained-range and cross-function translations with their source-built originals and negative controls. | `ctest --preset dev --no-tests=error -R '^cpu_' -V` |
| Script states | Reproduce a parent/child lookup, an ambiguous selection or a damaged sibling link with a synthetic tree. | `ctest --preset dev --no-tests=error -R '^script_(state_literals\|path_links)$' -V` |
| Texture recovery | Check a packed mip tail, a first refill byte, or a resident mip drawn on the software adapter. | `ctest --preset dev --no-tests=error -R '^(bcpack_prefix_and_mip_tail\|bc7_draw_warp)$' -V` |
| Static shader recompilation | Translate the single-texture pixel shader fixtures to desktop DXIL and check every output channel. [Supported instructions and bindings](docs/experiments.md#offline-pixel-shader-recompilation) | `ctest --preset dev -R '^pixel_aot_' -V` |
| D3D12 and ray tracing | Compare the CPU reference, custom BVH shader and DXR on another adapter. Record the driver and native wave width. | `ctest --preset dev -R '^bvh_(differential\|wave64)_hardware$' -V` |
| Windows runtime | Add a source-built allocation case with a measurable Windows result. | `ctest --preset dev -R 'era_heap\|desktop_memory_query'` |
| PE tooling | Reproduce an import-resolution edge case with a DLL or EXE built from source. | `ctest --preset dev -R 'static_exe_intake\|conditional_import_paths'` |
| Build and documentation | Try the first build from a clean clone. Identify the first missing prerequisite or unclear instruction. | Use the three-test command above. |

For task ownership, evidence requirements and PR review, see the [contribution guide](CONTRIBUTING.md). [Open an issue](https://github.com/R0Y00/Xbox2Windows/issues/new/choose) · [Community conduct](CODE_OF_CONDUCT.md) · [Private security reports](SECURITY.md)

## Case study and provenance

The installation metadata inspected for this case study lists a Scarlett executable. The main executable was absent from the inspected files. The public regression suite uses source-built fixtures and independently built shaders. [Intake methods and evidence](docs/experiments.md#static-executable-intake)

Windows Authenticode verification on the local copy reported `HashMismatch` for 12 DLLs and `NotSigned` for four. Three independently compared whole-file SHA-256 values matched the community reports. The cause of the digest mismatch and the original signing path remain undetermined. [Methods and fingerprints](https://github.com/R0Y00/Xbox2Windows/issues/52)

## License

Original project code and documentation use the [MIT License](LICENSE). Third-party games, assets, SDKs and trademarks belong to their respective owners. Xbox2Windows is an independent community project, unaffiliated with Rockstar Games, Take-Two, Microsoft or AMD.
