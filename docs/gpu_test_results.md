# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](gpu-test-results/RX9070GRE.txt) |
| NVIDIA GeForce RTX 4070 Ti (Ada), subgroup 32..32 | NVIDIA 621.14, Vulkan SDK 1.4.363.0, Windows 11 Pro 26300 | `dde82f27e` | 270/270 translated and valid | 176/211 pass, 35 skipped | 192/241 passed, 48 skipped; emulated 223/241 passed, 17 skipped | `ms_wave_info_s3`, every run (see below) | [local note](../gpu-test-results/RTX4070Ti_Windows.txt) |

## AMD Radeon RX 9070 GRE — 2026-11-23

```
GPU:        AMD Radeon RX 9070 GRE (RDNA4 / GFX1201, Navi 48, PCI 1002:7550), subgroup 32..64
Driver:     RADV (Mesa 26.3.1-arch3.5), Vulkan 1.4.363
Subgroup:   32..64 (minSubgroupSize 32, maxSubgroupSize 64; default 64)
OS:         CachyOS, kernel 7.3.5-2-cachyos
Commit:     05961f0eeb24486ce25ddc7dfb453ca87db2be5f (main, 2026-11-23)
Tools:      llvm-mc 23.1.5 (native), SPIRV-Tools v2026.4, CMake 4.4.5 + Ninja 1.13.2, GCC 16.2.1, Python 3.14.9
build.py:   264 shaders: 264 translated, 264 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   203 of 206 pass, 3 skipped (the three mesh fixtures built only by run_gpu_tests.py)
run_gpu_tests.py: 218/235 passed, 16 skipped, 1 failed (native and LEONIDA_W64_EMULATE=1 alike)
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0). Flaky on this RADV GPU: 1 pass and 2 fails in three isolated --filter reruns; same signature as the known round-44 mesh regression (#6172, fixed by #5741/#5855). Not a new failure.
Skipped:    16 skips are the documented "run by d3d12_*_test / w64_graphics.py" cases.
```

Setup: `cmake -S runtime -B runtime/build/linux -G Ninja -DCMAKE_BUILD_TYPE=Release -DLEONIDA_VULKAN_HEADERS_FETCH=ON`;
implicit Vulkan layers disabled (`VK_LOADER_LAYERS_DISABLE='~implicit~'`) because MangoHud/Steam layers are installed.
Inside `run_gpu_tests.py`, 241/241 shaders translated and passed `spirv-val` in both the native and emulated runs.
Every pixel test other than the flaky mesh fixture was exact (4096/4096).

AI-assisted (Hermes Agent): built `main` at `05961f0`, ran `build.py`, `check.py` and
`run_gpu_tests.py` (native and emulated) on this GPU, and verified the isolated reruns of the one failure.

## NVIDIA GeForce RTX 4070 Ti, Windows — 2026-11-23

```
GPU:        NVIDIA GeForce RTX 4070 Ti (Ada, AD104), subgroup 32..32
Driver:     NVIDIA 621.14 (Windows 32.0.16.2114), Vulkan SDK 1.4.363.0
OS:         Windows 11 Pro 10.0.26300
Commit:     dde82f27e994337094afc476ded0b1426f6e87e2 (main, 2026-11-23)
Tools:      llvm-mc 21.1.8 (scripts/build_llvm_mc.ps1), SPIRV-Tools from the Vulkan SDK, MSVC 19.44.35225 (VS 2022 Build Tools 17.14), Windows SDK 10.0.26100, Python 3.11
build.py:   270 shaders: 270 translated, 270 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   176 of 211 pass, 35 skipped (30 need full 64-wide compute subgroups, 3 mesh fixtures built only by run_gpu_tests.py, 2 need a 64-wide subgroup on request)
run_gpu_tests.py: 192/241 passed, 48 skipped, 1 failed
run_gpu_tests.py, LEONIDA_W64_EMULATE=1: 223/241 passed, 17 skipped, 1 failed
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0). Not flaky here: it failed in both full runs and in 5 of 5 isolated --filter reruns. The fixture was already in main at aa2988cc, where the #166 entry for the same GPU on NVIDIA 621.01 reports no failures, so either the driver (621.01 -> 621.14) or a change since aa2988cc is involved; not bisected. Same signature as #6172 (RADV) and #6497 (ANV).
Skipped:    native: the 31 wave64 tests (32-wide subgroups) plus the documented "run by d3d12_*_test / w64_graphics.py" cases; with emulation only the latter and ps_w64x_readlane (no 64-wide fragment subgroup).
```

Setup: `scriptsuild_runtime.ps1`; `LLVM_MC` pointed at the `llvm-mc.exe` built by `scriptsuild_llvm_mc.ps1`;
the Vulkan SDK's `Bin` folder on `PATH` for `spirv-as` and `spirv-val`. No game files used.

AI-assisted (Claude): built `main` at `dde82f27e`, ran `build.py`, `check.py` and `run_gpu_tests.py` (native and
emulated) on this GPU, and reran the one failure five times in isolation.
