# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](gpu-test-results/RX9070GRE.txt) |
| NVIDIA GeForce RTX 4090 (Ada, AD102), subgroup 32..32 | NVIDIA 621.14, Vulkan SDK 1.4.363.0, Windows 11 | `867d922` | 270/270 translated and valid | 176/211 pass, 35 skipped | 192/241 passed, 48 skipped (223/241 passed, 17 skipped with `LEONIDA_W64_EMULATE=1`) | `ms_wave_info_s3`, 3 of 3 reruns (see below) | see below |

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

## NVIDIA GeForce RTX 4090 — 2026-11-23

```
GPU:        NVIDIA GeForce RTX 4090 (Ada, AD102), subgroup 32..32
Driver:     NVIDIA 621.14, Vulkan 1.4.357 (Vulkan SDK 1.4.363.0 loader)
OS:         Windows 11 IoT Enterprise LTSC 2024 (10.0.26200)
Commit:     867d922f09b86583859796a174df89555fb32da8 (main, 2026-11-23)
Tools:      llvm-mc 23.1.6 (conda-forge llvm-tools), SPIRV-Tools v2026.4, MSVC 19.44 (VS Build Tools 2022), CMake 4.4.4, Python 3.13.15
build.py:   270 shaders: 270 translated, 270 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   176 of 211 pass, 35 skipped
run_gpu_tests.py: 192/241 passed, 48 skipped, 1 failed (native)
run_gpu_tests.py: 223/241 passed, 17 skipped, 1 failed (LEONIDA_W64_EMULATE=1)
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0). Fails 3 of 3 isolated --filter reruns, so it is not flaky on this GPU; same signature as #6172 and #5273. The failure is identical in the native and emulated runs.
Skipped:    native 48 = 30 "no 64-wide compute subgroup on this device", 1 "no 64-wide subgroup", 1 "cannot require a 64-wide fragment subgroup: subgroup 32..32", 16 "run by d3d12_*_test / w64_graphics.py". Emulated 17 = the 16 "run by" cases plus the fragment-subgroup one.
```

Setup: `cmake -S runtime -B runtime/build -G "Visual Studio 17 2022" -A x64`, then
`cmake --build runtime/build --config RelWithDebInfo --target vk_compute_test vk_store_test vk_harness vk_texture_test vk_bindless_test vk_sampler_test`.
Inside `run_gpu_tests.py`, 247/247 shaders translated and passed `spirv-val` in both runs. Emulating wave64 with
`LEONIDA_W64_EMULATE=1` runs the 31 tests that are skipped natively because this 32-wide device cannot provide a
64-wide compute subgroup.

AI-assisted (Claude Code): built `main` at `867d922`, ran `build.py`, `check.py` and `run_gpu_tests.py` (native and
emulated) on this GPU, and reran the one failure in isolation three times.
