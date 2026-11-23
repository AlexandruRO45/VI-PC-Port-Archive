# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](gpu-test-results/RX9070GRE.txt) |
| NVIDIA GeForce RTX 3060 Ti (Ampere, GA104), subgroup 32..32 | NVIDIA 610.88 (32.0.16.1088), Windows 10 19045 | `dde82f2` | 270/270 translated and valid | 176/211 pass, 35 skipped | 192/241 passed, 48 skipped (native); 223/241 passed, 17 skipped (`LEONIDA_W64_EMULATE=1`) | `ms_wave_info_s3`, known (see below) | [local note](../gpu-test-results/RTX3060Ti.txt) |

## NVIDIA GeForce RTX 3060 Ti — 2026-11-23

```
GPU:        NVIDIA GeForce RTX 3060 Ti (Ampere / GA104, PCI 10de:2489), subgroup 32..32
Driver:     NVIDIA 610.88 (32.0.16.1088), Windows 10 Pro 10.0.19045
Subgroup:   32..32 (default 32); wave64 runs emulated
CPU:        AMD Ryzen 5 1600, 16 GB RAM
OS:         Windows 10 Pro, build 19045
Commit:     dde82f27e994337094afc476ded0b1426f6e87e2 (main, 2026-11-23)
Tools:      llvm-mc 21.1.8 (AMDGPU target), SPIRV-Tools v2026.4, glslang 16.6.0, CMake 3.31.6 + Ninja 1.12.1, MSVC 19.44.35229, Windows SDK 10.0.26100, Python 3.14.3
build.py:   270 shaders: 270 translated, 270 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   176 of 211 pass, 35 skipped, 0 failed
run_gpu_tests.py: 192/241 passed, 48 skipped, 1 failed (native); 223/241 passed, 17 skipped, same failure (LEONIDA_W64_EMULATE=1)
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0). Same signature as the known round-44 mesh regression (#6172, fixed by #5741/#5855). Not a new failure.
Skipped:    48 skips native / 17 emulated: the wave64 cross-half cases need a 64-wide subgroup, ps_w64x_readlane cannot require a 64-wide fragment subgroup, and the rest are the documented "run by d3d12_*_test" cases.
```

Setup: `cmake -S runtime -B runtime/build -G "Visual Studio 17 2022" -A x64 -DLEONIDA_VULKAN_HEADERS_FETCH=ON`,
with `LLVM_MC`, `spirv-as`/`spirv-val` and `glslangValidator` on `PATH` before configuring (otherwise the
translated-shader tests are not built).

`LEONIDA_W64_EMULATE=1` turns the 48 native skips into 17: the 31 wave64 cross-half cases that a 32-wide
subgroup cannot run natively are translated with emulation instead, and all 31 pass. Every pixel test other
than the known mesh fixture was exact.

AI-assisted: built `main` at `dde82f2`, ran `build.py`, `check.py` and `run_gpu_tests.py` (native and
emulated) on this GPU.

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
