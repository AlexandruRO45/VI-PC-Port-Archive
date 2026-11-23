# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](gpu-test-results/RX9070GRE.txt) |
| NVIDIA GeForce RTX 3070 Ti (Ampere), subgroup 32..32 | NVIDIA 621.01, Vulkan 1.4.357, Windows 11 Pro 10.0.26300 | `dde82f2` | 270/270 translated and valid | 176/211 pass, 35 skipped | native 192/241 passed, 48 skipped; emulated 223/241 passed, 17 skipped | `ms_wave_info_s3`, 3 of 3 reruns (see below) | [local note](../gpu-test-results/RTX3070Ti.txt) |

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

## NVIDIA GeForce RTX 3070 Ti — 2026-11-23

```
GPU:        NVIDIA GeForce RTX 3070 Ti (Ampere, 8 GB), subgroup 32..32
Driver:     NVIDIA 621.01 (driverVersion 621.1.0.0), Vulkan 1.4.357
OS:         Windows 11 Pro 10.0.26300
Commit:     dde82f27e (main, 2026-11-23)
Tools:      llvm-mc 21.1.8 (scripts/build_llvm_mc.ps1), SPIRV-Tools v2026.4 (Vulkan SDK 1.4.363.0), CMake 3.31.6-msvc6 + Ninja (VS 2022 Build Tools), Python 3.15.0
build.py:   270 shaders: 270 translated, 270 pass spirv-val (6 mesh fixtures not built: ms_clip, ms_f16_pack, ms_tri, ms_v5_workgroup, ms_w64x_ballot, ms_wave_info_s3)
check.py:   176 of 211 pass, 35 skipped
run_gpu_tests.py: native 192/241 passed, 48 skipped, 1 failed; LEONIDA_W64_EMULATE=1 223/241 passed, 17 skipped, 1 failed
Failed:     ms_wave_info_s3: at (0,0) got (-1,-1,-1,-1), want (0.0078,0.0078,0.5,1.0), native and emulated. Not flaky here: 3 of 3 isolated --filter reruns fail the same way.
```

Built with `scripts\build_runtime.ps1` (RelWithDebInfo). Inside `run_gpu_tests.py`, 247/247 shaders translated and
passed `spirv-val` in both runs. The failure has the signature of #6172 (round-44 mesh regression, fixed by
#5741/#5855) and of the flaky RX 9070 GRE entry above, but it is not flaky on this GPU. The llvmpipe note in
[testing.md](testing.md) records NVIDIA giving the expected frame for `ms_wave_info_s3` earlier (#3838), so this may
be a regression on NVIDIA at `dde82f27e`.

AI-assisted (Claude Code): built `main` at `dde82f27e`, ran `build.py`, `check.py` and `run_gpu_tests.py` (native
and emulated) on this GPU, and the isolated reruns of the one failure.
