# GPU test results

This page keeps a local record of GPU test runs. The project's public tracker is
[#166](https://github.com/ItsNotRexed/Leonida/issues/166) (one comment per GPU and driver);
post counts and pass/fail lines there, never game shaders, logs or private paths.

| GPU | Driver, OS | Commit | build.py | check.py | run_gpu_tests.py | Failed | Entry |
|---|---|---|---|---|---|---|---|
| AMD Radeon RX 9070 GRE (RDNA4, GFX1201), subgroup 32..64 | RADV, Mesa 26.3.1-arch3.5, Vulkan 1.4.363, CachyOS | `05961f0` | 264/264 translated and valid | 203/206 pass, 3 skipped | 218/235 passed, 16 skipped (native and emulated) | `ms_wave_info_s3`, flaky (see below) | [local note](../gpu-test-results/RX9070GRE.txt) |
| NVIDIA GeForce RTX 5060 (Blackwell), subgroup 32..32 | NVIDIA 616.56, Vulkan 1.4.312, Windows 11 | `867d922` | N/A (Windows host) | N/A (Windows host) | 228/240 ctest passed, 12 skipped, 0 failed | none | [local note](../gpu-test-results/RTX5060.txt) |

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

## NVIDIA GeForce RTX 5060 — 2026-11-23

```
GPU:        NVIDIA GeForce RTX 5060 (Blackwell, desktop, 8 GB VRAM, PCI 10de:2d06), subgroup 32..32
Driver:     NVIDIA 616.56, Vulkan 1.4.312
Subgroup:   32..32 (minSubgroupSize 32, maxSubgroupSize 32; default 32)
OS:         Windows 11 Pro 64-bit (build 26100)
CPU:        Intel CC150 (8C/16T @ 3.50 GHz, Coffee Lake Refresh, LGA1151v2 Z390)
RAM:        32 GB DDR4-2666
Commit:     867d922f09b86583859796a174df89555fb32da8 (main, 2026-11-23)
Tools:      MSVC 19.44.35220 / VS 2026 v18.8.0, CMake 3.31.6 + Ninja 1.13.2, Python 3.14.2
ctest:      228 of 240 passed, 12 skipped, 0 failed (100% pass of runnable runtime tests)
d3d12_vulkan_device_test: PASS (copy comparison: 0 of 75 cases differ from the host; Resolve comparison: 60/60 matched; 20/20 heaps matched; placed castable formats D3D-90 matched; clears, zero-filled first use, uploads, texture/buffer/BC copies, queries, GetCopyableFootprints, GetResourceTiling and fence semantics all matched)
d3d12_depth_resolve_test: PASS (51/51 depth/stencil resolves matched NVIDIA GeForce RTX 5060; 37/37 matched WARP; 88 host comparisons matched; 12 refusals kept destination)
d3d12_predication_atomic_copy_test: PASS (59/59 checks passed)
xg_xs_test: PASS (16/16 checks passed)
Failed:     none
Skipped:    12 skips are package-dependent tests requiring retail package or movie assets.
```

Setup: built runtime with MSVC x64 in `RelWithDebInfo` configuration (`cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo`).
Ran full CTest suite on native Windows 11 host (`ctest --test-dir runtime/build -C RelWithDebInfo --output-on-failure -j 4`): 228 passed, 12 skipped (retail package / assets required), 0 failed.
Verified all 75 copy and 60 resolve comparisons in `d3d12_vulkan_device_test`, 51/51 depth/stencil resolves in `d3d12_depth_resolve_test`, and 59/59 checks in `d3d12_predication_atomic_copy_test`. Resizable BAR is active (8192 MiB BAR1), placing upload heaps directly into device-local memory.
