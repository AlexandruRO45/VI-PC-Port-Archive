# Vice-Tay

Vice-Tay is an experimental Windows compatibility layer for the Xbox Series build of Grand Theft Auto VI. It is based on [VIctory](https://github.com/Marvisak/VIctory). It runs the title's x86-64 code natively, implements the Xbox services it uses, and translates console GPU programs to DXIL for Direct3D 12. The runtime operates offline. The repository contains source code and synthetic test fixtures.

## Current status

The development build reaches the opening screens, loads saves, and has shown the prologue under player control. Rendering and frame pacing remain work in progress. Daylight scenes can show an overexposed sky, jagged shadows, and missing details. Ray-traced lighting is incomplete, and turning the camera or moving between scenes can cause long frames. Campaign completion and sustained stability are still unverified.

The latest local run on 25 November 2026 reached the 3D opening cinematic on `main` at `b033e17a`. The cinematic had dark silhouettes and visible lighting defects while shader compilation was active. The gameplay measurements below come from a separate run in the prologue.

Recent changes on `main` include recovery after texture remap queue overflow, pooled GPU upload buffers for texture copies, and reuse of pipeline references within a command list. [PR #14](https://github.com/TaynanGT/Vice-Tay/pull/14) records the changes and their sources.

## Local test results

### Test system

| Component | Test system |
| --- | --- |
| OS | Windows 11 Pro 26H2, build 26300.10075 |
| CPU | AMD Ryzen 9 9950X3D |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| GPU driver | NVIDIA 621.40 |
| RAM | 48 GiB DDR5 |

### 25 November 2026 — `main` at [`b033e17a`](https://github.com/TaynanGT/Vice-Tay/commit/b033e17a)

This was a diagnostics build in Series X mode with command relay level 2 (`VICTORY_RELAY=2`), frame capture enabled, and a 1920 × 1080 window limit. The run started with an existing host pipeline library and an empty shader cache. The renderer was still compiling shaders during the test.

| Segment | UTC sample times | Samples | Median FPS | FPS range | Median frame time |
| --- | --- | ---: | ---: | ---: | ---: |
| Early startup and loading | 20:40:25–20:41:01 | 8 | 28.0 | 25.5–30.7 | 35.7 ms |
| Later loading intervals | 20:41:06–20:41:39 | 7 | 15.2 | 0.6–22.3 | 65.6 ms |
| Opening 3D cinematic | 20:41:46–20:43:01 | 11 | 0.8 | 0.1–1.1 | 1,207 ms |

Each row summarises the renderer's periodic FPS and frame-time samples. The cinematic row includes intervals with at least 500 draw calls per frame. The first of those intervals appeared **88.5 s** after the first runtime log entry. The window was closed at **2 min 56 s**, during the cinematic, and the process exited with code 0.

### Earlier camera and transition measurements

The 24 November 2026 run used revision [`05a78e3`](https://github.com/TaynanGT/Vice-Tay/commit/05a78e37) in Series X mode at 2560 × 1440 internal resolution and 3840 × 2160 output. The command relay was set to level 2. Texture prefetching and the ray-tracing structure builder were disabled.

| Prologue segment | Method | FPS | Frame time |
| --- | --- | ---: | --- |
| Camera stationary | PresentMon, 268 presents over 35.76 s | 7.47 | 132.2 ms median, 157.2 ms at the 95th percentile |
| Turning the camera | Renderer log, 248 frames over 41.2 s | 6.01 | 166.3 ms mean |
| Scene transition | PresentMon, 155 presents over 35.51 s | 4.34 | 152.9 ms median, 521.7 ms at the 95th percentile |

PresentMon FPS uses the intervals between presents over the sampled segment. These three segments describe different types of work. A matched before-and-after route is still needed to quantify the effect of the latest renderer changes.

### Regression checks

On [`b033e17a`](https://github.com/TaynanGT/Vice-Tay/commit/b033e17a), the local graphics suite passed **262 tests** with **0 failures** and **9 ignored**. A Series X load-only check resolved every import in the executable and its four DLLs. [Windows CI](https://github.com/TaynanGT/Vice-Tay/actions/runs/45694584789) passed the workspace suite and Clippy in the normal and diagnostics configurations.

## Build and run

### Requirements

- Windows 10 or 11 x64 and an AVX2-capable CPU.
- A Direct3D 12 GPU with Shader Model 6.6. Rendering the game world needs mesh shader support.
- [Rustup](https://rustup.rs/) and the Visual Studio C++ build tools. The Rust release is pinned in [`rust-toolchain.toml`](rust-toolchain.toml).
- Your own compatible Xbox Series game files, stored outside this repository.

### Configuration

Copy the example configuration and set the paths in `victory.toml`. Game files, saves and generated caches belong outside the repository. The default cache and data folders are `%LOCALAPPDATA%\victory\cache` and `%LOCALAPPDATA%\victory\data`.

```powershell
Copy-Item .\victory.example.toml .\victory.toml
cargo build --locked --release -p victory
$victory = '.\target\release\victory.exe'
& $victory config
```

`VICTORY_CONFIG` selects a different configuration file. `VICTORY_GAME_DIR`, `VICTORY_CACHE_DIR`, and `VICTORY_DATA_DIR` override the corresponding entries. See [`victory.example.toml`](victory.example.toml) for the optional settings.

### Launch

Set `$title` to the actual executable in your game folder. The first command checks module loading and import resolution. Add `--execute` to launch. Series X is the default console profile.

```powershell
$title = 'D:\victory-data\game\game.exe'
& $victory run $title --console series-x
& $victory run $title --console series-x --execute
```

Prebuilding shaders and pipelines is optional. To fill the same cache used by `run`, pass the shader archive, executable and applicable shader DLLs to `translate`. The host pipeline library is specific to the GPU and driver. New configurations may spend time compiling pipelines as scenes load.

```powershell
$game = Split-Path -Parent $title
& $victory translate --cache --pipelines "$game\shaders.rpf" $title `
    "$game\XboxBvhBuilder.dll" "$game\amd_fidelityfx_upscaler_xbox.dll"
```

Use `--console series-s` to select the Series S profile. To use AMD's PC upscaler with that profile, set `VICTORY_FIDELITYFX_DIR` or `paths.fidelityfx_dir` to a [FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK) directory containing `amd_fidelityfx_upscaler_dx12.dll`. The log reports which implementation loads.

`VICTORY_WINDOW_MAX=1920x1080` limits the displayed window size while preserving its aspect ratio. It leaves the render resolution and swap-chain buffers at their original size.

### Controls

Use an XInput gamepad or the following keyboard controls while the title window has focus.

| Keyboard | Gamepad |
| --- | --- |
| `W A S D` / `I J K L` | Left / right stick |
| Arrow keys | D-pad |
| `Enter`, `Backspace`, `X`, `Y` | A, B, X, Y |
| `Q` / `E`, `Z` / `C` | Shoulders, triggers |
| `F` / `H` | Stick clicks |
| `Esc`, `Tab` | Menu, View |

## Development and diagnostics

[Architecture](docs/architecture.md) describes the loader, Xbox services, shader translator and renderer. Run `cargo xtask install-hooks` to enable the asset-policy check before each commit. [Asset policy](docs/asset-policy.md) covers what can be committed. Keep game-derived files, captures and dumps outside the repository.

```powershell
cargo test --workspace --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo build --profile diag --locked -p victory --features diagnostics
```

Shader validation uses `dxil.dll`. Set `VICTORY_DXIL` to a current x64 validator from a Windows SDK or the [DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler/releases) if the installed validator rejects Shader Model 6.6.

Use `victory --help` or `victory <command> --help` for the complete options. `inspect` reports executable structure and imports, `census` checks import coverage, `lib-exports` lists import-library symbols, and `shader-info` and `translate` describe or translate shader programs.

### Investigating a run

Report the Vice-Tay revision, Windows build, GPU and driver, console profile, renderer settings, and the scene that reproduces the problem. For performance reports, include the measured interval, the measurement tool, and whether the camera was moving. `crashes.log` is stored under the title's data directory.

| Setting | Use |
| --- | --- |
| `VICTORY_LOG` | Controls logging levels, for example `victory_loader=trace,info`. |
| `VICTORY_KEYS` | Schedules diagnostic input, for example `30:Enter,35:X`. |
| `VICTORY_CAPTURE` | Saves selected presented frames outside the repo in a diagnostics build. Set `VICTORY_CAPTURE_FRAMES` to choose frame numbers. |
| `VICTORY_CAPTURE_WORLD` | Also captures frames at offsets from the first frame with 500 draws. |
| `VICTORY_D3D12_DEBUG=1` | Enables the D3D12 debug layer. |
| `VICTORY_D3D12_DRED=1` | Records device removal data after a GPU failure. |
| `VICTORY_D3D12_WARP=1` | Uses the software D3D12 adapter for hardware versus software comparisons. |
| `VICTORY_CRASH_BREAKPOINTS=1` | Records first-chance breakpoints in `crashes.log`. |
| `VICTORY_PIPELINE_LOG` | Records pipeline creation, first use, and cache activity. |

If the loader reports a startup breakpoint at `+0x2237178` on Series 1.0.2.1, check that `appdata0.rpf` from the same package is present beside the executable. If Windows reports error 1455, check the system commit limit and paging-file capacity.

## Credits and licence

Vice-Tay builds on [VIctory](https://github.com/Marvisak/VIctory), with contributions and adaptations from [victory-fork](https://github.com/fxceless/victory-fork), [VItoriaBr](https://github.com/uiliannn/VItoriaBr), [KLBF Lab](https://github.com/PastiiJay/klbf-lab), and [VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD). Commit history records the sources of individual changes.

Vice-Tay is an independent project. Its source code is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE).
