# ztrx

**ztrx** runs the Xbox Series build of the game natively on Windows PCs with
NVIDIA graphics cards, from game files you supply yourself.

**This repository contains no game files or assets.** You need your own copy
of the game. Not affiliated with or endorsed by its developer or publisher.

## Status (2026-11-25)

Tested on Windows 11 with an **NVIDIA GeForce RTX 3050** (8 GB):

| | |
|---|---|
| Boot, first-boot screens, intro, opening credits | works |
| Prologue | playable from the cell to the boat chase: cell, cafeteria, robbery, police chase (wanted level, minimap), boat, binoculars |
| Stability | 1 h 30 min+ in one session without a crash |
| Frame rate | 1-3 fps in the world (the first run is slower while shaders translate; they are cached after that) |
| Known problems | anything in shadow is black and bright scenes overexpose (the game's ray-traced lighting does not run yet); some blocky artifacts; boats sit low in the water |

## Requirements

- Windows 10/11 x64, a CPU with AVX2
- A Direct3D 12 GPU with shader model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer)
- Rust (the version in `rust-toolchain.toml`)
- Several GB of free Windows commit memory (raise the paging file if `crashes.log` reports error 1455)

## Build

```powershell
cargo build --release -p victory
```

## Play

```powershell
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<your game folder>\<the game's exe>"
```

`scripts\play.ps1` keeps the shader cache and saves in folders you choose
(`-Cache`, `-Data`) and presses no buttons for you. Controller: XInput pads
(a DualShock/DualSense through DS4Windows works). With no pad connected the
keyboard acts as one while the game window has focus.

## Faster settings (optional, off by default)

`VICTORY_RELAY=2`, or `VICTORY_LATE_WAIT=3 VICTORY_WIDE_BACK=1` cut frame
time on the test machines. Set them in PowerShell before running the script.

## Debugging switches

| Switch | Effect |
|---|---|
| `VICTORY_CAPTURE=<dir>` + `VICTORY_CAPTURE_FRAMES=<list>` | save those frames as PNG (diagnostics build) |
| `VICTORY_KEYS=30:Enter,35:X` | press keys at those seconds (diagnostics build) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, report the commands in flight |
| `VICTORY_PIPELINE_WAIT_MS=24` | shorter per-frame wait for pipelines still being built |

Diagnostics build: `cargo build --profile diag -p victory --features diagnostics`

### Shader GPU tests with Agility SDK

On Windows x64 with the MSVC toolchain, the shader test executable can use
a local Agility SDK when the system D3D12 runtime lacks shader model 6.6.
Set the numeric version exported by the x64 `D3D12Core.dll` (608 for
package 1.608.2). Build the tests, then put the DLL in a `D3D12` folder
beside the test executable. Use SDK layers from the same package if
enabling the D3D12 debug layer.

```powershell
$env:VICTORY_TEST_AGILITY_SDK_VERSION = '608'
$env:CARGO_TARGET_DIR = Join-Path $PWD 'target\shader-agility'
$test = (cargo test --locked -p victory-shader --lib --no-run --message-format=json |
    ForEach-Object { ConvertFrom-Json $_ } |
    Where-Object { $_.reason -eq 'compiler-artifact' -and $_.target.name -eq 'victory_shader' -and $_.executable } |
    Select-Object -Last 1).executable
if ($LASTEXITCODE -ne 0 -or -not $test) { throw 'Shader test build failed' }
$sdk = Join-Path (Split-Path $test) 'D3D12'
New-Item -ItemType Directory -Force -Path $sdk | Out-Null
Copy-Item 'C:\path\to\sdk\x64\D3D12Core.dll' $sdk
& $test emit::host::sm66_device_probe --exact --ignored --nocapture
```

The probe reports which adapters can create a device with shader model 6.6.
Run the same executable with `--test-threads=1` for the normal shader suite.
Set `VICTORY_DXIL` to the path of the x64 `dxil.dll` if you need to pin
the validator version across runs.
Unset both environment variables to return to the default build.

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak) and victory-fork (fxceless).
