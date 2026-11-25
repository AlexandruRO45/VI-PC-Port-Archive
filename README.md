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

The script exits with `victory.exe`'s own exit code, so `$LASTEXITCODE` after
it tells a batch run whether the launch worked.

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
| `VICTORY_HOST_MAX_WAVE=32` | with `victory translate`: assume a host whose waves have at most this many lanes, as an NVIDIA card does — without it every wave-64 compute program looks refused. `VICTORY_WAVE_WHY=1` says why one keeps 64 lanes |
| `VICTORY_TRANSLATE_ONLY=<hash>,...` | with `victory translate`: those containers alone, by the 16-digit hash a run's log prints. Prefixes work, case is folded |
| `VICTORY_VIEW_LIMIT=<slot>` | with `victory translate`: assume the host's view heap is capped, shaders reading past `slot` getting its null view |
| `VICTORY_CARD_BUFFERS_MIN_MB=1` | copies of at least this many MiB of a title buffer go to the card's memory, where the card does the search for shader writes; the default is 16, so smaller copies stay on the CPU. The per-frame line names this switch — note `VICTORY_CARD_BUFFERS_MIN` is **not** a variable this build reads |
| `VICTORY_CARD_BUFFERS_MB=1024` | the total those copies may hold, default 512 |
| `VICTORY_TEXTURE_BUDGET_MB` | sets a limit on the bytes of texture copies the card may hold, instead of the default rule (a floor, then nine tenths of the budget the system gives the process) |
| `VICTORY_TEXTURE_GUARD=0` | hash view-memory textures at every check instead of once every eighth frame |

Diagnostics build: `cargo build --profile diag -p victory --features diagnostics`

### Running with captures or scripted input

`scripts\play.ps1` clears `VICTORY_CAPTURE`, `VICTORY_CAPTURE_FRAMES`,
`VICTORY_DUMP_TARGETS` and `VICTORY_KEYS` by default, because a normal run
should not capture or press anything. Pass `-Capture` to keep the capture
variables, `-Keys` to keep the key schedule, or `-KeepScriptedInput` for both:

```powershell
$env:VICTORY_CAPTURE = "$env:TEMP\ztrx-captures"
$env:VICTORY_CAPTURE_FRAMES = '60,120'
powershell -ExecutionPolicy Bypass -File scripts\play.ps1 -Game "<exe>" -Capture
```

Or run the diagnostics build directly, which the script is not involved in:

```powershell
cargo build --profile diag -p victory --features diagnostics
$env:VICTORY_CAPTURE = "$env:TEMP\ztrx-captures"
$env:VICTORY_CAPTURE_FRAMES = '60,120'
& .\target\diag\victory.exe run '<path to executable>' --execute
```

Set `VICTORY_CACHE_DIR` and `VICTORY_DATA_DIR` to your `-Cache` and `-Data`
paths to share the launcher's cache and saves.

## License

GPL-3.0-or-later (see `LICENSE`). Based on VIctory (Marvisak) and victory-fork (fxceless).
