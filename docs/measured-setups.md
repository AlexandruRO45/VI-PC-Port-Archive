# Measured setups

Tester setups for launching the title under VIctory, with the settings actually in force and what was measured with them. Each value says how it was obtained: **measured** (from the VIctory log of the run, the running process or a system tool), **observed** (seen on screen by the tester), **read** (from the source). Everything comes from running the tester's own copy of the game; nothing comes from the game's code or files. French version: [fr/configurations-mesurees.md](fr/configurations-mesurees.md).

## RTX 4070 Ti SUPER, VIctory `7b0f736` with `VICTORY_RELAY=2` (minecraftafk072, 24 November)

This is the setup of `main` at `b547bb2` (after #172), before #178 moved the pin to `fd021bf` and dropped `VICTORY_RELAY` from the launcher. It held more than 18 minutes in the world without a device loss. On today's `main`, `-Relay` sets the same `VICTORY_RELAY=2`, but with the newer pin `fd021bf`: the figures below hold for `7b0f736` only.

### Machine

| | |
| --- | --- |
| CPU | AMD Ryzen 9 7900X3D (its RDNA 2 integrated GPU is present; VIctory does not run on it) |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, driver 32.0.16.1742. `nvidia-smi` lists `victory.exe` on it (measured). |
| RAM | 32 GB |
| Display | 1920x1080 at 240 Hz, on the RTX |
| OS | Windows 11 Pro, build 26300 |

### Software

- AnyXboxX `main` at `b547bb2`.
- `third_party/victory` at `7b0f736` (fork [fxceless/victory-fork](https://github.com/fxceless/victory-fork)), built by the launcher: `cargo build --profile diag -p victory --features diagnostics`.
- Launched with `.\tools\lancer_victory.ps1 -GameDir <game folder>`, without `-Dred`, `-SkipPipelines` or `-NoTunedSettings`. The launcher runs `victory.exe run <game folder>\game_scarlett_master.exe --execute`, without `--console`, so with the default Series X profile (read).

### Settings in force

Read from the environment of the running `victory.exe` on 24 November at 22:18 UTC (measured, filtered on the `VICTORY_` prefix). They are the values that `tools/lancer_victory.ps1` sets at `b547bb2` (read).

| Variable | Value |
| --- | --- |
| `VICTORY_SHADER_LOOP_LIMIT` | `256` |
| `VICTORY_LATE_WAIT` | `3` |
| `VICTORY_WIDE_BACK` | `1` |
| `VICTORY_RELAY` | `2` |
| `VICTORY_CARD_BUFFERS_MIN_MB` | `1` |
| `VICTORY_CARD_BUFFERS_MB` | `1024` |
| `VICTORY_GAME_DIR` | the game folder |
| `VICTORY_D3D12_DRED` | not set (DRED off) |
| `VICTORY_SKIP_PIPELINES` | not set (no pipeline skipped) |

### Results

Run of 24 November from 21:31 UTC, these exact settings, VIctory log kept outside the repository, figures from its stats lines (one every 5 s), measured:

| | |
| --- | --- |
| World reached (1,000 draws per frame or more) | 110 s after launch. The shader cache had been filled at loop limit 32 the run before; the loop limit is part of the cache key, so part of it was rebuilt (read). |
| Time in the world | 18.1 min, then stopped by the tester's 20-minute limit |
| Device loss, `ERROR` lines | none, none |
| `WARN` lines | 60, mostly pipeline creation methods not implemented and shaders that do not translate |
| Frame rate in the world | 5.1 fps median over the stats lines with 100 draws or more, 195 ms per frame (106 ms on the GPU, 126 ms copying back) |
| Per minute | 4.9 to 5.3 fps from minute 2 to minute 15, 4.7 to 4.8 fps from minute 16 to minute 18 |
| Draws per frame | about 4,780 (median), about 345 dropped without a host pipeline |

Other runs on the same machine, the same day (measured, from their VIctory logs):

- With the launcher of `7b0f736` (#171: loop limit 32, DRED on): 18.8 min in the world, no device loss or `ERROR`; 4.61 fps from minute 3 to minute 6.9.
- With these settings plus `-Dred`: 6.9 min in the world, no device loss or `ERROR`; 4.70 fps from minute 3 to minute 6.9, against 5.13 fps without DRED. DRED costs about 18 ms per frame (+9.5 %) on this card.

A second launch with these settings, at 22:10 UTC from the tester's own terminal, without a kept log: still running after 12 minutes (measured with `Get-Process`); the tester reports it runs really well (observed).

Observed by the tester: with loop limit 32 the hair looks wrong; with 256, the loading screen and the intro flicker a little (probably the startup flicker of #174). The skin and subtitle problems that the fork's issue #89 ties to `VICTORY_RELAY` were not checked in these runs.

### Reproduce on this setup

At `b547bb2` the launcher sets all these values itself. The launcher only builds VIctory when `third_party\victory\target\diag\victory.exe` is missing (read), so rebuild by hand after moving the submodule:

```powershell
git checkout b547bb2
git submodule update --init third_party/victory
cargo build --profile diag -p victory --features diagnostics --manifest-path third_party/victory/Cargo.toml
.\tools\lancer_victory.ps1 -GameDir <game folder>
```

Open a new terminal first: the launcher leaves its variables in the PowerShell session, and a `VICTORY_D3D12_DRED` left by an earlier launch keeps DRED on.

On the current `main`, the closest launch is `.\tools\lancer_victory.ps1 -GameDir <game folder> -Relay`: the same settings with the pinned VIctory `fd021bf`, not measured on this machine yet. Rebuild VIctory after the submodule moves, as above.
