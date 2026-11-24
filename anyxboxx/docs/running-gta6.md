> Français : [lancer-gta6.md](fr/lancer-gta6.md)

# Running GTA VI (Xbox Series 1.0.2.1 build) with AnyXboxX

This guide installs and starts the project, and lists everything you need to repeat our tests on the Xbox Series 1.0.2.1 build of GTA VI, up to where we are today: the first cutscene. It does not provide the game and does not explain how to get it: you bring your own copy. Never put game files in this repository or attach them to an issue.

The launch goes through [VIctory](https://github.com/Marvisak/VIctory) (Marvisak, GPL-3.0 or later), tracked here as the `third_party/victory` submodule, pinned to the branch `anyxboxx/fork-7b0-lot8` of the maintainer's fork [LooperSalty/victory](https://github.com/LooperSalty/victory), which assembles the fork [fxceless/victory-fork](https://github.com/fxceless/victory-fork) (GPL-3.0) at `7b0f736` and eight of its pull requests (see below), at commit `d138351`. The script `tools/lancer_victory.ps1` builds it and starts it on your game folder.

## Pinned branch: batch 5 plus one pull request (25 November)

`third_party/victory` now points to the branch `anyxboxx/fork-7b0-lot8` of the maintainer's VIctory fork [LooperSalty/victory](https://github.com/LooperSalty/victory), at commit `d138351de32f860ab9df1aa1aa736fce0a40365e`. It is batch 5 (below) plus one pull request of the fork [fxceless/victory-fork](https://github.com/fxceless/victory-fork), cherry-picked with `-x`:

| Fork pull request | Author | What it does | Commit |
| --- | --- | --- | --- |
| #11 | cxchhh | Shaders: parked lane-mask pairs a few instructions apart are restored; relative VGPR reads keep parked lanes | `d138351` (fork commit `90228a7`) |

Why: a PIX capture of one cutscene frame showed that `IrradianceCache_Shade` and `IrradianceCache_ShadeSecondary` did not run on batch 5 (0 dispatch). Their two shaders did not translate ("a wave reduction where not every lane runs"), so the irradiance cache was never lit and everything outside direct light stayed black (shaded side of faces, interiors). With #11 both shaders translate and pass DXIL validation, and their pipelines are created and bound in game.

Measured or observed by the maintainer (Windows 11, RTX 4070 Super), with the launcher settings of `main` (`tools/lancer_victory.ps1`), from the VIctory log and a PIX capture (trial of 25 November, about 00:01 to 00:05):

- Batch 5: PIX capture, 0 dispatch for both irradiance cache passes. Measured.
- Batch 8: both shaders translate and validate, pipelines created and bound in game. Observed.
- Untranslated shaders: 6 instead of 9. Pipelines refused by the driver: 16, as on batch 5. Measured.
- 4.8 to 5.4 fps, about 6,550 draws per frame, no device loss in about 4 min 20. Measured.
- Maintainer's opinion: "this version is very good".

Limits. The fork's issue #75 reports about 80 more refused pipelines on an RTX 3070: this was not seen on the 4070 Super, so expect more refusals on other cards. Ray-traced lighting is still absent (the BVH builder stays off, about 350 dispatches dropped per frame). With the `LATE_WAIT` setting of the launcher, some walls of the first room can be transparent (fork issue #168, fixes in progress: #191, #206). Long-run stability is measured over only about 4 min 20.

**Updating an existing copy.** The submodule URL changed with #178 (seen on 24 November). In a copy of AnyXboxX that already has the submodule, run `git submodule sync third_party/victory` then `git submodule update --init third_party/victory`. Authors are credited in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Previous pin: batch 5, fork plus seven pull requests (24 November, night)

`third_party/victory` pointed to the branch `anyxboxx/fork-7b0-lot5` of the maintainer's VIctory fork [LooperSalty/victory](https://github.com/LooperSalty/victory), at commit `fd021bf26a7003749d668e5d9bf6b234bfab2e47`. That branch assembles the fork [fxceless/victory-fork](https://github.com/fxceless/victory-fork) (GPL-3.0, formerly `fxceless/victory-gta6`) at `7b0f736` and seven of its pull requests that were still open, cherry-picked with `-x`:

| Fork pull request | Author | What it does | Commit |
| --- | --- | --- | --- |
| #63 | arminkhaleghirod | Textures that changed lately are hashed at every check (fixes the startup flicker) | `d9770b9` |
| #71 | MartinIndra02 | D3D12 COPY queue for the relay card-copy fetches, parallel `CARD_COMPARE` | `5e758c7` |
| #32 | arminkhaleghirod | 16-bit float RGBA stands in for R9G9B9E5 render targets | `d214807`, `0e2c46a` |
| #82 | Kaivyyy | Depth bounds test passed through to host pipelines and command lists | `35cdf6f` |
| #83 | ryotdev | Keep empty-EXEC branches that can reactivate lanes | `194c736` |
| #81 | undeadhuman37 | A descriptor only read as data of eight dwords is a measured image | `6119c21` |
| #77 | m335671 | `v_fma_mixlo/mixhi` over an f16 first interpolation half keeps zero | `fd021bf` |

Measured or observed by the maintainer (Windows 11, RTX 4070 Super), from the VIctory log `victory-fork-lot5.log`, with the launcher settings of `main` (PR #172: no DRED, loop limit 256, `VICTORY_LATE_WAIT=3`, `VICTORY_WIDE_BACK=1`, card buffers) and **without** `VICTORY_RELAY`:

- Startup: no flicker. Observed.
- World after 2 min: 5.4 fps and 6,449 draws per frame (reference `main`: about 5.5 fps, about 4,410 draws; batch 4 with `VICTORY_RELAY=2`: 6.6 fps, 5,986 draws); 363 draws lost without a pipeline; 9 untranslated shaders (13 in batch 4); 16 pipelines refused by the driver (6 in batch 4); no device loss in about 2 min 30. Measured.
- Maintainer's opinion: "much better", the best lighting and graphics so far. Graphic problems remain on Lucia's face.

`VICTORY_RELAY` is no longer set by the launcher: the run judged best had it off, and the fork's issue #89 reports that it makes the characters' skin disappear and overlaps the subtitles. Known limits: ray tracing absent, shaders still untranslated, long-run stability not measured (our run lasted about 2 min 30). Progress of 24 November, from the VIctory logs: `d279e69` about 4 fps (13 min in the prison, then device lost) → `7b0f736` + `VICTORY_RELAY=2` about 4.5 fps → settings of PR #172 about 5.5 fps → batch 4 about 6.6 fps with `VICTORY_RELAY=2`, startup flicker fixed → this branch about 5.4 fps with more draws and the best image so far. Authors are credited in [THIRD_PARTY.md](../THIRD_PARTY.md).

## Previous pin: fork `7b0f736` (24 November, night)

`third_party/victory` points to `fxceless/victory-fork` at commit `7b0f736` ("Shader tests: recognise the software adapter from the device's own adapter"). It was first pinned at `d279e69` ("Make host pipeline waits configurable and prewarm pending requests (#1)", contributed by ZazaJr24); `7b0f736` adds `e9765a7` ("Fix the device loss introduced by the wave64 update") and `004bf69`. Newer commits, from `1684989` on, are not taken: the maintainer judged them a regression. The fork is not a GitHub fork attached to VIctory; its licence is GPL-3.0, compatible with AnyXboxX (GPL-3.0 or later). It replaces our former `anyxboxx/stable` integration.

Measured by the maintainer on 2026-11-24 (i7-13700KF + RTX 4070 SUPER, Windows 11) by running our own copy of the 1.0.2.1 build, from the VIctory log of the run (frame rate and draw counters as the program prints them), against our former `last` integration:

| | fork `d279e69` (first pin) | former `last` integration |
| --- | --- | --- |
| Frame rate in the world | 4.0 to 4.3 fps | 2.3 fps |
| Draws per frame | about 4,580 | about 2,860 |
| Draws lost without a pipeline | 428 | 400 |
| Indirect draws lost | 3 | 86 |
| Device lost | none for about 13 minutes, then lost (see known limits) | not recorded |

The world is reached and displayed. Nothing here comes from reading the game's code.

**Observed by the maintainer (Windows 11, RTX 4070 Super).** With the `fxceless/victory-fork` fork at commit `d279e69` and the launcher settings, the game was played at about 4 fps for about 13 minutes in the starting area (the prison): the character moves under the player's control and interacts with the first NPCs, then the host device is lost (known limit).

**Measured on `7b0f736` (maintainer, same machine, from the VIctory log of the run `victory-fork-7b0-relay.log`).** With `VICTORY_RELAY=2` and the launcher settings, pipeline cache already filled: world reached in about 1 min 25 s; 3.5 fps on the first frame of the world, then 4.5 fps after 2 min, about 4,975 draws per frame, 346 draws lost without a host pipeline; no device loss during about 2 min 40 s in the world, after which the maintainer closed the window. An earlier run of `7b0f736` without `VICTORY_RELAY`: 5.4 fps at about 3 min, no loss, window closed. The maintainer's opinion: "slightly better" than `d279e69`. Reference `d279e69`: 4.0 to 4.3 fps, about 4,580 draws, about 430 lost.

**Known limits (stability).** The device loss after about 13 minutes was observed on `d279e69`; the rest of this paragraph describes that run. `7b0f736` fixes a device loss tied to wave64; its long-run stability is not yet measured by us (our longest run on it is about 3 minutes). The fork's issue #26 reports 21.5 minutes stable on an RTX 4060 Ti with `VICTORY_RELAY=2` (third-party report, not reproduced here). The second run on `d279e69` held about 13 minutes in the world (first draws at 19:07:27 UTC), then lost the GPU device at 19:20:35 UTC: "host GPU device removed after 15 submissions sent ahead (VICTORY_LATE_WAIT=1) of 82 lists", the last list being a compute shader (`c1488f485ad606fb`). Measured, from the VIctory log of the run; the cause is not known. Plan on a few minutes to about 13 minutes in the world before a device loss, and relaunch. The message names `VICTORY_LATE_WAIT=1` while the launcher sets `3`: the setting actually in force in that run is to be checked. Checked since (read statically in `crates/victory-gpu/src/host.rs`, `late()`, same code at `d279e69` and `7b0f736`): `3` was in force. Stage 3 includes stages 1 and 2, and the `(VICTORY_LATE_WAIT=1)` and `(VICTORY_LATE_WAIT=2)` tags in log lines name the stage a counter belongs to, not the value set.

**First cold launch.** The first launch with an empty cache crashed at loading (pipeline compilation, then the game's internal crash tracked as VIctory #1464). The second launch, with the cache filled, held. If the first launch crashes, launch again.

The settings of the successful run are now the defaults of `tools/lancer_victory.ps1` (table in the Run section), except five variables that the pinned source never reads and DRED, which is now opt-in (see Run). They need the `diag` build (`--profile diag --features diagnostics`). Pass `-NoTunedSettings` to turn them off.

## What you get today

Measured on 2026-11-24 on an i7-13700KF + RTX 4070 SUPER, Windows 11, with the stable integration:

- Boot, profile screen, first-run settings, legal screens, loading and the pause menu work, at 30 fps in the menus.
- The first cutscene (opening credits of mission 1) is reached.
- The 3D scene behind the credits is black.
- In the world the game runs at 0.3 to 0.6 frames per second.
- It is not playable.

The section below describes a more advanced state, measured the same afternoon on a local integration branch that is not yet what `third_party/victory` pins.

## State on 24 November (afternoon)

Measured by running our own copy of the 1.0.2.1 build under VIctory, on a local integration branch (VIctory `main` + about 40 pull requests + local fixes not yet published), i7-13700KF + RTX 4070 SUPER. Nothing here comes from reading the game's code: these are run measurements (fps, timings, counts) and the output of `victory translate` on the shaders of our own copy.

- **World reached, character controllable (observed).** The tutorial "Move / Move the camera" (L / R) is displayed. The interface is composed over a 3D image that is still black.
- **Speed in the world (measured): 0.2 fps, then 2.3 to 2.5 fps.** Causes and fixes:
  - `VICTORY_D3D12_DRED=1` cost about 2.5x (text formatting and a `VirtualQuery` per draw). Only turn it on to investigate a device removed.
  - The copy back to the title spent 66% of the submission thread in `WriteProcessMemory`, itself in `NtQueryVirtualMemory` (about 2 ms per call). A direct `NtWriteVirtualMemory` brings it from 1.5 s to 0.13 s per frame.
  - Copies no longer call `VirtualQuery` first; the texture snapshot, comparison and hashing are spread over a thread pool.
  - What remains per frame (about 410 ms): GPU about 145 ms, copy back about 73 ms, hashing 20 to 60 ms, snapshot about 50 ms.
- **Untranslated shaders in game: 456 in the morning, about 170 in the afternoon (measured by the in-game count).** On the game's corpus (26,084 programs, `victory translate`) the emitted programs are: VS/PS 13,480 of 14,038, CS 2,212, MS 8,654 of 8,765 (counts as printed by the tool). Patterns solved during the day: descriptor read as data, scalar memory read as data, buffer of unknown stride, wave prefix in divergent flow (the InstanceCulling passes), mask kept in a VGPR lane, `readfirstlane` of a uniform address (ColorDownsample). What remains is mostly lane masks: "lane mask read as data" (586 programs) and "holding lanes elsewhere" (282) on the corpus.
- **Ray tracing.** The title's BVH builder causes a TDR (GPU timeout recovery); it is skipped with `VICTORY_SKIP_BVH_BUILDER=1`. The new local diagnostic option `VICTORY_RT_MISS=1` makes every ray miss: 36 of 47 ray-tracing containers are emitted instead of 18. A Codex analysis found that `ConstructBVH4` contains a bounded polling loop (256 iterations) on a value another pass produces.
- **Final composition.** As in VIctory #841, the scene exists at 2560x1440 but the presented target stays zero.

### Useful variables

All are diagnostics: they need the `diag` build (the `diagnostics` feature). Provenance, checked in the code of the integration worktree and against VIctory `main` on 2026-11-24:

| Variable | Effect | Where it comes from |
| --- | --- | --- |
| `VICTORY_KEYS` | Scripted key presses (see below). | VIctory `main` |
| `VICTORY_SHADER_LOOP_LIMIT=32` | Translated loops stop after that many iterations (decimal or `0x` hex, at least 1). | VIctory `main` |
| `VICTORY_D3D12_DRED=1` | Device-removed data. **Warning: any value turns it on, and it slows the world by about 2.5x; use it only to investigate a device removed.** | VIctory `main` |
| `VICTORY_SHOW_TARGET=busiest` | Shows the busiest render target instead of the final image. | Open pull request #638 (`Marvisak/VIctory`), in our integration |
| `VICTORY_CAPTURE_BUSY=<draws>[,<count>]` | Captures frames busier than that many draws (2 frames by default). | Open pull request #638 |
| `VICTORY_PRODUCERS=1` | Logs a pass that reads what only dropped (untranslated or skipped) work writes, with the dropped shader. | Open pull request #638 |
| `VICTORY_CAPPED_DYNAMIC=262144` | More dynamic descriptor slots on a capped heap (NVIDIA), at the cost of the title's highest slots. | Local experiment, published only on our fork (`anyxboxx/essai`, `anyxboxx/stable`) |
| `VICTORY_SKIP_BVH_BUILDER=1` | Leaves out the shaders of the title's BVH builder. | Our fork only (`anyxboxx/essai`), not proposed upstream |
| `VICTORY_PROFILE=<ms>` | Sampling profiler of the whole process; logs the busiest call stacks every 5 s (sampling period in ms, 4 by default). | Local, not published |
| `VICTORY_RT_MISS=1` | Every ray misses (ray-tracing diagnostic, off by default). | Local, not published |

## State on 24 November (evening)

Measured by running our own copy of the 1.0.2.1 build under VIctory, on the local integration branch (VIctory `main` + pull requests + local fixes not all published), i7-13700KF + RTX 4070 SUPER. These are run measurements (fps, counts, traces) and the output of `victory translate`; nothing comes from reading the game's code.

- **Series S mode is faster (measured).** `victory run --console series-s ...` gives about 3.1 to 3.3 fps in the world, against about 2.0 to 2.4 fps in Series X mode (about +50%), with the same image symptom. The game then renders internally at 1280x720 and presents at 1920x1080. Recommended for iterating.
- **Black image diagnosis (measured).** The geometry exists in the G-buffer: the characters of the cutscene are visible in 1440p and 720p. The final colour is a uniform tint that darkens over time (exposure), with the subtitles drawn on top. Almost all deferred lighting passes are translated and bound. A per-pass trace (local option `VICTORY_INDIRECT_TRACE`, not published) shows that about 379 mesh `ExecuteIndirect` calls start with zero arguments, because the GPU passes that produce those arguments are not translated yet: the translator refuses them with "an address from a phi" and "a lane mask holding lanes elsewhere read as data". This is the current work item.
- **Translation (measured).** Untranslated shaders in game: about 130 to 170 (in-game count). On the corpus (26,084 programs, `victory translate`): VS/PS 13,780 emitted of 14,038, CS 2,212, MS 8,654 of 8,765 (counts as printed by the tool).
- **Fixes published upstream** (`Marvisak/VIctory`): #1215 (speed: direct `NtWriteVirtualMemory` write, copies without `VirtualQuery`, profiler), #1225 (`readfirstlane` of a uniform address: ColorDownsample), #1231 (uniform loop counter: light probes). Measurement comments were also added on #497, #320, #1074, #1198 and #841.
- **Community.** The project has a Discord server (link at the top of the README).

Variable added to the table above (diagnostic, `diag` build):

| Variable | Effect | Where it comes from |
| --- | --- | --- |
| `VICTORY_INDIRECT_TRACE` | Per-pass trace of `ExecuteIndirect` calls and their arguments. | Local, not published |

## Requirements

- Windows 10 or 11, x64.
- A CPU with AVX2. Intel is fine: the instructions specific to AMD are emulated.
- A Direct3D 12 GPU with shader model 6.6 and mesh shaders (NVIDIA RTX 20 series or newer, AMD RDNA 2 or newer).
- 32 GB of RAM recommended, and a large enough Windows paging file: the game commits several GiB at startup. If it does not, `crashes.log` reports Windows error 1455: close other applications or enlarge the paging file.
- Your own copy of the Xbox Series 1.0.2.1 build: a folder that contains `game_scarlett_master.exe`. Not provided, not explained here.
- Free disk space for the shader cache (it grows as you play; see the status page for the figures to plan for).

## Install

1. [Git](https://git-scm.com/).
2. Rust through [rustup](https://rustup.rs/). The exact toolchain is pinned by the submodule's `rust-toolchain.toml` and is installed automatically on the first `cargo` command.
3. Visual Studio 2022 or 2026 Build Tools with the "Desktop development with C++" workload (MSVC).
4. Windows SDK 10.0.26100 (provides `dxc`).
5. Optional: the Windows optional feature "Graphics Tools" (D3D12 debug layer, needed only for `VICTORY_D3D12_DEBUG`).
6. Python 3.12, only for the AnyXboxX test suite (see Tests).

Get the code with its submodule:

```powershell
git clone --recurse-submodules https://github.com/LooperSalty/AnyXboxX.git
cd AnyXboxX
# or, in an existing clone:
git submodule update --init third_party/victory
```

## Build

```powershell
cargo build --profile diag -p victory --features diagnostics --manifest-path third_party/victory/Cargo.toml
```

The `diag` profile is required for the scripted keys, the frame captures and the diagnostic options below. The first build downloads and compiles all dependencies, so expect several minutes. The executable is `third_party\victory\target\diag\victory.exe` (or `%CARGO_TARGET_DIR%\diag\victory.exe` if you set that variable). `tools/lancer_victory.ps1` runs this build for you when the executable is missing.

## Run

```powershell
.\tools\lancer_victory.ps1 -GameDir <game folder>
```

The script sets the variables below (unless you pass `-NoTunedSettings`, which keeps only `VICTORY_GAME_DIR`) and then runs `victory.exe run <game folder>\game_scarlett_master.exe --execute`:

| Variable | Value | Why |
| --- | --- | --- |
| `VICTORY_GAME_DIR` | your game folder | No `victory.toml` is needed. |
| `VICTORY_SHADER_LOOP_LIMIT` | `256` | Translated shader loops were almost endless, so they are cut. At 32 (the former value) hair is drawn as glowing strands and the scene is darker; at 1024 the device was lost on an RTX 3060 Ti with the default 2-second timeout; 256 held (fork issue #18, and observed on Linux in runs 3 and 4 below). |
| `VICTORY_LATE_WAIT` | `3` | Host pipeline wait setting of the fork's pull request #1. |
| `VICTORY_WIDE_BACK` | `1` | Fork option, on in the measured run. |
| `VICTORY_RELAY` | not set | Fork option, no longer set by default (fork issue #89: skin disappears, subtitles overlap); `2` was on in the measured runs of `7b0f736` and batch 4. |
| `VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_CARD_BUFFERS_MB` | `1`, `1024` | Card buffer sizing, in MB. |

The former defaults `VICTORY_RT_MISS`, `VICTORY_WAVE32_READFIRSTLANE`, `VICTORY_WAVE32_CROSSLANE`, `VICTORY_SKIP_BVH_BUILDER` and `VICTORY_CAPPED_DYNAMIC` are gone: no file of the pinned commit reads them (the BVH builder is off unless `VICTORY_BVH_BUILDER=1`). `tests/test_lanceur_victory.py` now checks that VIctory reads every variable the launcher sets.

`-Dred` sets `VICTORY_D3D12_DRED=1` (device-removed data) to diagnose a device loss. It is off by default: VIctory turns it on whenever the variable exists, and it notes every command list operation, which slows each frame.

The first cold launch may crash while the pipelines compile: launch again once the cache is filled.

`-SkipPipelines <hash,...>` is passed to `VICTORY_SKIP_PIPELINES` (see below).

Where things are kept:

- Saves, settings and `crashes.log`: `%LOCALAPPDATA%\victory\data`.
- Derived data (shader and pipeline caches): `%LOCALAPPDATA%\victory\cache`.

Both are outside the repository on purpose. Delete the cache folder to start from a clean cache; keep the data folder if you want to keep your save.

## Getting to the first cutscene

The window caption shows the frame rate. The window must have the focus.

1. Profile screen: press Enter.
2. First-run settings: press Enter on each screen.
3. Legal notices: move down to "Next", wait for the countdown of the photosensitivity warning to end, tick the two boxes, then "Confirm".
4. Loading, then the first cutscene (opening credits).
5. Escape opens the pause menu.

Keyboard or pad. With no XInput pad connected, the keyboard acts as a pad while the title's window has focus:

| Keys | Pad |
| --- | --- |
| Enter, Backspace, X, Y | A, B, X, Y |
| Arrows | D-pad |
| W A S D / I J K L | left stick / right stick |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Escape | Menu |
| Tab | View |

Hold each key for a moment: the game reads its input at 30 Hz, so a very short tap can be missed. Later launches resume directly from the save.

## Scripted runs and diagnostics

These need the `diag` build.

| Variable | Effect |
| --- | --- |
| `VICTORY_KEYS=40:Enter,55:Enter,70:Enter` | Presses the keys at those seconds, with or without focus. |
| `VICTORY_CAPTURE=<folder>` and `VICTORY_CAPTURE_FRAMES=<list>` | Saves the presented frames (those frame numbers). |
| `VICTORY_SHOW_TARGET=busiest` | Shows the busiest render target instead of the final image (for example the G-buffer). |
| `VICTORY_D3D12_DRED=1` | On a GPU failure, reports the commands in flight. It slows the world by about 2.5x: turn it on only to investigate a device removed. |
| `VICTORY_D3D12_DEBUG=1` | D3D12 debug layer (Graphics Tools required). |
| `VICTORY_GPU_TIMING=1` | Times each command list and logs the slow ones. |
| `VICTORY_SKIP_PIPELINES=<hash,...>` | Omits the draws and dispatches that use the listed 16-digit shader hashes. |
| `VICTORY_LOG=<target>=debug,...` | Debug and trace output. |

To stop the game: `taskkill /F /IM victory.exe`. The `timeout` command of Git Bash does not kill the executable.

## Former branches

The branches `anyxboxx/stable` and `anyxboxx/essai` of `LooperSalty/victory` are no longer the pin: the fork above replaces them. The sections above that mention them describe earlier measurements.

## Safety

- On NVIDIA, staying in the world for a long time can reset the display driver (every monitor goes black for a moment), and one tester got a blue screen (0x119). Stop the game after a few minutes in the world.
- Run one instance at a time.

## Tests

```powershell
cargo test -p victory-gpu -p victory-shader --manifest-path third_party/victory/Cargo.toml
```

AnyXboxX itself (see the Setup section of [AGENTS.md](../AGENTS.md)):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check --select W605,F,E9 PrototypeScarlett tests tools
```

## Where to report

Open an issue on [AnyXboxX](https://github.com/LooperSalty/AnyXboxX/issues) or on [VIctory](https://github.com/Marvisak/VIctory/issues) (issue #264 collects the NVIDIA GPU hangs). Describe the symptom, your GPU and driver, and the relevant lines of `crashes.log`. Never attach a game file.
