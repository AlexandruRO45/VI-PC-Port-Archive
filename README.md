# victorymini: VIctory on a laptop with an 8 GB NVIDIA GPU

A fork of [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)
for laptops and 8 GB NVIDIA cards (32-lane waves): the Series S profile by default
with `-Light`, 393 compute programs NVIDIA refused now running, DualShock 4 pads
without XInput. Measured on one machine: RTX PRO 2000 Blackwell laptop GPU (8 GB),
Intel Core Ultra 9 285H, 31 GB RAM, Windows 11. World frame rate in the opening
cutscene went from 1.4 to about 3 fps (4.3 at best); not playable yet. Details and
what is still wrong: [On a laptop or an 8 GB NVIDIA card](#on-a-laptop-or-an-8-gb-nvidia-card).
The upstream fork's own description follows.

---

# VIctory on an RX 7900 XTX

A fork of [VIctory](https://github.com/Marvisak/VIctory), a compatibility layer
that runs an Xbox Series title natively on Windows from files the user supplies.
This one is about **speed on AMD RDNA 3**: the world at 30 frames a second on
one PC, without giving up the picture. It is not there yet.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## Where it stands (2026-11-25)

Tested on one machine only: Windows 11, Intel i9-12900K, 64 GB RAM, Radeon
RX 7900 XTX, 1440p, the Series X profile.

| | |
|---|---|
| Menus, videos, loading screens | 30 fps |
| Prologue, indoors | 50-75 ms a frame (13-20 fps) |
| First mission with the boat | 46-53 ms (19-22 fps) |
| Street chase with many cars and people | 100-165 ms (6-10 fps), single frames far longer |
| Outdoors on foot, the camp | 60-90 ms |
| Flying | 42-57 ms |
| Longest session so far | 41 minutes, no crash |
| Memory | 40-46 GB of private memory: **64 GB of RAM recommended** |

It started at 280 ms a frame in the same scenes two days earlier.

What is known to be wrong:

- Trees, plants and far buildings flicker outdoors in some places.
- Rooms with little direct light are too dark without `-RayTracing`.
- Textures of what just came into view appear late while the title streams.
- The first run with a new driver or a new build stutters while the driver makes
  pipelines (`victory warm` makes them ahead).
- Recording a frame in the world makes the title's own watchdog end it.

## How to run it

You need Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with shader
model 6.6 and mesh shaders, Rust (the toolchain is pinned), the MSVC build
tools, and your own dump of the title.

```powershell
git clone https://github.com/avatarmanz/VIctory-7900XTX
cd VIctory-7900XTX
copy victory.example.toml victory.toml      # then set your paths, outside this folder
cargo build --locked --profile diag -p victory --features diagnostics
scripts\play.ps1 -Exe <path of your copy's executable>
```

`scripts\play.ps1 -Safe` leaves out the newest, faster ways; try it first when
something is wrong. `-RayTracing` turns the title's ray-traced indirect light on.

### On a laptop or an 8 GB NVIDIA card

`scripts\play.ps1 -Exe <...> -Light` runs the Series S profile at a 1080p display
mode. Measured on an RTX PRO 2000 Blackwell laptop GPU (8 GB), 31 GB RAM, in the
opening cutscene (world frames over about 4 minutes):

| | world fps |
|---|---|
| Series X profile, 1080p | 1.4 |
| Series X profile, 720p | 1.2 |
| Series S profile | 1.9 |
| Series S, wave64 programs with LDS in lane branches run in waves of 32 | 2.2 |
| the same, key presses only until the world starts (same shots every run) | 3.0 (4.3 at best) |

- On GPUs with 32-lane waves (NVIDIA) 404 compute programs (culling, shadows,
  lighting, TAA) kept 64-lane waves and the driver refused them; 393 of them now run
  (`VICTORY_LDS_REGION_RACE`, on by default there; `0` turns it off).
- Going over the card's memory budget is far worse than textures waiting for room:
  a smaller `VICTORY_CARD_ROOM_FOR_TARGETS_MB` took the card to 10.5 GB of 7.1 GB and
  0.6 fps. Leave the defaults.
- Set Windows' power mode to *Best performance* and NVIDIA's *Power management mode*
  to *Prefer maximum performance*: in *Best power efficiency* the card stayed in P4
  (memory at 9 of 12 GHz, 33 of 56 W).
- DualShock 4 pads (and pads that report themselves as one) work without XInput.
- Still wrong: a dark rectangle over the opening skyline shot in the Series S
  profile; some faces black in dark scenes; textures of what comes into view late;
  11 compute programs (water volumetrics, three of the console upscaler's) still do
  not translate in waves of 32. A lower `VICTORY_SHADER_LOOP_LIMIT` (1024) and
  `VICTORY_PAGE_WRITES=1` did not help here.

Before the first run, ahead of time (both can be stopped and go on later):

```powershell
target\diag\victory.exe shaders <the shader archives>   # every shader translated, minutes
target\diag\victory.exe warm --used                     # pipelines earlier runs drew with, for a new driver or build
```

Commands, debugging switches and upstream's own description:
[docs/README-fxceless.md](docs/README-fxceless.md). Every `VICTORY_...` switch is
described in a comment where it is read.

## What is different here

In short; the reasons and the measurements are in [docs/findings.md](docs/findings.md).

- **List calls are written down, not carried out, on the title's threads**
  (`VICTORY_THIN`): eight threads of ours translate them.
- **Submissions are prepared by a thread of ours and joined**
  (`VICTORY_ASYNC_SUBMIT`, `VICTORY_JOIN_PARTS`), with what can be spread over
  threads spread: refreshes, compares (AVX2), descriptor records, landings.
- **What shaders wrote comes back beside the write guards**, by the dwords that
  changed, on a thread of its own (`VICTORY_RELAY`, `VICTORY_LAND`).
- **Buffer fills that need bytes still on the card are done on the card**
  (`VICTORY_CARD_STEPS`) instead of waiting for it.
- **The title's writes are caught more cheaply**: one view opened at a fault,
  blocks that rest, places in the title's code that look before they write.
- **Textures go up on a copy queue of their own** and a slot shows one only after
  its upload ran, which also ended a device loss that came every 15 minutes or so.
- **Boats float**: textures that shaders write and the title's CPU reads go back
  to title memory (from fxceless and Geckomacabre, on by default here).
- **Frame record and replay** with per-pass times and shader variants compared
  turn by turn ([docs/replay.md](docs/replay.md)).

## Credits

This is a tree of other people's work with more on top. The history is kept, and
commits taken from elsewhere name the commit and its author.

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors: the loader, the runtime services, the Direct3D 12 back end and
  the shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**: the fork
  this one started from, and the people whose pull requests it carries.
- **[DexNom/VIctory](https://github.com/DexNom/VIctory)**: ray tracing fixes,
  cube gradients, sparse textures.
- **[Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD)**:
  the ambient light idea, pipeline workers, and work by romainhedouin.
- **[FQY55kai/ZJOU-VICTORY](https://github.com/FQY55kai/ZJOU-VICTORY)**.
- **[myokex/victory-amd](https://github.com/myokex/victory-amd)**: the guard over
  every view of a page (with Spexivus, joroyt and tightfall), and Joshua Odom's
  reads into title memory.
- **Geckomacabre** and **cxy0072019-sketch**: the texture read-back.
- **[XCL](https://github.com/ivall/xcl)** (GPL-2.0-only) was read for how the same
  title behaves under another layer; no code of it is here.
- This fork's changes were made with [Claude Code](https://claude.com/claude-code).

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE); the one file that is not
covered is named in [docs/asset-policy.md](docs/asset-policy.md).
