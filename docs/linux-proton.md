# Building and running on Linux (Proton / Wine)

> **Linux only.** Windows is the primary target of this repository. Nothing
> here changes Windows: the switches below are environment variables that are
> off by default, and the rest is about how to set up Proton.

Tested on CachyOS (kernel 7.3), RX 7800 XT (RADV, NAVI32), GE-Proton11-14.

## Cross-compiling on Linux

Install the toolchain pinned in `rust-toolchain.toml` with the MSVC target and
[`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) (needs `clang`, `lld`):

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
```

The BCPack tables are not in the repository: copy
`crates/victory-formats/src/bcpack-tables.bin` from an upstream checkout
(`Initialize-FormatTables.ps1` lists the URL and SHA-256), then build:

```sh
cargo xwin build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
```

## What Proton needs

| What | Why |
|---|---|
| GE-Proton 11-13 or newer | older Proton Experimental builds hung in loading |
| `VKD3D_CONFIG=descriptor_heap` | the title's view heap is larger than vkd3d-proton's default limit |
| Microsoft `d3dcompiler_47.dll` in the prefix's `system32`, `WINEDLLOVERRIDES=d3dcompiler_47=n,b` | Wine's builtin one leaves the world black |
| A working audio device (PulseAudio / PipeWire socket) | without one, WASAPI never pumps and the story's loading screen waits forever |
| `ulimit -n 524288` | the title opens many files |
| `VKD3D_SHADER_CACHE_PATH` on a writable path | vkd3d-proton otherwise tries to write next to the game |
| `MESA_VK_WSI_PRESENT_MODE=fifo` | keeps 30 fps video and menu presentation in order on RADV / Xwayland |

Use X11 (Xwayland): with `PROTON_ENABLE_WAYLAND=1` `victory.exe` hung before
`main()` here.

## Opt-in switches (Linux only, no effect unless set)

```sh
export VICTORY_LOG_FILE='Z:\path\to\victory.log'  # Proton's steam.exe swallows stderr
export VICTORY_SKIP_TITLE_INT3=1                  # step over the title's non-fatal assert int3 (else it exits ~14 s in)
```

## Launch

```sh
export VICTORY_GAME_DIR='Z:\path\to\game'
export VICTORY_CACHE_DIR='Z:\path\to\victory-cache'
export VICTORY_DATA_DIR='Z:\path\to\victory-data'
"$PROTON_DIR/proton" run victory.exe run 'Z:\path\to\game\<title.exe>' --execute --console series-x
```

The first run with an empty shader cache can stall for minutes while
pipelines are made, and the title's own crash reporter may end it. Start it
again: the cache it left gets it through.

## Sandboxing with bubblewrap

If Proton runs inside `bwrap`, put `--dev /dev` **before**
`--dev-bind /dev/dri /dev/dri` (a later `--dev` hides `/dev/dri` and Vulkan
finds no GPU), and bind the audio sockets (`$XDG_RUNTIME_DIR/pulse/native`,
`$XDG_RUNTIME_DIR/pipewire-0`) with `XDG_RUNTIME_DIR` and `PULSE_SERVER` set.
