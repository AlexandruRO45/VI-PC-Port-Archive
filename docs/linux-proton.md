# Linux and Proton

VIctory is a Windows program. On Linux you can run the host tests, cross-build
`victory.exe`, and run that exe under Wine/Proton, where D3D12 goes through
vkd3d-proton. None of this is official support, and the title is **not
playable** on Linux any more than on Windows (see the status in the README).

You supply your own files. Keep them, and everything built or derived from
them, outside the repository: see [asset-policy.md](asset-policy.md). Never
commit game data, built binaries or logs that contain title bytes.

See also open PR [#195](https://github.com/Marvisak/VIctory/pull/195)
(Linux cross-build and Wine host stubs) and
[#191](https://github.com/Marvisak/VIctory/pull/191) (a Linux launcher and
smoke test for Proton). Once #195 is merged it is the reference for building
on Linux; this page keeps the Proton/umu run steps.

## Host tests

`main` does not build on Linux yet: `victory-gpu` stops at `error[E0425]:
cannot find function 'host_resource'` (tracked in
[#90](https://github.com/Marvisak/VIctory/issues/90)). Until
[#39](https://github.com/Marvisak/VIctory/pull/39) or
[#112](https://github.com/Marvisak/VIctory/pull/112) is merged, test on
one of those branches. Both conflict with current `main`, so they do not
merge onto it cleanly.

```sh
gh pr checkout 112              # or 39
cargo test --workspace          # needs no game data
cargo xtask install-hooks       # asset-policy check on every commit
```

## Tools on Linux vs running the title

The `victory` subcommands that only read files (`inspect`, `census`,
`lib-exports`, `shader-info`, `translate`, `config`) do CPU work and need no
Windows API, so they can build as a native Linux ELF once the workspace
builds on Linux (the E0425 fix and Linux gates in
[#112](https://github.com/Marvisak/VIctory/pull/112), tracking
[#90](https://github.com/Marvisak/VIctory/issues/90)). In a local test of
that approach, not merged anywhere, the workspace tests passed on Linux and
`translate` ran on the host without Wine.

`victory run` is different: running the title needs D3D12 and the rest of
the Windows runtime, so it stays the cross-built `victory.exe` under
Proton, as described below. A Linux build of the tools does not change that.

## Cross-building victory.exe

Needs the pinned toolchain with the `x86_64-pc-windows-msvc` target, LLVM
(`lld-link`, `clang-cl`, `llvm-lib`) and the Windows CRT and SDK from
[xwin](https://github.com/Jake-Shadle/xwin).

```sh
rustup target add x86_64-pc-windows-msvc
xwin --accept-license splat --output ~/.cache/xwin

X=$HOME/.cache/xwin
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-Lnative=$X/crt/lib/x86_64 -Lnative=$X/sdk/lib/um/x86_64 -Lnative=$X/sdk/lib/ucrt/x86_64"
export CC_x86_64_pc_windows_msvc=clang-cl
export CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-lib
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc -Wno-unused-command-line-argument -fuse-ld=lld-link /imsvc$X/crt/include /imsvc$X/sdk/include/ucrt /imsvc$X/sdk/include/um /imsvc$X/sdk/include/shared"
export CXXFLAGS_x86_64_pc_windows_msvc="$CFLAGS_x86_64_pc_windows_msvc"

cargo build --release -p victory --target x86_64-pc-windows-msvc
# target/x86_64-pc-windows-msvc/release/victory.exe

cargo build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
# target/x86_64-pc-windows-msvc/diag/victory.exe
```

The exe imports `VCRUNTIME140.dll`, the UCRT, `d3d12.dll`, `dxgi.dll`,
`d3dcompiler_47.dll` and `xinput1_4.dll`. Proton provides all of them; on a
plain Wine prefix without vcruntime140, run `winetricks vcrun2022`.

## Running under Proton

Tried with UMU-Proton via [umu-run](https://github.com/Open-Wine-Components/umu-launcher).
Paths given to `victory.exe` are Windows paths, so use `Z:` plus the Linux
path. The `VICTORY_*` variables override `victory.toml`, as on Windows.

```sh
export WINEPREFIX=$HOME/.local/share/victory/pfx
export GAMEID=umu-victory STORE=none
# PROTONPATH=<Proton dir>           # optional; umu-run uses UMU-Proton when unset

z() { printf 'Z:%s' "${1//\//\\}"; }
export VICTORY_GAME_DIR=$(z "$HOME/victory-data/game")
export VICTORY_DATA_DIR=$(z "$HOME/victory-data/data")
export VICTORY_CACHE_DIR=$(z "$HOME/victory-data/cache")
export VICTORY_LOG=info
# vkd3d-proton writes vkd3d-proton.cache into the working directory by
# default; keep it outside the checkout or `cargo xtask policy` flags it.
export VKD3D_SHADER_CACHE_PATH=$HOME/.local/share/victory/vkd3d-cache
mkdir -p "$VKD3D_SHADER_CACHE_PATH"

# From the repository root: the cargo output from the build step above
# (use .../diag/victory.exe for the diagnostics build).
VICTORY_BIN=$PWD/target/x86_64-pc-windows-msvc/release/victory.exe
# Host logs go outside the checkout, one folder per run.
LOGS=$HOME/.local/share/victory/logs/$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGS"

umu-run cmd.exe /c "$(z "$VICTORY_BIN") run $VICTORY_GAME_DIR\<the title's exe> --execute > $(z "$LOGS/host.out.log") 2> $(z "$LOGS/host.err.log")"
```

Proton does not pass the Windows console through, hence `cmd.exe /c` with
redirections to keep the log. Those logs can contain title-derived data, so
they live under `$LOGS`, not in the repository. If you copy `victory.exe`
somewhere else, point `VICTORY_BIN` at that copy, also outside the
repository. `crashes.log` is in the data folder as usual.

`VKD3D_SHADER_CACHE_PATH` must be a directory outside the repository. If a
`vkd3d-proton.cache` file ever lands in the checkout, delete it rather than
committing it; `cargo xtask policy` rejects it.

Seen on one machine (NVIDIA RTX 2060 SUPER, UMU-Proton 11.0, release build):
imports bind, the D3D12 device comes up on vkd3d-proton, the first-boot
screens and intro video run at 30 fps, and the title starts loading the
world. Some compute pipelines are refused with `E_INVALIDARG` (NVIDIA has
32-lane waves; see the README's shader row), and the title's view heap is
larger than vkd3d-proton allows on that GPU, so it is capped.

### Indirect draws under Wine

Wine's `D3DCompile` can refuse the shader that clamps indirect-draw
arguments, so indirect execution is dropped
([#170](https://github.com/Marvisak/VIctory/issues/170)). Two open fixes
take different routes: [#178](https://github.com/Marvisak/VIctory/pull/178)
falls back to DXC, and [#190](https://github.com/Marvisak/VIctory/pull/190)
builds the clamp as hand-written DXIL with no compiler at run time. They
conflict, so only one of them should be merged; measurements under Proton
are in the comments on both PRs.

### Open Wine/Proton fixes

Still open:

- [#76](https://github.com/Marvisak/VIctory/pull/76): xmem maps a run of
  physical pages over several placeholders. It merges cleanly onto
  current `main`, so you can stack it with `gh pr checkout 76`.

Already on `main`: [#41](https://github.com/Marvisak/VIctory/pull/41)
(reject non-software WARP adapters under Proton) and
[#53](https://github.com/Marvisak/VIctory/pull/53) (commit physical-page
sections whole under Wine).

These lists go stale as PRs land. If you keep a fork, sync its `main` from
upstream `main` (taking upstream's side on any conflict) before you rebase
your own branches onto it.

## Diagnostics

`VICTORY_D3D12_WARP`, `VICTORY_KEYS`, `VICTORY_STACKS` and the other
variables in the README's Debugging table only work in the `diag` build
above; a release build ignores them. Under Proton, `VICTORY_D3D12_WARP=1`
asks for the WARP adapter, and #41 refuses an adapter that is not a software
one instead of using the real GPU. Add `WINEDEBUG=+seh,+loaddll` for Wine's
view of faults and DLL loads.
