#!/usr/bin/env bash
# VIctory under Proton: run the Windows build of victory inside a Proton
# prefix, which supplies exactly what victory hosts on — a Win32 process
# environment with real D3D12 (via dxvk/vkd3d-proton). See docs/proton.md.
#
# The executable is built here with the gnu target (cargo win); nothing in
# this script touches the game itself.
set -euo pipefail

usage() {
    cat <<'USAGE'
usage: proton/run.sh [options] [--] [victory args...]

  --prefix DIR   Proton/Wine prefix (default: $XDG_DATA_HOME/victory/proton)
  --exe FILE     victory.exe to run (default: the cargo win output under target/)
  --proton DIR   Proton directory, or a plain wine binary
                 (default: newest GE-Proton, then newest Proton-*)
  --z DIR        link DIR into the prefix's Z: drive as Z:\<basename>
                 (GE-Proton 10+ maps Z: to /, where this is a no-op)
  --config NAME  victory config to use, relative to the repo root
                 (default: victory.toml if present; see docs/proton.md)
  --d3dcompiler FILE
                 copy a Win8.1/10-era d3dcompiler_47.dll into the prefix's
                 system32 and force it native; Wine's builtin lacks the
                 buffer intrinsics and [RootSignature(...)] (docs/proton.md)
  --texture-wait leave VICTORY_TEXTURE_NO_WAIT unset (the default is 1)
  --texture-no-wait
                 accepted for compatibility; the default is already on
  --watchdog     keep the title's stall watchdog (the default disarms it
                 with the VICTORY_NO_WATCHDOG hooks; docs/proton.md)
  --no-watchdog  accepted for compatibility; the default is already on
  --mangohud     force the fps/frametime overlay on (default: on under
                 GE-Proton, off elsewhere — MangoHud cannot present
                 through the Valve family's shared present mode)
  --no-mangohud  leave MANGOHUD unset (no overlay)
  --log-window   show the logs in Wine's console window (the exe runs
                 through start.exe); VICTORY_LOG_FILE picks the mirror log

Notes:
  VICTORY_COPY_MEMORY defaults to 1: under Wine the title memory used in
  place is never seen by the GPU and every frame stays black (#19, #53).

  The title's start screen listens to a pad. On a machine without one, a
  diagnostics build needs VICTORY_KEYS to leave it, for example
  VICTORY_KEYS=25:Enter,40:Enter,55:Enter,70:Enter; the release build has
  no scripted keys (docs/proton.md).

Examples:
  proton/run.sh config
  proton/run.sh --z /mnt/games inspect "Z:\\game\\scripts\\x64\\final.js.c.c.gmpc"
USAGE
}

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
prefix="${VICTORY_PROTON_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/victory/proton}"
exe="${VICTORY_EXE:-$repo/target/x86_64-pc-windows-gnu/release/victory.exe}"
proton="${PROTON:-}"
z_dir=""
config=""
d3dcompiler=""
texture_no_wait=1
watchdog=""
mangohud=1
mangohud_forced=""
log_window=""

while [[ $# -gt 0 && $1 != -- ]]; do
    case "$1" in
        --prefix) prefix=$2; shift 2 ;;
        --exe)    exe=$2; shift 2 ;;
        --proton) proton=$2; shift 2 ;;
        --z)      z_dir=$2; shift 2 ;;
        --config) config=$2; shift 2 ;;
        --d3dcompiler) d3dcompiler=$2; shift 2 ;;
        --texture-wait)    texture_no_wait=""; shift ;;
        --texture-no-wait) texture_no_wait=1; shift ;;
        --watchdog)    watchdog=1; shift ;;
        --no-watchdog) watchdog=""; shift ;;
        --mangohud)    mangohud=1; mangohud_forced=1; shift ;;
        --no-mangohud) mangohud=""; mangohud_forced=1; shift ;;
        --log-window)  log_window=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)      break ;;
    esac
done
[[ $# -gt 0 && $1 == -- ]] && shift
[[ $# -gt 0 ]] || set -- config

die() { echo "proton/run.sh: $*" >&2; exit 1; }

# Newest GE-Proton first, then the newest plain Proton-*; the two families
# version-sort independently (a Proton-9.0 is not newer than a GE-Proton10).
if [[ -z $proton ]]; then
    tools=${STEAM_COMPAT_TOOLS:-$HOME/.local/share/Steam/compatibilitytools.d}
    if [[ -d $tools ]]; then
        for family in 'GE-Proton*' 'Proton-*'; do
            newest=$(find "$tools" -maxdepth 1 -mindepth 1 -type d \
                -name "$family" -printf '%f\n' | sort -V | tail -n1)
            if [[ -n $newest ]]; then
                proton=$tools/$newest
                break
            fi
        done
    fi
fi
[[ -n $proton ]] || die "no Proton found; install one under compatibilitytools.d or pass --proton"

[[ -x "$proton/proton" || -x "$proton" ]] || die "$proton is neither a Proton directory nor an executable"
[[ -f $exe ]] || die "no $exe — build it with: cargo win"
exe=$(readlink -f -- "$exe")

# MangoHud 0.8.x cannot present through VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
# which the Valve Proton family's vkd3d uses: there the overlay ends the run
# early (exit code 5). GE-Proton tolerates the same warning, so the overlay
# defaults on under GE and off elsewhere; the flags and MANGOHUD override.
if [[ -z $mangohud_forced && -z ${MANGOHUD:-} ]]; then
    case "$(basename -- "$proton")" in
        GE-Proton*) ;;
        *) mangohud="" ;;
    esac
fi

# Steam uses these when launching Proton; the proton script wants them even
# when driven by hand. Export them before probing the prefix: getnativepath
# is itself a Proton verb and derives the prefix from them.
export WINEPREFIX=$prefix
export STEAM_COMPAT_DATA_PATH=$prefix
export STEAM_COMPAT_CLIENT_INSTALL_PATH=${STEAM_COMPAT_CLIENT_INSTALL_PATH:-$HOME/.steam/Steam}
# Proton injects through steam.exe unless it sees a UMU launch; that inject
# forks (GUI convention) and our output dies with the orphan. UMU_ID marks
# this as a non-Steam launch, so the script runs wine directly, waits, and
# stdout/exit code come back to this terminal.
export UMU_ID=${UMU_ID:-victory}

# Where the Wine prefix really lives: the proton script derives it from
# STEAM_COMPAT_DATA_PATH (a `pfx` dir inside it), silently overriding the
# WINEPREFIX we set. Ask it where C:\ is and take the prefix root above
# drive_c, falling back through the Steam convention to our plain prefix.
drive_c=""
if [[ -x "$proton/proton" ]]; then
    drive_c="$("$proton/proton" getnativepath 'C:\' 2>/dev/null || true)"
fi
real=""
if [[ -n $drive_c && $drive_c != / ]]; then
    real=$(dirname -- "${drive_c%/}")
fi
[[ -n $real && -d $real ]] || real=${prefix}/pfx
[[ -d $real ]] || real=$prefix

# Z:'s root. New Wine prefixes map it to <prefix>/drive_z, and we curate
# that: game folders linked in under --z appear as Z:\<name>. GE-Proton 10+
# instead maps z: to `/` - the whole host filesystem is visible under Z: -
# which also works (game_dir becomes e.g. Z:\mnt\games\title), so in that
# mode --z is a no-op and the docs say so. Create dosdevices first: a fresh
# prefix has no directory for the link, and the link is relative to it.
mkdir -p "$real/dosdevices"
zlink=$real/dosdevices/z:
if [[ -e $zlink && ! -L $zlink ]]; then
    mv "$zlink" "$real/drive_z"
    ln -sfn -- ../drive_z "$zlink"
fi
if [[ ! -e $zlink ]]; then
    mkdir -p "$real/drive_z"
    ln -sfn -- ../drive_z "$zlink"
fi
zroot=$(readlink -f -- "$zlink")
if [[ -n $z_dir ]]; then
    if [[ $zroot == / ]]; then
        echo "proton/run.sh: Z: is / under this Proton; --z is a no-op (name the host path, e.g. Z:\\mnt\\games)" >&2
    else
        mkdir -p "$zroot"
        ln -sfn -- "$z_dir" "$zroot/$(basename -- "$z_dir")"
    fi
fi

# The d3dcompiler truth: Wine's builtin is a vendored vkd3d-shader with no
# RWByteAddressBuffer.Interlocked*()/Load*() and no [RootSignature(...)];
# the 2010 winetricks redistributable predates D3D12 and embeds no root
# signature. A Win8.1/10-era native DLL is the working path; --d3dcompiler
# pins one in the prefix instead of leaving it to whatever winetricks did.
if [[ -n $d3dcompiler ]]; then
    [[ -f $d3dcompiler ]] || die "no $d3dcompiler"
    mkdir -p "$real/drive_c/windows/system32"
    cp -- "$d3dcompiler" "$real/drive_c/windows/system32/d3dcompiler_47.dll"
fi

# Wine's built-in d3d12/dxgi stubs answer E_INVALIDARG and draw nothing;
# the GPU path needs the dxvk/vkd3d-proton copies Proton syncs into the
# prefix as native. Recent Proton forces this itself; saying it too keeps
# hand-made prefixes behaving the same (docs/proton.md, #41).
overrides=${WINEDLLOVERRIDES:-"d3d12=n;d3d12core=n;dxgi=n"}
if [[ -n $d3dcompiler ]]; then
    case ";$overrides;" in
        *";d3dcompiler_47="*) ;;
        *) overrides="$overrides;d3dcompiler_47=n" ;;
    esac
fi
export WINEDLLOVERRIDES=$overrides

# The texture lane's remap-lock wait is a Wine-side mapping lock that costs
# seconds while the world streams; the default trades up to two textures a
# frame for it (#178). --texture-wait leaves the wait in place.
if [[ -n $texture_no_wait ]]; then
    export VICTORY_TEXTURE_NO_WAIT=1
fi

# Under Wine the in-place title-memory windows are made without an error,
# but the GPU never sees what the title writes there: every frame stays
# black from the first one (#19). Copying is the working path, and #53
# makes it the code default under Wine; until then the launcher sets it.
# VICTORY_COPY_MEMORY=0 keeps the memory in place once #53 lands.
export VICTORY_COPY_MEMORY=${VICTORY_COPY_MEMORY:-1}

# The title's stall watchdog ends the process when its main loop has not
# advanced for ~30 s; a cold pipeline compile under Wine stalls longer,
# so runs loop on the loading screen and then end themselves
# (victory-fork#416). The loader's VICTORY_NO_WATCHDOG hooks
# (victory-fork#364) jump over the kill; they are opt-in there and on by
# default here. --watchdog (or the variable set 0) keeps the title's
# watchdog — a real hang then ends the run as before.
no_watchdog=${VICTORY_NO_WATCHDOG:-1}
if [[ -n $watchdog || $no_watchdog == 0 ]]; then
    unset VICTORY_NO_WATCHDOG
else
    export VICTORY_NO_WATCHDOG=$no_watchdog
fi

# Shader caches live outside the prefix so a fresh prefix starts warm:
# vkd3d-proton's pipeline cache and Mesa's RADV cache both key on the
# shader bytes, not on the prefix. Without VKD3D_SHADER_CACHE_PATH the
# vkd3d cache lands in the process cwd (the repo root); without
# MESA_SHADER_CACHE_DIR Mesa uses ~/.cache/mesa_shader_cache. Both are
# overridable from the environment.
shader_cache=${XDG_DATA_HOME:-$HOME/.local/share}/victory/shader-cache
mkdir -p "$shader_cache/vkd3d" "$shader_cache/mesa"
export VKD3D_SHADER_CACHE_PATH=${VKD3D_SHADER_CACHE_PATH:-$shader_cache/vkd3d}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-$shader_cache/mesa}

# MangoHud's Vulkan layer puts fps, frametimes and GPU load on screen —
# the only view of the world's pace while the mesh layer presents black
# (docs/proton.md). The system layer answers MANGOHUD=1; MANGOHUD_CONFIG
# passes through untouched. The overlay is on screen only: VICTORY_CAPTURE
# reads the presented buffer without it. The default is family-conditional
# (above); --no-mangohud or MANGOHUD=0 leaves it off.
if [[ -n $mangohud ]]; then
    export MANGOHUD=${MANGOHUD:-1}
else
    unset MANGOHUD
fi

# Wine resolves the process' Unix cwd through its Z: mapping, so a config
# given relative to the repo root works on both sides unchanged.
cd "$repo"
if [[ -n $config ]]; then
    [[ -f $config ]] || die "no $repo/$config"
    export VICTORY_CONFIG=$config
fi

# A bare wine binary (not a Proton dir) gets the exe as-is: no dxvk/vkd3d
# sync, so file-side commands only.
if [[ ! -x "$proton/proton" ]]; then
    exec "$proton" "$exe" "$@"
fi

# Ask Proton for the exe's own Windows-side (Z:) path. With UMU_ID set,
# Proton's script takes the non-Steam path and runs wine on this argument
# instead of injecting through steam.exe (GE-Proton/Proton 9+). We keep
# `waitforexitandrun` rather than `runinprefix`: the latter skips Proton's
# prefix setup, so a fresh prefix never receives dxvk/vkd3d.
wexe=$("$proton/proton" getcompatpath "$exe" 2>/dev/null)
[[ -n $wexe ]] || die "proton getcompatpath failed for $exe"

# --log-window shows the run's logs in Wine's own console window, the live
# view the first runs had: start.exe without /exec goes through ShellExecute,
# so Wine allocates a fresh console (with /exec the child inherits the
# redirected handles and no console window appears). victory mirrors its log
# to CONOUT$ (which reaches that window) plus VICTORY_LOG_FILE.
if [[ -n $log_window ]]; then
    export VICTORY_LOG_CONSOLE=1
    export VICTORY_LOG_FILE=${VICTORY_LOG_FILE:-$HOME/.local/share/victory/proton-run.log}
    mkdir -p "$(dirname "$VICTORY_LOG_FILE")"
    exec "$proton/proton" waitforexitandrun 'C:\windows\system32\start.exe' /wait "$wexe" "$@"
fi
exec "$proton/proton" waitforexitandrun "$wexe" "$@"
