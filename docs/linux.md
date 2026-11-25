# Running under Wine/Proton on Linux

VIctory is a Windows program; on Linux it runs under Wine, in practice a
Proton build with vkd3d-proton for Direct3D 12. This is not a supported
platform. These notes are what has worked for testers.

## Setup

- A Proton build with vkd3d-proton (tested: proton-cachyos 11.0,
  vkd3d-proton 3.2.0; GE-Proton 11 is reported to work too). Run
  `victory.exe` with `proton runinprefix` in a prefix of its own.
- Use the native Direct3D 12 DLLs:
  `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi=n,b"`, with vkd3d-proton's
  `d3d12.dll` and `d3d12core.dll` next to `victory.exe`. Wine's builtin
  `libvkd3d-1.dll` takes a different path and is not what has been tested.
- NVIDIA: some translated shaders crash the driver's compiler, which ends
  the process. #491 (not merged yet) adds `VICTORY_PIPELINE_JOURNAL=1`:
  a pipeline whose creation ended a run is refused from the next run on,
  so each run gets further. Without #491 the variable does nothing.

## The wineserver patch (about four times faster)

Under Wine, much of a frame's time goes into the wineserver, which keeps
each process's mapped views in a list and walks it on every memory query.
VIctory asks many such questions per frame. ajsem1337's patch in #340, with
tests and contributions by neonbyte1, keeps the views in a tree instead. On an RTX 5080 with proton-cachyos 11.0 a
world frame took about 4.0 s with the stock server and about 1.0 s with
the patched one; testers in #340 report similar gains.

ajsem1337 has proposed the change to Wine itself
([Wine bug 60868](https://bugs.winehq.org/show_bug.cgi?id=60868)) and to
wine-cachyos ([CachyOS/wine-cachyos#50](https://github.com/CachyOS/wine-cachyos/pull/50)).
Once a Proton release ships it (for proton-cachyos, a release after that
pull request is merged; check its release notes), that Proton's own
wineserver already keeps views in a tree and none of the following is
needed. Until then, `tools/wine/` builds the patched server and makes a
Proton copy that uses it:

Two steps: build the server from the Wine source of the exact Proton you
run, then make a copy of that Proton that uses it. Pick your Proton:

**proton-cachyos 11.0** (tested with the title):

```sh
tools/wine/build-wineserver.sh https://github.com/CachyOS/wine-cachyos.git \
    cachyos-11.0-20260702-proton-native ~/wineserver-build   # the tag of your release
tools/wine/proton-with-wineserver.sh /usr/share/steam/compatibilitytools.d/proton-cachyos-native \
    ~/wineserver-build/build/server/wineserver ~/proton-victory
```

**Proton Experimental** (Steam's "Proton - Experimental"):

```sh
tools/wine/build-wineserver.sh https://github.com/ValveSoftware/wine.git \
    experimental_11.0 ~/wineserver-build
tools/wine/proton-with-wineserver.sh ~/.steam/steam/steamapps/common/"Proton - Experimental" \
    ~/wineserver-build/build/server/wineserver ~/proton-victory
```

Experimental moves often: build again after Steam updates it, or the
server and the rest of Wine stop matching.

**Valve Proton 10** (Steam's "Proton 10.0"):

```sh
tools/wine/build-wineserver.sh https://github.com/ValveSoftware/wine.git \
    proton_10.0 ~/wineserver-build
tools/wine/proton-with-wineserver.sh ~/.steam/steam/steamapps/common/"Proton 10.0" \
    ~/wineserver-build/build/server/wineserver ~/proton-victory
```

**GE-Proton 11** (e.g. GE-Proton11-14): GE's Wine is not in a single
repository. It is Valve's Wine with GE's patches, which GE's own script
applies. Prepare that source first, then pass its `wine` directory
instead of a git URL (`-` for the tag):

```sh
git clone --recurse-submodules --branch GE-Proton11-14 \
    https://github.com/GloriousEggroll/proton-ge-custom ~/proton-ge-src   # your release's tag
cd ~/proton-ge-src && ./patches/protonprep-valve-staging.sh &> patchlog.txt && cd -
tools/wine/build-wineserver.sh ~/proton-ge-src/wine - ~/wineserver-build
tools/wine/proton-with-wineserver.sh ~/.steam/steam/compatibilitytools.d/GE-Proton11-14 \
    ~/wineserver-build/build/server/wineserver ~/proton-victory
```

The patch applies to GE's Wine as GE's script leaves it, but this path
has not been built end to end yet.

**Upstream Wine** (a Wine you built or installed yourself, not Proton):
build from the same source and version as your Wine, e.g.
`https://gitlab.winehq.org/wine/wine.git` at its tag, and replace that
Wine's `wineserver` with the one built here.

Then run `victory.exe` with `~/proton-victory/proton runinprefix ...`.

- The server must come from the same Wine source as the Proton it runs
  with: the two speak a private protocol, and a mismatch fails at start-up.
  The build prints the protocol version (`SERVER_PROTOCOL_VERSION`).
- Proton ignores `WINESERVER`; ntdll starts the server found next to the
  real path of `ntdll.so`. That is why the copy is a tree of symlinks in
  which `ntdll.so`, the loader, `wine`, the `proton` script and the server
  are real files.
- Only one wineserver serves a prefix. Do not run anything else in the
  prefix with another Proton while the title runs: the first process
  started picks the server, and a run on the stock server is back to the
  old frame times without any warning.
- The patch is ported to these trees; the script uses the first that
  applies, and says which:

  | Proton / Wine | source to build from | patch |
  |---|---|---|
  | proton-cachyos 11.0 | `CachyOS/wine-cachyos`, tag of your release | `-cachyos11` |
  | Proton Experimental (11.0) | `ValveSoftware/wine`, `experimental_11.0` | `-cachyos11` |
  | GE-Proton 11 | `proton-ge-custom` at your release tag, after GE's prep script (see above) | `-cachyos11` |
  | Valve Proton 10 | `ValveSoftware/wine`, `proton_10.0` | `-proton10` |
  | upstream Wine | `gitlab.winehq.org/wine/wine`, `master` | `-upstream` |

  Each port builds without warnings and was checked against the original
  lookups by a randomized comparison; only proton-cachyos 11.0 has been
  run with the title. It is Wine code (LGPL-2.1-or-later), credit to
  ajsem1337 and neonbyte1.
