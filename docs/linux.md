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
VIctory asks many such questions per frame. ajsem1337's patch in #340 keeps
the views in a tree instead. On an RTX 5080 with proton-cachyos 11.0 a
world frame took about 4.0 s with the stock server and about 1.0 s with
the patched one; testers in #340 report similar gains.

`tools/wine/` builds it and makes a Proton copy that uses it:

```sh
# The Wine source of the exact Proton release you run (here: proton-cachyos 11.0).
tools/wine/build-wineserver.sh https://github.com/CachyOS/wine-cachyos.git \
    cachyos-11.0-20260702-proton-native ~/wineserver-build
# A copy of that Proton whose wineserver is the patched one.
tools/wine/proton-with-wineserver.sh /usr/share/steam/compatibilitytools.d/proton-cachyos-native \
    ~/wineserver-build/build/server/wineserver ~/proton-victory
```

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
  | GE-Proton 11 | Valve's wine at the commit GE pins, plus GE's patches | `-cachyos11` |
  | Valve Proton 10 | `ValveSoftware/wine`, `proton_10.0` | `-proton10` |
  | upstream Wine | `gitlab.winehq.org/wine/wine`, `master` | `-upstream` |

  Each port builds without warnings and was checked against the original
  lookups by a randomized comparison; only proton-cachyos 11.0 has been
  run with the title. It is Wine code (LGPL-2.1-or-later), credit to
  ajsem1337.
