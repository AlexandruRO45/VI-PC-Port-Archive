# Windows desktop launcher

VIctory's Windows launcher is a desktop application built with Python 3.13,
PySide6, and Qt Quick/QML. QML paints the interface; Python owns file picking,
launcher preferences, activity output, and the process bridge to victory.exe.
The selected game is still started by the existing Rust run --execute path.

## Run from a development checkout

Download a portable Python 3.13 runtime into local application data and install
the Qt for Python runtime there:

    py install --target="$env:LOCALAPPDATA\victory\launcher\python313" 3.13
    & "$env:LOCALAPPDATA\victory\launcher\python313\python.exe" -m pip install -r apps\victory\launcher\requirements.txt
    cargo run -p victory -- gui

The launcher reads its last game selection, console profile, and activity log
from %LOCALAPPDATA%\victory\launcher. The QML files are embedded in the Rust
binary and extracted there at startup, so the desktop app does not depend on
the checkout remaining beside victory.exe.

The maximized library screen balances two different, 3840 x 2160 Rockstar
artworks: a Vice City postcard as its translucent background and Jason and
Lucia on the rooftop as its foreground feature. Both are downloaded into
%LOCALAPPDATA%\victory\launcher\assets on first run. If either download is
unavailable, the launcher falls back to its local wallpaper or synthetic
Leonida skyline. Images remain outside the repository in accordance with the
[asset policy](../../../docs/asset-policy.md).

The desktop UI is a VIctory-branded Rockstar and GTA VI fan launcher. It uses
the original game's palette and setting as visual context, while keeping the
game location, console profile, launch status, and local activity easy to find.
The library also provides quick actions to rerun the selected game, check this
PC's Windows, processor, graphics adapter, and memory, choose the game folder,
open local data, view keyboard bindings, switch console profiles, and quit the
launcher. Graphics feature-level support is confirmed by the game at launch.

## Build a portable Windows launcher folder

Build VIctory first, then generate the bundled Python 3.13 / PySide6 runtime
beside it:

    cargo build --release -p victory
    .\apps\victory\launcher\build-runtime.ps1 -OutputDirectory .\target\release\launcher-runtime

Distribute target\release\victory.exe together with the complete
target\release\launcher-runtime\VictoryQmlRuntime directory. The packaged
runtime contains Python, PySide6, Qt Quick, and the QML controls needed by the
desktop window; end users do not need a separate Python installation.

The packaging script uses the portable Python runtime in local application data,
then places the bundled runtime beside victory.exe. Set VICTORY_PYTHON to an
explicit Python 3.13 executable to override runtime discovery.
