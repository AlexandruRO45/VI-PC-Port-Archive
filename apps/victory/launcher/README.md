# Windows desktop launcher

The VIctory desktop launcher provides Library, Activity, and Settings screens.
Its Python bridge, Qt Quick interface, and synthetic skyline are embedded in
the Rust executable. At startup they are written under
`%LOCALAPPDATA%\victory\launcher`, outside the checkout. The launcher does not
fetch artwork or make network requests.

## Run from a development checkout

Install Python 3.13 and the Qt for Python runtime into local application data,
then start the launcher:

```powershell
$launcher = Join-Path $env:LOCALAPPDATA 'victory\launcher'
py -3.13 -m venv (Join-Path $launcher 'venv')
& (Join-Path $launcher 'venv\Scripts\python.exe') -m pip install -r apps\victory\launcher\requirements.txt
cargo run -p victory -- gui
```

The selected executable and game directory are saved locally. The Play action
uses the existing Rust `run --execute` command. Build and configure the runtime
with files supplied by the user; no game or console SDK assets are included.
