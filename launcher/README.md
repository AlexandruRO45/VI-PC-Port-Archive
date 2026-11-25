# GTA VI Launcher

A small Windows launcher for the PC port runtimes in this archive. Point it at your own copy of the game's `.exe` once, pick a runtime and press **PLAY**.

**[Download GTA-VI-Launcher.zip](https://github.com/Guernica4u/GTA6-PC-Port-Archive/releases/latest/download/GTA-VI-Launcher.zip)** (Windows 10/11, about 50 KB)

Unzip it into a folder of its own. The launcher keeps everything next to itself: your settings, the runtimes it builds, their caches and the slideshow pictures.

## What it does

- **Runtimes:** the archive's top 3, read live from [the archive site](https://guernica4u.github.io/GTA6-PC-Port-Archive/), plus ztrx, Hyphen (hyphenn), VIctory and xcl. Hover over a runtime to see how far it gets.
- **Install & play:** the first PLAY on a runtime downloads its source from this archive and builds it. That needs [git](https://git-scm.com/download/win) and [Rust](https://rustup.rs) with the MSVC build tools. A console window shows the progress and stays open if something fails.
- **Slideshow:** shows every picture in the `backgrounds` folder next to the launcher. To add pictures, drag them onto the window. With no pictures, it shows the game's own splash screen.
- **Music:** plays music from your copy of the game's `audio.rpf`, using the Opus decoder built into Windows. The speaker button in the top bar mutes it. Music stops while the game runs.
- **Full screen:** the maximize button, F11, Alt+Enter, or double-click the window. Esc leaves full screen.

No game files are included or downloaded. The launcher only reads your own copy.

## Build from source

`build.bat` compiles `Launcher.cs` with the C# compiler that ships with Windows. Nothing else is needed. `GTA VI Launcher.exe --selftest` runs its checks. `make-icon.ps1` redraws `icon.ico`.
