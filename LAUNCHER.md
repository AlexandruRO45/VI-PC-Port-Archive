# Leonida Launcher

An optional one-window front end for `make_game.py`: find your game files, install the tools, build, play, and
keep the checkout up to date. `Build.bat` still works on its own; the launcher only calls `make_game.py` with the
options you pick.

## Start it

1. Install [Python 3.12+](https://www.python.org/downloads/) (tick "Add python.exe to PATH").
2. Get this repo (`git clone --recursive`, or **Code → Download ZIP**: the launcher connects a ZIP copy to GitHub
   the first time it updates, which needs Git).
3. Double-click **`Leonida Launcher.bat`**.

## Pages

- **Home**: build and play, plus the state of the game files, tools, free space and version.
- **Folders**: your own package folder (found with `make_game.find_dumps()` or picked by hand), the output folder
  and the work folder, with free space per drive.
- **Tools**: checks Python, .NET 8 SDK, Vulkan SDK, Git, Visual Studio C++ + Windows SDK, llvm-mc (AMDGPU) and
  ffmpeg. Each can be downloaded from its official source (progress and the reason if it fails) or pointed to with
  **Locate**. Installers open normally, so the user accepts each licence and the admin prompt. llvm-mc is built
  with `scripts/build_llvm_mc.ps1`.
- **Build**: the stages, WAV and job count map to `make_game.py --stages`, `--wav` and `--jobs`; the log is live
  and Stop kills the process tree.
- **Updates**: `git fetch` on a timer and `git pull --ff-only --autostash` plus `git submodule update` when no
  build is running (both can be turned off).

Nothing here contains, downloads or decrypts game data. Settings, downloads and portable tools stay in
`_launcher/` and are git-ignored. Standard library only; Pillow, when installed, draws the Home picture.

Fonts: Big Shoulders Display (SIL Open Font License, `_launcher/fonts/OFL.txt`), loaded for the launcher's own
process only.
