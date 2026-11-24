> Français : [jour-j.md](fr/jour-j.md)

# First launch of a console executable

Steps to follow once the package executable (for example `game_scarlett_master.exe`) is available. Everything runs from the AnyXboxX root.

## 1. Prepare (once)

```powershell
git submodule update --init
cmake -S runtime -B runtime/build -A x64
cmake --build runtime/build --config Release
cargo build --release --manifest-path rust/Cargo.toml
python tools/build_tombox.py d3d12_xs xg_xs xmem
```

## 2. Safe diagnosis

```powershell
python tools/jour_j.py ..
```

The tool:

- inventories the executable: SHA-256 hash, headers, sections, TLS and imports;
- tells, for each imported function, who provides it: the package, AnyXboxX, TomBox, Windows or nobody;
- runs `--rapport` and `--lancer`, which execute the program in Unicorn, isolated from the PC.

The report is written to `%LOCALAPPDATA%\AnyXboxX\jour-j\<date>\rapport.md`, outside the repository: it names functions of the game and must not be versioned.

## 3. Real launch (optional)

```powershell
python tools/jour_j.py .. --natif --duree 120
```

`anyxboxx.exe` then runs the game code directly on the processor, with TomBox's console DLLs (`ANYXBOXX_TOMBOX_DIR`) and the call trace (`ANYXBOXX_TRACE=1`).

**This mode does not isolate the game.** Use it only for an executable of known origin.

## 4. After the diagnosis

- The **missing** functions in the report become issues, one per DLL.
- The game's **shaders** go through `--rdna2 … --liste` then `--hlsl`; the cache (`emu/shader_cache.py`) avoids recompiling them.
- Nothing from the game enters the repository: not the executable, not its shaders, not its data. Only function names and findings may be published.
