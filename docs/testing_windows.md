# Testing a build on Windows

The checklist from "I have my own copy" to "I know what works and what doesn't", on a Windows PC. The
prerequisites and the build itself are in the README's [How to build](../README.md#how-to-build); the GPU test
programs, their setup per OS and the skips and driver problems each GPU has are in [testing.md](testing.md).

Everything runs locally: the tools read your package in place and write only to your `--out` and `--work` folders.
Nothing in this repository is the game, and nothing produced here should be shared — no archives, executables,
carved shaders, textures or logs that quote them.

If a step fails, read [what to send](#when-something-fails) before reporting it.

## 1. Check the copy, build nothing

```powershell
python -I make_game.py --find                 # copies found in Downloads, Desktop, Documents, Games, home, drive roots
python -I make_game.py --check                # sizes and names against the known 1.0.2.1 tables
python -I make_game.py --check --full-check   # the same plus the SHA-256 of every file (slow, a few hours)
```

`--check` is the fast signal: it recognises the version from file names and sizes without reading anything large.
If it rejects your folder, the version is not 1.0.2.1 or the folder is not the one holding `MicrosoftGame.Config`
and the `.rpf` archives. Nothing later in this guide can work until this passes.

`--check` returns as soon as the verification is done. `--full-check` alone also verifies and then goes on into the
build, so keep `--check` in front of it for a check that builds nothing.

## 2. Build the cheapest stage first

```powershell
python -I make_game.py --stages shaders --game "D:\path\to\your\package" --out "D:\Games\Leonida" --work "E:\leonida-work"
```

Pick `--work` on the drive with room: the full build needs about 400 GB, while `check,shaders,runtime` (enough to
run the loader) needs about 30 GB. The stages are `check`, `shaders`, `assets`, `runtime` and `pipelines`; the
README's [How to build](../README.md#how-to-build) says which of them overlap. A stopped build resumes where it
left off, so a failure in `assets` does not redo the shaders.

Watch the shader stage's summary: it prints how many shaders were carved, translated and accepted by `spirv-val`.
Expected on the full corpus: 26,503 translated (2,236 VS, 12,101 PS, 2,689 CS, 9,282 mesh, 195 geometry), all
valid. A lower number means a translator rule regressed on your machine — that is worth reporting.

## 3. Re-check a translator change on a slice

If you are testing a change to `tools/rdna2spirv.py`, do not run the whole corpus to find out if it works. Run a
slice against the disassembly a previous build left in `--work`, and compare translated and valid counts before and
after:

```powershell
python -I tools/rdna2spirv.py "E:\leonida-work\shaders\dxbc\disasm" --out C:\scratch --limit 200
python -I tools/rdna2spirv.py "E:\leonida-work\shaders\dxbc\disasm" --out C:\scratch --shard 0/8
```

`--limit 200` takes the first 200 shaders; `--shard 0/8` takes one of eight interleaved slices, so eight runs cover
everything. A change that lowers either count is a regression.

## 4. CPU, GPU and pipeline tests, no game files needed

The runtime tests use hand-written shaders and fixtures, so they run on any machine with the toolchain, without a
package:

```powershell
powershell -File scripts\build_runtime.ps1
ctest --test-dir runtime\build --output-on-failure
```

The GPU programs (`vk_compute_test`, `vk_store_test`, `vk_texture_test`, the sample harness in
`runtime/test/shaders`) are built when CMake finds the Vulkan SDK, and need a Vulkan device. Their setup, the
hand-written samples and the skips each GPU has are in [testing.md](testing.md); this page does not repeat them.

To check that your driver can create every pipeline the corpus translated, run this from the checkout root once
step 2 has written its shader index:

```powershell
.\runtime\build\vk_pipeline_check.exe "E:\leonida-work\shaders\spirv" --jobs 8
```

The index is `<work>\shaders\spirv\index.json`, the one `make_game.py` writes; a Visual Studio generator puts the
program under `runtime\build\RelWithDebInfo\` instead. An unreadable index is an error (exit 2), so a wrong path
does not pass as "zero shaders".

## 5. Test against the game itself

With a finished `--out` folder, the pieces that touch your copy are the loader and the sources of the exported
assets. The loader test maps the game's DLLs, binds every import and prints what it could not resolve, without
running any game code:

```powershell
runtime\build\gta6rt.exe --dry-run "D:\path\to\your\package\XboxBvhBuilder.dll" --game-dir "D:\path\to\your\package"
```

(`build_runtime.ps1` writes `runtime\build\gta6rt.exe`. If you configured CMake by hand with a multi-config
generator, the binary is in the configuration subfolder, `runtime\build\RelWithDebInfo\`.)

The report line is `<name> base=<addr> size=<size> entry=+<rva> shim=N host=N module=N unimplemented=N`. A
non-zero `unimplemented` count is the useful signal: it names the next shim that has to be written. The same
directory in `LEONIDA_GAME_DIR` makes the package-dependent tests (`pe_loader_test`, and `bink_decode_test` with
`LEONIDA_MOVIE` set) run instead of printing `SKIP`:

```powershell
$env:LEONIDA_GAME_DIR = "D:\path\to\your\package"
ctest --test-dir runtime\build -R "pe_loader|bink" --output-on-failure
```

The assets stage is the other place to look: it prints per-format counts (models, textures, audio banks) and the
names of resources it rejected. A rejected resource with a layout error is a decoder gap, not a build failure.

Calling the executable's entry point is the next step and still open; [boot.md](boot.md) has the `--run` runbook,
the two traces and `tools/boot_trace.py`. Run it with the network off: the executable's `ws2_32.dll` imports bind to
the host's.

## When something fails

Send the message, not the file. Useful reports contain:

- the exact command line and the last 30 lines of output;
- counts: how many shaders translated, how many passed `spirv-val`, how many `unimplemented` imports;
- the shader name and the offset `Unsupported` names, if a particular shader was rejected;
- your GPU and driver version for anything GPU-related.

Never attach a `.rpf`, `.ttpf`, `.dxbc`, `.fssm`, `.spv`, texture, model or audio file, and never paste a
disassembly of the game's code. If a rule looks wrong, describe the field and the value; that is enough to
reproduce it on another copy.

## Troubleshooting

**The build says a stage is missing a tool.** Run the stages separately (`--stages check,shaders`) to see which one
stops and what it asked for; the build lists everything missing in one go before it starts.

**`llvm-mc` without the AMDGPU target, or `spirv-val` not found.** Both are in the README's prerequisites, and
[testing.md](testing.md) has the Windows and WSL forms to put in `LLVM_MC` and `PATH`.

**A driver, layer or GPU crash in a test.** [testing.md](testing.md) keeps the known problems per GPU, driver and
Vulkan layer, with what to do about each.
