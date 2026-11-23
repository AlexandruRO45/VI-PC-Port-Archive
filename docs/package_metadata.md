# Package metadata: the launch chunk's small files

Beside `MicrosoftGame.Config` the package folder holds a few small files that no other note described:
`microsoft_gameconfig.bin`, `Catalog000.bin`, `resources.pri`, `Update.alignmentchunk` and the chunk layout
`layout_<guid>_xs.xml`. This page records what each one is, measured on package 1.0.2.1 (sizes and SHA-256 as in
`game/xbox-1.0.2.1.json`), and what the runtime does with the settings they carry. No value that identifies the title
is quoted here; `tools/gameconfig_hive.py` prints them on your own machine.

## The chunk layout

`layout_<guid>_xs.xml` lists 25 chunks:

| Chunk | Files |
|---|---|
| 1000 | `MicrosoftGame.Config`, the five `xbs_*.png` logos and splash, `resources.pri`, `appdata0.rpf` |
| 1001 | `game_scarlett_master.exe`, which the package does not contain ([runtime_imports.md](runtime_imports.md)) |
| 1002 | the 16 DLLs |
| 1003 | 46 loose `.ttpf` texture pools under `data\ttdb\` |
| 1004–1014 | one top-level `.rpf` each |
| 3000–3008 | the `packs\` archives and their pools |
| 3009 | `Update.alignmentchunk` only |

Chunk 3009 is the only one with an attribute, `Marker="Launch"`. In the GDK layout format the chunks before the launch
marker are the ones that must be installed before the title can start (a hypothesis from the public format, not
measured on a console); with the marker on the last chunk, that is every chunk. `Update.alignmentchunk` is one byte,
`0x30`.

## `microsoft_gameconfig.bin`: the config as a registry hive

The packaging tool compiles `MicrosoftGame.Config` into a Windows registry hive:

- 8,192 bytes: a `regf` base block (format 1.3, primary and secondary sequence numbers equal, checksum valid) and one
  4 KiB `hbin`. The base block's file-name field holds the packaging tool's temporary path, ending in `gc.bin`.
- One root key, named by a GUID in braces, with 10 values and 6 subkeys: `Executable` (one numbered subkey per
  `<Executable>`, `00000000` first), `ExtendedAttribute` and `Protocol` (both empty; the config has no list for
  them), `PersistentLocalStorage`, `ShellVisuals` and `VirtualMachine`.
- Value names are stored compressed (Latin-1), all cells sit in the one bin, and no value needs a big-data record.

`tools/gameconfig_hive.py check <game folder>` reads both files and compares each hive value with the config text it
is compiled from. On 1.0.2.1: 27 match, 0 differ, 0 config only, 2 hive only.

| Config | Hive | Stored as |
|---|---|---|
| `<Game configVersion>` | root `configVersion` | `REG_DWORD` |
| `<Identity>` `Name`, `Publisher`, `Version` | root values of the same names | `REG_SZ` |
| `<StoreId>` | root `StoreId` | `REG_SZ` |
| `<TitleId>` (8 hex digits) | root `TitleId` | `REG_DWORD`: the hex digits as a number |
| `<MSAAppId>` (16 hex digits) | root `MSAAppId` | `REG_QWORD`: the hex digits as a number |
| `<Resources>`, one `<Resource Language>` each | root `Languages` | one `REG_SZ`: the 39 tags joined with `;`, ending in `;` |
| `<Executable>` attributes | `Executable\00000000\<attribute>` | `REG_SZ`; `IsDevOnly="false"` becomes `REG_DWORD` 0 |
| `<PersistentLocalStorage><SizeMB>` | `PersistentLocalStorage\SizeMB` | `REG_DWORD` 2048, the quota `runtime/shims/xgameruntime.cpp` hard-codes |
| `<ShellVisuals>` attributes | `ShellVisuals\<attribute>` | `REG_SZ` |
| `<VirtualMachine>` children | `VirtualMachine\<element>` | `SMTEnabled` `true` as `REG_DWORD` 1; both `*TitleMemory` values as `REG_SZ` |
| (no element) | root `GameConfigRegRWVersion` | `REG_DWORD` 1 |
| (no element) | root `AdvancedUserModel` | `REG_DWORD` 1 |

What follows from it:

- The config text and the hive agree on every value, so the runtime, which reads `MicrosoftGame.Config` for the title
  id, the locales and the storage quota (`runtime/shims/xgameruntime.cpp`), reads the settings the hive carries.
  Which of the two files the console reads at launch was not measured.
- `AdvancedUserModel` has no element in the config; the packaging tool wrote it on its own. That it selects the GDK's
  advanced user model is a hypothesis from the name. The user shim (`runtime/shims/xgameruntime_user.cpp`) serves
  one signed-in default user either way.
- An edited config (a changed copy, a mod) shows up in `check` as `differs` or `config only`, because the hive stays
  as packaged.

## `VirtualMachine`: settings nothing reads yet

Both files declare `SMTEnabled` true and `Advanced` title memory for Series S (`XboxLockhartTitleMemory`) and
Series X (`XboxAnacondaTitleMemory`). No file under `runtime/` reads either setting. They matter in two places:

- **Thread placement.** With SMT enabled a console title schedules on two hardware threads per core. #2632 (closed,
  not merged) translated console affinity masks assuming one mask bit per console core of two SMT threads;
  `SMTEnabled` is the package's own statement that SMT is on.
- **Memory.** The budget the `Advanced` title-memory mode stands for is not in the package.

## Thread placement in the shipped DLLs

Read from the 16 DLLs' import and delay-import tables (the executable is not in the package):

| Module | Thread-placement and topology imports |
|---|---|
| `Bink2Scarlett.dll` | `SetThreadIdealProcessorEx` (`api-ms-win-core-processthreads-l1-1-1.dll`), `SetThreadAffinityMask` (`api-ms-win-core-processtopology-obsolete-l1-1-0.dll`) |
| `vcomp140.dll` | `GetLogicalProcessorInformationEx`, `GetNativeSystemInfo` |
| `concrt140.dll`, `msvcp140.dll` | `GetNativeSystemInfo` |
| the other 12 | none |

The CRT families bind to the host's own copies at run time ([runtime_imports.md](runtime_imports.md)), so Bink is the
one packaged module that places threads itself. Its thread start, read from its code:

1. `CreateThread` with `CREATE_SUSPENDED`.
2. If the caller passed a core number: `SetThreadIdealProcessorEx` with group 0 and that number, and, when a flag bit
   is also set, `SetThreadAffinityMask(thread, 1 << core)`.
3. `ResumeThread`. Neither placement call's return value is tested.

One more site is a stub that jumps straight to `SetThreadAffinityMask`. The core numbers come from the caller
(`BinkStartAsyncThread` is one of the DLL's 74 exports), that is from the executable, so they are unmeasured. For the
port this means:

- A core number the host can't satisfy makes the call fail, and Bink carries on with an unpinned thread: no crash path.
- When the call succeeds, the decode thread is hard-pinned to host logical processor `core`, which can be the SMT
  sibling of another pinned thread or an efficiency core.
- An affinity shim registered under `kernel32.dll`, `kernelbase.dll` and `api-ms-win-core-processthreads-l1-1-*`
  (the set #2632 used) sees Bink's `SetThreadIdealProcessorEx` but not its `SetThreadAffinityMask`, which arrives
  through `api-ms-win-core-processtopology-obsolete-l1-1-0.dll`.

## `Catalog000.bin`: identified, not decoded

815,104 bytes: a PE32+ header for AMD64 with subsystem 17 (the Windows SDK's `winnt.h` names it
`IMAGE_SUBSYSTEM_XBOX_CODE_CATALOG`), no entry point, image base 0, no data directories, and one unnamed read-only
section of 0xC6000 bytes. By its subsystem it is the console's code-integrity catalog, so it is left undecoded
(CONTRIBUTING.md: no protection workarounds). `make_game.py` uses it only as one of the files that identify a package
folder (`PROBES`), and `scripts/check_no_game_data.py` matches it as a PE executable, so it can't be committed by
mistake.

## `resources.pri`: identified, not decoded

1,152 bytes: a Package Resource Index with the magic `mrm_pri2` and five sections, `[mrm_decn_info]`,
`[mrm_pridescex]`, `[mrm_hschemaex]`, `[mrm_res_map2_]` and `[mrm_dataitem]`. Nothing in the runtime reads it.

## Reproduce

```powershell
python -I tools/gameconfig_hive.py check "D:\path\to\your\package"
python -I tools/gameconfig_hive.py dump "D:\path\to\your\package\microsoft_gameconfig.bin" --json
python -I tests/test_gameconfig_hive.py
```

`check --values` adds both values to a `differs` line; `--hive` and `--config` read other files than the folder's.
The unit test builds its hives and configs by hand, so it needs no game files.
