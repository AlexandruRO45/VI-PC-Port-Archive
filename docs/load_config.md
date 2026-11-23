# Load config: what Windows does at load that gta6rt does not

`runtime/src/pe_loader.cpp` maps an image, relocates it, binds its imports and delay imports, registers its `.pdata`
and runs its TLS. It never reads the load-config directory (`IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG`, index 10). Of the
data directories, it reads TLS, import, delay import, exception, base relocation and export. Windows uses the load
config at load time in three ways, so a mapped image does without all of them. `tools/load_config.py` reports, per
image, which of them an image relies on.

| What Windows does | Fields | Under gta6rt |
|---|---|---|
| Applies process settings from the process's own executable | `GlobalFlagsSet`/`Clear`, `CriticalSectionDefaultTimeout`, the four heap thresholds, `ProcessAffinityMask`, `ProcessHeapFlags`, `CSDVersion` | The process image is gta6rt.exe, so the game's values are never applied |
| Loads an image's dependencies with its flags | `DependentLoadFlags` | gta6rt binds imports itself |
| May overwrite function-pointer slots | the Control Flow Guard, XFG, RFG and guarded-memcpy check/dispatch pointers | The slot keeps what the linker stored. That is a default in the image's own code (a no-op check, a `jmp rax` dispatch), unless the slot holds 0 |
| Randomises the `/GS` security cookie | `SecurityCookie` | The MSVC CRT start-up replaces the linker's default when it finds it |
| Registers EH-continuation targets | `GuardEHContinuationTable` | Checked only when the process enforces CET shadow stacks, against modules Windows loaded |
| Applies dynamic value relocations | `DynamicValueRelocTableOffset`/`Section` | Not applied |
| Sets up an enclave, a hybrid (CHPE) image | `EnclaveConfigurationPointer`, `CHPEMetadataPointer` | Not supported |

An `attention:` line marks what a mapped image would need and gta6rt doesn't provide: a pointer slot that is 0 or
outside the image, non-zero process settings, dynamic relocations, CHPE, an enclave.

## Measured: the package's 16 DLLs

`python -I tools/load_config.py <package>\*.dll` on 1.0.2.1 (Windows 11 26200):

- 16 load configs: 312 bytes (11), 320 (4), 328 (1).
- **0 attention lines.** All 85 pointer slots the 16 images declare (5 each in 11 images, 6 in 5) hold an address in
  the image's own `.text`: the linker's defaults, which work without an OS loader. No dynamic relocations, CHPE or
  enclave. Every process setting and `DependentLoadFlags` is 0.
- The security cookie holds the linker's default (`0x2B992DDFA232`) in all 16, in `.data`. None sets
  `SECURITY_COOKIE_UNUSED`.
- `GuardFlags`:

  | Value | Images |
  |---|---|
  | `0x100` (`CF_INSTRUMENTED` only) | `Bink2Scarlett.dll`, `XboxBvhBuilder.dll`, both FidelityFX DLLs |
  | `0x10417500` | 9 of the CRT DLLs |
  | `0x417500` | `msvcp140_codecvt_ids.dll` |
  | `0x10017500` | `XCurl.dll` |
  | `0x10014500` | `libHttpClient.GDK.dll` |

  `0x10417500` decodes as CF instrumented, function table, protected delay-load IAT in its own section, export
  suppression info, long-jump table, EH-continuation table, stride 1.
- EH-continuation tables are in the ten CRT DLLs only (`msvcp140.dll` 129 targets, `vccorlib140.dll` 57,
  `msvcp140_2.dll` 49, `vcruntime140.dll` 17, `concrt140.dll` 9, `vcruntime140_1.dll` 6,
  `msvcp140_atomic_wait.dll` 5, `msvcp140_1.dll` 2, `msvcp140_codecvt_ids.dll` 2, `vcomp140.dll` 1). Those families
  bind to the host's own copies at run time ([runtime_imports.md](runtime_imports.md)), so none of the six DLLs
  gta6rt actually maps carries one.

## Measured: the executable

`game_scarlett_master.exe` 1.0.2.1 (own copy; size and SHA-256 match `docs/runtime_imports.md`): a 312-byte load
config and **0 attention lines**. Every process setting and `DependentLoadFlags` is 0, so gta6rt drops no setting
the title asked for. All five pointer slots (CFG check/dispatch, XFG check/dispatch/table dispatch) hold
own-code defaults, and `GuardFlags` is 0: the executable is not CFG-instrumented. The cookie holds the linker's
default, and there are no dynamic relocations, CHPE, enclave or EH-continuation table. Reproduce:

```powershell
python -I tools/load_config.py "D:\path\to\game_scarlett_master.exe"
python -I tools/load_config.py "D:\path\to\game_scarlett_master.exe" --json
```

It prints field values and section names only (no code or data of the image), and exits 0 with no `attention:`
line, 1 with one, 2 for an input it can't read.

## Open question: CET shadow stacks

`runtime/CMakeLists.txt` sets no `/CETCOMPAT` for gta6rt.exe, and the linker default was not checked here. If the
process ran with user-mode shadow stacks, an exception continuation or `longjmp` into a mapped image could be checked
against EH-continuation metadata Windows never registered for that image. That is a hypothesis, not a measurement.
The table above shows which images carry such tables.
