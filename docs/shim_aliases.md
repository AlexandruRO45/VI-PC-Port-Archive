# Shim aliases: imports that reach the host instead of their shim

The shim registry keys every row on the exact lower-cased `dll!name` (`runtime/shims/registry.cpp`, `find`), and an
`api-ms-*`, `ext-ms-*` or system-DLL import with no row binds to the host's function. Windows does not compare API-set
names that strictly. So a function shimmed under one name of a contract runs the host's code when an image imports it
under another name of the same contract, and nothing reports it: `gta6rt --dry-run` counts it under `host`. For
`ExitProcess` that is the failure #4043 fixed (the host's `ExitProcess` ends `gta6rt` before it reports).
`tools/shim_aliases.py` lists the cases from the shim sources and an image's import tables.

## The Windows rule

Windows resolves an API-set name without its last version segment. Measured with `ApiSetQueryApiSetPresence`
(the ntdll query `tools/pe_inspect.py` already uses) on Windows 11 Pro 26200:

| Name queried | Present |
|---|---|
| `api-ms-win-core-processthreads-l1-1-0` | yes |
| `api-ms-win-core-processthreads-l1-1-3` | yes |
| `api-ms-win-core-processthreads-l1-1-99` | yes, though no such version was ever published |
| `api-ms-win-core-processthreads-l1-2-0` | no: `l1-2` is a contract of its own |
| `api-ms-win-core-nonexistent-l1-1-0` | no |

So `…-l1-1-0.dll` and `…-l1-1-2.dll` reach the same host DLL, while the registry treats them as two unrelated names.
`tools/shim_aliases.py` calls the name without its last segment the contract (`api-ms-win-core-file-l1-2-0.dll` →
`api-ms-win-core-file-l1-2`) and marks a bypass whose DLL differs from a registered one only in that segment as a
`version alias`.

## What the shim table registers

`python -I tools/shim_aliases.py table` on `main` `ae12372` reads 379 rows: 138 functions under 29 DLL names
(literal `{"dll", "Name", ...}` rows and the dll-taking macros the same file or a header defines, such as
`EXIT_PROCESS(dll)` and `FILE_API(dll)`). `contracts` lists the API-set versions among them:

| Contract | Versions registered | Functions |
|---|---|---:|
| `api-ms-win-core-file-l1-1` | `-0` | 33 |
| `api-ms-win-core-file-l1-2` | `-0` | 33 |
| `api-ms-win-core-file-l2-1` | `-0` | 33 |
| `api-ms-win-core-libraryloader-l1-1` | `-0`, `-1` | 9 |
| `api-ms-win-core-libraryloader-l1-2` | `-0`, `-1`, `-2` | 9 |
| `api-ms-win-core-memory-l1-1` | `-0` to `-6` | 1 |
| `api-ms-win-core-processenvironment-l1-1` | `-0` | 2 |
| `api-ms-win-core-processthreads-l1-1` | `-0` | 1 (`ExitProcess`) |
| `api-ms-win-crt-runtime-l1-1` | `-0` | 17 |

An import of one of these functions through any version not listed (for example `ExitProcess` from
`api-ms-win-core-processthreads-l1-1-2.dll`, or a file function from `api-ms-win-core-file-l1-2-1.dll`) gets the
host's function. Which versions an image actually names is only known from its import table.

## Measured images

`check` on the package's 16 DLLs (1.0.2.1): 1,334 named imports, 133 through a shim, **0 bypasses**. The DLLs import
these functions only under names the table registers.

The executable is not in the package available here, so its imports are unmeasured. #6715 counts 53 of its imports
through a shim and 432 bound to the host. `check` tells how many of the 432 are shimmed functions under another name.
It prints only DLL and function names, as `gta6rt --dry-run` does:

```powershell
python -I tools/shim_aliases.py check "D:\path\to\game_scarlett_master.exe"
python -I tools/shim_aliases.py check "D:\path\to\your\package\Bink2Scarlett.dll" --json
python -I tools/shim_aliases.py contracts
```

Exit status: 0 with no bypass, 1 with one, 2 for an input it can't read. `--runtime` reads the shim sources from
another folder, for example a branch's `runtime/shims`.

## What a fix would look like

Matching rows by contract instead of by exact name in `registry.cpp`, with the exact name tried first, would make every
version of a registered contract reach the shim. That is a runtime change for a Windows build, and it is not made
here. Until then, a bypass `check` reports is fixed by adding the version to the row's macro list, as
`api-ms-win-core-memory-l1-1-0` to `-6` already are.

## Limits

- The parser reads `SHIM_TABLE(...)` bodies and function-like macros, not the C preprocessor in full. A row built
  from a variable or a generated table would be missed. `tests/test_shim_aliases.py` checks rows known to be in
  `runtime/shims` so that a missed format shows up as a failure.
- Ordinal imports are not matched (the table names functions).
- A bypass is not proof of a bug. Some functions may be shimmed under one name on purpose. The report says where the
  host's code runs instead.
